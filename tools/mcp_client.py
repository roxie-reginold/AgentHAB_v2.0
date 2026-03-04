"""Deploy rules to openHAB via MCP protocol."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession
from mcp.client.stdio import stdio_client
import mcp.client.stdio as mcp_stdio
from mcp.types import CallToolResult, TextContent

from tools.rule_parser import RuleParser


def _sanitize_uid(name: str) -> str:
    """Convert rule name to a valid UID (alphanumeric + underscores only)."""
    # Remove quotes and special characters
    uid = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Remove consecutive underscores
    uid = re.sub(r'_+', '_', uid)
    return uid.strip('_').lower()


def _extract_then_clause(rule_code: str) -> str:
    """Extract the then clause from DSL rule."""
    match = re.search(r'\bthen\b(.*?)\bend\b', rule_code, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _convert_dsl_to_javascript(then_clause: str, trigger_items: List[str]) -> str:
    """Convert DSL then clause to JavaScript for openHAB script action.
    
    This handles common patterns like:
    - sendCommand(Item, ON/OFF/value)
    - postUpdate(Item, value)
    - if (Item.state < value) { ... }
    """
    js_code = then_clause
    
    # Convert Item.state to items.getItem('Item').state
    js_code = re.sub(
        r'\b(\w+)\.state\b',
        r"items.getItem('\1').state",
        js_code
    )
    
    # Convert sendCommand(Item, value) to items.getItem('Item').sendCommand(value)
    js_code = re.sub(
        r'sendCommand\s*\(\s*(\w+)\s*,\s*([^)]+)\)',
        r"items.getItem('\1').sendCommand(\2)",
        js_code
    )
    
    # Convert postUpdate(Item, value) to items.getItem('Item').postUpdate(value)
    js_code = re.sub(
        r'postUpdate\s*\(\s*(\w+)\s*,\s*([^)]+)\)',
        r"items.getItem('\1').postUpdate(\2)",
        js_code
    )
    
    # Convert ON/OFF to strings if they're not already
    js_code = re.sub(r'\bON\b(?!\')', "'ON'", js_code)
    js_code = re.sub(r'\bOFF\b(?!\')', "'OFF'", js_code)
    
    # Convert "as Number" cast to parseFloat (openHAB JS helper)
    js_code = re.sub(
        r"items\.getItem\('(\w+)'\)\.state\s+as\s+Number",
        r"items.getItem('\1').numericState",
        js_code
    )
    
    return js_code.strip()


def _build_rule_payload(
    rule_code: str,
    rule_name: str,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Convert DSL rule to openHAB JSON rule format."""
    parser = RuleParser()
    parsed = parser.parse_rule(rule_code)
    
    if not parsed:
        raise ValueError("Could not parse rule from DSL code")
    
    # Use parsed name or provided name
    name = parsed.name or rule_name
    uid = _sanitize_uid(name)
    
    # Build triggers from parsed trigger items
    triggers = []
    for i, item in enumerate(parsed.trigger_items):
        triggers.append({
            "id": f"trigger_{i}",
            "type": "core.ItemStateChangeTrigger",
            "configuration": {"itemName": item}
        })
    
    # If no triggers found, use a generic time trigger (fallback)
    if not triggers:
        triggers.append({
            "id": "trigger_0",
            "type": "timer.GenericCronTrigger",
            "configuration": {"cronExpression": "0 * * * * ?"}  # Every minute
        })
    
    # Convert then clause to JavaScript
    then_clause = _extract_then_clause(rule_code)
    js_script = _convert_dsl_to_javascript(then_clause, parsed.trigger_items)
    
    # Build the rule payload
    rule_payload = {
        "uid": uid,
        "name": name,
        "description": metadata.get("request", "") if metadata else "",
        "triggers": triggers,
        "conditions": [],
        "actions": [{
            "id": "script_action",
            "type": "script.ScriptAction",
            "configuration": {
                "type": "application/javascript",
                "script": js_script
            }
        }],
        "tags": []
    }
    
    return rule_payload


async def _deploy_rule_async(
    rule_code: str,
    rule_name: str,
    metadata: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str]:
    """Deploy rule via MCP protocol (async implementation)."""
    
    # Build the JSON rule payload from DSL
    try:
        rule_payload = _build_rule_payload(rule_code, rule_name, metadata)
    except ValueError as e:
        return False, f"Failed to parse rule: {e}"
    
    # Locate the MCP server
    project_root = Path(__file__).parent.parent
    mcp_server = project_root.parent / "openhab-mcp" / "openhab_mcp_server.py"
    
    if not mcp_server.exists():
        return False, f"MCP server not found at {mcp_server}"
    
    command = sys.executable
    args = [str(mcp_server)]
    
    # Connect via MCP protocol
    ParamsClass = getattr(mcp_stdio, "StdioServerParameters", None)
    
    try:
        if ParamsClass:
            params = ParamsClass(command=command, args=args)
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    # Call create_rule tool
                    result: CallToolResult = await session.call_tool(
                        "create_rule",
                        {"rule": rule_payload}
                    )
                    
                    if result.isError:
                        return False, "MCP create_rule returned an error"
                    
                    # Parse response
                    data = None
                    if result.structuredContent is not None:
                        data = result.structuredContent
                    else:
                        for block in result.content:
                            if isinstance(block, TextContent):
                                try:
                                    data = json.loads(block.text)
                                    break
                                except Exception:
                                    continue
                    
                    # Unwrap "result" key if present
                    if isinstance(data, dict) and "result" in data and len(data) == 1:
                        data = data["result"]
                    
                    if data and isinstance(data, dict):
                        rule_uid = data.get("uid", rule_payload["uid"])
                        return True, f"Rule '{rule_uid}' deployed successfully"
                    
                    return True, "Rule deployed (response format unknown)"
        else:
            async with stdio_client(command, args) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    result: CallToolResult = await session.call_tool(
                        "create_rule",
                        {"rule": rule_payload}
                    )
                    
                    if result.isError:
                        return False, "MCP create_rule returned an error"
                    
                    return True, f"Rule '{rule_payload['uid']}' deployed successfully"
                    
    except Exception as e:
        return False, f"MCP deployment failed: {e}"


def deploy_rule_via_mcp(
    rule_code: str,
    *,
    rule_name: str,
    metadata: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str]:
    """Deploy a generated rule to openHAB via MCP protocol.
    
    Args:
        rule_code: The DSL rule code to deploy
        rule_name: Name for the rule
        metadata: Optional metadata (e.g., original request)
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    return asyncio.run(_deploy_rule_async(rule_code, rule_name, metadata))
