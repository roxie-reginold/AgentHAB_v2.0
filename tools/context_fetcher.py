"""Fetch live openHAB system context via the openHAB MCP server."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from mcp import ClientSession
from mcp.client.stdio import stdio_client
import mcp.client.stdio as mcp_stdio
from mcp.types import CallToolResult, TextContent

# Add openhab-mcp to path to import its models
# The openhab-mcp folder is a sibling to openHABAgents, not inside it.
_openhab_mcp_path = Path(__file__).parent.parent.parent / "openhab-mcp"
if str(_openhab_mcp_path) not in sys.path:
    sys.path.insert(0, str(_openhab_mcp_path))

from models import Item, Thing, Rule  # type: ignore  # imported from openhab-mcp package

from tools.rule_parser import ParsedRule, RuleParser


@dataclass
class SystemContext:
    """Complete system context including items, things, and rules."""

    items: List[Item]
    things: List[Thing]
    live_rules: List[Rule]  # Rules from live openHAB system
    local_rules: List[ParsedRule]  # Rules from generated_rules/ directory

    @property
    def all_rule_names(self) -> List[str]:
        """Get all rule names from both live and local sources."""
        names = [rule.name for rule in self.live_rules if rule.name]
        names.extend([rule.name for rule in self.local_rules if rule.name])
        return names

    @property
    def all_item_names(self) -> List[str]:
        """Get all item names."""
        return [item.name for item in self.items if item.name]

    def get_item(self, name: str) -> Optional[Item]:
        """Get item by name."""
        for item in self.items:
            if item.name == name:
                return item
        return None

    def get_thing(self, uid: str) -> Optional[Thing]:
        """Get thing by UID."""
        for thing in self.things:
            if thing.UID == uid:
                return thing
        return None


class SystemContextFetcher:
    """Fetches live openHAB system state AND local generated rules for validation via the MCP server."""

    def __init__(
        self,
        *,
        rules_dir: Optional[str] = None,
        mcp_command: Optional[str] = None,
        mcp_args: Optional[List[str]] = None,
    ):
        """Initialize context fetcher.

        Args:
            rules_dir: Directory containing generated rules (defaults to generated_rules/)
            mcp_command: Command to start the openHAB MCP server (defaults to sys.executable + openhab_mcp_server.py,
                         or OPENHAB_MCP_COMMAND env var if set)
            mcp_args: Arguments to pass to the MCP server command (defaults to [openhab_mcp_server.py] or
                      parsed from OPENHAB_MCP_ARGS env var if set)
        """
        self.rules_dir = Path(rules_dir or os.environ.get("OPENHAB_RULES_DIR", "generated_rules"))

        # Determine how to launch/connect to the MCP server.
        # By default, we spawn the local openhab_mcp_server.py via stdio.
        # The openhab-mcp folder is a sibling to openHABAgents, not inside it.
        project_root = Path(__file__).parent.parent  # openHABAgents/
        default_server_script = project_root.parent / "openhab-mcp" / "openhab_mcp_server.py"

        # Resolve command and args, robust to quoted/space-separated env values
        provided_cmd = mcp_command
        env_cmd = os.environ.get("OPENHAB_MCP_COMMAND")
        if provided_cmd:
            cmd_tokens = shlex.split(provided_cmd) if isinstance(provided_cmd, str) else [str(provided_cmd)]
        elif env_cmd:
            cmd_tokens = shlex.split(env_cmd)
        else:
            # Default to Python launching the bundled MCP server script
            cmd_tokens = [sys.executable, str(default_server_script)]

        self.mcp_command = cmd_tokens[0]
        base_args = cmd_tokens[1:]

        # Additional args from param/env
        if mcp_args is not None:
            extra_args = list(mcp_args)
        else:
            env_args = os.environ.get("OPENHAB_MCP_ARGS")
            extra_args = shlex.split(env_args) if env_args else []

        self.mcp_args = base_args + extra_args

        self.parser = RuleParser()

    async def _call_tool_json(
        self,
        session: ClientSession,
        name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Call an MCP tool and return its structured JSON content."""
        result: CallToolResult = await session.call_tool(name, arguments or {})

        if result.isError:
            raise RuntimeError(f"MCP tool '{name}' returned an error.")

        data = None
        if result.structuredContent is not None:
            data = result.structuredContent
        else:
            # Fallback: try to parse JSON from the first text content block.
            for block in result.content:
                if isinstance(block, TextContent):
                    try:
                        data = json.loads(block.text)
                        break
                    except Exception:
                        continue

        if data is None:
            raise RuntimeError(f"MCP tool '{name}' did not return structured JSON content.")

        # MCP FastMCP wraps tool returns in a "result" key - unwrap if present
        if isinstance(data, dict) and "result" in data and len(data) == 1:
            data = data["result"]

        return data

    async def _fetch_all_async(self) -> SystemContext:
        """Async helper to fetch items, things, and rules from MCP server + local .rules files."""
        print(f"Fetching system context via MCP server using command: {self.mcp_command} {' '.join(self.mcp_args)}")

        # Support both newer and older mcp stdio_client signatures:
        # - Newer: stdio_client(StdioServerParameters(command=..., args=[...]))
        # - Older: stdio_client(command, args_list)
        ParamsClass = getattr(mcp_stdio, "StdioServerParameters", None)
        if ParamsCore := ParamsClass:
            params = ParamsCore(command=self.mcp_command, args=list(self.mcp_args or []))
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # Fetch items
                    items_data = await self._call_tool_json(
                        session,
                        "list_items",
                        {"page": 1, "page_size": 1000},
                    )
                    raw_items = items_data.get("items", []) if isinstance(items_data, dict) else items_data
                    items = [Item.model_validate(it) for it in raw_items]
                    print(f"  ✓ Loaded {len(items)} items")

                    # Fetch things
                    things_data = await self._call_tool_json(
                        session,
                        "list_things",
                        {"page": 1, "page_size": 1000},
                    )
                    raw_things = things_data.get("things", []) if isinstance(things_data, dict) else things_data
                    things = [Thing.model_validate(th) for th in raw_things]
                    print(f"  ✓ Loaded {len(things)} things")

                    # Fetch rules (tolerate failures so we still return partial context)
                    try:
                        rules_data = await self._call_tool_json(session, "list_rules")
                        if isinstance(rules_data, dict) and "rules" in rules_data:
                            raw_rules = rules_data["rules"]
                        else:
                            raw_rules = rules_data
                        live_rules = [Rule.model_validate(r) for r in raw_rules]
                        print(f"  ✓ Loaded {len(live_rules)} live rules")
                    except Exception as e:
                        print(f"  ⚠ Warning: Could not load live rules via MCP: {e}")
                        live_rules = []
        else:
            async with stdio_client(self.mcp_command, list(self.mcp_args or [])) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # Fetch items
                    items_data = await self._call_tool_json(
                        session,
                        "list_items",
                        {"page": 1, "page_size": 1000},
                    )
                    raw_items = items_data.get("items", []) if isinstance(items_data, dict) else items_data
                    items = [Item.model_validate(it) for it in raw_items]
                    print(f"  ✓ Loaded {len(items)} items")

                    # Fetch things
                    things_data = await self._call_tool_json(
                        session,
                        "list_things",
                        {"page": 1, "page_size": 1000},
                    )
                    raw_things = things_data.get("things", []) if isinstance(things_data, dict) else things_data
                    things = [Thing.model_validate(th) for th in raw_things]
                    print(f"  ✓ Loaded {len(things)} things")

                    # Fetch rules (tolerate failures so we still return partial context)
                    try:
                        rules_data = await self._call_tool_json(session, "list_rules")
                        if isinstance(rules_data, dict) and "rules" in rules_data:
                            raw_rules = rules_data["rules"]
                        else:
                            raw_rules = rules_data
                        live_rules = [Rule.model_validate(r) for r in raw_rules]
                        print(f"  ✓ Loaded {len(live_rules)} live rules")
                    except Exception as e:
                        print(f"  ⚠ Warning: Could not load live rules via MCP: {e}")
                        live_rules = []

        # Load previously generated rules from disk
        local_rules = self._load_local_rules()
        print(f"  ✓ Loaded {len(local_rules)} local rules from {self.rules_dir}")

        return SystemContext(
            items=items,
            things=things,
            live_rules=live_rules,
            local_rules=local_rules,
        )

    def fetch_all(self) -> SystemContext:
        """Fetch items, things, and rules from MCP server + local .rules files (synchronous wrapper)."""
        return asyncio.run(self._fetch_all_async())

    def _load_local_rules(self) -> List[ParsedRule]:
        """Parse all .rules files in generated_rules/ directory."""
        parsed_rules: List[ParsedRule] = []

        if not self.rules_dir.exists():
            return parsed_rules

        for rules_file in self.rules_dir.glob("*.rules"):
            try:
                file_rules = self.parser.parse_rules_file(rules_file)
                parsed_rules.extend(file_rules)
            except Exception as e:
                print(f"  Warning: Could not parse {rules_file}: {e}")

        return parsed_rules


