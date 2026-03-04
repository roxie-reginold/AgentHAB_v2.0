"""Context-aware validator that checks rules against live openHAB system state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from tools.context_fetcher import SystemContext
from tools.rule_parser import RuleParser


@dataclass
class ContextValidationResult:
    """Result of context-aware validation."""
    
    verdict: str  # "valid" or "invalid"
    summary: str
    feedback: str
    fixes: List[str]
    warnings: List[str]
    raw_output: str
    
    @property
    def is_valid(self) -> bool:
        return self.verdict.lower().startswith("valid")
    
    def as_feedback_entry(self) -> str:
        """Format validation result as feedback for the generator."""
        if self.is_valid:
            if self.warnings:
                return f"{self.summary}\nWarnings:\n" + "\n".join(f"  - {w}" for w in self.warnings)
            return self.summary
        
        lines = [self.summary]
        if self.feedback:
            lines.append(self.feedback)
        if self.fixes:
            lines.append("Suggested fixes:")
            lines.extend(f"  - {fix}" for fix in self.fixes)
        return "\n".join(lines)


class ContextValidatorAgent:
    """LLM-based validator that checks rules against live system state."""
    
    SYSTEM_PROMPT = """You are an expert openHAB automation security and validation specialist.

Your role is to validate candidate openHAB DSL rules against the LIVE SYSTEM STATE to ensure:
1. All referenced items actually exist in the system
2. Item types are compatible with actions (e.g., can't dim a Switch, must use Dimmer)
3. No conflicts with DEPLOYED rules (duplicate triggers, contradictory actions)
4. No security vulnerabilities or dangerous patterns
5. Things backing items are online and functional

You have access to:
- Complete list of items with their types and current states
- Complete list of things with their status
- All DEPLOYED rules currently running in openHAB (for conflict detection)
- The candidate rule to validate

IMPORTANT: Only check for conflicts against DEPLOYED rules in openHAB. Ignore any local files or drafts.

IMPORTANT SECURITY REASONING:
- Use your knowledge to identify dangerous automation patterns (HVAC conflicts, water hazards, etc.)
- Detect potential infinite loops where rules trigger each other
- Flag contradictory actions (e.g., locking and unlocking simultaneously)
- Consider timing issues and race conditions
- Think about physical safety implications

Respond ONLY as a JSON object with these keys:
{
  "verdict": "valid" or "invalid",
  "summary": "Brief one-sentence summary",
  "feedback": "Detailed explanation of issues (empty if valid)",
  "fixes": ["List", "of", "specific", "fixes"],
  "warnings": ["Non-blocking", "warnings", "if any"]
}

Do NOT include markdown, code fences, or additional commentary."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,  # Lower temperature for more consistent validation
        llm: ChatOpenAI | None = None,
    ) -> None:
        self.llm = llm or ChatOpenAI(model=model, temperature=temperature)
        self.parser = RuleParser()
        # IMPORTANT: Escape JSON braces in SYSTEM_PROMPT so ChatPromptTemplate
        # doesn't treat JSON keys like "verdict" as template variables.
        safe_system_prompt = self.SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", safe_system_prompt),
                (
                    "user",
                    "User request:\n{request}\n\n"
                    "LIVE SYSTEM STATE:\n{system_context}\n\n"
                    "CANDIDATE RULE TO VALIDATE:\n{candidate_code}\n\n"
                    "Validate this rule against the live system state and existing rules. "
                    "Check for item existence, type compatibility, conflicts, and security issues."
                ),
            ]
        )
    
    def validate(
        self,
        *,
        candidate_code: str,
        system_context: SystemContext,
        request: str,
    ) -> ContextValidationResult:
        """Validate candidate rule against live system context.
        
        Args:
            candidate_code: The generated rule to validate
            system_context: Live system state (items, things, rules)
            request: Original user request for context
            
        Returns:
            ContextValidationResult with validation verdict and feedback
        """
        # Format system context for LLM
        context_str = self._format_system_context(system_context, candidate_code)
        
        # Invoke LLM
        prompt_messages = self.prompt.invoke(
            {
                "request": request,
                "system_context": context_str,
                "candidate_code": candidate_code,
            }
        )
        response = self.llm.invoke(prompt_messages)
        output = (response.content or "").strip()
        
        # Parse response
        parsed = self._parse_output(output)
        
        return ContextValidationResult(
            verdict=parsed.get("verdict", "invalid"),
            summary=parsed.get("summary", "Context validator could not parse response."),
            feedback=parsed.get("feedback", ""),
            fixes=parsed.get("fixes", []),
            warnings=parsed.get("warnings", []),
            raw_output=output,
        )
    
    def _format_system_context(self, context: SystemContext, candidate_code: str) -> str:
        """Format system context for LLM consumption."""
        lines = []
        
        # Parse candidate rule to identify referenced items
        parsed_candidate = self.parser.parse_rule(candidate_code)
        referenced_items = parsed_candidate.all_items if parsed_candidate else []
        
        # 1. Items section - focus on referenced items plus show all available
        lines.append("=== AVAILABLE ITEMS ===")
        if context.items:
            # Show referenced items first (with details)
            referenced_found = []
            for item_name in referenced_items:
                item = context.get_item(item_name)
                if item:
                    referenced_found.append(item)
                    lines.append(
                        f"  ✓ {item.name} (type: {item.type}, state: {item.state or 'NULL'}, "
                        f"tags: {item.tags or []})"
                    )
            
            # List items NOT referenced but showing they exist
            if len(referenced_found) < len(referenced_items):
                missing = set(referenced_items) - {i.name for i in referenced_found}
                lines.append("\n  MISSING ITEMS (referenced but not found in system):")
                for item_name in missing:
                    lines.append(f"    ✗ {item_name} - DOES NOT EXIST")
            
            # Show sample of other available items
            other_items = [i for i in context.items if i.name not in referenced_items]
            if other_items:
                lines.append(f"\n  Other available items ({len(other_items)} total): " +
                           ", ".join(i.name for i in other_items[:20]))
                if len(other_items) > 20:
                    lines.append(f"    ... and {len(other_items) - 20} more")
        else:
            lines.append("  (No items in system)")
        
        # 2. Things section - show status
        lines.append("\n=== THINGS STATUS ===")
        if context.things:
            online_count = sum(1 for t in context.things if t.statusInfo and t.statusInfo.status == "ONLINE")
            offline_count = len(context.things) - online_count
            lines.append(f"  Total: {len(context.things)} ({online_count} online, {offline_count} offline)")
            
            # Show offline things as they might affect validation
            offline = [t for t in context.things if t.statusInfo and t.statusInfo.status != "ONLINE"]
            if offline:
                lines.append("  Offline things:")
                for thing in offline[:10]:
                    lines.append(f"    - {thing.UID} ({thing.statusInfo.status if thing.statusInfo else 'UNKNOWN'})")
        else:
            lines.append("  (No things in system)")
        
        # 3. Deployed rules section - crucial for conflict detection
        # NOTE: Only check DEPLOYED (live) rules for conflicts. Local .rules files
        # are just drafts on disk and should not trigger conflict warnings.
        lines.append("\n=== DEPLOYED RULES IN OPENHAB ===")
        
        if context.live_rules:
            lines.append(f"Currently deployed rules ({len(context.live_rules)}):")
            for rule in context.live_rules[:10]:  # Show first 10
                lines.append(f"  - {rule.name or rule.uid}")
                if rule.triggers:
                    trigger_summary = ", ".join(
                        f"{t.type}({t.configuration})" for t in rule.triggers[:2]
                    )
                    lines.append(f"    Triggers: {trigger_summary}")
            if len(context.live_rules) > 10:
                lines.append(f"    ... and {len(context.live_rules) - 10} more")
        else:
            lines.append("  (No rules currently deployed in openHAB)")
        
        return "\n".join(lines)
    
    @staticmethod
    def _parse_output(output: str) -> dict:
        """Parse LLM JSON output."""
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            # Try to clean up common issues
            sanitized = output.strip().strip("`")
            # Remove markdown json tags
            sanitized = sanitized.replace("```json", "").replace("```", "")
            try:
                return json.loads(sanitized)
            except json.JSONDecodeError:
                return {
                    "verdict": "invalid",
                    "summary": "Context validator response could not be parsed.",
                    "feedback": sanitized,
                    "fixes": [],
                    "warnings": [],
                }

