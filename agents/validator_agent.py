from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


@dataclass
class ValidationResult:
    verdict: str
    summary: str
    feedback: str
    fixes: List[str]
    raw_output: str

    @property
    def is_valid(self) -> bool:
        return self.verdict.lower().startswith("valid")

    def as_feedback_entry(self) -> str:
        if self.is_valid:
            return self.summary
        lines = [self.summary]
        if self.feedback:
            lines.append(self.feedback)
        if self.fixes:
            lines.extend(self.fixes)
        return "\n".join(lines)


class ValidatorAgent:
    """LangChain agent responsible for validating generated openHAB rules."""

    SYSTEM_PROMPT = (
        "You are an openHAB *syntax* validator. Given a user request, supporting context, "
        "and the candidate openHAB DSL code, determine whether the code is syntactically valid and "
        "structurally sound.\n\n"
        "Your responsibilities:\n"
        "- Check openHAB DSL syntax (rule blocks, if/else, brackets, parentheses, etc.).\n"
        "- Check for missing or malformed triggers/actions.\n"
        "- Flag obviously broken code-level logic (e.g., unreachable code due to malformed condition), "
        "but ONLY when it can be determined from syntax/structure alone.\n\n"
        "IMPORTANT LIMITATION:\n"
        "- DO NOT check whether items, things, channels, or other resources actually exist or are defined.\n"
        "- Assume that all referenced items/things/channels are valid and defined in the system.\n"
        "- DO NOT mark a rule as invalid just because an item name might not exist; that is the responsibility "
        "of a separate context validator that has access to the live system state.\n\n"
        "Respond strictly as a JSON object with keys: "
        "'verdict' ('valid' or 'invalid'), 'summary' (short sentence), "
        "'feedback' (string, may be empty), and optional 'fixes' (array of strings). "
        "Do not include markdown, code fences, or additional commentary."
    )

    def __init__(
        self,
        *,
        model: str = "gemini-3-flash-preview",
        temperature: float = 1.5,
        llm: ChatGoogleGenerativeAI | None = None,
    ) -> None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.llm = llm or ChatGoogleGenerativeAI(
            model=model, temperature=temperature, google_api_key=api_key
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT + "\nPrior validator feedback:\n{feedback}"),
                (
                    "user",
                    "User request:\n{request}\n\nContext snippets:\n{context}\n\n"
                    "Candidate openHAB code:\n{candidate_code}",
                ),
            ]
        )

    def validate(
        self,
        *,
        request: str,
        context: str,
        feedback: str,
        candidate_code: str,
    ) -> ValidationResult:
        inputs = {
            "input": candidate_code,
            "request": request,
            "context": context,
            "feedback": feedback,
            "candidate_code": candidate_code,
        }
        prompt_messages = self.prompt.invoke(inputs)
        response = self.llm.invoke(prompt_messages)
        # Gemini 3 returns content as list of blocks; .text gives string
        output = getattr(response, "text", None) or response.content
        if isinstance(output, list):
            output = "".join(
                b.get("text", b) if isinstance(b, dict) else str(b) for b in output
            )
        output = (output or "").strip()
        parsed = self._parse_output(output)
        return ValidationResult(
            verdict=parsed.get("verdict", "invalid"),
            summary=parsed.get("summary", "Validator could not parse response."),
            feedback=parsed.get("feedback", ""),
            fixes=parsed.get("fixes", []),
            raw_output=output,
        )

    @staticmethod
    def _parse_output(output: str) -> dict:
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            sanitized = output.strip().strip("`")
            try:
                return json.loads(sanitized)
            except json.JSONDecodeError:
                return {
                    "verdict": "invalid",
                    "summary": "Unable to decode validator response.",
                    "feedback": sanitized,
                    "fixes": [],
                }

