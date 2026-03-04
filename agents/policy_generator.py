from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


@dataclass
class GenerationResult:
    openhab_code: str
    reasoning: Optional[str] = None


def _load_api_keys() -> List[str]:
    """
    Collect all Gemini API keys from the environment.

    Reads GEMINI_API_KEY / GOOGLE_API_KEY as the primary key, then
    GEMINI_API_KEY_2, GEMINI_API_KEY_3, … for additional rotation keys.
    """
    keys: List[str] = []
    primary = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if primary:
        keys.append(primary)
    for i in range(2, 20):
        extra = os.environ.get(f"GEMINI_API_KEY_{i}")
        if extra:
            keys.append(extra)
        else:
            break
    return keys


class PolicyGeneratorAgent:
    """LangChain agent responsible for synthesising openHAB rules."""

    SYSTEM_PROMPT = (
        "You are an expert openHAB policy engineer creating DSL rules. "
        "Rely on the supplied context snippets summarising syntax, grammar, and examples. "
        "Incorporate prior validator feedback when present. Respond with openHAB code only."
    )

    def __init__(
        self,
        *,
        model: str = "gemini-3-flash-preview",
        temperature: float = 1.5,
        llm: ChatGoogleGenerativeAI | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._api_keys = _load_api_keys()
        self._key_index = 0  # which key is currently active

        if llm is not None:
            self.llm = llm
        else:
            if not self._api_keys:
                raise ValueError("No Gemini API key found. Set GEMINI_API_KEY in .env.")
            self.llm = self._make_llm(self._api_keys[0])

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.SYSTEM_PROMPT
                    + "\n\nContext snippets:\n{context}\n\n"
                    "Outstanding feedback to resolve:\n{feedback}\n\n"
                    "Prior candidate rule (revise if provided):\n{prior_code}",
                ),
                ("user", "User request:\n{request}"),
            ]
        )

    def _make_llm(self, api_key: str) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self._model,
            temperature=self._temperature,
            google_api_key=api_key,
        )

    def _rotate_key(self) -> bool:
        """Switch to the next available API key. Returns True if a new key was available."""
        if len(self._api_keys) <= 1:
            return False
        self._key_index = (self._key_index + 1) % len(self._api_keys)
        new_key = self._api_keys[self._key_index]
        self.llm = self._make_llm(new_key)
        masked = new_key[:8] + "…"
        print(f"  ↻ Rotated to API key {self._key_index + 1}/{len(self._api_keys)} ({masked})")
        return True

    @staticmethod
    def _parse_retry_delay(error_msg: str) -> float | None:
        """Extract the suggested retry delay (seconds) from a rate-limit error message."""
        match = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", error_msg, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 2  # small buffer on top of suggested delay
        return None

    @staticmethod
    def _is_quota_error(err: str) -> bool:
        return any(code in err for code in ("429", "RESOURCE_EXHAUSTED"))

    @staticmethod
    def _is_retryable(err: str) -> bool:
        return any(code in err for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))

    def generate(
        self,
        *,
        request: str,
        context: str,
        feedback: str,
        prior_code: str,
        max_retries: int = 6,
    ) -> GenerationResult:
        """Invoke the agent and return the generated openHAB code."""
        prompt_messages = self.prompt.invoke(
            {
                "request": request,
                "context": context,
                "feedback": feedback,
                "prior_code": prior_code,
            }
        )
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt_messages)
                break
            except Exception as exc:
                err = str(exc)
                if not self._is_retryable(err) or attempt >= max_retries - 1:
                    raise

                # On quota exhaustion, try rotating to the next key first
                if self._is_quota_error(err) and self._rotate_key():
                    print(f"  ⚠ Quota exhausted on key {self._key_index} – switched key, retrying immediately...")
                    continue

                # Fall back to waiting (use API-suggested delay or exponential backoff)
                wait = self._parse_retry_delay(err) or (2 ** attempt)
                print(f"  ⚠ Rate limited – retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)

        # Gemini returns content as list of blocks; .text gives string
        code = (getattr(response, "text", None) or response.content or "")
        if isinstance(code, list):
            code = "".join(
                b.get("text", b) if isinstance(b, dict) else str(b) for b in code
            )
        code = str(code).strip() if code else ""
        return GenerationResult(openhab_code=code, reasoning=None)
