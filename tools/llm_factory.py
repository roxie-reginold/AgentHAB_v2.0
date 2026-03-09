"""LLM factory – returns a LangChain chat model for the configured provider.

Supported providers (set via LLM_PROVIDER env var or the `provider` argument):
  gemini   – Google Gemini via langchain-google-genai (default)
  lmstudio – LM Studio local server (OpenAI-compatible, http://localhost:1234/v1)
  ollama   – Any locally-served model via Ollama (http://localhost:11434 by default)

Environment variables
---------------------
LLM_PROVIDER          "gemini" | "lmstudio" | "ollama"   (default: "gemini")

Gemini:
  GEMINI_API_KEY      Required API key
  GOOGLE_API_KEY      Fallback for GEMINI_API_KEY
  GEMINI_MODEL        Default model name  (default: "gemini-2.0-flash")

LM Studio:
  LMSTUDIO_BASE_URL   Server URL  (default: "http://localhost:1234/v1")
  LMSTUDIO_MODEL      Model identifier shown in LM Studio (e.g. "google/gemma-3-4b")

Ollama:
  OLLAMA_BASE_URL     Server URL  (default: "http://localhost:11434")
  OLLAMA_MODEL        Model name  (default: "llama3.2")
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel


def get_provider() -> str:
    """Return the active LLM provider name (lower-cased)."""
    return os.environ.get("LLM_PROVIDER", "gemini").lower()


def build_llm(
    model: str,
    temperature: float,
    *,
    provider: str | None = None,
    api_key: str | None = None,
) -> BaseChatModel:
    """Build and return the appropriate LangChain chat model.

    Args:
        model:       Model name/identifier.
        temperature: Sampling temperature.
        provider:    Override the LLM_PROVIDER env var.
        api_key:     Gemini API key override (ignored for local providers).
    """
    resolved_provider = (provider or get_provider()).lower()

    if resolved_provider == "lmstudio":
        return _build_lmstudio(model=model, temperature=temperature)
    elif resolved_provider == "ollama":
        return _build_ollama(model=model, temperature=temperature)
    else:
        return _build_gemini(model=model, temperature=temperature, api_key=api_key)


def _build_gemini(
    model: str,
    temperature: float,
    *,
    api_key: str | None = None,
) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

    resolved_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not resolved_key:
        raise ValueError(
            "No Gemini API key found. Set GEMINI_API_KEY in your .env file, "
            "or switch to LLM_PROVIDER=lmstudio to use LM Studio."
        )
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=resolved_key,
    )


def _build_lmstudio(model: str, temperature: float) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is not installed. Run: pip install langchain-openai"
        ) from exc

    base_url = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        # LM Studio accepts any non-empty string as the API key
        api_key="lm-studio",
    )


def _build_ollama(model: str, temperature: float) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is not installed. Run: pip install langchain-ollama"
        ) from exc

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    return ChatOllama(model=model, temperature=temperature, base_url=base_url)


def default_model(provider: str | None = None) -> str:
    """Return the configured default model name for the given provider."""
    resolved = (provider or get_provider()).lower()
    if resolved == "lmstudio":
        return os.environ.get("LMSTUDIO_MODEL", "google/gemma-3-4b")
    if resolved == "ollama":
        return os.environ.get("OLLAMA_MODEL", "llama3.2")
    return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
