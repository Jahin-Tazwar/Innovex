"""Environment configuration. No secret values live in this file."""

from __future__ import annotations

import os


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    """Read once at import. Restart the service to pick up changes."""

    # Which provider Person B's interpreter talks to.
    # One of: anthropic | openai | gemini | stub
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "stub").strip().lower()
    LLM_MODEL: str = os.getenv("LLM_MODEL", "").strip()

    # Only the name that matches LLM_PROVIDER needs to be set.
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # Judge allows 30s per request; we stop the model well before that so there
    # is always time left to run the optimizer and still answer.
    LLM_TIMEOUT_SECONDS: float = _num("LLM_TIMEOUT_SECONDS", 12.0)
    LLM_MAX_RETRIES: int = int(_num("LLM_MAX_RETRIES", 1))

    SOLVER_TIMEOUT_SECONDS: float = _num("SOLVER_TIMEOUT_SECONDS", 10.0)

    PORT: int = int(_num("PORT", 8000))

    # Verbose per-request logging. Never logs note text or model output.
    DEBUG: bool = _flag("DEBUG", False)

    @property
    def api_key(self) -> str:
        return {
            "anthropic": self.ANTHROPIC_API_KEY,
            "openai": self.OPENAI_API_KEY,
            "gemini": self.GOOGLE_API_KEY,
        }.get(self.LLM_PROVIDER, "")

    @property
    def llm_configured(self) -> bool:
        return self.LLM_PROVIDER != "stub" and bool(self.api_key)


settings = Settings()
