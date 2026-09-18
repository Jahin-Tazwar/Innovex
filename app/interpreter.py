"""
LLM operator-note interpretation -- PERSON B OWNS THIS FILE.

Person A has wired the plumbing; the model call itself is yours. Everything
downstream is already built, so you only have to satisfy this one signature:

    async def interpret_notes(notes: list[str], battery: Battery) -> list[dict]

Return one raw dict per note, in note order. Shape you should aim for:

    {
      "note_index": 0,
      "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
      "explanation": "Solar availability is reduced during panel cleaning."
    }

You do NOT need to be perfect about it. app/guardrails.py rebuilds every entry,
repairs missing/duplicate/out-of-order note_index values, sorts and dedupes
hours, converts "20" or "20%" into 0.2, sets `applies` for you, and degrades
anything unusable to no_op. Raise if the provider fails -- main.py catches it
and still returns a valid schedule.

Two conventions the prompt must state explicitly (the judge checks both):
  * End hour is EXCLUSIVE. "1 PM to 3 PM" -> hours [13, 14].
  * `factor` is the fraction REMAINING. "80% reduction" -> factor 0.2.

Provider is chosen with LLM_PROVIDER + the matching *_API_KEY env var; see
app/config.py. Add your SDK to requirements.txt if you do not use raw httpx.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .config import settings
from .schemas import Battery

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You convert campus energy operator notes into structured directives.
TODO(B): write the real prompt here. Cover the six directive types, the
exclusive end hour, and factor-as-remaining-fraction.
"""


class InterpreterUnavailable(RuntimeError):
    """Raised when the model could not be reached or gave unusable output."""


async def interpret_notes(notes: List[str], battery: Battery) -> List[Dict[str, Any]]:
    """Interpret every operator note. One raw entry per note, in note order."""
    if settings.LLM_PROVIDER == "stub":
        # Dev placeholder so the pipeline runs end to end before B lands.
        # Everything degrades to no_op, which is a valid (if unscored) answer.
        logger.warning("LLM_PROVIDER=stub -- notes are not being interpreted")
        return []

    if not settings.api_key:
        raise InterpreterUnavailable(
            f"no API key configured for provider '{settings.LLM_PROVIDER}'"
        )

    # TODO(B): one call for all notes, low temperature, structured/JSON output,
    # directive_type constrained to the six allowed values, few-shot paraphrases.
    # Honour settings.LLM_TIMEOUT_SECONDS and settings.LLM_MAX_RETRIES.
    raise InterpreterUnavailable(
        f"interpreter for provider '{settings.LLM_PROVIDER}' is not implemented yet"
    )
