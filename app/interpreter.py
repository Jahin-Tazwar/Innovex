"""
LLM operator-note interpretation.

This is the mandatory language-model stage: it is the only path from note text
to a directive. There is no keyword matcher and no phrase table anywhere in the
service. If the model returns nothing usable, notes become no_op and the
schedule is built without them -- we never guess a constraint from the text.

Everything this module emits is treated as untrusted. app/guardrails.py rebuilds
each entry into the exact shape the Problem Statement requires, so the prompt
optimises for semantic accuracy rather than perfect JSON discipline.

Providers: groq and openai share one OpenAI-compatible code path. Set
LLM_PROVIDER plus the matching key; see app/config.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Sequence

import httpx

from .config import settings
from .schemas import Battery

logger = logging.getLogger(__name__)

# OpenAI-compatible endpoints, keyed by provider.
_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
}

_DEFAULT_MODELS = {
    # Groq retired the Llama 3.3 endpoints; gpt-oss-120b is the strongest
    # instruction-following model this account can reach (GET /v1/models).
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
}


class InterpreterUnavailable(RuntimeError):
    """The model could not be reached, or gave nothing usable."""


# The prompt is kept deliberately tight. Groq's free tier caps us at 8000
# tokens per minute, and every token here is spent on every request, so the
# examples are compressed to one line each rather than a full few-shot turn.
SYSTEM_PROMPT = """\
Convert campus energy operator notes into directives for a 24-hour electricity \
optimizer. Return exactly one entry per note, in order.

TYPES (use only these):
solar_reduction          {"hours":[int],"factor":0..1}
minimum_battery_reserve  {"hours":[int],"minimum_energy_kwh":num}
no_charge_window         {"hours":[int]}
no_discharge_window      {"hours":[int]}
max_grid_window          {"hours":[int],"max_grid_kwh":num}
no_op                    null

RULE A - END HOUR IS EXCLUSIVE. "1 PM to 3 PM"->[13,14]. "6 PM until 9 PM"->
[18,19,20]. "noon to 2 PM"->[12,13]. "during the 10 AM hour"->[10]. Hours are
unique ints 0-23, ascending. Midnight=0, noon=12.

RULE B - factor IS THE FRACTION REMAINING, not the loss. "drops to 20%"->0.2.
"80% reduction"->0.2. "one-fifth of normal"->0.2. "halved"->0.5. "no solar"->0.

RULE C - A reserve given as a percentage or fraction of the battery must be
converted to kWh using the capacity stated in the user message. With a 200 kWh
battery: "50% of capacity"->100, "half the battery"->100, "a quarter"->50.

no_op = anything not about solar output, battery charging/discharging/reserve,
or grid import limits: menus, deadlines, meetings, bookings, exams, unrelated
maintenance. Also no_op if it applies to another day. Never invent numbers or
types. Keep explanation under 8 words.

EXAMPLES:
"PV drops to about 20% between 13:00 and 15:00" -> solar_reduction {"hours":[13,14],"factor":0.2}
"panel washing one until three, one-fifth of normal output" -> solar_reduction {"hours":[13,14],"factor":0.2}
"expect an 80% reduction in rooftop solar 1-3 PM" -> solar_reduction {"hours":[13,14],"factor":0.2}
"hold off charging the battery between 2 PM and 4 PM" -> no_charge_window {"hours":[14,15]}
"do not draw the battery down 5 to 8 PM" -> no_discharge_window {"hours":[17,18,19]}
"keep at least 120 kWh in reserve from 6 PM until 9 PM" -> minimum_battery_reserve {"hours":[18,19,20],"minimum_energy_kwh":120}
"grid import must stay under 150 kWh from 7 PM to 10 PM" -> max_grid_window {"hours":[19,20,21],"max_grid_kwh":150}
"the sports office moved the registration deadline" -> no_op null

Return ONLY this JSON object:
{"interpretations":[{"note_index":0,"directive_type":"...","structured_adjustment":{...} or null,"explanation":"..."}]}
"""


def _build_messages(
    notes: Sequence[str], battery: Battery | None = None
) -> List[Dict[str, str]]:
    listing = "\n".join(f"[{index}] {note.strip()}" for index, note in enumerate(notes))
    # Capacity is needed for RULE C: notes phrase reserves as "50% of capacity".
    context = (
        f"Battery capacity: {battery.capacity_kwh:g} kWh.\n\n" if battery else ""
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{context}Interpret these operator notes:\n\n{listing}\n",
        },
    ]


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _extract_entries(content: str) -> List[Dict[str, Any]]:
    """Pull the interpretation list out of a model response."""
    text = _FENCE.sub("", content.strip())
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Last resort: grab the outermost {...} the model wrapped in prose.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise InterpreterUnavailable("model response was not JSON")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise InterpreterUnavailable("model response was not JSON") from exc

    if isinstance(parsed, list):
        entries = parsed
    elif isinstance(parsed, dict):
        entries = (
            parsed.get("interpretations")
            or parsed.get("directive_interpretation")
            or parsed.get("results")
        )
        if entries is None:
            # A single-entry object is acceptable too.
            entries = [parsed] if "directive_type" in parsed else []
    else:
        entries = []

    if not isinstance(entries, list):
        raise InterpreterUnavailable("model response had no interpretation list")
    return [entry for entry in entries if isinstance(entry, dict)]


class RateLimited(InterpreterUnavailable):
    """Provider returned 429. Carries the server's own wait hint."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"rate limited, retry after {retry_after:.1f}s")
        self.retry_after = retry_after


def _retry_after_seconds(response: httpx.Response) -> float:
    """Seconds to wait, from Retry-After or the token-bucket reset header."""
    raw = response.headers.get("retry-after") or response.headers.get(
        "x-ratelimit-reset-tokens", ""
    )
    raw = raw.strip().lower()
    if not raw:
        return 2.0
    match = re.match(r"^(?:(\d+(?:\.\d+)?)m)?(\d+(?:\.\d+)?)s?$", raw)
    if match:
        minutes = float(match.group(1) or 0)
        return minutes * 60 + float(match.group(2))
    try:
        return float(raw)
    except ValueError:
        return 2.0


# Identical note sets recur across judge retries and our own test runs; a hit
# costs nothing and spends no tokens against the per-minute budget.
_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_CACHE_LIMIT = 256


async def _call_openai_compatible(
    notes: Sequence[str], provider: str, battery: Battery
) -> List[Dict[str, Any]]:
    url = _ENDPOINTS[provider]
    model = settings.LLM_MODEL or _DEFAULT_MODELS[provider]
    payload = {
        "model": model,
        "messages": _build_messages(notes, battery),
        "temperature": 0,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }
    if settings.LLM_REASONING_EFFORT and "gpt-oss" in model:
        # gpt-oss spends hidden reasoning tokens against the same per-minute
        # budget. "low" gave identical answers at ~45% of the tokens and a
        # third of the latency on the public cases.
        payload["reasoning_effort"] = settings.LLM_REASONING_EFFORT
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code == 429:
            raise RateLimited(_retry_after_seconds(response))
        if response.status_code != 200:
            # The body may echo our payload; log the status only.
            raise InterpreterUnavailable(
                f"{provider} returned HTTP {response.status_code}"
            )
        body = response.json()

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InterpreterUnavailable("unexpected response envelope") from exc

    return _extract_entries(content or "")


async def interpret_notes(notes: List[str], battery: Battery) -> List[Dict[str, Any]]:
    """Interpret every operator note. One raw entry per note, in note order."""
    provider = settings.LLM_PROVIDER

    if provider == "stub":
        # Offline mode for optimizer work. Everything becomes no_op.
        logger.warning("LLM_PROVIDER=stub -- notes are not being interpreted")
        return []

    if provider not in _ENDPOINTS:
        raise InterpreterUnavailable(f"provider '{provider}' is not supported")

    if not settings.api_key:
        raise InterpreterUnavailable(f"no API key configured for '{provider}'")

    cache_key = json.dumps(
        [settings.LLM_MODEL, battery.capacity_kwh, list(notes)],
        separators=(",", ":"),
        sort_keys=True,
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        logger.info("interpretation cache hit")
        return cached

    deadline = asyncio.get_running_loop().time() + settings.LLM_BUDGET_SECONDS
    attempts = max(1, settings.LLM_MAX_RETRIES + 1)
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            entries = await _call_openai_compatible(notes, provider, battery)
            if entries:
                if len(_CACHE) >= _CACHE_LIMIT:
                    _CACHE.clear()
                _CACHE[cache_key] = entries
                return entries
            last_error = InterpreterUnavailable("model returned an empty list")
            wait = 0.4 * (attempt + 1)
        except RateLimited as exc:
            last_error = exc
            wait = exc.retry_after
            logger.warning("rate limited; provider asked for %.1fs", wait)
        except (httpx.HTTPError, InterpreterUnavailable) as exc:
            last_error = exc
            wait = 0.4 * (attempt + 1)
            logger.warning(
                "interpretation attempt %d/%d failed: %s",
                attempt + 1,
                attempts,
                type(exc).__name__,
            )

        if attempt + 1 >= attempts:
            break
        # Only wait if there is still room inside the request budget; the judge
        # times us out at 30s and a valid schedule beats a perfect one that
        # arrives too late.
        remaining = deadline - asyncio.get_running_loop().time()
        if wait >= remaining:
            logger.warning("skipping retry: %.1fs wait exceeds remaining budget", wait)
            break
        await asyncio.sleep(wait)

    raise InterpreterUnavailable(
        str(last_error) if last_error else "interpretation failed"
    )
