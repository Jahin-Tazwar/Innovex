"""
Deterministic guardrails.

This is the layer that makes LLM output safe. Nothing the model returns is
trusted: every entry is rebuilt from scratch into the exact shape the Problem
Statement requires (sections 05, 08). Anything we cannot rebuild confidently
degrades to no_op instead of inventing a constraint or raising.

Contract:
    normalize(raw_entries, notes, battery) -> (interpretations, directives)

  * `interpretations` always has exactly len(notes) entries, in note_index
    order 0..N-1. This is what goes straight into the response.
  * `directives` holds only the applicable ones, for the optimizer.

The function never raises. That is deliberate -- a bad model response must
degrade, not 500.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schemas import Battery, Directive, DirectiveInterpretation, DirectiveType

_WINDOW_TYPES = {
    DirectiveType.SOLAR_REDUCTION,
    DirectiveType.MINIMUM_BATTERY_RESERVE,
    DirectiveType.NO_CHARGE_WINDOW,
    DirectiveType.NO_DISCHARGE_WINDOW,
    DirectiveType.MAX_GRID_WINDOW,
}

NO_OP_EXPLANATION = "This note does not affect today's energy schedule."


def _finite(value: Any) -> Optional[float]:
    """Coerce to a finite float, or None. Tolerates numeric strings."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
    elif isinstance(value, str):
        try:
            out = float(value.strip().rstrip("%"))
        except (TypeError, ValueError):
            return None
    else:
        return None
    return out if math.isfinite(out) else None


def _clean_hours(value: Any) -> List[int]:
    """Unique ints in 0..23, ascending. Junk entries are dropped."""
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return []
    seen = set()
    for item in value:
        num = _finite(item)
        if num is None:
            continue
        hour = int(round(num))
        if 0 <= hour <= 23:
            seen.add(hour)
    return sorted(seen)


def _clean_factor(value: Any) -> Optional[float]:
    """
    factor is the fraction of solar that REMAINS (80% reduction -> 0.2).

    Models sometimes emit 20 or "20%" meaning 0.2, so anything in (1, 100] is
    read as a percentage. Small overshoots are clamped rather than discarded.
    """
    num = _finite(value)
    if num is None:
        return None
    if 1.0 < num <= 100.0:
        num = num / 100.0
    if num < 0.0:
        num = 0.0
    if num > 1.0:
        if num > 1.05:  # not a rounding artefact -- do not guess
            return None
        num = 1.0
    return num


def _directive_type(value: Any) -> Optional[DirectiveType]:
    if isinstance(value, DirectiveType):
        return value
    if not isinstance(value, str):
        return None
    try:
        return DirectiveType(value.strip().lower())
    except ValueError:
        return None


def _no_op(note_index: int, explanation: str = "") -> DirectiveInterpretation:
    return DirectiveInterpretation(
        note_index=note_index,
        applies=False,
        directive_type=DirectiveType.NO_OP,
        structured_adjustment=None,
        explanation=(explanation or NO_OP_EXPLANATION)[:400],
    )


def _build(
    note_index: int,
    raw: Dict[str, Any],
    battery: Battery,
) -> Tuple[DirectiveInterpretation, Optional[Directive]]:
    """Rebuild one entry into canonical form, or degrade it to no_op."""
    explanation = str(raw.get("explanation") or "")[:400]
    dtype = _directive_type(raw.get("directive_type"))

    if dtype is None or dtype is DirectiveType.NO_OP:
        return _no_op(note_index, explanation), None

    adjustment = raw.get("structured_adjustment")
    if not isinstance(adjustment, dict):
        # Some models flatten the fields onto the entry itself. Accept that.
        adjustment = raw

    hours = _clean_hours(adjustment.get("hours"))
    if dtype in _WINDOW_TYPES and not hours:
        return _no_op(note_index, explanation), None

    payload: Dict[str, Any] = {"hours": hours}
    directive = Directive(directive_type=dtype, hours=hours)

    if dtype is DirectiveType.SOLAR_REDUCTION:
        factor = _clean_factor(adjustment.get("factor"))
        if factor is None:
            return _no_op(note_index, explanation), None
        payload["factor"] = factor
        directive.factor = factor

    elif dtype is DirectiveType.MINIMUM_BATTERY_RESERVE:
        reserve = _finite(adjustment.get("minimum_energy_kwh"))
        if reserve is None or reserve < 0:
            return _no_op(note_index, explanation), None
        # A reserve above capacity is unsatisfiable; cap it so the LP stays feasible.
        reserve = min(reserve, battery.capacity_kwh)
        payload["minimum_energy_kwh"] = reserve
        directive.minimum_energy_kwh = reserve

    elif dtype is DirectiveType.MAX_GRID_WINDOW:
        cap = _finite(adjustment.get("max_grid_kwh"))
        if cap is None or cap < 0:
            return _no_op(note_index, explanation), None
        payload["max_grid_kwh"] = cap
        directive.max_grid_kwh = cap

    interpretation = DirectiveInterpretation(
        note_index=note_index,
        applies=True,
        directive_type=dtype,
        structured_adjustment=payload,
        explanation=explanation,
    )
    return interpretation, directive


def normalize(
    raw_entries: Any,
    notes: Sequence[str],
    battery: Battery,
) -> Tuple[List[DirectiveInterpretation], List[Directive]]:
    """
    Turn whatever the model produced into exactly one valid entry per note.

    Missing, duplicate and out-of-range note_index values are all repaired here;
    the judge treats those as schema failures, so we never let them through.
    """
    by_index: Dict[int, Dict[str, Any]] = {}

    if isinstance(raw_entries, dict):
        raw_entries = raw_entries.get("directive_interpretation") or raw_entries.get(
            "interpretations"
        )

    if isinstance(raw_entries, list):
        for position, entry in enumerate(raw_entries):
            if not isinstance(entry, dict):
                continue
            index = _finite(entry.get("note_index"))
            index = int(index) if index is not None else position
            if 0 <= index < len(notes) and index not in by_index:
                by_index[index] = entry

    interpretations: List[DirectiveInterpretation] = []
    directives: List[Directive] = []

    for note_index in range(len(notes)):
        entry = by_index.get(note_index)
        if entry is None:
            interpretations.append(_no_op(note_index))
            continue
        interpretation, directive = _build(note_index, entry, battery)
        interpretations.append(interpretation)
        if directive is not None:
            directives.append(directive)

    return interpretations, directives
