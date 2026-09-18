"""
Shared directive semantics (Problem Statement 5.3, 09).

Both the optimizer and the replay validator import from here on purpose: if the
two ever disagreed about what a directive means, we would produce plans that
pass our own checks and fail the judge's.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

from .schemas import Battery, Directive, DirectiveType, HourEntry


def effective_solar(
    hours: Sequence[HourEntry], directives: Sequence[Directive]
) -> List[float]:
    """Base solar after every solar_reduction directive is applied."""
    solar = [float(h.solar_kwh) for h in hours]
    for directive in directives:
        if directive.directive_type is not DirectiveType.SOLAR_REDUCTION:
            continue
        factor = 1.0 if directive.factor is None else directive.factor
        for hour in directive.hours:
            if 0 <= hour < len(solar):
                # Overlapping reductions compound; the stricter one wins out.
                solar[hour] *= factor
    return solar


def blocked_hours(
    directives: Sequence[Directive], directive_type: DirectiveType
) -> Set[int]:
    blocked: Set[int] = set()
    for directive in directives:
        if directive.directive_type is directive_type:
            blocked.update(directive.hours)
    return blocked


def no_charge_hours(directives: Sequence[Directive]) -> Set[int]:
    return blocked_hours(directives, DirectiveType.NO_CHARGE_WINDOW)


def no_discharge_hours(directives: Sequence[Directive]) -> Set[int]:
    return blocked_hours(directives, DirectiveType.NO_DISCHARGE_WINDOW)


def reserve_floors(
    battery: Battery, directives: Sequence[Directive], horizon: int = 24
) -> List[float]:
    """
    Per-hour lower bound on battery energy after the hour.

    The base minimum always applies; a minimum_battery_reserve directive can
    only raise it (Problem Statement 5.3, 9.2).
    """
    floors = [float(battery.minimum_energy_kwh)] * horizon
    for directive in directives:
        if directive.directive_type is not DirectiveType.MINIMUM_BATTERY_RESERVE:
            continue
        reserve = directive.minimum_energy_kwh
        if reserve is None:
            continue
        for hour in directive.hours:
            if 0 <= hour < horizon:
                floors[hour] = max(floors[hour], float(reserve))
    return floors


def grid_caps(
    directives: Sequence[Directive], horizon: int = 24
) -> Dict[int, float]:
    """Hour -> maximum grid import. Hours absent from the map are uncapped."""
    caps: Dict[int, float] = {}
    for directive in directives:
        if directive.directive_type is not DirectiveType.MAX_GRID_WINDOW:
            continue
        cap = directive.max_grid_kwh
        if cap is None:
            continue
        for hour in directive.hours:
            if 0 <= hour < horizon:
                # Overlapping caps: the tightest one binds.
                caps[hour] = min(caps.get(hour, float(cap)), float(cap))
    return caps


def grid_cap_for(caps: Dict[int, float], hour: int) -> Optional[float]:
    return caps.get(hour)
