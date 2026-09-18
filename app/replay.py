"""
Replay validator -- our local stand-in for the judge (Problem Statement 11.2,
11.3; Participant Guide 09).

The judge re-simulates the returned hourly_plan hour by hour against the
GROUND-TRUTH directives rather than our reported interpretation. This module
does the same thing, so anything it flags is a case we would have lost.

Person A wrote this so the smoke-test harness has something to check against.
Person C: run it on your solver output while you work.

    errors = replay(plan, request.hours_sorted(), request.battery, directives)
    errors == [] means the plan is valid.
"""

from __future__ import annotations

from typing import List, Sequence

from .energy import (
    effective_solar,
    grid_caps,
    no_charge_hours,
    no_discharge_hours,
    reserve_floors,
)
from .schemas import (
    TOLERANCE,
    Battery,
    BatteryAction,
    Directive,
    HourEntry,
    HourPlanEntry,
)


def replay(
    plan: Sequence[HourPlanEntry],
    hours: Sequence[HourEntry],
    battery: Battery,
    directives: Sequence[Directive],
    tolerance: float = TOLERANCE,
) -> List[str]:
    """Return a list of human-readable rule violations. Empty means valid."""
    errors: List[str] = []

    if len(plan) != 24:
        errors.append(f"hourly_plan has {len(plan)} entries, expected 24")
        return errors

    by_hour = {entry.hour: entry for entry in plan}
    if sorted(by_hour) != list(range(24)):
        errors.append("hourly_plan hours are not exactly 0..23 with no duplicates")
        return errors

    solar = effective_solar(hours, directives)
    floors = reserve_floors(battery, directives)
    caps = grid_caps(directives)
    no_charge = no_charge_hours(directives)
    no_discharge = no_discharge_hours(directives)

    energy = float(battery.initial_energy_kwh)

    for index, hour_entry in enumerate(hours):
        hour = hour_entry.hour
        entry = by_hour[hour]

        for label, value in (
            ("grid_kwh", entry.grid_kwh),
            ("solar_used_kwh", entry.solar_used_kwh),
            ("battery_kwh", entry.battery_kwh),
            ("battery_energy_after_kwh", entry.battery_energy_after_kwh),
        ):
            if value != value or value in (float("inf"), float("-inf")):
                errors.append(f"h{hour}: {label} is not finite")
            elif value < -tolerance:
                errors.append(f"h{hour}: {label} is negative ({value})")

        charge = discharge = 0.0
        if entry.battery_action is BatteryAction.CHARGE:
            charge = entry.battery_kwh
            if charge > battery.max_charge_kwh_per_hour + tolerance:
                errors.append(
                    f"h{hour}: charge {charge} exceeds max_charge_kwh_per_hour "
                    f"{battery.max_charge_kwh_per_hour}"
                )
            if hour in no_charge:
                errors.append(f"h{hour}: charging inside a no_charge_window")
        elif entry.battery_action is BatteryAction.DISCHARGE:
            discharge = entry.battery_kwh
            if discharge > battery.max_discharge_kwh_per_hour + tolerance:
                errors.append(
                    f"h{hour}: discharge {discharge} exceeds "
                    f"max_discharge_kwh_per_hour {battery.max_discharge_kwh_per_hour}"
                )
            if hour in no_discharge:
                errors.append(f"h{hour}: discharging inside a no_discharge_window")
        elif entry.battery_kwh > tolerance:
            errors.append(f"h{hour}: battery_kwh must be 0 when idle")

        if entry.solar_used_kwh > solar[index] + tolerance:
            errors.append(
                f"h{hour}: solar_used_kwh {entry.solar_used_kwh} exceeds effective "
                f"solar {solar[index]:.4f}"
            )

        supply = entry.grid_kwh + entry.solar_used_kwh + discharge
        demand = float(hour_entry.demand_kwh) + charge
        if abs(supply - demand) > tolerance:
            errors.append(
                f"h{hour}: energy balance off by {supply - demand:.4f} "
                f"(supply {supply:.4f} vs demand {demand:.4f})"
            )

        cap = caps.get(hour)
        if cap is not None and entry.grid_kwh > cap + tolerance:
            errors.append(
                f"h{hour}: grid_kwh {entry.grid_kwh} exceeds max_grid_window cap {cap}"
            )

        energy += charge - discharge
        if abs(entry.battery_energy_after_kwh - energy) > tolerance:
            errors.append(
                f"h{hour}: battery_energy_after_kwh {entry.battery_energy_after_kwh} "
                f"does not follow the stated action (expected {energy:.4f})"
            )
        energy = entry.battery_energy_after_kwh

        if energy > battery.capacity_kwh + tolerance:
            errors.append(f"h{hour}: battery energy {energy} exceeds capacity")
        if energy < floors[hour] - tolerance:
            errors.append(
                f"h{hour}: battery energy {energy} below required reserve "
                f"{floors[hour]}"
            )

    if abs(energy - battery.initial_energy_kwh) > tolerance:
        errors.append(
            f"end-of-day battery energy {energy} != initial "
            f"{battery.initial_energy_kwh}"
        )

    return errors


def verify_schedule(
    scenario: dict,
    directives_raw: Sequence,
    plan: Sequence[HourPlanEntry],
) -> tuple[bool, List[str]]:
    """
    Dict-in convenience wrapper around replay(), for test fixtures.

    Takes a raw scenario dict and raw directive dicts (the shape the LLM/judge
    speaks) instead of parsed models, so edge-case suites can be written as
    plain literals.

        is_valid, problems = verify_schedule(scenario, directives, plan)
    """
    from .guardrails import normalize  # local import keeps the module graph flat

    battery = Battery.model_validate(scenario["battery"])
    hours = sorted(
        (HourEntry.model_validate(h) for h in scenario["hours"]),
        key=lambda h: h.hour,
    )

    if directives_raw and isinstance(directives_raw[0], Directive):
        directives = list(directives_raw)
    else:
        _, directives = normalize(
            list(directives_raw), [""] * len(directives_raw), battery
        )

    errors = replay(plan, hours, battery, directives)
    return not errors, errors


def totals_from_plan(
    plan: Sequence[HourPlanEntry], hours: Sequence[HourEntry]
) -> tuple[float, float, float]:
    """(total_grid_kwh, total_cost_bdt, peak_grid_kwh) recomputed from the plan."""
    tariff = {h.hour: float(h.tariff_bdt_per_kwh) for h in hours}
    total_grid = sum(entry.grid_kwh for entry in plan)
    total_cost = sum(entry.grid_kwh * tariff.get(entry.hour, 0.0) for entry in plan)
    peak = max((entry.grid_kwh for entry in plan), default=0.0)
    return total_grid, total_cost, peak
