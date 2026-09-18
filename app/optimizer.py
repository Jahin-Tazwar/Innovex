"""
24-hour schedule optimizer -- PERSON C OWNS THIS FILE.

The LP below is C's draft (it was sitting in README.md), ported onto the frozen
contract by A. Changes made during the port:

  * directives now arrive pre-validated as `Directive` objects, so the big
    if/elif block that parsed raw dicts is gone -- app/energy.py does it, and
    app/replay.py reads the same helpers so the checker cannot drift from the
    model.
  * hours are indexed by position in a 0..23-sorted list rather than assuming
    the request arrived in order.
  * an uncapped grid variable gets upBound=None instead of float("inf") (CBC
    chokes on an infinite bound written into the LP file).
  * simultaneous charge+discharge in the same hour is netted into one action by
    build_plan(); picking `charge` while discharge was also non-zero would have
    made battery_kwh disagree with battery_energy_after_kwh.
  * infeasible/failed solve returns the baseline instead of raising, so the case
    is still valid rather than a 500.
  * totals are no longer computed here -- main.py derives them from the rounded
    plan so reported numbers always reproduce from hourly_plan.

Interface:
    solve(hours, battery, directives) -> list[HourPlanEntry]
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import pulp

from .config import settings
from .energy import (
    effective_solar,
    grid_caps,
    no_charge_hours,
    no_discharge_hours,
    reserve_floors,
)
from .schemas import Battery, BatteryAction, Directive, HourEntry, HourPlanEntry

logger = logging.getLogger(__name__)

EPSILON = 1e-9


def build_plan(
    hours: Sequence[HourEntry],
    battery: Battery,
    grid: Sequence[float],
    solar_used: Sequence[float],
    charge: Sequence[float],
    discharge: Sequence[float],
) -> List[HourPlanEntry]:
    """
    Turn solver arrays into response entries.

    Nets simultaneous charge/discharge into a single action and rebuilds the
    battery trajectory from the netted values, so battery_energy_after_kwh
    always agrees with the action and magnitude we report.
    """
    plan: List[HourPlanEntry] = []
    energy = float(battery.initial_energy_kwh)

    for index, hour in enumerate(hours):
        net = float(charge[index]) - float(discharge[index])
        if net > EPSILON:
            action, magnitude = BatteryAction.CHARGE, net
        elif net < -EPSILON:
            action, magnitude = BatteryAction.DISCHARGE, -net
        else:
            action, magnitude = BatteryAction.IDLE, 0.0

        energy += net
        plan.append(
            HourPlanEntry(
                hour=hour.hour,
                grid_kwh=max(0.0, float(grid[index])),
                solar_used_kwh=max(0.0, float(solar_used[index])),
                battery_action=action,
                battery_kwh=max(0.0, magnitude),
                battery_energy_after_kwh=max(0.0, energy),
            )
        )
    return plan


def baseline_plan(
    hours: Sequence[HourEntry],
    battery: Battery,
    directives: Sequence[Directive],
) -> List[HourPlanEntry]:
    """
    Safe fallback: use whatever solar is available, buy the rest, battery idle.

    Always satisfies energy balance, battery bounds, rate limits, end-of-day
    neutrality, no_charge_window and no_discharge_window. It does not satisfy
    minimum_battery_reserve or max_grid_window and it is not cheap -- it exists
    so a solver failure still returns a schedule instead of a 500.
    """
    solar = effective_solar(hours, directives)
    grid: List[float] = []
    used: List[float] = []
    for index, hour in enumerate(hours):
        take = min(solar[index], float(hour.demand_kwh))
        used.append(take)
        grid.append(float(hour.demand_kwh) - take)
    zeros = [0.0] * len(hours)
    return build_plan(hours, battery, grid, used, zeros, zeros)


def solve(
    hours: Sequence[HourEntry],
    battery: Battery,
    directives: Sequence[Directive],
) -> List[HourPlanEntry]:
    """Cheapest valid 24-hour schedule, or the baseline if the LP cannot solve."""
    horizon = len(hours)
    solar = effective_solar(hours, directives)
    floors = reserve_floors(battery, directives, horizon)
    caps = grid_caps(directives, horizon)
    blocked_charge = no_charge_hours(directives)
    blocked_discharge = no_discharge_hours(directives)

    demand = [float(h.demand_kwh) for h in hours]
    tariff = [float(h.tariff_bdt_per_kwh) for h in hours]

    problem = pulp.LpProblem("GridWise", pulp.LpMinimize)

    grid = [
        pulp.LpVariable(
            f"grid_{h}", lowBound=0, upBound=caps.get(hours[h].hour)  # None = uncapped
        )
        for h in range(horizon)
    ]
    solar_used = [
        pulp.LpVariable(f"solar_{h}", lowBound=0, upBound=solar[h])
        for h in range(horizon)
    ]
    charge = [
        pulp.LpVariable(
            f"charge_{h}",
            lowBound=0,
            upBound=0 if hours[h].hour in blocked_charge
            else battery.max_charge_kwh_per_hour,
        )
        for h in range(horizon)
    ]
    discharge = [
        pulp.LpVariable(
            f"discharge_{h}",
            lowBound=0,
            upBound=0 if hours[h].hour in blocked_discharge
            else battery.max_discharge_kwh_per_hour,
        )
        for h in range(horizon)
    ]
    energy_after = [
        pulp.LpVariable(
            f"energy_{h}",
            lowBound=floors[hours[h].hour],
            upBound=battery.capacity_kwh,
        )
        for h in range(horizon)
    ]

    problem += pulp.lpSum(grid[h] * tariff[h] for h in range(horizon))

    for h in range(horizon):
        problem += (
            grid[h] + solar_used[h] + discharge[h] == demand[h] + charge[h],
            f"balance_{h}",
        )
        previous = (
            float(battery.initial_energy_kwh) if h == 0 else energy_after[h - 1]
        )
        problem += (
            energy_after[h] == previous + charge[h] - discharge[h],
            f"state_{h}",
        )

    # End-of-day neutrality: the starting charge is a buffer, not free energy.
    problem += (energy_after[horizon - 1] == float(battery.initial_energy_kwh),
                "neutrality")

    try:
        solver = pulp.PULP_CBC_CMD(
            msg=False, timeLimit=int(settings.SOLVER_TIMEOUT_SECONDS)
        )
        status = problem.solve(solver)
    except Exception as exc:  # noqa: BLE001 -- missing/failing CBC binary
        logger.error("solver crashed (%s); falling back to baseline", type(exc).__name__)
        return baseline_plan(hours, battery, directives)

    if pulp.LpStatus[status] != "Optimal":
        logger.error(
            "LP status %s; falling back to baseline", pulp.LpStatus.get(status, status)
        )
        return baseline_plan(hours, battery, directives)

    def value(variable) -> float:
        raw = pulp.value(variable)
        return 0.0 if raw is None else float(raw)

    return build_plan(
        hours,
        battery,
        [value(v) for v in grid],
        [value(v) for v in solar_used],
        [value(v) for v in charge],
        [value(v) for v in discharge],
    )
