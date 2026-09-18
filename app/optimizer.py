
"""
24-hour schedule optimizer & test runner.

This module combines the PuLP-based LP schedule optimizer with an inline
test harness to evaluate edge cases, infeasibility scenarios, and validator replays.
"""

from __future__ import annotations

import logging
import sys
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
from .replay import verify_schedule
from .schemas import Battery, BatteryAction, Directive, HourEntry, HourPlanEntry

logger = logging.getLogger(__name__)

EPSILON = 1e-9

# =====================================================================
# OPTIMIZER CORE LOGIC
# =====================================================================

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
    problem += (
        energy_after[horizon - 1] == float(battery.initial_energy_kwh),
        "neutrality",
    )

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


# Alias for backwards compatibility with legacy call sites
solve_energy_schedule = solve


# =====================================================================
# TEST HARNESS & EDGE CASE SUITE
# =====================================================================

def parse_scenario(scenario_dict: dict) -> tuple[List[HourEntry], Battery]:
    """Helper to convert raw dictionary fixtures into Pydantic schema instances."""
    hours = [HourEntry(**h) for h in scenario_dict["hours"]]
    battery = Battery(**scenario_dict["battery"])
    return hours, battery


def parse_directives(directives_list: list) -> List[Directive]:
    """Helper to convert raw dictionary directives into Pydantic schema instances."""
    return [Directive(**d) for d in directives_list]


def get_base_scenario() -> dict:
    """Base scenario fixture for testing."""
    return {
        "scenario_id": "EDGE-TEST",
        "hours": [
            {"hour": h, "demand_kwh": 100.0, "solar_kwh": 50.0, "tariff_bdt_per_kwh": 10.0}
            for h in range(24)
        ],
        "battery": {
            "capacity_kwh": 200.0,
            "initial_energy_kwh": 100.0,
            "minimum_energy_kwh": 20.0,
            "max_charge_kwh_per_hour": 50.0,
            "max_discharge_kwh_per_hour": 50.0,
        },
    }


def run_test(name: str, scenario_raw: dict, directives_raw: list) -> None:
    """Executes a scenario through the solver and runs replay validation."""
    print(f"\n--- Testing: {name} ---")
    try:
        hours, battery = parse_scenario(scenario_raw)
        directives = parse_directives(directives_raw)

        # 1. Run Solver
        result = solve(hours, battery, directives)
        print("Solver Status: EXECUTED (No 500 Raised)")

        # 2. Replay Verification
        is_valid, logs = verify_schedule(scenario_raw, directives_raw, result)
        print(f"Replay Valid: {is_valid}")
        if not is_valid:
            print(f"Replay Violation Logs: {logs}")
    except Exception as e:
        print(f"FAILED (Raised 500 / Exception): {e}")


if __name__ == "__main__":
    # --- Edge Case Test Scenarios ---

    # 1. Tighter max_grid_window causing LP Infeasibility
    scenario1 = get_base_scenario()
    directives1 = [
        {
            "applies": True,
            "directive_type": "max_grid_window",
            "structured_adjustment": {"hours": [12], "max_grid_kwh": 0.0},
        }
    ]

    # 2. minimum_battery_reserve at capacity or hour 23 fighting end-of-day neutrality
    scenario2 = get_base_scenario()
    directives2 = [
        {
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [23], "minimum_energy_kwh": 200.0},
        }
    ]

    # 3. Solar Reduction with factor 0 (Complete Solar Outage)
    scenario3 = get_base_scenario()
    directives3 = [
        {
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {"hours": list(range(24)), "factor": 0.0},
        }
    ]

    # 4. All-zero solar, flat tariff, demand exceeding grid cap
    scenario4 = get_base_scenario()
    for h_entry in scenario4["hours"]:
        h_entry["solar_kwh"] = 0.0
    directives4 = [
        {
            "applies": True,
            "directive_type": "max_grid_window",
            "structured_adjustment": {"hours": [10], "max_grid_kwh": 10.0},
        }
    ]

    # 5. Overlapping / Duplicate Directives
    scenario5 = get_base_scenario()
    directives5 = [
        {
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [10, 11, 12], "minimum_energy_kwh": 80.0},
        },
        {
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [12, 13], "minimum_energy_kwh": 120.0},
        },
    ]

    # Run Test Suite
    run_test("1. Infeasible Grid Cap", scenario1, directives1)
    run_test("2. Hour 23 Neutrality vs Reserve Conflict", scenario2, directives2)
    run_test("3. Total Solar Outage (Factor 0)", scenario3, directives3)
    run_test("4. Demand > Grid Cap (Unresolvable)", scenario4, directives4)
    run_test("5. Overlapping Directives", scenario5, directives5)
