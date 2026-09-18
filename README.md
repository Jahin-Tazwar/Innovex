# Innovex
import pulp

from typing import List, Dict, Any, Tuple

from pydantic import BaseModel



class HourlyPlanEntry(BaseModel):

    hour: int

    grid_kwh: float

    solar_used_kwh: float

    battery_action: str  # "charge", "discharge", "idle"

    battery_kwh: float

    battery_energy_after_kwh: float



class OptimizationResult(BaseModel):

    hourly_plan: List[HourlyPlanEntry]

    total_grid_kwh: float

    total_cost_bdt: float

    peak_grid_kwh: float



def solve_energy_schedule(

    request_data: dict, 

    directives: List[dict]

) -> OptimizationResult:

    hours_data = request_data["hours"]

    battery = request_data["battery"]

    

    # Base battery constraints

    capacity = float(battery["capacity_kwh"])

    initial_energy = float(battery["initial_energy_kwh"])

    base_min_energy = float(battery["minimum_energy_kwh"])

    max_charge = float(battery["max_charge_kwh_per_hour"])

    max_discharge = float(battery["max_discharge_kwh_per_hour"])



    # Prepare hourly parameter vectors

    solar_effective = [float(h["solar_kwh"]) for h in hours_data]

    demand = [float(h["demand_kwh"]) for h in hours_data]

    tariff = [float(h["tariff_bdt_per_kwh"]) for h in hours_data]

    

    min_reserve = [base_min_energy] * 24

    no_charge_hours = set()

    no_discharge_hours = set()

    max_grid_cap = [float('inf')] * 24



    # Apply directives to optimization parameters

    for d in directives:

        if not d.get("applies", False):

            continue

        

        dtype = d.get("directive_type")

        adj = d.get("structured_adjustment") or {}

        target_hours = adj.get("hours", [])



        if dtype == "solar_reduction":

            factor = float(adj.get("factor", 1.0))

            for h in target_hours:

                if 0 <= h < 24:

                    solar_effective[h] *= factor



        elif dtype == "minimum_battery_reserve":

            req_min = float(adj.get("minimum_energy_kwh", base_min_energy))

            for h in target_hours:

                if 0 <= h < 24:

                    min_reserve[h] = max(min_reserve[h], req_min)



        elif dtype == "no_charge_window":

            for h in target_hours:

                if 0 <= h < 24:

                    no_charge_hours.add(h)



        elif dtype == "no_discharge_window":

            for h in target_hours:

                if 0 <= h < 24:

                    no_discharge_hours.add(h)



        elif dtype == "max_grid_window":

            grid_cap = float(adj.get("max_grid_kwh", float('inf')))

            for h in target_hours:

                if 0 <= h < 24:

                    max_grid_cap[h] = min(max_grid_cap[h], grid_cap)



    # Instantiate LP Problem

    prob = pulp.LpProblem("GridWise_Optimization", pulp.LpMinimize)



    # Decision Variables

    grid_kwh = [pulp.LpVariable(f"grid_{h}", lowBound=0, upBound=max_grid_cap[h]) for h in range(24)]

    solar_used = [pulp.LpVariable(f"solar_used_{h}", lowBound=0, upBound=solar_effective[h]) for h in range(24)]

    charge = [pulp.LpVariable(f"charge_{h}", lowBound=0, upBound=0 if h in no_charge_hours else max_charge) for h in range(24)]

    discharge = [pulp.LpVariable(f"discharge_{h}", lowBound=0, upBound=0 if h in no_discharge_hours else max_discharge) for h in range(24)]

    e_after = [pulp.LpVariable(f"e_after_{h}", lowBound=min_reserve[h], upBound=capacity) for h in range(24)]



    # Objective Function: Minimize Total BDT Cost

    prob += pulp.lpSum([grid_kwh[h] * tariff[h] for h in range(24)])



    # Constraints

    for h in range(24):

        # 1. Energy balance equation

        prob += (grid_kwh[h] + solar_used[h] + discharge[h] == demand[h] + charge[h], f"EnergyBalance_{h}")

        

        # 2. Battery state transitions

        prev_e = initial_energy if h == 0 else e_after[h - 1]

        prob += (e_after[h] == prev_e + charge[h] - discharge[h], f"BatteryState_{h}")



    # 3. End-of-day battery neutrality

    prob += (e_after[23] == initial_energy, "Neutrality")



    # Solve model silently

    solver = pulp.PULP_CBC_CMD(msg=False)

    status = prob.solve(solver)



    if status != pulp.LpStatusOptimal:

        raise ValueError("Optimization model failed to find an optimal solution.")



    # Process results into output schema

    hourly_plan: List[HourlyPlanEntry] = []

    total_grid = 0.0

    total_cost = 0.0

    peak_grid = 0.0



    for h in range(24):

        g = round(pulp.value(grid_kwh[h]), 4)

        s = round(pulp.value(solar_used[h]), 4)

        c = round(pulp.value(charge[h]), 4)

        d = round(pulp.value(discharge[h]), 4)

        e = round(pulp.value(e_after[h]), 4)



        if c > 0.001:

            action = "charge"

            b_kwh = c

        elif d > 0.001:

            action = "discharge"

            b_kwh = d

        else:

            action = "idle"

            b_kwh = 0.0



        hourly_plan.append(HourlyPlanEntry(

            hour=h,

            grid_kwh=g,

            solar_used_kwh=s,

            battery_action=action,

            battery_kwh=b_kwh,

            battery_energy_after_kwh=e

        ))



        total_grid += g

        total_cost += g * tariff[h]

        if g > peak_grid:

            peak_grid = g



    return OptimizationResult(

        hourly_plan=hourly_plan,

        total_grid_kwh=round(total_grid, 2),

        total_cost_bdt=round(total_cost, 2),

        peak_grid_kwh=round(peak_grid, 2)

    ) 


