"""
Request/response contract, transcribed from the Problem Statement (07, 10).

Field names and types are fixed by the spec and are relied on across the whole
service, so they should not be changed casually.

Directive shapes (Problem Statement 4.1):
    solar_reduction          {"hours": [...], "factor": float}
    minimum_battery_reserve  {"hours": [...], "minimum_energy_kwh": float}
    no_charge_window         {"hours": [...]}
    no_discharge_window      {"hours": [...]}
    max_grid_window          {"hours": [...], "max_grid_kwh": float}
    no_op                    null
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Judge tolerance is 0.01 kWh / 0.01 BDT. We round plan values well inside it so
# the numbers we report always reproduce exactly from the plan we return.
ROUND_DP = 6
TOLERANCE = 0.01


class DirectiveType(str, Enum):
    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


class BatteryAction(str, Enum):
    CHARGE = "charge"
    DISCHARGE = "discharge"
    IDLE = "idle"


# --------------------------------------------------------------------------
# Request  (Problem Statement 07)
# --------------------------------------------------------------------------


class HourEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float


class Battery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capacity_kwh: float = Field(ge=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenario_id: str = Field(min_length=1)
    operator_notes: List[str] = Field(min_length=1, max_length=3)
    hours: List[HourEntry] = Field(min_length=24, max_length=24)
    battery: Battery

    def hours_sorted(self) -> List[HourEntry]:
        """Hours in 0..23 order regardless of the order they arrived in."""
        return sorted(self.hours, key=lambda h: h.hour)


# --------------------------------------------------------------------------
# Response  (Problem Statement 10)
# --------------------------------------------------------------------------


class DirectiveInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    note_index: int = Field(ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[Dict[str, Any]] = None
    explanation: str = ""


class HourPlanEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hour: int = Field(ge=0, le=23)
    grid_kwh: float = Field(ge=0)
    solar_used_kwh: float = Field(ge=0)
    battery_action: BatteryAction
    battery_kwh: float = Field(ge=0)
    battery_energy_after_kwh: float = Field(ge=0)


class OptimizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


# --------------------------------------------------------------------------
# Internal type the optimizer consumes: validated directives only, never raw
# model output or note text. `hours` is always sorted unique ints 0..23.
# --------------------------------------------------------------------------


class Directive(BaseModel):
    """A single validated, applicable directive. no_op never reaches here."""

    model_config = ConfigDict(extra="forbid")

    directive_type: DirectiveType
    hours: List[int] = Field(default_factory=list)
    factor: Optional[float] = None  # solar_reduction: fraction REMAINING
    minimum_energy_kwh: Optional[float] = None  # minimum_battery_reserve
    max_grid_kwh: Optional[float] = None  # max_grid_window


class HealthResponse(BaseModel):
    status: str = "ok"
