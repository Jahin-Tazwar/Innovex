"""
GridWise LLM -- service entry point.

Pipeline, in order:

    request -> schema validation -> LLM interpretation
            -> deterministic guardrails -> optimizer
            -> replay validation -> rounded response

Every stage degrades instead of failing: if the model is unreachable we still
return a valid schedule, and if the optimizer produces an invalid plan we fall
back to a safe baseline. The service should never return 5xx for a valid
request (Participant Guide 08, "failure rate").
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import guardrails, interpreter, optimizer, replay
from .config import settings
from .schemas import (
    ROUND_DP,
    Directive,
    DirectiveType,
    HealthResponse,
    HourPlanEntry,
    OptimizeRequest,
    OptimizeResponse,
)

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("gridwise")

app = FastAPI(
    title="GridWise LLM",
    description="LLM-assisted operator directive interpretation and 24-hour "
    "campus energy optimization.",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Error handling -- controlled responses only, never a stack trace or a secret.
# --------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    """Structurally invalid request -> 400 (Problem Statement 6.1)."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "invalid_request", "detail": _safe_errors(exc)},
    )


@app.exception_handler(json.JSONDecodeError)
async def on_json_error(request: Request, exc: json.JSONDecodeError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "invalid_json", "detail": "Request body is not valid JSON."},
    )


@app.exception_handler(Exception)
async def on_unhandled(request: Request, exc: Exception):
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error"},
    )


def _safe_errors(exc: RequestValidationError) -> List[Dict[str, Any]]:
    """Field-level detail with no input echo, so nothing sensitive leaks back."""
    out: List[Dict[str, Any]] = []
    for error in exc.errors()[:20]:
        out.append(
            {
                "field": ".".join(str(part) for part in error.get("loc", ())),
                "message": str(error.get("msg", ""))[:200],
            }
        )
    return out


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def root() -> Dict[str, Any]:
    return {
        "service": "GridWise LLM",
        "health": "/health",
        "optimize": "POST /optimize-energy",
    }


@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/optimize-energy", response_model=OptimizeResponse)
async def optimize_energy(payload: OptimizeRequest) -> OptimizeResponse:
    started = time.perf_counter()

    hours = payload.hours_sorted()
    if [h.hour for h in hours] != list(range(24)):
        return JSONResponse(  # type: ignore[return-value]
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "invalid_scenario",
                "detail": "hours must contain each hour 0..23 exactly once.",
            },
        )
    if any(not note or not note.strip() for note in payload.operator_notes):
        return JSONResponse(  # type: ignore[return-value]
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "invalid_scenario",
                "detail": "operator_notes entries must be non-empty strings.",
            },
        )

    # 1. LLM interpretation. A failure here must not fail the request: we still
    #    owe the caller a valid schedule.
    raw_entries: Any = []
    try:
        raw_entries = await interpreter.interpret_notes(
            payload.operator_notes, payload.battery
        )
    except Exception as exc:  # noqa: BLE001 -- provider errors are expected
        logger.warning("interpreter failed (%s); continuing with no directives",
                       type(exc).__name__)

    # 2. Deterministic guardrails. Never raises; always len(notes) entries.
    interpretations, directives = guardrails.normalize(
        raw_entries, payload.operator_notes, payload.battery
    )

    # 3. Optimize. Sync and CPU-bound, so keep it off the event loop.
    try:
        plan = await run_in_threadpool(
            optimizer.solve, hours, payload.battery, directives
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("optimizer failed (%s); using baseline", type(exc).__name__)
        plan = optimizer.baseline_plan(hours, payload.battery, directives)

    # 4. Replay against our own checker. An invalid plan scores zero, so a valid
    #    expensive fallback is strictly better than shipping the broken one.
    errors = replay.replay(plan, hours, payload.battery, directives)
    if errors:
        logger.error("plan failed replay (%d issues, first: %s)", len(errors), errors[0])
        fallback = optimizer.baseline_plan(hours, payload.battery, directives)
        if not replay.replay(fallback, hours, payload.battery, directives):
            plan = fallback

    plan = _round_plan(plan, payload.battery)
    total_grid, total_cost, peak_grid = replay.totals_from_plan(plan, hours)

    elapsed_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "scenario=%s notes=%d applied=%d cost=%.2f in %.0fms",
        payload.scenario_id,
        len(payload.operator_notes),
        len(directives),
        total_cost,
        elapsed_ms,
    )

    return OptimizeResponse(
        scenario_id=payload.scenario_id,
        directive_interpretation=interpretations,
        hourly_plan=plan,
        total_grid_kwh=round(total_grid, ROUND_DP),
        total_cost_bdt=round(total_cost, ROUND_DP),
        peak_grid_kwh=round(peak_grid, ROUND_DP),
        plan_summary=_summarize(directives, total_cost, peak_grid),
    )


# --------------------------------------------------------------------------
# Response assembly
# --------------------------------------------------------------------------


def _round_plan(plan: List[HourPlanEntry], battery) -> List[HourPlanEntry]:
    """
    Round every reported value and rebuild the battery trajectory from the
    rounded magnitudes, so battery_energy_after_kwh always reconciles exactly
    with the action we report. Totals are then summed from these same numbers.
    """
    energy = round(float(battery.initial_energy_kwh), ROUND_DP)
    rounded: List[HourPlanEntry] = []
    for entry in plan:
        magnitude = round(max(0.0, entry.battery_kwh), ROUND_DP)
        if entry.battery_action.value == "charge":
            energy = round(energy + magnitude, ROUND_DP)
        elif entry.battery_action.value == "discharge":
            energy = round(energy - magnitude, ROUND_DP)
        else:
            magnitude = 0.0
        rounded.append(
            HourPlanEntry(
                hour=entry.hour,
                grid_kwh=round(max(0.0, entry.grid_kwh), ROUND_DP),
                solar_used_kwh=round(max(0.0, entry.solar_used_kwh), ROUND_DP),
                battery_action=entry.battery_action,
                battery_kwh=magnitude,
                battery_energy_after_kwh=max(0.0, energy),
            )
        )
    return rounded


_SUMMARY_LABELS = {
    DirectiveType.SOLAR_REDUCTION: "reduced solar availability",
    DirectiveType.MINIMUM_BATTERY_RESERVE: "a raised battery reserve",
    DirectiveType.NO_CHARGE_WINDOW: "a no-charge window",
    DirectiveType.NO_DISCHARGE_WINDOW: "a no-discharge window",
    DirectiveType.MAX_GRID_WINDOW: "a grid import cap",
}


def _summarize(
    directives: List[Directive], total_cost: float, peak_grid: float
) -> str:
    """
    Human-readable summary, built deterministically.

    Deliberately not model-generated: the LLM requirement is satisfied by the
    interpretation path, and a model call here would only add latency.
    """
    if directives:
        applied = ", ".join(
            sorted({_SUMMARY_LABELS.get(d.directive_type, "") for d in directives} - {""})
        )
        head = f"Applied {len(directives)} operator directive(s): {applied}. "
    else:
        head = "No operator note changed today's schedule. "
    return (
        head
        + "Solar is used first, the battery shifts energy from cheap hours into "
        + "expensive ones and returns to its starting level by hour 23, giving a "
        + f"total grid cost of {total_cost:.2f} BDT with a peak draw of "
        + f"{peak_grid:.2f} kWh."
    )
