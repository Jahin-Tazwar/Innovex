#!/usr/bin/env python3
"""
Run every public sample case against a running service and score it the way
the judge would.

    python scripts/run_public_cases.py                       # localhost:8000
    python scripts/run_public_cases.py --base-url https://... # deployed
    python scripts/run_public_cases.py --case SAMPLE-03 -v    # one case, verbose

For each case it checks three things independently:

  INTERP  directive_type + hours + numeric values vs the published expectation
  VALID   the returned hourly_plan replayed against the GROUND-TRUTH directives
          (what the judge does -- not our own interpretation)
  COST    min(1, reference_cost / our_cost)

Exit code is 0 only when every case is valid and every interpretation matches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import guardrails, replay  # noqa: E402
from app.schemas import TOLERANCE, Battery, HourEntry, OptimizeResponse  # noqa: E402

SAMPLES = Path(__file__).resolve().parents[1] / "samples" / "public_cases.json"


def close(a: float, b: float, tol: float = TOLERANCE) -> bool:
    return abs(float(a) - float(b)) <= tol


def compare_interpretation(
    got: List[Dict[str, Any]], expected: List[Dict[str, Any]]
) -> Tuple[int, int, List[str]]:
    """Returns (matched_notes, total_notes, problems)."""
    problems: List[str] = []
    total = len(expected)
    matched = 0

    by_index = {entry.get("note_index"): entry for entry in got}
    if [entry.get("note_index") for entry in got] != list(range(len(got))):
        problems.append("note_index values are not 0..N-1 in order")
    if len(got) != total:
        problems.append(f"returned {len(got)} entries for {total} notes")

    for want in expected:
        index = want.get("note_index")
        mine = by_index.get(index)
        if mine is None:
            problems.append(f"note {index}: missing")
            continue

        want_type = want.get("directive_type")
        got_type = mine.get("directive_type")
        if want_type != got_type:
            problems.append(f"note {index}: type {got_type!r}, expected {want_type!r}")
            continue

        if bool(mine.get("applies")) != bool(want.get("applies")):
            problems.append(f"note {index}: applies={mine.get('applies')}")
            continue

        want_adj = want.get("structured_adjustment")
        got_adj = mine.get("structured_adjustment")
        if want_adj is None:
            if got_adj is not None:
                problems.append(f"note {index}: expected null structured_adjustment")
                continue
            matched += 1
            continue

        if not isinstance(got_adj, dict):
            problems.append(f"note {index}: structured_adjustment is not an object")
            continue

        ok = True
        if list(want_adj.get("hours", [])) != list(got_adj.get("hours", [])):
            problems.append(
                f"note {index}: hours {got_adj.get('hours')}, "
                f"expected {want_adj.get('hours')}"
            )
            ok = False
        for key in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
            if key in want_adj:
                if key not in got_adj or not close(got_adj[key], want_adj[key]):
                    problems.append(
                        f"note {index}: {key}={got_adj.get(key)}, "
                        f"expected {want_adj[key]}"
                    )
                    ok = False
        if ok:
            matched += 1

    return matched, total, problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--case", help="run a single case id, e.g. SAMPLE-03")
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    pack = json.loads(SAMPLES.read_text(encoding="utf-8"))
    cases = pack["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"no case with id {args.case}")
            return 2

    base = args.base_url.rstrip("/")
    print(f"target {base}   cases {len(cases)}\n")
    print(f"{'case':<12} {'interp':>8} {'valid':>7} {'cost':>7}  {'ms':>6}")
    print("-" * 48)

    interp_matched = interp_total = 0
    valid_cases = 0
    cost_ratios: List[float] = []
    failures: List[str] = []

    with httpx.Client(timeout=args.timeout) as client:
        for case in cases:
            payload = case["input"]
            expected = case["expected_output"]

            try:
                response = client.post(f"{base}/optimize-energy", json=payload)
                elapsed = response.elapsed.total_seconds() * 1000
            except Exception as exc:  # noqa: BLE001
                print(f"{case['id']:<12} {'-':>8} {'ERROR':>7} {'-':>7}  {'-':>6}")
                failures.append(f"{case['id']}: request failed -- {exc}")
                continue

            if response.status_code != 200:
                print(f"{case['id']:<12} {'-':>8} {'HTTP':>7} {'-':>7}  {elapsed:6.0f}")
                failures.append(f"{case['id']}: HTTP {response.status_code}")
                continue

            body = response.json()
            try:
                parsed = OptimizeResponse.model_validate(body)
            except Exception as exc:  # noqa: BLE001
                print(f"{case['id']:<12} {'-':>8} {'SCHEMA':>7} {'-':>7}  {elapsed:6.0f}")
                failures.append(f"{case['id']}: response schema -- {exc}")
                continue

            if parsed.scenario_id != payload["scenario_id"]:
                failures.append(f"{case['id']}: scenario_id not echoed")

            matched, total, problems = compare_interpretation(
                body.get("directive_interpretation", []),
                expected["directive_interpretation"],
            )
            interp_matched += matched
            interp_total += total
            failures.extend(f"{case['id']}: {p}" for p in problems)

            # Replay against ground truth, exactly like the judge.
            battery = Battery.model_validate(payload["battery"])
            hours = sorted(
                (HourEntry.model_validate(h) for h in payload["hours"]),
                key=lambda h: h.hour,
            )
            _, truth = guardrails.normalize(
                expected["directive_interpretation"],
                payload["operator_notes"],
                battery,
            )
            errors = replay.replay(parsed.hourly_plan, hours, battery, truth)

            # Reported totals must reproduce from hourly_plan.
            total_grid, total_cost, peak = replay.totals_from_plan(
                parsed.hourly_plan, hours
            )
            for label, mine, recomputed in (
                ("total_grid_kwh", parsed.total_grid_kwh, total_grid),
                ("total_cost_bdt", parsed.total_cost_bdt, total_cost),
                ("peak_grid_kwh", parsed.peak_grid_kwh, peak),
            ):
                if not close(mine, recomputed):
                    errors.append(
                        f"{label} {mine} does not match plan ({recomputed:.4f})"
                    )

            valid = not errors
            valid_cases += int(valid)
            failures.extend(f"{case['id']}: {e}" for e in errors[:6])

            reference = float(expected["total_cost_bdt"])
            if valid:
                ratio = 1.0 if total_cost <= TOLERANCE and reference <= TOLERANCE else (
                    min(1.0, reference / total_cost) if total_cost > TOLERANCE else 0.0
                )
            else:
                ratio = 0.0
            cost_ratios.append(ratio)

            print(
                f"{case['id']:<12} {matched}/{total:<6} "
                f"{'ok' if valid else 'FAIL':>7} {ratio:7.3f}  {elapsed:6.0f}"
            )
            if args.verbose:
                print(f"    reference {reference:.2f} BDT / ours {total_cost:.2f} BDT")
                for problem in problems + errors[:6]:
                    print(f"    - {problem}")

    print("-" * 48)
    count = len(cases)
    interp_pct = 100 * interp_matched / interp_total if interp_total else 0.0
    mean_cost = sum(cost_ratios) / len(cost_ratios) if cost_ratios else 0.0
    print(
        f"interpretation {interp_matched}/{interp_total} ({interp_pct:.0f}%)   "
        f"valid {valid_cases}/{count}   mean cost ratio {mean_cost:.3f}"
    )

    if failures:
        print(f"\n{len(failures)} problem(s):")
        for line in failures[:40]:
            print(f"  - {line}")

    return 0 if (valid_cases == count and interp_matched == interp_total) else 1


if __name__ == "__main__":
    raise SystemExit(main())
