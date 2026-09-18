# GridWise LLM — Smart Campus Energy Optimization

**BUP CSE Fest 2026 · Online Preliminary · Team Innovex**

An HTTP service that reads plain-English campus operator notes with a language
model, converts them into machine-checkable directives, validates those
directives deterministically, and returns a cost-minimal 24-hour energy
schedule that obeys every one of them.

| | |
|---|---|
| Health endpoint | `GET /health` → `{"status":"ok"}` |
| Main endpoint | `POST /optimize-energy` |
| Live base URL | <!-- TODO: paste the deployed Render URL here before submitting --> |
| Docker image | `ghcr.io/jahin-tazwar/innovex:latest` |
| Interactive schema | `GET /docs` |

---

## 1. Architecture

The model understands language. It is never trusted with arithmetic. Every
interpretation is rebuilt by deterministic code before it can reach the
optimizer.

```
POST /optimize-energy
      │
      ▼
┌─────────────────┐   FastAPI + Pydantic. Structurally invalid → 400,
│ schema layer    │   semantically invalid → 422. Never 5xx on valid input.
└────────┬────────┘
         ▼
┌─────────────────┐   app/interpreter.py
│ LLM             │   One call, all 1–3 notes at once. Emits a raw directive
│ interpretation  │   per note. Output is treated as UNTRUSTED from here on.
└────────┬────────┘
         ▼
┌─────────────────┐   app/guardrails.py — the trust boundary.
│ guardrails      │   Rebuilds every entry from scratch: allowed types only,
│ (deterministic) │   one entry per note in note_index order, hours forced to
│                 │   unique ints 0–23 ascending, factor coerced into [0,1]
│                 │   ("20" and "20%" → 0.2), reserve capped at capacity,
│                 │   applies set from the type. Anything unusable degrades to
│                 │   no_op. This function never raises.
└────────┬────────┘
         ▼
┌─────────────────┐   app/optimizer.py — linear program solved with PuLP/CBC.
│ optimizer       │   Directives enter as hard constraints via app/energy.py.
│                 │   Infeasible or failed solve → safe baseline, never a 500.
└────────┬────────┘
         ▼
┌─────────────────┐   app/replay.py — re-simulates the finished plan hour by
│ replay check    │   hour against every rule, the same way the judge does.
│                 │   A plan that fails is swapped for the valid baseline.
└────────┬────────┘
         ▼
     response      Totals are summed from the ROUNDED hourly_plan, so
                   total_grid_kwh / total_cost_bdt / peak_grid_kwh always
                   reproduce exactly from the plan we return.
```

**Why the LLM cannot be bypassed:** `interpret_notes()` is the only thing that
turns note text into a directive. There is no keyword matcher, no phrase table
and no fallback interpreter — if the model returns nothing, notes become
`no_op` and the schedule is built without them. The model's structured output
is what produces the optimizer's constraints.

**What the model is deliberately *not* used for:** `plan_summary` is generated
deterministically from the applied directives. Spending a round-trip on prose
would only add latency.

### Files

| Path | Role |
|---|---|
| `app/schemas.py` | Frozen request/response contract |
| `app/interpreter.py` | LLM call — notes → raw directives |
| `app/guardrails.py` | Deterministic validation and repair |
| `app/energy.py` | Shared directive semantics (optimizer and replay read the same helpers, so they cannot drift) |
| `app/optimizer.py` | The LP, plus an edge-case suite (`python -m app.optimizer`) |
| `app/replay.py` | Local re-implementation of the judge's validator |
| `app/main.py` | Endpoints, error handling, response assembly |
| `scripts/run_public_cases.py` | Scores all 10 public cases like the judge |

---

## 2. Quickstart (clean machine)

Requires **Python 3.11+**. No other system dependencies — the CBC solver ships
inside the `pulp` wheel.

```bash
git clone https://github.com/Jahin-Tazwar/Innovex.git
cd Innovex

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# edit .env — set LLM_PROVIDER and the matching API key (see §3)

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service is ready in under a second. Verify:

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok"}
```

Run a real scenario — `samples/example_request.json` is public sample case 1,
complete with 24 hours and 2 operator notes:

```bash
curl -s -X POST http://127.0.0.1:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d @samples/example_request.json
```

Expected shape (abridged):

```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {"note_index": 0, "applies": true, "directive_type": "solar_reduction",
     "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
     "explanation": "Panel washing reduces usable solar."},
    {"note_index": 1, "applies": false, "directive_type": "no_op",
     "structured_adjustment": null,
     "explanation": "This note does not affect today's energy schedule."}
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 90.0, "solar_used_kwh": 0.0,
     "battery_action": "idle", "battery_kwh": 0.0,
     "battery_energy_after_kwh": 120.0}
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 175.0,
  "plan_summary": "Applied 1 operator directive(s)..."
}
```

---

## 3. Configuration

Every value is read from the environment. **No secrets are committed** — see
`.env.example` for the full list of names with empty values.

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | `groq` \| `openai` \| `stub` | `stub` |
| `LLM_MODEL` | Model id; blank uses the provider default | `llama-3.3-70b-versatile` |
| `GROQ_API_KEY` | Required when `LLM_PROVIDER=groq` | — |
| `OPENAI_API_KEY` | Required when `LLM_PROVIDER=openai` | — |
| `LLM_TIMEOUT_SECONDS` | Model call budget; judge kills us at 30s | `12` |
| `LLM_MAX_RETRIES` | Retries on a failed model call | `1` |
| `SOLVER_TIMEOUT_SECONDS` | CBC time limit | `10` |
| `PORT` | Listen port (Render/Railway/Fly inject this) | `8000` |
| `DEBUG` | Verbose logging | `false` |

**Model in use for submission:** Groq — `openai/gpt-oss-120b`, called through
Groq's OpenAI-compatible `/chat/completions` endpoint at `temperature: 0`,
`response_format: json_object`, `reasoning_effort: low`. One call interprets all
1–3 notes together. The `openai` provider shares the same code path, so an
OpenAI key is a drop-in backup.

| Variable | Purpose | Default |
|---|---|---|
| `LLM_REASONING_EFFORT` | `low` \| `medium` \| `high`, gpt-oss only | `low` |
| `LLM_BUDGET_SECONDS` | Total interpreter wall-clock across retries | `22` |

`LLM_PROVIDER=stub` runs the service with no model at all: every note degrades
to `no_op` and a valid (but directive-free) schedule is still returned. It
exists for offline optimizer work and is **not** a valid submission mode.

---

## 4. Testing against the public samples

With the service running:

```bash
python scripts/run_public_cases.py
# or against the deployed service:
python scripts/run_public_cases.py --base-url https://<your-app>.onrender.com
```

The script POSTs all 10 public cases and scores each one the way the judge
does, on three independent axes:

- **INTERP** — `directive_type`, `hours` and numeric values vs the published
  expectation, entry ordering, and `applies` semantics
- **VALID** — the returned `hourly_plan` replayed against the **ground-truth**
  directives from the sample pack, not against our own interpretation; it also
  re-derives the three totals from `hourly_plan` and compares
- **COST** — `min(1, reference_cost / our_cost)`

Expected output when everything is working:

```
case           interp   valid    cost      ms
------------------------------------------------
SAMPLE-01    2/2           ok   1.000      ...
...
interpretation 18/18 (100%)   valid 10/10   mean cost ratio 1.000
```

Exit code is `0` only when every case is valid and every interpretation
matches. Useful flags: `--case SAMPLE-03`, `-v` for per-case detail.

The optimizer's own edge-case suite (infeasible grid caps, reserve-vs-neutrality
conflicts, total solar outage, overlapping directives) runs standalone and
needs no server or API key:

```bash
python -m app.optimizer
```

---

## 5. Docker fallback

Built and published by GitHub Actions on every push
(`.github/workflows/docker-publish.yml`), which also starts the container and
curls `/health` before the build is allowed to pass.

```bash
docker pull ghcr.io/jahin-tazwar/innovex:latest

docker run --rm -p 8000:8000 \
  -e LLM_PROVIDER=anthropic \
  -e ANTHROPIC_API_KEY=... \
  ghcr.io/jahin-tazwar/innovex:latest

curl -s http://127.0.0.1:8000/health
```

The image exposes port **8000**, binds `0.0.0.0`, runs as a non-root user
(uid 10001) and has a built-in `HEALTHCHECK`. **No credentials are baked in** —
every key arrives via `-e` at runtime. Pin an exact build with the immutable
`:sha-<commit>` tag instead of `:latest`.

---

## 6. Dependencies and credits

| Package | Why |
|---|---|
| `fastapi` + `uvicorn` | HTTP service and ASGI server |
| `pydantic` v2 | Request/response validation, the frozen contract |
| `pulp` | Linear programming; bundles the CBC solver |
| `httpx` | Async HTTP to the model provider; sync client in the test script |
| `python-dotenv` | Local `.env` loading |

The LP formulation, guardrail logic, energy accounting and replay validator are
the team's own work. AI coding assistants were used during development, as
permitted by the rulebook. The public sample-case pack in `samples/` is the
organizers' own artifact, included verbatim for local testing.

---

## 7. Secret handling

- `.env` and `.env.*` are gitignored; only `.env.example` (names, no values) is
  committed.
- No key, token or raw model output appears in logs. Request logging records
  scenario id, note count, directive count, cost and latency — never note text.
- Error responses are `{"error": "..."}` with a short field-level detail and no
  stack trace, no input echo.
- The Docker image contains no credentials.

---

## 8. Known limitations

- **Infeasible scenarios.** If hard directives genuinely contradict each other
  (e.g. a reserve at hour 23 above the level end-of-day neutrality forces), no
  valid plan exists. We return the safe baseline rather than a 500. §5.1 of the
  problem statement promises scoring scenarios will not be contradictory.
- **The baseline fallback is not cheap.** It uses available solar, buys the
  rest and leaves the battery idle. It satisfies balance, battery bounds, rate
  limits, neutrality and the two window directives, but not
  `minimum_battery_reserve` or `max_grid_window`. It exists so a solver failure
  still yields a valid response.
- **Free-tier hosting sleeps.** A cold Render instance takes ~30s to wake. The
  `/health` endpoint is used as a keep-alive target during the judging window.
- **Model latency dominates.** The LP solves in ~80 ms; everything above that
  in p95 is the provider round-trip.
- Overlapping directives of the same type compound (two `solar_reduction`
  factors on one hour multiply; the tightest grid cap and the highest reserve
  win). The spec does not define this case explicitly; we chose the
  interpretation that never relaxes a stated constraint.
