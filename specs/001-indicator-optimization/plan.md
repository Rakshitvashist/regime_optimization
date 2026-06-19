# Implementation Plan: Indicator Optimization from Historical Data

**Branch**: `001-indicator-optimization` | **Date**: 2026-06-10 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-indicator-optimization/spec.md`

## Summary

Add a learning step that searches each symbol's stored history for the indicator parameters **and**
consensus weights that best predict forward price moves, scored by a **hybrid objective** (directional
hit-ratio + realized return). It learns a **global** configuration across the universe plus
**per-symbol overrides** kept only when they beat the global on a held-out window, persists each
result as a versioned artifact, and lets the existing `ConsensusPredictor` consume the tuned config
via a precedence chain (per-symbol → global → default). Selection is always on out-of-sample data with
a forward-return embargo to guarantee no look-ahead bias.

Technical approach: split the problem into a cheap **continuous weight sub-problem** (the consensus is
differentiable in the category/voting weights given fixed indicator signals → gradient descent on a
smooth surrogate loss, honoring the user's "backpropagation" intent) and an expensive **discrete
indicator-parameter sub-problem** (talib periods/thresholds are non-differentiable and require
recompute → derivative-free search, e.g. TPE/Bayesian or staged coordinate search, under a bounded
evaluation budget). A walk-forward train/held-out split with an embargo gap drives validation;
artifacts are JSON; runs are seeded and reproducible.

## Technical Context

**Language/Version**: Python 3.10+ (matches existing scripts; talib/numba/sklearn stack)

**Primary Dependencies**: pandas, numpy, scipy, TA-Lib (`talib`), numba, scikit-learn (existing);
**proposed addition**: a derivative-free optimizer library (Optuna preferred — TPE sampler, seedable,
pruning, parallelism) for the discrete indicator-parameter sub-problem. Weight sub-problem uses plain
numpy gradient descent (no new dependency) or scipy.optimize.

**Storage**: Filesystem. Inputs = existing OHLCV CSVs (`nifty50_host/`, `nifty500_host/`) and/or
processed indicator CSVs (`processed_indicators/`), treated read-only. Outputs = versioned JSON config
artifacts under `configs/` and per-run summary JSON/console reports.

**Testing**: pytest for unit/contract tests (no test suite exists yet — to be introduced); the existing
`HitRatioBacktester` is the validation/scoring engine reused inside the objective.

**Target Platform**: Local/CI batch execution on Windows + Linux (developer runs the pipeline; same
environment as `update_dashboard.py`).

**Project Type**: Single-project Python CLI/batch pipeline (flat script layout, orchestrated by
`update_dashboard.py`).

**Performance Goals**: Single-symbol weight-only optimization completes in seconds–low minutes; a
full NIFTY 50 batch (weights + bounded indicator-param search) completes within a normal pipeline run
(target < ~30 min on a typical multi-core dev machine) by parallelizing across symbols and bounding
evaluation budget. No hard real-time constraint — this is an offline learning step.

**Constraints**:
- **No look-ahead bias (non-negotiable)**: forward-return embargo of at least `max(forward_days)` bars
  between train and held-out windows; rolling stats/scalers fit on past only.
- Inputs are immutable; the optimizer never writes to OHLCV/indicator data.
- Recomputing 400+ indicators per candidate is the dominant cost → indicator-param search runs under a
  bounded budget and/or on a reduced tunable subset; weight search avoids recompute entirely.
- Deterministic/seeded → identical inputs reproduce identical artifacts and metrics.

**Scale/Scope**: NIFTY 50 (≈50 symbols) for MVP, extensible to NIFTY 500 (≈500 symbols). ~400
indicators; tunable parameter surface bounded to a documented subset (lookback periods, key thresholds,
multipliers) plus ~6 category weights + voting threshold + regime multipliers.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Constitution v1.0.0 — evaluated against all five principles:

| Principle | Gate | Status |
|-----------|------|--------|
| I. Data Integrity & Reproducibility | Optimizer consumes stored CSVs read-only; never mutates/back-fills; runs are seeded and reproducible (FR-001, FR-014). | ✅ PASS |
| II. No Look-Ahead Bias (NON-NEGOTIABLE) | Train/held-out split with forward-return embargo; signals use only data ≤ decision bar; forward outcomes only score, never construct (FR-002, FR-003). Design adds explicit embargo = max(forward_days). | ✅ PASS |
| III. Backtest-Validated Signals | Every run reports tuned vs baseline held-out hit-ratio; non-improving configs rejected (FR-006, FR-009). Reuses `HitRatioBacktester`. | ✅ PASS |
| IV. Modular Indicators & Transparent Consensus | Tunable surface explicitly bounded/documented; config is decomposable and traceable; `IndicatorLibrary` becomes config-driven without losing per-indicator inspectability (FR-005). | ✅ PASS |
| V. Observability & Auditability | Per-run human-readable progress + summary; per-symbol skip/fail reasons surfaced, no silent partial success (FR-010, FR-011, FR-012). | ✅ PASS |

**Result**: PASS — no violations. No entries required in Complexity Tracking. (Re-checked after Phase 1
design below — still PASS.)

## Project Structure

### Documentation (this feature)

```text
specs/001-indicator-optimization/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output — method & methodology decisions
├── data-model.md        # Phase 1 output — entities, config schema, state
├── quickstart.md        # Phase 1 output — runnable validation guide
├── contracts/           # Phase 1 output — CLI + artifact schemas
│   ├── cli-optimize.md          # CLI command contract
│   ├── tuned-config.schema.json # Tuned configuration artifact schema
│   └── run-summary.schema.json  # Optimization run summary schema
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

The repo uses a flat, script-based layout where each stage is a standalone script invoked by
`update_dashboard.py`. New code follows that convention: one CLI entry script plus a small
`optimization/` package for the logic, a `configs/` artifact store, and a `tests/` suite.

```text
# Existing (unchanged inputs / integration points)
nifty_50.py, nifty_500.py          # data download (read-only producers)
start.py                           # IndicatorLibrary — refactored to accept an indicator-param config
Consensus_predictor .py            # ConsensusPredictor — accepts a weight config + records config level used
Hit_ratio_backtester .py           # HitRatioBacktester — reused as the scoring engine
generate_summary.py / _500.py      # predictor consumers — resolve tuned config via precedence
update_dashboard.py                # pipeline orchestrator — optional new "optimize" step

# New
optimize_indicators.py             # CLI entry: single-symbol / batch / global / per-symbol runs
optimization/
├── __init__.py
├── config.py                      # Config dataclasses: defaults, param ranges, (de)serialization
├── search_space.py                # Bounded, documented tunable surface (params + ranges)
├── objective.py                   # Hybrid objective (accuracy + return) over held-out window
├── splitter.py                    # Walk-forward train/held-out split with embargo
├── weight_optimizer.py            # Gradient descent on consensus weights (differentiable surrogate)
├── param_optimizer.py             # Derivative-free search over indicator parameters (Optuna/TPE)
├── runner.py                      # Orchestrates a run: scope, baseline, search, accept/reject, persist
└── store.py                       # Versioned artifact read/write + precedence resolution

configs/                           # Tuned configuration artifacts (JSON), versioned
├── global/
└── per_symbol/

tests/
├── unit/                          # splitter embargo, objective, search-space bounds, store I/O
├── contract/                      # artifact schema conformance, CLI contract
└── integration/                   # end-to-end single-symbol + batch, no-look-ahead assertions
```

**Structure Decision**: Single-project flat layout (Option 1, adapted to the repo's existing
script-per-stage convention). The optimizer is a new CLI entry (`optimize_indicators.py`) backed by an
`optimization/` package; tuned artifacts live in `configs/`; the existing predictor/indicator/backtester
modules are extended (config-driven) rather than replaced, preserving the current pipeline.

## Complexity Tracking

> No constitution violations — section intentionally empty.

The one notable refactor (making `IndicatorLibrary` config-driven so indicator parameters can vary) is
required by FR-005 and is consistent with Principle IV (modular, inspectable indicators); it is not a
constitutional deviation and needs no justification entry.
