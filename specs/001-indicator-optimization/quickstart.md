# Quickstart & Validation Guide: Indicator Optimization

A runnable guide proving the feature works end-to-end. Implementation details live in `tasks.md` and the
code; this file is the validation/run script. Schemas and entities are referenced, not duplicated — see
[`data-model.md`](./data-model.md) and [`contracts/`](./contracts/).

## Prerequisites

- The existing pipeline works: OHLCV downloaded (`nifty50_host/`) and indicators computed
  (`processed_indicators/*_with_indicators.csv`). If not, run the current pipeline first:
  ```text
  python update_dashboard.py --skip-push --skip-nifty500
  ```
- Python deps installed: existing stack (`pandas numpy scipy TA-Lib numba scikit-learn`) plus the
  optimizer dependency (`optuna`) and `pytest` for tests.
- Optimizer code present: `optimize_indicators.py` + `optimization/` package + `configs/` store.

## Scenario 1 — Single-symbol optimization improves held-out accuracy (User Story 1, P1)

```text
python optimize_indicators.py --scope symbol --symbol RELIANCE --seed 42
```

**Expected**:
- Console prints baseline vs tuned held-out hit-ratio and the hybrid objective for both.
- If tuned beats baseline → a versioned artifact `configs/per_symbol/RELIANCE/RELIANCE.v1.json` is
  written, conforming to [`tuned-config.schema.json`](./contracts/tuned-config.schema.json).
- If not → console reports `no_improvement` and **no** artifact is written.
- Exit code `0`.

**Validates**: FR-001, FR-002, FR-004, FR-005, FR-006, SC-003.

## Scenario 2 — No look-ahead bias (Principle II, non-negotiable)

```text
pytest tests/integration/test_no_lookahead.py -q
```

**Expected**: Tests assert that (a) the held-out block starts at least `embargo` bars (≥ max
forward-day horizon) after `train_end`, and (b) perturbing data at/after each decision bar never changes
that bar's signal. All pass.

**Validates**: FR-003, FR-002 embargo.

## Scenario 3 — Non-improving run never overwrites a good config

```text
# Run twice with identical inputs; the second run should find no improvement over the first.
python optimize_indicators.py --scope symbol --symbol RELIANCE --seed 42
python optimize_indicators.py --scope symbol --symbol RELIANCE --seed 42
```

**Expected**: Second run reports `no_improvement` (or reproduces the identical config); the stored
artifact is **not** replaced by a worse one. No artifact regresses in held-out objective.

**Validates**: FR-009, SC-004.

## Scenario 4 — Reproducibility (determinism)

```text
python optimize_indicators.py --scope symbol --symbol TCS --seed 7 --dry-run
python optimize_indicators.py --scope symbol --symbol TCS --seed 7 --dry-run
```

**Expected**: Both runs print byte-identical tuned config values and metrics.

**Validates**: FR-014, SC-005.

## Scenario 5 — Predictor consumes the tuned config via precedence (User Story 2, P2)

```text
# After Scenario 1 produced a per-symbol override:
python generate_summary.py --input processed_indicators --output dashboard/public/summary.json
```

**Expected**: For RELIANCE the prediction records config level `symbol` (override used); for a symbol
with only a global config it records `global`; for a symbol with neither it records `default`.

**Validates**: FR-008, FR-013, User Story 2 acceptance scenarios.

## Scenario 6 — Universe batch + global config + full accounting (User Story 3, P3)

```text
python optimize_indicators.py --scope batch --input processed_indicators --jobs 4 --seed 42
```

**Expected**:
- Learns a global config, then per-symbol overrides; writes `configs/global/global.vN.json` if improved.
- Prints a summary in which **every** universe member is `improved | no_improvement | skipped | failed`,
  with reasons for skips/failures. Counts in the summary sum to the universe size (no silent omissions).
- Optional `runs/<run_id>.json` conforms to [`run-summary.schema.json`](./contracts/run-summary.schema.json).
- Exit code `0` (or `2` if any symbol `failed`, with failures listed).

**Validates**: FR-010, FR-011, FR-012, SC-006.

## Scenario 7 — Insufficient-history symbol is skipped, not failed

```text
python optimize_indicators.py --scope symbol --symbol <THIN_HISTORY_SYMBOL>
```

**Expected**: Reports `skipped` with reason `insufficient history (needs >= N bars, has M)`; exit code
`0`; no artifact written.

**Validates**: FR-010, spec Edge Cases.

## Success-criteria validation (aggregate)

After a full `--scope batch` run, confirm against [spec.md](./spec.md) Success Criteria:

| Check | How |
|-------|-----|
| SC-001 (≥60% of eligible symbols improve) | From run summary: `improved / (targeted − skipped)` ≥ 0.60. |
| SC-002 (avg ≥ +3pp where improved) | Mean held-out hit-ratio delta over `improved` symbols ≥ 0.03. |
| SC-003 (100% report both metrics) | Every symbol entry has tuned + baseline objective. |
| SC-004 (0% accepted worse) | No accepted artifact has held-out objective < its baseline. |
| SC-005 (determinism) | Scenario 4 passes. |
| SC-006 (full accounting) | `len(symbols) == totals.targeted` and statuses sum correctly. |

## Run the full test suite

```text
pytest -q
```

Expected: unit (splitter embargo, objective math, search-space bounds, store I/O), contract (artifact +
run-summary schema conformance, CLI contract), and integration (Scenarios 1–7) tests all pass.
