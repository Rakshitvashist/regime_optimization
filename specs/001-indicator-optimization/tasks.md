---

description: "Task list for Indicator Optimization from Historical Data"
---

# Tasks: Indicator Optimization from Historical Data

**Input**: Design documents from `specs/001-indicator-optimization/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: INCLUDED. The constitution makes validation mandatory — Principle II (No Look-Ahead Bias,
NON-NEGOTIABLE) and Principle III (Backtest-Validated Signals) — and `quickstart.md` specifies concrete
pytest files. Test tasks are therefore part of each story, not optional.

**Organization**: Tasks are grouped by user story (US1/US2/US3) for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1, US2, US3 (Setup/Foundational/Polish have no story label)
- Exact file paths are included in each task.

## Path Conventions

Single-project, flat script layout (repo root), per `plan.md`. New code: `optimize_indicators.py` (CLI),
`optimization/` package, `configs/` artifact store, `tests/`. Existing modules extended in place. Note:
two existing files contain a literal space in their names — `Consensus_predictor .py` and
`Hit_ratio_backtester .py` — referenced verbatim below.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and structure

- [X] T001 Create the optimizer package skeleton: `optimization/__init__.py`, and empty module files `optimization/config.py`, `optimization/search_space.py`, `optimization/splitter.py`, `optimization/objective.py`, `optimization/data_loader.py`, `optimization/weight_optimizer.py`, `optimization/param_optimizer.py`, `optimization/runner.py`, `optimization/store.py`; create `configs/global/` and `configs/per_symbol/` and `runs/` directories (with `.gitkeep`); create `tests/unit/`, `tests/contract/`, `tests/integration/` with `__init__.py`.
- [X] T002 Create `requirements.txt` at repo root capturing the existing stack (pandas, numpy, scipy, TA-Lib, numba, scikit-learn, yfinance) plus new deps `optuna` and `pytest`.
- [X] T003 [P] Add `tests/conftest.py` with a fixture that loads a small real symbol CSV from `processed_indicators/` (and a synthetic OHLCV fallback) for fast, deterministic tests; add `pytest.ini` setting `testpaths = tests`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core, shared building blocks every user story depends on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Implement config dataclasses in `optimization/config.py`: `IndicatorConfig` (indicator_params, weights.category, weights.voting_threshold, weights.regime_multipliers, is_default) with a `default()` factory whose values EXACTLY reproduce the current hardcoded values in `start.py` and `Consensus_predictor .py`; include sparse-override merge (`merge_with_default`) and JSON (de)serialization. (data-model.md → IndicatorConfig)
- [X] T005 [P] Declare the bounded tunable surface in `optimization/search_space.py`: each entry = name, kind (int/float/categorical), low/high or choices, and `affects_recompute` (True for indicator params, False for weights); include a `validate(config)` that rejects out-of-range values. (data-model.md → SearchSpace; contracts/tuned-config.schema.json)
- [X] T006 Implement the walk-forward splitter in `optimization/splitter.py`: given a symbol series + `forward_days` + `heldout_frac`, return `EvaluationWindow` with an embargo gap ≥ `max(forward_days)` removed between train_end and heldout_start; raise/return a skip with reason when `n_bars_total` < minimum required (longest lookback + embargo + min train/held-out). (data-model.md → EvaluationWindow; Principle II)
- [X] T007 Implement the hybrid objective in `optimization/objective.py`: `score = w_acc*hit_ratio + w_ret*norm_return` aggregated across horizons, computed on a given window by delegating to `HitRatioBacktester` in `Hit_ratio_backtester .py`; return a `Metrics` object (hit_ratio, mean_directional_return, objective_score, signal_count, train_score). (data-model.md → Objective/Metrics; Principle III)
- [X] T008 Implement `optimization/data_loader.py`: load a `*_with_indicators.csv` into `indicators_df`, `quality_df`, `price_series`, `atr_series`, `adx_series` (mirroring the construction in `generate_summary.py` lines ~60–101) and optionally load raw OHLCV from `nifty50_host/` for indicator recompute; treat all inputs read-only. (Principle I)
- [X] T009 Refactor `start.py` `IndicatorLibrary.calculate_all_indicators(df, params=None)` to accept an optional indicator-parameter config, threading values (MA/RSI/BB/ATR periods, BB std, supertrend multiplier, thresholds) from the config; `params=None` MUST reproduce today's exact output. Keep per-indicator modularity/inspectability (Principle IV).
- [X] T010 Refactor `Consensus_predictor .py` `ConsensusPredictor` to accept a weight config (category weights, voting_threshold, regime multipliers) instead of the hardcoded `cat_weights`/`0.15`/`0.2`/`1.2`; defaults MUST reproduce current behavior. Add a `config_level` field plumbed into the returned predictions frame (value filled by US2). (FR-005, FR-008)

**Checkpoint**: Shared core ready — user stories can begin.

---

## Phase 3: User Story 1 - Tune indicators against history and measure the gain (Priority: P1) 🎯 MVP

**Goal**: Run optimization on one symbol over a train window and report tuned vs default held-out
hit-ratio and hybrid objective, learning both indicator parameters and consensus weights, with no
look-ahead.

**Independent Test**: `python optimize_indicators.py --scope symbol --symbol RELIANCE --seed 42` prints
baseline vs tuned held-out metrics; tuned ≥ baseline or a clear "no improvement"; passes the
no-look-ahead test. (quickstart.md Scenarios 1, 2, 4)

### Tests for User Story 1 ⚠️ (write first, ensure they FAIL before implementing)

- [X] T011 [P] [US1] Unit test the splitter embargo in `tests/unit/test_splitter.py`: assert heldout_start ≥ train_end + embargo and embargo ≥ max(forward_days).
- [X] T012 [P] [US1] Unit test the hybrid objective in `tests/unit/test_objective.py`: verify score = w_acc*acc + w_ret*ret on a hand-built series and correct horizon aggregation.
- [X] T013 [P] [US1] Integration test in `tests/integration/test_no_lookahead.py`: perturbing data at/after each decision bar never changes that bar's signal, and held-out scoring uses no train-window future (Principle II / FR-003).
- [X] T014 [P] [US1] Integration test in `tests/integration/test_single_symbol.py`: a single-symbol run returns both tuned and baseline held-out metrics and is deterministic for a fixed seed (FR-006, FR-014, SC-003, SC-005).

### Implementation for User Story 1

- [X] T015 [P] [US1] Implement `optimization/weight_optimizer.py`: gradient descent on the consensus weights using a differentiable surrogate of the hybrid objective (smooth confidence×return + soft directional accuracy); deterministic init; returns best weight config + train score. (research.md R1)
- [X] T016 [P] [US1] Implement `optimization/param_optimizer.py`: derivative-free search (Optuna TPE, seeded sampler, `--budget` trials, pruning) over the `affects_recompute` search-space entries; recomputes indicators via the refactored `IndicatorLibrary` per trial with within-run caching keyed by param set. (research.md R1, R8)
- [X] T017 [US1] Implement `optimization/runner.py` `run_symbol(...)`: build window (T006), compute baseline metrics (default config), run param + weight search per `--tune`, select best candidate on the HELD-OUT window, apply accept/reject vs baseline with `--min-improvement`, and return an `OptimizationRun` result (status improved/no_improvement/skipped). Depends on T015, T016. (FR-001..FR-006, FR-009)
- [X] T018 [US1] Implement `optimize_indicators.py` CLI for `--scope symbol`: parse all args from contracts/cli-optimize.md, invoke `run_symbol`, print human-readable baseline-vs-tuned summary and progress, exit code 0. (contracts/cli-optimize.md)
- [X] T019 [US1] Add deterministic tie-break + seed wiring across optimizers in `optimization/runner.py` and `optimization/param_optimizer.py` (lexicographic tie-break on equal scores) so identical inputs reproduce identical results. (FR-014, SC-005, research.md R7)
- [X] T020 [US1] Handle insufficient-history in `optimization/runner.py`: when the splitter reports too few bars, return status `skipped` with reason `insufficient history (needs >= N bars, has M)` and write nothing. (FR-010, quickstart Scenario 7)

**Checkpoint**: Single-symbol optimization is fully functional and independently testable (MVP).

---

## Phase 4: User Story 2 - Persist and reuse the tuned configuration (Priority: P2)

**Goal**: Persist accepted configs as durable, versioned artifacts and have the predictor consume them
via precedence (per-symbol → global → default), recording which level was used.

**Independent Test**: After an optimize run, a versioned artifact exists; running `generate_summary.py`
records `config_level=symbol` for that symbol, `global`/`default` otherwise. (quickstart.md Scenarios 3, 5)

### Tests for User Story 2 ⚠️

- [X] T021 [P] [US2] Contract test in `tests/contract/test_tuned_config_schema.py`: a written artifact validates against `specs/001-indicator-optimization/contracts/tuned-config.schema.json` (use `jsonschema`).
- [X] T022 [P] [US2] Integration test in `tests/integration/test_persist_and_precedence.py`: accepted run writes `vN`, a non-improving rerun does NOT overwrite (SC-004), and precedence resolution returns symbol→global→default in the right order (FR-008, FR-009).

### Implementation for User Story 2

- [X] T023 [US2] Implement `optimization/store.py` write side: serialize an accepted `TunedConfigArtifact` to `configs/global/global.vNN.json` or `configs/per_symbol/<SYMBOL>/<SYMBOL>.vNN.json` with monotonic version, never overwriting in place; embed full provenance (window, seed, objective, heldout+baseline metrics, injected `created_at`). (data-model.md; contracts/tuned-config.schema.json; FR-007)
- [X] T024 [US2] Implement `optimization/store.py` read/precedence side: `resolve_config(symbol)` returns latest accepted per-symbol override (if it beats global) → latest accepted global → default; expose the chosen level. (FR-008, FR-013)
- [X] T025 [US2] Wire `optimization/runner.py` to persist via `store.py` only when `accepted` and not `--dry-run`; non-improving runs leave the store unchanged. (FR-009, SC-004)
- [X] T026 [US2] Complete `config_level` recording in `Consensus_predictor .py` (from T010): the predictions frame carries the resolved level for each prediction. (FR-008, Principle V)
- [X] T027 [US2] Integrate precedence into `generate_summary.py`: before `predictor.predict(...)`, call `store.resolve_config(symbol)` and apply the indicator-param + weight config (recompute indicators when param overrides exist); fall back to default with the level recorded. (FR-008, User Story 2)
- [X] T027a [US2] Integrate precedence into `generate_summary_500.py`, first reconciling its different predict call shape (it builds synthetic binary indicator signals — `generate_summary_500.py:~90,197`): confirm the resolved config applies to that path or adapt the integration accordingly. (FR-008, User Story 2)

**Checkpoint**: Tuned configs persist and are consumed by the live prediction pipeline; US1 still works.

---

## Phase 5: User Story 3 - Optimize across the universe and review results (Priority: P3)

**Goal**: Batch-optimize the whole universe — learn a global config, then per-symbol overrides adopted
only when they beat global — and emit a summary accounting for every symbol.

**Independent Test**: `python optimize_indicators.py --scope batch --jobs 4` prints a summary where every
universe member is improved/no_improvement/skipped/failed with reasons, counts summing to the universe
size. (quickstart.md Scenario 6)

### Tests for User Story 3 ⚠️

- [X] T028 [P] [US3] Contract test in `tests/contract/test_run_summary_schema.py`: a written run summary validates against `specs/001-indicator-optimization/contracts/run-summary.schema.json` and `len(symbols) == totals.targeted`.
- [X] T029 [P] [US3] Integration test in `tests/integration/test_batch_accounting.py`: a batch over a small symbol set accounts for every symbol (improved+no_improvement+skipped+failed == targeted), with reasons on skip/fail (FR-010, FR-012, SC-006).

### Implementation for User Story 3

- [X] T030 [US3] Implement `run_global(...)` in `optimization/runner.py`: optimize the aggregate held-out objective across all universe symbols to produce one global config; reuses T015/T016/T017 building blocks. (FR-013, research.md R5)
- [X] T031 [US3] Implement `run_batch(...)` in `optimization/runner.py`: run global, then per-symbol overrides, adopting an override only when it beats the global on that symbol's held-out window. Wrap each symbol in try/except so an unexpected error is recorded as `status=failed` with the error message and the batch continues to the next symbol (no silent abort). (FR-012, FR-013, SC-004, Principle V)
- [X] T032 [US3] Add cross-symbol parallelism in `optimization/runner.py` via a process pool honoring `--jobs`, preserving determinism per symbol (per-symbol seeds derived from `--seed`). (research.md R8, FR-014)
- [X] T033 [US3] Implement run-summary emission: aggregate per-symbol statuses into the stdout summary AND optional `runs/<run_id>.json` conforming to the run-summary schema; ensure NO symbol is silently omitted. (FR-011, FR-012, SC-006, Principle V)
- [X] T034 [US3] Extend `optimize_indicators.py` CLI for `--scope global` and `--scope batch`, including exit code 2 when any symbol `failed` (failures listed), 0 otherwise. (contracts/cli-optimize.md)

**Checkpoint**: All three user stories are independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements spanning stories

- [X] T035 [P] Add an off-by-default optimize step to `update_dashboard.py` (e.g. `--optimize` flag invoking `optimize_indicators.py --scope batch`) so the existing dashboard run is unchanged unless requested. (research.md R9)
- [X] T036 [P] Document optimizer usage in `README.md` (or `docs/optimizer.md`): CLI examples, config artifact location, precedence, and the no-look-ahead/reproducibility guarantees.
- [X] T037 [P] Add remaining unit tests in `tests/unit/test_store.py` and `tests/unit/test_search_space.py` (versioned write/no-overwrite, precedence ordering, range validation).
- [ ] T038 Run all `quickstart.md` scenarios 1–7 end-to-end and confirm Success Criteria SC-001…SC-006 from the summary of a full `--scope batch` run.
- [X] T039 [P] Add within-run indicator-recompute caching/perf pass in `optimization/param_optimizer.py` and verify a NIFTY 50 batch completes within the Performance Goal. (research.md R8)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories.
- **User Stories (Phase 3+)**: All depend on Foundational.
  - **US1 (P1)**: After Foundational. No dependency on other stories.
  - **US2 (P2)**: After Foundational; uses US1's runner result to persist (depends on T017 producing an accepted run). Independently testable.
  - **US3 (P3)**: After Foundational; reuses US1's optimizer/runner building blocks (T015–T017). Independently testable.
- **Polish (Phase 6)**: After the desired stories complete (T035 depends on US3; T038 depends on US1–US3).

### Within Each User Story

- Tests written first and FAIL before implementation.
- Foundational config/splitter/objective before optimizers; optimizers before runner; runner before CLI.
- Store write (T023) before precedence read (T024) before pipeline integration (T027).

### Parallel Opportunities

- Setup: T003 ∥ (after T001).
- Foundational: T005 ∥ T006 ∥ T007 ∥ T008 can proceed in parallel after T004; T009 and T010 are independent files and can run in parallel.
- US1 tests T011 ∥ T012 ∥ T013 ∥ T014; implementations T015 ∥ T016 (different files) before T017.
- US2 tests T021 ∥ T022. US3 tests T028 ∥ T029.
- Polish: T035 ∥ T036 ∥ T037 ∥ T039.
- With capacity, US2 and US3 can be developed in parallel once US1's core (T015–T017) lands.

---

## Parallel Example: User Story 1

```bash
# Tests for US1 together (write first, expect failure):
Task: "Unit test splitter embargo in tests/unit/test_splitter.py"
Task: "Unit test hybrid objective in tests/unit/test_objective.py"
Task: "Integration no-look-ahead test in tests/integration/test_no_lookahead.py"
Task: "Integration single-symbol test in tests/integration/test_single_symbol.py"

# Then the two optimizers in parallel (different files):
Task: "Implement optimization/weight_optimizer.py"
Task: "Implement optimization/param_optimizer.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1: Setup.
2. Phase 2: Foundational (CRITICAL — blocks all stories; includes the `start.py` / `Consensus_predictor .py` config refactors).
3. Phase 3: User Story 1 — single-symbol optimize + report gain, no look-ahead.
4. **STOP and VALIDATE**: run quickstart Scenarios 1, 2, 4 for one symbol.
5. Demo the measured held-out improvement.

### Incremental Delivery

1. Setup + Foundational → core ready.
2. US1 → test independently → MVP (measured improvement on a symbol).
3. US2 → persistence + predictor consumes tuned config → dashboard reflects optimized signals.
4. US3 → universe batch + global config + full accounting.
5. Polish → pipeline integration, docs, perf, full Success-Criteria validation.

### Parallel Team Strategy

After Foundational: Dev A → US1 core; once T015–T017 land, Dev B → US2 (store/precedence/integration),
Dev C → US3 (batch/global/parallelism). Stories integrate independently.

---

## Notes

- [P] = different files, no incomplete dependencies. [Story] label maps each task to its user story.
- Two existing modules have a literal space in their filenames (`Consensus_predictor .py`,
  `Hit_ratio_backtester .py`) — reference them exactly when editing.
- Defaults of every refactored module MUST reproduce current behavior (regression-safe).
- No look-ahead (Principle II) and reproducibility (Principle I) are verified by tests, not assumed.
- Commit after each task or logical group; stop at any checkpoint to validate a story independently.
