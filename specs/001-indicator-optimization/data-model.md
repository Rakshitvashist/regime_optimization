# Phase 1 Data Model: Indicator Optimization

Entities derived from the spec's Key Entities and Functional Requirements. These are logical models
(fields + rules + relationships); the on-disk JSON shapes are formalized in `contracts/`.

---

## Entity: IndicatorConfig

The tunable settings that govern how signals are produced and combined. Has a **default** form
(reproducing today's hardcoded behavior) and zero or more **tuned** forms.

| Field | Type | Notes / Validation |
|-------|------|--------------------|
| `indicator_params` | map<string, number\|int> | e.g. `rsi_period`, `bb_period`, `bb_std`, `atr_period`, `supertrend_mult`, MA periods, threshold values. Every key MUST exist in the declared search space; every value MUST fall within that key's allowed range. |
| `weights.category` | map<string, float> | Per-category consensus weights (`trend`, `momentum`, `volatility`, `volume`, `price_action`, `other`). Each ≥ 0. |
| `weights.voting_threshold` | float | In `[0, 1)`. |
| `weights.regime_multipliers` | map<string, float> | e.g. trend boost in trending regime, momentum boost in ranging regime. Each ≥ 0. |
| `is_default` | bool | True for the baseline default config. |

**Rules**
- Default config values MUST equal the current hardcoded values in `start.py` / `Consensus_predictor .py`
  (so default-config predictions are byte-for-byte today's behavior).
- Any field absent from a tuned config inherits the default (sparse overrides allowed).
- The complete tunable surface (keys + ranges) is declared once in `search_space.py` and mirrored in
  `contracts/tuned-config.schema.json`.

---

## Entity: SearchSpace

Declarative bound on what may be tuned. Not persisted per-run (it is code/config), but referenced by
every run for validation and auditability (Principle IV).

| Field | Type | Notes |
|-------|------|-------|
| `param_name` | string | Tunable indicator parameter or weight key. |
| `kind` | enum(`int`, `float`, `categorical`) | Determines optimizer treatment. |
| `low` / `high` | number | Inclusive bounds (for numeric kinds). |
| `choices` | list | For categorical kinds. |
| `affects_recompute` | bool | True if changing it requires recomputing indicators (indicator params) vs only re-running the predictor (weights). Drives R8 performance handling. |

---

## Entity: EvaluationWindow

A contiguous slice of one symbol's stored history split for learning vs validation.

| Field | Type | Notes / Validation |
|-------|------|--------------------|
| `symbol` | string | Universe member. |
| `train_start` / `train_end` | date | Training block bounds. |
| `embargo_bars` | int | ≥ `max(forward_days)` (default 10). Gap removed between train_end and heldout_start. |
| `heldout_start` / `heldout_end` | date | Held-out block bounds; strictly after embargo. |
| `n_bars_total` | int | MUST be ≥ minimum required (longest indicator lookback + embargo + minimum train/held-out sizes), else the symbol is skipped (FR-010). |

**Rules**
- `train_end` < `heldout_start`, with at least `embargo_bars` bars between them (Principle II).
- All bounds derived only from validated stored data; the actual covered window is reported (spec Edge
  Cases: stale/partial data).

---

## Entity: Objective

The scoring definition applied to a config over a held-out window.

| Field | Type | Notes |
|-------|------|-------|
| `forward_days` | list<int> | Default `[5, 10]`. |
| `w_acc` | float | Weight on directional hit-ratio. Default 0.5. |
| `w_ret` | float | Weight on normalized mean directional return. Default 0.5. |
| `score` | float (derived) | `w_acc * hit_ratio + w_ret * norm_return`, aggregated across horizons. |

**Rules**: `w_acc`, `w_ret` explicit, documented, and stored on every artifact (FR-004). Computed via
the existing `HitRatioBacktester` (Principle III).

---

## Entity: OptimizationRun

One execution that searches for a better config for one scope.

| Field | Type | Notes |
|-------|------|-------|
| `run_id` | string | Unique id. |
| `scope` | enum(`global`, `symbol`) | Global → produces global config; symbol → produces per-symbol override candidate. |
| `symbol` | string\|null | Set when scope = symbol. |
| `window` | EvaluationWindow (or per-symbol set for global) | The split(s) used. |
| `objective` | Objective | Scoring config. |
| `seed` | int | RNG seed (reproducibility). |
| `budget` | int | Max search trials for the indicator-param sub-problem. |
| `baseline_metrics` | Metrics | Held-out metrics of the baseline (default for global; global for per-symbol). |
| `tuned_metrics` | Metrics | Held-out metrics of the best candidate. |
| `selected_config` | IndicatorConfig\|null | Best candidate, or null if "no improvement". |
| `accepted` | bool | True only if `tuned_metrics` beats `baseline_metrics` by ≥ min margin (FR-009). |
| `status` | enum(`improved`, `no_improvement`, `skipped`, `failed`) | Drives the summary (FR-012). |
| `skip_reason` / `error` | string\|null | Required when status is `skipped`/`failed` (FR-010, Principle V). |

**State transitions**
```
created → running → (improved | no_improvement | skipped | failed)
                         │
              accepted=true only on "improved"
```
- `improved` → persist a new `TunedConfigArtifact`, baseline retained as history.
- `no_improvement` → baseline/default retained; nothing overwritten (SC-004).
- `skipped` → insufficient data; reason recorded, not counted as failure.
- `failed` → unexpected error; recorded per-symbol, batch continues (Principle V).

---

## Entity: Metrics

| Field | Type | Notes |
|-------|------|-------|
| `hit_ratio` | map<horizon, float> | Directional accuracy per forward horizon. |
| `mean_directional_return` | map<horizon, float> | Avg % return in predicted direction. |
| `objective_score` | float | The hybrid score. |
| `signal_count` | map<horizon, int> | Signals evaluated (guards tiny-sample claims). |
| `train_score` | float | For overfitting visibility (train-vs-held-out gap). |

---

## Entity: TunedConfigArtifact

The durable, versioned output of an accepted run; what the predictor consumes.

| Field | Type | Notes |
|-------|------|-------|
| `schema_version` | string | Artifact schema version. |
| `scope` / `symbol` | enum / string\|null | Level of this config. |
| `config` | IndicatorConfig | The selected tuned settings. |
| `objective` | Objective | Including `w_acc`/`w_ret`, `forward_days`. |
| `window` | EvaluationWindow summary | Train/embargo/held-out bounds actually used. |
| `seed` | int | Reproducibility. |
| `heldout_metrics` / `baseline_metrics` | Metrics | For audit + improvement proof. |
| `created_at` | datetime (injected) | Provenance; passed in, not read inside pure logic (R7). |
| `version` | int | Monotonic per (scope, symbol); never overwritten in place. |

**Relationships**
- `TunedConfigArtifact` is produced by exactly one `OptimizationRun` (when `accepted`).
- `IndicatorConfig` is embedded in both the run and the artifact.
- A symbol may have many artifact versions; **precedence at prediction time**: latest accepted
  per-symbol override (if it beats global) → latest accepted global → default (FR-008).

---

## Storage layout (filesystem)

```text
configs/
├── global/
│   └── global.v<NN>.json          # versioned global TunedConfigArtifact
└── per_symbol/
    └── <SYMBOL>/
        └── <SYMBOL>.v<NN>.json     # versioned per-symbol override artifact
runs/                               # optional: per-run summaries (run-summary schema)
└── <run_id>.json
```

All paths are read-only with respect to OHLCV/indicator inputs; only `configs/` and `runs/` are written.
