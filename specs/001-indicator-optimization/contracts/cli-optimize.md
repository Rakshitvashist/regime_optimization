# CLI Contract: `optimize_indicators.py`

The optimizer's external interface is a command-line tool, consistent with the repo's script-per-stage
convention (invoked directly or via `update_dashboard.py`). Text/JSON in → artifacts + summary out;
progress to stdout, errors to stderr.

## Synopsis

```text
python optimize_indicators.py --scope <global|symbol|batch> [options]
```

## Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--scope` | enum(`global`, `symbol`, `batch`) | required | `global`: learn one config across the universe. `symbol`: learn one per-symbol override (requires `--symbol`). `batch`: learn global, then per-symbol overrides for every universe member. |
| `--symbol` | string | — | Required when `--scope symbol`. e.g. `RELIANCE`. |
| `--input` | path | `processed_indicators` | Directory of `*_with_indicators.csv` (weights-only) and/or raw OHLCV dir for indicator-param recompute. |
| `--configs-dir` | path | `configs` | Output artifact store (read for baselines/precedence, written for accepted configs). |
| `--forward-days` | int list | `5,10` | Forward horizons for the objective. |
| `--w-acc` | float | `0.5` | Objective weight on directional hit-ratio. |
| `--w-ret` | float | `0.5` | Objective weight on directional return. |
| `--embargo` | int | `max(forward_days)` | Embargo bars between train and held-out (≥ max horizon). |
| `--heldout-frac` | float | `0.3` | Fraction of usable history reserved as the held-out tail. |
| `--budget` | int | `100` | Max search trials for the indicator-parameter sub-problem. |
| `--tune` | enum(`weights`, `params`, `both`) | `both` | Which sub-problems to run. |
| `--min-improvement` | float | `0.0` (strictly greater) | Minimum held-out objective gain required to accept a config. |
| `--seed` | int | `42` | RNG seed (reproducibility). |
| `--jobs` | int | `1` | Parallel worker processes for batch/global. |
| `--dry-run` | flag | off | Compute and report metrics but do not write artifacts. |

## Behavior contract

1. MUST treat all input data as read-only; MUST NOT modify OHLCV or indicator CSVs (Principle I).
2. MUST build a walk-forward split with `--embargo ≥ max(--forward-days)` and select on the held-out
   block only; MUST NOT use any data at/after a signal's decision bar to construct that signal
   (Principle II / FR-003).
3. MUST compute baseline held-out metrics (default for global; global for per-symbol) and accept a
   tuned config only if its held-out objective exceeds baseline by `--min-improvement` (FR-006, FR-009).
4. On acceptance MUST write a new versioned `TunedConfigArtifact` (never overwrite in place); on
   non-acceptance MUST leave the store unchanged and report `no_improvement` (SC-004).
5. MUST skip symbols lacking sufficient history with an explicit reason; MUST surface per-symbol
   failures and continue the batch (FR-010, FR-012, Principle V).
6. MUST be deterministic for fixed inputs + `--seed` (FR-014, SC-005).

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Completed; summary emitted (including runs with `no_improvement`/`skipped`). |
| `1` | Fatal configuration/IO error (e.g. input dir missing, invalid args). |
| `2` | Completed but one or more symbols `failed` during a batch (summary still emitted). |

## stdout summary (human-readable) — required fields

Per the observability principle, every run prints a final summary accounting for **every** targeted
symbol:

```text
Scope: batch | seed=42 | objective: 0.5*acc + 0.5*ret | horizons=[5,10] | embargo=10
Universe: 50 symbols
  improved:       31   (avg held-out hit-ratio +4.2pp vs default)
  no_improvement: 12
  skipped:         5   (insufficient history)
  failed:          2   (see errors below)
Global config: improved (held-out 0.58 vs default 0.54) -> configs/global/global.v3.json
Per-symbol overrides written: 18
Failures:
  - <SYMBOL>: <error one-liner>
Skipped:
  - <SYMBOL>: insufficient history (needs >= N bars, has M)
```

A machine-readable run summary conforming to `run-summary.schema.json` MAY also be written under
`runs/<run_id>.json`.
