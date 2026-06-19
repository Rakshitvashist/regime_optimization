# Indicator Optimizer (feature 001-indicator-optimization)

Learns optimal **indicator parameters** and **consensus weights** from each symbol's
stored history, so the consensus predictor uses settings validated against realized
forward price moves instead of hand-chosen defaults.

- **Objective**: hybrid — directional hit-ratio + ATR-relative directional return.
- **Levels**: a **global** config plus **per-symbol overrides** (kept only when they
  beat the global on held-out data).
- **No look-ahead**: walk-forward split with a forward-return embargo (≥ max horizon).
- **Reproducible**: seeded; non-improving runs never overwrite a good config.

See the full spec under `specs/001-indicator-optimization/` (spec, plan, research,
data-model, contracts, quickstart).

## Install

```bash
pip install -r requirements.txt   # adds optuna, pytest, jsonschema to the existing stack
```

TA-Lib is required for the indicator-parameter recompute path (`--tune params|both`).
The weights-only path (`--tune weights`) needs only numpy/pandas.

## CLI

```bash
# One symbol (MVP): tune weights only, report baseline vs tuned held-out metrics
python optimize_indicators.py --scope symbol --symbol RELIANCE --tune weights --seed 42

# One symbol: tune indicator params + weights (needs TA-Lib + Optuna)
python optimize_indicators.py --scope symbol --symbol RELIANCE --tune both --budget 100

# Whole universe: learn a global config, then per-symbol overrides
python optimize_indicators.py --scope batch --input processed_indicators --jobs 4

# Global config only
python optimize_indicators.py --scope global --input processed_indicators
```

Key flags (full contract: `specs/001-indicator-optimization/contracts/cli-optimize.md`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--scope` | (required) | `symbol` \| `global` \| `batch` |
| `--method` | `numgrad` | `numgrad` (numerical-gradient weight search) \| `nn` (pure-NumPy backprop, linear per-indicator model + L1/L2) |
| `--nn-l1` / `--nn-l2` | `1e-3` | NN L1 (sparsity/selection) / L2 (stability) |
| `--nn-lr` / `--nn-epochs` | `0.1` / `300` | NN learning rate / training epochs |
| `--tune` | `both` | `weights` \| `params` \| `both` (numgrad method only) |
| `--forward-days` | `5,10` | objective horizons |
| `--w-acc` / `--w-ret` | `0.5` / `0.5` | objective mix (accuracy vs return) |
| `--embargo` | `max(forward_days)` | look-ahead embargo bars |
| `--budget` | `100` | Optuna trials for parameter search |
| `--min-improvement` | `0.0` | min held-out gain required to accept |
| `--seed` | `42` | reproducibility |
| `--jobs` | `1` | parallel workers (batch) |
| `--dry-run` | off | compute + report, write nothing |

Exit codes: `0` success, `1` config/IO error, `2` batch completed with ≥1 failed symbol.

## Artifacts & precedence

Accepted configs are written as versioned JSON (never overwritten in place):

```
configs/
├── global/global.vNN.json
└── per_symbol/<SYMBOL>/<SYMBOL>.vNN.json
runs/<run_id>.json          # batch run summaries
```

At prediction time (`generate_summary.py`, `generate_summary_500.py`) the config is
resolved by precedence and the chosen level is recorded on each prediction
(`config_level` column):

```
per-symbol override  →  global config  →  default
```

If the optimizer package or its artifacts are absent, prediction falls back to the
default config (today's behaviour) unchanged.

## Pipeline integration

`update_dashboard.py --optimize` runs a batch optimization (after indicators, before
summaries) so the dashboard reflects tuned configs. Without `--optimize`, the pipeline
is unchanged.

## How the optimization works

- **Weights** (category weights, voting threshold, regime boosts) are tuned by
  gradient ascent on the objective over the TRAIN window — gradient "backpropagation"
  through the weighting layer (`optimization/weight_optimizer.py`).
- **Indicator parameters** (periods, BB std, …) are discrete and require an indicator
  recompute per candidate, so they are searched with Optuna's seeded TPE sampler under
  a trial budget, with within-run caching (`optimization/param_optimizer.py`).
- Candidates are **selected on the held-out window**, never on training, and accepted
  only if they beat the baseline by `--min-improvement`.

## Two optimization methods (compare head-to-head)

| | `numgrad` (default) | `nn` (neural net, backprop) |
|---|---|---|
| Parameters learned | ~9 consensus knobs (6 category weights + threshold + 2 regime boosts) | ~400 per-indicator weights + bias |
| How | Numerical (finite-difference) gradient ascent | Hand-coded backpropagation (real chain rule), full-batch GD + momentum, in pure NumPy (no PyTorch) |
| Regularization | bounded search space | L1 (sparsity → indicator selection) + L2 (stability) |
| Indicator params | Optuna TPE (`--tune params/both`) | not tuned (NN reweights existing indicators) |
| Strength | Robust, low-overfit, interpretable | More granular; L1 reveals *which* indicators matter |
| Risk | Limited expressiveness | Overfitting on small per-symbol data — strongest on the pooled global config |

Both use the **same** train/held-out embargo split and the **same** hybrid objective, so
their held-out numbers are directly comparable.

```bash
# A/B the same symbol on held-out (identical split + objective):
python optimize_indicators.py --scope symbol --symbol RELIANCE --method numgrad --seed 42
python optimize_indicators.py --scope symbol --symbol RELIANCE --method nn --seed 42
```

The `nn` run reports its non-zero weight count and top indicators (the L1 selection result).
NN configs are not yet persisted through the schema store (experimental method); they are
reported for comparison.

### Global (pooled) NN — where the neural net should win

A per-symbol NN is data-starved. The `nn` method is most powerful on the **global** scope,
where every symbol's TRAIN rows are **pooled into one fit** (≈ N_symbols × more data), then
the single global model is validated on each symbol's held-out window:

```bash
# Pooled neural-net global config across the universe, vs the numgrad global:
python optimize_indicators.py --scope global --input processed_indicators --method nn --seed 42
python optimize_indicators.py --scope global --input processed_indicators --method numgrad --seed 42
```

The pooled `nn` global reports how many symbols were pooled and the surviving (non-zero) L1
indicator weights — a universe-wide view of which indicators actually carry signal. Tune
`--nn-l1` up if the held-out objective trails the train objective (overfitting). In `--scope
batch` with `--method nn`, the pooled global is learned first, then per-symbol NN overrides are
evaluated (judged against the default baseline, since the NN global config has a different shape
than the numgrad config).

## Tests

```bash
pytest -q
```

Covers: splitter embargo, hybrid objective math, no-look-ahead, single-symbol run +
determinism, artifact/run-summary schema conformance, persistence/precedence,
search-space bounds, and batch accounting.
