# Regime Optimization — Volatility Forecasting & Market-Regime Research

A research codebase for **Indian markets** (NIFTY, Bank Nifty, Sensex, Reliance, and
MCX commodities) that learns indicator/consensus parameters from stored history and,
more importantly, forecasts **realized volatility, market regimes, and spikes** —
honestly, with strict no-look-ahead validation.

> **Key honest finding:** intraday *direction* is near-efficient (~0.53 AUC), but
> **daily realized volatility is genuinely forecastable** (HAR-RV R² ≈ 0.43–0.60),
> and the **simple HAR model beats gradient boosting and transformers**. Direction is
> a mirage; volatility is the real, defensible signal.

## What's here

### Volatility forecasting (the strong result)
- `daily_rv_forecast.py` — daily realized variance from 1-min bars; HAR vs GBM vs GARCH vs naive (R²/RMSE/QLIKE).
- `vol_cone.py` — HAR forecast + probability cones, overnight-adjusted, GARCH + HMM-regime conditioning.
- `vol_features.py` — HAR-RV, Garman-Klass/Yang-Zhang/Parkinson, time-of-day, Hawkes intensity (all causal).
- `sigma_backtest.py` — calibration backtest: does the realized price land inside the 1σ/1.5σ/2σ bands the expected % of the time?
- `hmm_regime.py` — causal (filtered, walk-forward) Gaussian-HMM volatility regime detector.

### Live dashboard (React + Python)
- `vol_server.py` — stdlib `http.server` backend + React (CDN) frontend; σ-band vol cones, price ranges, regime, cross-asset correlation, live intraday panel. Auto-refresh.
- `vol_dashboard_data.py` — computes everything the dashboard shows.
- `build_vol_dashboard.py` — static HTML variant.

### Intraday signal research
- `analyze_indicators.py`, `predict_selected.py`, `backtest_selected.py`, `ml_direction.py`,
  `ml_cascade.py`, `walk_forward.py`, `movement_skill.py`, `gate_spike_test.py`,
  `indicators_by_regime.py`, `prespike_pv.py` — hit-ratio, movement/spike, and
  regime-conditioned indicator studies.
- `optimization/` — the indicator-optimization package (config, search space, objective,
  splitter with embargo, weight/param/NN/DL optimizers, liquidity/volume-profile/spike
  feature modules, store).

### Pipeline / data producers (data not in repo)
- `nifty_50.py`, `nifty_500.py`, `nifty_index.py` — download OHLCV (yfinance).
- `start.py` — `IndicatorLibrary` (~400 TA-Lib indicators), **audited causal** (no look-ahead).
- `resample_intraday.py`, `compute_indicators_intraday.py` — intraday pipeline.
- `Consensus_predictor .py`, `Hit_ratio_backtester .py`, `generate_summary*.py`, `update_dashboard.py`.

## Setup

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
# GPU (RTX 50xx / Blackwell needs cu128):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# optional: uv pip install hmmlearn arch
```

Tests: `python -m pytest tests/unit -q`
(On Windows set `KMP_DUPLICATE_LIB_OK=TRUE` to avoid the OpenMP duplicate-runtime abort.)

## Run the volatility dashboard

```bash
python vol_server.py --port 8000      # -> http://localhost:8000
```

## Notes
- **No look-ahead is non-negotiable.** A leak audit of the legacy indicator library
  (`np.gradient`/`argrelextrema`/`np.roll`) was the key correction — it had inflated
  earlier results; everything here is causal/walk-forward.
- Market data (CSVs, `processed_*`, `intraday/`) is multi-GB and **gitignored** —
  regenerate via the pipeline scripts.

*Research code, not investment advice.*
