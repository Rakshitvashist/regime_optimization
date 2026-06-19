"""
Shared pytest fixtures.

Prefers a small real symbol CSV from processed_indicators/ when present; otherwise
falls back to deterministic synthetic data so the core logic is testable without
TA-Lib or downloaded market data. All synthetic data is seeded -> reproducible.
"""
import glob
import os

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Indicator column names deliberately contain category keywords used by
# ConsensusPredictor._group_indicators (trend/momentum/volatility/volume/price_action).
_TREND = ["price_above_sma_20", "ema_slope_50"]
_MOMENTUM = ["rsi_14", "macd_hist", "cci_20"]
_VOLATILITY = ["bb_pos_20", "atr_norm_14"]
_VOLUME = ["obv_ma_20", "vwap_dist"]
INDICATOR_COLS = _TREND + _MOMENTUM + _VOLATILITY + _VOLUME


def make_ohlcv(n=400, seed=0):
    """Deterministic OHLCV with mild momentum so direction is partly predictable."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, n)
    for i in range(1, n):
        rets[i] += 0.12 * rets[i - 1]  # slight autocorrelation -> learnable signal
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1.0 + rng.normal(0, 0.003, n))
    vol = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def make_indicators(ohlcv, seed=0):
    """Indicator signals in [-1, 1]; some partly aligned with the 5-day forward sign."""
    rng = np.random.default_rng(seed + 1)
    close = ohlcv["close"]
    n = len(close)
    fwd_sign = np.sign(close.pct_change(5).shift(-5)).fillna(0.0).to_numpy()
    # alignment strength per column (some predictive, some noise)
    strength = {
        "price_above_sma_20": 0.55, "ema_slope_50": 0.40,
        "rsi_14": 0.50, "macd_hist": 0.30, "cci_20": 0.35,
        "bb_pos_20": 0.00, "atr_norm_14": 0.05,
        "obv_ma_20": 0.20, "vwap_dist": 0.15,
    }
    data = {}
    for col in INDICATOR_COLS:
        a = strength[col]
        sig = a * fwd_sign + (1 - a) * rng.normal(0, 1, n)
        data[col] = np.clip(sig, -1.0, 1.0)
    return pd.DataFrame(data, index=close.index)


@pytest.fixture
def ohlcv():
    return make_ohlcv()


@pytest.fixture
def indicators(ohlcv):
    return make_indicators(ohlcv)


@pytest.fixture
def symbol_csv(tmp_path, ohlcv, indicators):
    """Write a `<SYMBOL>_with_indicators.csv` (generate_summary format) and return (dir, symbol, path)."""
    df = ohlcv.join(indicators)
    df.index.name = "date"
    symbol = "SYNTH"
    out_dir = tmp_path / "processed_indicators"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_with_indicators.csv"
    df.reset_index().to_csv(path, index=False)
    return str(out_dir), symbol, str(path)


@pytest.fixture
def real_or_synth_csv(symbol_csv):
    """Use a real processed_indicators CSV if one exists; else the synthetic one."""
    real = glob.glob(os.path.join(REPO_ROOT, "processed_indicators", "*_with_indicators.csv"))
    if real:
        sym = os.path.basename(real[0]).split("_")[0]
        return os.path.join(REPO_ROOT, "processed_indicators"), sym, real[0]
    return symbol_csv
