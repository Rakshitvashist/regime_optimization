"""
vix_features.py  —  causal India-VIX (implied volatility) features, aligned.

VIX is the market's implied vol — directly on-target for predicting realized
movement. Features here are strictly causal: the value at bar t is the last VIX
known at or before the end of bar t (same timing as the instrument's close), so
no look-ahead. Bad near-zero VIX prints (corrupt 2016/2021 rows) are cleaned.

Use:
    from vix_features import load_vix_1min, vix_features
    vc = load_vix_1min("INDIA_VIX.csv")            # cleaned 1-min VIX close (UTC)
    vf = vix_features(target_df.index, vc, "30min") # DataFrame aligned to bars
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULE = {"5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "60min", "4h": "240min", "1d": "1D"}


def rule_for_symbol(symbol: str) -> str:
    """'NIFTY-50-30m' -> '30min'."""
    tf = symbol.rsplit("-", 1)[-1]
    return RULE.get(tf, "30min")


def load_vix_1min(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["DateTime"], utc=True)
    c = df.set_index("DateTime").sort_index()["Close"].astype(float)
    c = c[~c.index.duplicated(keep="last")]
    c = c.where(c >= 5.0).ffill()      # India VIX never < ~8; <5 = corrupt -> ffill
    return c


def vix_features(target_index: pd.DatetimeIndex, vix_close_1min: pd.Series,
                 rule: str) -> pd.DataFrame:
    """Causal VIX features aligned to target_index (same resample as the bars)."""
    idx = target_index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    # resample VIX the same way the instrument bars were (last value in window),
    # then carry the last known value onto each target bar (causal).
    v = vix_close_1min.resample(rule, label="left", closed="left").last().ffill()
    v = v.reindex(idx, method="ffill")

    out = pd.DataFrame(index=target_index)
    out["vix_level"] = v.to_numpy()
    out["vix_chg_1"] = v.diff(1).to_numpy()
    out["vix_chg_5"] = v.diff(5).to_numpy()
    roll_m = v.rolling(50, min_periods=10).mean()
    roll_s = v.rolling(50, min_periods=10).std()
    out["vix_z"] = ((v - roll_m) / (roll_s + 1e-9)).to_numpy()
    # VIX relative to its own recent range (regime): 0 = bottom, 1 = top
    lo = v.rolling(100, min_periods=20).min()
    hi = v.rolling(100, min_periods=20).max()
    out["vix_pctile"] = ((v - lo) / (hi - lo + 1e-9)).to_numpy()
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
