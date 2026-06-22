"""
market_state.py  —  the "market X-ray": diagnostic statistics (Stage 2-3).

These DESCRIBE the market's current state — they are not alpha, they are context
(the honest use of the advanced-stats layer):

  Hurst exponent     trending (>0.55) / mean-reverting (<0.45) / random walk
  Distribution       skewness, excess kurtosis -> fat-tailed vs near-normal
  Tail index (Hill)  EVT tail heaviness (lower = fatter tail = more crash-prone)
  VaR / CVaR (95%)   daily downside risk
  Drawdown           max and current drawdown

All from daily closes built off the 1-min bars. Pure numpy/scipy, causal as a
point-in-time snapshot (uses the full available history up to now).
"""
from __future__ import annotations

import numpy as np, pandas as pd
from scipy.stats import skew, kurtosis

import daily_rv_forecast as drv


def _hurst(x, max_lag=60):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 40:
        return float("nan")
    lags = range(2, min(max_lag, len(x) // 2))
    tau = [np.sqrt(np.mean((x[lag:] - x[:-lag]) ** 2)) for lag in lags]
    tau = np.array(tau); lg = np.array(list(lags)); m = tau > 0
    return float(np.polyfit(np.log(lg[m]), np.log(tau[m]), 1)[0])


def market_state(df):
    cl = df["Close"].astype(float).groupby(drv._day_index(df.index)).last()
    lc = np.log(cl); ret = lc.diff().dropna()
    r = ret.to_numpy()
    H = _hurst(lc.dropna().to_numpy())
    sk = float(skew(r)); ku = float(kurtosis(r))  # excess kurtosis (0 = normal)
    a = np.sort(np.abs(r))[::-1]; k = max(20, int(0.05 * len(a)))
    hill = float(1.0 / np.mean(np.log(a[:k] / a[k]))) if k < len(a) and a[k] > 0 else float("nan")
    q5 = float(np.percentile(r, 5))
    var95 = q5 * 100; cvar95 = float(r[r <= q5].mean()) * 100
    dd = (cl / cl.cummax() - 1.0)
    trend = "trending" if H > 0.55 else ("mean-revert" if H < 0.45 else "random walk")
    tail = "fat-tailed" if ku > 1.0 else "near-normal"
    return {
        "hurst": round(H, 3), "trend": trend,
        "skew": round(sk, 2), "kurtosis": round(ku, 2), "tail": tail,
        "tail_index": round(hill, 2),
        "var95": round(var95, 2), "cvar95": round(cvar95, 2),
        "max_dd": round(float(dd.min()) * 100, 1),
        "cur_dd": round(float(dd.iloc[-1]) * 100, 1),
    }
