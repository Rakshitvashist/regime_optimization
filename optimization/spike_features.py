"""
optimization/spike_features.py

Causal (no look-ahead) "SPIKE-IMMINENT" price-volume features.

Empirically (see prespike_pv.py on RELIANCE 5m, ~173k bars), a price spike is
preceded by VOLUME EXPANSION — current volume well above average and rising into
the move (+7.6% probability tilt). Volume says a spike is *coming*; OBV direction
hints *which way*. The effect is intraday-only (it washes out on daily bars), so
these features matter on volume-bearing series (stocks), and auto-vanish on the
volume-less indices (constant -> dropped by the dead-column filter).

Features (see FEATURE_NAMES):
  spike_volz        volume z-score now (the leading "spike coming" tell)
  spike_voltrend    is volume rising over the lead-up window
  spike_obv_dir     OBV drift = accumulation(+)/distribution(-) -> spike direction
  spike_compression range coil vs longer range: <0 compressed (pre-burst)
  spike_absorption  move-per-volume; low = price drifting on light volume
  spike_score       combined: volz gated by OBV direction (signed spike-imminence)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "spike_volz", "spike_voltrend", "spike_obv_dir",
    "spike_compression", "spike_absorption", "spike_score",
]
N_FEATURES = len(FEATURE_NAMES)


def spike_features(df: pd.DataFrame, lookback: int = 12) -> np.ndarray:
    """Build the (T, N_FEATURES) causal spike-imminent feature block."""
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = (df["volume"].astype(float) if "volume" in df.columns
         else pd.Series(np.ones(len(df)), index=df.index))
    L = max(2, int(lookback))
    eps = 1e-9

    v_mean20 = v.rolling(20, min_periods=1).mean()
    v_std20 = v.rolling(20, min_periods=1).std().fillna(0.0)
    volz = ((v - v_mean20) / (v_std20 + eps)).clip(-4, 4)

    vL = v.rolling(L, min_periods=1).mean()
    v4L = v.rolling(4 * L, min_periods=1).mean()
    voltrend = ((vL - vL.shift(L)) / (v4L + eps)).clip(-3, 3)

    ret = c.pct_change().fillna(0.0)
    obv = (np.sign(ret) * v).cumsum()
    obv_dir = np.tanh((obv - obv.shift(L)) / (vL * L + eps))

    rng = (h - l)
    compression = (rng.rolling(L, min_periods=1).mean()
                   / (rng.rolling(4 * L, min_periods=1).mean() + eps) - 1.0).clip(-1, 2)

    absorption = v.rolling(L).corr(ret.abs()).fillna(0.0).clip(-1, 1)

    # Signed spike-imminence: how strong the "volume says move" tilt is, signed by
    # OBV direction. High |value| = spike likely; sign = expected direction.
    score = (volz.clip(0, 4) / 4.0) * obv_dir

    feats = np.stack([
        volz.to_numpy(), voltrend.to_numpy(), obv_dir.to_numpy(float),
        compression.to_numpy(), absorption.to_numpy(), score.to_numpy(float),
    ], axis=1).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0)
