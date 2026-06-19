"""
optimization/volume_profile.py

Causal (no look-ahead) VOLUME-PROFILE features — where volume actually traded by
price level, over a rolling window. This is the "volume profile pattern" view:

  POC  (Point of Control)  : price bin with the most traded volume — a magnet /
                             fair-value level price keeps returning to.
  Value Area (VAH..VAL)    : the price band holding ~70% of the window's volume.
  Concentration            : how peaked the profile is at the POC.

Trading patterns this exposes (the model / hit-ratio screen can use them):
  - Price ABOVE the POC  -> bullish acceptance;  BELOW -> bearish.
  - Price breaking ABOVE VAH / BELOW VAL -> range breakout (trend ignition).
  - Narrow value area      -> balance / coil  -> often PRECEDES a rally.
  - POC migrating up/down   -> accumulation / distribution drift (trend).

Look-ahead safety (Principle II): the profile at bar t is built only from bars
[t-window+1 .. t]. Nothing reads the future.

Features (see FEATURE_NAMES):
  vp_dist_poc          signed normalized distance of close from POC   (~[-2,2])
  vp_va_pos            +1 above value area, -1 below, 0 inside
  vp_poc_trend         signed POC drift over the window (accumulation/distribution)
  vp_value_width       value-area width vs price: <0 compressed (coiled), >0 wide
  vp_poc_concentration volume share at the POC bin, centered to [-1,1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "vp_dist_poc", "vp_va_pos", "vp_poc_trend",
    "vp_value_width", "vp_poc_concentration",
]
N_FEATURES = len(FEATURE_NAMES)


def _profile_one(prices: np.ndarray, vols: np.ndarray, n_bins: int,
                 value_area: float):
    """Return (poc, vah, val, concentration) for one window."""
    lo, hi = float(prices.min()), float(prices.max())
    if hi <= lo:
        last = float(prices[-1])
        return last, last, last, 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(prices, edges) - 1, 0, n_bins - 1)
    vol_by_bin = np.bincount(idx, weights=vols, minlength=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    total = vol_by_bin.sum() + 1e-9
    poc_bin = int(vol_by_bin.argmax())
    conc = vol_by_bin[poc_bin] / total

    # Grow the value area outward from the POC until it holds `value_area` volume.
    lo_b = hi_b = poc_bin
    acc = vol_by_bin[poc_bin]
    while acc < value_area * total and (lo_b > 0 or hi_b < n_bins - 1):
        left = vol_by_bin[lo_b - 1] if lo_b > 0 else -1.0
        right = vol_by_bin[hi_b + 1] if hi_b < n_bins - 1 else -1.0
        if right >= left:
            hi_b += 1
            acc += vol_by_bin[hi_b]
        else:
            lo_b -= 1
            acc += vol_by_bin[lo_b]
    return centers[poc_bin], centers[hi_b], centers[lo_b], conc


def profile_features(df: pd.DataFrame, window: int = 120, n_bins: int = 24,
                     value_area: float = 0.70) -> np.ndarray:
    """Build the (T, N_FEATURES) causal volume-profile feature block."""
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    vol = (df["volume"].to_numpy(float) if "volume" in df.columns
           else np.ones(len(df)))
    # Volume-less series (indices) -> fall back to a TIME profile (Market Profile /
    # TPO): every bar contributes one unit of time, so POC = price where the most
    # TIME was spent. This is the original volume-less Market Profile concept.
    if not np.any(np.nan_to_num(vol)):
        vol = np.ones(len(df))
    tp = (high + low + close) / 3.0          # typical price per bar
    T = len(close)
    eps = 1e-9

    poc = np.full(T, np.nan)
    vah = np.full(T, np.nan)
    val = np.full(T, np.nan)
    conc = np.zeros(T)
    for t in range(T):
        s = max(0, t - window + 1)
        p, vh, vl, cc = _profile_one(tp[s:t + 1], vol[s:t + 1], n_bins, value_area)
        poc[t], vah[t], val[t], conc[t] = p, vh, vl, cc

    dist_poc = np.clip((close - poc) / (close + eps), -0.2, 0.2) / 0.1   # ~[-2,2]
    va_pos = np.where(close > vah, 1.0, np.where(close < val, -1.0, 0.0))

    k = max(1, window // 4)
    poc_trend = np.zeros(T)
    dpoc = (poc[k:] - poc[:-k]) / (close[k:] + eps)
    poc_trend[k:] = np.sign(dpoc) * np.minimum(np.abs(dpoc) / 0.05, 2.0)

    value_width = np.clip((vah - val) / (close + eps), 0.0, 0.3) / 0.1 - 1.0
    conc_f = np.clip(conc, 0.0, 1.0) * 2.0 - 1.0

    feats = np.stack([dist_poc, va_pos, poc_trend, value_width, conc_f],
                     axis=1).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0)
