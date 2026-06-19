"""
optimization/liquidity_features.py

Causal (no look-ahead) liquidity-sweep + price/volume features for *trend
anticipation* — designed to fire BEFORE the move, which is the behaviour the
user asked for ("detect trend before it occurs").

Faithful NumPy port of the Pine v5 "Liquidity Sweep - Purple Box":

    A  = anchor candle, reset every `lookback` bars; its high/low are the
         liquidity-pool edges (resting stops sit just beyond them).
    B  = first candle AFTER A that breaks A's high (up) or low (down) — the
         stop-run / sweep that grabs that liquidity.
    C  = confirmation: price then breaks A in the OPPOSITE direction:
            B swept the LOW  -> C reclaims the HIGH  => bullish setup (+1)
            B swept the HIGH -> C breaks   the LOW   => bearish setup (-1)

The C event typically precedes the directional leg (stop-hunt then reversal),
so it is a *leading* signal rather than a lagging confirmation.

Look-ahead safety (Principle II, NON-NEGOTIABLE): every value at bar t is built
from bars <= t only — the state machine walks forward in time and never reads a
future bar. Safe to use as an LSTM input feature alongside forward-return targets.

Price/volume relationship: a sweep on heavy volume is a genuine liquidity grab;
the same pattern on thin volume is noise. The features below let the model gate
the sweep by participation (volume surge, signed money-flow, sweep×volume).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Order of the appended feature columns (kept stable for interpretability).
FEATURE_NAMES = [
    "liq_sweep_event",   # +1 bullish C-confirm, -1 bearish C-confirm, else 0
    "liq_pool_pos",      # signed proximity of close to the A-pool edges in [-1,1]
    "vol_surge",         # volume vs rolling mean (z-like), clipped
    "sweep_x_volume",    # sweep_event * vol_surge  (a sweep WITH participation)
    "signed_vol_mom",    # OBV-style signed-volume momentum, normalized
]
N_FEATURES = len(FEATURE_NAMES)


def liquidity_sweep_state(high: np.ndarray, low: np.ndarray,
                          lookback: int = 10):
    """
    Walk the A/B/C state machine forward in time.

    Returns
    -------
    sweep_event : (T,) float32  -> +1 bullish C, -1 bearish C, 0 otherwise
    pool_high   : (T,) float32  -> current A high (the upside liquidity edge)
    pool_low    : (T,) float32  -> current A low  (the downside liquidity edge)
    """
    T = len(high)
    sweep_event = np.zeros(T, dtype=np.float32)
    pool_high = np.full(T, np.nan, dtype=np.float32)
    pool_low = np.full(T, np.nan, dtype=np.float32)

    a_high = a_low = np.nan
    a_bar = -1
    b_up = b_down = b_done = False

    for i in range(T):
        # STEP 1 — reset anchor A every `lookback` bars (matches bar_index % N == 0)
        if i % lookback == 0:
            a_high, a_low, a_bar = high[i], low[i], i
            b_up = b_down = b_done = False

        # STEP 2 — candle B: first bar after A that breaks A's high or low
        if not np.isnan(a_high) and not b_done and i > a_bar:
            if high[i] > a_high and not b_up and not b_down:
                b_up, b_done = True, True
            elif low[i] < a_low and not b_up and not b_down:
                b_down, b_done = True, True

        # STEP 3 — candle C: opposite break confirms the sweep
        if b_done and b_down and high[i] >= a_high:        # B down -> C up
            sweep_event[i] = 1.0
            b_up = b_down = b_done = False
        elif b_done and b_up and low[i] <= a_low:          # B up -> C down
            sweep_event[i] = -1.0
            b_up = b_down = b_done = False

        pool_high[i] = a_high
        pool_low[i] = a_low

    return sweep_event, pool_high, pool_low


def extra_features(df: pd.DataFrame, lookback: int = 10) -> np.ndarray:
    """
    Build the (T, N_FEATURES) causal liquidity + price/volume feature block.

    `df` must have lowercase columns: high, low, close, volume.
    Output is float32, NaN-free, and roughly scaled to ~[-3, 3] (the caller
    clips to [-5, 5] like the other OHLCV features).
    """
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    volume = (df["volume"].to_numpy(dtype=np.float64)
              if "volume" in df.columns else np.ones(len(df)))
    T = len(close)
    eps = 1e-8

    sweep_event, pool_high, pool_low = liquidity_sweep_state(high, low, lookback)

    # Signed proximity to the pool edges: +1 hugging the upside pool (A-high),
    # -1 hugging the downside pool (A-low). Tells the model a sweep may be near.
    pool_mid = (pool_high + pool_low) / 2.0
    pool_rng = (pool_high - pool_low) + eps
    pool_pos = np.clip((close - pool_mid) / (pool_rng / 2.0), -2.0, 2.0)
    pool_pos = np.nan_to_num(pool_pos, nan=0.0).astype(np.float32)

    # Volume surge vs its own rolling mean over the lookback window (z-like).
    v = pd.Series(volume)
    v_mean = v.rolling(lookback, min_periods=1).mean()
    v_std = v.rolling(lookback, min_periods=1).std().fillna(0.0)
    vol_surge = ((v - v_mean) / (v_std + eps)).to_numpy()
    vol_surge = np.clip(np.nan_to_num(vol_surge, nan=0.0), -3.0, 3.0).astype(np.float32)

    # A sweep that happens WITH heavy participation (real liquidity grab).
    sweep_x_volume = (sweep_event * np.clip(vol_surge, 0.0, 3.0)).astype(np.float32)

    # OBV-style signed-volume momentum, normalized by rolling |signed-volume|.
    ret = np.zeros(T)
    ret[1:] = np.sign(close[1:] - close[:-1])
    signed_vol = ret * volume
    sv = pd.Series(signed_vol)
    sv_mom = sv.rolling(lookback, min_periods=1).sum()
    sv_scale = sv.abs().rolling(lookback, min_periods=1).sum() + eps
    signed_vol_mom = np.clip((sv_mom / sv_scale).to_numpy(), -1.0, 1.0)
    signed_vol_mom = np.nan_to_num(signed_vol_mom, nan=0.0).astype(np.float32)

    feats = np.stack(
        [sweep_event, pool_pos, vol_surge, sweep_x_volume, signed_vol_mom],
        axis=1,
    ).astype(np.float32)
    return feats  # (T, N_FEATURES)
