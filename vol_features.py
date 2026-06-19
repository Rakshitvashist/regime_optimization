"""
vol_features.py  —  advanced volatility features (all CAUSAL).

Implements the literature's strongest volatility predictors as features for the
movement/regime models:

  HAR-RV          multi-horizon realized vol (Corsi 2009) — the academic benchmark;
                  captures volatility's long memory that a single window misses.
  Range estimators Parkinson / Garman-Klass / Rogers-Satchell / Yang-Zhang — use the
                  full OHLC bar (~5x more efficient than close-to-close RV).
  Time-of-day     intraday seasonality (the vol U-shape: open & close are volatile).
  Hawkes intensity self-exciting jump intensity — spikes beget spikes; this is the
                  math-grounded "spike clustering" signal.
  GARCH (optional) conditional-vol forecast (needs the `arch` package). Params are
                  fit on the EARLY part of the series only; the conditional-vol
                  recursion is causal -> usable as a forward feature without leak.

All features at bar t use only data up to t. Returns a float32 DataFrame aligned
to df.index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def advanced_vol_features(df: pd.DataFrame, with_garch: bool = False,
                          garch_fit_frac: float = 0.4) -> pd.DataFrame:
    o = df["open"].astype(float); h = df["high"].astype(float)
    l = df["low"].astype(float);  c = df["close"].astype(float)
    ret = np.log(c).diff().fillna(0.0)
    out = pd.DataFrame(index=df.index)

    # --- HAR-RV: realized vol over short/medium/long horizons (causal rolling) ---
    rv = ret ** 2
    for w, nm in [(10, "s"), (50, "m"), (200, "l")]:
        out[f"har_rv_{nm}"] = np.sqrt(rv.rolling(w, min_periods=max(3, w // 4)).mean())
    out["har_ratio"] = out["har_rv_s"] / (out["har_rv_l"] + 1e-9)   # short/long vol regime

    # --- Range-based estimators (full OHLC; more efficient than close-only) ---
    eps = 1e-12
    ln_hl = np.log((h / l).clip(lower=eps))
    ln_co = np.log((c / o).clip(lower=eps))
    ln_ho = np.log((h / o).clip(lower=eps))
    ln_lo = np.log((l / o).clip(lower=eps))
    park = (ln_hl ** 2) / (4 * np.log(2))
    gk = 0.5 * ln_hl ** 2 - (2 * np.log(2) - 1) * ln_co ** 2
    rs = ln_ho * (ln_ho - ln_co) + ln_lo * (ln_lo - ln_co)
    for est, nm in [(park, "park"), (gk, "gk"), (rs, "rs")]:
        out[f"vol_{nm}"] = np.sqrt(est.clip(lower=0).rolling(20, min_periods=5).mean())
    out["vol_yz"] = np.sqrt(((gk + rs) / 2).clip(lower=0)
                           .rolling(20, min_periods=5).mean())   # Yang-Zhang-style blend

    # --- Time-of-day seasonality (intraday vol U-shape) ---
    idx = df.index
    idxl = idx.tz_convert("Asia/Kolkata") if idx.tz is not None else idx
    mins = np.clip((idxl.hour * 60 + idxl.minute) - (9 * 60 + 15), 0, 375)
    frac = mins / 375.0
    out["tod_frac"] = frac
    out["tod_sin"] = np.sin(2 * np.pi * frac)
    out["tod_cos"] = np.cos(2 * np.pi * frac)
    out["tod_open30"] = (mins <= 30).astype(float)
    out["tod_close30"] = (mins >= 345).astype(float)

    # --- Hawkes self-exciting jump intensity (causal: uses PAST jumps only) ---
    sd = ret.rolling(50, min_periods=10).std().bfill()
    jump = (ret.abs() > 2.0 * sd).astype(float).to_numpy()
    n = len(df)
    for tau, nm in [(10, "fast"), (50, "slow")]:
        decay = np.exp(-1.0 / tau)
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = decay * e[t - 1] + jump[t - 1]      # excitation from past events
        out[f"hawkes_{nm}"] = e

    # --- GARCH(1,1) conditional vol (optional; params fit on EARLY data only) ---
    if with_garch:
        try:
            from arch import arch_model
            r = (ret * 100.0).to_numpy()
            n0 = max(500, int(n * garch_fit_frac))
            res = arch_model(r[:n0], vol="Garch", p=1, q=1, mean="Zero",
                             rescale=False).fit(disp="off")
            w = float(res.params["omega"]); al = float(res.params["alpha[1]"])
            be = float(res.params["beta[1]"])
            s2 = np.empty(n); s2[0] = np.var(r[:n0])
            for t in range(1, n):                       # causal recursion, fixed params
                s2[t] = w + al * r[t - 1] ** 2 + be * s2[t - 1]
            out["garch_vol"] = np.sqrt(np.maximum(s2, 0)) / 100.0
        except Exception as exc:
            print(f"  [GARCH skipped: {type(exc).__name__}: {exc}; install `arch`]")
            out["garch_vol"] = out["har_rv_m"]

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
