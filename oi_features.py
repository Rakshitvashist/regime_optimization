"""
oi_features.py  —  Open-Interest positioning features & directional signal (CAUSAL).

Futures Open Interest (OI) is the count of outstanding contracts. Combined with the
price change over the same bar it reveals *who* is driving a move — fresh money or
position-closing — which a price-only feature set cannot see:

    price   OI     interpretation              directional read
    -----   ----   -------------------------   ----------------
    up      up     long buildup                strong bullish  (+1.0)
    down    up     short buildup               strong bearish  (-1.0)
    up      down   short covering              weak bullish    (+0.3)  (rally not backed by new longs)
    down    down   long unwinding              weak bearish    (-0.3)  (selloff not backed by new shorts)

So the directional vote is  sign(dPrice) * conviction, where conviction is high when
OI confirms the move (fresh positions) and low when OI falls (positions closing).

DATA NOTE — the project's *_OI.csv files are 3-minute bars covering only ~1 month
(2026-05 onward). That is far too little to TRAIN the year-spanning walk-forward
models, so OI is used here as a *recent-state* layer: a live direction vote for the
consensus predictor and a positioning snapshot for the market X-ray. Everything is
causal (bar t uses only data <= t): OI is forward-filled onto the price grid with the
last *known* value, and all rolling stats use trailing windows only.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

# Map the consensus/price symbol to the *_OI.csv filename prefix.
OI_PREFIX = {
    "NIFTY_50": "NIFTY",
    "NIFTY": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
}


# --------------------------------------------------------------------------- IO
def _to_ist(idx: pd.Index) -> pd.DatetimeIndex:
    """Parse to a tz-aware IST DatetimeIndex (the *_OI.csv use +0530)."""
    di = pd.DatetimeIndex(pd.to_datetime(idx, utc=True))
    return di.tz_convert("Asia/Kolkata")


def load_oi(path: str) -> pd.Series:
    """Load a single *_OI.csv -> OpenInterest Series indexed by IST datetime."""
    df = pd.read_csv(path, usecols=["DateTime", "OpenInterest"])
    s = pd.Series(df["OpenInterest"].astype(float).values, index=_to_ist(df["DateTime"]))
    return s[~s.index.duplicated(keep="last")].sort_index()


def aggregate_oi(symbol: str, data_dir: str = ".") -> pd.DataFrame:
    """Aggregate OI across all expiry files for `symbol` (e.g. 'NIFTY', 'BANKNIFTY').

    Returns a frame on the union 3-min grid with:
        total_oi    sum of OI across expiries (overall positioning)
        near_oi     OI of the nearest expiry (front month)
        rollover    1 - near_oi/total_oi  (share already rolled to later expiries)
    Each expiry is forward-filled onto the union grid before summing (causal).
    """
    prefix = OI_PREFIX.get(symbol, symbol)
    paths = sorted(glob.glob(os.path.join(data_dir, f"{prefix}*FUT_OI.csv")))
    if not paths:
        raise FileNotFoundError(f"no *_OI.csv for {prefix} in {data_dir}")

    series, expiries = [], []
    for p in paths:
        s = load_oi(p)
        series.append(s)
        # Expiry date parsed from the file's Expiry column (e.g. 30JUN2026).
        exp = pd.read_csv(p, usecols=["Expiry"], nrows=1)["Expiry"].iloc[0]
        expiries.append(pd.to_datetime(exp, format="%d%b%Y"))

    grid = series[0].index
    for s in series[1:]:
        grid = grid.union(s.index)
    cols = [s.reindex(grid).ffill() for s in series]          # causal ffill onto union grid
    mat = pd.concat(cols, axis=1)
    mat.columns = [str(e.date()) for e in expiries]

    order = np.argsort(expiries)                              # nearest expiry first
    near = mat.iloc[:, order[0]]
    total = mat.sum(axis=1)
    out = pd.DataFrame({
        "total_oi": total,
        "near_oi": near,
        "rollover": 1.0 - near / total.replace(0, np.nan),
    }, index=grid)
    for c in mat.columns:                                    # per-expiry OI (prefixed)
        out[f"exp:{c}"] = mat[c]
    return out


# --------------------------------------------------------------- core features
def compute_oi_features(price: pd.Series, oi: pd.Series, z_win: int = 100) -> pd.DataFrame:
    """Causal OI features aligned to `price.index`.

    Args:
        price: Close price Series (any cadence) with a tz-aware/naive DatetimeIndex.
        oi:    OpenInterest Series (e.g. total_oi from aggregate_oi).
        z_win: trailing window (in price bars) for the OI-change z-score.

    Returns a float DataFrame indexed like `price` with:
        oi              forward-filled OI on the price grid
        d_oi            OI change vs previous bar
        d_oi_pct        OI % change
        oi_chg_z        trailing z-score of OI change (conviction / unusual activity)
        d_price         price change vs previous bar
        buildup         categorical {1:long buildup, -1:short buildup,
                                      2:short covering, -2:long unwinding, 0:flat}
        oi_signal       directional vote in [-1, +1]  (see module docstring)
    """
    px = price.astype(float)
    oi_al = _align(oi, px.index)

    d_oi = oi_al.diff()
    d_oi_pct = oi_al.pct_change().replace([np.inf, -np.inf], np.nan)
    mu = d_oi.rolling(z_win, min_periods=max(5, z_win // 5)).mean()
    sd = d_oi.rolling(z_win, min_periods=max(5, z_win // 5)).std()
    oi_chg_z = ((d_oi - mu) / (sd + 1e-9)).clip(-5, 5)

    d_price = px.diff()
    sp = np.sign(d_price).fillna(0.0)
    rising_oi = (d_oi > 0)

    # buildup category
    buildup = pd.Series(0, index=px.index, dtype=int)
    buildup[(sp > 0) & rising_oi] = 1     # long buildup
    buildup[(sp < 0) & rising_oi] = -1    # short buildup
    buildup[(sp > 0) & ~rising_oi] = 2    # short covering
    buildup[(sp < 0) & ~rising_oi] = -2   # long unwinding

    conviction = np.where(rising_oi.to_numpy(), 1.0, 0.3)  # fresh money vs unwinding
    oi_signal = pd.Series(sp.to_numpy() * conviction, index=px.index).fillna(0.0)

    out = pd.DataFrame({
        "oi": oi_al,
        "d_oi": d_oi,
        "d_oi_pct": d_oi_pct,
        "oi_chg_z": oi_chg_z,
        "d_price": d_price,
        "buildup": buildup,
        "oi_signal": oi_signal,
    }, index=px.index)
    return out.astype(float)


def oi_direction_signal(price: pd.Series, oi: pd.Series) -> pd.Series:
    """Directional OI vote in [-1, +1], aligned to `price.index`.

    Drop-in as an extra column for ConsensusPredictor.predict's `indicators_df`
    (name it e.g. 'oi_buildup' so it lands in the 'other' category, or 'oi_trend'
    to weight it with the trend bucket)."""
    return compute_oi_features(price, oi)["oi_signal"]


def oi_state(price: pd.Series, oi: pd.Series, lookback: int = 125) -> dict:
    """Point-in-time OI positioning snapshot for the market X-ray (market_state).

    `lookback` is the number of recent OI bars used for the rising-OI day count
    framing (125 ~= one trading day of 3-min bars)."""
    f = compute_oi_features(price, oi)
    tail = f.dropna(subset=["oi"]).tail(lookback)
    if tail.empty:
        return {}
    last = tail.iloc[-1]
    label = {1: "long buildup", -1: "short buildup",
             2: "short covering", -2: "long unwinding", 0: "flat"}[int(last["buildup"])]
    rising = float((tail["d_oi"] > 0).mean()) * 100.0
    net = float((tail["oi_signal"] > 0).sum() - (tail["oi_signal"] < 0).sum())
    return {
        "oi_now": round(float(last["oi"]), 0),
        "oi_chg_pct_window": round(float(last["oi"] / tail["oi"].iloc[0] - 1.0) * 100, 2),
        "oi_buildup": label,
        "oi_chg_z": round(float(last["oi_chg_z"]), 2),
        "oi_rising_bars_pct": round(rising, 1),
        "oi_net_bias": "bullish" if net > 0 else ("bearish" if net < 0 else "neutral"),
    }


def oi_dashboard_block(price: pd.Series, symbol: str, data_dir: str = ".",
                       spark_bars: int = 130) -> dict | None:
    """Everything the dashboard's Positioning panel needs, or None if no OI files.

    Bundles the oi_state snapshot + rollover + a recent total-OI sparkline +
    a per-expiry OI breakdown (latest bar)."""
    try:
        agg = aggregate_oi(symbol, data_dir)
    except FileNotFoundError:
        return None
    st = oi_state(price, agg["total_oi"])
    if not st:
        return None
    total = agg["total_oi"].dropna()
    tail = total.tail(spark_bars)
    exp_cols = [c for c in agg.columns if c.startswith("exp:")]
    last = agg[exp_cols].iloc[-1]
    tot = float(last.sum()) or 1.0
    expiries = [{"expiry": c[4:], "oi": round(float(last[c]), 0),
                 "share": round(float(last[c]) / tot * 100, 1)} for c in exp_cols]
    return {
        **st,
        "rollover": round(float(agg["rollover"].iloc[-1]) * 100, 1),
        "asof": str(total.index[-1]),
        "expiries": expiries,
        "spark": {"t": [i.strftime("%m-%d %H:%M") for i in tail.index],
                  "oi": [round(float(v), 0) for v in tail.values]},
    }


# --------------------------------------------------------------------- helpers
def _align(oi: pd.Series, target: pd.Index) -> pd.Series:
    """Forward-fill OI onto `target` using only past values (causal reindex)."""
    src = oi.sort_index()
    src = src[~src.index.duplicated(keep="last")]
    # Match tz so reindex aligns; coerce target to match source's tz state.
    if isinstance(target, pd.DatetimeIndex) and isinstance(src.index, pd.DatetimeIndex):
        if src.index.tz is not None and target.tz is None:
            target = target.tz_localize(src.index.tz)
        elif src.index.tz is None and target.tz is not None:
            src.index = src.index.tz_localize(target.tz)
        elif src.index.tz is not None and target.tz is not None:
            src.index = src.index.tz_convert(target.tz)
    return src.reindex(target, method="ffill")


if __name__ == "__main__":
    for sym in ("NIFTY", "BANKNIFTY"):
        agg = aggregate_oi(sym)
        px = pd.Series(  # use the OI grid itself as a stand-in price for the smoke test
            np.arange(len(agg), dtype=float), index=agg.index)
        st = oi_state(px, agg["total_oi"])
        print(f"\n{sym}: {len(agg)} bars, {agg.index[0]} -> {agg.index[-1]}")
        print(f"  total_oi last={agg['total_oi'].iloc[-1]:,.0f}  rollover={agg['rollover'].iloc[-1]:.1%}")
        print(f"  state={st}")
