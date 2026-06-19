"""Empirical: is there a price-volume signature BEFORE a price spike?
Causal features measured at bar t; spike labelled from the FORWARD window only."""
import sys, os, argparse
import numpy as np, pandas as pd

def atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def analyze(path, H, k, L):
    df = pd.read_csv(path)
    df.columns = [x.lower() for x in df.columns]
    C, Hh, Ll, V = df["close"], df["high"], df["low"], df["volume"].astype(float)
    a = atr(Hh, Ll, C)
    ret = C.pct_change()
    # --- causal price-volume features measured AT bar t (use only past/current) ---
    feats = {}
    feats["vol_z"]        = (V - V.rolling(20, min_periods=1).mean()) / (V.rolling(20, min_periods=1).std() + 1e-9)
    feats["vol_ratio"]    = V.rolling(L, min_periods=1).mean() / (V.rolling(4*L, min_periods=1).mean() + 1e-9)   # recent vs longer volume
    rng = (Hh - Ll)
    feats["range_comp"]   = rng.rolling(L, min_periods=1).mean() / (rng.rolling(4*L, min_periods=1).mean() + 1e-9)  # <1 = compressed/coiled
    obv = (np.sign(C.diff().fillna(0)) * V).cumsum()
    feats["obv_slope"]    = (obv - obv.shift(L)) / (V.rolling(L, min_periods=1).mean()*L + 1e-9)                  # accumulation drift
    feats["absorption"]   = (V.rolling(L).corr(ret.abs()))                                                       # do moves need volume? low = absorption
    feats["vol_trend"]    = (V.rolling(L, min_periods=1).mean() - V.rolling(L, min_periods=1).mean().shift(L)) / (V.rolling(4*L, min_periods=1).mean() + 1e-9)

    fwd = (C.shift(-H) - C) / (a + 1e-9)          # forward move in ATRs
    up   = (fwd > k)
    dn   = (fwd < -k)
    base_up, base_dn = up.mean(), dn.mean()
    print(f"\n=== {os.path.basename(path)} | H={H} bars, spike=|move|>{k}*ATR | leadup L={L} ===")
    print(f"base rate: up-spike {base_up:.3f} | down-spike {base_dn:.3f}  (n={len(C)})")
    print(f"{'feature':12s} {'mean@up':>9s} {'mean@dn':>9s} {'mean@norm':>10s} "
          f"{'P(up|hi)':>9s} {'lift':>7s} {'P(up|lo)':>9s}")
    norm = ~(up | dn)
    for name, f in feats.items():
        f = f.replace([np.inf, -np.inf], np.nan)
        m = f.notna() & fwd.notna()
        fu, fd, fn = f[up & m].mean(), f[dn & m].mean(), f[norm & m].mean()
        hi = f > f[m].quantile(0.75); lo = f < f[m].quantile(0.25)
        p_hi = up[hi & m].mean(); p_lo = up[lo & m].mean()
        print(f"{name:12s} {fu:9.3f} {fd:9.3f} {fn:10.3f} "
              f"{p_hi:9.3f} {p_hi-base_up:+7.3f} {p_lo:9.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--H", type=int, default=12)
    p.add_argument("--k", type=float, default=1.5)
    p.add_argument("--L", type=int, default=12)
    a = p.parse_args()
    analyze(a.path, a.H, a.k, a.L)
