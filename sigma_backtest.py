"""
sigma_backtest.py  —  calibration backtest of the sigma price bands.

For each historical day t (out-of-sample, walk-forward), we forecast the H-day
volatility with HAR, build the band  P_t * exp(+/- k*sigma_H), and check whether
the ACTUAL price H days later, P_{t+H}, landed inside it. The coverage = how
often the realized price lay within each band.

A well-calibrated forecast (with ~normal returns) gives:
    1.0 sigma  ~ 68.3%
    1.5 sigma  ~ 86.6%
    2.0 sigma  ~ 95.4%
Higher observed = bands too wide (vol over-forecast); lower = too narrow / fat tails.

Reported for H=20 and H=7 (or any --horizons), per instrument.

Usage:
  python sigma_backtest.py --inputs NIFTY_50.csv BANKNIFTY.csv GOLD.csv CRUDEOIL.csv
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

import daily_rv_forecast as drv
from vol_cone import daily_total_var, har_table

KS = [1.0, 1.5, 2.0]
EXPECT = {1.0: 0.683, 1.5: 0.866, 2.0: 0.954}


def coverage(df, H, folds, start_frac, overnight=True):
    rv = daily_total_var(df, overnight=overnight)
    F, target = har_table(rv, H)
    cl = df["Close"].astype(float).groupby(drv._day_index(df.index)).last()
    lc = np.log(cl)
    rH = (lc.shift(-H) - lc).reindex(rv.index)          # realized forward H-day return
    both = F.notna().all(axis=1) & target.notna()
    Xb = F[both].to_numpy(); yb = target[both].to_numpy(); rHb = rH[both].to_numpy()
    n = len(Xb); start = int(n * start_frac); step = max(1, (n - start) // folds)
    hits = {k: [] for k in KS}; below = {k: [] for k in KS}
    for f in range(folds):
        lo = start + f * step
        hi = (start + (f + 1) * step) if f < folds - 1 else n
        if lo - H < 100:
            continue
        m = LinearRegression().fit(Xb[:lo - H], yb[:lo - H])
        sigH = np.sqrt(np.exp(m.predict(Xb[lo:hi])) * H)
        r = rHb[lo:hi]; v = np.isfinite(r)
        for k in KS:
            hits[k].append(np.abs(r[v]) < k * sigH[v])
    cov = {k: float(np.mean(np.concatenate(hits[k]))) for k in KS}
    n_obs = int(sum(len(h) for h in hits[1.0]))
    return cov, n_obs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sigma-band coverage backtest.")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=[20, 7])
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--start-frac", type=float, default=0.4)
    a = ap.parse_args(argv)
    print(f"Expected coverage:  1s ~ {EXPECT[1.0]:.1%}   1.5s ~ {EXPECT[1.5]:.1%}   2s ~ {EXPECT[2.0]:.1%}\n")
    print(f"{'instrument':12s} {'H':>3s} {'n':>6s} | {'1s':>16s} {'1.5s':>16s} {'2s':>16s}")
    for path in a.inputs:
        name = path.replace(".csv", "")
        df = drv._load_1min(path)
        for H in a.horizons:
            cov, n = coverage(df, H, a.folds, a.start_frac)
            def cell(k):
                d = cov[k] - EXPECT[k]
                return f"{cov[k]:6.1%} ({d:+.1%})"
            print(f"{name:12s} {H:3d} {n:6d} | {cell(1.0):>16s} {cell(1.5):>16s} {cell(2.0):>16s}")
    print("\n(% = realized price landed inside the band; (+/-) = vs theoretical. "
          "near 0 = well-calibrated; + = bands too wide; - = too narrow / fat tails.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
