"""
vol_ml_hybrid.py  —  can ML push HAR realized-vol forecasting higher? (honest test)

We already found ML *replacing* HAR loses (GBM R² 0.10 vs HAR 0.43). So this AUGMENTS
HAR with information it lacks, and only the walk-forward decides if it helps:

  HAR        log RV daily/weekly/monthly (OLS) — the benchmark to beat
  HAR-CJ     + continuous (bipower) and JUMP components (HAR-CJ, Andersen et al.)
  HAR-CJ-SV  + realized SEMIvariance (up/down) — downside vol is more persistent
  GBM        gradient boosting on ALL features (the "ML replaces HAR" baseline)
  HYBRID     HAR base + GBM on the RESIDUAL using extras (VIX, skew/kurt, vol-of-vol)
             -> the honest "ML on top of HAR"

All causal/walk-forward; realized measures (RV, bipower, semivar) from 1-min bars.
Reports R² / RMSE(vol%) / QLIKE per model so you see exactly what (if anything) wins.

Usage:
  python vol_ml_hybrid.py --inputs NIFTY_50.csv BANKNIFTY.csv GOLD.csv --H 20 --vix INDIA_VIX.csv
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

import daily_rv_forecast as drv


def daily_measures(df):
    """Per-day RV, bipower (BV), realized up/down semivariance from 1-min returns."""
    lc = np.log(df["Close"].astype(float))
    day = drv._day_index(df.index)
    g = pd.DataFrame({"lc": lc.to_numpy(), "day": day})
    g["r"] = g.groupby("day")["lc"].diff()
    rows = {}
    for dy, sub in g.groupby("day"):
        r = sub["r"].dropna().to_numpy()
        if len(r) < 3:
            continue
        rv = float(np.sum(r ** 2))
        bv = float((np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
        rows[dy] = (rv, min(bv, rv), float(np.sum(np.clip(r, 0, None) ** 2)),
                    float(np.sum(np.clip(r, None, 0) ** 2)))
    M = pd.DataFrame(rows, index=["rv", "bv", "rsp", "rsn"]).T
    return M[M["rv"] > 0]


def build(df, H, vix_path):
    M = daily_measures(df)
    rv, bv = M["rv"], M["bv"]
    C = bv; J = (rv - bv).clip(lower=0)
    L = lambda s: np.log(s.clip(lower=1e-12))
    F = pd.DataFrame(index=M.index)
    # HAR
    F["rv_d"] = L(rv); F["rv_w"] = L(rv.rolling(5, min_periods=3).mean()); F["rv_m"] = L(rv.rolling(22, min_periods=10).mean())
    # HAR-CJ
    F["c_d"] = L(C); F["c_w"] = L(C.rolling(5, min_periods=3).mean()); F["c_m"] = L(C.rolling(22, min_periods=10).mean())
    F["j_d"] = L(J + 1e-10); F["j_w"] = L(J.rolling(5, min_periods=3).mean() + 1e-10)
    # semivariance + extras
    F["sv_up"] = L(M["rsp"]); F["sv_dn"] = L(M["rsn"])
    F["volofvol"] = L(rv).rolling(22, min_periods=10).std()
    dret = drv.daily_close_returns(df).reindex(M.index)
    F["skew20"] = dret.rolling(20, min_periods=10).skew()
    F["kurt20"] = dret.rolling(20, min_periods=10).kurt()
    if vix_path:
        from vix_features import load_vix_1min, vix_features
        try:
            vf = vix_features(M.index.map(lambda d: pd.Timestamp(d)), load_vix_1min(vix_path), "1D")
            F["vix"] = np.log(np.clip(vf["vix_level"].to_numpy(), 1e-6, None))
        except Exception:
            pass
    F["target"] = L(rv.rolling(H).mean().shift(-H))
    return F.replace([np.inf, -np.inf], np.nan)


def _gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=2.0, early_stopping=True, validation_fraction=0.15, random_state=42)


def wf(F, cols, H, folds, start_frac, model="ols", har_cols=None):
    both = F[cols + ["target"]].notna().all(axis=1)
    X = F.loc[both, cols].to_numpy(); y = F.loc[both, "target"].to_numpy()
    Xh = F.loc[both, har_cols].to_numpy() if har_cols else None
    n = len(X); start = int(n * start_frac); step = max(1, (n - start) // folds)
    oy, op = [], []
    for f in range(folds):
        lo = start + f * step; hi = (start + (f + 1) * step) if f < folds - 1 else n
        if lo - H < 120:
            continue
        tr = slice(0, lo - H); te = slice(lo, hi)
        if model == "ols":
            m = LinearRegression().fit(X[tr], y[tr]); p = m.predict(X[te])
        elif model == "gbm":
            m = _gbm().fit(X[tr], y[tr]); p = m.predict(X[te])
        else:  # hybrid: HAR base + GBM on residual
            h = LinearRegression().fit(Xh[tr], y[tr])
            res = y[tr] - h.predict(Xh[tr])
            g = _gbm().fit(X[tr], res)
            p = h.predict(Xh[te]) + g.predict(X[te])
        oy.append(y[te]); op.append(p)
    yt = np.concatenate(oy); yp = np.concatenate(op)
    rmse = np.sqrt(mean_squared_error(drv.annvol(yt), drv.annvol(yp)))
    return r2_score(yt, yp), rmse, drv.qlike(np.exp(yt), np.exp(yp))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--start-frac", type=float, default=0.4)
    ap.add_argument("--vix", default=None)
    a = ap.parse_args(argv)
    HAR = ["rv_d", "rv_w", "rv_m"]
    CJ = ["c_d", "c_w", "c_m", "j_d", "j_w"]
    CJSV = CJ + ["sv_up", "sv_dn"]
    ALL = lambda F: [c for c in F.columns if c != "target"]
    print(f"{'instrument':12s} {'model':10s} {'R2':>7s} {'RMSE%':>7s} {'QLIKE':>7s}")
    for path in a.inputs:
        nm = path.replace(".csv", ""); F = build(df := drv._load_1min(path), a.H, a.vix)
        runs = [("HAR", HAR, "ols", None), ("HAR-CJ", CJ, "ols", None),
                ("HAR-CJ-SV", CJSV, "ols", None), ("GBM-all", ALL(F), "gbm", None),
                ("HYBRID", ALL(F), "hybrid", HAR)]
        best = None
        for name, cols, mdl, hc in runs:
            try:
                r2, rmse, ql = wf(F, cols, a.H, a.folds, a.start_frac, mdl, hc)
                mark = ""
                if name == "HAR":
                    har_r2 = r2
                elif r2 > har_r2 + 0.005:
                    mark = "  <-- beats HAR"
                print(f"{nm:12s} {name:10s} {r2:7.3f} {rmse:7.2f} {ql:7.3f}{mark}")
            except Exception as e:
                print(f"{nm:12s} {name:10s} ERR {type(e).__name__}: {e}")
        print()
    print("(Only a model that beats HAR R2 by >0.005 walk-forward is worth keeping.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
