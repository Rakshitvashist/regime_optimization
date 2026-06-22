"""
vol_ml_rich.py  —  does a RICH cross-asset + macro feature set let ML beat HAR?

Earlier we found ML loses to HAR when fed features redundant with the instrument's
own vol (HAR/CJ/semivar/skew). This adds genuinely NON-REDUNDANT information —
cross-asset (crude & gold realized vol + returns), the macro risk-on/off regime,
VIX, and rolling cross-asset correlations — then tests, walk-forward:

  HAR        baseline (log RV d/w/m)
  +macro OLS HAR + macro/cross-asset regressors (HAR-X, linear)
  GBM-rich   gradient boosting on the full rich set
  HYBRID     HAR base + GBM on residual using the rich set

Honest: only a model that beats HAR R2 by >0.005 walk-forward is worth keeping.

Usage:  python vol_ml_rich.py --inputs NIFTY_50.csv BANKNIFTY.csv --H 20 --vix INDIA_VIX.csv
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

import daily_rv_forecast as drv
import vol_ml_hybrid as vmh


def _dk(s):
    s = s.copy(); s.index = [pd.Timestamp(d).date() for d in s.index]; return s


def _logrv(path):
    return _dk(np.log(vmh.daily_measures(drv._load_1min(path))["rv"]))


def _dret(path):
    df = drv._load_1min(path)
    cl = df["Close"].astype(float).groupby(drv._day_index(df.index)).last()
    return _dk(np.log(cl).diff())


def build(target, H, vix_path):
    M = vmh.daily_measures(drv._load_1min(target)); M.index = [pd.Timestamp(d).date() for d in M.index]
    rv, bv = M["rv"], M["bv"]; L = lambda s: np.log(s.clip(lower=1e-12))
    F = pd.DataFrame(index=M.index)
    F["rv_d"] = L(rv); F["rv_w"] = L(rv.rolling(5, min_periods=3).mean()); F["rv_m"] = L(rv.rolling(22, min_periods=10).mean())
    F["c_d"] = L(bv); F["j_d"] = L((rv - bv).clip(lower=0) + 1e-10)
    F["sv_dn"] = L(M["rsn"])
    # --- cross-asset realized vol (NON-redundant) ---
    cr, go = _logrv("CRUDEOIL.csv"), _logrv("GOLD.csv")
    F["crude_rv"] = cr.reindex(F.index); F["gold_rv"] = go.reindex(F.index)
    F["crude_rv_w"] = cr.rolling(5, min_periods=3).mean().reindex(F.index)
    F["gold_rv_w"] = go.rolling(5, min_periods=3).mean().reindex(F.index)
    # --- cross-asset returns + rolling correlation (risk-on/off) ---
    nr, crr, gor = _dret(target), _dret("CRUDEOIL.csv"), _dret("GOLD.csv")
    F["crude_ret5"] = crr.rolling(5).sum().reindex(F.index)
    F["gold_ret5"] = gor.rolling(5).sum().reindex(F.index)
    al = pd.DataFrame({"n": nr, "c": crr, "g": gor}).dropna()
    F["corr_nc"] = al["n"].rolling(30).corr(al["c"]).reindex(F.index)
    F["corr_ng"] = al["n"].rolling(30).corr(al["g"]).reindex(F.index)
    # --- macro risk regime ---
    try:
        from macro_regime import compute as macro_compute
        mac = macro_compute(); mac.index = [pd.Timestamp(d).date() for d in mac.index]
        F["macro_regime"] = mac["regime"].reindex(F.index)
        F["macro_goldeq"] = mac["gold_minus_eq"].reindex(F.index)
    except Exception:
        pass
    if vix_path:
        from vix_features import load_vix_1min, vix_features
        try:
            v = load_vix_1min(vix_path); vd = v.groupby(drv._day_index(v.index)).last()
            F["vix"] = _dk(np.log(vd.clip(lower=1e-6))).reindex(F.index)
        except Exception:
            pass
    F["target"] = L(rv.rolling(H).mean().shift(-H))
    return F.replace([np.inf, -np.inf], np.nan)


def _gbm():
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_leaf_nodes=15,
        l2_regularization=3.0, min_samples_leaf=40, early_stopping=True,
        validation_fraction=0.15, random_state=42)


def wf(F, cols, H, folds, sf, model, har=None):
    sub = F[cols + ["target"]].dropna()
    X = sub[cols].to_numpy(); y = sub["target"].to_numpy()
    Xh = sub[har].to_numpy() if har else None
    n = len(X); st = int(n * sf); step = max(1, (n - st) // folds); oy, op = [], []
    for f in range(folds):
        lo = st + f * step; hi = (st + (f + 1) * step) if f < folds - 1 else n
        if lo - H < 120:
            continue
        tr = slice(0, lo - H); te = slice(lo, hi)
        if model == "ols":
            p = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
        elif model == "gbm":
            p = _gbm().fit(X[tr], y[tr]).predict(X[te])
        else:
            h = LinearRegression().fit(Xh[tr], y[tr]); g = _gbm().fit(X[tr], y[tr] - h.predict(Xh[tr]))
            p = h.predict(Xh[te]) + g.predict(X[te])
        oy.append(y[te]); op.append(p)
    yt = np.concatenate(oy); yp = np.concatenate(op)
    return r2_score(yt, yp), np.sqrt(mean_squared_error(drv.annvol(yt), drv.annvol(yp)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--H", type=int, default=20); ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--start-frac", type=float, default=0.4); ap.add_argument("--vix", default=None)
    a = ap.parse_args(argv)
    HAR = ["rv_d", "rv_w", "rv_m"]
    print(f"{'instrument':12s} {'model':12s} {'R2':>7s} {'RMSE%':>7s} {'vs HAR':>8s}")
    for p in a.inputs:
        nm = p.replace(".csv", ""); F = build(p, a.H, a.vix)
        rich = [c for c in F.columns if c != "target"]
        macro = HAR + [c for c in F.columns if c.startswith(("crude", "gold", "macro", "corr", "vix")) and c in F]
        base = None
        for name, cols, mdl, hc in [("HAR", HAR, "ols", None), ("HAR-X(macro)", macro, "ols", None),
                                    ("GBM-rich", rich, "gbm", None), ("HYBRID-rich", rich, "hybrid", HAR)]:
            try:
                r2, rmse = wf(F, cols, a.H, a.folds, a.start_frac, mdl, hc)
                if name == "HAR":
                    base = r2; tag = "—"
                else:
                    tag = f"{r2-base:+.3f}" + (" WIN" if r2 > base + 0.005 else "")
                print(f"{nm:12s} {name:12s} {r2:7.3f} {rmse:7.2f} {tag:>8s}")
            except Exception as e:
                print(f"{nm:12s} {name:12s} ERR {type(e).__name__}: {str(e)[:40]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
