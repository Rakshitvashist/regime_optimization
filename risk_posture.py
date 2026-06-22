"""
risk_posture.py  —  multi-factor RISK scorecard (the "make a decision" tool).

Blends the weak-but-real signals into ONE number that answers "how much risk is
ahead, right now?" — NOT direction (that's dead). Transparent by design:

  factors (causal)  : vol level, vol rising, vol-of-vol (+accel), recent move/return,
                      Hurst trend, regime, time-in-regime, BOCPD change-point, macro
  model             : logistic regression on standardized factors -> interpretable
                      weights ("a classic factor scorecard"); HistGBM as the nonlinear
                      upper bound; current-vol-alone as the persistence baseline
  target            : forward-H realized vol lands in the high (top-30%) bucket
  honesty           : walk-forward AUC, and the LIFT over persistence — if the
                      multi-factor blend doesn't beat "vol is already high", we show it.

Output: 0-100 posture score, traffic-light level, the factors driving it now, the
factor weights, and the backtested accuracy.

Usage:  python risk_posture.py --inputs NIFTY_50.csv BANKNIFTY.csv --H 20
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import daily_rv_forecast as drv
from vol_cone import daily_total_var
from regime_switch import _base_features

PRETTY = {"rv": "vol level", "rv_chg5": "vol rising", "vov": "vol-of-vol",
          "vov_accel": "vol-of-vol accel", "ret5": "recent return", "absret5": "recent move size",
          "hurst": "trendiness", "regime_now": "regime", "duration": "time in regime",
          "bocpd": "change-point alarm", "macro": "macro risk-off"}
HI_Q = 0.70   # "high-vol ahead" = forward vol in the top 30%


def _features(path, H=20):
    F, reg, idx = _base_features(path)          # 10 causal factors, date-indexed
    try:
        from macro_regime import compute as mc
        m = mc(); m.index = [pd.Timestamp(d).date() for d in m.index]
        F["macro"] = m["regime"].reindex(idx).to_numpy()
    except Exception:
        F["macro"] = np.nan
    rv = daily_total_var(drv._load_1min(path), overnight=True)
    rvv = rv.to_numpy(float)
    n = len(rvv); c = np.concatenate([[0.0], np.nancumsum(rvv)])
    fwd = np.full(n, np.nan)                     # mean realized var over the NEXT H days
    for t in range(n - H):
        fwd[t] = (c[t + 1 + H] - c[t + 1]) / H
    return F, idx, fwd


def _level(s):
    if s < 33:
        return "calm", "Normal risk — size and stops as usual."
    if s < 55:
        return "caution", "Slightly elevated — trim oversized positions."
    if s < 75:
        return "elevated", "Elevated — reduce size, widen stops, expect bigger swings."
    return "high", "High risk — defensive: cut size, hedge, expect a vol spike."


def _fit_pack(X, m, y):
    """Median-impute (train) + standardize (train) + fit logistic. Returns (lr, mu, sd, med)."""
    med = np.nanmedian(X[m], axis=0); med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isnan(X), med, X)
    mu = Xi[m].mean(0); sd = Xi[m].std(0) + 1e-9
    Z = (Xi - mu) / sd
    lr = LogisticRegression(max_iter=600, C=1.0).fit(Z[m], y[m].astype(int))
    return lr, mu, sd, med, Z


def _wf(X, fwd, vcol, H, folds, sf):
    n = len(X); st = int(n * sf); step = max(1, (n - st) // folds)
    yL, pL, pG, pB = [], [], [], []
    for f in range(folds):
        lo = st + f * step; hi = (st + (f + 1) * step) if f < folds - 1 else n
        if lo - H < 150:
            continue
        trm = np.zeros(n, bool); trm[:lo - H] = True; trm &= np.isfinite(fwd)
        tem = np.zeros(n, bool); tem[lo:hi] = True; tem &= np.isfinite(fwd)
        if trm.sum() < 200 or tem.sum() < 30:
            continue
        thr = np.nanquantile(fwd[trm], HI_Q)      # threshold from TRAIN only
        y = (fwd > thr).astype(float)
        if len(np.unique(y[trm])) < 2 or len(np.unique(y[tem])) < 2:
            continue
        lr, mu, sd, med, Z = _fit_pack(X, trm, y)
        # GBM handles NaN but not all-NaN/constant columns -> drop degenerate ones
        vc = [j for j in range(X.shape[1])
              if np.unique(X[trm][~np.isnan(X[trm][:, j]), j]).size >= 2]
        gb = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05,
            max_leaf_nodes=15, l2_regularization=2.0, random_state=42).fit(X[trm][:, vc], y[trm].astype(int))
        yL.append(y[tem].astype(int))
        pL.append(lr.predict_proba(Z[tem])[:, 1])
        pG.append(gb.predict_proba(X[tem][:, vc])[:, 1])
        pB.append(Z[tem][:, vcol])                # persistence baseline: current vol level
    if not yL:
        return None
    y = np.concatenate(yL)
    return (float(roc_auc_score(y, np.concatenate(pL))),
            float(roc_auc_score(y, np.concatenate(pG))),
            float(roc_auc_score(y, np.concatenate(pB))), float(y.mean()))


def posture(path, H=20, folds=8, sf=0.4):
    F, idx, fwd = _features(path, H)
    cols = list(F.columns)
    X = F.to_numpy(float); X = np.where(np.isfinite(X), X, np.nan)
    n = len(X); vcol = cols.index("rv")
    wf = _wf(X, fwd, vcol, H, folds, sf)
    # live model: fit on every row whose forward target is known, score the latest bar
    valid = np.isfinite(fwd)
    thr = np.nanquantile(fwd[valid], HI_Q)
    y = (fwd > thr).astype(float)
    lr, mu, sd, med, Z = _fit_pack(X, valid, y)
    z_last = Z[-1]
    p = float(lr.predict_proba(z_last.reshape(1, -1))[0, 1])
    score = int(round(p * 100))
    contrib = lr.coef_[0] * z_last                # signed push of each factor right now
    order = np.argsort(-np.abs(contrib))
    drivers = [{"factor": PRETTY.get(cols[j], cols[j]),
                "push": "up" if contrib[j] > 0 else "down",
                "c": round(float(contrib[j]), 2)} for j in order[:4] if abs(contrib[j]) > 1e-6]
    worder = np.argsort(-np.abs(lr.coef_[0]))
    weights = [{"factor": PRETTY.get(cols[j], cols[j]), "coef": round(float(lr.coef_[0][j]), 2)}
               for j in worder]
    level, action = _level(score)
    out = {"score": score, "level": level, "action": action, "drivers": drivers,
           "weights": weights, "H": H, "asof": str(idx[-1])}
    if wf:
        out.update({"auc": round(wf[0], 3), "auc_gb": round(wf[1], 3),
                    "auc_base": round(wf[2], 3), "lift": round(wf[0] - wf[2], 3),
                    "base_rate": round(wf[3], 3)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--H", type=int, default=20); ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--start-frac", type=float, default=0.4)
    a = ap.parse_args(argv)
    print(f"Multi-factor RISK posture (high-vol in next {a.H}d). Logistic vs GBM vs persistence.\n")
    print(f"{'instrument':12s} {'score':>5s} {'level':>9s} {'AUC':>6s} {'AUC-gb':>7s} "
          f"{'persist':>8s} {'lift':>6s}  drivers")
    for p in a.inputs:
        r = posture(p, a.H, a.folds, a.start_frac)
        nm = p.replace(".csv", "")
        dr = ", ".join(f"{d['factor']}{'+' if d['push']=='up' else '-'}" for d in r["drivers"][:3])
        print(f"{nm:12s} {r['score']:5d} {r['level']:>9s} {r.get('auc',float('nan')):6.3f} "
              f"{r.get('auc_gb',float('nan')):7.3f} {r.get('auc_base',float('nan')):8.3f} "
              f"{r.get('lift',float('nan')):+6.3f}  {dr}")
    print("\n(lift>0 => the multi-factor blend beats 'vol is already high' persistence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
