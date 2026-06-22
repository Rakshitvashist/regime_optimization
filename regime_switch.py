"""
regime_switch.py  —  regime-SWITCH early-warning signal (the predictable part).

Predicting the exact next regime barely beats persistence; predicting a SWITCH
(regime[t+H] != regime[t]) does work (~AUC 0.65). This pushes that higher with:

  + early-warning features : vol-of-vol acceleration, Hurst dynamics, return/vol
                             momentum, time-in-regime
  + cross-asset / macro    : crude & gold realized-vol changes, macro risk regime,
                             rolling cross-asset correlation drift
  + BOCPD                  : Bayesian Online Change-Point probability (Adams-MacKay)
                             — an independent "distribution just shifted" signal

Outputs P(regime switch in next H days) and the live switch probability.
Honest metric: walk-forward switch AUC, base features vs rich+BOCPD.

Usage:  python regime_switch.py --inputs BANKNIFTY.csv NIFTY_50.csv --H 10
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import daily_rv_forecast as drv
from vol_cone import daily_total_var, daily_regime
from market_state import _hurst


def bocpd(x, hazard=1.0 / 120.0):
    """Bayesian online change-point: returns per-step changepoint probability."""
    x = np.asarray(x, float); x = np.nan_to_num(x)
    T = len(x)
    mu = np.array([0.0]); kp = np.array([1.0]); al = np.array([1.0]); be = np.array([1.0])
    rl = np.array([1.0]); cp = np.zeros(T)
    for i in range(T):
        xi = x[i]
        scale = np.sqrt(be * (kp + 1) / (al * kp))
        pred = tdist.pdf(xi, df=2 * al, loc=mu, scale=scale)
        growth = rl * pred * (1 - hazard)
        change = float(np.sum(rl * pred * hazard))
        rl = np.concatenate([[change], growth]); rl /= rl.sum() + 1e-300
        cp[i] = rl[0]
        mu_new = (kp * mu + xi) / (kp + 1)
        be_new = be + (kp * (xi - mu) ** 2) / (2 * (kp + 1))
        mu = np.concatenate([[0.0], mu_new]); kp = np.concatenate([[1.0], kp + 1])
        al = np.concatenate([[1.0], al + 0.5]); be = np.concatenate([[1.0], be_new])
    return cp


def _logrv(path):
    M = daily_total_var(drv._load_1min(path), overnight=True)
    s = np.log(M); s.index = [pd.Timestamp(d).date() for d in s.index]; return s


def _dret(path):
    df = drv._load_1min(path); cl = df["Close"].astype(float).groupby(drv._day_index(df.index)).last()
    s = np.log(cl).diff(); s.index = [pd.Timestamp(d).date() for d in s.index]; return s


def build_rich(path):
    df = drv._load_1min(path); rv = daily_total_var(df, overnight=True)
    reg = daily_regime(rv).to_numpy().astype(int)
    idx = [pd.Timestamp(d).date() for d in rv.index]
    cl = df["Close"].astype(float).groupby(drv._day_index(df.index)).last()
    lc = np.log(cl); lc.index = [pd.Timestamp(d).date() for d in lc.index]; lc = lc.reindex(idx)
    ret = lc.diff(); logrv = np.log(rv); logrv.index = idx
    F = pd.DataFrame(index=idx)
    # --- base early-warning ---
    F["rv"] = logrv.to_numpy(); F["rv_chg5"] = logrv.diff(5).to_numpy()
    vov = logrv.rolling(22, min_periods=10).std()
    F["vov"] = vov.to_numpy(); F["vov_accel"] = vov.diff(5).to_numpy()
    F["ret5"] = ret.rolling(5).sum().to_numpy(); F["absret5"] = ret.abs().rolling(5).mean().to_numpy()
    hur = lc.rolling(80).apply(lambda w: _hurst(w.to_numpy()), raw=False)
    F["hurst"] = hur.to_numpy(); F["hurst_chg"] = hur.diff(10).to_numpy()
    F["regime_now"] = reg.astype(float)
    dur = np.zeros(len(reg))
    for i in range(1, len(reg)):
        dur[i] = dur[i - 1] + 1 if reg[i] == reg[i - 1] else 0
    F["duration"] = dur
    # --- cross-asset / macro ---
    cr, go = _logrv("CRUDEOIL.csv").reindex(idx), _logrv("GOLD.csv").reindex(idx)
    F["crude_rv_chg"] = cr.diff(5).to_numpy(); F["gold_rv_chg"] = go.diff(5).to_numpy()
    nr, crr, gor = ret, _dret("CRUDEOIL.csv").reindex(idx), _dret("GOLD.csv").reindex(idx)
    al = pd.DataFrame({"n": nr, "c": crr, "g": gor})
    F["corr_nc_chg"] = al["n"].rolling(30).corr(al["c"]).diff(10).to_numpy()
    F["corr_ng_chg"] = al["n"].rolling(30).corr(al["g"]).diff(10).to_numpy()
    try:
        from macro_regime import compute as mc
        mac = mc(); mac.index = [pd.Timestamp(d).date() for d in mac.index]
        F["macro_regime"] = mac["regime"].reindex(idx).to_numpy()
    except Exception:
        pass
    # --- BOCPD on standardized returns ---
    rstd = (ret / (ret.rolling(20, min_periods=5).std() + 1e-9)).fillna(0).to_numpy()
    F["bocpd"] = bocpd(rstd)
    return F, reg


BASE = ["rv", "rv_chg5", "vov", "ret5", "absret5", "hurst", "regime_now", "duration"]


def _base_features(path):
    """Lightweight per-instrument switch features (no cross-asset) + BOCPD — fast,
    works on any 1-min file including commodities."""
    df = drv._load_1min(path); rv = daily_total_var(df, overnight=True)
    reg = daily_regime(rv).to_numpy().astype(int)
    idx = [pd.Timestamp(d).date() for d in rv.index]
    cl = df["Close"].astype(float).groupby(drv._day_index(df.index)).last()
    lc = np.log(cl); lc.index = [pd.Timestamp(d).date() for d in lc.index]; lc = lc.reindex(idx)
    ret = lc.diff(); logrv = np.log(rv); logrv.index = idx
    F = pd.DataFrame(index=idx)
    F["rv"] = logrv.to_numpy(); F["rv_chg5"] = logrv.diff(5).to_numpy()
    vov = logrv.rolling(22, min_periods=10).std()
    F["vov"] = vov.to_numpy(); F["vov_accel"] = vov.diff(5).to_numpy()
    F["ret5"] = ret.rolling(5).sum().to_numpy(); F["absret5"] = ret.abs().rolling(5).mean().to_numpy()
    F["hurst"] = lc.rolling(80).apply(lambda w: _hurst(w.to_numpy()), raw=False).to_numpy()
    dur = np.zeros(len(reg))
    for i in range(1, len(reg)):
        dur[i] = dur[i - 1] + 1 if reg[i] == reg[i - 1] else 0
    F["regime_now"] = reg.astype(float); F["duration"] = dur
    rstd = (ret / (ret.rolling(20, min_periods=5).std() + 1e-9)).fillna(0).to_numpy()
    F["bocpd"] = bocpd(rstd)
    return F, reg, idx


def _fit_predict(X, y, tr, vc):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
        max_leaf_nodes=15, l2_regularization=2.0, random_state=42)
    clf.fit(X[tr][:, vc], y[tr]); return clf


def switch_now(path, H=10):
    """Live P(regime switch in next H days) + holdout AUC + current BOCPD level."""
    F, reg, idx = _base_features(path)
    sw = (np.roll(reg, -H) != reg).astype(float); sw[-H:] = np.nan
    X = F.to_numpy(float); X = np.where(np.isfinite(X), X, np.nan)
    ok = np.isfinite(sw); lab = np.where(ok)[0]
    auc = base = None
    if len(lab) > 260:
        cut = int(len(lab) * 0.7); tr, te = lab[:cut], lab[cut:]
        if len(np.unique(sw[tr].astype(int))) > 1 and len(np.unique(sw[te].astype(int))) > 1:
            vc = [j for j in range(X.shape[1]) if np.unique(X[tr][~np.isnan(X[tr][:, j]), j]).size >= 2]
            if vc:
                clf = _fit_predict(X, sw.astype(int), tr, vc)
                auc = float(roc_auc_score(sw[te].astype(int), clf.predict_proba(X[te][:, vc])[:, 1]))
                base = float(sw[te].astype(int).mean())
    vc = [j for j in range(X.shape[1]) if np.unique(X[lab][~np.isnan(X[lab][:, j]), j]).size >= 2]
    clf = _fit_predict(X, sw.astype(int), lab, vc)
    p = float(clf.predict_proba(X[-1:][:, vc])[:, 1][0])
    return {"p_switch": round(p, 3), "auc": round(auc, 3) if auc is not None else None,
            "switch_base": round(base, 3) if base is not None else None,
            "bocpd": round(float(np.nan_to_num(F["bocpd"].iloc[-1])), 3),
            "regime_now": int(reg[-1]), "asof": str(idx[-1]), "H": H}


def wf_switch(F, reg, cols, H, folds, sf):
    sw = (np.roll(reg, -H) != reg).astype(float); sw[-H:] = np.nan
    ok = np.isfinite(sw)                                  # only the TARGET must be valid
    X = F[cols].to_numpy(float)
    X = np.where(np.isfinite(X), X, np.nan)               # HistGBM handles NaN, not inf
    n = len(X); st = int(n * sf); step = max(1, (n - st) // folds)
    oy, op = [], []
    for f in range(folds):
        lo = st + f * step; hi = (st + (f + 1) * step) if f < folds - 1 else n
        if lo - H < 150:
            continue
        trm = np.zeros(n, bool); trm[:lo - H] = True; trm &= ok
        tem = np.zeros(n, bool); tem[lo:hi] = True; tem &= ok
        if trm.sum() < 200 or tem.sum() < 40 or len(np.unique(sw[trm].astype(int))) < 2:
            continue
        Xtr = X[trm]
        vc = [j for j in range(X.shape[1])
              if np.unique(Xtr[~np.isnan(Xtr[:, j]), j]).size >= 2]   # drop constant/all-NaN cols
        if not vc:
            continue
        clf = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05,
            max_leaf_nodes=15, l2_regularization=2.0, random_state=42)
        clf.fit(Xtr[:, vc], sw[trm].astype(int))
        op.append(clf.predict_proba(X[tem][:, vc])[:, 1]); oy.append(sw[tem].astype(int))
    yt = np.concatenate(oy); yp = np.concatenate(op)
    return roc_auc_score(yt, yp), float(yt.mean())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--H", type=int, default=10); ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--start-frac", type=float, default=0.4)
    a = ap.parse_args(argv)
    print(f"Regime-SWITCH prediction (t+{a.H}). AUC: base features vs rich + BOCPD.\n")
    print(f"{'instrument':12s} {'switch%':>8s} {'AUC base':>9s} {'AUC rich':>9s} {'gain':>6s}")
    for p in a.inputs:
        nm = p.replace(".csv", ""); F, reg = build_rich(p)
        rich = [c for c in F.columns]
        ab, base = wf_switch(F, reg, BASE, a.H, a.folds, a.start_frac)
        ar, _ = wf_switch(F, reg, rich, a.H, a.folds, a.start_frac)
        print(f"{nm:12s} {base*100:7.1f}% {ab:9.3f} {ar:9.3f} {ar-ab:+6.3f}")
    print("\n(rich+BOCPD > base => the extra early-warning + change-point signal helps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
