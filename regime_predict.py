"""
regime_predict.py  —  can we PREDICT the next regime (not just detect the current)?

Detection (HMM filtered) = "what regime now". Prediction = "what regime in H days"
and "will it SWITCH". Regimes are sticky (~0.78 stay-prob), so the bar to beat is
PERSISTENCE (assume next = current). We test whether a model with richer features
beats it, and whether regime SWITCHES are predictable (the valuable, rare events).

Features (causal): realized vol, vol-of-vol, vol momentum, Hurst (trend/MR),
return momentum, current regime, time-in-regime.
Targets: regime at t+H ; and switch = regime[t+H] != regime[t].

Honest metric: model accuracy vs persistence; switch AUC vs 0.5.

Usage:  python regime_predict.py --inputs BANKNIFTY.csv NIFTY_50.csv --H 10
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import daily_rv_forecast as drv
from vol_cone import daily_total_var, daily_regime
from market_state import _hurst


def build(path):
    df = drv._load_1min(path)
    rv = daily_total_var(df, overnight=True)
    reg = daily_regime(rv).to_numpy().astype(int)        # causal filtered regime
    cl = df["Close"].astype(float).groupby(drv._day_index(df.index)).last()
    lc = np.log(cl).reindex(rv.index)
    ret = lc.diff()
    logrv = np.log(rv)
    F = pd.DataFrame(index=rv.index)
    F["rv"] = logrv
    F["rv_chg5"] = logrv.diff(5)
    F["vov"] = logrv.rolling(22, min_periods=10).std()
    F["ret5"] = ret.rolling(5).sum()
    F["absret5"] = ret.abs().rolling(5).mean()
    F["hurst"] = lc.rolling(80).apply(lambda w: _hurst(w.to_numpy()), raw=False)
    F["regime_now"] = reg.astype(float)
    dur = np.zeros(len(reg))
    for i in range(1, len(reg)):
        dur[i] = dur[i - 1] + 1 if reg[i] == reg[i - 1] else 0
    F["duration"] = dur
    return F, reg, rv.index


def wf_predict(F, reg, H, folds, sf):
    feats = [c for c in F.columns]
    tgt = np.roll(reg, -H).astype(float); tgt[-H:] = np.nan          # regime at t+H
    sw = (np.roll(reg, -H) != reg).astype(float); sw[-H:] = np.nan   # switch?
    ok = F.notna().all(axis=1).to_numpy() & np.isfinite(tgt)
    X = F.to_numpy(); n = len(X)
    st = int(n * sf); step = max(1, (n - st) // folds)
    pt, pp, pers, sy, spitch = [], [], [], [], []
    for f in range(folds):
        lo = st + f * step; hi = (st + (f + 1) * step) if f < folds - 1 else n
        if lo - H < 150:
            continue
        trm = np.zeros(n, bool); trm[:lo - H] = True; trm &= ok
        tem = np.zeros(n, bool); tem[lo:hi] = True; tem &= ok
        if trm.sum() < 200 or tem.sum() < 40:
            continue
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
            max_leaf_nodes=15, l2_regularization=2.0, random_state=42)
        clf.fit(X[trm], tgt[trm].astype(int))
        pp.append(clf.predict(X[tem])); pt.append(tgt[tem].astype(int))
        pers.append(F["regime_now"].to_numpy()[tem].astype(int))       # persistence
        # switch model
        sc = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
            max_leaf_nodes=15, l2_regularization=2.0, random_state=1)
        if len(np.unique(sw[trm].astype(int))) > 1:
            sc.fit(X[trm], sw[trm].astype(int))
            spitch.append(sc.predict_proba(X[tem])[:, 1]); sy.append(sw[tem].astype(int))
    yt = np.concatenate(pt); yp = np.concatenate(pp); pr = np.concatenate(pers)
    acc = accuracy_score(yt, yp); pacc = accuracy_score(yt, pr)
    out = {"model_acc": acc, "persist_acc": pacc, "n": len(yt)}
    if sy:
        sy = np.concatenate(sy); sp = np.concatenate(spitch)
        out["switch_base"] = float(sy.mean())
        out["switch_auc"] = float(roc_auc_score(sy, sp)) if len(np.unique(sy)) > 1 else float("nan")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--start-frac", type=float, default=0.4)
    a = ap.parse_args(argv)
    print(f"Predict regime at t+{a.H} (3-state HMM). Beat PERSISTENCE (next=current)?\n")
    print(f"{'instrument':12s} {'model':>7s} {'persist':>8s} {'gain':>6s} | "
          f"{'switch%':>8s} {'switchAUC':>10s}")
    for p in a.inputs:
        nm = p.replace(".csv", "")
        F, reg, idx = build(p)
        r = wf_predict(F, reg, a.H, a.folds, a.start_frac)
        g = r["model_acc"] - r["persist_acc"]
        print(f"{nm:12s} {r['model_acc']:7.3f} {r['persist_acc']:8.3f} {g:+6.3f} | "
              f"{r.get('switch_base',float('nan'))*100:7.1f}% {r.get('switch_auc',float('nan')):10.3f}")
    print("\n(model>persist => prediction adds value; switchAUC>0.55 => switches are foreseeable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
