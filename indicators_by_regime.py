"""
indicators_by_regime.py  —  WHICH indicators carry the spike signal in WHICH regime.

The payoff of the whole signal stack: label each bar with the causal HMM regime
(quiet / normal / explosive), then, separately within each regime, train a movement
(spike) predictor and report (a) how predictable movement is in that regime and
(b) which indicators drive it. So you learn "in explosive regimes indicators X,Y
matter; in quiet, Z" — regime-conditional indicator optimization.

No look-ahead: regime labels are causal/walk-forward (hmm_regime.compute_regime),
features are causal, the movement target is a session-aware forward move, and the
per-regime model is trained on an earlier time block and tested on a later one.

Usage:
  python indicators_by_regime.py --input processed_intraday --symbol BANKNIFTY-30m
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from optimization import data_loader
import analyze_indicators as ai
from hmm_regime import compute_regime
from vix_features import rule_for_symbol

REGIME_NAMES = {0: "quiet", 1: "normal", 2: "explosive"}


def _gbm():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=42)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Which indicators matter in which regime.")
    ap.add_argument("--input", default="processed_intraday")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--abs-q", type=float, default=0.6, help="spike = |move| top (1-q) frac")
    ap.add_argument("--heldout-frac", type=float, default=0.3)
    ap.add_argument("--vix", default=None)
    ap.add_argument("--topn", type=int, default=8)
    a = ap.parse_args(argv)

    by_sym = {s: pth for s, pth in data_loader.list_symbols(a.input)}
    if a.symbol not in by_sym:
        print(f"ERROR: {a.symbol} not found"); return 1
    df = data_loader.load_symbol_frame(by_sym[a.symbol])
    rule = rule_for_symbol(a.symbol)

    # causal regime labels
    regime = compute_regime(df, a.states, 8, 0.4, a.vix, rule)
    # indicator candidate features
    X = ai.build_candidate_frame(df, vp_window=120)
    feats = list(X.columns); Xv = X.to_numpy(np.float32)
    # session-aware spike (absolute-movement) target
    close = df["close"].to_numpy(float); T = len(df); H = a.H
    days = pd.Series(df.index).dt.normalize().to_numpy()
    fwd = np.full(T, np.nan); fwd[:T-H] = (close[H:] - close[:T-H]) / close[:T-H]
    same = np.zeros(T, bool); same[:T-H] = days[H:] == days[:T-H]
    fwd = np.where(same, fwd, np.nan)
    valid = np.isfinite(fwd) & (regime >= 0)
    thr = np.nanquantile(np.abs(fwd[valid]), a.abs_q)
    y = (np.abs(fwd) >= thr).astype(int)

    # temporal split (train earlier, test later) shared across regimes
    cut = int(T * (1 - a.heldout_frac)); emb = H
    is_tr = np.zeros(T, bool); is_tr[:cut - emb] = True
    is_te = np.zeros(T, bool); is_te[cut:] = True

    print(f"=== indicators-by-regime  {a.symbol}  | spike=|fwd {H}b move|>={thr*100:.3f}% "
          f"| {len(feats)} features ===")
    for r in range(a.states):
        trm = is_tr & valid & (regime == r)
        tem = is_te & valid & (regime == r)
        nm = REGIME_NAMES.get(r, f"r{r}")
        if trm.sum() < 300 or tem.sum() < 100:
            print(f"\n[{nm}] too few bars (train {trm.sum()}, test {tem.sum()})"); continue
        clf = _gbm(); clf.fit(Xv[trm], y[trm])
        p = clf.predict_proba(Xv[tem])[:, 1]
        auc = roc_auc_score(y[tem], p)
        imp = permutation_importance(clf, Xv[tem], y[tem], n_repeats=4,
                                     random_state=42, scoring="roc_auc",
                                     max_samples=min(4000, int(tem.sum())))
        order = np.argsort(imp.importances_mean)[::-1][:a.topn]
        print(f"\n[{nm}]  bars: train {trm.sum():,} test {tem.sum():,} | "
              f"spike base {y[tem].mean():.3f} | movement AUC {auc:.3f}")
        print("   top indicators (regime-specific):")
        for i in order:
            if imp.importances_mean[i] <= 0:
                continue
            print(f"     {feats[i]:24s} {imp.importances_mean[i]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
