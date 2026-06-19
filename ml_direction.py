"""
ml_direction.py  —  ML direction + movement predictor (gradient-boosted trees).

Directly predicts, from the engineered features (liquidity sweep + spike-imminent
+ volume/time profile + divergences + indicators), the DIRECTION of the next H
bars — and, optionally, only flags it when a MOVEMENT (spike) is likely.

Why GBM, not the LSTM: for a tabular feature set like this, gradient-boosted
trees are usually as good or better than an LSTM, train in seconds, handle NaNs
natively, and give feature importances (so you see WHAT drives the call). The
torch LSTM path (optimize_indicators --method dl) remains for sequence learning.

No look-ahead: features are causal; the target is the FORWARD return, computed
WITHIN the same session (never across the overnight gap); train/test are split
with an embargo so no train bar's forward window leaks into the test window.

Saves model + feature list to configs_intraday/ml/<SYMBOL>.joblib for live use.

Usage:
  python ml_direction.py --input processed_intraday --symbol RELIANCE-30m --H 5
"""
from __future__ import annotations

import argparse, json, os
import numpy as np, pandas as pd
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score

from optimization import data_loader
import analyze_indicators as ai


def _atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def build_xy(df, H, k, vp_window):
    X = ai.build_candidate_frame(df, vp_window=vp_window)
    close = df["close"].to_numpy(float)
    atr = _atr(df).to_numpy(float)
    T = len(close)
    fwd = np.full(T, np.nan)
    fwd[:T - H] = (close[H:] - close[:T - H])
    # session-aware: null any forward window that crosses the overnight gap
    days = pd.Series(df.index).dt.normalize().to_numpy()
    same_session = np.zeros(T, bool)
    same_session[:T - H] = days[H:] == days[:T - H]
    fwd = np.where(same_session, fwd, np.nan)
    rel = fwd / (atr + 1e-9)               # forward move in ATRs
    direction = (fwd > 0).astype(int)      # 1 = up, 0 = down
    is_move = np.abs(rel) > k              # a real "movement" (spike)
    return X, direction, rel, is_move, np.isfinite(fwd)


def main(argv=None):
    p = argparse.ArgumentParser(description="ML direction/movement predictor.")
    p.add_argument("--input", default="processed_intraday")
    p.add_argument("--symbol", required=True, help="e.g. RELIANCE-30m")
    p.add_argument("--H", type=int, default=5, help="forward horizon in bars")
    p.add_argument("--k", type=float, default=1.0, help="spike = |move| > k*ATR")
    p.add_argument("--heldout-frac", type=float, default=0.3)
    p.add_argument("--configs-dir", default="configs_intraday")
    p.add_argument("--move-only", action="store_true",
                   help="train/evaluate only on bars where a movement (spike) occurs")
    args = p.parse_args(argv)

    path = None
    for sym, pth in data_loader.list_symbols(args.input):
        if sym == args.symbol:
            path = pth
    if path is None:
        print(f"ERROR: {args.symbol} not in {args.input}"); return 1
    df = data_loader.load_symbol_frame(path)

    X, direction, rel, is_move, valid = build_xy(df, args.H, args.k, 120)
    feats = list(X.columns)
    Xv = X.to_numpy(np.float32)
    T = len(df)
    emb = args.H
    cut = int(T * (1 - args.heldout_frac))
    tr = np.zeros(T, bool); tr[:cut - emb] = True
    te = np.zeros(T, bool); te[cut:] = True
    sel = valid & (is_move if args.move_only else np.ones(T, bool))
    tr &= sel; te &= sel

    Xtr, ytr = Xv[tr], direction[tr]
    Xte, yte = Xv[te], direction[te]
    print(f"{args.symbol}: H={args.H} bars | train={tr.sum():,} test={te.sum():,} "
          f"| features={len(feats)} | move_only={args.move_only}")
    if tr.sum() < 200 or te.sum() < 100:
        print("  insufficient samples"); return 0

    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=None, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=42)
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba > 0.5).astype(int)
    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba)
    base = max(yte.mean(), 1 - yte.mean())     # majority-class baseline
    print(f"  held-out accuracy: {acc:.4f}  (majority baseline {base:.4f})  AUC {auc:.4f}")

    # accuracy on the most-confident calls (this is "movement before it happens")
    conf = np.abs(proba - 0.5)
    for q in (0.5, 0.75, 0.9):
        thr = np.quantile(conf, q)
        m = conf >= thr
        if m.sum() > 20:
            print(f"  top-{int((1-q)*100):2d}% most-confident calls: "
                  f"acc {accuracy_score(yte[m], pred[m]):.4f}  (n={m.sum()})")

    # which features drive the call
    imp = permutation_importance(clf, Xte, yte, n_repeats=4, random_state=42,
                                 scoring="accuracy", max_samples=min(4000, te.sum()))
    order = np.argsort(imp.importances_mean)[::-1][:12]
    print("  top features driving direction:")
    for i in order:
        print(f"    {feats[i]:22s} {imp.importances_mean[i]:+.4f}")

    outdir = os.path.join(args.configs_dir, "ml")
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, f"{args.symbol}.joblib")
    joblib.dump({"model": clf, "features": feats, "H": args.H, "k": args.k,
                 "vp_window": 120, "symbol": args.symbol,
                 "heldout_acc": float(acc), "auc": float(auc)}, outp)
    print(f"  saved -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
