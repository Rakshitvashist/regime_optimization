"""
walk_forward.py  —  walk-forward retraining harness.

A single train/held-out split can flatter a model that only worked in one regime.
This retrains on an EXPANDING window and predicts the next out-of-sample block,
rolling forward across the whole history — exactly how you'd run it in production
(retrain periodically, predict forward). It reports per-fold and aggregate OOS
accuracy, so you see whether the edge holds across 2017, 2020, 2022, 2024...

Each fold:
  train on bars [0 .. test_start - embargo]   (move-only direction model)
  test  on bars [test_start .. test_end]      (never seen, strictly future)

Usage:
  python walk_forward.py --input processed_intraday --symbol RELIANCE-30m --folds 8
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from optimization import data_loader
from ml_direction import build_xy


def _gbm():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=42)


def run(df, H, k, folds, start_frac, move_only, vp_window=120):
    X, direction, rel, is_move, valid = build_xy(df, H, k, vp_window)
    Xv = X.to_numpy(np.float32); T = len(df)
    idx = df.index
    start = int(T * start_frac)
    step = (T - start) // folds
    base = (is_move if move_only else np.ones(T, bool))
    rows, oos_true, oos_pred, oos_conf = [], [], [], []

    for f in range(folds):
        te_lo = start + f * step
        te_hi = (start + (f + 1) * step) if f < folds - 1 else T
        trm = np.zeros(T, bool); trm[:te_lo - H] = True; trm &= valid & base
        tem = np.zeros(T, bool); tem[te_lo:te_hi] = True; tem &= valid & base
        if trm.sum() < 300 or tem.sum() < 50:
            continue
        clf = _gbm(); clf.fit(Xv[trm], direction[trm])
        p = clf.predict_proba(Xv[tem])[:, 1]; pred = (p > 0.5).astype(int)
        yt = direction[tem]
        acc = accuracy_score(yt, pred)
        conf = np.abs(p - 0.5)
        m = conf >= np.quantile(conf, 0.75)
        cacc = accuracy_score(yt[m], pred[m]) if m.sum() > 20 else float("nan")
        rows.append({"fold": f + 1, "train_n": int(trm.sum()), "test_n": int(tem.sum()),
                     "period": f"{idx[te_lo].date()}..{idx[te_hi-1].date()}",
                     "acc": round(acc, 4), "conf25_acc": round(cacc, 4)})
        oos_true.append(yt); oos_pred.append(pred); oos_conf.append(conf)

    if not rows:
        print("insufficient data for folds"); return
    yt = np.concatenate(oos_true); pr = np.concatenate(oos_pred); cf = np.concatenate(oos_conf)
    print(pd.DataFrame(rows).to_string(index=False))
    base_acc = max(yt.mean(), 1 - yt.mean())
    m = cf >= np.quantile(cf, 0.75)
    print(f"\nAGGREGATE walk-forward ({len(rows)} folds, move_only={move_only}):")
    print(f"  OOS accuracy           : {accuracy_score(yt, pr):.4f}  (majority base {base_acc:.4f})")
    print(f"  OOS conf-25% accuracy  : {accuracy_score(yt[m], pr[m]):.4f}  (n={int(m.sum())})")
    accs = [r["acc"] for r in rows]
    print(f"  fold acc: min {min(accs):.3f}  mean {np.mean(accs):.3f}  max {max(accs):.3f}  "
          f"| folds>base: {sum(a>base_acc for a in accs)}/{len(accs)}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Walk-forward retraining harness.")
    p.add_argument("--input", default="processed_intraday")
    p.add_argument("--symbol", required=True)
    p.add_argument("--H", type=int, default=5)
    p.add_argument("--k", type=float, default=1.0)
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--start-frac", type=float, default=0.4)
    p.add_argument("--all-bars", action="store_true", help="don't restrict to move bars")
    args = p.parse_args(argv)

    by_sym = {s: pth for s, pth in data_loader.list_symbols(args.input)}
    if args.symbol not in by_sym:
        print(f"ERROR: {args.symbol} not found"); return 1
    df = data_loader.load_symbol_frame(by_sym[args.symbol])
    print(f"=== walk-forward {args.symbol} | H={args.H} k={args.k} ===")
    run(df, args.H, args.k, args.folds, args.start_frac, not args.all_bars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
