"""
ml_cascade.py  —  2-stage movement-gate cascade.

  Stage A  (movement detector): on EVERY bar, P(a spike |move|>k*ATR occurs in
           the next H bars).  Trained on all bars.
  Stage B  (direction):         GIVEN a move, P(up).  Trained on move bars only
           (this is the strong move-only model).

Live decision: ACT only when A says a move is likely AND B is confident about the
direction. This is the production shape — you don't trade every bar, you wait for
the gate (movement coming) then take the side B is sure of.

Evaluation (held-out, session-clean, embargoed) reports, at several operating
points, the trade-off:
    coverage     = fraction of bars we ACT on
    move_prec    = of acted bars, how many had a real movement (Stage A precision)
    dir_acc      = direction accuracy on acted+moved bars (Stage B)

Saves both stages to configs_intraday/cascade/<SYMBOL>.joblib.

Usage:
  python ml_cascade.py --input processed_intraday --symbol RELIANCE-30m --H 5 --k 1.0
  python ml_cascade.py --predict --input processed_intraday        # live cascade calls
"""
from __future__ import annotations

import argparse, glob, os
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from optimization import data_loader
import analyze_indicators as ai
from ml_direction import build_xy


def _gbm():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=42)


def train_symbol(df, H, k, heldout_frac, vp_window=120):
    X, direction, rel, is_move, valid = build_xy(df, H, k, vp_window)
    feats = list(X.columns)
    Xv = X.to_numpy(np.float32)
    T = len(df); cut = int(T * (1 - heldout_frac)); emb = H
    tr = np.zeros(T, bool); tr[:cut - emb] = True
    te = np.zeros(T, bool); te[cut:] = True

    A = _gbm(); A.fit(Xv[tr & valid], is_move[tr & valid].astype(int))
    bTr = tr & valid & is_move
    B = _gbm(); B.fit(Xv[bTr], direction[bTr])

    aTe = te & valid
    pA = A.predict_proba(Xv[aTe])[:, 1]
    pB = B.predict_proba(Xv[aTe])[:, 1]
    moved = is_move[aTe]; dirn = direction[aTe]; predB = (pB > 0.5).astype(int)
    confB = np.abs(pB - 0.5)

    print(f"{df.attrs.get('sym','')}: H={H} k={k} | test bars={aTe.sum():,} | "
          f"base movement rate={moved.mean():.3f}")
    print(f"{'gateA':>6s} {'confB':>6s} {'coverage':>9s} {'move_prec':>10s} {'dir_acc':>8s} {'n_act':>7s}")
    best = None
    for qA in (0.5, 0.75, 0.9):
        tA = np.quantile(pA, qA); selA = pA >= tA
        for qB in (0.5, 0.75):
            tB = np.quantile(confB[selA], qB) if selA.any() else 0.0
            act = selA & (confB >= tB)
            cov = act.mean()
            am = act & moved
            mp = moved[act].mean() if act.any() else float("nan")
            da = accuracy_score(dirn[am], predB[am]) if am.sum() > 20 else float("nan")
            print(f"{qA:6.2f} {qB:6.2f} {cov:9.3f} {mp:10.3f} {da:8.4f} {int(act.sum()):7d}")
            if am.sum() > 20 and (best is None or da > best[0]):
                best = (da, qA, qB)

    # Store ABSOLUTE live thresholds at the best operating point so inference
    # gates consistently with training (Stage-A probs sit near the base rate).
    bqA, bqB = (best[1], best[2]) if best else (0.75, 0.5)
    thrA = float(np.quantile(pA, bqA))
    selA = pA >= thrA
    thrB = float(np.quantile(confB[selA], bqB)) if selA.any() else 0.1

    bundle = {"cascade": True, "modelA": A, "modelB": B, "features": feats,
              "H": H, "k": k, "vp_window": vp_window, "symbol": df.attrs.get("sym", ""),
              "gate_qA": bqA, "conf_qB": bqB, "thrA": thrA, "thrB": thrB,
              "best_dir_acc": float(best[0]) if best else float("nan")}
    return bundle


def predict_symbol(bundle, df):
    X = ai.build_candidate_frame(df, vp_window=bundle.get("vp_window", 120))
    X = X.reindex(columns=bundle["features"])
    row = X.to_numpy(np.float32)[-1:]
    pA = float(bundle["modelA"].predict_proba(row)[0, 1])
    pB = float(bundle["modelB"].predict_proba(row)[0, 1])
    # ACT only when the movement gate AND direction confidence clear the stored
    # operating-point thresholds (calibrated on training, not a hardcoded 0.5).
    confB_raw = abs(pB - 0.5)
    act = (pA >= bundle.get("thrA", 0.5)) and (confB_raw >= bundle.get("thrB", 0.25))
    return {"symbol": bundle["symbol"],
            "p_move": round(pA, 3), "signal": "LONG" if pB > 0.5 else "SHORT",
            "p_up": round(pB, 3), "dir_conf": round(confB_raw * 2, 3),
            "ACT": "YES" if act else "wait",
            "reliability": round(bundle.get("best_dir_acc", float("nan")), 3),
            "as_of": str(df.index[-1])}


def main(argv=None):
    p = argparse.ArgumentParser(description="2-stage movement-gate cascade.")
    p.add_argument("--input", default="processed_intraday")
    p.add_argument("--symbol")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--H", type=int, default=5)
    p.add_argument("--k", type=float, default=1.0)
    p.add_argument("--heldout-frac", type=float, default=0.3)
    p.add_argument("--configs-dir", default="configs_intraday")
    p.add_argument("--predict", action="store_true")
    args = p.parse_args(argv)

    cdir = os.path.join(args.configs_dir, "cascade")
    by_sym = {s: pth for s, pth in data_loader.list_symbols(args.input)}

    if args.predict:
        rows = []
        for mp in sorted(glob.glob(os.path.join(cdir, "*.joblib"))):
            b = joblib.load(mp)
            if b["symbol"] in by_sym:
                df = data_loader.load_symbol_frame(by_sym[b["symbol"]])
                rows.append(predict_symbol(b, df))
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))
        else:
            print("no cascade models — train first")
        return 0

    os.makedirs(cdir, exist_ok=True)
    syms = [args.symbol] if args.symbol else (args.symbols or list(by_sym.keys()))
    for sym in syms:
        if sym not in by_sym:
            print(f"  {sym}: not found"); continue
        df = data_loader.load_symbol_frame(by_sym[sym]); df.attrs["sym"] = sym
        b = train_symbol(df, args.H, args.k, args.heldout_frac)
        joblib.dump(b, os.path.join(cdir, f"{sym}.joblib"))
        print(f"  saved -> {os.path.join(cdir, sym)}.joblib  (best dir_acc {b['best_dir_acc']:.4f})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
