"""
movement_skill.py  —  is MOVEMENT (a spike, regardless of direction) predictable?

Direction is near-efficient; volatility clusters, so "will a big move happen in the
next H bars" should be more forecastable. This measures it HONESTLY with walk-forward
(expanding retrain, predict forward), reporting:

  AUC                 signal strength for movement (0.5 = none)
  base move rate      P(|fwd move|>k*ATR) unconditionally
  P(move | top-X%)    when the model is most sure a move is coming, how often one
                      actually happens  -> the usable lift over the base rate

Target & features are causal (post leak-fix). No P&L claim — this just asks whether
the *movement* signal is real and tradeable (e.g. via option straddles).

Usage:
  python movement_skill.py --input processed_intraday --symbol NIFTY-50-30m --H 5 --k 1.0
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from optimization import data_loader
from ml_direction import build_xy


def _gbm():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=42)


def run(df, H, k, folds, start_frac, vp_window=120, abs_q=None,
        vix_path=None, rule="30min", vs_implied=False, adv_vol=False, garch=False):
    X, direction, rel, is_move, valid = build_xy(df, H, k, vp_window)
    vix_level = None
    if adv_vol:
        from vol_features import advanced_vol_features
        av = advanced_vol_features(df, with_garch=garch)
        X = pd.concat([X, av], axis=1)
        print(f"  [+advanced vol features ({av.shape[1]}): HAR-RV / GK-YZ / "
              f"time-of-day / Hawkes{' / GARCH' if garch else ''}]")
    if vix_path:
        from vix_features import load_vix_1min, vix_features
        vf = vix_features(df.index, load_vix_1min(vix_path), rule)
        X = pd.concat([X, vf], axis=1)
        vix_level = vf["vix_level"].to_numpy()
        print(f"  [+VIX features: {list(vf.columns)}]")
    Xv = X.to_numpy(np.float32); T = len(df); idx = df.index

    # session-aware forward return (for absolute / vs-implied targets)
    close = df["close"].to_numpy(float)
    days = pd.Series(df.index).dt.normalize().to_numpy()
    fwd = np.full(T, np.nan); fwd[:T - H] = (close[H:] - close[:T - H]) / close[:T - H]
    same = np.zeros(T, bool); same[:T - H] = days[H:] == days[:T - H]
    fwd = np.where(same, fwd, np.nan)

    if vs_implied:
        # THE TRADEABLE TEST: will realized move EXCEED what VIX implies?
        # The VIX(30d annualized) -> intraday H-bar move scaling is NOT the textbook
        # sqrt(t) (intraday excludes overnight vol etc.), so we CALIBRATE it from
        # data: per fold, implied = c * (VIX/100) where c = mean(|realized|) /
        # mean(VIX/100) on TRAIN only (causal). Base rate then ~0.5 (a fair test),
        # and the AUC is clean mispricing-detection skill. y is built in the loop.
        if vix_level is None:
            raise SystemExit("--vs-implied requires --vix")
        valid = valid & np.isfinite(fwd) & (vix_level > 0)
        y_all = None
        print("  [target: realized |move| > VIX-implied; VIX->intraday scaling fit "
              "on TRAIN each fold (calibrated, base ~0.5)]")
    elif abs_q is not None:
        valid = valid & np.isfinite(fwd)
        thr = np.nanquantile(np.abs(fwd), abs_q)
        y_all = (np.abs(fwd) >= thr).astype(int)
        print(f"  [absolute movement: |fwd {H}-bar return| >= {thr*100:.3f}% "
              f"(q={abs_q}), base={y_all[valid].mean():.3f}]")
    else:
        y_all = is_move.astype(int)
    start = int(T * start_frac); step = (T - start) // folds
    rows, oy, op, cals = [], [], [], []
    absfwd = np.abs(fwd)
    for f in range(folds):
        te_lo = start + f * step
        te_hi = (start + (f + 1) * step) if f < folds - 1 else T
        trm = np.zeros(T, bool); trm[:te_lo - H] = True; trm &= valid
        tem = np.zeros(T, bool); tem[te_lo:te_hi] = True; tem &= valid
        if trm.sum() < 300 or tem.sum() < 80:
            continue
        if vs_implied:
            # calibrate VIX->realized scaling on TRAIN, apply to all bars (causal)
            vv = vix_level / 100.0
            c = absfwd[trm].sum() / (vv[trm].sum() + 1e-12)
            cals.append(c)
            yfold = (absfwd > c * vv).astype(int)
        else:
            yfold = y_all
        clf = _gbm(); clf.fit(Xv[trm], yfold[trm])
        p = clf.predict_proba(Xv[tem])[:, 1]; y = yfold[tem]
        thr = np.quantile(p, 0.8); top = p >= thr
        rows.append({"fold": f + 1, "test_n": int(tem.sum()),
                     "period": f"{idx[te_lo].date()}..{idx[te_hi-1].date()}",
                     "AUC": round(roc_auc_score(y, p), 3),
                     "base": round(y.mean(), 3),
                     "P_move@top20": round(y[top].mean(), 3)})
        oy.append(y); op.append(p)
    if not rows:
        print("insufficient data"); return
    y = np.concatenate(oy); p = np.concatenate(op)
    print(pd.DataFrame(rows).to_string(index=False))
    auc = roc_auc_score(y, p); base = y.mean()
    print(f"\nAGGREGATE walk-forward movement skill ({len(rows)} folds):")
    print(f"  AUC                 : {auc:.4f}   (0.5 = no skill)")
    print(f"  base move rate      : {base:.4f}" +
          (f"   [VIX->move scale c={np.mean(cals):.4f}]" if cals else ""))
    for q, lbl in ((0.8, "top-20%"), (0.9, "top-10%"), (0.95, "top- 5%")):
        thr = np.quantile(p, q); m = p >= thr
        print(f"  P(move | {lbl}) : {y[m].mean():.4f}   "
              f"(lift {y[m].mean()-base:+.3f}, n={int(m.sum())})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Walk-forward movement-skill measurement.")
    ap.add_argument("--input", default="processed_intraday")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--start-frac", type=float, default=0.4)
    ap.add_argument("--abs-q", type=float, default=None,
                    help="use ABSOLUTE movement target: |fwd return| >= this quantile "
                         "(e.g. 0.6 = top 40%); keeps volatility-clustering signal")
    ap.add_argument("--vix", default=None, help="path to INDIA_VIX.csv (adds VIX features)")
    ap.add_argument("--vs-implied", action="store_true",
                    help="target = realized move > VIX-implied move (the tradeable edge)")
    ap.add_argument("--adv-vol", action="store_true",
                    help="add HAR-RV / Garman-Klass-Yang-Zhang / time-of-day / Hawkes features")
    ap.add_argument("--garch", action="store_true", help="also add GARCH cond-vol (needs `arch`)")
    a = ap.parse_args(argv)
    by_sym = {s: pth for s, pth in data_loader.list_symbols(a.input)}
    if a.symbol not in by_sym:
        print(f"ERROR: {a.symbol} not found"); return 1
    df = data_loader.load_symbol_frame(by_sym[a.symbol])
    from vix_features import rule_for_symbol
    print(f"=== movement skill {a.symbol} | H={a.H} k={a.k} "
          f"| target={'absolute' if a.abs_q else 'ATR-relative'} "
          f"| VIX={'yes' if a.vix else 'no'} ===")
    run(df, a.H, a.k, a.folds, a.start_frac, abs_q=a.abs_q,
        vix_path=a.vix, rule=rule_for_symbol(a.symbol), vs_implied=a.vs_implied,
        adv_vol=a.adv_vol, garch=a.garch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
