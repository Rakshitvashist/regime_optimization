"""
ml_predict.py  —  live direction call from the trained ML models.

Loads the models saved by ml_direction.py (configs_intraday/ml/<SYMBOL>.joblib),
rebuilds the causal features on the latest data, and emits the CURRENT call:

  signal       LONG / SHORT
  p_up         model probability of up over the next H bars
  confidence   |p_up - 0.5| * 2   (0..1)
  reliability  the model's held-out accuracy (how much to trust it)
  as_of        timestamp of the bar the call is made on

This is the real-time inference step: run it on fresh bars to get the next call.
Features are aligned to each model's saved feature list, so it stays consistent
with training even if a column is missing on the latest data.

Usage:
  python ml_predict.py                       # all trained models
  python ml_predict.py --symbols RELIANCE-30m
  python ml_predict.py --min-confidence 0.4  # only show strong calls
"""
from __future__ import annotations

import argparse, glob, os
import numpy as np, pandas as pd
import joblib

from optimization import data_loader
import analyze_indicators as ai


def predict_one(bundle, csv_path):
    df = data_loader.load_symbol_frame(csv_path)
    X = ai.build_candidate_frame(df, vp_window=bundle.get("vp_window", 120))
    X = X.reindex(columns=bundle["features"])      # align to trained features
    row = X.to_numpy(np.float32)[-1:]              # latest bar
    p_up = float(bundle["model"].predict_proba(row)[0, 1])
    conf = abs(p_up - 0.5) * 2.0
    return {
        "symbol": bundle["symbol"],
        "signal": "LONG" if p_up > 0.5 else "SHORT",
        "p_up": round(p_up, 4),
        "confidence": round(conf, 4),
        "reliability": round(bundle.get("heldout_acc", float("nan")), 4),
        "H_bars": bundle.get("H"),
        "as_of": str(df.index[-1]),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Live ML direction calls.")
    p.add_argument("--input", default="processed_intraday")
    p.add_argument("--configs-dir", default="configs_intraday")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--min-confidence", type=float, default=0.0)
    args = p.parse_args(argv)

    mdir = os.path.join(args.configs_dir, "ml")
    models = sorted(glob.glob(os.path.join(mdir, "*.joblib")))
    if not models:
        print(f"ERROR: no models in {mdir} — run ml_direction.py first"); return 1
    by_sym = {s: pth for s, pth in data_loader.list_symbols(args.input)}

    rows = []
    for mp in models:
        sym = os.path.splitext(os.path.basename(mp))[0]
        if args.symbols and sym not in args.symbols:
            continue
        if sym not in by_sym:
            continue
        rows.append(predict_one(joblib.load(mp), by_sym[sym]))

    df = pd.DataFrame(rows)
    if df.empty:
        print("no calls"); return 0
    df = df[df["confidence"] >= args.min_confidence].sort_values("confidence", ascending=False)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
