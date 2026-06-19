"""
predict_selected.py  —  prediction from each stock's selected high-hit indicators.

Consumes the per-stock selection configs written by analyze_indicators.py
(configs/selected/<SYMBOL>.json) and produces a CURRENT directional call per
stock by majority-voting ONLY those validated indicators, each applied with the
orientation chosen on the train window (+1 as-is, -1 contrarian).

This is the "find the prediction from the couple of indicators that give a good
hit ratio" step. Each call carries:
  - signal      : LONG / SHORT / FLAT
  - confidence  : share of selected indicators agreeing with the vote
  - reliability : the ensemble held-out hit ratio (how well this set scored OOS)

No look-ahead: signals are read at the latest bar; orientations were fixed on
train; held-out hit was measured on the embargoed out-of-sample window.

Usage:
  python predict_selected.py --input processed_indicators
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from optimization import data_loader
import analyze_indicators as ai


def predict_symbol(symbol, csv_path, cfg, at=-1):
    df = data_loader.load_symbol_frame(csv_path)
    cand = ai.build_candidate_frame(df, vp_window=cfg.get("vp_window", 120),
                                    vp_bins=cfg.get("vp_bins", 24))
    selected = cfg.get("selected", [])
    votes, fired = [], []
    for s in selected:
        name = s["indicator"]
        if name not in cand.columns:
            continue
        val = cand[name].to_numpy(float)[at]
        sgn = np.sign(val) * s["orientation"]
        if sgn != 0:
            votes.append(sgn)
            fired.append({"indicator": name, "vote": int(sgn),
                          "heldout_hit": s["heldout_hit"]})
    if not votes:
        return {"symbol": symbol, "signal": "FLAT", "confidence": 0.0,
                "reliability": cfg.get("ensemble_heldout_hit"), "n_fired": 0,
                "agreeing": []}
    net = float(np.sum(votes))
    direction = 1 if net > 0 else (-1 if net < 0 else 0)
    agree = sum(1 for v in votes if v == direction)
    conf = agree / len(votes)
    signal = "LONG" if direction > 0 else ("SHORT" if direction < 0 else "FLAT")
    agreeing = [f["indicator"] for f in fired if f["vote"] == direction]
    return {"symbol": symbol, "signal": signal, "confidence": round(conf, 3),
            "reliability": cfg.get("ensemble_heldout_hit"),
            "n_fired": len(votes), "agreeing": agreeing,
            "as_of": str(df.index[at].date())}


def main(argv=None):
    p = argparse.ArgumentParser(description="Predict from selected high-hit indicators.")
    p.add_argument("--input", default="processed_indicators")
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--out", default="runs/selected_predictions.csv")
    p.add_argument("--min-reliability", type=float, default=0.0,
                   help="only show calls whose ensemble held-out hit >= this")
    args = p.parse_args(argv)

    sel_dir = os.path.join(args.configs_dir, "selected")
    cfgs = sorted(glob.glob(os.path.join(sel_dir, "*.json")))
    if not cfgs:
        print(f"ERROR: no selection configs in {sel_dir} — run analyze_indicators.py first",
              file=sys.stderr)
        return 1

    rows = []
    for cfg_path in cfgs:
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        sym = cfg["symbol"]
        csv = ai_find_csv(args.input, sym)
        if csv is None:
            continue
        rows.append(predict_symbol(sym, csv, cfg))

    df = pd.DataFrame(rows)
    if df.empty:
        print("No predictions produced.")
        return 0
    df = df[df["reliability"].fillna(0) >= args.min_reliability]
    df = df.sort_values(["signal", "reliability", "confidence"], ascending=[True, False, False])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 160); pd.set_option("display.max_rows", 80)
    show = df.copy()
    show["agreeing"] = show["agreeing"].apply(lambda xs: ", ".join(xs[:4]))
    print(show[["symbol", "signal", "confidence", "reliability", "n_fired",
                "as_of", "agreeing"]].to_string(index=False))
    n_long = (df.signal == "LONG").sum(); n_short = (df.signal == "SHORT").sum()
    print(f"\nLONG: {n_long} | SHORT: {n_short} | FLAT: {(df.signal=='FLAT').sum()}")
    print(f"Wrote {args.out}")
    return 0


def ai_find_csv(input_dir, symbol):
    for sym, path in data_loader.list_symbols(input_dir):
        if sym.upper() == symbol.upper():
            return path
    return None


if __name__ == "__main__":
    raise SystemExit(main())
