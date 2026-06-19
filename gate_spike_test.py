"""
gate_spike_test.py  —  does gating entries on a VOLUME SPIKE-SIGNAL raise hit rate?

(b) from the plan: take each instrument's selected-indicator ensemble vote and
only act when the spike-imminent signal (spike_volz) is elevated. Compare the
held-out hit ratio / edge UNGATED vs GATED.

Usage:
  python gate_spike_test.py --input processed_intraday --configs-dir configs_intraday \
      --symbols RELIANCE-5m RELIANCE-15m RELIANCE-30m
"""
from __future__ import annotations

import argparse, glob, json, os
import numpy as np, pandas as pd

from optimization import data_loader, splitter
import analyze_indicators as ai


def run(csv, cfg, gate_q):
    df = data_loader.load_symbol_frame(csv)
    cand = ai.build_candidate_frame(df, vp_window=cfg.get("vp_window", 120))
    close = df["close"].to_numpy(float); T = len(close)
    horizons = cfg.get("horizons", [5, 10]); d = min(horizons)
    win = splitter.make_window(cfg["symbol"], df.index, horizons, heldout_frac=0.3, embargo=None)
    if isinstance(win, splitter.InsufficientHistory):
        return None
    ho = set(range(*win.heldout_slice.indices(T)))

    net = np.zeros(T)
    for s in cfg.get("selected", []):
        if s["indicator"] in cand.columns:
            net += s["orientation"] * np.sign(cand[s["indicator"]].to_numpy(float))
    vote = np.sign(net)

    volz = (cand["spike_volz"].to_numpy(float) if "spike_volz" in cand.columns
            else np.zeros(T))
    finite = np.isfinite(volz)
    thr = np.quantile(volz[finite], gate_q) if finite.any() else 0.0
    gate = volz > thr

    fwd = np.full(T, np.nan); fwd[:T - d] = (close[d:] - close[:T - d]) / (close[:T - d] + 1e-9)

    def stats(extra):
        idx = [t for t in range(T) if t in ho and vote[t] != 0 and np.isfinite(fwd[t]) and extra[t]]
        if not idx:
            return 0, float("nan"), float("nan")
        v = vote[idx]; f = fwd[idx]
        return len(idx), float(np.mean(np.sign(v) == np.sign(f))), float(np.mean(v * f) * 100)

    n0, h0, e0 = stats(np.ones(T, bool))
    n1, h1, e1 = stats(gate)
    return {"symbol": cfg["symbol"], "thr": round(float(thr), 2),
            "n_ungated": n0, "hit_ungated": round(h0, 4), "edge_ungated_pct": round(e0, 3),
            "n_gated": n1, "hit_gated": round(h1, 4), "edge_gated_pct": round(e1, 3),
            "kept_pct": round(100 * n1 / max(n0, 1), 1),
            "hit_lift": round(h1 - h0, 4) if np.isfinite(h1) and np.isfinite(h0) else None}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="processed_intraday")
    p.add_argument("--configs-dir", default="configs_intraday")
    p.add_argument("--symbols", nargs="*", help="default: all configs present")
    p.add_argument("--gate-q", type=float, default=0.5, help="vol-z quantile gate (default median)")
    args = p.parse_args(argv)

    by_sym = {s: pth for s, pth in data_loader.list_symbols(args.input)}
    seldir = os.path.join(args.configs_dir, "selected")
    syms = args.symbols or [os.path.splitext(os.path.basename(p))[0]
                            for p in sorted(glob.glob(os.path.join(seldir, "*.json")))]
    rows = []
    for sym in syms:
        cfgp = os.path.join(seldir, f"{sym}.json")
        if not os.path.exists(cfgp) or sym not in by_sym:
            continue
        r = run(by_sym[sym], json.load(open(cfgp, encoding="utf-8")), args.gate_q)
        if r:
            rows.append(r)
    if not rows:
        print("no results"); return 0
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    valid = df[df["hit_lift"].notna()]
    print(f"\nmean hit lift from spike-gating: {valid['hit_lift'].mean():+.4f} "
          f"| improved: {(valid['hit_lift'] > 0).sum()}/{len(valid)} "
          f"| avg signals kept: {df['kept_pct'].mean():.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
