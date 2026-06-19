"""
backtest_selected.py  —  out-of-sample backtest of predict_selected.py's calls.

For each stock it replays the selected-indicator vote (from configs/selected/) on
the HELD-OUT window (the embargoed out-of-sample slice — the indicators were
chosen on TRAIN, so this is an honest test) and reports:

  Directional (every firing bar):
    n_signals    how many bars the vote was non-zero
    hit_ratio    P(vote direction == realized forward direction)   [horizon = min]
    avg_edge%    mean of  direction * forward_return  per firing bar

  Traded equity (NON-overlapping: enter on a signal, hold `hold` bars, then flat):
    n_trades, win_rate, total_return%   (sum of direction * trade return)

This tells you whether the live calls are trustworthy BEFORE acting on them.

Usage:
  python backtest_selected.py --input processed_indicators
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from optimization import data_loader, splitter
import analyze_indicators as ai


def _vote_series(cand, selected):
    """Per-bar net vote across the selected indicators (orientation applied)."""
    T = len(cand)
    net = np.zeros(T)
    for s in selected:
        name = s["indicator"]
        if name in cand.columns:
            net += s["orientation"] * np.sign(cand[name].to_numpy(float))
    return np.sign(net)            # -1 / 0 / +1 per bar


def backtest_symbol(symbol, csv_path, cfg, hold=5):
    horizons = cfg.get("horizons", [5, 10])
    d = min(horizons)
    df = data_loader.load_symbol_frame(csv_path)
    cand = ai.build_candidate_frame(df, vp_window=cfg.get("vp_window", 120),
                                    vp_bins=cfg.get("vp_bins", 24))
    close = df["close"].to_numpy(float)
    T = len(close)
    win = splitter.make_window(symbol, df.index, horizons,
                               heldout_frac=0.3, embargo=None)
    if isinstance(win, splitter.InsufficientHistory):
        return None
    ho = range(*win.heldout_slice.indices(T))
    vote = _vote_series(cand, cfg.get("selected", []))

    # --- directional stats over every firing bar in held-out ----------------
    fwd = np.full(T, np.nan)
    fwd[:T - d] = (close[d:] - close[:T - d]) / (close[:T - d] + 1e-9)
    hits = edges = nsig = 0
    for t in ho:
        if t >= T - d or vote[t] == 0 or not np.isfinite(fwd[t]):
            continue
        nsig += 1
        edge = vote[t] * fwd[t]
        edges += edge
        hits += 1 if edge > 0 else 0
    hit_ratio = hits / nsig if nsig else float("nan")
    avg_edge = edges / nsig if nsig else float("nan")

    # --- non-overlapping traded equity --------------------------------------
    ho_list = list(ho)
    lo, hi = ho_list[0], ho_list[-1]
    t = lo
    trades = []
    while t <= hi - hold and t < T - hold:
        if vote[t] != 0:
            ret = vote[t] * (close[t + hold] - close[t]) / (close[t] + 1e-9)
            trades.append(ret)
            t += hold
        else:
            t += 1
    n_tr = len(trades)
    win_rate = float(np.mean([r > 0 for r in trades])) if trades else float("nan")
    total_ret = float(np.sum(trades)) if trades else 0.0

    return {
        "symbol": symbol,
        "reliability": cfg.get("ensemble_heldout_hit"),
        "n_signals": nsig,
        "hit_ratio": round(hit_ratio, 4) if nsig else None,
        "avg_edge_pct": round(avg_edge * 100, 3) if nsig else None,
        "n_trades": n_tr,
        "win_rate": round(win_rate, 4) if n_tr else None,
        "total_return_pct": round(total_ret * 100, 2),
        "heldout": f"{win.heldout_start.date()}..{win.heldout_end.date()}",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Backtest the selected-indicator calls.")
    p.add_argument("--input", default="processed_indicators")
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--out", default="runs/selected_backtest.csv")
    p.add_argument("--hold", type=int, default=5, help="bars to hold per trade")
    args = p.parse_args(argv)

    cfgs = sorted(glob.glob(os.path.join(args.configs_dir, "selected", "*.json")))
    if not cfgs:
        print(f"ERROR: no selection configs — run analyze_indicators.py first", file=sys.stderr)
        return 1
    by_sym = {s: pth for s, pth in data_loader.list_symbols(args.input)}

    rows = []
    for cp in cfgs:
        cfg = json.load(open(cp, encoding="utf-8"))
        sym = cfg["symbol"]
        if sym not in by_sym:
            continue
        r = backtest_symbol(sym, by_sym[sym], cfg, hold=args.hold)
        if r:
            rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No backtests produced.")
        return 0
    df = df.sort_values("total_return_pct", ascending=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 170); pd.set_option("display.max_rows", 80)
    print(df.to_string(index=False))
    valid = df[df["hit_ratio"].notna()]
    print(f"\n=== Universe ({len(df)} stocks) ===")
    print(f"  mean held-out hit ratio   : {valid['hit_ratio'].mean():.4f}")
    print(f"  mean avg edge per signal  : {valid['avg_edge_pct'].mean():.3f}%")
    print(f"  mean total return (hold={args.hold}): {df['total_return_pct'].mean():.2f}%")
    print(f"  profitable stocks         : {(df['total_return_pct'] > 0).sum()} / {len(df)}")
    print(f"  hit ratio > 0.52          : {(valid['hit_ratio'] > 0.52).sum()} / {len(valid)}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
