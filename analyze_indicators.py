"""
analyze_indicators.py  —  per-stock indicator hit-ratio analysis (all 50 / 500).

Answers, FOR EACH STOCK, the question "which couple of indicators actually give
a good hit ratio?" — cleanly (no look-ahead), so they can drive prediction.

Pipeline per symbol (no look-ahead, Principle II):
  1. Walk-forward split with a forward-return embargo (reuses optimization.splitter).
  2. TRAIN window: rank every indicator by directional hit ratio. An indicator
     whose TRAIN hit ratio is < 0.5 is *inverted* (used as a contrarian signal),
     so its edge = |hit - 0.5|. Pick the top-K by edge (with a min fire-rate).
  3. HELD-OUT window: validate the selected top-K with their TRAIN-decided
     orientation, and a simple majority-vote ENSEMBLE of them. Selection is on
     train, scoring on held-out — the held-out number is the trustworthy one.
  4. Rally precursors: which indicators fire BEFORE a big up/down move
     (fwd 10d move beyond +/- `rally_atr` * ATR) — "behaviour before a rally".
  5. Liquidity-sweep CONDITIONAL edge: P(up | bullish sweep + volume surge) vs
     the base rate, and the same for bearish sweeps.

Outputs:
  <out>/<SYMBOL>.json     per-stock detail
  <out>/summary.csv       universe table (held-out edge per stock)
  console                 ranked summary

Pure numpy/pandas on the stored indicator CSVs — no TA-Lib / torch / GPU needed,
so it runs on all 50 in well under a minute.

Usage:
  python analyze_indicators.py --input processed_indicators --top-k 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from optimization import data_loader, splitter, liquidity_features as lf
from optimization import volume_profile as vp
from optimization import spike_features as sf

OHLCV = {"open", "high", "low", "close", "volume"}


def build_candidate_frame(df: pd.DataFrame, vp_window: int = 120,
                          vp_bins: int = 24) -> pd.DataFrame:
    """Stored indicator columns + causal liquidity + volume-profile features.

    Shared by the analysis (ranking) and predict_selected.py (inference), so the
    selected indicators resolve to the exact same columns at prediction time.
    """
    ind = df[data_loader.indicator_columns(df)].astype(float)
    ex = lf.extra_features(df, lookback=10)
    vpf = vp.profile_features(df, window=vp_window, n_bins=vp_bins)
    spk = sf.spike_features(df)
    extra_df = pd.concat([
        pd.DataFrame(ex, index=df.index, columns=lf.FEATURE_NAMES),
        pd.DataFrame(vpf, index=df.index, columns=vp.FEATURE_NAMES),
        pd.DataFrame(spk, index=df.index, columns=sf.FEATURE_NAMES),
    ], axis=1)
    out = pd.concat([ind, extra_df], axis=1).replace([np.inf, -np.inf], np.nan)
    # Drop dead columns: all-NaN or (near-)constant. On indices (volume=0) the
    # volume indicators (obv_*, price_volume_corr, ...) become NaN/constant and
    # would otherwise pollute the ranking — this removes them automatically.
    keep = [c for c in out.columns
            if out[c].notna().any() and float(out[c].std(skipna=True) or 0.0) > 1e-12]
    return out[keep]


def _forward_returns(close: np.ndarray, horizons):
    """dict d -> (signed fwd return array, NaN at tail)."""
    out = {}
    T = len(close)
    for d in horizons:
        r = np.full(T, np.nan, dtype=np.float64)
        r[: T - d] = (close[d:] - close[: T - d]) / (close[: T - d] + 1e-9)
        out[d] = r
    return out


def _hit_and_count(sig_col: np.ndarray, fwd: dict):
    """Mean directional hit ratio + valid-signal count across horizons."""
    s = np.sign(sig_col)
    hits = cnt = 0.0
    for d, r in fwd.items():
        rs = np.sign(r)
        valid = (s != 0) & np.isfinite(s) & np.isfinite(r)
        cnt += valid.sum()
        hits += ((s == rs) & valid).sum()
    return (hits / cnt if cnt else 0.5), cnt


def analyze_symbol(symbol, csv_path, horizons, heldout_frac, embargo,
                   top_k, min_fire, rally_atr, vp_window=120, vp_bins=24):
    df = data_loader.load_symbol_frame(csv_path)
    if "close" not in df.columns:
        return {"symbol": symbol, "status": "skipped", "reason": "no close column"}

    ind = build_candidate_frame(df, vp_window=vp_window, vp_bins=vp_bins)

    close = df["close"].to_numpy(float)
    atr = data_loader.compute_atr(df).to_numpy(float)

    win = splitter.make_window(symbol, df.index, horizons,
                               heldout_frac=heldout_frac, embargo=embargo)
    if isinstance(win, splitter.InsufficientHistory):
        return {"symbol": symbol, "status": "skipped", "reason": win.reason}

    tr, ho = win.train_slice, win.heldout_slice
    cols = list(ind.columns)
    M = ind.to_numpy()
    n_train = len(range(*tr.indices(len(df))))

    fwd_tr = _forward_returns(close, horizons)
    fwd_ho = _forward_returns(close, horizons)
    # restrict each horizon array to the window via masking (keep index alignment)
    def _slice_fwd(fwd, sl):
        idx = np.zeros(len(close), dtype=bool); idx[sl] = True
        return {d: np.where(idx, r, np.nan) for d, r in fwd.items()}
    fwd_tr = _slice_fwd(fwd_tr, tr)
    fwd_ho = _slice_fwd(fwd_ho, ho)

    # ---- TRAIN: rank indicators by edge ------------------------------------
    ranked = []
    fire_floor = min_fire * n_train * len(horizons)
    for j, name in enumerate(cols):
        hit, cnt = _hit_and_count(M[:, j], fwd_tr)
        if cnt < fire_floor:
            continue
        orient = 1 if hit >= 0.5 else -1
        ranked.append((name, hit, abs(hit - 0.5), orient, cnt))
    ranked.sort(key=lambda x: x[2], reverse=True)
    top = ranked[:top_k]

    # ---- HELD-OUT: validate each + ensemble --------------------------------
    selected = []
    vote = np.zeros(len(close))
    for name, hit_tr, edge_tr, orient, _ in top:
        j = cols.index(name)
        hit_ho, cnt_ho = _hit_and_count(orient * M[:, j], fwd_ho)
        selected.append({
            "indicator": name, "orientation": int(orient),
            "train_hit": round(hit_tr, 4), "heldout_hit": round(hit_ho, 4),
            "heldout_signals": int(cnt_ho),
        })
        vote += orient * np.sign(M[:, j])
    ens_hit, ens_cnt = _hit_and_count(vote, fwd_ho)

    # ---- Rally precursors (big forward 10d move) ---------------------------
    d_rally = max(horizons)
    fr = (close[d_rally:] - close[:-d_rally]) / (close[:-d_rally] + 1e-9)
    fr = np.concatenate([fr, np.full(d_rally, np.nan)])
    atr_rel = atr / (close + 1e-9)
    up = fr > rally_atr * atr_rel
    dn = fr < -rally_atr * atr_rel
    rally = {"up_rate": round(float(np.nanmean(up)), 4),
             "down_rate": round(float(np.nanmean(dn)), 4)}
    precursors = {"up": [], "down": []}
    for direction, mask in (("up", up), ("down", dn)):
        base = np.nanmean(mask)
        lifts = []
        for name, *_ in top:
            j = cols.index(name)
            sig_on = np.sign(M[:, j]) == (1 if direction == "up" else -1)
            both = sig_on & np.isfinite(fr)
            if both.sum() < 30:
                continue
            p = mask[both].mean()
            lifts.append((name, round(float(p), 4), round(float(p - base), 4)))
        lifts.sort(key=lambda x: x[2], reverse=True)
        precursors[direction] = [{"indicator": n, "p_rally": p, "lift": l}
                                 for n, p, l in lifts[:3]]

    # ---- Liquidity-sweep conditional edge ----------------------------------
    # vol_surge is dropped on volume-less indices -> default to zeros (no volume
    # confirmation possible there; the price-based sweep event still works).
    sweep = (ind["liq_sweep_event"].to_numpy(float)
             if "liq_sweep_event" in ind.columns else np.zeros(len(ind)))
    vsurge = (ind["vol_surge"].to_numpy(float)
              if "vol_surge" in ind.columns else np.zeros(len(ind)))
    f5 = fwd_ho[min(horizons)]
    base_up = float(np.nanmean(np.sign(_forward_returns(close, [min(horizons)])[min(horizons)]) > 0))
    def _cond(mask):
        m = mask & np.isfinite(f5)
        if m.sum() < 10:
            return None
        return {"n": int(m.sum()),
                "p_up": round(float((f5[m] > 0).mean()), 4),
                "mean_fwd": round(float(np.nanmean(f5[m])), 5)}
    sweep_edge = {
        "base_p_up": round(base_up, 4),
        "bull_sweep_volup": _cond((sweep > 0) & (vsurge > 0)),
        "bear_sweep_volup": _cond((sweep < 0) & (vsurge > 0)),
    }

    return {
        "symbol": symbol, "status": "ok",
        "window": {"train_end": str(win.train_end.date()),
                   "heldout_start": str(win.heldout_start.date()),
                   "heldout_end": str(win.heldout_end.date()),
                   "embargo_bars": win.embargo_bars},
        "top_indicators": selected,
        "ensemble_heldout_hit": round(ens_hit, 4),
        "ensemble_signals": int(ens_cnt),
        "rally": rally,
        "rally_precursors": precursors,
        "sweep_edge": sweep_edge,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Per-stock indicator hit-ratio analysis.")
    p.add_argument("--input", default="processed_indicators")
    p.add_argument("--out", default="runs/indicator_analysis")
    p.add_argument("--configs-dir", default="configs",
                   help="writes per-stock selected-indicator configs to <dir>/selected/")
    p.add_argument("--forward-days", default="5,10")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--heldout-frac", type=float, default=0.3)
    p.add_argument("--embargo", type=int, default=None)
    p.add_argument("--min-fire", type=float, default=0.05)
    p.add_argument("--rally-atr", type=float, default=1.5,
                   help="rally = fwd move beyond this many ATRs")
    p.add_argument("--vp-window", type=int, default=120,
                   help="volume-profile lookback window in bars (default 120 ≈ 6 months)")
    p.add_argument("--vp-bins", type=int, default=24,
                   help="volume-profile price bins (default 24)")
    args = p.parse_args(argv)

    horizons = [int(x) for x in args.forward_days.split(",") if x.strip()]
    if not os.path.isdir(args.input):
        print(f"ERROR: input dir not found: {args.input}", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)
    sel_dir = os.path.join(args.configs_dir, "selected")
    os.makedirs(sel_dir, exist_ok=True)
    symbols = data_loader.list_symbols(args.input)
    if not symbols:
        print(f"ERROR: no *_with_indicators.csv in {args.input}", file=sys.stderr)
        return 1

    rows = []
    print(f"Analyzing {len(symbols)} symbols | horizons {horizons} | top-{args.top_k}\n")
    for sym, path in symbols:
        res = analyze_symbol(sym, path, horizons, args.heldout_frac, args.embargo,
                             args.top_k, args.min_fire, args.rally_atr,
                             vp_window=args.vp_window, vp_bins=args.vp_bins)
        with open(os.path.join(args.out, f"{sym}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        if res["status"] != "ok":
            print(f"  {sym:14s} SKIPPED ({res['reason']})")
            continue
        names = ", ".join(d["indicator"] for d in res["top_indicators"][:3])
        print(f"  {sym:14s} ensemble_HO={res['ensemble_heldout_hit']:.3f} | "
              f"top: {names}")
        for d in res["top_indicators"]:
            rows.append({"symbol": sym, **d, "ensemble_heldout_hit": res["ensemble_heldout_hit"]})

        # Persist a compact selection config the predictor votes on (item 2).
        sel_cfg = {
            "symbol": sym, "horizons": horizons,
            "vp_window": args.vp_window, "vp_bins": args.vp_bins,
            "ensemble_heldout_hit": res["ensemble_heldout_hit"],
            "selected": [{"indicator": d["indicator"], "orientation": d["orientation"],
                          "train_hit": d["train_hit"], "heldout_hit": d["heldout_hit"]}
                         for d in res["top_indicators"]],
            "window": res["window"],
        }
        with open(os.path.join(sel_dir, f"{sym}.json"), "w", encoding="utf-8") as f:
            json.dump(sel_cfg, f, indent=2)

    if rows:
        summary = pd.DataFrame(rows)
        summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)

        print("\n=== Indicators most often selected across the universe ===")
        agg = (summary.groupby("indicator")
               .agg(times_selected=("symbol", "count"),
                    mean_heldout_hit=("heldout_hit", "mean"))
               .sort_values("times_selected", ascending=False).head(20))
        print(agg.to_string(formatters={"mean_heldout_hit": "{:.4f}".format}))

        ok = summary.drop_duplicates("symbol")
        print(f"\nMean ensemble held-out hit ratio across stocks: "
              f"{ok['ensemble_heldout_hit'].mean():.4f}")
        print(f"Stocks with ensemble held-out hit > 0.52: "
              f"{(ok['ensemble_heldout_hit'] > 0.52).sum()} / {len(ok)}")
        print(f"\nWrote per-stock JSON + summary.csv to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
