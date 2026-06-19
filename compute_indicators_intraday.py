"""
compute_indicators_intraday.py  —  indicators on resampled intraday bars.

Reads intraday/<NAME>_<TF>.csv (from resample_intraday.py), runs the existing
IndicatorLibrary (start.py, ~400 TA-Lib indicators), and writes
processed_intraday/<NAME>-<TF>_with_indicators.csv.

The "<NAME>-<TF>" naming (hyphen, not underscore) keeps each instrument+timeframe
a DISTINCT symbol for analyze_indicators / predict_selected / backtest_selected
(they split the symbol on '_').

Usage:
  python compute_indicators_intraday.py --input intraday --out processed_intraday
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

from optimization import _legacy

OHLCV = ["open", "high", "low", "close", "volume"]


def main(argv=None):
    p = argparse.ArgumentParser(description="Compute indicators on resampled intraday bars.")
    p.add_argument("--input", default="intraday")
    p.add_argument("--out", default="processed_intraday")
    p.add_argument("--glob", default="*.csv", help="filter, e.g. 'BANKNIFTY_*.csv'")
    args = p.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.input, args.glob)))
    if not files:
        print(f"ERROR: no files matching {args.input}/{args.glob}", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)

    IndicatorLibrary = _legacy.load_indicator_library_cls()
    lib = IndicatorLibrary()

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]          # e.g. RELIANCE_5m
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        dtcol = next(c for c in df.columns if c in ("datetime", "date"))
        df[dtcol] = pd.to_datetime(df[dtcol], utc=True)
        df = df.set_index(dtcol)
        for c in OHLCV:
            if c not in df.columns:
                df[c] = 0.0
        try:
            ind = lib.calculate_all_indicators(df.copy())
            ind = ind.reindex(df.index)
            out = pd.concat([df[OHLCV], ind], axis=1)
            out.index.name = "Date"
            sym = name.replace("_", "-")                            # RELIANCE-5m
            outp = os.path.join(args.out, f"{sym}_with_indicators.csv")
            out.to_csv(outp)
            print(f"  {name:18s} {len(out):>8,} bars, {ind.shape[1]} indicators -> {outp}")
        except Exception as exc:
            print(f"  {name:18s} FAILED: {type(exc).__name__}: {exc}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
