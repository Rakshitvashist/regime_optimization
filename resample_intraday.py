"""
resample_intraday.py  —  resample 1-min OHLCV to multiple timeframes.

Input  : 1-min CSV with columns DateTime,Open,High,Low,Close,Volume
          (DateTime tz-aware IST, e.g. 2016-10-03 09:15:00+05:30)
Output : intraday/<NAME>_<TF>.csv  for TF in 5m,15m,30m,1h,4h,1d

Session safety: the NSE/BSE session is 09:15–15:30 with a ~17.75h overnight gap.
Every timeframe here is <= 4h < the gap, so clock-based resampling NEVER merges
two days into one bar — empty overnight bins are simply dropped. So no special
per-day grouping is needed and there is no cross-session leakage.

Aggregation: Open=first, High=max, Low=min, Close=last, Volume=sum.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

# label -> pandas resample rule
TIMEFRAMES = {
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "60min",
    "4h":  "240min",
    "1d":  "1D",
}
AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}


def load_1min(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    dtcol = next(c for c in df.columns if c.lower() in ("datetime", "date"))
    df[dtcol] = pd.to_datetime(df[dtcol], utc=True).dt.tz_convert("Asia/Kolkata")
    df = df.set_index(dtcol).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule == "1D":
        # one bar per calendar trading day
        out = df.resample("1D").agg(AGG)
    else:
        out = df.resample(rule, label="left", closed="left").agg(AGG)
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def main(argv=None):
    p = argparse.ArgumentParser(description="Resample 1-min OHLCV to multiple timeframes.")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="1-min CSV files (e.g. RELIANCE.csv BANKNIFTY.csv ...)")
    p.add_argument("--out", default="intraday")
    p.add_argument("--timeframes", nargs="+", default=list(TIMEFRAMES.keys()),
                   help=f"subset of {list(TIMEFRAMES.keys())}")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    for path in args.inputs:
        name = os.path.splitext(os.path.basename(path))[0].upper()
        df = load_1min(path)
        has_vol = df["Volume"].abs().sum() > 0
        print(f"{name:10s} 1m rows={len(df):,} | {df.index[0].date()}..{df.index[-1].date()} | "
              f"volume={'YES' if has_vol else 'none (index)'}")
        for tf in args.timeframes:
            rule = TIMEFRAMES[tf]
            r = resample(df, rule)
            r.index.name = "DateTime"
            outp = os.path.join(args.out, f"{name}_{tf}.csv")
            r.to_csv(outp)
            print(f"   {tf:3s} -> {len(r):>8,} bars  {outp}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
