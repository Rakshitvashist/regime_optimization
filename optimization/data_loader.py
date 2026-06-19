"""
T008 - Load stored data into predictor inputs (read-only; Principle I).

Mirrors the input construction in generate_summary.py (indicators_df, quality_df,
price_series, atr_series, adx_series) but computes ATR with a pure-pandas
fallback so the weights-only optimization path needs neither TA-Lib nor live
data. TA-Lib is imported lazily only if an indicator recompute is requested.

Inputs are treated strictly read-only — nothing here writes to OHLCV/indicator
files.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def list_symbols(input_dir: str) -> List[Tuple[str, str]]:
    """Return (symbol, csv_path) for every *_with_indicators.csv in input_dir."""
    out = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*_with_indicators.csv"))):
        symbol = os.path.basename(path).split("_")[0]
        out.append((symbol, path))
    return out


def load_symbol_frame(csv_path: str) -> pd.DataFrame:
    """Load one symbol CSV -> date-indexed, lowercased, de-duplicated frame."""
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df = df[~df.index.duplicated(keep="last")]
    return df


def indicator_columns(df: pd.DataFrame) -> List[str]:
    """Non-OHLCV *numeric* columns are treated as indicator signals.

    Mirrors generate_summary._indicator_cols: metadata text columns
    (symbol/company/industry/index, etc.) and any other non-numeric columns are
    excluded so the float cast in build_inputs cannot fail on string values.
    """
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    return [c for c in df.columns if c not in OHLCV_COLS and c in numeric]


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-style ATR via pure pandas (no TA-Lib dependency)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def build_inputs(df: pd.DataFrame, atr_period: int = 14):
    """Return (indicators_df, quality_df, price_series, atr_series, adx_series).

    quality_df defaults to 0.5 everywhere, matching generate_summary.py's
    starting quality matrix. adx_series is None -> the predictor uses its
    price-distance regime fallback (talib-free).
    """
    df = df.copy()
    df = df.fillna(0)
    ind_cols = indicator_columns(df)
    indicators_df = df[ind_cols].astype(float)
    quality_df = pd.DataFrame(
        np.full(indicators_df.shape, 0.5),
        index=indicators_df.index,
        columns=indicators_df.columns,
    )
    price_series = df["close"].astype(float)
    atr_series = compute_atr(df, atr_period)
    adx_series = None
    return indicators_df, quality_df, price_series, atr_series, adx_series
