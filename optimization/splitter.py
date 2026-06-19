"""
T006 - Walk-forward train / held-out splitter with a forward-return embargo.

Enforces Constitution Principle II (No Look-Ahead Bias, NON-NEGOTIABLE): the
held-out block begins at least `embargo` bars AFTER the training block ends,
where `embargo >= max(forward_days)`. This removes the calendar overlap between
forward-return labels at the tail of training and the start of held-out.

A symbol with too little history returns an InsufficientHistory result (FR-010)
instead of a window, so callers can skip it with a reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

# Minimum bars each block must retain to be meaningful.
MIN_TRAIN_BARS = 60
MIN_HELDOUT_BARS = 30


@dataclass
class EvaluationWindow:
    symbol: str
    index: pd.Index            # full bar index used
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    heldout_start: pd.Timestamp
    heldout_end: pd.Timestamp
    embargo_bars: int
    n_bars_total: int
    # positional slices into `index`
    train_slice: slice
    heldout_slice: slice


@dataclass
class InsufficientHistory:
    symbol: str
    reason: str
    have: int
    need: int


def min_required_bars(forward_days: List[int], longest_lookback: int = 0,
                      heldout_frac: float = 0.3) -> int:
    """Smallest history that yields valid train + embargo + held-out blocks."""
    embargo = max(forward_days)
    # held-out must also leave room for its own forward labels at its tail.
    base = longest_lookback + MIN_TRAIN_BARS + embargo + MIN_HELDOUT_BARS + embargo
    return base


def make_window(symbol: str, index: pd.Index, forward_days: List[int],
                heldout_frac: float = 0.3, embargo: Optional[int] = None,
                longest_lookback: int = 0):
    """Build an EvaluationWindow, or InsufficientHistory if there isn't enough data."""
    n = len(index)
    emb = int(max(forward_days) if embargo is None else embargo)
    if emb < max(forward_days):
        raise ValueError(
            f"embargo ({emb}) must be >= max(forward_days) ({max(forward_days)}) "
            "to prevent look-ahead leakage (Principle II)."
        )

    need = min_required_bars(forward_days, longest_lookback, heldout_frac)
    if n < need:
        return InsufficientHistory(
            symbol=symbol,
            reason=f"insufficient history (needs >= {need} bars, has {n})",
            have=n, need=need,
        )

    # Reserve the held-out tail; trim its end by `emb` so its forward labels exist.
    heldout_len = max(MIN_HELDOUT_BARS, int(round(n * heldout_frac)))
    heldout_end_pos = n - emb                 # last usable bar for held-out labels
    heldout_start_pos = heldout_end_pos - heldout_len
    train_end_pos = heldout_start_pos - emb   # embargo gap
    train_start_pos = longest_lookback        # skip warm-up region

    if train_end_pos - train_start_pos < MIN_TRAIN_BARS:
        return InsufficientHistory(
            symbol=symbol,
            reason=f"insufficient train bars after embargo (needs >= {MIN_TRAIN_BARS})",
            have=max(0, train_end_pos - train_start_pos), need=MIN_TRAIN_BARS,
        )

    train_slice = slice(train_start_pos, train_end_pos)
    heldout_slice = slice(heldout_start_pos, heldout_end_pos)

    return EvaluationWindow(
        symbol=symbol,
        index=index,
        train_start=index[train_start_pos],
        train_end=index[train_end_pos - 1],
        heldout_start=index[heldout_start_pos],
        heldout_end=index[heldout_end_pos - 1],
        embargo_bars=emb,
        n_bars_total=n,
        train_slice=train_slice,
        heldout_slice=heldout_slice,
    )
