"""
T007 - Hybrid objective (FR-004).

    score = w_acc * directional_hit_ratio + w_ret * tanh(mean_atr_relative_return)

- Directional hit-ratio comes from the trusted `HitRatioBacktester` (Principle III).
- The return component is the mean DIRECTIONAL price move expressed in ATR units
  (scale-free across symbols, R4), squashed with tanh so it is comparable in
  magnitude to the [0,1] hit-ratio before weighting.
- Both components are aggregated across horizons by SIGNAL-COUNT-weighted mean.

`evaluate()` is pure (numpy/pandas + HitRatioBacktester) and takes an already
computed predictions frame, so it is unit-testable without TA-Lib / scikit-learn.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import _legacy


@dataclass
class Metrics:
    hit_ratio: Dict[str, float] = field(default_factory=dict)
    mean_directional_return: Dict[str, float] = field(default_factory=dict)  # ATR units
    signal_count: Dict[str, int] = field(default_factory=dict)
    objective_score: float = 0.0
    train_score: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "hit_ratio": dict(self.hit_ratio),
            "mean_directional_return": dict(self.mean_directional_return),
            "signal_count": dict(self.signal_count),
            "objective_score": self.objective_score,
        }
        if self.train_score is not None:
            d["train_score"] = self.train_score
        return d


def _atr_relative_return(predictions: pd.DataFrame, price: pd.Series,
                         atr: pd.Series, days: int) -> (float, int):
    """Mean (signal * forward_abs_move / ATR) over actionable signals; ATR units."""
    actionable = predictions[predictions["signal"] != 0]
    if len(actionable) == 0:
        return 0.0, 0
    fwd_change = price.shift(-days) - price
    dr = (actionable["signal"] * fwd_change.reindex(actionable.index)) / \
        atr.reindex(actionable.index).replace(0.0, np.nan)
    dr = dr.replace([np.inf, -np.inf], np.nan).dropna()
    if len(dr) == 0:
        return 0.0, 0
    return float(dr.mean()), int(len(dr))


def evaluate(predictions: pd.DataFrame, price: pd.Series, atr: pd.Series,
             forward_days: List[int], w_acc: float = 0.5, w_ret: float = 0.5) -> Metrics:
    """Score a predictions frame on a window. Higher is better."""
    backtester = _legacy.load_backtester_cls()(forward_days=forward_days)
    stats = backtester.backtest(predictions, price)

    hit_ratio: Dict[str, float] = {}
    ret: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for d in forward_days:
        key = f"{d}d"
        hit_ratio[key] = float(stats.get(f"overall_{d}d", 0.5))
        counts[key] = int(stats.get(f"signal_count_{d}d", 0))
        r, _ = _atr_relative_return(predictions, price, atr, d)
        ret[key] = r

    total = sum(counts.values())
    if total > 0:
        hit_mean = sum(hit_ratio[f"{d}d"] * counts[f"{d}d"] for d in forward_days) / total
        ret_mean = sum(ret[f"{d}d"] * counts[f"{d}d"] for d in forward_days) / total
    else:
        hit_mean, ret_mean = 0.5, 0.0

    score = w_acc * hit_mean + w_ret * math.tanh(ret_mean)
    return Metrics(
        hit_ratio=hit_ratio,
        mean_directional_return=ret,
        signal_count=counts,
        objective_score=float(score),
    )
