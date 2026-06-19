"""
T005 - Bounded, documented tunable surface.

Each `SpaceEntry` declares a tunable name, its kind, allowed range/choices, and
whether changing it requires recomputing indicators (`affects_recompute=True`
for indicator parameters) or only re-running the predictor (`False` for
weights). `validate()` rejects out-of-range values so every search candidate is
auditable (Principle IV; contracts/tuned-config.schema.json).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SpaceEntry:
    name: str
    kind: str  # "int" | "float" | "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[Tuple] = None
    affects_recompute: bool = False
    group: str = ""  # "indicator_param" | "weight.category" | "weight.scalar"


# Indicator parameters (changing these requires an indicator recompute).
INDICATOR_PARAM_SPACE: List[SpaceEntry] = [
    SpaceEntry("rsi_period", "int", 5, 30, affects_recompute=True, group="indicator_param"),
    SpaceEntry("bb_period", "int", 10, 40, affects_recompute=True, group="indicator_param"),
    SpaceEntry("bb_std", "float", 1.5, 3.0, affects_recompute=True, group="indicator_param"),
    SpaceEntry("atr_period", "int", 7, 30, affects_recompute=True, group="indicator_param"),
    SpaceEntry("supertrend_mult", "float", 1.5, 5.0, affects_recompute=True, group="indicator_param"),
    SpaceEntry("ma_fast", "int", 5, 50, affects_recompute=True, group="indicator_param"),
    SpaceEntry("ma_slow", "int", 50, 200, affects_recompute=True, group="indicator_param"),
]

# Consensus weights (changing these only re-runs the predictor; no recompute).
_CATEGORIES = ["trend", "price_action", "momentum", "volatility", "volume", "other"]
WEIGHT_SPACE: List[SpaceEntry] = (
    [SpaceEntry(f"category.{c}", "float", 0.0, 3.0, affects_recompute=False, group="weight.category")
     for c in _CATEGORIES]
    + [
        SpaceEntry("voting_threshold", "float", 0.0, 0.6, affects_recompute=False, group="weight.scalar"),
        SpaceEntry("trend_trending_boost", "float", 0.0, 1.0, affects_recompute=False, group="weight.scalar"),
        SpaceEntry("momentum_ranging_boost", "float", 0.0, 1.0, affects_recompute=False, group="weight.scalar"),
    ]
)

FULL_SPACE: List[SpaceEntry] = INDICATOR_PARAM_SPACE + WEIGHT_SPACE
_BY_NAME: Dict[str, SpaceEntry] = {e.name: e for e in FULL_SPACE}


def entries(kind: str = "all") -> List[SpaceEntry]:
    """Return space entries: 'all', 'params' (recompute), or 'weights'."""
    if kind == "params":
        return list(INDICATOR_PARAM_SPACE)
    if kind == "weights":
        return list(WEIGHT_SPACE)
    return list(FULL_SPACE)


def clamp(entry: SpaceEntry, value: float) -> float:
    """Clamp a numeric value into the entry's bounds (ints rounded)."""
    v = float(value)
    if entry.low is not None:
        v = max(entry.low, v)
    if entry.high is not None:
        v = min(entry.high, v)
    return int(round(v)) if entry.kind == "int" else v


def validate(config) -> List[str]:
    """Return a list of violation messages for an IndicatorConfig (empty == valid)."""
    dense = config.merge_with_default()
    problems: List[str] = []

    for name, val in dense.indicator_params.items():
        e = _BY_NAME.get(name)
        if e is None:
            continue  # default-only params not in the tunable surface are allowed as-is
        if e.low is not None and (val < e.low or val > e.high):
            problems.append(f"{name}={val} out of range [{e.low}, {e.high}]")

    for cat, w in dense.weights.category.items():
        e = _BY_NAME.get(f"category.{cat}")
        if e and (w < e.low or w > e.high):
            problems.append(f"category.{cat}={w} out of range [{e.low}, {e.high}]")

    vt = dense.weights.voting_threshold
    e = _BY_NAME["voting_threshold"]
    if vt < e.low or vt > e.high:
        problems.append(f"voting_threshold={vt} out of range [{e.low}, {e.high}]")

    for name, val in dense.weights.regime_multipliers.items():
        e = _BY_NAME.get(name)
        if e and (val < e.low or val > e.high):
            problems.append(f"{name}={val} out of range [{e.low}, {e.high}]")
    return problems
