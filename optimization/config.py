"""
T004 - Configuration dataclasses.

`IndicatorConfig` holds the full tunable surface: indicator parameters plus the
consensus weights. `IndicatorConfig.default()` reproduces EXACTLY the values
currently hardcoded in `start.py` and `Consensus_predictor .py`, so a default
config yields today's behaviour (regression-safe). Sparse overrides merge onto
the default. JSON (de)serialization matches contracts/tuned-config.schema.json.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


# --- Defaults mirrored from the existing code (do not drift) -----------------
# Consensus_predictor .py: cat_weights, voting_threshold, regime multipliers.
_DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "trend": 1.2,
    "price_action": 1.1,
    "momentum": 1.0,
    "volatility": 0.8,
    "volume": 0.9,
    "other": 0.7,
}
_DEFAULT_VOTING_THRESHOLD = 0.15
# predict(): trend quality *= (1 + 0.2*regime); momentum *= 1.2 when ranging.
_DEFAULT_REGIME_MULTIPLIERS: Dict[str, float] = {
    "trend_trending_boost": 0.2,
    "momentum_ranging_boost": 0.2,
}
# start.py: representative tunable indicator parameters (see search_space.py).
_DEFAULT_INDICATOR_PARAMS: Dict[str, float] = {
    "rsi_period": 14,
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14,
    "supertrend_mult": 3.0,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "ma_fast": 20,
    "ma_slow": 50,
}


@dataclass
class WeightConfig:
    category: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_CATEGORY_WEIGHTS))
    voting_threshold: float = _DEFAULT_VOTING_THRESHOLD
    regime_multipliers: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_REGIME_MULTIPLIERS)
    )

    def to_dict(self) -> dict:
        return {
            "category": dict(self.category),
            "voting_threshold": self.voting_threshold,
            "regime_multipliers": dict(self.regime_multipliers),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeightConfig":
        return cls(
            category=dict(d.get("category", _DEFAULT_CATEGORY_WEIGHTS)),
            voting_threshold=float(d.get("voting_threshold", _DEFAULT_VOTING_THRESHOLD)),
            regime_multipliers=dict(d.get("regime_multipliers", _DEFAULT_REGIME_MULTIPLIERS)),
        )


@dataclass
class IndicatorConfig:
    indicator_params: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_INDICATOR_PARAMS)
    )
    weights: WeightConfig = field(default_factory=WeightConfig)
    is_default: bool = False
    # --- DL optimization metadata (Feature 002) ---
    # Path to the saved PyTorch checkpoint (.pt) for this config.
    dl_model_path: Optional[str] = None
    # The indicator parameter values discovered by the DL model (for logging/audit).
    dl_params_discovered: Optional[Dict[str, float]] = None
    # DL method used: None / 'dl' / 'lstm' / 'regime_lstm'
    dl_method: Optional[str] = None

    # --- factories -----------------------------------------------------------
    @classmethod
    def default(cls) -> "IndicatorConfig":
        """The current production behaviour, as a config."""
        return cls(
            indicator_params=dict(_DEFAULT_INDICATOR_PARAMS),
            weights=WeightConfig(),
            is_default=True,
        )

    # --- serialization -------------------------------------------------------
    def to_dict(self) -> dict:
        d = {
            "indicator_params": dict(self.indicator_params),
            "weights": self.weights.to_dict(),
            "is_default": self.is_default,
        }
        # Only include DL fields when present (backward-compatible)
        if self.dl_model_path is not None:
            d["dl_model_path"] = self.dl_model_path
        if self.dl_params_discovered is not None:
            d["dl_params_discovered"] = dict(self.dl_params_discovered)
        if self.dl_method is not None:
            d["dl_method"] = self.dl_method
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "IndicatorConfig":
        return cls(
            indicator_params=dict(d.get("indicator_params", _DEFAULT_INDICATOR_PARAMS)),
            weights=WeightConfig.from_dict(d.get("weights", {})),
            is_default=bool(d.get("is_default", False)),
            dl_model_path=d.get("dl_model_path"),
            dl_params_discovered=d.get("dl_params_discovered"),
            dl_method=d.get("dl_method"),
        )

    # --- sparse override merge ----------------------------------------------
    def merge_with_default(self) -> "IndicatorConfig":
        """Return a dense config: this config's values overlaid on the defaults.

        Keys absent here inherit the default value (FR-005 sparse overrides).
        """
        base = IndicatorConfig.default()
        params = dict(base.indicator_params)
        params.update(self.indicator_params or {})
        cat = dict(base.weights.category)
        cat.update(self.weights.category or {})
        regime = dict(base.weights.regime_multipliers)
        regime.update(self.weights.regime_multipliers or {})
        vt = self.weights.voting_threshold
        return IndicatorConfig(
            indicator_params=params,
            weights=WeightConfig(
                category=cat,
                voting_threshold=vt if vt is not None else base.weights.voting_threshold,
                regime_multipliers=regime,
            ),
            is_default=False,
        )

    def copy(self) -> "IndicatorConfig":
        return copy.deepcopy(self)
