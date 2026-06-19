"""
Indicator Optimization package (feature 001-indicator-optimization).

Learns optimal indicator parameters and consensus weights from stored history
(hybrid accuracy + ATR-relative-return objective), with out-of-sample selection
and a forward-return embargo guaranteeing no look-ahead bias.

Heavy / optional dependencies (talib, optuna, scikit-learn) are imported lazily
inside the modules that need them, so the core logic (config, search space,
splitter, objective, weight optimizer, store) is importable and testable with
only numpy + pandas.
"""

__all__ = [
    "config",
    "search_space",
    "splitter",
    "objective",
    "data_loader",
    "weight_optimizer",
    "param_optimizer",
    "runner",
    "store",
    # Feature 002 — Deep Learning Indicator Optimizer
    "differentiable_indicators",
    "dl_optimizer",
    "regime_lstm",
    "dl_param_search",
    "liquidity_features",
    "volume_profile",
    "spike_features",
]
