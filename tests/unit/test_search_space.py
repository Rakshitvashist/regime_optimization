"""T037 - search-space bounds, clamping, and validation."""
from optimization import search_space
from optimization.config import IndicatorConfig, WeightConfig


def test_default_config_is_valid():
    assert search_space.validate(IndicatorConfig.default()) == []


def test_out_of_range_weight_is_flagged():
    cfg = IndicatorConfig.default()
    cfg.weights = WeightConfig(category={"trend": 99.0}, voting_threshold=0.15,
                               regime_multipliers={})
    problems = search_space.validate(cfg)
    assert any("category.trend" in p for p in problems)


def test_out_of_range_voting_threshold_is_flagged():
    cfg = IndicatorConfig.default()
    cfg.weights = WeightConfig(category={}, voting_threshold=5.0, regime_multipliers={})
    problems = search_space.validate(cfg)
    assert any("voting_threshold" in p for p in problems)


def test_clamp_respects_bounds_and_int_kind():
    e_int = search_space._BY_NAME["rsi_period"]
    assert search_space.clamp(e_int, 999) == e_int.high
    assert search_space.clamp(e_int, -5) == e_int.low
    assert isinstance(search_space.clamp(e_int, 14.7), int)
    e_float = search_space._BY_NAME["voting_threshold"]
    assert search_space.clamp(e_float, 10.0) == e_float.high


def test_entries_partition_by_recompute():
    params = search_space.entries("params")
    weights = search_space.entries("weights")
    assert all(e.affects_recompute for e in params)
    assert all(not e.affects_recompute for e in weights)
