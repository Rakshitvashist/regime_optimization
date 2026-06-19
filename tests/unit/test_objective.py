"""T012 - hybrid objective math (accuracy + ATR-relative return weighting)."""
import math

import numpy as np
import pandas as pd

from optimization import objective


def _perfect_buy_setup(n=80):
    idx = pd.bdate_range("2020-01-01", periods=n)
    price = pd.Series(100.0 + np.arange(n), index=idx)        # strictly increasing
    atr = pd.Series(1.0, index=idx)
    preds = pd.DataFrame(
        {"signal": 1.0, "confidence": 0.5, "consensus_score": 0.5}, index=idx
    )
    return preds, price, atr


def test_all_correct_gives_hit_ratio_one():
    preds, price, atr = _perfect_buy_setup()
    m = objective.evaluate(preds, price, atr, [5, 10], w_acc=1.0, w_ret=0.0)
    for d in ("5d", "10d"):
        assert m.hit_ratio[d] == 1.0
        assert m.signal_count[d] > 0
    # w_acc=1, w_ret=0 and all-correct -> score is exactly the hit-ratio (1.0)
    assert m.objective_score == 1.0


def test_return_component_is_tanh_of_atr_relative_return():
    preds, price, atr = _perfect_buy_setup()
    m = objective.evaluate(preds, price, atr, [5, 10], w_acc=0.0, w_ret=1.0)
    # forward moves are positive and signal=+1 -> positive ATR-relative return
    assert m.mean_directional_return["5d"] > 0
    # score == tanh(signal-count-weighted mean return)
    counts = m.signal_count
    total = counts["5d"] + counts["10d"]
    ret_mean = (m.mean_directional_return["5d"] * counts["5d"]
                + m.mean_directional_return["10d"] * counts["10d"]) / total
    assert math.isclose(m.objective_score, math.tanh(ret_mean), rel_tol=1e-9)


def test_weights_blend_components():
    preds, price, atr = _perfect_buy_setup()
    m = objective.evaluate(preds, price, atr, [5, 10], w_acc=0.5, w_ret=0.5)
    assert 0.5 <= m.objective_score <= 1.0  # 0.5*1.0 + 0.5*tanh(positive)
