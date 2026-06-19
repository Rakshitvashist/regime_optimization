"""T013 - no look-ahead bias (Constitution Principle II, NON-NEGOTIABLE)."""
import numpy as np
import pandas as pd

from optimization import _legacy, data_loader, splitter
from tests.conftest import make_ohlcv, make_indicators


def test_embargo_gap_prevents_label_overlap():
    win = splitter.make_window("SYM", pd.bdate_range("2020-01-01", periods=500),
                               [5, 10], heldout_frac=0.3)
    # No training forward-return label (max horizon = 10) can reach into held-out.
    gap = win.heldout_slice.start - win.train_slice.stop
    assert gap >= max([5, 10])


def test_perturbing_future_does_not_change_a_past_signal():
    ohlcv = make_ohlcv(n=300, seed=3)
    indicators = make_indicators(ohlcv, seed=3)
    qual = pd.DataFrame(0.5, index=indicators.index, columns=indicators.columns)
    price = ohlcv["close"].astype(float)
    atr = data_loader.compute_atr(ohlcv, 14)

    Consensus = _legacy.load_consensus_module().ConsensusPredictor
    predictor = Consensus()

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        base = predictor.predict(indicators, qual, price, atr, None)

    i = 150  # a decision bar in the middle
    # Perturb ALL data strictly AFTER bar i.
    pert = indicators.copy()
    rng = np.random.default_rng(99)
    pert.iloc[i + 1:] = rng.uniform(-1, 1, pert.iloc[i + 1:].shape)
    with contextlib.redirect_stdout(io.StringIO()):
        after = predictor.predict(pert, qual, price, atr, None)

    # The signal at bar i must be unchanged by anything that happens after it.
    assert base["signal"].iloc[i] == after["signal"].iloc[i]
    assert np.isclose(base["consensus_score"].iloc[i], after["consensus_score"].iloc[i])
