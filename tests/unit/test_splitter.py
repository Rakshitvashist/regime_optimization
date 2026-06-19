"""T011 - splitter embargo / no-look-ahead at the structural level."""
import pandas as pd
import pytest

from optimization import splitter


def _index(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_heldout_starts_after_train_plus_embargo():
    fwd = [5, 10]
    win = splitter.make_window("SYM", _index(400), fwd, heldout_frac=0.3)
    assert not isinstance(win, splitter.InsufficientHistory)
    # embargo must be >= max horizon (Principle II)
    assert win.embargo_bars >= max(fwd)
    # positional gap between train end and held-out start must be >= embargo
    gap = win.heldout_slice.start - win.train_slice.stop
    assert gap >= win.embargo_bars
    # blocks must not overlap
    assert win.train_slice.stop <= win.heldout_slice.start


def test_explicit_embargo_below_max_horizon_is_rejected():
    with pytest.raises(ValueError):
        splitter.make_window("SYM", _index(400), [5, 10], embargo=3)


def test_insufficient_history_is_reported_not_raised():
    res = splitter.make_window("TINY", _index(40), [5, 10])
    assert isinstance(res, splitter.InsufficientHistory)
    assert "insufficient history" in res.reason
    assert res.have == 40 and res.need > 40
