"""NN extended to the global config: one pooled per-indicator model across symbols."""
import pandas as pd

from optimization import batch, data_loader
from tests.conftest import make_ohlcv, make_indicators


def _write(directory, symbol, n, seed):
    o = make_ohlcv(n=n, seed=seed)
    df = o.join(make_indicators(o, seed=seed))
    df.index.name = "date"
    df.reset_index().to_csv(directory / f"{symbol}_with_indicators.csv", index=False)


def _universe(tmp_path):
    d = tmp_path / "processed_indicators"
    d.mkdir()
    _write(d, "AAA", 400, 1)
    _write(d, "BBB", 400, 2)
    _write(d, "CCC", 400, 3)
    return data_loader.list_symbols(str(d))


def test_global_nn_pools_and_reports(tmp_path):
    symbols_paths = _universe(tmp_path)
    res = batch.run_global(symbols_paths, forward_days=[5, 10], method="nn",
                           seed=42, nn_epochs=150)
    assert res.scope == "global"
    assert res.status in ("improved", "no_improvement")
    assert res.baseline_metrics and res.tuned_metrics
    assert "objective_score" in res.tuned_metrics
    # the pooled-NN note records how many symbols were pooled
    assert any("nn-global" in n for n in res.notes)
    if res.accepted:
        assert "nn" in res.selected_config
        assert "top_indicators" in res.selected_config["nn"]


def test_global_nn_is_deterministic(tmp_path):
    symbols_paths = _universe(tmp_path)
    a = batch.run_global(symbols_paths, method="nn", seed=7, nn_epochs=120)
    b = batch.run_global(symbols_paths, method="nn", seed=7, nn_epochs=120)
    assert a.tuned_metrics == b.tuned_metrics
    assert a.status == b.status


def test_global_numgrad_still_works(tmp_path):
    symbols_paths = _universe(tmp_path)
    res = batch.run_global(symbols_paths, method="numgrad", seed=42)
    assert res.scope == "global"
    assert res.status in ("improved", "no_improvement")
    assert res.tuned_metrics is not None
