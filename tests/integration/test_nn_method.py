"""NN method: hand-coded backprop trains, regularizes, and is deterministic."""
import numpy as np

from optimization import data_loader, nn_weight_optimizer as nnopt, runner
from tests.conftest import make_ohlcv, make_indicators


def _inputs(n=400, seed=1):
    o = make_ohlcv(n=n, seed=seed)
    ind = make_indicators(o, seed=seed)
    price = o["close"].astype(float)
    atr = data_loader.compute_atr(o, 14)
    return ind, price, atr


def test_training_reduces_loss_and_is_deterministic():
    ind, price, atr = _inputs()
    m1 = nnopt.train(ind, price, atr, [5, 10], l1=1e-3, l2=1e-3, epochs=200)
    m2 = nnopt.train(ind, price, atr, [5, 10], l1=1e-3, l2=1e-3, epochs=200)
    # loss decreased over training
    assert m1.history[-1] < m1.history[0]
    # deterministic: identical weights across runs (FR-014)
    assert np.allclose(m1.W, m2.W)
    assert m1.b == m2.b
    # learned something non-trivial
    assert np.any(np.abs(m1.W) > 1e-4)


def test_l1_increases_sparsity():
    ind, price, atr = _inputs()
    weak = nnopt.train(ind, price, atr, [5, 10], l1=0.0, l2=1e-4, epochs=200)
    strong = nnopt.train(ind, price, atr, [5, 10], l1=0.05, l2=1e-4, epochs=200)
    nz_weak = int(np.sum(np.abs(weak.W) > 1e-4))
    nz_strong = int(np.sum(np.abs(strong.W) > 1e-4))
    # stronger L1 -> fewer surviving indicators (automatic selection)
    assert nz_strong <= nz_weak


def test_run_symbol_nn_reports_comparison(real_or_synth_csv):
    _, symbol, csv_path = real_or_synth_csv
    res = runner.run_symbol(symbol, csv_path, method="nn", seed=42, nn_epochs=150)
    assert res.status in ("improved", "no_improvement")
    assert res.baseline_metrics is not None and res.tuned_metrics is not None
    assert "objective_score" in res.tuned_metrics
    if res.accepted:
        assert res.selected_config is not None and "nn" in res.selected_config
        # selection result is present
        assert "top_indicators" in res.selected_config["nn"]


def test_run_symbol_nn_is_deterministic(real_or_synth_csv):
    _, symbol, csv_path = real_or_synth_csv
    a = runner.run_symbol(symbol, csv_path, method="nn", seed=7, nn_epochs=120)
    b = runner.run_symbol(symbol, csv_path, method="nn", seed=7, nn_epochs=120)
    assert a.tuned_metrics == b.tuned_metrics
    assert a.status == b.status
