"""T029 - a batch run accounts for every symbol (improved/no_improvement/skipped/failed)."""
import pandas as pd

from optimization import batch
from tests.conftest import make_ohlcv, make_indicators


def _write_symbol(directory, symbol, n, seed):
    o = make_ohlcv(n=n, seed=seed)
    df = o.join(make_indicators(o, seed=seed))
    df.index.name = "date"
    df.reset_index().to_csv(directory / f"{symbol}_with_indicators.csv", index=False)


def test_batch_accounts_for_all_symbols(tmp_path):
    d = tmp_path / "processed_indicators"
    d.mkdir()
    _write_symbol(d, "AAA", 400, 1)   # eligible
    _write_symbol(d, "BBB", 400, 2)   # eligible
    _write_symbol(d, "TINY", 50, 3)   # too short -> skipped

    global_result, results = batch.run_batch(
        str(d), forward_days=[5, 10], tune="weights", jobs=1, seed=42,
        configs_dir=None)

    obj_meta = {"forward_days": [5, 10], "w_acc": 0.5, "w_ret": 0.5, "embargo_bars": 10}
    summary = batch.build_run_summary("batch-test", "batch", 42, obj_meta,
                                      global_result, results, "2026-06-10T00:00:00Z")

    t = summary["totals"]
    assert t["targeted"] == 3
    assert t["improved"] + t["no_improvement"] + t["skipped"] + t["failed"] == 3
    # The short symbol must be skipped with a reason, not silently dropped (FR-010).
    tiny = [s for s in summary["symbols"] if s["symbol"] == "TINY"][0]
    assert tiny["status"] == "skipped" and tiny["reason"]
    # No symbol is silently omitted (SC-006).
    assert len(summary["symbols"]) == 3
