"""T022 - versioned persistence (no overwrite) + precedence resolution."""
import os

from optimization import store
from optimization.config import IndicatorConfig
from optimization.runner import OptimizationResult


def _metrics(score=0.6):
    return {
        "hit_ratio": {"5d": 0.58}, "mean_directional_return": {"5d": 0.3},
        "signal_count": {"5d": 100}, "objective_score": score,
    }


def _result(scope="symbol", symbol="ACME", accepted=True):
    return OptimizationResult(
        symbol=symbol, scope=scope, status="improved" if accepted else "no_improvement",
        accepted=accepted, seed=42,
        objective={"forward_days": [5, 10], "w_acc": 0.5, "w_ret": 0.5, "embargo_bars": 10},
        baseline_metrics=_metrics(0.5), tuned_metrics=_metrics(0.6),
        selected_config=IndicatorConfig.default().to_dict() if accepted else None,
        window={"train_start": "2020-01-01", "train_end": "2021-01-01", "embargo_bars": 10,
                "heldout_start": "2021-01-20", "heldout_end": "2021-06-01"},
    )


def test_versioned_write_never_overwrites(tmp_path):
    cfg = str(tmp_path)
    p1 = store.save(_result(symbol="ACME"), cfg, created_at="t1")
    p2 = store.save(_result(symbol="ACME"), cfg, created_at="t2")
    assert p1 and p2 and p1 != p2          # distinct versions
    assert os.path.exists(p1) and os.path.exists(p2)  # v1 retained after v2 written
    assert "ACME.v01.json" in p1 and "ACME.v02.json" in p2


def test_non_improving_run_writes_nothing(tmp_path):
    cfg = str(tmp_path)
    assert store.save(_result(symbol="ACME", accepted=False), cfg, created_at="t1") is None
    assert not os.path.exists(os.path.join(cfg, "per_symbol", "ACME"))


def test_precedence_symbol_then_global_then_default(tmp_path):
    cfg = str(tmp_path)
    # default when nothing stored
    _, level = store.resolve_config("ACME", cfg)
    assert level == "default"
    # global only -> global
    store.save(_result(scope="global"), cfg, created_at="t1")
    _, level = store.resolve_config("ACME", cfg)
    assert level == "global"
    # per-symbol override present -> symbol wins
    store.save(_result(scope="symbol", symbol="ACME"), cfg, created_at="t2")
    _, level = store.resolve_config("ACME", cfg)
    assert level == "symbol"


def test_run_symbol_persists_when_accepted(real_or_synth_csv, tmp_path):
    from optimization import runner
    input_dir, symbol, csv_path = real_or_synth_csv
    res = runner.run_symbol(symbol, csv_path, tune="weights", seed=42,
                            configs_dir=str(tmp_path), created_at="2026-06-10T00:00:00Z")
    if res.accepted:
        assert res.artifact_path and os.path.exists(res.artifact_path)
    else:
        assert res.artifact_path is None
