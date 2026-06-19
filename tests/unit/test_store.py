"""T037 - store versioning, build_artifact provenance, and config round-trip."""
import os

from optimization import store
from optimization.config import IndicatorConfig
from optimization.runner import OptimizationResult


def _metrics(score=0.6):
    return {"hit_ratio": {"5d": 0.58}, "mean_directional_return": {"5d": 0.3},
            "signal_count": {"5d": 100}, "objective_score": score}


def _result(symbol="ZZZ", accepted=True, scope="symbol"):
    return OptimizationResult(
        symbol=symbol if scope == "symbol" else None, scope=scope,
        status="improved" if accepted else "no_improvement", accepted=accepted, seed=7,
        objective={"forward_days": [5, 10], "w_acc": 0.5, "w_ret": 0.5, "embargo_bars": 10},
        baseline_metrics=_metrics(0.5), tuned_metrics=_metrics(0.6),
        selected_config=IndicatorConfig.default().to_dict() if accepted else None,
        window={"train_start": "2020-01-01", "train_end": "2021-01-01", "embargo_bars": 10,
                "heldout_start": "2021-01-20", "heldout_end": "2021-06-01"},
    )


def test_build_artifact_carries_provenance():
    art = store.build_artifact(_result(), created_at="2026-06-10T00:00:00Z")
    assert art["schema_version"] == store.SCHEMA_VERSION
    assert art["seed"] == 7
    assert art["created_at"] == "2026-06-10T00:00:00Z"
    assert art["objective"] == {"forward_days": [5, 10], "w_acc": 0.5, "w_ret": 0.5}
    assert art["heldout_metrics"]["objective_score"] == 0.6
    assert art["baseline_metrics"]["objective_score"] == 0.5


def test_version_increments_and_keeps_history(tmp_path):
    cfg = str(tmp_path)
    p1 = store.save(_result(), cfg, created_at="t1")
    p2 = store.save(_result(), cfg, created_at="t2")
    p3 = store.save(_result(), cfg, created_at="t3")
    assert all(os.path.exists(p) for p in (p1, p2, p3))
    assert p1.endswith("ZZZ.v01.json") and p2.endswith("ZZZ.v02.json") and p3.endswith("ZZZ.v03.json")


def test_config_dict_round_trip():
    cfg = IndicatorConfig.default()
    back = IndicatorConfig.from_dict(cfg.to_dict())
    assert back.weights.voting_threshold == cfg.weights.voting_threshold
    assert back.weights.category == cfg.weights.category
    assert back.indicator_params == cfg.indicator_params
