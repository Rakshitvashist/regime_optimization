"""T028 - run-summary artifact conforms to schema; every symbol is accounted for."""
import json
import os

import jsonschema

from optimization import batch
from optimization.runner import OptimizationResult

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA_PATH = os.path.join(
    REPO, "specs", "001-indicator-optimization", "contracts", "run-summary.schema.json"
)


def _metrics(score):
    return {"hit_ratio": {"5d": 0.55}, "mean_directional_return": {"5d": 0.2},
            "signal_count": {"5d": 80}, "objective_score": score}


def _sym(symbol, status, accepted=False):
    return OptimizationResult(
        symbol=symbol, scope="symbol", status=status, accepted=accepted, seed=42,
        objective={"forward_days": [5, 10], "w_acc": 0.5, "w_ret": 0.5, "embargo_bars": 10},
        baseline_metrics=_metrics(0.5) if status in ("improved", "no_improvement") else None,
        tuned_metrics=_metrics(0.6) if status in ("improved", "no_improvement") else None,
        skip_reason="insufficient history (needs >= 120 bars, has 40)" if status == "skipped" else None,
        error="ValueError: boom" if status == "failed" else None,
    )


def test_run_summary_validates_and_accounts_for_all_symbols():
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    global_result = OptimizationResult(
        symbol=None, scope="global", status="improved", accepted=True, seed=42,
        objective={"forward_days": [5, 10], "w_acc": 0.5, "w_ret": 0.5, "embargo_bars": 10},
        baseline_metrics=_metrics(0.50), tuned_metrics=_metrics(0.57),
    )
    per_symbol = [_sym("AAA", "improved", True), _sym("BBB", "no_improvement"),
                  _sym("CCC", "skipped"), _sym("DDD", "failed")]
    obj_meta = {"forward_days": [5, 10], "w_acc": 0.5, "w_ret": 0.5, "embargo_bars": 10}

    summary = batch.build_run_summary("batch-test-001", "batch", 42, obj_meta,
                                      global_result, per_symbol, "2026-06-10T00:00:00Z")
    jsonschema.validate(summary, schema)

    # Every targeted symbol accounted for; counts sum to the universe (SC-006).
    assert len(summary["symbols"]) == summary["totals"]["targeted"] == 4
    t = summary["totals"]
    assert t["improved"] + t["no_improvement"] + t["skipped"] + t["failed"] == t["targeted"]
