"""T021 - a written tuned-config artifact conforms to the contract schema."""
import json
import os

import jsonschema

from optimization import store
from optimization.config import IndicatorConfig
from optimization.runner import OptimizationResult

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA_PATH = os.path.join(
    REPO, "specs", "001-indicator-optimization", "contracts", "tuned-config.schema.json"
)


def _metrics():
    return {
        "hit_ratio": {"5d": 0.58, "10d": 0.55},
        "mean_directional_return": {"5d": 0.3, "10d": 0.2},
        "signal_count": {"5d": 120, "10d": 110},
        "objective_score": 0.61,
        "train_score": 0.63,
    }


def _accepted_result(scope="symbol", symbol="RELIANCE"):
    return OptimizationResult(
        symbol=symbol, scope=scope, status="improved", accepted=True, seed=42,
        objective={"forward_days": [5, 10], "w_acc": 0.5, "w_ret": 0.5, "embargo_bars": 10},
        baseline_metrics=_metrics(), tuned_metrics=_metrics(),
        selected_config=IndicatorConfig.default().to_dict(),
        window={
            "train_start": "2020-01-01", "train_end": "2021-06-01",
            "embargo_bars": 10, "heldout_start": "2021-06-20", "heldout_end": "2022-01-01",
        },
    )


def test_artifact_validates_against_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    artifact = store.build_artifact(_accepted_result(), created_at="2026-06-10T00:00:00Z")
    artifact["version"] = 1
    jsonschema.validate(artifact, schema)  # raises on non-conformance


def test_global_artifact_has_null_symbol():
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    artifact = store.build_artifact(_accepted_result(scope="global", symbol="RELIANCE"),
                                    created_at="2026-06-10T00:00:00Z")
    artifact["version"] = 1
    assert artifact["symbol"] is None
    jsonschema.validate(artifact, schema)
