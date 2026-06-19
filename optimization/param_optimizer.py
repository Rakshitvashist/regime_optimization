"""
T016 - Indicator-parameter optimizer (derivative-free Bayesian search).

Indicator parameters (lookback periods, BB std, etc.) are discrete/expensive and
require an indicator recompute per candidate, so they are searched with Optuna's
seeded TPE sampler under a bounded trial budget (research.md R1/R8). Optuna and
TA-Lib are imported lazily; if either is unavailable (or there is no raw OHLCV to
recompute from), `optimize` raises `ParamSearchUnavailable` and the runner falls
back to weights-only optimization with a recorded note.
"""
from __future__ import annotations

from typing import Callable, Dict

from . import search_space
from .config import IndicatorConfig, WeightConfig


class ParamSearchUnavailable(RuntimeError):
    """Raised when Optuna/TA-Lib are missing so parameter search cannot run."""


def _make_sampler(seed: int):
    import optuna  # lazy
    return optuna.samplers.TPESampler(seed=seed)


def optimize(base_config: IndicatorConfig,
             score_params_train: Callable[[Dict[str, float]], float],
             budget: int = 100, seed: int = 42):
    """Search indicator parameters to maximise `score_params_train(params)`.

    `score_params_train` recomputes indicators for the given params and returns
    the TRAIN objective score. Returns (best_params_dict, best_score).
    Raises ParamSearchUnavailable if Optuna cannot be imported.
    """
    try:
        import optuna  # noqa: F401
    except Exception as exc:  # pragma: no cover - env-dependent
        raise ParamSearchUnavailable(f"optuna unavailable: {exc}")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    space = search_space.entries("params")
    # Within-run cache: identical parameter sets recompute indicators once (R8).
    _cache: Dict[tuple, float] = {}

    def _objective(trial):
        params: Dict[str, float] = {}
        for e in space:
            if e.kind == "int":
                params[e.name] = trial.suggest_int(e.name, int(e.low), int(e.high))
            else:
                params[e.name] = trial.suggest_float(e.name, float(e.low), float(e.high))
        key = tuple(sorted(params.items()))
        if key in _cache:
            return _cache[key]
        # deterministic tie-break handled by Optuna seed + fixed enqueue
        score = score_params_train(params)
        _cache[key] = score
        return score

    study = optuna.create_study(direction="maximize", sampler=_make_sampler(seed))
    # Enqueue the default params first so the search never starts below baseline.
    default = IndicatorConfig.default().indicator_params
    study.enqueue_trial({e.name: default[e.name] for e in space if e.name in default})
    study.optimize(_objective, n_trials=budget, show_progress_bar=False)

    return dict(study.best_params), float(study.best_value)
