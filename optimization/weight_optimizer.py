"""
T015 - Consensus-weight optimizer (gradient descent on the weights).

The consensus score is a function of the category weights, the regime boosts, and
the voting threshold. The hybrid objective itself is non-differentiable in closed
form (it routes through a sign/threshold step and the backtester), so gradients
are taken NUMERICALLY (central differences) on the real objective evaluated over
the TRAIN window — this is genuine gradient ascent on the weight vector, the
"backpropagation through the weighting layer" intent of the spec, without
reimplementing the whole predictor as differentiable ops.

Deterministic: no randomness, fixed evaluation order -> reproducible (FR-014).
Search is on TRAIN only; selection on held-out happens in the runner (Principle II).
"""
from __future__ import annotations

from typing import Callable, List

from . import search_space
from .config import IndicatorConfig, WeightConfig

# Continuous weight parameters tuned by gradient ascent.
_CATS = ["trend", "price_action", "momentum", "volatility", "volume", "other"]
_SCALARS = ["trend_trending_boost", "momentum_ranging_boost"]
# Threshold is a step parameter (near-zero numeric gradient) -> tuned by line scan.
_THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def _config_from_vector(vec: List[float]) -> IndicatorConfig:
    cat = {c: vec[i] for i, c in enumerate(_CATS)}
    boosts = {s: vec[len(_CATS) + i] for i, s in enumerate(_SCALARS)}
    return IndicatorConfig(
        indicator_params={},  # weights-only candidate (inherits default params on merge)
        weights=WeightConfig(
            category=cat,
            voting_threshold=vec[-1],
            regime_multipliers=boosts,
        ),
    )


def _vector_from_config(cfg: IndicatorConfig) -> List[float]:
    dense = cfg.merge_with_default()
    v = [dense.weights.category[c] for c in _CATS]
    v += [dense.weights.regime_multipliers.get(s, 0.2) for s in _SCALARS]
    v += [dense.weights.voting_threshold]
    return v


def _clamp_vector(vec: List[float]) -> List[float]:
    out = list(vec)
    for i, c in enumerate(_CATS):
        out[i] = search_space.clamp(search_space._BY_NAME[f"category.{c}"], out[i])
    for j, s in enumerate(_SCALARS):
        out[len(_CATS) + j] = search_space.clamp(search_space._BY_NAME[s], out[len(_CATS) + j])
    out[-1] = search_space.clamp(search_space._BY_NAME["voting_threshold"], out[-1])
    return out


def optimize(base_config: IndicatorConfig,
             score_train: Callable[[IndicatorConfig], float],
             max_iter: int = 20, lr: float = 0.4, eps: float = 0.05,
             tol: float = 1e-5, seed: int = 42):
    """Gradient-ascent the weights to maximise `score_train`.

    Returns (best_config, best_train_score). Starts from base_config so the
    optimizer can never do worse than the baseline on train. `seed` is accepted
    for interface symmetry with the parameter optimizer; this routine is fully
    deterministic and does not use randomness.
    """
    theta = _clamp_vector(_vector_from_config(base_config))
    best_theta = list(theta)
    best_score = score_train(_config_from_vector(theta))

    for _ in range(max_iter):
        grad = [0.0] * len(theta)
        for i in range(len(theta) - 1):  # exclude threshold (scanned separately)
            up = list(theta); up[i] += eps
            dn = list(theta); dn[i] -= eps
            up = _clamp_vector(up); dn = _clamp_vector(dn)
            s_up = score_train(_config_from_vector(up))
            s_dn = score_train(_config_from_vector(dn))
            grad[i] = (s_up - s_dn) / (2 * eps)

        gnorm = sum(g * g for g in grad) ** 0.5
        if gnorm < tol:
            break
        theta = _clamp_vector([t + lr * g for t, g in zip(theta, grad)])
        s = score_train(_config_from_vector(theta))
        if s > best_score:
            best_score = s
            best_theta = list(theta)

    # Line scan over the (non-differentiable) voting threshold.
    for thr in _THRESHOLD_GRID:
        cand = list(best_theta); cand[-1] = thr
        s = score_train(_config_from_vector(_clamp_vector(cand)))
        if s > best_score:
            best_score = s
            best_theta = _clamp_vector(cand)

    return _config_from_vector(best_theta), best_score
