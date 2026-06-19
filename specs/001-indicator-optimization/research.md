# Phase 0 Research: Indicator Optimization

This document resolves the open technical questions implied by the spec and Technical Context. Each
item follows: **Decision / Rationale / Alternatives considered**.

---

## R1. Optimization method (the "backpropagation" question)

**Decision**: Split the problem into two sub-problems and use a different method for each.

1. **Consensus-weight sub-problem (continuous, cheap)** → **gradient descent on a differentiable
   surrogate loss**. Given fixed indicator signals, the consensus score is a differentiable function of
   the category weights, voting threshold, and regime multipliers. Replace the hard sign/threshold
   objective with a smooth surrogate (e.g. a logistic/`tanh` confidence times realized forward return,
   or a soft-hinge on direction) so gradients exist. Optimize weights with plain gradient descent
   (numpy) or `scipy.optimize`. This is literally "backpropagation" through the weighting layer and
   directly honors the user's stated intent.
2. **Indicator-parameter sub-problem (discrete/expensive)** → **derivative-free Bayesian search**
   (Optuna TPE sampler). talib lookback periods and thresholds are integer/discrete and require a full
   indicator recompute per candidate; the true hit-ratio objective is a non-differentiable step
   function. A sample-efficient global optimizer (TPE) with a bounded trial budget and pruning fits.

The runner alternates/stages them: optimize indicator parameters on the training window with a bounded
budget, then optimize weights on the resulting signals; both are scored and finally selected on the
held-out window.

**Rationale**: Pure backpropagation cannot flow through discrete talib periods or a step-function
hit-ratio, but it is the right tool for the continuous weighting layer. Splitting the problem keeps the
expensive recompute (indicator params) under a sample-efficient search while still giving the user a
genuine gradient-trained weighting layer. It also matches the existing code seams: weights live in
`ConsensusPredictor`, parameters live in `IndicatorLibrary`.

**Alternatives considered**:
- *Single global derivative-free optimizer over everything (CMA-ES / genetic)*: simpler conceptually
  but wastes evaluations on the continuous weights that gradients solve far faster, and every trial
  pays the indicator recompute cost.
- *Full differentiable pipeline (reimplement indicators as differentiable ops, straight-through
  estimators for periods)*: maximally "backprop" but a large, fragile reimplementation of 400 talib
  indicators — rejected as disproportionate.
- *Pure grid/random search*: transparent but sample-inefficient over a high-dimensional space;
  acceptable as a fallback for the parameter sub-problem if Optuna is undesired.

---

## R2. Train / held-out methodology & look-ahead prevention (Principle II — non-negotiable)

**Decision**: Walk-forward split — earliest contiguous block = **training**, later contiguous block =
**held-out** — with an **embargo gap of `max(forward_days)` bars** (default 10) removed between them.
Selection metric is computed only on the held-out block. All rolling stats, scalers, and the adaptive
quality weights are fit on training data only and applied forward.

**Rationale**: Forward-return labels (5/10-day) at the tail of the training window overlap calendar
days in the held-out window; without an embargo equal to the longest horizon, training labels leak
future information into the test boundary. An embargo gap is the standard fix and makes the
"no candidate scored on data at/after its decision timestamp" requirement (FR-003) concrete and
testable.

**Alternatives considered**:
- *K-fold cross-validation*: invalid for time series (shuffles future into past).
- *Single split without embargo*: simplest but violates Principle II at the boundary.
- *Full walk-forward with multiple rolling folds*: more robust generalization estimate; deferred as an
  optional enhancement — MVP uses a single train/embargo/held-out split, with the splitter designed to
  support multiple folds later.

---

## R3. Overfitting control & accept/reject rule

**Decision**: (a) Select solely on held-out performance; (b) bound the tunable surface (small,
documented set of params + ranges — see `contracts/`/`search_space.py`); (c) **accept a tuned config
only if it strictly beats its baseline on held-out** by a configurable minimum margin (default > 0),
otherwise retain baseline and report "no improvement" (FR-009); (d) report the train-vs-held-out gap so
overfitting is visible (spec Edge Cases).

**Rationale**: Out-of-sample selection + a bounded search space + a minimum-improvement gate are the
cheapest effective guards against curve-fitting on noisy financial data, and they make SC-004
("0% accepted configs worse than baseline") enforceable.

**Alternatives considered**: explicit complexity penalties / regularization terms in the objective —
deferred; the bounded search space already limits capacity, and a penalty adds a hyperparameter to
justify.

---

## R4. Hybrid objective definition (FR-004)

**Decision**: `score = w_acc * directional_hit_ratio + w_ret * normalized_mean_directional_return`,
evaluated on the held-out window via `HitRatioBacktester`, aggregated across configured horizons
(default 5 & 10 day) by signal-count-weighted mean. **Return normalization**: the mean directional
return is expressed in **ATR-relative units** — each forward return is divided by the bar's ATR (the
same ATR series the predictor already computes) before averaging — so the return component is scale-free
and comparable across symbols and price levels, with no cross-symbol leakage (normalization uses only
that symbol's own ATR, computed on data ≤ the decision bar). `w_acc`/`w_ret` are explicit, documented,
defaulted (0.5/0.5), and stored in the artifact alongside the normalization method. The differentiable
surrogate used for gradient training of weights is a smooth analogue of the same objective
(confidence-weighted realized return + soft directional accuracy).

**Rationale**: Reuses the existing, trusted backtester for the reported/selection metric while giving
the gradient optimizer a smooth proxy; keeping the mix explicit and stored satisfies reproducibility
and auditability.

**Alternatives considered**: pure hit-ratio (loses magnitude information, the user asked for "correct
values" → returns matter); pure Sharpe/risk-adjusted (richer but needs a position-sizing model not yet
in scope) — folded into a future enhancement.

---

## R5. Global vs per-symbol learning & precedence (FR-008, FR-013)

**Decision**: Learn one **global** config by optimizing the **equal-weighted mean of per-symbol
held-out objective scores** across all eligible universe symbols (symbols skipped for insufficient
history are excluded from the mean, not scored as zero); then, per symbol, learn a candidate
**override** and adopt it only if it beats the global on that symbol's held-out window. Prediction-time precedence: **per-symbol override → global →
default**, with the chosen level recorded on every prediction.

**Rationale**: Global config is robust and low-overfit; per-symbol overrides capture genuine
idiosyncrasy but are gated by held-out improvement so they cannot degrade results. The recorded level
satisfies auditability (Principle V).

**Alternatives considered**: per-symbol only (more overfit, no shared prior); global only (ignores
per-stock behavior). The spec explicitly chose both.

---

## R6. Configuration artifact format, versioning & storage (FR-007, FR-014)

**Decision**: JSON artifacts under `configs/global/` and `configs/per_symbol/<SYMBOL>/`, each carrying:
schema version, scope, indicator-param values, weight values, objective + mix, train/embargo/held-out
window bounds, RNG seed, achieved held-out metrics, baseline metrics, and a creation timestamp (passed
in, not read from system clock inside pure logic, to keep runs reproducible). Latest-by-version
resolution; never overwrite in place — write a new version, keep history. Schemas live in
`contracts/tuned-config.schema.json` and `contracts/run-summary.schema.json`.

**Rationale**: JSON matches the project's existing artifact style (`summary.json`); embedding full
provenance makes every accepted config reproducible and auditable (Principles I & V).

**Alternatives considered**: pickling fitted objects (opaque, version-fragile — rejected); a single
mutable config file (loses history and reproducibility — rejected).

---

## R7. Reproducibility & determinism (FR-014, SC-005)

**Decision**: All stochastic steps take an explicit seed (stored in the artifact); Optuna uses a seeded
sampler; gradient descent uses deterministic initialization; timestamps are injected as inputs rather
than read inside core logic. Tie-breaks use a documented deterministic rule (e.g. lowest parameter-
vector lexicographic order) so equal-scoring candidates resolve identically (spec Edge Cases).

**Rationale**: Required by Principle I and SC-005 (identical inputs → identical artifact/metrics in
100% of runs).

**Alternatives considered**: none — determinism is a hard requirement.

---

## R8. Performance & scale (recompute cost)

**Decision**: (a) Weight optimization reuses already-computed indicator signals → no recompute;
(b) indicator-parameter search runs under a bounded trial budget on a reduced high-impact subset, with
Optuna pruning; (c) parallelize across symbols (process pool) for batch/global runs; (d) cache indicator
computations keyed by parameter set within a run. talib+numba keep per-recompute cost low.

**Rationale**: The dominant cost is recomputing 400+ indicators per parameter candidate; isolating that
to a bounded, prunable, parallelized search keeps a NIFTY 50 batch within a normal pipeline window
(Performance Goals) while weight tuning stays near-instant.

**Alternatives considered**: recompute every indicator for every candidate with no budget/caching
(intractable at 50–500 symbols); precompute a fixed parameter grid to CSV (huge storage, still a grid).

---

## R9. Integration with the existing pipeline

**Decision**: Extend, don't replace. `IndicatorLibrary.calculate_all_indicators` gains an optional
indicator-param config (defaults reproduce today's hardcoded values). `ConsensusPredictor` accepts a
weight config and records the config level used. `generate_summary*.py` resolve the tuned config via
`optimization/store.py` precedence before predicting. `update_dashboard.py` gains an optional, off-by-
default "optimize" step so the dashboard run is unchanged unless explicitly invoked.

**Rationale**: Preserves the working pipeline and honors backward compatibility (defaults = current
behavior); satisfies FR-008 and keeps Principle IV's modularity.

**Alternatives considered**: a parallel standalone pipeline (duplicates logic, drifts from production);
hard-wiring tuned configs into the scripts (breaks reproducibility/fallback). Both rejected.

---

## Open items carried into design

None blocking. All NEEDS CLARIFICATION from the spec were resolved by the user (objective = hybrid,
scope = params + weights, level = global + per-symbol). Remaining choices above are design defaults that
are explicit, documented, and overridable.
