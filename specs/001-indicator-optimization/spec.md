# Feature Specification: Indicator Optimization from Historical Data

**Feature Branch**: `001-indicator-optimization`

**Created**: 2026-06-10

**Status**: Draft

**Input**: User description: "for this project i want to optimize the indicators used in this like for example using the historical data indicators values check what values must be optimal to get correct values... using backpropagation got it what my point is"

## Overview

Today the consensus prediction system treats every technical indicator with fixed, hand-chosen
settings (e.g. lookback periods, signal thresholds) and combines them into a buy/sell/hold
decision using fixed category weights. Nobody has verified that those settings or weights are the
*best* ones for actually predicting future price direction — they are defaults inherited from
convention.

This feature introduces a **learning step**: using each symbol's stored historical price data, the
system automatically searches for the indicator settings and consensus weights that would have most
reliably predicted the correct forward price direction, then saves that tuned configuration so the
predictor can use it going forward. In short — let history tell us which indicator values are
"correct," instead of guessing.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Tune indicators against history and measure the gain (Priority: P1)

As an analyst, I want to run an optimization over a symbol's (or the universe's) historical data so
that the system discovers the indicator settings and consensus weights that best predicted the
correct forward direction, and I can see how much more accurate the tuned configuration is versus
today's default settings.

**Why this priority**: This is the core of the request — without the ability to learn better
settings and prove they are better, nothing else has value. It is the minimum viable slice.

**Independent Test**: Run the optimization on a chosen symbol over a defined training window,
then compare directional hit-ratio of the tuned configuration vs. the default configuration on a
held-out (later, unseen) window. Delivers value if the tuned configuration measurably improves
out-of-sample directional accuracy.

**Acceptance Scenarios**:

1. **Given** a symbol with sufficient stored history, **When** the analyst runs optimization over a
   training window, **Then** the system produces a tuned configuration (indicator settings and/or
   weights) and reports its out-of-sample directional hit-ratio alongside the default baseline.
2. **Given** a completed optimization run, **When** the analyst inspects the result, **Then** the
   tuned configuration's out-of-sample hit-ratio is reported and is greater than or equal to the
   default baseline on the same held-out window (or the run clearly flags "no improvement found").
3. **Given** an optimization run, **When** it evaluates candidate settings, **Then** no candidate is
   ever scored using price information dated at or after the decision timestamp (no look-ahead).

---

### User Story 2 - Persist and reuse the tuned configuration (Priority: P2)

As an analyst, I want the tuned configuration saved durably and automatically picked up by the
consensus predictor, so that subsequent predictions and reports use the optimized settings without
manual wiring.

**Why this priority**: Optimization that cannot be reused produces a one-off number, not an ongoing
improvement. Persistence turns the experiment into a usable capability, but it depends on Story 1.

**Independent Test**: After an optimization run, confirm a saved configuration artifact exists for
the symbol/scope, then run a normal prediction and verify it consumed the tuned configuration rather
than the defaults.

**Acceptance Scenarios**:

1. **Given** a completed optimization run, **When** it finishes, **Then** the tuned configuration is
   written to a durable, versioned artifact identified by symbol/scope and training window.
2. **Given** a saved tuned configuration exists, **When** the predictor runs for that symbol,
   **Then** it uses the tuned configuration and records which configuration version it used.
3. **Given** no tuned configuration exists for a symbol, **When** the predictor runs, **Then** it
   falls back to the default configuration and records that it did so.

---

### User Story 3 - Optimize across the universe and review results (Priority: P3)

As an analyst, I want to run optimization across all symbols in the universe in one batch and review
a summary of which symbols improved, by how much, and which failed or showed no gain.

**Why this priority**: Scaling from one symbol to the full NIFTY universe is the operational payoff,
but it builds directly on the single-symbol capability and is not required to prove the concept.

**Independent Test**: Run a batch optimization over the configured universe and confirm a summary
report lists per-symbol baseline vs tuned hit-ratio, improvement delta, and any skipped/failed
symbols.

**Acceptance Scenarios**:

1. **Given** the configured symbol universe, **When** a batch optimization runs, **Then** each
   symbol is optimized independently and a summary reports per-symbol baseline, tuned hit-ratio, and
   improvement delta.
2. **Given** a batch run where some symbols lack sufficient history, **When** it completes, **Then**
   those symbols are reported as skipped with a reason rather than silently omitted.

---

### Edge Cases

- **Insufficient history**: A symbol has too few bars to form a train/test split or to compute the
  longest-lookback indicators → the symbol is skipped with an explicit reason; optimization is not
  attempted on degenerate data.
- **No improvement found**: The search cannot beat the default baseline on the held-out window → the
  run reports "no improvement" and the default configuration is retained rather than saving a worse
  one.
- **Overfitting**: A configuration scores very well on the training window but poorly on the
  held-out window → selection is based on held-out performance, and the gap between train and
  held-out performance is reported so overfitting is visible.
- **Stale/partial data**: The stored history for a symbol ends well before the present, or has gaps →
  optimization uses only validated stored data and reports the actual window covered.
- **Ties / instability**: Multiple candidate configurations score nearly identically → a documented
  tie-break rule selects one deterministically so runs are reproducible.
- **Universe member with no data file** → reported as skipped, not failed silently.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST optimize indicator configuration for a target scope (a single symbol
  at minimum) using only that symbol's validated stored historical OHLCV data.
- **FR-002**: The system MUST split each symbol's history into a training window and a later,
  non-overlapping held-out window, and MUST select the winning configuration based on held-out
  performance, not training performance.
- **FR-003**: The system MUST evaluate candidate configurations strictly without look-ahead bias —
  every signal MUST be derivable from information available at or before its decision timestamp, and
  forward price outcomes MUST only be used to score, never to construct, signals.
- **FR-004**: The system MUST score configurations using a **hybrid objective** that combines
  directional accuracy (correct prediction of forward price direction over the configured horizon(s),
  consistent with the existing hit-ratio backtesting concept) with realized return magnitude in the
  predicted direction (measured in ATR-relative units so it is scale-free across symbols), so that
  high-confidence correct calls are rewarded more than marginal ones and
  confident wrong calls are penalized more than near-misses. The relative weighting of the accuracy
  and return components MUST be an explicit, documented, and reproducible setting.
- **FR-005**: The system MUST optimize **both** individual indicator parameters (e.g. lookback
  periods, signal thresholds) **and** the consensus/category weights. The set of tunable parameters
  and their allowed ranges MUST be explicitly defined so that the search space is bounded and
  auditable.
- **FR-006**: The system MUST produce, for every run, the tuned configuration's held-out directional
  hit-ratio and the default configuration's held-out hit-ratio on the same window, so improvement is
  directly comparable.
- **FR-007**: The system MUST persist each tuned configuration as a durable, versioned artifact
  identified by scope (symbol/universe), training window, objective, and a timestamp, so results are
  reproducible.
- **FR-008**: The consensus predictor MUST resolve which configuration to use via a defined
  precedence — per-symbol override (if present and validated as better than global) → global tuned
  configuration → default configuration — and MUST record which level was used for each prediction.
- **FR-009**: The system MUST refuse to overwrite a known-good configuration with one that does not
  improve held-out performance; a non-improving run MUST retain the prior/default configuration and
  report "no improvement."
- **FR-010**: The system MUST skip symbols with insufficient history (too few bars for the split or
  for the longest-lookback indicator) and report each skip with an explicit reason.
- **FR-011**: The system MUST support running optimization across the full configured symbol universe
  in one batch and MUST emit a summary of per-symbol baseline vs tuned performance, improvement
  deltas, and skipped/failed symbols.
- **FR-012**: The system MUST emit human-readable progress and a final summary for every run
  (processed, improved, skipped, failed), surfacing per-symbol failures rather than reporting a
  silent partial success.
- **FR-013**: The system MUST support **both** a global tuned configuration (one configuration
  learned to perform best on average across the whole universe) **and** per-symbol overrides (a
  symbol-specific configuration that is adopted only when it beats the global configuration on that
  symbol's held-out window). The prediction-time precedence and level-recording are governed by FR-008
  (per-symbol override → global → default), which is the single authority for that resolution.
- **FR-014**: Optimization runs MUST be reproducible — the same stored data, scope, and settings
  MUST yield the same tuned configuration and reported metrics.

### Key Entities *(include if feature involves data)*

- **Indicator Configuration**: The set of tunable settings that govern how signals are produced and
  combined — per-indicator parameters (e.g. lookback periods, signal thresholds) and/or
  consensus/category weights. Has a default (current) form and zero or more tuned forms.
- **Optimization Run**: A single execution that searches for a better configuration over a defined
  training window for a defined scope. Records the scope, training and held-out windows, objective,
  candidates evaluated (at least in summary), the selected configuration, and the baseline-vs-tuned
  metrics.
- **Tuned Configuration Artifact**: The durable, versioned output of an optimization run — the
  selected configuration plus the metadata needed to reproduce and audit it (scope, windows,
  objective, timestamp, achieved held-out hit-ratio, baseline hit-ratio).
- **Scope**: The target/level of an optimization run — a single symbol (producing a per-symbol
  override), or the full configured universe (producing a global configuration). Per-symbol overrides
  are adopted only when they beat the global configuration on held-out data.
- **Evaluation Window**: A contiguous slice of a symbol's stored history, divided into training and
  held-out portions, over which configurations are learned and validated.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For at least 60% of symbols that have sufficient history, the tuned configuration
  achieves a higher out-of-sample directional hit-ratio than the default configuration on the same
  held-out window.
- **SC-002**: Across symbols that improve, the average out-of-sample directional hit-ratio improves
  by at least 3 percentage points over the default baseline.
- **SC-003**: 100% of optimization runs report both the tuned and the baseline held-out hit-ratio,
  so no result is presented without a comparison.
- **SC-004**: 0% of accepted (saved) configurations are worse than their baseline on the held-out
  window — a per-symbol override is kept only if it beats the global configuration, and the global is
  kept only if it beats the default; non-improving runs never overwrite a good configuration.
- **SC-005**: Re-running optimization with identical inputs and seed reproduces the identical tuned
  configuration values and reported metrics in 100% of cases. (Determinism covers the configuration and
  metrics; provenance fields such as `created_at` are excluded from this comparison.)
- **SC-006**: A full-universe batch run completes with a per-symbol summary in which every universe
  member is accounted for as improved, unchanged, skipped (with reason), or failed (with reason) —
  no silent omissions.

## Assumptions

- The optimization consumes the existing stored OHLCV CSVs as read-only inputs; it does not fetch
  new data or alter historical rows (per the project's data-integrity principle).
- "Backpropagation" in the request is read as the user's shorthand for *automatically learning
  optimal settings from historical data*; the spec fixes the goal (learned, validated, reproducible
  optimal settings) and leaves the specific search/learning method to the planning phase.
- "Correct values" is interpreted as configurations that best predict realized forward price
  direction, scored by a hybrid objective that combines directional accuracy (the existing hit-ratio
  backtesting concept over the configured horizon(s), default 5- and 10-day forward periods) with
  realized return magnitude in the predicted direction.
- Both indicator parameters and consensus weights are in scope to be tuned; a global configuration is
  learned across the universe and per-symbol overrides are adopted only where they beat the global on
  held-out data.
- The symbol universe is the committed constituent list(s) already used by the project (NIFTY 50,
  and NIFTY 500 where applicable).
- Out-of-sample (held-out) performance, not training performance, is the basis for accepting a
  configuration, to guard against overfitting.
- Single-symbol optimization (Story 1) is the MVP; universe-wide batch optimization (Story 3) is an
  extension of the same capability.

## Dependencies

- Validated stored historical OHLCV data per symbol (produced by the existing ingestion pipeline).
- The existing indicator library and consensus predictor as the configurable system being tuned.
- The existing hit-ratio backtesting concept as the scoring/validation basis.
