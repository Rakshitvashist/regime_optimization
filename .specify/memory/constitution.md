<!--
SYNC IMPACT REPORT
==================
Version change: (uninitialized template) → 1.0.0
Bump rationale: Initial ratification — first concrete constitution replacing the raw template.

Modified principles: N/A (initial adoption)
Added principles:
  - I. Data Integrity & Reproducibility
  - II. No Look-Ahead Bias (NON-NEGOTIABLE)
  - III. Backtest-Validated Signals
  - IV. Modular Indicators & Transparent Consensus
  - V. Observability & Auditability
Added sections:
  - Data & Risk Constraints
  - Development Workflow & Quality Gates
Removed sections: None

Templates requiring updates:
  - .specify/templates/plan-template.md  ✅ compatible (uses dynamic "[Gates determined based on constitution file]" placeholder; no hardcoded principles to edit)
  - .specify/templates/spec-template.md  ✅ compatible (no constitution-coupled mandatory sections)
  - .specify/templates/tasks-template.md ✅ compatible (no principle-driven task categories to add/remove)
  - .specify/templates/checklist-template.md ✅ compatible (generic)
  - .specify/templates/commands/*.md ⚠ none present (directory empty)

Follow-up TODOs: None — ratification date set to initial adoption date (2026-06-10).
-->

# ML Regime Constitution
<!-- Quantitative trading / ML regime-consensus system for Indian equities (NIFTY 50/500) -->

## Core Principles

### I. Data Integrity & Reproducibility

All market data MUST be sourced through the project's ingestion pipeline (`nifty_50.py`,
`nifty_500.py`) and persisted as immutable, append-only OHLCV CSVs keyed by symbol and date.
Incremental updates MUST only append rows after the last stored date and MUST NOT mutate or
back-fill historical rows. Every prediction, backtest, or report MUST be reproducible from the
stored data plus committed code alone — no manual edits, no out-of-band data sources, no
hidden state. Required OHLCV columns (`open, high, low, close, volume`) MUST be validated before
any indicator computation; missing columns MUST raise rather than silently impute.

*Rationale: A trading system's conclusions are only as trustworthy as the data lineage behind
them. Immutable, reproducible data is the precondition for every other principle.*

### II. No Look-Ahead Bias (NON-NEGOTIABLE)

No indicator, signal, prediction, or backtest may use information not available at the decision
timestamp. Indicators MUST be computed using only data up to and including the current bar.
Backtests MUST evaluate signals against strictly forward-dated prices and MUST NOT leak future
returns into feature construction, normalization, or model fitting. Any rolling statistic,
scaler, or regime classifier MUST be fit on past data only. A change that cannot demonstrate
absence of look-ahead leakage MUST NOT merge.

*Rationale: Look-ahead bias is the single most common way quantitative systems produce
impressive-but-fake results. It is non-negotiable because a leaked future is indistinguishable
from fraud once capital is at risk.*

### III. Backtest-Validated Signals

No predictive signal or strategy change ships without a hit-ratio backtest
(`Hit_ratio_backtester.py`) over the relevant forward horizons. Reported performance MUST
include directional accuracy, signal counts, and confidence-stratified results — never a single
cherry-picked metric. Claims of improvement MUST be backed by a before/after backtest on the
same data and horizons. Confidence thresholds and voting parameters that materially change
output MUST be re-validated, not assumed stable.

*Rationale: Consensus voting across ~400 indicators can rationalize almost any narrative;
out-of-sample directional accuracy is the only honest arbiter of whether a signal works.*

### IV. Modular Indicators & Transparent Consensus

Each indicator MUST be self-contained, normalized to a common scale (−1 strong sell … +1 strong
buy), and independently inspectable. The consensus layer MUST keep its inputs traceable: for any
prediction it MUST be possible to recover which categories (trend, momentum, volatility, volume,
price action) and which indicators drove the score. Category grouping, regime awareness, and
weighting logic MUST be explicit and documented — no opaque magic constants without rationale.

*Rationale: A consensus you cannot decompose is a black box; transparency is what lets a human
catch a broken indicator before it costs money.*

### V. Observability & Auditability

Every batch run (ingestion, prediction, backtest, dashboard generation) MUST emit
human-readable progress and a summary of what was processed, skipped, and failed — including
per-symbol failures rather than a silent partial success. Reports and dashboards
(`report.py`, `generate_summary*.py`, `update_dashboard.py`) MUST state the data window and
generation context so a reader can tell what they are looking at. Failures MUST be surfaced, not
swallowed; a run that processed 47 of 50 symbols MUST say so.

*Rationale: Silent partial failure in a daily pipeline quietly corrupts the signal over time;
auditable runs make degradation visible while it is still cheap to fix.*

## Data & Risk Constraints

- Symbol universe is defined by the committed constituent lists (`ind_nifty50list.csv`,
  NIFTY 500 list); changes to the universe MUST be intentional and version-controlled.
- Stored CSVs are the source of truth; downstream code MUST treat them as read-only inputs.
- The system produces analytical signals, not financial advice or automated order execution.
  Any extension toward live execution MUST introduce explicit risk controls and is out of scope
  until a dedicated amendment defines them.
- External market-data providers (e.g., yfinance) are unreliable; ingestion MUST degrade
  gracefully (retry/skip with logging) and MUST NOT corrupt existing data on partial failure.

## Development Workflow & Quality Gates

- Changes that touch indicator math, normalization, consensus weighting, or regime detection
  MUST be accompanied by a backtest comparison (Principle III) before merge.
- Reviews MUST verify compliance with Principles I–V, with explicit attention to look-ahead
  bias (Principle II) on any change to feature construction or evaluation.
- New indicators MUST follow the existing self-contained, normalized pattern (Principle IV).
- Complexity MUST be justified: prefer the simplest construction that the backtest supports
  over speculative sophistication.

## Governance

This constitution supersedes ad-hoc practice for the ML Regime project. Amendments MUST be made
by editing this file, MUST document the change and its rationale, and MUST bump the version per
the policy below. Versioning is semantic: MAJOR for backward-incompatible governance or
principle removals/redefinitions, MINOR for newly added or materially expanded principles or
sections, PATCH for clarifications and non-semantic refinements. All reviews and substantive
changes MUST verify compliance with the principles above; deviations MUST be justified in the
relevant plan's Complexity Tracking and either remedied or explicitly accepted. For runtime
development guidance and project structure, consult the current plan and `CLAUDE.md`.

**Version**: 1.0.0 | **Ratified**: 2026-06-10 | **Last Amended**: 2026-06-10
