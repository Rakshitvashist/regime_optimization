"""
T017 / T019 / T020 - Orchestrate one optimization run for a single symbol.

Flow (FR-001..FR-006, FR-009, FR-010):
  1. Load read-only inputs, build a walk-forward window with embargo.
  2. If history is insufficient -> status 'skipped' with reason (T020).
  3. Score the DEFAULT config on the held-out window (baseline).
  4. Search on the TRAIN window only (no look-ahead, Principle II):
       - indicator parameters via Optuna (if available), then
       - consensus weights via gradient ascent.
  5. Score the tuned candidate on the held-out window.
  6. Accept only if it beats the baseline by > min_improvement (FR-009); else
     'no_improvement'. Selection is on held-out, search on train.

Determinism (T019/FR-014): the weight optimizer is deterministic; Optuna is
seeded; acceptance uses strict '>' so ties keep the baseline -> reproducible.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from . import _legacy, data_loader, objective, splitter, store, weight_optimizer
from . import param_optimizer
from . import nn_weight_optimizer as nnopt
from .config import IndicatorConfig

# DL optimizers (lazily imported — PyTorch may not be installed)
def _get_dl_optimizer():
    from optimization import dl_optimizer
    return dl_optimizer

def _get_regime_lstm():
    from optimization import regime_lstm
    return regime_lstm

_THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


@dataclass
class OptimizationResult:
    symbol: str
    scope: str
    status: str                      # improved | no_improvement | skipped | failed
    accepted: bool
    seed: int
    objective: dict
    baseline_metrics: Optional[dict] = None
    tuned_metrics: Optional[dict] = None
    selected_config: Optional[dict] = None
    window: Optional[dict] = None
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    artifact_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def _silence():
    return contextlib.redirect_stdout(io.StringIO())


def run_symbol(symbol: str, csv_path: str,
               forward_days: Optional[List[int]] = None,
               w_acc: float = 0.5, w_ret: float = 0.5,
               heldout_frac: float = 0.3, embargo: Optional[int] = None,
               tune: str = "both", budget: int = 100,
               min_improvement: float = 0.0, seed: int = 42,
               config_level: str = "symbol",
               configs_dir: Optional[str] = None, created_at: Optional[str] = None,
               dry_run: bool = False,
               baseline_config: Optional[IndicatorConfig] = None,
               method: str = "numgrad",
               nn_l1: float = 1e-3, nn_l2: float = 1e-3,
               nn_lr: float = 0.1, nn_epochs: int = 300,
               dl_epochs: int = 500,
               dl_lr: float = 1e-3,
               dl_device: str = "cuda",
               dl_hidden: int = 256,
               dl_layers: int = 3,
               dl_seq_len: int = 60,
               vp_window: int = 120) -> OptimizationResult:
    forward_days = forward_days or [5, 10]
    obj_meta = {
        "forward_days": forward_days, "w_acc": w_acc, "w_ret": w_ret,
        "embargo_bars": int(max(forward_days) if embargo is None else embargo),
    }

    df = data_loader.load_symbol_frame(csv_path)
    indicators_df, quality_df, price, atr, adx = data_loader.build_inputs(df)

    win = splitter.make_window(symbol, indicators_df.index, forward_days,
                               heldout_frac=heldout_frac, embargo=embargo)
    if isinstance(win, splitter.InsufficientHistory):
        return OptimizationResult(
            symbol=symbol, scope="symbol", status="skipped", accepted=False,
            seed=seed, objective=obj_meta, skip_reason=win.reason,
        )

    # Mutable "active" indicator set (swapped if param search recomputes).
    state = {"ind": indicators_df, "qual": quality_df}
    Consensus = _legacy.load_consensus_module().ConsensusPredictor
    notes: List[str] = []

    def _predict(cfg: IndicatorConfig, sl: slice) -> pd.DataFrame:
        ind = state["ind"].iloc[sl]
        qual = state["qual"].iloc[sl]
        pr = price.iloc[sl]
        at = atr.iloc[sl]
        predictor = Consensus(weight_config=cfg.weights.to_dict(), config_level=config_level)
        with _silence():
            return predictor.predict(ind, qual, pr, at, None)

    def _evaluate(cfg: IndicatorConfig, sl: slice) -> objective.Metrics:
        preds = _predict(cfg, sl)
        return objective.evaluate(preds, price.iloc[sl], atr.iloc[sl],
                                  forward_days, w_acc, w_ret)

    def _score_train(cfg: IndicatorConfig) -> float:
        return _evaluate(cfg, win.train_slice).objective_score

    # Baseline is the default config for a standalone run, or the supplied
    # global config when this is a per-symbol override run inside a batch (US3):
    # the override is then accepted only if it beats the global on held-out.
    baseline = (baseline_config or IndicatorConfig.default())
    baseline_heldout = _evaluate(baseline, win.heldout_slice)

    win_dict = {
        "train_start": str(win.train_start.date()),
        "train_end": str(win.train_end.date()),
        "embargo_bars": win.embargo_bars,
        "heldout_start": str(win.heldout_start.date()),
        "heldout_end": str(win.heldout_end.date()),
    }

    # --- NN method: hand-coded backprop, linear per-indicator model -----------
    if method == "nn":
        tr, ho = win.train_slice, win.heldout_slice
        model = nnopt.train(state["ind"].iloc[tr], price.iloc[tr], atr.iloc[tr],
                            forward_days, l1=nn_l1, l2=nn_l2, lr=nn_lr,
                            epochs=nn_epochs, seed=seed)
        # Threshold line-scan on TRAIN only (no look-ahead).
        best_thr, best_train = _THRESHOLD_GRID[0], -1e18
        for thr in _THRESHOLD_GRID:
            model.threshold = thr
            preds_tr = model.predictions_frame(state["ind"].iloc[tr])
            s = objective.evaluate(preds_tr, price.iloc[tr], atr.iloc[tr],
                                   forward_days, w_acc, w_ret).objective_score
            if s > best_train:
                best_train, best_thr = s, thr
        model.threshold = best_thr

        preds_ho = model.predictions_frame(state["ind"].iloc[ho])
        tuned_heldout = objective.evaluate(preds_ho, price.iloc[ho], atr.iloc[ho],
                                           forward_days, w_acc, w_ret)
        tuned_heldout.train_score = best_train
        accepted = tuned_heldout.objective_score > baseline_heldout.objective_score + min_improvement
        status = "improved" if accepted else "no_improvement"
        cfg = model.to_config()
        top = ", ".join(c for c, _ in cfg["top_indicators"][:8])
        notes.append(f"nn: {cfg['nonzero_count']} non-zero indicator weights; top: {top}")
        # NN config shape differs from the tuned-config schema -> not persisted via
        # the schema store in this experimental method.
        if accepted and configs_dir and not dry_run:
            notes.append("nn config not persisted via schema store (experimental method)")
        return OptimizationResult(
            symbol=symbol, scope="symbol", status=status, accepted=accepted, seed=seed,
            objective=obj_meta, baseline_metrics=baseline_heldout.to_dict(),
            tuned_metrics=tuned_heldout.to_dict(),
            selected_config={"nn": cfg} if accepted else None,
            window=win_dict, notes=notes,
        )

    # --- DL method: LSTM / Regime-LSTM indicator parameter predictor ----------
    if method in ("dl", "lstm", "regime_lstm"):
        tr, ho = win.train_slice, win.heldout_slice
        # Per-stock prior (item a): feed this symbol's analyze-selected indicators
        # to the DL model so the net focuses on what already works for it.
        extra_cols = None
        try:
            import json as _json, os as _os
            sel_path = _os.path.join(configs_dir or "configs", "selected", f"{symbol}.json")
            if _os.path.exists(sel_path):
                sel = _json.load(open(sel_path, encoding="utf-8"))
                extra_cols = [s["indicator"] for s in sel.get("selected", [])]
                notes.append(f"per-stock prior: {len(extra_cols)} selected indicators fed to DL")
        except Exception:
            extra_cols = None
        dl_kwargs = dict(
            # Pass the RAW frame (with real OHLCV) so the DL model sees real
            # high/low/volume — build_inputs strips OHLCV, which would force the
            # synthetic proxy and make the liquidity-sweep/volume features useless.
            indicators_df = df,
            price         = price,
            atr           = atr,
            forward_days  = forward_days,
            train_slice   = tr,
            heldout_slice = ho,
            seq_len       = dl_seq_len,
            hidden_size   = dl_hidden,
            num_layers    = dl_layers if method != "lstm" else (dl_layers + 1),
            bidirectional = (method == "lstm"),
            vp_window     = vp_window,
            extra_indicator_cols = extra_cols,
            epochs        = dl_epochs,
            lr            = dl_lr,
            w_acc         = w_acc,
            w_ret         = w_ret,
            seed          = seed,
            device        = dl_device,
        )
        try:
            if method == "regime_lstm":
                dl_mod = _get_regime_lstm()
            else:
                dl_mod = _get_dl_optimizer()
            dl_model = dl_mod.train(**dl_kwargs)
        except Exception as exc:
            notes.append(f"{method} training failed ({exc}); falling back to numgrad")
            # Fall through to standard path below
            method = "numgrad"
        else:
            from .config import WeightConfig
            cfg  = dl_model.to_config()
            disc = dl_model.discovered_params

            # DL-discovered category weights (fall back to defaults if absent).
            dl_weights = (WeightConfig.from_dict(cfg["weights"])
                          if cfg.get("weights", {}).get("category")
                          else IndicatorConfig.default().weights)
            dl_cfg = IndicatorConfig(
                indicator_params     = disc,
                weights              = dl_weights,
                is_default           = False,
                dl_params_discovered = disc,
                dl_method            = method,
            )

            # --- Use the DISCOVERED params for real: recompute the indicators
            # with them (via the existing TA-Lib IndicatorLibrary) so the learned
            # parameter values actually drive the signals we score. This is the
            # whole point of the search — the future predictions use these params.
            # If TA-Lib / recompute is unavailable, fall back to the model's own
            # soft aggregation so the run still completes (with a note).
            recompute_ok = False
            try:
                IndicatorLibrary = _legacy.load_indicator_library_cls()
                lib = IndicatorLibrary()
                with _silence():
                    recomputed = lib.calculate_all_indicators(df.copy(), params=disc)
                recomputed = recomputed.reindex(indicators_df.index).fillna(0).astype(float)
                state["ind"] = recomputed
                state["qual"] = pd.DataFrame(0.5, index=recomputed.index,
                                             columns=recomputed.columns)
                recompute_ok = True
            except Exception as exc:
                notes.append(f"{method}: indicator recompute unavailable "
                             f"({type(exc).__name__}: {exc}); scoring on stored indicators")

            if recompute_ok:
                # Threshold scan on TRAIN through the real ConsensusPredictor,
                # using the recomputed (discovered-param) indicators.
                best_thr, best_train = _THRESHOLD_GRID[0], -1e18
                for thr in _THRESHOLD_GRID:
                    dl_cfg.weights.voting_threshold = thr
                    s = _evaluate(dl_cfg, win.train_slice).objective_score
                    if s > best_train:
                        best_train, best_thr = s, thr
                dl_cfg.weights.voting_threshold = best_thr
                tuned_heldout = _evaluate(dl_cfg, win.heldout_slice)
                scoring = "recomputed via TA-Lib with discovered params"
            else:
                # Fallback: the model's own soft aggregation + threshold scan.
                best_thr, best_train = _THRESHOLD_GRID[0], -1e18
                for thr in _THRESHOLD_GRID:
                    dl_model.discovered_threshold = thr
                    s = objective.evaluate(
                        dl_model.predictions_frame(state["ind"].iloc[tr]),
                        price.iloc[tr], atr.iloc[tr],
                        forward_days, w_acc, w_ret).objective_score
                    if s > best_train:
                        best_train, best_thr = s, thr
                dl_model.discovered_threshold = best_thr
                dl_cfg.weights.voting_threshold = best_thr
                tuned_heldout = objective.evaluate(
                    dl_model.predictions_frame(state["ind"].iloc[ho]),
                    price.iloc[ho], atr.iloc[ho], forward_days, w_acc, w_ret)
                scoring = "stored-indicator fallback (TA-Lib unavailable)"

            tuned_heldout.train_score = best_train
            accepted = tuned_heldout.objective_score > baseline_heldout.objective_score + min_improvement
            status   = "improved" if accepted else "no_improvement"
            notes.append(
                f"{method}: discovered params={disc}; threshold={best_thr}; "
                f"train_score={best_train:.4f}; scoring={scoring}"
            )

            # Persist DL checkpoint if accepted
            checkpoint_path = None
            if accepted and configs_dir and not dry_run:
                import os
                os.makedirs(configs_dir, exist_ok=True)
                ckpt_name = f"{symbol or 'global'}_{method}_checkpoint.pt"
                checkpoint_path = os.path.join(configs_dir, ckpt_name)
                dl_model.save_checkpoint(checkpoint_path)
                notes.append(f"DL checkpoint saved: {checkpoint_path}")
            dl_cfg.dl_model_path = checkpoint_path

            result = OptimizationResult(
                symbol=symbol, scope="symbol", status=status, accepted=accepted, seed=seed,
                objective=obj_meta, baseline_metrics=baseline_heldout.to_dict(),
                tuned_metrics=tuned_heldout.to_dict(),
                selected_config=dl_cfg.to_dict() if accepted else None,
                window=win_dict, notes=notes,
            )
            if accepted and configs_dir and not dry_run:
                result.artifact_path = store.save(
                    result, configs_dir, created_at or "1970-01-01T00:00:00Z")
            return result

    candidate = baseline.copy()
    candidate.is_default = False

    # --- (a) indicator-parameter search (recompute; optional) ----------------
    if tune in ("params", "both"):
        try:
            IndicatorLibrary = _legacy.load_indicator_library_cls()

            def _score_params(params: Dict[str, float]) -> float:
                lib = IndicatorLibrary()
                with _silence():
                    recomputed = lib.calculate_all_indicators(df.copy(), params=params)
                recomputed = recomputed.reindex(indicators_df.index).fillna(0).astype(float)
                qual = pd.DataFrame(0.5, index=recomputed.index, columns=recomputed.columns)
                saved = state["ind"], state["qual"]
                state["ind"], state["qual"] = recomputed, qual
                try:
                    return _score_train(candidate)
                finally:
                    state["ind"], state["qual"] = saved

            best_params, _ = param_optimizer.optimize(
                candidate, _score_params, budget=budget, seed=seed)
            candidate.indicator_params = best_params
            # Recompute once with the chosen params and make it active.
            lib = IndicatorLibrary()
            with _silence():
                recomputed = lib.calculate_all_indicators(df.copy(), params=best_params)
            recomputed = recomputed.reindex(indicators_df.index).fillna(0).astype(float)
            state["ind"] = recomputed
            state["qual"] = pd.DataFrame(0.5, index=recomputed.index, columns=recomputed.columns)
        except param_optimizer.ParamSearchUnavailable as exc:
            notes.append(f"parameter search skipped ({exc}); optimized weights only")
        except Exception as exc:  # recompute failed (e.g. no TA-Lib)
            notes.append(f"parameter search unavailable ({type(exc).__name__}: {exc}); weights only")

    # --- (b) consensus-weight search (gradient ascent on train) --------------
    if tune in ("weights", "both"):
        best_wcfg, _ = weight_optimizer.optimize(candidate, _score_train, seed=seed)
        candidate.weights = best_wcfg.weights

    # --- selection on held-out ----------------------------------------------
    tuned_heldout = _evaluate(candidate, win.heldout_slice)
    tuned_heldout.train_score = _score_train(candidate)

    accepted = tuned_heldout.objective_score > baseline_heldout.objective_score + min_improvement
    status = "improved" if accepted else "no_improvement"

    result = OptimizationResult(
        symbol=symbol, scope="symbol", status=status, accepted=accepted,
        seed=seed, objective=obj_meta,
        baseline_metrics=baseline_heldout.to_dict(),
        tuned_metrics=tuned_heldout.to_dict(),
        selected_config=candidate.to_dict() if accepted else None,
        window=win_dict,
        notes=notes,
    )

    # Persist only an accepted config; a non-improving run never overwrites (FR-009, SC-004).
    if accepted and configs_dir and not dry_run:
        result.artifact_path = store.save(
            result, configs_dir, created_at or "1970-01-01T00:00:00Z")

    return result
