"""
T030 / T031 / T032 / T033 - Universe-wide optimization.

- run_global: learn ONE global config by maximising the equal-weighted mean of
  per-symbol held-out objective scores across all eligible symbols (research.md R5).
- run_batch: run global, then per-symbol overrides judged against the global
  config (adopted only when they beat it on held-out), capturing per-symbol
  failures so the batch never aborts silently (FR-012, Principle V).
- build_run_summary: a schema-conforming summary accounting for EVERY targeted
  symbol (FR-011, SC-006; contracts/run-summary.schema.json).

Pure numpy/pandas weight path (no TA-Lib) so the batch is runnable on stored
indicator CSVs. Optional cross-symbol parallelism via a process pool (--jobs).
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import io
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import _legacy, data_loader, objective, splitter, store, weight_optimizer
from . import nn_weight_optimizer as nnopt
from .config import IndicatorConfig
from .runner import OptimizationResult, run_symbol


def _silence():
    return contextlib.redirect_stdout(io.StringIO())


# --- global config -----------------------------------------------------------
def _prepare(symbol: str, csv_path: str, forward_days, heldout_frac, embargo):
    df = data_loader.load_symbol_frame(csv_path)
    ind, qual, price, atr, _ = data_loader.build_inputs(df)
    win = splitter.make_window(symbol, ind.index, forward_days,
                               heldout_frac=heldout_frac, embargo=embargo)
    if isinstance(win, splitter.InsufficientHistory):
        return None
    # Keep the raw OHLCV frame so the DL path can recompute indicators with the
    # globally-discovered params (the numgrad path never needs it).
    return {"symbol": symbol, "ind": ind, "qual": qual, "price": price,
            "atr": atr, "win": win, "df": df}


def _eval(prep, cfg: IndicatorConfig, which: str, forward_days, w_acc, w_ret,
          Consensus) -> objective.Metrics:
    sl = prep["win"].train_slice if which == "train" else prep["win"].heldout_slice
    predictor = Consensus(weight_config=cfg.weights.to_dict(), config_level="global")
    with _silence():
        preds = predictor.predict(prep["ind"].iloc[sl], prep["qual"].iloc[sl],
                                  prep["price"].iloc[sl], prep["atr"].iloc[sl], None)
    return objective.evaluate(preds, prep["price"].iloc[sl], prep["atr"].iloc[sl],
                              forward_days, w_acc, w_ret)


def _aggregate(metrics_list: List[objective.Metrics], forward_days) -> dict:
    """Average per-horizon metrics across symbols; sum signal counts."""
    n = max(1, len(metrics_list))
    hit = {f"{d}d": sum(m.hit_ratio.get(f"{d}d", 0.0) for m in metrics_list) / n
           for d in forward_days}
    ret = {f"{d}d": sum(m.mean_directional_return.get(f"{d}d", 0.0) for m in metrics_list) / n
           for d in forward_days}
    cnt = {f"{d}d": sum(m.signal_count.get(f"{d}d", 0) for m in metrics_list)
           for d in forward_days}
    score = sum(m.objective_score for m in metrics_list) / n
    return {"hit_ratio": hit, "mean_directional_return": ret,
            "signal_count": cnt, "objective_score": score}


def run_global(symbols_paths: List[Tuple[str, str]],
               forward_days: Optional[List[int]] = None,
               w_acc: float = 0.5, w_ret: float = 0.5,
               heldout_frac: float = 0.3, embargo: Optional[int] = None,
               min_improvement: float = 0.0, seed: int = 42,
               configs_dir: Optional[str] = None, created_at: Optional[str] = None,
               dry_run: bool = False, method: str = "numgrad",
               nn_l1: float = 1e-3, nn_l2: float = 1e-3,
               nn_lr: float = 0.1, nn_epochs: int = 400,
               dl_epochs: int = 500,
               dl_lr: float = 1e-3,
               dl_device: str = "cuda",
               dl_hidden: int = 256,
               dl_layers: int = 3,
               dl_seq_len: int = 60,
               vp_window: int = 120) -> OptimizationResult:
    forward_days = forward_days or [5, 10]
    obj_meta = {"forward_days": forward_days, "w_acc": w_acc, "w_ret": w_ret,
                "embargo_bars": int(max(forward_days) if embargo is None else embargo)}
    Consensus = _legacy.load_consensus_module().ConsensusPredictor

    preps = [p for p in (_prepare(s, c, forward_days, heldout_frac, embargo)
                         for s, c in symbols_paths) if p is not None]
    if not preps:
        return OptimizationResult(symbol=None, scope="global", status="skipped",
                                  accepted=False, seed=seed, objective=obj_meta,
                                  skip_reason="no eligible symbols with sufficient history")

    def gscore_train(cfg: IndicatorConfig) -> float:
        return sum(_eval(p, cfg, "train", forward_days, w_acc, w_ret, Consensus).objective_score
                   for p in preps) / len(preps)

    baseline = IndicatorConfig.default()
    base_heldout = [_eval(p, baseline, "heldout", forward_days, w_acc, w_ret, Consensus)
                    for p in preps]
    base_agg = _aggregate(base_heldout, forward_days)

    # Aggregate window spanning all eligible symbols (schema-valid).
    window = {
        "train_start": str(min(p["win"].train_start for p in preps).date()),
        "train_end": str(max(p["win"].train_end for p in preps).date()),
        "embargo_bars": preps[0]["win"].embargo_bars,
        "heldout_start": str(min(p["win"].heldout_start for p in preps).date()),
        "heldout_end": str(max(p["win"].heldout_end for p in preps).date()),
    }

    # --- NN global: ONE pooled per-indicator backprop model across all symbols ---
    # Pooling every symbol's TRAIN rows is the regime where ~400 weights are
    # justified (≈ N_symbols × more data); selection stays on each symbol's
    # held-out window with the same embargo (no look-ahead).
    if method == "nn":
        common_cols = sorted(set.intersection(*[set(p["ind"].columns) for p in preps]))
        model = nnopt.train_pooled(preps, forward_days, common_cols,
                                   l1=nn_l1, l2=nn_l2, lr=nn_lr, epochs=nn_epochs)
        # Threshold line-scan on pooled TRAIN (mean per-symbol train objective).
        best_thr, best_train = 0.15, -1e18
        for thr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            model.threshold = thr
            s = sum(objective.evaluate(
                        model.predictions_frame(p["ind"].iloc[p["win"].train_slice]),
                        p["price"].iloc[p["win"].train_slice],
                        p["atr"].iloc[p["win"].train_slice],
                        forward_days, w_acc, w_ret).objective_score
                    for p in preps) / len(preps)
            if s > best_train:
                best_train, best_thr = s, thr
        model.threshold = best_thr
        cand_ho = [objective.evaluate(
                       model.predictions_frame(p["ind"].iloc[p["win"].heldout_slice]),
                       p["price"].iloc[p["win"].heldout_slice],
                       p["atr"].iloc[p["win"].heldout_slice],
                       forward_days, w_acc, w_ret) for p in preps]
        cand_agg = _aggregate(cand_ho, forward_days)
        accepted = cand_agg["objective_score"] > base_agg["objective_score"] + min_improvement
        cfg = model.to_config()
        top = ", ".join(c for c, _ in cfg["top_indicators"][:8])
        return OptimizationResult(
            symbol=None, scope="global",
            status="improved" if accepted else "no_improvement", accepted=accepted,
            seed=seed, objective=obj_meta, baseline_metrics=base_agg, tuned_metrics=cand_agg,
            selected_config={"nn": cfg} if accepted else None, window=window,
            notes=[f"nn-global: {len(preps)} symbols pooled, {cfg['nonzero_count']} non-zero "
                   f"indicator weights; top: {top}",
                   "nn config not persisted via schema store (experimental method)"],
        )

    # --- DL global: GPU-pooled LSTM across ALL symbols simultaneously ---------
    # On RTX 5090 (31 GB VRAM): trains on all NIFTY 50/500 symbols pooled in
    # one forward/backward pass — discovers globally optimal indicator periods.
    if method in ("dl", "lstm", "regime_lstm"):
        try:
            from optimization import dl_optimizer as dl_mod
            print(f"\n  DL global ({method}): pooling {len(preps)} symbols on GPU...")
            dl_model = dl_mod.train_pooled(
                prep_list    = preps,
                forward_days = forward_days,
                seq_len      = dl_seq_len,
                hidden_size  = dl_hidden,
                num_layers   = dl_layers if method != "lstm" else (dl_layers + 1),
                bidirectional= (method == "lstm"),
                vp_window    = vp_window,
                epochs       = dl_epochs,
                lr           = dl_lr,
                w_acc        = w_acc,
                w_ret        = w_ret,
                seed         = seed,
                device       = dl_device,
            )
        except Exception as exc:
            return OptimizationResult(
                symbol=None, scope="global", status="failed", accepted=False,
                seed=seed, objective=obj_meta, baseline_metrics=base_agg,
                notes=[f"DL global failed: {exc}; re-run with --method numgrad"],
            )

        disc = dl_model.discovered_params
        from optimization.config import IndicatorConfig as IC, WeightConfig
        cfg_dict   = dl_model.to_config()
        dl_weights = (WeightConfig.from_dict(cfg_dict["weights"])
                      if cfg_dict.get("weights", {}).get("category")
                      else IC.default().weights)
        dl_cfg = IC(indicator_params=disc, weights=dl_weights, is_default=False,
                    dl_params_discovered=disc, dl_method=method)

        # --- Use the DISCOVERED params for real: recompute each symbol's
        # indicators with the globally-learned params (via the existing TA-Lib
        # IndicatorLibrary) so the learned values actually drive the scored
        # signals — the whole point of the search. base_agg above was computed
        # on the stored (default-param) indicators, so this is an apples-to-apples
        # "discovered vs default" comparison. Falls back to the model's soft
        # aggregation if TA-Lib / recompute is unavailable.
        recompute_ok, extra_note = False, None
        try:
            IndicatorLibrary = _legacy.load_indicator_library_cls()
            lib = IndicatorLibrary()
            for p in preps:
                with _silence():
                    rec = lib.calculate_all_indicators(p["df"].copy(), params=disc)
                rec = rec.reindex(p["ind"].index).fillna(0).astype(float)
                p["ind"]  = rec
                p["qual"] = pd.DataFrame(0.5, index=rec.index, columns=rec.columns)
            recompute_ok = True
        except Exception as exc:
            extra_note = (f"indicator recompute unavailable "
                          f"({type(exc).__name__}: {exc}); scored on stored indicators")

        _TGRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        best_thr, best_train = _TGRID[0], -1e18
        if recompute_ok:
            # Threshold scan on mean TRAIN objective via the real ConsensusPredictor.
            for thr in _TGRID:
                dl_cfg.weights.voting_threshold = thr
                s = sum(_eval(p, dl_cfg, "train", forward_days, w_acc, w_ret,
                              Consensus).objective_score for p in preps) / len(preps)
                if s > best_train:
                    best_train, best_thr = s, thr
            dl_cfg.weights.voting_threshold = best_thr
            cand_ho = [_eval(p, dl_cfg, "heldout", forward_days, w_acc, w_ret, Consensus)
                       for p in preps]
            scoring = "recomputed via TA-Lib with discovered params"
        else:
            for thr in _TGRID:
                dl_model.discovered_threshold = thr
                s = sum(objective.evaluate(
                            dl_model.predictions_frame(p["ind"].iloc[p["win"].train_slice]),
                            p["price"].iloc[p["win"].train_slice],
                            p["atr"].iloc[p["win"].train_slice],
                            forward_days, w_acc, w_ret).objective_score
                        for p in preps) / len(preps)
                if s > best_train:
                    best_train, best_thr = s, thr
            dl_model.discovered_threshold = best_thr
            dl_cfg.weights.voting_threshold = best_thr
            cand_ho = [objective.evaluate(
                           dl_model.predictions_frame(p["ind"].iloc[p["win"].heldout_slice]),
                           p["price"].iloc[p["win"].heldout_slice],
                           p["atr"].iloc[p["win"].heldout_slice],
                           forward_days, w_acc, w_ret) for p in preps]
            scoring = "stored-indicator fallback (TA-Lib unavailable)"

        cand_agg = _aggregate(cand_ho, forward_days)
        accepted = cand_agg["objective_score"] > base_agg["objective_score"] + min_improvement
        status   = "improved" if accepted else "no_improvement"

        # Save DL checkpoint
        checkpoint_path = None
        if accepted and configs_dir and not dry_run:
            import os
            os.makedirs(configs_dir, exist_ok=True)
            checkpoint_path = os.path.join(configs_dir, f"global_{method}_checkpoint.pt")
            dl_model.save_checkpoint(checkpoint_path)
        dl_cfg.dl_model_path = checkpoint_path

        notes = [
            f"{method}-global: {len(preps)} symbols pooled; discovered={disc}",
            f"threshold={best_thr}; train_score={best_train:.4f}; scoring={scoring}",
        ]
        if extra_note:
            notes.append(extra_note)
        if checkpoint_path is not None:
            notes.append(f"checkpoint: {checkpoint_path}")

        result = OptimizationResult(
            symbol=None, scope="global", status=status, accepted=accepted, seed=seed,
            objective=obj_meta, baseline_metrics=base_agg, tuned_metrics=cand_agg,
            selected_config=dl_cfg.to_dict() if accepted else None, window=window,
            notes=notes,
        )
        if accepted and configs_dir and not dry_run:
            result.artifact_path = store.save(result, configs_dir,
                                              created_at or "1970-01-01T00:00:00Z")
        return result



    best_wcfg, _ = weight_optimizer.optimize(baseline, gscore_train, seed=seed)
    candidate = baseline.copy(); candidate.is_default = False
    candidate.weights = best_wcfg.weights

    cand_heldout = [_eval(p, candidate, "heldout", forward_days, w_acc, w_ret, Consensus)
                    for p in preps]
    cand_agg = _aggregate(cand_heldout, forward_days)

    accepted = cand_agg["objective_score"] > base_agg["objective_score"] + min_improvement
    status = "improved" if accepted else "no_improvement"

    result = OptimizationResult(
        symbol=None, scope="global", status=status, accepted=accepted, seed=seed,
        objective=obj_meta, baseline_metrics=base_agg, tuned_metrics=cand_agg,
        selected_config=candidate.to_dict() if accepted else None, window=window,
    )
    if accepted and configs_dir and not dry_run:
        result.artifact_path = store.save(result, configs_dir,
                                          created_at or "1970-01-01T00:00:00Z")
    return result


# --- batch -------------------------------------------------------------------
def _run_one_override(args) -> OptimizationResult:
    """Top-level (picklable) worker: per-symbol override judged vs global config."""
    (symbol, csv_path, kwargs, global_cfg_dict) = args
    try:
        baseline_cfg = (IndicatorConfig.from_dict(global_cfg_dict)
                        if global_cfg_dict else None)
        return run_symbol(symbol, csv_path, baseline_config=baseline_cfg, **kwargs)
    except Exception as exc:  # per-symbol failure must not abort the batch (FR-012)
        return OptimizationResult(
            symbol=symbol, scope="symbol", status="failed", accepted=False,
            seed=kwargs.get("seed", 42),
            objective={"forward_days": kwargs.get("forward_days", [5, 10]),
                       "w_acc": kwargs.get("w_acc", 0.5), "w_ret": kwargs.get("w_ret", 0.5),
                       "embargo_bars": 0},
            error=f"{type(exc).__name__}: {exc}",
        )


def run_batch(input_dir: str, forward_days: Optional[List[int]] = None,
              w_acc: float = 0.5, w_ret: float = 0.5, heldout_frac: float = 0.3,
              embargo: Optional[int] = None, tune: str = "both", budget: int = 100,
              min_improvement: float = 0.0, seed: int = 42, jobs: int = 1,
              configs_dir: Optional[str] = None, created_at: Optional[str] = None,
              dry_run: bool = False, method: str = "numgrad",
              nn_l1: float = 1e-3, nn_l2: float = 1e-3,
              nn_lr: float = 0.1, nn_epochs: int = 400,
              dl_epochs: int = 500,
              dl_lr: float = 1e-3,
              dl_device: str = "cuda",
              dl_hidden: int = 256,
              dl_layers: int = 3,
              dl_seq_len: int = 60,
              vp_window: int = 120):
    """Return (global_result, [per_symbol_results]) covering every targeted symbol."""
    forward_days = forward_days or [5, 10]
    symbols_paths = data_loader.list_symbols(input_dir)

    global_result = run_global(
        symbols_paths, forward_days=forward_days, w_acc=w_acc, w_ret=w_ret,
        heldout_frac=heldout_frac, embargo=embargo, min_improvement=min_improvement,
        seed=seed, configs_dir=configs_dir, created_at=created_at, dry_run=dry_run,
        method=method, nn_l1=nn_l1, nn_l2=nn_l2, nn_lr=nn_lr, nn_epochs=nn_epochs,
        dl_epochs=dl_epochs, dl_lr=dl_lr, dl_device=dl_device,
        dl_hidden=dl_hidden, dl_layers=dl_layers, dl_seq_len=dl_seq_len,
        vp_window=vp_window)

    # numgrad global config is an IndicatorConfig dict usable as a per-symbol
    # baseline; an NN global config has a different shape, so NN per-symbol
    # overrides are judged against the default instead.
    override_baseline = global_result.selected_config if method == "numgrad" else None

    # Per-symbol overrides judged against the global config (numgrad) / default (nn).
    base_kwargs = dict(forward_days=forward_days, w_acc=w_acc, w_ret=w_ret,
                       heldout_frac=heldout_frac, embargo=embargo, tune=tune,
                       budget=budget, min_improvement=min_improvement, seed=seed,
                       config_level="symbol", configs_dir=configs_dir,
                       created_at=created_at, dry_run=dry_run, method=method,
                       nn_l1=nn_l1, nn_l2=nn_l2, nn_lr=nn_lr, nn_epochs=nn_epochs,
                       dl_epochs=dl_epochs, dl_lr=dl_lr, dl_device=dl_device,
                       dl_hidden=dl_hidden, dl_layers=dl_layers, dl_seq_len=dl_seq_len,
                       vp_window=vp_window)
    work = [(sym, path, dict(base_kwargs), override_baseline) for sym, path in symbols_paths]

    results: List[OptimizationResult] = []
    if jobs and jobs > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_run_one_override, work))
    else:
        results = [_run_one_override(w) for w in work]

    return global_result, results


# --- run summary (T033) ------------------------------------------------------
def build_run_summary(run_id: str, scope: str, seed: int, objective_meta: dict,
                      global_result: Optional[OptimizationResult],
                      per_symbol: List[OptimizationResult], created_at: str) -> dict:
    def _delta(r):
        if r.tuned_metrics and r.baseline_metrics:
            return r.tuned_metrics["objective_score"] - r.baseline_metrics["objective_score"]
        return None

    totals = {"targeted": len(per_symbol), "improved": 0, "no_improvement": 0,
              "skipped": 0, "failed": 0}
    symbols = []
    for r in per_symbol:
        totals[r.status] = totals.get(r.status, 0) + 1
        symbols.append({
            "symbol": r.symbol, "status": r.status, "artifact_path": r.artifact_path,
            "heldout_objective": (r.tuned_metrics or {}).get("objective_score") if r.tuned_metrics else None,
            "baseline_objective": (r.baseline_metrics or {}).get("objective_score") if r.baseline_metrics else None,
            "improvement_delta": _delta(r),
            "reason": r.skip_reason or r.error,
        })

    gr = None
    if global_result is not None:
        gr = {
            "status": global_result.status,
            "artifact_path": global_result.artifact_path,
            "heldout_objective": (global_result.tuned_metrics or {}).get("objective_score")
            if global_result.tuned_metrics else None,
            "baseline_objective": (global_result.baseline_metrics or {}).get("objective_score")
            if global_result.baseline_metrics else None,
        }

    return {
        "run_id": run_id, "scope": scope, "seed": seed,
        "objective": {
            "forward_days": objective_meta["forward_days"], "w_acc": objective_meta["w_acc"],
            "w_ret": objective_meta["w_ret"], "embargo_bars": objective_meta["embargo_bars"],
        },
        "global_result": gr, "totals": totals, "symbols": symbols, "created_at": created_at,
    }
