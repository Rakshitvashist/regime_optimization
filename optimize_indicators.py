"""
optimize_indicators.py - CLI entry for indicator optimization (feature 001).

Contract: specs/001-indicator-optimization/contracts/cli-optimize.md

This MVP build implements `--scope symbol` (US1). `--scope global` and
`--scope batch` (US3) are wired to a not-yet-implemented stub and will be
completed in tasks T030-T034.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone

from optimization import batch, data_loader, runner


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Optimize indicator params + consensus weights from history.")
    p.add_argument("--scope", choices=["symbol", "global", "batch"], required=True)
    p.add_argument("--symbol", default=None, help="Required when --scope symbol")
    p.add_argument("--input", default="processed_indicators",
                   help="Directory of *_with_indicators.csv")
    p.add_argument("--configs-dir", default="configs")
    p.add_argument("--forward-days", default="5,10",
                   help="Comma-separated forward horizons (default 5,10)")
    p.add_argument("--w-acc", type=float, default=0.5)
    p.add_argument("--w-ret", type=float, default=0.5)
    p.add_argument("--embargo", type=int, default=None)
    p.add_argument("--heldout-frac", type=float, default=0.3)
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--tune", choices=["weights", "params", "both"], default="both")
    p.add_argument("--method", choices=["numgrad", "nn", "dl", "lstm", "regime_lstm"],
                   default="numgrad",
                   help="numgrad: numerical-gradient weight search (default). "
                        "nn: NumPy backprop per-indicator model. "
                        "dl: LSTM+Attention learns optimal indicator periods from history (GPU). "
                        "lstm: Deeper bidirectional LSTM variant. "
                        "regime_lstm: Regime-aware LSTM (trending/ranging/volatile).")
    p.add_argument("--nn-l1", type=float, default=1e-3, help="NN L1 (sparsity/selection)")
    p.add_argument("--nn-l2", type=float, default=1e-3, help="NN L2 (stability)")
    p.add_argument("--nn-lr", type=float, default=0.1, help="NN learning rate")
    p.add_argument("--nn-epochs", type=int, default=300, help="NN training epochs")
    # --- DL (Feature 002) hyperparameters ---
    p.add_argument("--dl-seq-len", type=int, default=60,
                   help="DL: OHLCV lookback window in bars (default 60 ≈ 3 months)")
    p.add_argument("--dl-hidden", type=int, default=256,
                   help="DL: LSTM hidden size (default 256)")
    p.add_argument("--dl-layers", type=int, default=3,
                   help="DL: LSTM number of layers (default 3)")
    p.add_argument("--dl-epochs", type=int, default=500,
                   help="DL: training epochs (default 500)")
    p.add_argument("--dl-lr", type=float, default=1e-3,
                   help="DL: learning rate (default 0.001)")
    p.add_argument("--dl-device", default="cuda",
                   help="DL: compute device — 'cuda' (RTX 5090, default) or 'cpu'")
    p.add_argument("--vp-window", type=int, default=120,
                   help="Volume-profile lookback window in bars (default 120 ≈ 6 months)")
    p.add_argument("--min-improvement", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _forward_days(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def _find_csv(input_dir: str, symbol: str):
    for sym, path in data_loader.list_symbols(input_dir):
        if sym.upper() == symbol.upper():
            return path
    return None


def _print_symbol_result(res, w_acc, w_ret, forward_days):
    print(f"Symbol: {res.symbol} | seed={res.seed} | "
          f"objective: {w_acc}*acc + {w_ret}*ret | horizons={forward_days} | "
          f"embargo={res.objective['embargo_bars']}")
    if res.status == "skipped":
        print(f"  status: SKIPPED — {res.skip_reason}")
        return
    if res.status == "failed":
        print(f"  status: FAILED — {res.error}")
        return
    b = res.baseline_metrics["objective_score"]
    t = res.tuned_metrics["objective_score"]
    print(f"  baseline  held-out objective: {b:.4f}  hit_ratio={res.baseline_metrics['hit_ratio']}")
    print(f"  tuned     held-out objective: {t:.4f}  hit_ratio={res.tuned_metrics['hit_ratio']}")
    print(f"  status: {res.status.upper()} (delta={t - b:+.4f})")
    if res.artifact_path:
        print(f"  saved: {res.artifact_path}")
    for n in res.notes:
        print(f"  note: {n}")


def main(argv=None) -> int:
    args = _parse_args(argv)
    forward_days = _forward_days(args.forward_days)

    if args.scope == "symbol":
        if not args.symbol:
            print("ERROR: --symbol is required when --scope symbol", file=sys.stderr)
            return 1
        if not os.path.isdir(args.input):
            print(f"ERROR: input dir not found: {args.input}", file=sys.stderr)
            return 1
        csv_path = _find_csv(args.input, args.symbol)
        if csv_path is None:
            print(f"ERROR: no *_with_indicators.csv for symbol '{args.symbol}' in {args.input}",
                  file=sys.stderr)
            return 1
        res = runner.run_symbol(
            args.symbol, csv_path, forward_days=forward_days,
            w_acc=args.w_acc, w_ret=args.w_ret, heldout_frac=args.heldout_frac,
            embargo=args.embargo, tune=args.tune, budget=args.budget,
            min_improvement=args.min_improvement, seed=args.seed,
            configs_dir=args.configs_dir,
            created_at=datetime.now(timezone.utc).isoformat(),
            dry_run=args.dry_run, method=args.method,
            nn_l1=args.nn_l1, nn_l2=args.nn_l2, nn_lr=args.nn_lr, nn_epochs=args.nn_epochs,
            dl_epochs=args.dl_epochs, dl_lr=args.dl_lr, dl_device=args.dl_device,
            dl_hidden=args.dl_hidden, dl_layers=args.dl_layers, dl_seq_len=args.dl_seq_len,
            vp_window=args.vp_window,
        )
        _print_symbol_result(res, args.w_acc, args.w_ret, forward_days)
        return 0

    # --- global / batch (US3) ----------------------------------------------
    if not os.path.isdir(args.input):
        print(f"ERROR: input dir not found: {args.input}", file=sys.stderr)
        return 1
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = f"{args.scope}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"

    if args.scope == "global":
        symbols_paths = data_loader.list_symbols(args.input)
        gres = batch.run_global(
            symbols_paths, forward_days=forward_days, w_acc=args.w_acc, w_ret=args.w_ret,
            heldout_frac=args.heldout_frac, embargo=args.embargo,
            min_improvement=args.min_improvement, seed=args.seed,
            configs_dir=None if args.dry_run else args.configs_dir,
            created_at=created_at, dry_run=args.dry_run, method=args.method,
            nn_l1=args.nn_l1, nn_l2=args.nn_l2, nn_lr=args.nn_lr, nn_epochs=args.nn_epochs,
            dl_epochs=args.dl_epochs, dl_lr=args.dl_lr, dl_device=args.dl_device,
            dl_hidden=args.dl_hidden, dl_layers=args.dl_layers, dl_seq_len=args.dl_seq_len,
            vp_window=args.vp_window)
        print(f"Global config ({args.method}) over {len(symbols_paths)} symbols | seed={args.seed}")
        if gres.status == "skipped":
            print(f"  SKIPPED — {gres.skip_reason}")
            return 0
        print(f"  baseline held-out objective: {gres.baseline_metrics['objective_score']:.4f}")
        print(f"  tuned    held-out objective: {gres.tuned_metrics['objective_score']:.4f}")
        print(f"  status: {gres.status.upper()}")
        if gres.artifact_path:
            print(f"  saved: {gres.artifact_path}")
        for n in gres.notes:
            print(f"  note: {n}")
        return 0

    # scope == batch
    global_result, results = batch.run_batch(
        args.input, forward_days=forward_days, w_acc=args.w_acc, w_ret=args.w_ret,
        heldout_frac=args.heldout_frac, embargo=args.embargo, tune=args.tune,
        budget=args.budget, min_improvement=args.min_improvement, seed=args.seed,
        jobs=args.jobs, configs_dir=None if args.dry_run else args.configs_dir,
        created_at=created_at, dry_run=args.dry_run, method=args.method,
        nn_l1=args.nn_l1, nn_l2=args.nn_l2, nn_lr=args.nn_lr, nn_epochs=args.nn_epochs,
        dl_epochs=args.dl_epochs, dl_lr=args.dl_lr, dl_device=args.dl_device,
        dl_hidden=args.dl_hidden, dl_layers=args.dl_layers, dl_seq_len=args.dl_seq_len,
        vp_window=args.vp_window)

    obj_meta = {"forward_days": forward_days, "w_acc": args.w_acc, "w_ret": args.w_ret,
                "embargo_bars": int(max(forward_days) if args.embargo is None else args.embargo)}
    summary = batch.build_run_summary(run_id, "batch", args.seed, obj_meta,
                                      global_result, results, created_at)

    if not args.dry_run:
        os.makedirs("runs", exist_ok=True)
        with open(os.path.join("runs", f"{run_id}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    _print_batch_summary(summary, global_result, results)
    return 2 if summary["totals"]["failed"] > 0 else 0


def _print_batch_summary(summary, global_result, results):
    o = summary["objective"]
    t = summary["totals"]
    print(f"Scope: batch | seed={summary['seed']} | "
          f"objective: {o['w_acc']}*acc + {o['w_ret']}*ret | "
          f"horizons={o['forward_days']} | embargo={o['embargo_bars']}")
    print(f"Universe: {t['targeted']} symbols")
    print(f"  improved:       {t['improved']}")
    print(f"  no_improvement: {t['no_improvement']}")
    print(f"  skipped:        {t['skipped']}")
    print(f"  failed:         {t['failed']}")
    if global_result is not None and global_result.status != "skipped":
        gb = global_result.baseline_metrics["objective_score"]
        gt = global_result.tuned_metrics["objective_score"]
        print(f"Global config: {global_result.status} (held-out {gt:.4f} vs default {gb:.4f})"
              + (f" -> {global_result.artifact_path}" if global_result.artifact_path else ""))
    failures = [s for s in summary["symbols"] if s["status"] == "failed"]
    skips = [s for s in summary["symbols"] if s["status"] == "skipped"]
    if failures:
        print("Failures:")
        for s in failures:
            print(f"  - {s['symbol']}: {s['reason']}")
    if skips:
        print("Skipped:")
        for s in skips:
            print(f"  - {s['symbol']}: {s['reason']}")


if __name__ == "__main__":
    raise SystemExit(main())
