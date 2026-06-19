"""
T023 / T024 - Versioned tuned-config artifact store + precedence resolution.

Write side (T023): persist an ACCEPTED OptimizationResult as a versioned JSON
artifact under configs/global/ or configs/per_symbol/<SYMBOL>/, never overwriting
in place (monotonic vNN). Embeds full provenance (window, seed, objective,
held-out + baseline metrics, injected created_at) so every config is reproducible
and auditable (FR-007; contracts/tuned-config.schema.json).

Read side (T024): `resolve_config(symbol)` returns the config the predictor should
use, by precedence: latest accepted per-symbol override -> latest accepted global
-> default; plus the level label recorded on predictions (FR-008).
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Optional, Tuple

from .config import IndicatorConfig

SCHEMA_VERSION = "1.0.0"
_VER_RE = re.compile(r"\.v(\d+)\.json$")


# --- paths -------------------------------------------------------------------
def _global_dir(configs_dir: str) -> str:
    return os.path.join(configs_dir, "global")


def _symbol_dir(configs_dir: str, symbol: str) -> str:
    return os.path.join(configs_dir, "per_symbol", symbol.upper())


def _versions(directory: str, stem: str):
    """Return sorted (version:int, path) for <stem>.vNN.json files in directory."""
    out = []
    for path in glob.glob(os.path.join(directory, f"{stem}.v*.json")):
        m = _VER_RE.search(path)
        if m:
            out.append((int(m.group(1)), path))
    return sorted(out)


def _next_version(directory: str, stem: str) -> int:
    vs = _versions(directory, stem)
    return (vs[-1][0] + 1) if vs else 1


# --- write -------------------------------------------------------------------
def build_artifact(result, created_at: str) -> dict:
    """Construct a schema-conforming TunedConfigArtifact dict from a result."""
    obj = result.objective
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": result.scope,
        "symbol": result.symbol if result.scope == "symbol" else None,
        "version": 0,  # filled in by save()
        "config": result.selected_config,
        "objective": {
            "forward_days": obj["forward_days"],
            "w_acc": obj["w_acc"],
            "w_ret": obj["w_ret"],
        },
        "window": result.window,
        "seed": result.seed,
        "heldout_metrics": result.tuned_metrics,
        "baseline_metrics": result.baseline_metrics,
        "created_at": created_at,
    }


def save(result, configs_dir: str, created_at: str) -> Optional[str]:
    """Persist an accepted result; returns the artifact path (or None if not accepted)."""
    if not result.accepted or result.selected_config is None:
        return None

    if result.scope == "global":
        directory = _global_dir(configs_dir)
        stem = "global"
    else:
        directory = _symbol_dir(configs_dir, result.symbol)
        stem = result.symbol.upper()

    os.makedirs(directory, exist_ok=True)
    version = _next_version(directory, stem)
    artifact = build_artifact(result, created_at)
    artifact["version"] = version

    path = os.path.join(directory, f"{stem}.v{version:02d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    return path


# --- read / precedence -------------------------------------------------------
def load_latest(configs_dir: str, scope: str, symbol: Optional[str] = None) -> Optional[dict]:
    """Load the highest-version artifact for a scope, or None if absent."""
    if scope == "global":
        directory, stem = _global_dir(configs_dir), "global"
    else:
        directory, stem = _symbol_dir(configs_dir, symbol), symbol.upper()
    vs = _versions(directory, stem)
    if not vs:
        return None
    with open(vs[-1][1], encoding="utf-8") as f:
        return json.load(f)


def resolve_config(symbol: str, configs_dir: str = "configs") -> Tuple[IndicatorConfig, str]:
    """Return (config, level) using precedence: per-symbol -> global -> default (FR-008)."""
    art = load_latest(configs_dir, "symbol", symbol)
    if art is not None:
        return IndicatorConfig.from_dict(art["config"]), "symbol"
    art = load_latest(configs_dir, "global")
    if art is not None:
        return IndicatorConfig.from_dict(art["config"]), "global"
    return IndicatorConfig.default(), "default"
