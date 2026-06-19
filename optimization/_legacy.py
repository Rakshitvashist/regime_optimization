"""
Loader for the existing modules whose filenames contain a literal space
(`Consensus_predictor .py`, `Hit_ratio_backtester .py`) and are therefore not
importable by normal `import`. Loads them by file path via importlib (the same
approach used in generate_summary.py).
"""
from __future__ import annotations

import importlib.util
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(modname: str, filename: str):
    path = os.path.join(_REPO, filename)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_backtester_cls():
    return _load("hit_ratio_backtester", "Hit_ratio_backtester .py").HitRatioBacktester


def load_consensus_module():
    return _load("consensus_predictor", "Consensus_predictor .py")


def load_indicator_library_cls():
    """Load IndicatorLibrary from start.py. Requires TA-Lib/numba; raises on missing deps."""
    return _load("start", "start.py").IndicatorLibrary
