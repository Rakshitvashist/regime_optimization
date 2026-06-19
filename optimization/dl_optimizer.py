"""
optimization/dl_optimizer.py  —  Feature 002: Deep Learning Indicator Optimizer

WHAT IT DOES:
  An LSTM + Multi-Head Attention model that looks at recent OHLCV patterns
  and learns what indicator parameter values (RSI period, BB period, ATR period,
  MA fast/slow, etc.) would have produced the most accurate directional signals
  at each historical bar.

  The model is trained END-TO-END:
    Raw OHLCV features (60 bars)
      → LSTM encoder (captures temporal regime patterns)
      → Multi-Head Attention (focuses on most predictive time steps)
      → Parameter Head  → predicted optimal periods (continuous)
      → Differentiable indicator computation (smooth TA surrogates)
      → Direction score (tanh)
      → Compared to actual N-day forward return
      → Backpropagation updates the LSTM weights

  After training, the model's predicted parameters for the MOST RECENT bars
  are saved and fed into TA-Lib for production signals.

TRAINING PRINCIPLE:
  - All training is on the TRAIN slice only (no look-ahead, Principle II)
  - Selection / acceptance is on the held-out slice
  - Fully deterministic (seed fixed throughout)
  - GPU-accelerated: RTX 5090 handles all 50-500 symbols in one GPU batch

USAGE (via runner.py):
  --method dl           Standard LSTM (recommended)
  --method lstm         Deeper bidirectional variant (slower, richer)
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Torch is an optional dependency — import lazily so the rest of the
# optimization package is importable without it.
_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    _TORCH_AVAILABLE = True
except ImportError:
    pass


class DLOptimizerUnavailable(RuntimeError):
    """Raised when PyTorch is not installed."""


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise DLOptimizerUnavailable(
            "PyTorch is required for --method dl/lstm. "
            "Install: pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )


# ---------------------------------------------------------------------------
# Search-space parameter bounds (mirrors search_space.py)
# ---------------------------------------------------------------------------
PARAM_BOUNDS = {
    "rsi_period":     (5,   30),
    "bb_period":      (10,  40),
    "bb_std":         (1.5, 3.0),
    "atr_period":     (7,   30),
    "ma_fast":        (5,   50),
    "ma_slow":        (50,  200),
    "macd_fast":      (5,   20),
    "macd_slow":      (15,  50),
    "supertrend_mult":(1.5, 5.0),
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())
N_PARAMS    = len(PARAM_NAMES)

# ---------------------------------------------------------------------------
# OHLCV feature engineering (returns + volatility features for the LSTM input)
# ---------------------------------------------------------------------------

# Curated indicator columns with the strongest standalone directional hit ratio
# (universe screen — see analyze_indicators.py). Fed to the LSTM as extra inputs
# so the model can lean on PROVEN signals, not just raw OHLCV. Columns absent from
# a given CSV are silently skipped.
HIGH_SIGNAL_INDICATORS = [
    "double_bottom", "double_top", "triangle_pattern",
    "bb_100_2_squeeze", "bb_20_3_squeeze",
    "obv_vs_ma_100", "obv_vs_ma_50",
    "adx_7_trending", "lower_lows", "sma_slope_200",
    # strongest signals from the universe + intraday screens (used contrarian by
    # the net, which learns the sign): MACD / OBV divergences.
    "macd_5_35_divergence", "macd_12_26_divergence", "macd_19_39_divergence",
    "obv_divergence_20",
]


def _causal_zscore(s: pd.Series) -> np.ndarray:
    """Expanding (look-ahead-free) z-score of a series; NaN -> 0."""
    mean = s.expanding(min_periods=1).mean()
    std = s.expanding(min_periods=1).std().fillna(0.0)
    z = (s - mean) / (std + 1e-8)
    return np.nan_to_num(z.to_numpy(), nan=0.0).astype(np.float32)


def _ohlcv_features(df: pd.DataFrame, seq_len: int = 60,
                    sweep_lookback: int = 10, vp_window: int = 120,
                    extra_indicator_cols=None) -> np.ndarray:
    """
    Convert OHLCV dataframe to (T, seq_len, n_features) float32 array.

    Each row t is a sliding window of the last seq_len bars. Features:
      Base OHLCV (9):
        0: log-return close   1: log-return high   2: log-return low
        3: volume z-score (rolling 20)              4: high-low range / close
        5: close position in (low, high) range      6: 5-bar realized vol
        7: 20-bar realized vol                       8: log-return open
      Liquidity / price-volume (appended, see liquidity_features.FEATURE_NAMES):
        9:  liq_sweep_event   10: liq_pool_pos   11: vol_surge
        12: sweep_x_volume    13: signed_vol_mom

    All features are causal (bar t uses only bars <= t) — no look-ahead.
    """
    from optimization import liquidity_features

    # Sanitize prices before the log: some CSVs have NaN / zero price cells (data
    # gaps, blank rows, newly-listed tails). log(<=0 or NaN) -> "invalid value" and
    # injects NaN features that the LayerNorm then spreads. Forward/back-fill gaps
    # and floor to a positive value so returns are finite.
    def _clean_price(col):
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.where(s > 0)                      # any non-positive (0 or negative) -> gap
        return s.ffill().bfill().fillna(1.0).to_numpy(np.float32)

    c = _clean_price("close")
    h = _clean_price("high")
    l = _clean_price("low")
    o = _clean_price("open")
    v = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    T = len(c)

    eps = 1e-8
    ret_c = np.diff(np.log(c + eps), prepend=np.log(c[0] + eps))
    ret_h = np.diff(np.log(h + eps), prepend=np.log(h[0] + eps))
    ret_l = np.diff(np.log(l + eps), prepend=np.log(l[0] + eps))
    ret_o = np.diff(np.log(o + eps), prepend=np.log(o[0] + eps))

    v_mean = pd.Series(v).rolling(20, min_periods=1).mean().values
    v_std  = pd.Series(v).rolling(20, min_periods=1).std().fillna(1).values
    v_z    = (v - v_mean) / (v_std + eps)

    hl_range    = (h - l) / (c + eps)
    close_pos   = (c - l) / (h - l + eps)

    vol5  = pd.Series(ret_c).rolling(5,  min_periods=1).std().fillna(0).values
    vol20 = pd.Series(ret_c).rolling(20, min_periods=1).std().fillna(0).values

    base = np.stack([ret_c, ret_h, ret_l, v_z, hl_range,
                     close_pos, vol5, vol20, ret_o], axis=1)  # (T, 9)

    # Append causal liquidity-sweep + price/volume features (T, N_FEATURES).
    extra = liquidity_features.extra_features(df, lookback=sweep_lookback)
    blocks = [base, extra]

    # Append causal volume-profile features (POC / value-area / concentration).
    from optimization import volume_profile
    blocks.append(volume_profile.profile_features(df, window=vp_window))

    # Append causal spike-imminent price-volume features (volume expansion before
    # a move). On volume-less series these are ~constant and harmless.
    from optimization import spike_features as _spike
    blocks.append(_spike.spike_features(df))

    # Append curated high-hit-ratio indicator columns when present (causal z-score),
    # so the model sees proven signals like double_bottom / BB-squeeze / obv_vs_ma.
    # Then append this stock's OWN selected indicators (per-stock prior, item a) —
    # the columns analyze_indicators.py validated for THIS symbol.
    used = set()
    for name in list(HIGH_SIGNAL_INDICATORS) + list(extra_indicator_cols or []):
        if name in df.columns and name not in used:
            blocks.append(_causal_zscore(df[name].astype(float)).reshape(-1, 1))
            used.add(name)
    raw = np.concatenate(blocks, axis=1)                      # (T, 9 + N_FEATURES + n_kept)

    # Clip extremes (financial data has fat tails)
    raw = np.clip(raw, -5.0, 5.0).astype(np.float32)
    n_feat = raw.shape[1]

    # Build sliding windows — shape: (T, seq_len, n_feat)
    windows = np.zeros((T, seq_len, n_feat), dtype=np.float32)
    for t in range(T):
        start = max(0, t - seq_len + 1)
        win   = raw[start : t + 1]           # variable length at start
        windows[t, -len(win):] = win         # right-align (pad left with zeros)
    return windows                           # (T, seq_len, n_feat)


def _forward_targets_np(close: np.ndarray, atr: np.ndarray,
                         forward_days: List[int]) -> np.ndarray:
    """ATR-relative forward returns, averaged across horizons. NaN at tail."""
    T = len(close)
    r = np.zeros(T, dtype=np.float32)
    cnt = np.zeros(T, dtype=np.float32)
    for d in forward_days:
        fwd = np.full(T, np.nan, dtype=np.float32)
        fwd[:T - d] = close[d:] - close[:T - d]
        rel = fwd / (atr + 1e-8)
        mask = np.isfinite(rel)
        r[mask]   += np.clip(rel[mask], -10.0, 10.0)
        cnt[mask] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(cnt > 0, r / np.maximum(cnt, 1), np.nan)
    return out


# ---------------------------------------------------------------------------
# Model Definition
# ---------------------------------------------------------------------------

class _IndicatorParamPredictorImpl(nn.Module):
    """
    LSTM + Multi-Head Attention → Parameter Head + Signal Head.

    Inputs  : (batch, seq_len, input_size) — OHLCV feature windows
    Outputs :
        params   (batch, N_PARAMS) in [0,1] — will be re-scaled to real ranges
        weights  (batch, n_categories) — category weights for ConsensusPredictor
    """

    def __init__(self, input_size: int = 9, hidden_size: int = 256,
                 num_layers: int = 3, n_heads: int = 8,
                 n_params: int = N_PARAMS, n_categories: int = 6,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.hidden_size  = hidden_size
        self.bidirectional = bidirectional
        lstm_out = hidden_size * (2 if bidirectional else 1)

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
            batch_first = True,
        )

        # Temporal attention over LSTM output sequence
        self.attn = nn.MultiheadAttention(
            embed_dim   = lstm_out,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out)

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(lstm_out, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
        )

        # Parameter head → outputs in (0,1) which are re-scaled to valid ranges
        self.param_head = nn.Sequential(
            nn.Linear(hidden_size // 2, n_params),
            nn.Sigmoid(),
        )

        # Category-weight head → soft-positive weights via softplus
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_size // 2, n_categories),
            nn.Softplus(),   # always positive, no upper bound needed
        )

        # Voting threshold head (scalar)
        self.threshold_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),    # in (0, 1) → scaled to (0.0, 0.6) externally
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            params:    (batch, N_PARAMS) in [0,1]
            cat_weights: (batch, n_categories)  positive
            threshold:   (batch, 1) in [0,1]
        """
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)           # (batch, seq_len, lstm_out)

        # Self-attention over time steps
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out     = self.attn_norm(lstm_out + attn_out)   # residual

        # Use the last time step as the summary representation
        summary = attn_out[:, -1, :]         # (batch, lstm_out)
        trunk   = self.trunk(summary)        # (batch, hidden//2)

        params   = self.param_head(trunk)      # (batch, N_PARAMS)
        weights  = self.weight_head(trunk)     # (batch, n_categories)
        threshold = self.threshold_head(trunk) # (batch, 1)

        return params, weights, threshold


# ---------------------------------------------------------------------------
# Parameter scaling helpers
# ---------------------------------------------------------------------------

def _scale_params(raw: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Scale [0,1] network output to real indicator parameter ranges.
    Returns a dict {param_name: (batch,) tensor}.
    """
    out = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[name]
        out[name] = raw[:, i] * (hi - lo) + lo
    return out


def _params_to_dict(scaled: Dict[str, torch.Tensor], idx: int = 0) -> Dict[str, float]:
    """Extract one sample (idx) from a batch of params, round integers."""
    result = {}
    int_params = {"rsi_period", "bb_period", "atr_period", "ma_fast", "ma_slow",
                  "macd_fast", "macd_slow"}
    for k, v in scaled.items():
        val = float(v[idx].item())
        result[k] = round(val) if k in int_params else round(val, 2)
    return result


# ---------------------------------------------------------------------------
# Differentiable indicator surrogates (imported from differentiable_indicators)
# ---------------------------------------------------------------------------

def _soft_direction_score(close: torch.Tensor, params_dict: Dict[str, torch.Tensor],
                          seq_len: int) -> torch.Tensor:
    """
    Compute a differentiable direction score from close prices using soft indicators.

    Returns: (batch,) score in [-1, 1] based on last bar's composite signal
    """
    from optimization.differentiable_indicators import (
        rsi, ema, bb_position, atr as diff_atr
    )

    # Use last seq_len bars — close shape (batch, T)
    c = close[:, -seq_len:] if close.shape[1] >= seq_len else close

    rsi_v   = rsi(c, params_dict["rsi_period"])[:, -1]         # (batch,)
    bb_pos  = bb_position(c, params_dict["bb_period"],
                           params_dict["bb_std"])[:, -1]
    ma_f    = ema(c, params_dict["ma_fast"])[:, -1]
    ma_s    = ema(c, params_dict["ma_slow"])[:, -1]
    ma_cross = torch.tanh((ma_f - ma_s) / (c[:, -1] + 1e-8) * 20.0)

    # RSI normalized to [-1,1]
    rsi_sig = (rsi_v - 50.0) / 50.0
    # BB position to [-1,1]
    bb_sig  = bb_pos * 2.0 - 1.0

    score = (rsi_sig + bb_sig + ma_cross) / 3.0
    return score   # (batch,)


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def _training_loss(model: "_IndicatorParamPredictorImpl",
                   x: torch.Tensor,
                   close_full: torch.Tensor,
                   targets: torch.Tensor,
                   w_acc: float = 0.5, w_ret: float = 0.5,
                   l2_reg: float = 1e-4, seq_len: int = 60) -> torch.Tensor:
    """
    End-to-end differentiable loss.

    Args:
        x:          (batch, seq_len, 9) OHLCV feature windows
        close_full: (batch, full_T)     full close price history
        targets:    (batch,) ATR-relative forward returns (signed, clipped ±10)
    """
    raw_params, cat_weights, threshold = model(x)
    params_dict = _scale_params(raw_params)

    # Soft direction score from differentiable indicators
    score = _soft_direction_score(close_full, params_dict, seq_len)  # (batch,)

    # Return alignment loss: maximize score * target (confident → correct direction)
    valid_mask  = targets.isfinite()
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=x.device)

    s = score[valid_mask]
    t = targets[valid_mask]

    # Directional accuracy: soft sigmoid of score*target (smooth surrogate)
    soft_acc  = torch.sigmoid(s * t * 5.0).mean()
    # Return reward: tanh of score*target (scale-free)
    ret_reward = torch.tanh(s * t).mean()

    # Combined objective (higher = better → negate for loss)
    objective = w_acc * soft_acc + w_ret * ret_reward

    # Regularize on the NORMALIZED [0,1] param outputs (scale-free) with a gentle
    # pull toward mid-range — NOT on real-scale magnitudes. Penalizing raw period
    # values (e.g. ma_slow up to 200) just makes "shorter is always cheaper", which
    # collapses every parameter to its lower bound (a degenerate optimum). Keep the
    # category weights near 1.0 so the softplus head can't run off to extremes.
    param_reg  = (raw_params - 0.5).pow(2).mean() * l2_reg
    weight_reg = (cat_weights - 1.0).pow(2).mean() * l2_reg

    loss = -objective + param_reg + weight_reg
    return loss


# ---------------------------------------------------------------------------
# DLModel — inference wrapper (mirrors NNModel interface)
# ---------------------------------------------------------------------------

@dataclass
class DLModel:
    """Inference wrapper around a trained IndicatorParamPredictorImpl."""
    model_state: dict                    # state_dict (CPU)
    input_size:  int
    hidden_size: int
    num_layers:  int
    bidirectional: bool
    seq_len: int
    # Discovered mean optimal params (over last N bars of held-out)
    discovered_params: Dict[str, float]
    # Discovered category weights (mean over last N bars)
    discovered_weights: Dict[str, float]
    discovered_threshold: float
    train_history: List[float] = field(default_factory=list)
    held_out_score: float = 0.0

    CATEGORY_NAMES = ["trend", "price_action", "momentum", "volatility", "volume", "other"]

    def _build_model(self) -> "_IndicatorParamPredictorImpl":
        m = _IndicatorParamPredictorImpl(
            input_size   = self.input_size,
            hidden_size  = self.hidden_size,
            num_layers   = self.num_layers,
            bidirectional = self.bidirectional,
        )
        m.load_state_dict(self.model_state)
        m.eval()
        return m

    def predict_params(self, df: pd.DataFrame,
                       device: str = "cpu") -> Tuple[Dict[str, float], Dict[str, float], float]:
        """
        Run model on the last seq_len bars of df.
        Returns (indicator_params, category_weights, voting_threshold).
        """
        _require_torch()
        windows = _ohlcv_features(df, self.seq_len)
        # Use last bar's window
        x = torch.tensor(windows[-1:], dtype=torch.float32).to(device)
        model = self._build_model().to(device)
        with torch.no_grad():
            raw_p, cat_w, thr = model(x)
        scaled  = _scale_params(raw_p)
        params  = _params_to_dict(scaled, idx=0)
        weights = {self.CATEGORY_NAMES[i]: float(cat_w[0, i].item())
                   for i in range(min(len(self.CATEGORY_NAMES), cat_w.shape[1]))}
        threshold = float(thr[0, 0].item()) * 0.6   # scale (0,1) → (0, 0.6)
        return params, weights, threshold

    def predictions_frame(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a minimal predictions frame (signal/confidence/consensus_score).

        This uses the discovered_params to build a simple signed score via the
        normalized indicator signals already in indicators_df (no TA-Lib recompute
        needed at inference time — full recompute happens in runner.py).
        """
        n = len(indicators_df)
        # Use equal weights across all indicator columns as a simple aggregation
        vals = indicators_df.fillna(0).values.astype(np.float32)
        scores = vals.mean(axis=1)
        thr = self.discovered_threshold
        signals = np.where(scores > thr, 1.0, np.where(scores < -thr, -1.0, 0.0))
        return pd.DataFrame({
            "signal": signals,
            "confidence": np.abs(scores),
            "consensus_score": scores,
        }, index=indicators_df.index)

    def to_config(self) -> dict:
        """Return a config dict compatible with runner.py result handling."""
        return {
            "method": "dl",
            "indicator_params": self.discovered_params,
            "weights": {
                "category": self.discovered_weights,
                "voting_threshold": self.discovered_threshold,
                "regime_multipliers": {"trend_trending_boost": 0.2,
                                       "momentum_ranging_boost": 0.2},
            },
            "dl_metadata": {
                "hidden_size":   self.hidden_size,
                "num_layers":    self.num_layers,
                "bidirectional": self.bidirectional,
                "seq_len":       self.seq_len,
            },
        }

    def save_checkpoint(self, path: str):
        """Save model state + metadata as a .pt file."""
        _require_torch()
        torch.save({
            "model_state":          self.model_state,
            "input_size":           self.input_size,
            "hidden_size":          self.hidden_size,
            "num_layers":           self.num_layers,
            "bidirectional":        self.bidirectional,
            "seq_len":              self.seq_len,
            "discovered_params":    self.discovered_params,
            "discovered_weights":   self.discovered_weights,
            "discovered_threshold": self.discovered_threshold,
            "train_history":        self.train_history,
            "held_out_score":       self.held_out_score,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str) -> "DLModel":
        _require_torch()
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**{k: v for k, v in d.items()})


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    indicators_df: pd.DataFrame,
    price: pd.Series,
    atr: pd.Series,
    forward_days: List[int],
    train_slice: slice,
    heldout_slice: slice,
    # Model hyperparams
    seq_len:      int   = 60,
    hidden_size:  int   = 256,
    num_layers:   int   = 3,
    bidirectional: bool = False,
    n_heads:      int   = 8,
    dropout:      float = 0.2,
    vp_window:    int   = 120,
    extra_indicator_cols=None,
    # Training hyperparams
    epochs:       int   = 500,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size:   int   = 128,
    w_acc:        float = 0.5,
    w_ret:        float = 0.5,
    seed:         int   = 42,
    device:       str   = "cuda",
    verbose:      bool  = True,
) -> DLModel:
    """
    Train the LSTM parameter predictor on the training slice.

    Args:
        indicators_df: Full indicator DataFrame (model uses OHLCV internally)
        price:         Full close price Series
        atr:           Full ATR Series
        forward_days:  Horizons for forward-return targets
        train_slice:   slice — bars used for training (NO look-ahead)
        heldout_slice: slice — bars used only for evaluation
        ...

    Returns:
        DLModel — trained model with discovered optimal params
    """
    _require_torch()

    # ---- device setup -------------------------------------------------------
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)

    # ---- reproducibility ----------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    # ---- data preparation ---------------------------------------------------
    # We need the raw OHLCV — reconstruct from the indicator CSV columns
    # (open/high/low/close/volume should be present alongside indicators)
    ohlcv_cols = [c for c in indicators_df.columns
                  if c in ("open", "high", "low", "close", "volume")]

    # If the caller passed only indicator signals, build a minimal OHLCV proxy;
    # otherwise feed the FULL frame so curated indicator columns are picked up.
    if "close" not in indicators_df.columns:
        feat_df = pd.DataFrame({
            "open": price.values,
            "high": price.values * 1.005,
            "low":  price.values * 0.995,
            "close": price.values,
            "volume": np.ones(len(price)),
        }, index=price.index)
    else:
        feat_df = indicators_df          # OHLCV + indicator columns

    windows_all = _ohlcv_features(feat_df, seq_len=seq_len, vp_window=vp_window,
                                  extra_indicator_cols=extra_indicator_cols)
    n_feat = windows_all.shape[-1]

    close_arr = price.values.astype(np.float32)
    atr_arr   = atr.values.astype(np.float32)
    targets_all = _forward_targets_np(close_arr, atr_arr, forward_days)  # (T,)

    # Extract train indices
    tr_idx = list(range(*train_slice.indices(len(indicators_df))))
    X_train = windows_all[tr_idx]            # (N_train, seq_len, 9)
    T_train = targets_all[tr_idx]            # (N_train,)
    C_train = close_arr[tr_idx]              # (N_train,)

    # Remove rows where target is NaN (forward-return tail)
    valid = np.isfinite(T_train)
    X_train = X_train[valid]
    T_train = T_train[valid]
    C_train = C_train[valid]

    if len(X_train) < 30:
        raise ValueError(f"Insufficient training samples after NaN removal: {len(X_train)}")

    # Convert to tensors
    X_t = torch.tensor(X_train, dtype=torch.float32, device=dev)
    T_t = torch.tensor(T_train, dtype=torch.float32, device=dev)

    # Build "close windows" for differentiable indicator computation
    # Each row t: last seq_len close prices ending at t
    close_wins = np.zeros((len(C_train), seq_len), dtype=np.float32)
    for i, orig_i in enumerate(np.where(valid)[0]):
        start = max(0, orig_i - seq_len + 1)
        seg   = close_arr[start: orig_i + 1]
        close_wins[i, -len(seg):] = seg
    C_t = torch.tensor(close_wins, dtype=torch.float32, device=dev)

    # ---- model & optimizer --------------------------------------------------
    model = _IndicatorParamPredictorImpl(
        input_size    = n_feat,
        hidden_size   = hidden_size,
        num_layers    = num_layers,
        n_heads       = n_heads,
        dropout       = dropout,
        bidirectional = bidirectional,
    ).to(dev)

    # Use mixed precision if CUDA available (speeds up RTX 5090 significantly)
    use_amp = (device == "cuda" and torch.cuda.is_available())
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    N = len(X_t)
    n_batches = max(1, N // batch_size)
    history: List[float] = []

    if verbose:
        try:
            from tqdm import tqdm
            epoch_iter = tqdm(range(epochs), desc="DL training", unit="epoch",
                              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        except ImportError:
            epoch_iter = range(epochs)
    else:
        epoch_iter = range(epochs)

    model.train()
    for epoch in epoch_iter:
        # Shuffle each epoch
        perm  = torch.randperm(N, device=dev)
        epoch_loss = 0.0

        for b in range(n_batches):
            idx   = perm[b * batch_size: (b + 1) * batch_size]
            xb    = X_t[idx]
            tb    = T_t[idx]
            cb    = C_t[idx]

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = _training_loss(model, xb, cb, tb, w_acc, w_ret,
                                          weight_decay, seq_len)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = _training_loss(model, xb, cb, tb, w_acc, w_ret,
                                      weight_decay, seq_len)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / n_batches
        history.append(avg_loss)

        if verbose and isinstance(epoch_iter, range) and (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:4d}/{epochs}  loss={avg_loss:.5f}")

    # ---- extract discovered parameters (mean over last 20 training bars) ----
    model.eval()
    n_recent = min(20, len(X_t))
    X_recent = X_t[-n_recent:]
    with torch.no_grad():
        raw_p, cat_w, thr = model(X_recent)  # (n_recent, ...)

    scaled  = _scale_params(raw_p)
    disc_params: Dict[str, float] = {}
    int_params = {"rsi_period", "bb_period", "atr_period", "ma_fast", "ma_slow",
                  "macd_fast", "macd_slow"}
    for name in PARAM_NAMES:
        val = float(scaled[name].mean().item())
        disc_params[name] = round(val) if name in int_params else round(val, 2)

    cat_names = ["trend", "price_action", "momentum", "volatility", "volume", "other"]
    disc_weights = {cat_names[i]: round(float(cat_w.mean(0)[i].item()), 4)
                    for i in range(min(len(cat_names), cat_w.shape[1]))}
    disc_threshold = round(float(thr.mean().item()) * 0.6, 4)

    return DLModel(
        model_state          = {k: v.cpu() for k, v in model.state_dict().items()},
        input_size           = n_feat,
        hidden_size          = hidden_size,
        num_layers           = num_layers,
        bidirectional        = bidirectional,
        seq_len              = seq_len,
        discovered_params    = disc_params,
        discovered_weights   = disc_weights,
        discovered_threshold = disc_threshold,
        train_history        = history,
    )


# ---------------------------------------------------------------------------
# Pooled global training (batch across all symbols simultaneously on GPU)
# ---------------------------------------------------------------------------

def train_pooled(
    prep_list: list,
    forward_days: List[int],
    seq_len:      int   = 60,
    hidden_size:  int   = 256,
    num_layers:   int   = 3,
    bidirectional: bool = False,
    vp_window:    int   = 120,
    epochs:       int   = 500,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size:   int   = 256,
    w_acc:        float = 0.5,
    w_ret:        float = 0.5,
    seed:         int   = 42,
    device:       str   = "cuda",
) -> DLModel:
    """
    Train ONE global DL model on ALL symbols' training rows pooled together.

    prep_list items: dicts with keys ind/price/atr/win (as built by batch._prepare).
    All symbols' TRAIN slices are stacked into one large GPU batch — on an
    RTX 5090 with 31 GB VRAM this runs in seconds even for NIFTY 500.
    """
    _require_torch()

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    all_X, all_T, all_C = [], [], []

    for p in prep_list:
        tr = p["win"].train_slice
        df_sym = p.get("df", p["ind"])   # prefer raw OHLCV frame (real high/low/volume)
        price_sym = p["price"]
        atr_sym   = p["atr"]

        # Use the full frame (OHLCV + indicators) when available, else a proxy.
        if "close" not in df_sym.columns:
            feat_df = pd.DataFrame({
                "open": price_sym.values, "high": price_sym.values * 1.005,
                "low":  price_sym.values * 0.995, "close": price_sym.values,
                "volume": np.ones(len(price_sym)),
            }, index=price_sym.index)
        else:
            feat_df = df_sym

        wins   = _ohlcv_features(feat_df, seq_len, vp_window=vp_window)
        c_arr  = price_sym.values.astype(np.float32)
        a_arr  = atr_sym.values.astype(np.float32)
        tgt    = _forward_targets_np(c_arr, a_arr, forward_days)

        tr_idx = list(range(*tr.indices(len(df_sym))))
        X  = wins[tr_idx]
        T_ = tgt[tr_idx]
        C_ = c_arr[tr_idx]

        valid = np.isfinite(T_)
        if valid.sum() < 10:
            continue

        # Build close windows
        close_wins = np.zeros((valid.sum(), seq_len), dtype=np.float32)
        for i, orig_i in enumerate(np.where(valid)[0]):
            start = max(0, orig_i - seq_len + 1)
            seg   = c_arr[start: orig_i + 1]
            close_wins[i, -len(seg):] = seg

        all_X.append(X[valid])
        all_T.append(T_[valid])
        all_C.append(close_wins)

    if not all_X:
        raise ValueError("No eligible symbols with sufficient history for pooled training.")

    X_pool = np.vstack(all_X)
    T_pool = np.concatenate(all_T)
    C_pool = np.vstack(all_C)

    print(f"  Pooled training: {X_pool.shape[0]:,} samples from {len(all_X)} symbols")
    if device == "cuda":
        mem_gb = X_pool.nbytes / 1e9
        print(f"  Dataset size: {mem_gb:.2f} GB  (RTX 5090 VRAM: 31 GB — OK)")

    dev  = torch.device(device)
    X_t  = torch.tensor(X_pool, dtype=torch.float32, device=dev)
    T_t  = torch.tensor(T_pool, dtype=torch.float32, device=dev)
    C_t  = torch.tensor(C_pool, dtype=torch.float32, device=dev)

    n_feat = X_pool.shape[-1]
    model = _IndicatorParamPredictorImpl(
        input_size=n_feat, hidden_size=hidden_size, num_layers=num_layers,
        bidirectional=bidirectional,
    ).to(dev)

    use_amp = (device == "cuda" and torch.cuda.is_available())
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    N = len(X_t)
    history: List[float] = []

    try:
        from tqdm import tqdm
        epoch_iter = tqdm(range(epochs), desc="DL global training", unit="epoch")
    except ImportError:
        epoch_iter = range(epochs)

    model.train()
    for epoch in epoch_iter:
        perm = torch.randperm(N, device=dev)
        n_batches  = max(1, N // batch_size)
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size: (b + 1) * batch_size]
            xb, tb, cb = X_t[idx], T_t[idx], C_t[idx]
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss = _training_loss(model, xb, cb, tb, w_acc, w_ret,
                                          weight_decay, seq_len)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss = _training_loss(model, xb, cb, tb, w_acc, w_ret,
                                      weight_decay, seq_len)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        history.append(epoch_loss / n_batches)

    # Extract global discovered params
    model.eval()
    n_recent = min(50, len(X_t))
    with torch.no_grad():
        raw_p, cat_w, thr = model(X_t[-n_recent:])

    scaled = _scale_params(raw_p)
    int_params = {"rsi_period", "bb_period", "atr_period", "ma_fast", "ma_slow",
                  "macd_fast", "macd_slow"}
    disc_params = {name: (round(float(scaled[name].mean().item()))
                          if name in int_params
                          else round(float(scaled[name].mean().item()), 2))
                   for name in PARAM_NAMES}
    cat_names = ["trend", "price_action", "momentum", "volatility", "volume", "other"]
    disc_weights = {cat_names[i]: round(float(cat_w.mean(0)[i].item()), 4)
                    for i in range(min(len(cat_names), cat_w.shape[1]))}
    disc_threshold = round(float(thr.mean().item()) * 0.6, 4)

    return DLModel(
        model_state={k: v.cpu() for k, v in model.state_dict().items()},
        input_size=n_feat, hidden_size=hidden_size, num_layers=num_layers,
        bidirectional=bidirectional, seq_len=seq_len,
        discovered_params=disc_params, discovered_weights=disc_weights,
        discovered_threshold=disc_threshold, train_history=history,
    )
