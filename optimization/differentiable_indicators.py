"""
optimization/differentiable_indicators.py

Smooth, differentiable PyTorch re-implementations of key technical indicators.

PURPOSE:
  During DL model training we need to backpropagate gradients through indicator
  computations so the LSTM can learn which period values are predictive. Standard
  TA-Lib uses C extensions with no gradient support.

  These are TRAINING-ONLY approximations — they trade mathematical exactness for
  differentiability. TA-Lib is still used for all production signals; the DL model
  learns optimal period *ranges* that are then rounded to integers and fed into TA-Lib
  for actual prediction.

SUPPORTED:
  - EMA(close, period)  — exact recursive formula (differentiable via scan)
  - SMA(close, period)  — avg pool approximation
  - RSI(close, period)  — Wilder smoothing with soft-clip
  - ATR(high, low, close, period)  — Wilder ATR
  - BBANDS(close, period, std)  — upper / middle / lower
  - MACD(close, fast, slow, signal)  — triple EMA

All functions accept float-valued 'period' tensors (continuous relaxation) so
gradients flow through the period parameter during training.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema_alpha(period: torch.Tensor) -> torch.Tensor:
    """Wilder multiplier: 2 / (period + 1). period can be float (continuous)."""
    return 2.0 / (period + 1.0)


_MATRIX_MAX_T = 512   # above this, fall back to the O(T) loop (avoids B*T*T memory)


def _ema_loop(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Reference sequential EMA with out[0]=x[0] (no in-place writes)."""
    T = x.shape[1]
    one_m = 1.0 - alpha
    outs = [x[:, 0]]
    for t in range(1, T):
        outs.append(alpha * x[:, t] + one_m * outs[-1])
    return torch.stack(outs, dim=1)


def ema(x: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
    """
    Differentiable EMA, VECTORIZED via the closed form
        out[t] = b^t * x[0] + sum_{k=1..t} a * b^(t-k) * x[k],   a=2/(p+1), b=1-a
    so the whole series is one batched mat-vec instead of T sequential GPU
    kernels — orders of magnitude faster on GPU. Exact (matches the recurrence).

    Args:
        x: (batch, seq_len) ; period: (batch,) or scalar float tensor
    Returns:
        (batch, seq_len) EMA values
    """
    batch, T = x.shape
    alpha = _ema_alpha(period).reshape(-1)             # (batch,)
    if alpha.numel() == 1:
        alpha = alpha.expand(batch)
    if T > _MATRIX_MAX_T:
        return _ema_loop(x, alpha)

    a = alpha.view(batch, 1, 1)
    b = (1.0 - alpha).view(batch, 1, 1)
    t = torch.arange(T, device=x.device, dtype=x.dtype)
    E = t.view(1, T, 1) - t.view(1, 1, T)              # (1,T,T) = t - k
    pos = ((t.view(1, 1, T) >= 1) & (t.view(1, 1, T) <= t.view(1, T, 1)))
    w = a * (b ** E.clamp(min=0)) * pos.to(x.dtype)    # (batch,T,T) k=1..t weights
    ema_k = torch.bmm(w, x.unsqueeze(-1)).squeeze(-1)  # (batch,T)
    col0 = b.view(batch, 1) ** t.view(1, T)            # (batch,T) = b^t  (x[0] term)
    return col0 * x[:, :1] + ema_k


def _ema_zeroinit(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """EMA with zero initial state: out[t] = sum_{k=0..t} a*b^(t-k)*x[k]."""
    batch, T = x.shape
    if T > _MATRIX_MAX_T:                               # loop fallback
        one_m = 1.0 - alpha
        acc = torch.zeros(batch, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            acc = alpha * x[:, t] + one_m * acc
            outs.append(acc)
        return torch.stack(outs, dim=1)
    a = alpha.view(batch, 1, 1)
    b = (1.0 - alpha).view(batch, 1, 1)
    t = torch.arange(T, device=x.device, dtype=x.dtype)
    E = t.view(1, T, 1) - t.view(1, 1, T)              # t - k
    causal = (t.view(1, 1, T) <= t.view(1, T, 1)).to(x.dtype)
    w = a * (b ** E.clamp(min=0)) * causal
    return torch.bmm(w, x.unsqueeze(-1)).squeeze(-1)


def sma(x: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
    """
    Smooth SMA using soft window via exponential kernel.

    For small memory footprint uses a running-sum approximation that degrades
    gracefully to EMA so gradients are always defined.
    """
    # Use EMA as a smooth SMA surrogate — exact SMA is non-differentiable at
    # integer period boundaries; EMA(period) ≈ SMA(period) for training purposes.
    return ema(x, period)


def rsi(close: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
    """
    Differentiable RSI.

    Returns values in [0, 100].
    """
    batch, T = close.shape
    delta = close[:, 1:] - close[:, :-1]   # (batch, T-1)

    gain = F.relu(delta)
    loss = F.relu(-delta)

    # Wilder smoothing: zero-init EMA with alpha = 1/period (VECTORIZED).
    alpha = (1.0 / period).reshape(-1)
    if alpha.numel() == 1:
        alpha = alpha.expand(batch)

    avg_gain = _ema_zeroinit(gain, alpha)              # (batch, T-1)
    avg_loss = _ema_zeroinit(loss, alpha)
    rs = avg_gain / (avg_loss + 1e-8)
    rsi_tail = 100.0 - 100.0 / (1.0 + rs)              # maps to out[1..T-1]

    first = torch.full((batch, 1), 50.0, device=close.device, dtype=close.dtype)
    return torch.cat([first, rsi_tail], dim=1)         # (batch, T)


def atr(high: torch.Tensor, low: torch.Tensor, close: torch.Tensor,
        period: torch.Tensor) -> torch.Tensor:
    """Differentiable Wilder ATR."""
    batch, T = close.shape
    prev_close = torch.cat([close[:, :1], close[:, :-1]], dim=1)

    tr = torch.stack([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], dim=-1).max(dim=-1).values   # (batch, T)

    return ema(tr, period)


def bbands(close: torch.Tensor, period: torch.Tensor,
           std_dev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable Bollinger Bands.

    Returns: (upper, middle, lower) each (batch, seq_len)
    """
    middle = sma(close, period)

    # Rolling std via (E[x²] - E[x]²)^0.5  — approximate via EMA
    close_sq = close ** 2
    mean_sq   = sma(close_sq, period)
    mean      = middle
    variance  = (mean_sq - mean ** 2).clamp(min=1e-8)
    std       = variance.sqrt()

    upper = middle + std_dev.view(-1, 1) * std
    lower = middle - std_dev.view(-1, 1) * std
    return upper, middle, lower


def macd(close: torch.Tensor, fast: torch.Tensor,
         slow: torch.Tensor, signal_period: torch.Tensor
         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable MACD.

    Returns: (macd_line, signal_line, histogram) each (batch, seq_len)
    """
    fast_ema   = ema(close, fast)
    slow_ema   = ema(close, slow)
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def bb_position(close: torch.Tensor, period: torch.Tensor,
                std_dev: torch.Tensor) -> torch.Tensor:
    """BB %B position in [0,1]: (close - lower) / (upper - lower)."""
    upper, _, lower = bbands(close, period, std_dev)
    pos = (close - lower) / (upper - lower + 1e-8)
    return pos.clamp(0.0, 1.0)


def compute_features(close: torch.Tensor, high: torch.Tensor,
                     low: torch.Tensor, volume: torch.Tensor,
                     params: dict) -> torch.Tensor:
    """
    Compute a feature vector using the given (soft/continuous) params.

    Args:
        close, high, low, volume: (batch, seq_len) tensors
        params: dict with keys:
            rsi_period, bb_period, bb_std, atr_period, ma_fast, ma_slow,
            macd_fast, macd_slow  — all (batch,) float tensors

    Returns:
        (batch, seq_len, n_features) — normalized features for the LSTM
    """
    feats = []

    # RSI signal (normalized to [-1, 1])
    r = rsi(close, params["rsi_period"])
    feats.append(((r - 50.0) / 50.0).unsqueeze(-1))

    # BB position (normalized to [-1, 1])
    bb_pos = bb_position(close, params["bb_period"], params["bb_std"])
    feats.append((bb_pos * 2.0 - 1.0).unsqueeze(-1))

    # ATR normalized by close
    at = atr(high, low, close, params["atr_period"])
    atr_norm = (at / (close + 1e-8)).clamp(0, 0.1) / 0.05 - 1.0
    feats.append(atr_norm.unsqueeze(-1))

    # MA crossover (fast vs slow)
    ma_f = ema(close, params["ma_fast"])
    ma_s = ema(close, params["ma_slow"])
    cross = ((ma_f - ma_s) / (close + 1e-8)).clamp(-0.1, 0.1) / 0.05
    feats.append(cross.unsqueeze(-1))

    # MACD histogram (normalized)
    _, _, hist = macd(close, params["macd_fast"], params["macd_slow"],
                      torch.full_like(params["macd_fast"], 9.0))
    hist_norm = (hist / (close + 1e-8)).clamp(-0.05, 0.05) / 0.025
    feats.append(hist_norm.unsqueeze(-1))

    return torch.cat(feats, dim=-1)  # (batch, seq_len, 5)
