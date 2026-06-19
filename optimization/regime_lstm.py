"""
optimization/regime_lstm.py  —  Regime-Aware Parameter Predictor

WHAT IT DOES:
  Different market regimes need different indicator settings:
    TRENDING  → shorter RSI (8-12), wider BB std (2.5+), longer MA slow (100-200)
    RANGING   → standard RSI (14-21), tighter BB (1.5-2.0), shorter lookbacks
    VOLATILE  → wider ATR multipliers, conservative voting thresholds

  This module adds a second LSTM head that explicitly detects the regime
  (trending / ranging / volatile) and gates separate parameter MLPs for each
  regime. Final parameters = weighted sum of regime-specific params, weighted
  by predicted regime probabilities.

  Architecture:
    OHLCV window (batch, seq_len, 9)
        ↓
    Shared LSTM encoder (hidden=256, layers=3)
        ↓
    ┌──────────────────────────────────────┐
    │  Regime Head (softmax → 3 probs)     │  trending / ranging / volatile
    │  Param Head ×3 (one per regime)      │  → optimal periods per regime
    │  Weight Head ×3                      │  → category weights per regime
    └──────────────────────────────────────┘
        ↓
    Gated output = Σ regime_prob[i] * regime_params[i]

USAGE:
  --method regime_lstm
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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

from optimization.dl_optimizer import (
    DLOptimizerUnavailable, DLModel, PARAM_NAMES, N_PARAMS, PARAM_BOUNDS,
    _require_torch, _ohlcv_features, _forward_targets_np, _scale_params,
    _params_to_dict, _training_loss,
)


# ---------------------------------------------------------------------------
# Regime-aware model
# ---------------------------------------------------------------------------

class _RegimeAwareLSTMImpl(nn.Module):
    """
    Regime-gated LSTM parameter predictor.

    Three regimes: 0=trending, 1=ranging, 2=volatile
    """
    N_REGIMES    = 3
    REGIME_NAMES = ["trending", "ranging", "volatile"]

    def __init__(self, input_size: int = 9, hidden_size: int = 256,
                 num_layers: int = 3, n_params: int = N_PARAMS,
                 n_categories: int = 6, dropout: float = 0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)

        # Shared encoder
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.lstm_norm = nn.LayerNorm(hidden_size)

        # Regime detector
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.N_REGIMES),
            nn.Softmax(dim=-1),            # regime probabilities
        )

        # Per-regime parameter heads
        self.param_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, n_params),
                nn.Sigmoid(),              # raw params in [0,1]
            )
            for _ in range(self.N_REGIMES)
        ])

        # Per-regime category weight heads
        self.weight_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, n_categories),
                nn.Softplus(),
            )
            for _ in range(self.N_REGIMES)
        ])

        # Threshold
        self.threshold_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            params:       (batch, N_PARAMS) gated by regime probs
            cat_weights:  (batch, n_categories) gated
            threshold:    (batch, 1)
            regime_probs: (batch, N_REGIMES) — for interpretability / logging
        """
        x = self.input_norm(x)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden)
        summary = self.lstm_norm(out[:, -1, :])   # last time step

        regime_probs = self.regime_head(summary)  # (batch, N_REGIMES)

        # Compute per-regime params and weights
        all_params  = torch.stack([h(summary) for h in self.param_heads],  dim=1)  # (B, 3, n_params)
        all_weights = torch.stack([h(summary) for h in self.weight_heads], dim=1)  # (B, 3, n_cat)

        # Gate: weighted sum by regime probabilities
        rp = regime_probs.unsqueeze(-1)   # (B, 3, 1)
        gated_params  = (all_params  * rp).sum(dim=1)   # (B, n_params)
        gated_weights = (all_weights * rp).sum(dim=1)   # (B, n_cat)

        threshold = self.threshold_head(summary)         # (B, 1)

        return gated_params, gated_weights, threshold, regime_probs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    indicators_df: pd.DataFrame,
    price: pd.Series,
    atr: pd.Series,
    forward_days: List[int],
    train_slice: slice,
    heldout_slice: slice,
    seq_len:      int   = 60,
    hidden_size:  int   = 256,
    num_layers:   int   = 3,
    dropout:      float = 0.2,
    vp_window:    int   = 120,
    extra_indicator_cols=None,
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
    Train the regime-aware LSTM. Returns a DLModel compatible with runner.py.
    """
    _require_torch()

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use the full frame (OHLCV + indicators) when available, else a proxy.
    if "close" not in indicators_df.columns:
        feat_df = pd.DataFrame({
            "open": price.values, "high": price.values * 1.005,
            "low": price.values * 0.995, "close": price.values,
            "volume": np.ones(len(price)),
        }, index=price.index)
    else:
        feat_df = indicators_df

    windows_all = _ohlcv_features(feat_df, seq_len, vp_window=vp_window,
                                  extra_indicator_cols=extra_indicator_cols)
    n_feat = windows_all.shape[-1]
    close_arr   = price.values.astype(np.float32)
    atr_arr     = atr.values.astype(np.float32)
    targets_all = _forward_targets_np(close_arr, atr_arr, forward_days)

    tr_idx  = list(range(*train_slice.indices(len(indicators_df))))
    X_train = windows_all[tr_idx]
    T_train = targets_all[tr_idx]
    C_train = close_arr[tr_idx]

    valid   = np.isfinite(T_train)
    X_train = X_train[valid]
    T_train = T_train[valid]

    # Close windows
    close_wins = np.zeros((valid.sum(), seq_len), dtype=np.float32)
    for i, orig_i in enumerate(np.where(valid)[0]):
        start = max(0, orig_i - seq_len + 1)
        seg   = close_arr[start: orig_i + 1]
        close_wins[i, -len(seg):] = seg

    if len(X_train) < 30:
        raise ValueError(f"Insufficient samples: {len(X_train)}")

    X_t = torch.tensor(X_train, dtype=torch.float32, device=dev)
    T_t = torch.tensor(T_train, dtype=torch.float32, device=dev)
    C_t = torch.tensor(close_wins, dtype=torch.float32, device=dev)

    model = _RegimeAwareLSTMImpl(
        input_size=n_feat, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
    ).to(dev)

    use_amp = device == "cuda" and torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    N = len(X_t)
    history: List[float] = []

    try:
        from tqdm import tqdm
        it = tqdm(range(epochs), desc="Regime-LSTM training", unit="epoch")
    except ImportError:
        it = range(epochs)

    model.train()
    for epoch in it:
        perm = torch.randperm(N, device=dev)
        nb   = max(1, N // batch_size)
        ep_loss = 0.0
        for b in range(nb):
            idx = perm[b * batch_size: (b + 1) * batch_size]
            xb, tb, cb = X_t[idx], T_t[idx], C_t[idx]
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    gated_p, gated_w, thr, _ = model(xb)
                    # Compute soft direction score using gated params
                    from optimization.dl_optimizer import _soft_direction_score, _scale_params
                    scaled = _scale_params(gated_p)
                    from optimization.differentiable_indicators import rsi, ema, bb_position
                    c = cb
                    rsi_v  = rsi(c, scaled["rsi_period"])[:, -1]
                    bb_pos = bb_position(c, scaled["bb_period"], scaled["bb_std"])[:, -1]
                    ma_f   = ema(c, scaled["ma_fast"])[:, -1]
                    ma_s   = ema(c, scaled["ma_slow"])[:, -1]
                    ma_x   = torch.tanh((ma_f - ma_s) / (c[:, -1] + 1e-8) * 20)
                    score  = ((rsi_v - 50) / 50 + bb_pos * 2 - 1 + ma_x) / 3
                    vm     = tb.isfinite()
                    if vm.sum() == 0:
                        continue
                    soft_acc   = torch.sigmoid(score[vm] * tb[vm] * 5).mean()
                    ret_reward = torch.tanh(score[vm] * tb[vm]).mean()
                    loss = -(w_acc * soft_acc + w_ret * ret_reward) + weight_decay * (gated_p - 0.5).pow(2).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                gated_p, gated_w, thr, _ = model(xb)
                from optimization.dl_optimizer import _scale_params
                from optimization.differentiable_indicators import rsi, ema, bb_position
                scaled = _scale_params(gated_p)
                c = cb
                rsi_v  = rsi(c, scaled["rsi_period"])[:, -1]
                bb_pos = bb_position(c, scaled["bb_period"], scaled["bb_std"])[:, -1]
                ma_f   = ema(c, scaled["ma_fast"])[:, -1]
                ma_s   = ema(c, scaled["ma_slow"])[:, -1]
                ma_x   = torch.tanh((ma_f - ma_s) / (c[:, -1] + 1e-8) * 20)
                score  = ((rsi_v - 50) / 50 + bb_pos * 2 - 1 + ma_x) / 3
                vm     = tb.isfinite()
                if vm.sum() == 0:
                    continue
                soft_acc   = torch.sigmoid(score[vm] * tb[vm] * 5).mean()
                ret_reward = torch.tanh(score[vm] * tb[vm]).mean()
                loss = -(w_acc * soft_acc + w_ret * ret_reward) + weight_decay * (gated_p - 0.5).pow(2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            ep_loss += loss.item()
        scheduler.step()
        history.append(ep_loss / nb)

    # Extract discovered params + regime breakdown
    model.eval()
    n_recent = min(20, len(X_t))
    with torch.no_grad():
        raw_p, cat_w, thr, regime_probs = model(X_t[-n_recent:])

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

    dominant_regime_idx = int(regime_probs.mean(0).argmax().item())
    regime_name = _RegimeAwareLSTMImpl.REGIME_NAMES[dominant_regime_idx]
    regime_mean = regime_probs.mean(0).tolist()

    if verbose:
        print(f"\n  Regime-LSTM: dominant regime = {regime_name.upper()}")
        for i, rn in enumerate(_RegimeAwareLSTMImpl.REGIME_NAMES):
            print(f"    {rn:12s}: {regime_mean[i]:.1%}")
        print(f"  Discovered params: {disc_params}")

    return DLModel(
        model_state={k: v.cpu() for k, v in model.state_dict().items()},
        input_size=n_feat, hidden_size=hidden_size, num_layers=num_layers,
        bidirectional=False, seq_len=seq_len,
        discovered_params=disc_params, discovered_weights=disc_weights,
        discovered_threshold=disc_threshold, train_history=history,
    )
