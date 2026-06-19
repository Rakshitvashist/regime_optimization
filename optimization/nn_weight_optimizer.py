"""
Pure-NumPy backprop optimizer: a LINEAR per-indicator model with L1+L2.

This is a genuine neural network trained by hand-coded backpropagation (forward
pass -> loss -> chain-rule backward pass), with NO autograd framework / PyTorch.

Model (per day t, x_t = the ~400 indicator signals already in [-1, 1]):

        score_t = tanh( W · x_t + b )            # one linear layer + tanh

Loss (differentiable surrogate of the hybrid objective): reward conviction in the
correct direction, i.e. align the signed/confident score with the realized
ATR-relative forward return r_t, plus regularization:

        L = -mean( score_t · r_t )  +  l1·‖W‖₁  +  l2·‖W‖₂²

- L2 keeps weights small/stable; L1 drives many weights to ~0 -> SPARSE weights,
  which is automatic indicator selection ("which indicators are actually useful").
- Deterministic: weights init at 0, full-batch gradient descent with momentum,
  no randomness -> reproducible.

Trains on the TRAIN slice only; the runner selects on the held-out window with the
same embargo, so there is no look-ahead (Constitution Principle II).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class NNModel:
    columns: List[str]                 # indicator column order for W
    W: np.ndarray                      # shape (n_indicators,)
    b: float
    threshold: float = 0.15
    history: List[float] = field(default_factory=list)  # training loss per epoch

    # --- inference -----------------------------------------------------------
    def scores(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.W + self.b)

    def predictions_frame(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        """Build a predictions frame (signal/confidence/consensus_score) for scoring."""
        X = indicators_df[self.columns].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        score = self.scores(X)
        signal = np.where(score > self.threshold, 1.0,
                          np.where(score < -self.threshold, -1.0, 0.0))
        return pd.DataFrame(
            {"signal": signal, "confidence": np.abs(score), "consensus_score": score},
            index=indicators_df.index,
        )

    def weights_dict(self) -> Dict[str, float]:
        return {c: float(w) for c, w in zip(self.columns, self.W)}

    def top_indicators(self, k: int = 15, eps: float = 1e-4) -> List[Tuple[str, float]]:
        """Surviving (non-zero) indicators ranked by |weight| — the selection result."""
        pairs = [(c, float(w)) for c, w in zip(self.columns, self.W) if abs(w) > eps]
        pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return pairs[:k]

    def to_config(self) -> dict:
        """Serializable config block for a tuned NN result."""
        return {
            "method": "nn",
            "per_indicator_weights": self.weights_dict(),
            "bias": float(self.b),
            "threshold": float(self.threshold),
            "nonzero_count": int(np.sum(np.abs(self.W) > 1e-4)),
            "top_indicators": self.top_indicators(),
        }


def _forward_targets(price: pd.Series, atr: pd.Series, forward_days, clip: float = 5.0):
    """ATR-relative forward return per day, averaged over horizons; NaN at the tail."""
    r = np.zeros(len(price))
    counts = np.zeros(len(price))
    a = atr.replace(0.0, np.nan).to_numpy(dtype=float)
    p = price.to_numpy(dtype=float)
    for d in forward_days:
        fwd = np.full(len(p), np.nan)
        fwd[:-d] = p[d:] - p[:-d]
        rr = fwd / a
        mask = np.isfinite(rr)
        r[mask] += rr[mask]
        counts[mask] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(counts > 0, r / np.maximum(counts, 1), np.nan)
    return np.clip(r, -clip, clip)


def fit_arrays(columns: List[str], X: np.ndarray, r: np.ndarray,
               l1: float = 1e-3, l2: float = 1e-3, lr: float = 0.1,
               epochs: int = 300, momentum: float = 0.9,
               threshold: float = 0.15) -> NNModel:
    """Core hand-coded backprop fit on already-masked (finite) arrays.

    Used for both a single symbol and the POOLED global model (X/r stacked across
    many symbols' train rows).
    """
    n, k = X.shape
    W = np.zeros(k)          # deterministic init
    b = 0.0
    vW = np.zeros(k)
    vb = 0.0
    history: List[float] = []
    if n == 0:
        return NNModel(columns=columns, W=W, b=b, threshold=threshold, history=history)

    for _ in range(epochs):
        # ---- forward ----
        pre = X @ W + b                      # (n,)
        score = np.tanh(pre)                 # (n,)
        data_loss = -np.mean(score * r)
        loss = data_loss + l1 * np.sum(np.abs(W)) + l2 * np.sum(W * W)
        history.append(float(loss))

        # ---- backward (chain rule by hand) ----
        # dL/dscore = -r/n ; dscore/dpre = 1 - tanh^2 = 1 - score^2
        dpre = (-r / n) * (1.0 - score * score)      # (n,)
        gW = X.T @ dpre + l1 * np.sign(W) + 2.0 * l2 * W
        gb = float(np.sum(dpre))

        # ---- momentum gradient descent ----
        vW = momentum * vW - lr * gW
        vb = momentum * vb - lr * gb
        W = W + vW
        b = b + vb

    return NNModel(columns=columns, W=W, b=b, threshold=threshold, history=history)


def _symbol_arrays(indicators_df: pd.DataFrame, price: pd.Series, atr: pd.Series,
                   forward_days, columns: List[str]):
    """(X, r) for one symbol's slice over a fixed column order, finite-masked."""
    X = np.nan_to_num(indicators_df[columns].to_numpy(dtype=float), nan=0.0)
    r = _forward_targets(price, atr, forward_days)
    mask = np.isfinite(r)
    return X[mask], r[mask]


def train(indicators_df: pd.DataFrame, price: pd.Series, atr: pd.Series,
          forward_days, l1: float = 1e-3, l2: float = 1e-3, lr: float = 0.1,
          epochs: int = 300, momentum: float = 0.9, threshold: float = 0.15,
          seed: int = 42) -> NNModel:
    """Fit the per-indicator model by backprop on a single symbol's (train) slice."""
    columns = list(indicators_df.columns)
    X, r = _symbol_arrays(indicators_df, price, atr, forward_days, columns)
    return fit_arrays(columns, X, r, l1=l1, l2=l2, lr=lr, epochs=epochs,
                      momentum=momentum, threshold=threshold)


def train_pooled(prep_list, forward_days, columns: List[str],
                 l1: float = 1e-3, l2: float = 1e-3, lr: float = 0.1,
                 epochs: int = 400, momentum: float = 0.9,
                 threshold: float = 0.15) -> NNModel:
    """Fit ONE global model on TRAIN rows pooled across all symbols (more data -> the
    regime where ~400 weights are justified). `prep_list` items are dicts with
    keys ind/price/atr/win (as built by batch._prepare)."""
    Xs, rs = [], []
    for p in prep_list:
        tr = p["win"].train_slice
        Xi, ri = _symbol_arrays(p["ind"].iloc[tr], p["price"].iloc[tr],
                                p["atr"].iloc[tr], forward_days, columns)
        if len(ri):
            Xs.append(Xi)
            rs.append(ri)
    if not Xs:
        return NNModel(columns=columns, W=np.zeros(len(columns)), b=0.0, threshold=threshold)
    X = np.vstack(Xs)
    r = np.concatenate(rs)
    return fit_arrays(columns, X, r, l1=l1, l2=l2, lr=lr, epochs=epochs,
                      momentum=momentum, threshold=threshold)
