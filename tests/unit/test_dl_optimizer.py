"""
tests/unit/test_dl_optimizer.py

Unit tests for the Deep Learning Indicator Optimizer (Feature 002).

These tests are designed to pass WITHOUT a GPU and even WITHOUT PyTorch
(the torch-absent case is tested explicitly). When PyTorch is available,
all model/training tests run on CPU to keep CI fast.
"""
from __future__ import annotations

import importlib
import sys
import types
import unittest

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_series(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, 0.01, n)
    prices = 1000.0 * np.exp(np.cumsum(log_returns))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(prices, index=idx, name="close")


def _make_ohlcv_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    close = _make_price_series(n, seed)
    rng = np.random.default_rng(seed + 1)
    df = pd.DataFrame({
        "open":   close.values * (1 + rng.normal(0, 0.003, n)),
        "high":   close.values * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low":    close.values * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close":  close.values,
        "volume": rng.integers(100_000, 1_000_000, n).astype(float),
    }, index=close.index)
    return df


def _make_indicators_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Minimal indicators DataFrame (all float, no TA-Lib needed)."""
    rng = np.random.default_rng(seed)
    cols = ["rsi", "macd", "bb_upper", "bb_lower", "atr", "ma_fast", "ma_slow",
            "trend_score", "momentum_score"]
    data = rng.uniform(-1.0, 1.0, (n, len(cols)))
    idx  = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(data, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# Test: torch unavailable path
# ---------------------------------------------------------------------------

class TestTorchUnavailable:
    """Verify graceful error when torch is not installed."""

    def test_require_torch_raises(self, monkeypatch):
        """_require_torch() should raise DLOptimizerUnavailable when torch absent."""
        # Temporarily shadow torch in sys.modules
        import optimization.dl_optimizer as dlopt
        original = dlopt._TORCH_AVAILABLE
        dlopt._TORCH_AVAILABLE = False
        try:
            with pytest.raises(dlopt.DLOptimizerUnavailable):
                dlopt._require_torch()
        finally:
            dlopt._TORCH_AVAILABLE = original


# ---------------------------------------------------------------------------
# Test: OHLCV feature engineering
# ---------------------------------------------------------------------------

class TestOhlcvFeatures:

    def test_shape(self):
        from optimization.dl_optimizer import _ohlcv_features
        from optimization.liquidity_features import N_FEATURES as N_LIQ
        from optimization.volume_profile import N_FEATURES as N_VP
        from optimization.spike_features import N_FEATURES as N_SPK
        df  = _make_ohlcv_df(300)
        out = _ohlcv_features(df, seq_len=60)
        exp = 9 + N_LIQ + N_VP + N_SPK   # base + liquidity + volume-profile + spike
        assert out.shape == (300, 60, exp), f"Expected (300,60,{exp}), got {out.shape}"

    def test_no_nan(self):
        from optimization.dl_optimizer import _ohlcv_features
        df  = _make_ohlcv_df(300)
        out = _ohlcv_features(df, seq_len=60)
        assert np.isfinite(out).all(), "OHLCV features contain NaN/Inf"

    def test_clipped(self):
        from optimization.dl_optimizer import _ohlcv_features
        df  = _make_ohlcv_df(300)
        out = _ohlcv_features(df, seq_len=60)
        assert out.max() <= 5.0 and out.min() >= -5.0, "Features exceed clip bounds"

    def test_short_series(self):
        """Works correctly when T < seq_len (pads left with zeros)."""
        from optimization.dl_optimizer import _ohlcv_features
        from optimization.liquidity_features import N_FEATURES as N_LIQ
        from optimization.volume_profile import N_FEATURES as N_VP
        from optimization.spike_features import N_FEATURES as N_SPK
        df  = _make_ohlcv_df(30)
        out = _ohlcv_features(df, seq_len=60)
        assert out.shape == (30, 60, 9 + N_LIQ + N_VP + N_SPK)
        # First row: only bar 0 exists → other 59 slots should be zero
        assert (out[0, :59, :] == 0.0).all()


# ---------------------------------------------------------------------------
# Test: forward targets
# ---------------------------------------------------------------------------

class TestForwardTargets:

    def test_tail_nan(self):
        from optimization.dl_optimizer import _forward_targets_np
        close = np.linspace(100, 200, 100).astype(np.float32)
        atr   = np.ones(100, dtype=np.float32) * 1.5
        tgt   = _forward_targets_np(close, atr, [5, 10])
        # Targets average over AVAILABLE horizons, so a bar is NaN only when
        # every horizon runs off the end -> the last min(forward_days)=5 bars.
        assert np.isnan(tgt[-5:]).all()
        # Bars in [-10:-5] still have the 5-day horizon available -> finite.
        assert np.isfinite(tgt[-10:-5]).all()
        # Early bars should be finite
        assert np.isfinite(tgt[:80]).all()

    def test_clipped(self):
        from optimization.dl_optimizer import _forward_targets_np
        close = np.array([100.0, 1000.0] + [100.0] * 98, dtype=np.float32)  # huge jump
        atr   = np.ones(100, dtype=np.float32) * 0.01
        tgt   = _forward_targets_np(close, atr, [5])
        assert np.nanmax(np.abs(tgt)) <= 10.0


# ---------------------------------------------------------------------------
# Test: Differentiable indicators
# ---------------------------------------------------------------------------

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestDifferentiableIndicators:

    def _tensor(self, arr):
        import torch
        return torch.tensor(arr, dtype=torch.float32)

    def test_ema_shape(self):
        from optimization.differentiable_indicators import ema
        import torch
        x = self._tensor(np.linspace(100, 200, 100).reshape(1, -1))
        period = torch.tensor([14.0])
        out = ema(x, period)
        assert out.shape == (1, 100)

    def test_rsi_range(self):
        """RSI should be in [0, 100]."""
        from optimization.differentiable_indicators import rsi
        import torch
        rng  = np.random.default_rng(0)
        vals = 100.0 + np.cumsum(rng.normal(0, 1, 100)).astype(np.float32)
        x    = self._tensor(vals.reshape(1, -1))
        period = torch.tensor([14.0])
        out  = rsi(x, period)
        assert float(out.min()) >= -1.0   # allow tiny numerical error
        assert float(out.max()) <= 101.0

    def test_rsi_gradient_flows(self):
        """Gradients should flow back through RSI w.r.t. period."""
        from optimization.differentiable_indicators import rsi
        import torch
        rng  = np.random.default_rng(0)
        vals = 100.0 + np.cumsum(rng.normal(0, 1, 100)).astype(np.float32)
        x    = self._tensor(vals.reshape(1, -1))
        period = torch.tensor([14.0], requires_grad=True)
        out  = rsi(x, period)
        out.sum().backward()
        assert period.grad is not None
        assert period.grad.isfinite().all()

    def test_bb_position_range(self):
        """BB %B should be in [0, 1]."""
        from optimization.differentiable_indicators import bb_position
        import torch
        rng  = np.random.default_rng(1)
        vals = 100.0 + np.cumsum(rng.normal(0, 0.5, 100)).astype(np.float32)
        x    = self._tensor(vals.reshape(1, -1))
        period = torch.tensor([20.0])
        std    = torch.tensor([2.0])
        out    = bb_position(x, period, std)
        assert float(out.min()) >= -0.01
        assert float(out.max()) <= 1.01


# ---------------------------------------------------------------------------
# Test: Model shape (forward pass)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestModelForwardPass:

    def test_output_shapes(self):
        """Model forward pass should produce correctly shaped outputs."""
        from optimization.dl_optimizer import (
            _IndicatorParamPredictorImpl, N_PARAMS
        )
        import torch
        model = _IndicatorParamPredictorImpl(
            input_size=9, hidden_size=32, num_layers=1, n_heads=2
        )
        model.eval()
        batch_sz, seq_len = 4, 60
        x = torch.randn(batch_sz, seq_len, 9)
        with torch.no_grad():
            params, weights, threshold = model(x)
        assert params.shape    == (batch_sz, N_PARAMS),  f"params shape: {params.shape}"
        assert weights.shape   == (batch_sz, 6),          f"weights shape: {weights.shape}"
        assert threshold.shape == (batch_sz, 1),          f"threshold shape: {threshold.shape}"

    def test_params_in_unit_range(self):
        """Raw param outputs should be in [0,1] (sigmoid-activated)."""
        from optimization.dl_optimizer import _IndicatorParamPredictorImpl
        import torch
        model = _IndicatorParamPredictorImpl(input_size=9, hidden_size=32, num_layers=1, n_heads=2)
        model.eval()
        x = torch.randn(8, 60, 9)
        with torch.no_grad():
            params, _, _ = model(x)
        assert float(params.min()) >= 0.0
        assert float(params.max()) <= 1.0

    def test_weights_positive(self):
        """Category weights should be positive (softplus-activated)."""
        from optimization.dl_optimizer import _IndicatorParamPredictorImpl
        import torch
        model = _IndicatorParamPredictorImpl(input_size=9, hidden_size=32, num_layers=1, n_heads=2)
        model.eval()
        x = torch.randn(8, 60, 9)
        with torch.no_grad():
            _, weights, _ = model(x)
        assert float(weights.min()) > 0.0

    def test_no_nan_output(self):
        """Forward pass should never produce NaN."""
        from optimization.dl_optimizer import _IndicatorParamPredictorImpl
        import torch
        model = _IndicatorParamPredictorImpl(input_size=9, hidden_size=64, num_layers=2, n_heads=4)
        x = torch.randn(16, 60, 9)
        params, weights, threshold = model(x)
        assert params.isfinite().all()
        assert weights.isfinite().all()
        assert threshold.isfinite().all()

    def test_param_scaling(self):
        """Scaled params should be within declared PARAM_BOUNDS."""
        from optimization.dl_optimizer import (
            _IndicatorParamPredictorImpl, _scale_params, PARAM_BOUNDS
        )
        import torch
        model = _IndicatorParamPredictorImpl(input_size=9, hidden_size=32, num_layers=1, n_heads=2)
        model.eval()
        x = torch.randn(32, 60, 9)
        with torch.no_grad():
            raw_p, _, _ = model(x)
        scaled = _scale_params(raw_p)
        for name, tensor in scaled.items():
            lo, hi = PARAM_BOUNDS[name]
            assert float(tensor.min()) >= lo - 1e-4, f"{name} below lower bound"
            assert float(tensor.max()) <= hi + 1e-4, f"{name} above upper bound"


# ---------------------------------------------------------------------------
# Test: Regime LSTM
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestRegimeLSTM:

    def test_regime_probs_sum_to_one(self):
        """Regime probabilities should sum to 1 (softmax)."""
        from optimization.regime_lstm import _RegimeAwareLSTMImpl
        import torch
        model = _RegimeAwareLSTMImpl(input_size=9, hidden_size=32, num_layers=1)
        model.eval()
        x = torch.randn(8, 60, 9)
        with torch.no_grad():
            _, _, _, regime_probs = model(x)
        sums = regime_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_output_shapes(self):
        from optimization.regime_lstm import _RegimeAwareLSTMImpl
        from optimization.dl_optimizer import N_PARAMS
        import torch
        model = _RegimeAwareLSTMImpl(input_size=9, hidden_size=32, num_layers=1)
        model.eval()
        x = torch.randn(4, 60, 9)
        with torch.no_grad():
            params, weights, threshold, regime_probs = model(x)
        assert params.shape       == (4, N_PARAMS)
        assert weights.shape      == (4, 6)
        assert threshold.shape    == (4, 1)
        assert regime_probs.shape == (4, 3)


# ---------------------------------------------------------------------------
# Test: DLModel dataclass
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestDLModel:

    def _make_model(self):
        from optimization.dl_optimizer import _IndicatorParamPredictorImpl, DLModel, N_PARAMS
        import torch
        m = _IndicatorParamPredictorImpl(input_size=9, hidden_size=32, num_layers=1, n_heads=2)
        return DLModel(
            model_state          = {k: v.cpu() for k, v in m.state_dict().items()},
            input_size           = 9,
            hidden_size          = 32,
            num_layers           = 1,
            bidirectional        = False,
            seq_len              = 60,
            discovered_params    = {"rsi_period": 12, "bb_period": 20, "bb_std": 2.0,
                                    "atr_period": 14, "ma_fast": 20, "ma_slow": 50,
                                    "macd_fast": 12, "macd_slow": 26, "supertrend_mult": 3.0},
            discovered_weights   = {"trend": 1.2, "price_action": 1.1, "momentum": 1.0,
                                    "volatility": 0.8, "volume": 0.9, "other": 0.7},
            discovered_threshold = 0.15,
            train_history        = [0.5, 0.4, 0.3],
        )

    def test_to_config_keys(self):
        m = self._make_model()
        cfg = m.to_config()
        assert "method" in cfg
        assert cfg["method"] == "dl"
        assert "indicator_params" in cfg
        assert "weights" in cfg

    def test_predictions_frame_shape(self):
        m = self._make_model()
        ind = _make_indicators_df(100)
        preds = m.predictions_frame(ind)
        assert len(preds) == 100
        assert "signal" in preds.columns
        assert "confidence" in preds.columns

    def test_save_load_checkpoint(self, tmp_path):
        m = self._make_model()
        path = str(tmp_path / "test_checkpoint.pt")
        m.save_checkpoint(path)
        import os
        assert os.path.exists(path)
        loaded = type(m).load_checkpoint(path)
        assert loaded.discovered_params == m.discovered_params
        assert loaded.hidden_size == m.hidden_size

    def test_deterministic_predictions(self):
        """Two calls with same model should give identical predictions."""
        m = self._make_model()
        ind = _make_indicators_df(50)
        p1 = m.predictions_frame(ind)
        p2 = m.predictions_frame(ind)
        pd.testing.assert_frame_equal(p1, p2)


# ---------------------------------------------------------------------------
# Test: Quick training smoke test (CPU, tiny model, few epochs)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestTrainingSmoke:
    """Smoke tests: can the model train without crashing for a few epochs?"""

    def test_single_symbol_train(self):
        """Single-symbol DL training on CPU (tiny model for speed)."""
        from optimization.dl_optimizer import train as dl_train
        from optimization import data_loader

        n = 300
        df   = _make_ohlcv_df(n)
        close = pd.Series(df["close"].values, index=df.index, name="close")
        atr_  = pd.Series(np.ones(n) * 1.5, index=df.index, name="atr")

        # Build a minimal indicators_df (no TA-Lib needed)
        ind_df = _make_indicators_df(n)

        train_sl  = slice(0, 200)
        heldout_sl = slice(210, 300)

        model = dl_train(
            indicators_df = ind_df,
            price         = close,
            atr           = atr_,
            forward_days  = [5],
            train_slice   = train_sl,
            heldout_slice = heldout_sl,
            hidden_size   = 32,
            num_layers    = 1,
            epochs        = 5,
            lr            = 1e-3,
            device        = "cpu",
            verbose       = False,
        )

        assert model is not None
        assert isinstance(model.discovered_params, dict)
        assert len(model.train_history) == 5
        assert all(np.isfinite(h) for h in model.train_history), "NaN in training history"

    def test_regime_lstm_train(self):
        """Regime-LSTM training smoke test on CPU."""
        from optimization.regime_lstm import train as regime_train

        n = 300
        df    = _make_ohlcv_df(n)
        close = pd.Series(df["close"].values, index=df.index)
        atr_  = pd.Series(np.ones(n) * 1.5, index=df.index)
        ind_df = _make_indicators_df(n)

        model = regime_train(
            indicators_df = ind_df,
            price         = close,
            atr           = atr_,
            forward_days  = [5],
            train_slice   = slice(0, 200),
            heldout_slice = slice(210, 300),
            hidden_size   = 32,
            num_layers    = 1,
            epochs        = 5,
            device        = "cpu",
            verbose       = False,
        )
        assert model is not None
        assert isinstance(model.discovered_params, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
