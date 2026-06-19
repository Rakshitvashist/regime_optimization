"""
generate_summary_500.py
Reads raw OHLCV CSVs from nifty500_host/, computes technical indicators,
consensus predictions, price targets, and hit-ratio backtesting, then
writes dashboard/public/summary_500.json with all 500 Nifty 500 stocks.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import glob
import os
import json
import importlib.util
import numpy as np
import talib

# ── Dynamic import for files with spaces ────────────────────────────────────
def _load_module(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_cp = _load_module("consensus_predictor",  "Consensus_predictor .py")
_hb = _load_module("hit_ratio_backtester", "Hit_ratio_backtester .py")

ConsensusPredictor   = _cp.ConsensusPredictor
PriceTargetPredictor = _cp.PriceTargetPredictor
HitRatioBacktester   = _hb.HitRatioBacktester

# ── Tuned-config precedence (feature 001-indicator-optimization, FR-008) ──────
try:
    from optimization import store as _opt_store
except Exception:        # optimizer package optional; fall back to defaults
    _opt_store = None


def _predictor_for(symbol, default_predictor):
    """Build a per-symbol predictor from the resolved tuned weights (FR-008).

    The Nifty-500 path constructs its own (synthetic) indicator signals but the
    predict() signature is identical, so only the weight config + level differ.
    """
    if _opt_store is None:
        return default_predictor
    cfg, level = _opt_store.resolve_config(symbol)
    return ConsensusPredictor(weight_config=cfg.weights.to_dict(), config_level=level)


# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR    = "nifty500_host"
OUTPUT_FILE = os.path.join("dashboard", "public", "summary_500.json")
MIN_ROWS    = 60  # minimum history rows required

def _safe(val, default=0.0):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def compute_indicators(df):
    """Compute raw TA-Lib indicators and synthetic indicator signals for consensus."""
    C = df['close'].values.astype(float)
    H = df['high'].values.astype(float)
    L = df['low'].values.astype(float)
    V = df['volume'].values.astype(float)

    n = len(C)
    idx = df.index

    # ── Raw indicators ────────────────────────────────────────────────────────
    rsi      = talib.RSI(C, timeperiod=14)
    _, _, macdhist = talib.MACD(C, 12, 26, 9)
    adx      = talib.ADX(H, L, C, timeperiod=14)
    bb_up, bb_mid, bb_lo = talib.BBANDS(C, timeperiod=20)
    atr      = talib.ATR(H, L, C, timeperiod=14)
    ema20    = talib.EMA(C, timeperiod=20)
    ema50    = talib.EMA(C, timeperiod=50)
    ema200   = talib.EMA(C, timeperiod=200)
    sma20    = talib.SMA(C, timeperiod=20)
    stoch_k, stoch_d = talib.STOCH(H, L, C)
    cci      = talib.CCI(H, L, C, timeperiod=14)
    willr    = talib.WILLR(H, L, C, timeperiod=14)
    obv      = talib.OBV(C, V)
    mfi      = talib.MFI(H, L, C, V, timeperiod=14)
    roc      = talib.ROC(C, timeperiod=10)

    # Volume surge (current vs 20-day average)
    vol_ma   = pd.Series(V, index=idx).rolling(20).mean().values
    vol_surge = np.where(vol_ma > 0, V / (vol_ma + 1e-10), 1.0)

    # Z-score of price vs 20-day mean
    price_s   = pd.Series(C, index=idx)
    roll_mean = price_s.rolling(20).mean()
    roll_std  = price_s.rolling(20).std()
    zscore    = ((price_s - roll_mean) / (roll_std + 1e-10)).values

    # SuperTrend (simplified ATR-based signal)
    atr_s       = pd.Series(atr, index=idx).fillna(0)
    mid         = (pd.Series(H, index=idx) + pd.Series(L, index=idx)) / 2
    upper_band  = mid + 3 * atr_s
    lower_band  = mid - 3 * atr_s
    close_s     = pd.Series(C, index=idx)
    supertrend  = np.where(close_s >= lower_band, 1.0, -1.0)

    # ── Synthetic binary indicator signals (for ConsensusPredictor) ───────────
    def _nb(cond): return np.where(cond, 1.0, -1.0)

    indicators = pd.DataFrame(index=idx)
    # Oscillators (extreme zones) -> map True to 1 or -1, False to 0
    indicators['rsi_oversold']       = (rsi < 30).astype(float)
    indicators['rsi_overbought']     = (rsi > 70).astype(float) * -1
    indicators['rsi_mid_bull']       = ((rsi > 50) & (rsi <= 70)).astype(float)
    indicators['bb_near_lower']      = ((C - bb_lo) / (bb_up - bb_lo + 1e-10) < 0.2).astype(float)
    indicators['bb_near_upper']      = ((C - bb_lo) / (bb_up - bb_lo + 1e-10) > 0.8).astype(float) * -1
    indicators['stoch_oversold']     = (stoch_k < 20).astype(float)
    indicators['stoch_overbought']   = (stoch_k > 80).astype(float) * -1
    indicators['cci_oversold']       = (cci < -100).astype(float)
    indicators['cci_overbought']     = (cci > 100).astype(float) * -1
    indicators['willr_oversold']     = (willr < -80).astype(float)
    indicators['willr_overbought']   = (willr > -20).astype(float) * -1
    indicators['mfi_oversold']       = (mfi < 20).astype(float)
    indicators['mfi_overbought']     = (mfi > 80).astype(float) * -1
    
    # Trend/Directional (always -1 or 1 to balance consensus)
    indicators['macd_positive']      = _nb(macdhist > 0)
    indicators['ema20_above_ema50']  = _nb(ema20 > ema50)
    indicators['ema50_above_ema200'] = _nb(ema50 > ema200)
    indicators['price_above_ema20']  = _nb(C > ema20)
    indicators['price_above_sma20']  = _nb(C > sma20)
    indicators['roc_positive']       = _nb(roc > 0)
    indicators['supertrend_signal']  = _nb(supertrend > 0)
    indicators['obv_rising']         = _nb(pd.Series(obv, index=idx).diff() > 0)
    
    # Momentum confirmations
    indicators['adx_strong']         = (adx > 25).astype(float)
    indicators['vol_surge_signal']   = (vol_surge > 1.5).astype(float)
    indicators['macd_cross_above']   = ((macdhist > 0) & (np.roll(macdhist, 1) <= 0)).astype(float)
    indicators = indicators.fillna(0)

    raw = {
        'rsi':         _safe(rsi[-1], 50),
        'macd':        _safe(macdhist[-1]),
        'adx':         _safe(adx[-1]),
        'bb_pos':      _safe((C[-1] - bb_lo[-1]) / (bb_up[-1] - bb_lo[-1] + 1e-10), 0.5),
        'atr':         _safe(atr[-1]),
        'supertrend':  float(supertrend[-1]),
        'volume_surge':_safe(vol_surge[-1], 1.0),
        'zscore':      _safe(zscore[-1]),
    }
    atr_series = pd.Series(atr, index=idx)
    adx_series = pd.Series(adx, index=idx)
    return indicators, raw, atr_series, adx_series


def process_stock(file_path, predictor, target_predictor, backtester):
    """Process one stock CSV and return its summary record."""
    df = pd.read_csv(file_path)
    if df.empty or len(df) < MIN_ROWS:
        return None

    # ── Normalise columns ─────────────────────────────────────────────────────
    df.columns = [c.strip() for c in df.columns]
    col_map = {c: c.lower().replace(' ', '_') for c in df.columns}
    df = df.rename(columns=col_map)

    # Symbol / meta
    symbol   = str(df['symbol'].iloc[-1])   if 'symbol'   in df.columns else os.path.basename(file_path).split('_')[0]
    company  = str(df['company'].iloc[-1])  if 'company'  in df.columns else symbol
    industry = str(df['industry'].iloc[-1]) if 'industry' in df.columns else 'Unknown'

    # Date index
    date_col = 'date' if 'date' in df.columns else df.columns[df.columns.str.lower() == 'date'][0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()
    df = df[~df.index.duplicated(keep='last')]

    # Keep only numeric OHLCV
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df['volume'] = df['volume'].fillna(0)

    if len(df) < MIN_ROWS:
        return None

    # ── Compute indicators ────────────────────────────────────────────────────
    indicators_df, raw, atr_series, adx_series = compute_indicators(df)
    price_series = df['close'].astype(float)

    quality_df = pd.DataFrame(
        np.full(indicators_df.shape, 0.5),
        index=indicators_df.index,
        columns=indicators_df.columns
    )

    # Adaptive quality weights (last 100-day lookback)
    lookback = 100
    if len(df) > lookback + 10:
        future_rets = price_series.pct_change(5).shift(-5)
        try:
            perf_df    = predictor.calculate_indicator_performance(
                indicators_df.iloc[-lookback-10:-10],
                future_rets.iloc[-lookback-10:-10]
            )
            quality_df = predictor.get_adaptive_weights(perf_df, quality_df)
        except Exception:
            pass

    # ── Predictions (tuned config via precedence, FR-008) ─────────────────────
    sym_predictor = _predictor_for(symbol, predictor)
    predictions = sym_predictor.predict(indicators_df, quality_df, price_series, atr_series, adx_series)

    hit_stats    = backtester.backtest(predictions, price_series)
    target_stats = backtester.backtest_targets(
        predictions, df['high'].astype(float), df['low'].astype(float), look_ahead=10
    )

    latest_pred      = predictions.iloc[-1]
    latest_price     = _safe(df['close'].iloc[-1])
    latest_atr       = _safe(raw['atr'], latest_price * 0.015)
    latest_consensus = _safe(latest_pred['consensus_score'])

    vol            = price_series.pct_change().rolling(20).std().iloc[-1]
    vol            = _safe(vol, 0.01)
    trend_strength = float(np.clip(_safe(raw['adx']) / 50, 0, 1))

    targets = target_predictor.predict_targets(
        current_price   = latest_price,
        consensus_score = latest_consensus,
        atr             = latest_atr,
        volatility      = vol,
        trend_strength  = trend_strength
    )

    bullish_pct = _safe(latest_pred.get('bullish_pct', 50))
    bearish_pct = _safe(latest_pred.get('bearish_pct', 50))
    confidence  = _safe(latest_pred.get('confidence', 0))
    signal_raw  = int(latest_pred.get('signal', 0))
    exp_move    = _safe(latest_pred.get('expected_move_pct', 0))

    # Sidebar consensus badge (sync with consensus_score)
    consensus_badge = float(latest_consensus)

    # ── Chart history (last 100 days) ─────────────────────────────────────────
    chart_data = []
    for dt, row in df.tail(100).iterrows():
        o = _safe(row['open']); h = _safe(row['high'])
        l = _safe(row['low']);  c = _safe(row['close'])
        # Ensure OHLC integrity for lightweight-charts
        h = max(o, h, l, c)
        l = min(o, h, l, c)
        chart_data.append({
            "time":   str(dt.date()) if hasattr(dt, 'date') else str(dt),
            "open":   round(o, 4),
            "high":   round(h, 4),
            "low":    round(l, 4),
            "close":  round(c, 4),
            "volume": _safe(row['volume'])
        })

    return {
        "symbol":   symbol,
        "company":  company,
        "industry": industry,
        "price":    latest_price,
        "change":   _safe(df['close'].iloc[-1] - df['close'].iloc[-2]),
        "consensus": consensus_badge,
        "indicators": {
            "rsi":          raw['rsi'],
            "macd":         raw['macd'],
            "adx":          raw['adx'],
            "bb_pos":       raw['bb_pos'],
            "atr":          raw['atr'],
            "supertrend":   raw['supertrend'],
            "volume_surge": raw['volume_surge'],
            "zscore":       raw['zscore'],
        },
        "prediction": {
            "signal":            signal_raw,
            "direction":         targets.get('direction', 'LONG'),
            "confidence":        round(confidence, 4),
            "confidence_level":  "high" if confidence > 0.6 else "medium" if confidence > 0.3 else "low",
            "consensus_score":   round(latest_consensus, 4),
            "bullish_pct":       round(bullish_pct, 2),
            "bearish_pct":       round(bearish_pct, 2),
            "target_1":          round(_safe(targets.get('target_1', latest_price)), 2),
            "target_2":          round(_safe(targets.get('target_2', latest_price)), 2),
            "target_3":          round(_safe(targets.get('target_3', latest_price)), 2),
            "stop_loss":         round(_safe(targets.get('stop_loss', latest_price)), 2),
            "prob_target_1":     round(_safe(targets.get('prob_target_1', 0.5)), 4),
            "prob_target_2":     round(_safe(targets.get('prob_target_2', 0.4)), 4),
            "prob_target_3":     round(_safe(targets.get('prob_target_3', 0.3)), 4),
            "expected_move_pct": round(exp_move, 4),
        },
        "hit_ratio": {
            "overall_5d":             _safe(hit_stats.get('overall_5d', 0.5), 0.5),
            "overall_10d":            _safe(hit_stats.get('overall_10d', 0.5), 0.5),
            "buy_hit_5d":             _safe(hit_stats.get('buy_hit_5d', 0.5), 0.5),
            "sell_hit_5d":            _safe(hit_stats.get('sell_hit_5d', 0.5), 0.5),
            "high_conf_5d":           _safe(hit_stats.get('high_conf_5d', 0.5), 0.5),
            "high_conf_10d":          _safe(hit_stats.get('high_conf_10d', 0.5), 0.5),
            "signal_count_5d":        int(_safe(hit_stats.get('signal_count_5d', 0), 0)),
            "win_count_5d":           int(_safe(hit_stats.get('win_count_5d', 0), 0)),
            "avg_directional_return": _safe(hit_stats.get('avg_directional_return_pct', 0.0), 0.0),
            "t1_hit_rate":            _safe(target_stats.get('t1_hit_rate', 0), 0),
            "t2_hit_rate":            _safe(target_stats.get('t2_hit_rate', 0), 0),
            "t3_hit_rate":            _safe(target_stats.get('t3_hit_rate', 0), 0),
            "sl_hit_rate":            _safe(target_stats.get('sl_hit_rate', 0), 0),
            "total_signals":          int(_safe(target_stats.get('total_signals', 0), 0)),
        },
        "history": chart_data
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_1d_max.csv")))
    print(f"Found {len(files)} Nifty 500 stock files in '{DATA_DIR}'")

    predictor        = ConsensusPredictor(voting_threshold=0.15)
    target_predictor = PriceTargetPredictor()
    backtester       = HitRatioBacktester(forward_days=[5, 10])

    summary  = []
    errors   = []

    for i, file_path in enumerate(files, 1):
        sym = os.path.basename(file_path).split('_')[0]
        try:
            record = process_stock(file_path, predictor, target_predictor, backtester)
            if record:
                summary.append(record)
                print(f"[{i:3d}/{len(files)}] ✓ {sym}")
            else:
                print(f"[{i:3d}/{len(files)}] ⚠ {sym} — skipped (too few rows)")
        except Exception as e:
            errors.append((sym, str(e)))
            print(f"[{i:3d}/{len(files)}] ✗ {sym} — ERROR: {e}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary written → {OUTPUT_FILE}")
    print(f"  Stocks processed : {len(summary)}")
    print(f"  Errors           : {len(errors)}")
    if errors:
        print("  Failed symbols:")
        for sym, err in errors:
            print(f"    {sym}: {err}")


if __name__ == "__main__":
    main()
