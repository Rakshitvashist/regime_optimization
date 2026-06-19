import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import glob
import os
import json
import importlib.util
import numpy as np
import talib

# ── Dynamic import for files with spaces in their names ─────────────────────
def _load_module(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_cp  = _load_module("consensus_predictor",   "Consensus_predictor .py")
_hb  = _load_module("hit_ratio_backtester",  "Hit_ratio_backtester .py")

ConsensusPredictor   = _cp.ConsensusPredictor
PriceTargetPredictor = _cp.PriceTargetPredictor
HitRatioBacktester   = _hb.HitRatioBacktester

# ── Tuned-config precedence (feature 001-indicator-optimization, FR-008) ──────
# When the optimizer package is present, each symbol's prediction uses its tuned
# configuration (per-symbol override -> global -> default) and records the level.
try:
    from optimization import store as _opt_store
except Exception:        # optimizer package optional; fall back to defaults
    _opt_store = None


def _predictor_for(symbol, default_predictor):
    """Build a per-symbol predictor from the resolved tuned weights (FR-008)."""
    if _opt_store is None:
        return default_predictor
    cfg, level = _opt_store.resolve_config(symbol)
    return ConsensusPredictor(weight_config=cfg.weights.to_dict(), config_level=level)

# ── Constants ────────────────────────────────────────────────────────────────
OHLCV = {'open', 'high', 'low', 'close', 'volume'}


def _indicator_cols(df):
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    return [c for c in df.columns if c.lower() not in OHLCV and c in numeric_cols]


def _safe(val, default=0.0):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ── Main ─────────────────────────────────────────────────────────────────────
def generate_summary():
    data_dir    = "processed_indicators"
    output_file = os.path.join("dashboard", "public", "summary.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    files = glob.glob(os.path.join(data_dir, "*_with_indicators.csv"))
    print(f"Aggregating data from {len(files)} files...")

    predictor        = ConsensusPredictor(voting_threshold=0.15)
    target_predictor = PriceTargetPredictor()
    backtester       = HitRatioBacktester(forward_days=[5, 10])

    summary = []

    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            if df.empty or len(df) < 50:
                continue

            symbol = os.path.basename(file_path).split('_')[0]

            # ── Index ────────────────────────────────────────────────────────
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            df = df[~df.index.duplicated(keep='last')]   # remove duplicate dates
            df.columns = [c.lower() for c in df.columns]
            df = df.fillna(0)

            # ── Separate OHLCV / indicators ──────────────────────────────────
            ind_cols      = _indicator_cols(df)
            indicators_df = df[ind_cols].astype(float)
            quality_df    = pd.DataFrame(
                np.full(indicators_df.shape, 0.5),
                index=indicators_df.index,
                columns=indicators_df.columns
            )

            C = df['close'].values.astype(float)
            H = df['high'].values.astype(float)
            L = df['low'].values.astype(float)
            V = df['volume'].values.astype(float)
            price_series = df['close'].astype(float)

            # ── Raw TA-Lib indicators ────────────────────────────────────────
            raw_rsi      = talib.RSI(C, timeperiod=14)[-1]
            _, _, macdhist = talib.MACD(C, 12, 26, 9)
            raw_macd     = macdhist[-1]
            
            adx_series = pd.Series(talib.ADX(H, L, C, timeperiod=14), index=df.index)
            raw_adx    = adx_series.iloc[-1]
            
            bb_up, bb_mid, bb_lo = talib.BBANDS(C, timeperiod=20)
            raw_bb_pos   = (C[-1] - bb_lo[-1]) / (bb_up[-1] - bb_lo[-1] + 1e-10)
            raw_atr      = talib.ATR(H, L, C, timeperiod=14)[-1]

            atr_series = pd.Series(talib.ATR(H, L, C, timeperiod=14), index=df.index)

            # ── Adaptive Quality Weights ─────────────────────────────────────
            # Calculate how each indicator performed over the last 100 days
            # and adjust weights dynamically
            lookback = 100
            if len(df) > lookback:
                # 5-day future returns for performance tracking
                future_rets = price_series.pct_change(5).shift(-5)
                
                perf_df = predictor.calculate_indicator_performance(
                    indicators_df.iloc[-lookback-10:-10], 
                    future_rets.iloc[-lookback-10:-10]
                )
                quality_df = predictor.get_adaptive_weights(perf_df, quality_df)

            # ── Consensus predictor (tuned config via precedence, FR-008) ─────
            sym_predictor = _predictor_for(symbol, predictor)
            predictions = sym_predictor.predict(
                indicators_df, quality_df, price_series, atr_series, adx_series
            )

            # ── Hit-ratio backtester ─────────────────────────────────────────
            hit_stats = backtester.backtest(predictions, price_series)
            
            # Target Hit Analysis (New)
            target_stats = backtester.backtest_targets(
                predictions, 
                df['high'].astype(float), 
                df['low'].astype(float),
                look_ahead=10
            )

            # ── Latest values ────────────────────────────────────────────────
            latest_pred     = predictions.iloc[-1]
            latest_price    = _safe(df['close'].iloc[-1])
            latest_atr      = _safe(raw_atr, latest_price * 0.015)
            latest_consensus = _safe(latest_pred['consensus_score'])

            vol = price_series.pct_change().rolling(20).std().iloc[-1]
            vol = _safe(vol, 0.01)

            trend_strength = float(np.clip(_safe(raw_adx) / 50, 0, 1))

            # ── Price targets ────────────────────────────────────────────────
            targets = target_predictor.predict_targets(
                current_price   = latest_price,
                consensus_score = latest_consensus,
                atr             = latest_atr,
                volatility      = vol,
                trend_strength  = trend_strength
            )

            # ── Consensus breakdown ──────────────────────────────────────────
            bullish_pct = _safe(latest_pred.get('bullish_pct', 50))
            bearish_pct = _safe(latest_pred.get('bearish_pct', 50))
            confidence  = _safe(latest_pred.get('confidence', 0))
            signal_raw  = int(latest_pred.get('signal', 0))
            exp_move    = _safe(latest_pred.get('expected_move_pct', 0))

            # ── Chart data ───────────────────────────────────────────────────
            chart_data = []
            for _, row in df.tail(100).iterrows():
                chart_data.append({
                    "time":   str(row.name.date()) if hasattr(row.name, 'date') else str(row.name),
                    "open":   _safe(row['open']),
                    "high":   _safe(row['high']),
                    "low":    _safe(row['low']),
                    "close":  _safe(row['close']),
                    "volume": _safe(row['volume'])
                })

            # ── Consensus from indicator columns (for sidebar badge) ──────────
            consensus_badge = float(latest_consensus)

            # ── Breakdown for UI Category Cards & Modal ──────────────────────
            grouped_cols = predictor._group_indicators(indicators_df.columns)
            indicator_breakdown = {}
            def _sig(v):
                if v > 0: return 1
                if v < 0: return -1
                return 0
            for cat, cols in grouped_cols.items():
                if not cols: continue
                cat_data = indicators_df[cols].iloc[-1]
                signals = {col: _sig(val) for col, val in cat_data.items() if not np.isnan(val)}
                buy     = sum(1 for v in signals.values() if v == 1)
                sell    = sum(1 for v in signals.values() if v == -1)
                neutral = sum(1 for v in signals.values() if v == 0)
                indicator_breakdown[cat] = {
                    "buy": buy,
                    "sell": sell,
                    "neutral": neutral,
                    "signals": signals
                }

            # ── Append record ─────────────────────────────────────────────────
            summary.append({
                "symbol":   symbol,
                "company":  str(df.get('company', pd.Series([symbol])).iloc[-1] if 'company' in df else symbol),
                "industry": str(df.get('industry', pd.Series(['Unknown'])).iloc[-1] if 'industry' in df else 'Unknown'),
                "price":    latest_price,
                "change":   _safe(df['close'].iloc[-1] - df['close'].iloc[-2]),
                "consensus": consensus_badge,

                # ── 8 technical indicators ───────────────────────────────────
                "indicators": {
                    "rsi":          _safe(raw_rsi, 50),
                    "macd":         _safe(raw_macd),
                    "adx":          _safe(raw_adx),
                    "bb_pos":       _safe(raw_bb_pos, 0.5),
                    "atr":          _safe(raw_atr),
                    "supertrend":   _safe(df.get('supertrend_10_3', pd.Series([0])).iloc[-1] if 'supertrend_10_3' in df else 0),
                    "volume_surge": _safe(df.get('volume_surge_20', pd.Series([1])).iloc[-1] if 'volume_surge_20' in df else 1),
                    "zscore":       _safe(df.get('zscore_20', pd.Series([0])).iloc[-1] if 'zscore_20' in df else 0),
                },
                "indicator_breakdown": indicator_breakdown,

                # ── Consensus prediction ─────────────────────────────────────
                "prediction": {
                    "signal":           signal_raw,
                    "direction":        targets.get('direction', 'LONG'),
                    "confidence":       round(confidence, 4),
                    "consensus_score":  round(latest_consensus, 4),
                    "bullish_pct":      round(bullish_pct, 2),
                    "bearish_pct":      round(bearish_pct, 2),
                    "target_1":         round(_safe(targets.get('target_1', latest_price)), 2),
                    "target_2":         round(_safe(targets.get('target_2', latest_price)), 2),
                    "target_3":         round(_safe(targets.get('target_3', latest_price)), 2),
                    "stop_loss":        round(_safe(targets.get('stop_loss', latest_price)), 2),
                    "prob_target_1":    round(_safe(targets.get('prob_target_1', 0.5)), 4),
                    "prob_target_2":    round(_safe(targets.get('prob_target_2', 0.4)), 4),
                    "prob_target_3":    round(_safe(targets.get('prob_target_3', 0.3)), 4),
                    "expected_move_pct": round(exp_move, 4),
                },

                # ── Backtester hit ratios ────────────────────────────────────
                "hit_ratio": {
                    "overall_5d":              _safe(hit_stats.get('overall_5d', 0.5), 0.5),
                    "overall_10d":             _safe(hit_stats.get('overall_10d', 0.5), 0.5),
                    "buy_hit_5d":              _safe(hit_stats.get('buy_hit_5d', 0.5), 0.5),
                    "sell_hit_5d":             _safe(hit_stats.get('sell_hit_5d', 0.5), 0.5),
                    "high_conf_5d":            _safe(hit_stats.get('high_conf_5d', 0.5), 0.5),
                    "high_conf_10d":           _safe(hit_stats.get('high_conf_10d', 0.5), 0.5),
                    "signal_count_5d":         int(_safe(hit_stats.get('signal_count_5d', 0), 0)),
                    "win_count_5d":            int(_safe(hit_stats.get('win_count_5d', 0), 0)),
                    "avg_directional_return":  _safe(hit_stats.get('avg_directional_return_pct', 0.0), 0.0),
                    
                    # Target Hits (New)
                    "t1_hit_rate":             _safe(target_stats.get('t1_hit_rate', 0), 0),
                    "t2_hit_rate":             _safe(target_stats.get('t2_hit_rate', 0), 0),
                    "t3_hit_rate":             _safe(target_stats.get('t3_hit_rate', 0), 0),
                    "sl_hit_rate":             _safe(target_stats.get('sl_hit_rate', 0), 0),
                    "total_signals":           int(_safe(target_stats.get('total_signals', 0), 0))
                },

                "history": chart_data
            })

            print(f"[SUCCESS] Processed {symbol}")

        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary generated: {output_file}  ({len(summary)} stocks)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="processed_indicators", help="Input directory")
    parser.add_argument("--output", default=os.path.join("dashboard", "public", "summary.json"), help="Output file")
    args = parser.parse_args()

    def generate_summary_with_args(data_dir, output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        files = glob.glob(os.path.join(data_dir, "*_with_indicators.csv"))
        print(f"Aggregating data from {len(files)} files...")

        predictor        = ConsensusPredictor(voting_threshold=0.15)
        target_predictor = PriceTargetPredictor()
        backtester       = HitRatioBacktester(forward_days=[5, 10])

        summary = []

        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                if df.empty or len(df) < 50:
                    continue

                symbol = os.path.basename(file_path).split('_')[0]

                # ── Index ────────────────────────────────────────────────────────
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                df = df[~df.index.duplicated(keep='last')]   # remove duplicate dates
                df.columns = [c.lower() for c in df.columns]
                df = df.fillna(0)

                # ── Separate OHLCV / indicators ──────────────────────────────────
                ind_cols      = _indicator_cols(df)
                indicators_df = df[ind_cols].astype(float)
                quality_df    = pd.DataFrame(
                    np.full(indicators_df.shape, 0.5),
                    index=indicators_df.index,
                    columns=indicators_df.columns
                )

                C = df['close'].values.astype(float)
                H = df['high'].values.astype(float)
                L = df['low'].values.astype(float)
                V = df['volume'].values.astype(float)
                price_series = df['close'].astype(float)

                # ── Raw TA-Lib indicators ────────────────────────────────────────
                raw_rsi      = talib.RSI(C, timeperiod=14)[-1]
                _, _, macdhist = talib.MACD(C, 12, 26, 9)
                raw_macd     = macdhist[-1]
                
                adx_series = pd.Series(talib.ADX(H, L, C, timeperiod=14), index=df.index)
                raw_adx    = adx_series.iloc[-1]
                
                bb_up, bb_mid, bb_lo = talib.BBANDS(C, timeperiod=20)
                raw_bb_pos   = (C[-1] - bb_lo[-1]) / (bb_up[-1] - bb_lo[-1] + 1e-10)
                raw_atr      = talib.ATR(H, L, C, timeperiod=14)[-1]

                atr_series = pd.Series(talib.ATR(H, L, C, timeperiod=14), index=df.index)

                # ── Adaptive Quality Weights ─────────────────────────────────────
                lookback = 100
                if len(df) > lookback:
                    future_rets = price_series.pct_change(5).shift(-5)
                    perf_df = predictor.calculate_indicator_performance(
                        indicators_df.iloc[-lookback-10:-10], 
                        future_rets.iloc[-lookback-10:-10]
                    )
                    quality_df = predictor.get_adaptive_weights(perf_df, quality_df)

                # ── Consensus predictor (tuned config via precedence, FR-008) ─────
                sym_predictor = _predictor_for(symbol, predictor)
                predictions = sym_predictor.predict(
                    indicators_df, quality_df, price_series, atr_series, adx_series
                )

                # ── Hit-ratio backtester ─────────────────────────────────────────
                hit_stats = backtester.backtest(predictions, price_series)
                target_stats = backtester.backtest_targets(
                    predictions, df['high'].astype(float), df['low'].astype(float), look_ahead=10
                )

                # ── Latest values ────────────────────────────────────────────────
                latest_pred     = predictions.iloc[-1]
                latest_price    = _safe(df['close'].iloc[-1])
                latest_atr      = _safe(raw_atr, latest_price * 0.015)
                latest_consensus = _safe(latest_pred['consensus_score'])

                vol = price_series.pct_change().rolling(20).std().iloc[-1]
                vol = _safe(vol, 0.01)
                trend_strength = float(np.clip(_safe(raw_adx) / 50, 0, 1))

                # ── Price targets ────────────────────────────────────────────────
                targets = target_predictor.predict_targets(
                    current_price=latest_price, consensus_score=latest_consensus, atr=latest_atr, volatility=vol, trend_strength=trend_strength
                )

                bullish_pct = _safe(latest_pred.get('bullish_pct', 50))
                bearish_pct = _safe(latest_pred.get('bearish_pct', 50))
                confidence  = _safe(latest_pred.get('confidence', 0))
                signal_raw  = int(latest_pred.get('signal', 0))
                exp_move    = _safe(latest_pred.get('expected_move_pct', 0))

                # ── Chart data ───────────────────────────────────────────────────
                chart_data = []
                for _, row in df.tail(100).iterrows():
                    chart_data.append({
                        "time":   str(row.name.date()) if hasattr(row.name, 'date') else str(row.name),
                        "open":   _safe(row['open']),
                        "high":   _safe(row['high']),
                        "low":    _safe(row['low']),
                        "close":  _safe(row['close']),
                        "volume": _safe(row['volume'])
                    })

                consensus_badge = float(latest_consensus)

                # ── Breakdown for UI Category Cards & Modal ──────────────────────
                grouped_cols = predictor._group_indicators(indicators_df.columns)
                indicator_breakdown = {}
                def _sig(v):
                    if v > 0: return 1
                    if v < 0: return -1
                    return 0
                for cat, cols in grouped_cols.items():
                    if not cols: continue
                    cat_data = indicators_df[cols].iloc[-1]
                    signals = {col: _sig(val) for col, val in cat_data.items() if not np.isnan(val)}
                    buy     = sum(1 for v in signals.values() if v == 1)
                    sell    = sum(1 for v in signals.values() if v == -1)
                    neutral = sum(1 for v in signals.values() if v == 0)
                    indicator_breakdown[cat] = {
                        "buy": buy,
                        "sell": sell,
                        "neutral": neutral,
                        "signals": signals
                    }

                # ── Append record ─────────────────────────────────────────────────
                summary.append({
                    "symbol":   symbol,
                    "company":  str(df.get('company', pd.Series([symbol])).iloc[-1] if 'company' in df else symbol),
                    "industry": str(df.get('industry', pd.Series(['Unknown'])).iloc[-1] if 'industry' in df else 'Unknown'),
                    "price":    latest_price,
                    "change":   _safe(df['close'].iloc[-1] - df['close'].iloc[-2]),
                    "consensus": consensus_badge,
                    "indicators": {
                        "rsi": _safe(raw_rsi, 50), "macd": _safe(raw_macd), "adx": _safe(raw_adx), "bb_pos": _safe(raw_bb_pos, 0.5),
                        "atr": _safe(raw_atr), "supertrend": _safe(df.get('supertrend_10_3', pd.Series([0])).iloc[-1] if 'supertrend_10_3' in df else 0),
                        "volume_surge": _safe(df.get('volume_surge_20', pd.Series([1])).iloc[-1] if 'volume_surge_20' in df else 1),
                        "zscore": _safe(df.get('zscore_20', pd.Series([0])).iloc[-1] if 'zscore_20' in df else 0),
                    },
                    "indicator_breakdown": indicator_breakdown,
                    "prediction": {
                        "signal": signal_raw, "direction": targets.get('direction', 'LONG'), "confidence": round(confidence, 4),
                        "consensus_score": round(latest_consensus, 4), "bullish_pct": round(bullish_pct, 2), "bearish_pct": round(bearish_pct, 2),
                        "target_1": round(_safe(targets.get('target_1', latest_price)), 2), "target_2": round(_safe(targets.get('target_2', latest_price)), 2),
                        "target_3": round(_safe(targets.get('target_3', latest_price)), 2), "stop_loss": round(_safe(targets.get('stop_loss', latest_price)), 2),
                        "prob_target_1": round(_safe(targets.get('prob_target_1', 0.5)), 4), "prob_target_2": round(_safe(targets.get('prob_target_2', 0.4)), 4),
                        "prob_target_3": round(_safe(targets.get('prob_target_3', 0.3)), 4), "expected_move_pct": round(exp_move, 4),
                    },
                    "hit_ratio": {
                        "overall_5d": _safe(hit_stats.get('overall_5d', 0.5), 0.5), "overall_10d": _safe(hit_stats.get('overall_10d', 0.5), 0.5),
                        "buy_hit_5d": _safe(hit_stats.get('buy_hit_5d', 0.5), 0.5), "sell_hit_5d": _safe(hit_stats.get('sell_hit_5d', 0.5), 0.5),
                        "high_conf_5d": _safe(hit_stats.get('high_conf_5d', 0.5), 0.5), "high_conf_10d": _safe(hit_stats.get('high_conf_10d', 0.5), 0.5),
                        "signal_count_5d": int(_safe(hit_stats.get('signal_count_5d', 0), 0)), "win_count_5d": int(_safe(hit_stats.get('win_count_5d', 0), 0)),
                        "avg_directional_return": _safe(hit_stats.get('avg_directional_return_pct', 0.0), 0.0),
                        "t1_hit_rate": _safe(target_stats.get('t1_hit_rate', 0), 0), "t2_hit_rate": _safe(target_stats.get('t2_hit_rate', 0), 0),
                        "t3_hit_rate": _safe(target_stats.get('t3_hit_rate', 0), 0), "sl_hit_rate": _safe(target_stats.get('sl_hit_rate', 0), 0),
                        "total_signals": int(_safe(target_stats.get('total_signals', 0), 0))
                    },
                    "history": chart_data
                })
                print(f"[SUCCESS] Processed {symbol}")
            except Exception as e:
                print(f"[ERROR] {file_path}: {e}")

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary generated: {output_file}  ({len(summary)} stocks)")

    generate_summary_with_args(args.input, args.output)

