"""
Hit Ratio Backtester
Tests how often consensus predictions correctly predict price direction.
Integrates with ConsensusPredictor output from Consensus_predictor.py
"""

import numpy as np
import pandas as pd


class HitRatioBacktester:
    """
    Backtests consensus prediction accuracy against historical prices.
    Measures hit ratios for N-day forward periods, broken down by
    confidence level, buy vs sell, and high-confidence only.
    """

    def __init__(self, forward_days=None):
        self.forward_days = forward_days or [5, 10]

    def backtest(self, predictions_df: pd.DataFrame, price_series: pd.Series) -> dict:
        """
        Backtest consensus signals against future price movements.

        Args:
            predictions_df: Output DataFrame from ConsensusPredictor.predict()
                            Must have columns: signal, confidence, consensus_score
            price_series:   Closing price Series (same index as predictions_df)

        Returns:
            dict with keys like:
                overall_5d, overall_10d   - directional accuracy for all signals
                buy_hit_5d, sell_hit_5d   - accuracy for buy/sell separately
                high_conf_5d              - accuracy for confidence > 0.4 signals
                signal_count_5d           - total signals evaluated
                win_count_5d              - number correct
                avg_directional_return_pct - avg % return in predicted direction
        """
        results = {}

        for days in self.forward_days:
            fwd_returns = price_series.pct_change(days).shift(-days)
            actionable = predictions_df[predictions_df['signal'] != 0]

            if len(actionable) == 0:
                results[f'overall_{days}d']      = 0.5
                results[f'buy_hit_{days}d']      = 0.5
                results[f'sell_hit_{days}d']     = 0.5
                results[f'signal_count_{days}d'] = 0
                results[f'win_count_{days}d']    = 0
                continue

            ret = fwd_returns.reindex(actionable.index).dropna()
            sig = actionable.loc[ret.index, 'signal']

            correct = ((sig == 1) & (ret > 0)) | ((sig == -1) & (ret < 0))

            results[f'overall_{days}d']      = round(float(correct.mean()), 4)
            results[f'signal_count_{days}d'] = int(len(correct))
            results[f'win_count_{days}d']    = int(correct.sum())

            # Buy-only
            buy_mask = sig == 1
            results[f'buy_hit_{days}d'] = round(float((ret[buy_mask] > 0).mean()), 4) \
                if buy_mask.any() else 0.5

            # Sell-only
            sell_mask = sig == -1
            results[f'sell_hit_{days}d'] = round(float((ret[sell_mask] < 0).mean()), 4) \
                if sell_mask.any() else 0.5

        # High-confidence signals (confidence > 0.4)
        hc = predictions_df[
            (predictions_df['signal'] != 0) &
            (predictions_df['confidence'] > 0.4)
        ]
        for days in self.forward_days:
            fwd = price_series.pct_change(days).shift(-days)
            if len(hc) > 0:
                hc_ret = fwd.reindex(hc.index).dropna()
                hc_sig = hc.loc[hc_ret.index, 'signal']
                hc_correct = ((hc_sig == 1) & (hc_ret > 0)) | ((hc_sig == -1) & (hc_ret < 0))
                results[f'high_conf_{days}d'] = round(float(hc_correct.mean()), 4) \
                    if len(hc_correct) > 0 else 0.5
            else:
                results[f'high_conf_{days}d'] = 0.5

        # Average directional return (positive = correct direction on average)
        fwd5 = price_series.pct_change(5).shift(-5)
        actionable = predictions_df[predictions_df['signal'] != 0]
        if len(actionable) > 0:
            r = fwd5.reindex(actionable.index).dropna() * 100
            s = actionable.loc[r.index, 'signal']
            results['avg_directional_return_pct'] = round(float((r * s).mean()), 4)
        else:
            results['avg_directional_return_pct'] = 0.0

        return results

    def get_signal_history(self, predictions_df: pd.DataFrame,
                           price_series: pd.Series,
                           days: int = 5) -> pd.DataFrame:
        """
        Row-by-row outcome log for each signal.

        Returns DataFrame with columns:
            signal, confidence, consensus_score, fwd_return, correct
        """
        fwd = price_series.pct_change(days).shift(-days)
        actionable = predictions_df[predictions_df['signal'] != 0].copy()
        actionable['fwd_return'] = fwd.reindex(actionable.index)
        actionable['correct'] = (
            ((actionable['signal'] == 1) & (actionable['fwd_return'] > 0)) |
            ((actionable['signal'] == -1) & (actionable['fwd_return'] < 0))
        )
        return actionable[
            ['signal', 'confidence', 'consensus_score', 'fwd_return', 'correct']
        ].dropna()


    def backtest_targets(self, 
                         predictions_df: pd.DataFrame, 
                         high_series: pd.Series, 
                         low_series: pd.Series,
                         look_ahead: int = 10) -> dict:
        """
        Analyze Target vs SL hits.
        Checks if Target 1/2/3 was hit before the SL over the look_ahead period.
        """
        actionable = predictions_df[predictions_df['signal'] != 0].copy()
        if len(actionable) == 0:
            return {"t1_hit_rate": 0, "t2_hit_rate": 0, "t3_hit_rate": 0, "sl_hit_rate": 0}

        results = {
            't1_hits': 0, 't2_hits': 0, 't3_hits': 0, 'sl_hits': 0, 'total': 0
        }

        for idx, row in actionable.iterrows():
            sig = row['signal']
            t1, t2, t3 = row['target_1'], row['target_2'], row['target_3']
            sl = row['stop_loss']
            
            # Get data for the next 'look_ahead' days
            try:
                start_loc = high_series.index.get_loc(idx)
                fwd_high = high_series.iloc[start_loc+1 : start_loc+look_ahead+1]
                fwd_low = low_series.iloc[start_loc+1 : start_loc+look_ahead+1]
            except (KeyError, IndexError):
                continue
                
            if len(fwd_high) == 0:
                continue
                
            results['total'] += 1
            
            if sig == 1: # LONG
                # Find first day where SL hit or T1 hit
                sl_hit_idx = np.where(fwd_low <= sl)[0]
                t1_hit_idx = np.where(fwd_high >= t1)[0]
                t2_hit_idx = np.where(fwd_high >= t2)[0]
                t3_hit_idx = np.where(fwd_high >= t3)[0]
                
                # Check if T1 was hit before SL
                first_sl = sl_hit_idx[0] if len(sl_hit_idx) > 0 else 999
                if len(t1_hit_idx) > 0 and t1_hit_idx[0] < first_sl:
                    results['t1_hits'] += 1
                if len(t2_hit_idx) > 0 and t2_hit_idx[0] < first_sl:
                    results['t2_hits'] += 1
                if len(t3_hit_idx) > 0 and t3_hit_idx[0] < first_sl:
                    results['t3_hits'] += 1
                if len(sl_hit_idx) > 0 and (len(t1_hit_idx) == 0 or first_sl < t1_hit_idx[0]):
                    results['sl_hits'] += 1
                    
            elif sig == -1: # SHORT
                sl_hit_idx = np.where(fwd_high >= sl)[0]
                t1_hit_idx = np.where(fwd_low <= t1)[0]
                t2_hit_idx = np.where(fwd_low <= t2)[0]
                t3_hit_idx = np.where(fwd_low <= t3)[0]
                
                first_sl = sl_hit_idx[0] if len(sl_hit_idx) > 0 else 999
                if len(t1_hit_idx) > 0 and t1_hit_idx[0] < first_sl:
                    results['t1_hits'] += 1
                if len(t2_hit_idx) > 0 and t2_hit_idx[0] < first_sl:
                    results['t2_hits'] += 1
                if len(t3_hit_idx) > 0 and t3_hit_idx[0] < first_sl:
                    results['t3_hits'] += 1
                if len(sl_hit_idx) > 0 and (len(t1_hit_idx) == 0 or first_sl < t1_hit_idx[0]):
                    results['sl_hits'] += 1

        # Calculate final rates
        total = results['total'] if results['total'] > 0 else 1
        return {
            "t1_hit_rate": round(results['t1_hits'] / total, 4),
            "t2_hit_rate": round(results['t2_hits'] / total, 4),
            "t3_hit_rate": round(results['t3_hits'] / total, 4),
            "sl_hit_rate": round(results['sl_hits'] / total, 4),
            "total_signals": results['total']
        }


if __name__ == "__main__":
    print("HitRatioBacktester - Ready for use")
    print("""
Example:
    from HitRatioBacktester import HitRatioBacktester
    bt = HitRatioBacktester(forward_days=[5, 10])
    stats = bt.backtest(predictions_df, price_series)
    print(stats)
    """)
