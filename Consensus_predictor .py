"""
Consensus Voting Predictor
Aggregates 400+ indicators using weighted voting to predict:
1. Direction (buy/sell/hold)
2. Confidence score
3. Price targets
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
# NOTE: scikit-learn is imported lazily inside PriceTargetPredictor (T010) so that
# ConsensusPredictor can be used without scikit-learn installed.

class ConsensusPredictor:
    """
    Multi-indicator consensus voting system
    Predicts direction, confidence, and price targets
    
    IMPROVEMENTS IMPLEMENTED:
    1. Market Regime Awareness (Trending vs Ranging)
    2. Time-Decay Indicator Quality (Recent performance weights)
    3. Category Grouping (Prevents indicator redundancy)
    4. Meta-Ensemble Voting (Non-linear combination)
    """
    
    def __init__(self, voting_threshold: float = 0.15, confidence_levels: Dict = None,
                 weight_config: Dict = None, config_level: str = "default"):
        """
        Args:
            voting_threshold: Minimum consensus score for signal (-1 to +1)
            confidence_levels: Dict mapping confidence to thresholds
            weight_config: Optional dict overriding the default consensus weights
                (feature 001-indicator-optimization). Shape:
                {'category': {cat: w, ...}, 'voting_threshold': float,
                 'regime_multipliers': {'trend_trending_boost': float,
                                         'momentum_ranging_boost': float}}
            config_level: Provenance label recorded on every prediction
                ('default' | 'global' | 'symbol'). (FR-008)
        """
        wc = weight_config or {}
        self.voting_threshold = float(wc.get('voting_threshold', voting_threshold))
        self.confidence_levels = confidence_levels or {
            'very_high': 0.6,
            'high': 0.4,
            'medium': 0.2,
            'low': 0.0
        }
        # Default category weights (previously hardcoded inside predict()).
        self.cat_weights = {
            'trend': 1.2, 'price_action': 1.1, 'momentum': 1.0,
            'volatility': 0.8, 'volume': 0.9, 'other': 0.7,
        }
        self.cat_weights.update(wc.get('category', {}) or {})
        # Regime contextual boosts (previously hardcoded 0.2 / 1.2 in predict()).
        _regime = wc.get('regime_multipliers', {}) or {}
        self.regime_trend_boost = float(_regime.get('trend_trending_boost', 0.2))
        self.regime_momentum_boost = float(_regime.get('momentum_ranging_boost', 0.2))
        # Weight of the Open-Interest positioning vote in the final consensus blend
        # (0 = ignore). Treated like one extra high-quality "super-indicator".
        self.oi_weight = float(wc.get('oi_weight', 0.0))
        self.config_level = config_level
        # Categories for grouping
        self.categories = {
            'trend': ['ma', 'ema', 'sma', 'supertrend', 'ichimoku', 'adx', 'psar'],
            'momentum': ['rsi', 'macd', 'stoch', 'cci', 'roc', 'williams', 'mfi'],
            'volatility': ['bb', 'atr', 'std', 'keltner', 'bollinger', 'zscore'],
            'volume': ['obv', 'surge', 'vwap', 'emv', 'mfi_vol'],
            'price_action': ['above', 'below', 'cross', 'candle', 'pivot']
        }
        
    def _detect_regime(self, current_price: pd.Series, adx: pd.Series = None) -> pd.Series:
        """
        Detect market regime: 1 (Trending), 0 (Ranging/Sideways)
        """
        if adx is not None:
            # ADX > 25 usually indicates a strong trend
            return (adx > 25).astype(int)
        
        # Fallback: Use price distance from a 20-day MA
        ma20 = current_price.rolling(20).mean()
        std20 = current_price.rolling(20).std()
        dist = np.abs(current_price - ma20) / (std20 + 1e-10)
        return (dist > 1.5).astype(int)

    def _group_indicators(self, columns: list) -> Dict[str, list]:
        """Group indicator columns by their category"""
        grouped = {cat: [] for cat in self.categories}
        grouped['other'] = []
        
        for col in columns:
            found = False
            col_lower = col.lower()
            for cat, keywords in self.categories.items():
                if any(k in col_lower for k in keywords):
                    grouped[cat].append(col)
                    found = True
                    break
            if not found:
                grouped['other'].append(col)
        return grouped

    def predict(self,
                indicators_df: pd.DataFrame,
                quality_df: pd.DataFrame,
                current_price: pd.Series,
                atr: pd.Series = None,
                adx: pd.Series = None,
                oi_signal: pd.Series = None) -> pd.DataFrame:
        """
        Generate predictions using advanced Meta-Ensemble consensus

        oi_signal: optional Open-Interest directional vote in [-1, +1], aligned to
            indicators_df.index (see oi_features.oi_direction_signal). Blended into
            the final consensus with weight self.oi_weight (0 => ignored).
        """
        print("Calculating advanced consensus predictions...")
        
        # 1. Regime Detection
        regime = self._detect_regime(current_price, adx)
        
        # 2. Category Grouping
        grouped_cols = self._group_indicators(indicators_df.columns)
        
        # 3. Handle data cleaning
        indicators_clean = indicators_df.fillna(0)
        quality_clean = quality_df.fillna(0.5)
        
        # 4. Meta-Ensemble: Calculate consensus per category
        category_scores = {}
        for cat, cols in grouped_cols.items():
            if not cols:
                continue
            
            # Weighted average for this category
            cat_ind = indicators_clean[cols]
            cat_qual = quality_clean[cols]
            
            # Contextual weighting: Trend indicators get +20% weight in trending regime
            if cat == 'trend':
                cat_qual = cat_qual.multiply(1 + self.regime_trend_boost * regime, axis=0)
            elif cat == 'momentum' and regime.iloc[-1] == 0:
                cat_qual = cat_qual.multiply(1 + self.regime_momentum_boost, axis=0) # Momentum better in ranges
            
            w_sum = (cat_ind * cat_qual).sum(axis=1)
            q_sum = cat_qual.sum(axis=1)
            category_scores[cat] = w_sum / (q_sum + 1e-10)
            
        # 5. Combine Category Scores (Meta-Ensemble)
        # Instead of 400 indicators, we now combine ~6 category "super-indicators"
        cat_df = pd.DataFrame(category_scores)
        
        # Final consensus is a weighted average of categories.
        # Weights are configurable via weight_config; defaults set in __init__.
        cat_weights = self.cat_weights

        final_w_sum = sum(cat_df[cat] * cat_weights.get(cat, 1.0) for cat in cat_df.columns)
        final_q_sum = sum(np.ones(len(cat_df)) * cat_weights.get(cat, 1.0) for cat in cat_df.columns)

        # 5b. Open-Interest positioning vote (extra super-indicator).
        oi_aligned = None
        if oi_signal is not None and self.oi_weight > 0:
            oi_aligned = (oi_signal.reindex(indicators_df.index)
                          .clip(-1, 1).fillna(0.0))
            final_w_sum = final_w_sum + self.oi_weight * oi_aligned
            final_q_sum = final_q_sum + self.oi_weight

        consensus_score = final_w_sum / final_q_sum
        
        # 6. Generate final signals
        signals = np.zeros(len(consensus_score))
        signals[consensus_score > self.voting_threshold] = 1
        signals[consensus_score < -self.voting_threshold] = -1
        
        # 7. Confidence & Counts
        confidence = np.abs(consensus_score)
        bullish_count = (indicators_clean > 0).sum(axis=1)
        bearish_count = (indicators_clean < 0).sum(axis=1)
        
        # Target calculation (enhanced with regime)
        if atr is not None:
            # In trending markets, targets can be further away
            multiplier = 1.0 + (0.5 * regime)
            
            # Historical Targets for Backtesting
            # Target 1: 1.5x ATR, Target 2: 3x ATR, Target 3: 5x ATR
            t1_dist = atr * 1.5 * multiplier
            t2_dist = atr * 3.0 * multiplier
            t3_dist = atr * 5.0 * multiplier
            sl_dist = atr * 1.0 # 1x ATR SL
            
            expected_move_pct = (atr / current_price) * confidence * multiplier
        else:
            vol = current_price.pct_change().rolling(20).std()
            t1_dist = current_price * vol * 2
            t2_dist = current_price * vol * 4
            t3_dist = current_price * vol * 6
            sl_dist = current_price * vol * 1.5
            expected_move_pct = vol * confidence * 2
            
        # Directional Targets
        target_1 = np.where(signals >= 0, current_price + t1_dist, current_price - t1_dist)
        target_2 = np.where(signals >= 0, current_price + t2_dist, current_price - t2_dist)
        target_3 = np.where(signals >= 0, current_price + t3_dist, current_price - t3_dist)
        stop_loss = np.where(signals >= 0, current_price - sl_dist, current_price + sl_dist)
        
        results = pd.DataFrame({
            'signal': signals,
            'confidence': confidence,
            'consensus_score': consensus_score,
            'regime': regime,
            'bullish_pct': (bullish_count / indicators_clean.shape[1]) * 100,
            'bearish_pct': (bearish_count / indicators_clean.shape[1]) * 100,
            'target_1': target_1,
            'target_2': target_2,
            'target_3': target_3,
            'stop_loss': stop_loss,
            'expected_move_pct': expected_move_pct * 100,
            'current_price': current_price,
            'config_level': self.config_level
        }, index=indicators_df.index)

        if oi_aligned is not None:
            results['oi_signal'] = oi_aligned

        return results

    def calculate_indicator_performance(self,
                                       indicators_df: pd.DataFrame,
                                       future_returns: pd.Series,
                                       lookback: int = 20,
                                       decay: float = 0.95) -> pd.DataFrame:
        """
        Calculate performance with TIME-DECAY (Recent signals matter more)
        """
        print(f"Calculating time-decayed performance (decay={decay})...")
        performance = []
        
        # Weights for time decay
        weights = np.array([decay**i for i in range(len(future_returns))])[::-1]
        
        for col in indicators_df.columns:
            sig = indicators_df[col].fillna(0)
            ret = future_returns.fillna(0)
            
            # Directional alignment
            correct = (np.sign(sig) == np.sign(ret)).astype(float)
            
            # Weighted accuracy
            weighted_acc = np.average(correct, weights=weights)
            
            # Weighted correlation (approx)
            w_corr = (sig * ret * weights).sum() / (weights.sum() + 1e-10)
            
            performance.append({
                'indicator': col,
                'accuracy': weighted_acc,
                'contribution': w_corr,
                'sharpe_proxy': weighted_acc * np.abs(w_corr)
            })
            
        return pd.DataFrame(performance).sort_values('sharpe_proxy', ascending=False)

    
    def get_adaptive_weights(self,
                            performance_df: pd.DataFrame,
                            quality_df: pd.DataFrame,
                            adaptation_rate: float = 0.5) -> pd.DataFrame:
        """
        Calculate adaptive weights based on historical performance
        
        Args:
            performance_df: DataFrame from calculate_indicator_performance
            quality_df: Original quality scores
            adaptation_rate: How much to weight performance (0=ignore, 1=full)
            
        Returns:
            Updated quality DataFrame with adaptive weights
        """
        # Create mapping from performance
        perf_map = performance_df.set_index('indicator')['sharpe_proxy'].to_dict()
        
        # Normalize performance scores to 0-1
        perf_values = np.array(list(perf_map.values()))
        perf_min, perf_max = perf_values.min(), perf_values.max()
        perf_normalized = {k: (v - perf_min) / (perf_max - perf_min + 1e-10) 
                          for k, v in perf_map.items()}
        
        # Update quality scores
        adaptive_quality = quality_df.copy()
        
        for col in quality_df.columns:
            if col in perf_normalized:
                original_quality = quality_df[col].iloc[0]
                performance_quality = perf_normalized[col]
                
                # Blend original and performance-based quality
                adaptive_quality[col] = (
                    (1 - adaptation_rate) * original_quality +
                    adaptation_rate * performance_quality
                )
        
        return adaptive_quality


class PriceTargetPredictor:
    """
    Predict specific price targets and probability of hitting them
    """
    
    def __init__(self):
        from sklearn.preprocessing import StandardScaler  # lazy import (T010)
        self.scaler = StandardScaler()
        
    def predict_targets(self,
                       current_price: float,
                       consensus_score: float,
                       atr: float,
                       volatility: float,
                       trend_strength: float = 0.5) -> Dict:
        """
        Calculate multiple price targets with probabilities
        
        Args:
            current_price: Current stock price
            consensus_score: Consensus score from predictor (-1 to +1)
            atr: Average True Range
            volatility: Historical volatility (std of returns)
            trend_strength: Trend strength indicator (0 to 1)
            
        Returns:
            Dict with targets and probabilities
        """
        direction = np.sign(consensus_score)
        confidence = np.abs(consensus_score)
        
        # Calculate target distances based on ATR and volatility
        # More aggressive targets when confidence is high
        target_1x = atr * (1 + confidence * 0.5)  # Conservative
        target_2x = atr * (2 + confidence)          # Moderate
        target_3x = atr * (3 + confidence * 2)      # Aggressive
        
        if direction > 0:  # Bullish
            targets = {
                'target_1': current_price + target_1x,
                'target_2': current_price + target_2x,
                'target_3': current_price + target_3x,
                'stop_loss': current_price - atr * 0.5
            }
        else:  # Bearish
            targets = {
                'target_1': current_price - target_1x,
                'target_2': current_price - target_2x,
                'target_3': current_price - target_3x,
                'stop_loss': current_price + atr * 0.5
            }
        
        # Estimate probabilities (simplified model)
        # Higher confidence + stronger trend = higher probability
        base_prob = 0.5 + confidence * 0.3 + trend_strength * 0.2
        
        probabilities = {
            'prob_target_1': min(base_prob, 0.85),
            'prob_target_2': min(base_prob * 0.7, 0.70),
            'prob_target_3': min(base_prob * 0.4, 0.50)
        }
        
        # Expected days to hit (rough estimate)
        base_days = 5
        days_estimate = {
            'days_target_1': base_days,
            'days_target_2': base_days * 2,
            'days_target_3': base_days * 4
        }
        
        return {
            **targets,
            **probabilities,
            **days_estimate,
            'direction': 'LONG' if direction > 0 else 'SHORT',
            'confidence': confidence
        }
    
    def calculate_risk_reward(self, targets: Dict, current_price: float) -> Dict:
        """Calculate risk-reward ratios"""
        stop_loss = targets['stop_loss']
        risk = abs(current_price - stop_loss)
        
        rr_ratios = {}
        for i in range(1, 4):
            target = targets[f'target_{i}']
            reward = abs(target - current_price)
            rr_ratios[f'rr_target_{i}'] = reward / risk if risk > 0 else 0
        
        return rr_ratios


if __name__ == "__main__":
    print("Consensus Predictor - Ready for use")
    print("\nExample usage:")
    print("""
    from consensus_predictor import ConsensusPredictor
    
    predictor = ConsensusPredictor(voting_threshold=0.15)
    predictions = predictor.predict(indicators_df, quality_df, prices, atr)
    """)