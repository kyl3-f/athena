# src/processing/feature_engineer.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available, using fallback calculations")

logger = logging.getLogger(__name__)

class OptionsFeatureEngine:
    """
    Extract features from options data including gamma exposure
    """
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate (adjust as needed)
        
    def calculate_black_scholes_greeks(self, S: float, K: float, T: float, r: float, 
                                     sigma: float, option_type: str = 'call') -> Dict:
        """
        Calculate Black-Scholes Greeks (Delta, Gamma, Theta, Vega)
        """
        try:
            from scipy.stats import norm
            import math
            
            if T <= 0 or sigma <= 0 or S <= 0:
                return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
            
            d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
                theta = (-S*norm.pdf(d1)*sigma/(2*math.sqrt(T)) - 
                        r*K*math.exp(-r*T)*norm.cdf(d2)) / 365
            else:  # put
                delta = norm.cdf(d1) - 1
                theta = (-S*norm.pdf(d1)*sigma/(2*math.sqrt(T)) + 
                        r*K*math.exp(-r*T)*norm.cdf(-d2)) / 365
            
            gamma = norm.pdf(d1) / (S*sigma*math.sqrt(T))
            vega = S*norm.pdf(d1)*math.sqrt(T) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            }
            
        except Exception as e:
            logger.warning(f"Error calculating Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    def calculate_gamma_exposure(self, options_data: List[Dict], current_price: float) -> Dict:
        """
        Calculate total gamma exposure for the underlying
        GEX = Sum(Gamma * Open_Interest * 100 * Spot^2)
        """
        try:
            total_call_gamma = 0
            total_put_gamma = 0
            total_gamma_exposure = 0
            
            for option in options_data:
                # Extract option details
                strike = option.get('strike_price', 0)
                oi = option.get('open_interest', 0)
                contract_type = option.get('contract_type', '').upper()
                iv = option.get('implied_volatility', 0.3)  # Default to 30% IV
                expiry = option.get('expiration_date', '')
                
                if not all([strike, oi, contract_type, expiry]):
                    continue
                
                # Calculate time to expiry
                try:
                    exp_date = pd.to_datetime(expiry)
                    days_to_expiry = (exp_date - pd.Timestamp.now()).days
                    time_to_expiry = max(days_to_expiry / 365, 0.01)  # Minimum 1 day
                except:
                    time_to_expiry = 0.01
                
                # Calculate gamma using Black-Scholes
                greeks = self.calculate_black_scholes_greeks(
                    S=current_price,
                    K=strike,
                    T=time_to_expiry,
                    r=self.risk_free_rate,
                    sigma=iv,
                    option_type=contract_type.lower()
                )
                
                gamma = greeks['gamma']
                
                # Calculate gamma exposure
                # GEX = Gamma * Open Interest * 100 * Spot^2
                gamma_exposure = gamma * oi * 100 * (current_price ** 2)
                
                if contract_type == 'CALL':
                    total_call_gamma += gamma_exposure
                elif contract_type == 'PUT':
                    total_put_gamma += gamma_exposure
                
                total_gamma_exposure += gamma_exposure
            
            return {
                'total_gamma_exposure': total_gamma_exposure,
                'call_gamma_exposure': total_call_gamma,
                'put_gamma_exposure': total_put_gamma,
                'gamma_ratio': total_call_gamma / (abs(total_put_gamma) + 1e-8),
                'gamma_imbalance': (total_call_gamma - abs(total_put_gamma)) / (total_call_gamma + abs(total_put_gamma) + 1e-8)
            }
            
        except Exception as e:
            logger.error(f"Error calculating gamma exposure: {e}")
            return {
                'total_gamma_exposure': 0,
                'call_gamma_exposure': 0,
                'put_gamma_exposure': 0,
                'gamma_ratio': 0,
                'gamma_imbalance': 0
            }


class AdvancedFeatureEngineer:
    """
    Create comprehensive features for ML model training
    Includes technical indicators, market microstructure, and options features
    """
    
    def __init__(self):
        self.lookback_periods = [5, 10, 20, 50, 100]
        self.volume_periods = [5, 10, 20]
        self.options_engine = OptionsFeatureEngine()
        
    def create_comprehensive_features(self, df: pd.DataFrame, symbol: str, 
                                    options_data: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Create comprehensive feature set from clean price data
        """
        logger.info(f"Creating comprehensive features for {symbol} ({len(df)} data points)")
        
        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp').copy()
        df = df.set_index('timestamp')
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Technical indicators
        df = self._add_technical_indicators(df)
        
        # Volume features
        df = self._add_volume_features(df)
        
        # Statistical features
        df = self._add_statistical_features(df)
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Market microstructure features
        df = self._add_microstructure_features(df)
        
        # Options features (if options data available)
        if options_data:
            df = self._add_options_features(df, options_data, symbol)
        
        # Forward-looking labels (for supervised learning)
        df = self._add_future_labels(df)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns at different timeframes
        for period in self.lookback_periods:
            df[f'return_{period}'] = df['c'].pct_change(period)
            df[f'log_return_{period}'] = np.log(df['c'] / df['c'].shift(period))
        
        # Price momentum
        df['price_momentum_5'] = df['c'] / df['c'].shift(5) - 1
        df['price_momentum_20'] = df['c'] / df['c'].shift(20) - 1
        
        # Price position within recent range
        for period in [10, 20, 50]:
            rolling_high = df['h'].rolling(period).max()
            rolling_low = df['l'].rolling(period).min()
            df[f'price_position_{period}'] = (df['c'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
        
        # Price acceleration
        df['price_acceleration'] = df['c'].pct_change().diff()
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        if TALIB_AVAILABLE:
            try:
                # Trend indicators
                df['sma_10'] = talib.SMA(df['c'], timeperiod=10)
                df['sma_20'] = talib.SMA(df['c'], timeperiod=20)
                df['sma_50'] = talib.SMA(df['c'], timeperiod=50)
                df['ema_12'] = talib.EMA(df['c'], timeperiod=12)
                df['ema_26'] = talib.EMA(df['c'], timeperiod=26)
                
                # MACD
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['c'])
                
                # RSI
                df['rsi_14'] = talib.RSI(df['c'], timeperiod=14)
                
                # Bollinger Bands
                df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['c'])
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-8)
                df['bb_position'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
                
                # Stochastic
                df['stoch_k'], df['stoch_d'] = talib.STOCH(df['h'], df['l'], df['c'])
                
                # Average True Range
                df['atr_14'] = talib.ATR(df['h'], df['l'], df['c'], timeperiod=14)
                
                # Williams %R
                df['williams_r'] = talib.WILLR(df['h'], df['l'], df['c'])
                
                # Commodity Channel Index
                df['cci'] = talib.CCI(df['h'], df['l'], df['c'])
                
            except Exception as e:
                logger.warning(f"Error with TA-Lib indicators: {e}")
                self._add_fallback_indicators(df)
        else:
            self._add_fallback_indicators(df)
        
        return df
    
    def _add_fallback_indicators(self, df: pd.DataFrame):
        """Fallback indicators if TA-Lib is not available"""
        # Simple moving averages
        df['sma_10'] = df['c'].rolling(10).mean()
        df['sma_20'] = df['c'].rolling(20).mean()
        df['sma_50'] = df['c'].rolling(50).mean()
        
        # Simple RSI
        df['rsi_14'] = self._calculate_rsi(df['c'], 14)
        
        # Simple Bollinger Bands
        df['bb_middle'] = df['c'].rolling(20).mean()
        bb_std = df['c'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-8)
        df['bb_position'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        # Volume moving averages
        for period in self.volume_periods:
            df[f'volume_sma_{period}'] = df['v'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['v'] / (df[f'volume_sma_{period}'] + 1e-8)
        
        # Volume-price relationship
        if 'vwap_20' in df.columns:
            df['price_vs_vwap'] = df['c'] / (df['vwap_20'] + 1e-8) - 1
        
        # On-Balance Volume
        df['price_change_sign'] = np.sign(df['c'].diff())
        df['obv'] = (df['price_change_sign'] * df['v']).cumsum()
        
        # Volume Rate of Change
        df['volume_roc_5'] = df['v'].pct_change(5)
        
        # Money Flow Index approximation
        df['money_flow_raw'] = df['typical_price'] * df['v']
        positive_flow = df['money_flow_raw'].where(df['c'] > df['c'].shift(1), 0)
        negative_flow = df['money_flow_raw'].where(df['c'] < df['c'].shift(1), 0)
        df['mfi_14'] = 100 - (100 / (1 + positive_flow.rolling(14).sum() / (negative_flow.rolling(14).sum() + 1e-8)))
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical and volatility features"""
        # Volatility measures
        for period in [5, 10, 20]:
            returns = df['c'].pct_change()
            df[f'volatility_{period}'] = returns.rolling(period).std() * np.sqrt(252)
            df[f'skewness_{period}'] = returns.rolling(period).skew()
            df[f'kurtosis_{period}'] = returns.rolling(period).kurt()
        
        # Realized volatility (using high-frequency returns)
        df['realized_vol_20'] = (df['log_return_1min'] ** 2).rolling(20).sum() * np.sqrt(252)
        
        # Price efficiency measure (autocorrelation)
        df['autocorr_5'] = df['return_1min'].rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        # Extract time components
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Market session features
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16) & 
                               (df['day_of_week'] < 5))
        df['is_morning_session'] = (df['hour'] >= 9) & (df['hour'] < 12)
        df['is_afternoon_session'] = (df['hour'] >= 13) & (df['hour'] < 16)
        df['is_power_hour'] = (df['hour'] == 15)  # 3-4 PM ET
        df['is_opening_hour'] = (df['hour'] == 9) & (df['minute'] >= 30)
        
        # Time since market open
        market_open_minutes = 9 * 60 + 30  # 9:30 AM in minutes
        current_minutes = df['hour'] * 60 + df['minute']
        df['minutes_since_open'] = np.where(
            df['is_market_open'],
            current_minutes - market_open_minutes,
            np.nan
        )
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Bid-ask spread proxy
        df['spread_proxy'] = (df['h'] - df['l']) / (df['c'] + 1e-8)
        
        # Price impact measures
        df['price_impact'] = abs(df['c'].diff()) / np.sqrt(df['v'] + 1e-8)
        
        # Tick direction and momentum
        df['tick_direction'] = np.sign(df['c'].diff())
        df['tick_momentum_5'] = df['tick_direction'].rolling(5).sum()
        df['tick_momentum_20'] = df['tick_direction'].rolling(20).sum()
        
        # Amihud illiquidity measure
        daily_return = abs(df['c'].pct_change())
        dollar_volume = df['v'] * df['c']
        df['illiquidity'] = daily_return / (dollar_volume + 1e-8)
        
        # Kyle's Lambda (price impact of trades)
        df['kyle_lambda'] = abs(df['c'].diff()) / (df['v'] + 1e-8)
        
        return df
    
    def _add_options_features(self, df: pd.DataFrame, options_data: List[Dict], symbol: str) -> pd.DataFrame:
        """Add options-related features including gamma exposure"""
        logger.info(f"Adding options features for {symbol}")
        
        try:
            # Calculate gamma exposure for each timestamp
            gamma_features = []
            
            for timestamp in df.index:
                current_price = df.loc[timestamp, 'c']
                
                # Calculate gamma exposure
                gex_data = self.options_engine.calculate_gamma_exposure(options_data, current_price)
                
                gamma_features.append({
                    'timestamp': timestamp,
                    'total_gamma_exposure': gex_data['total_gamma_exposure'],
                    'call_gamma_exposure': gex_data['call_gamma_exposure'],
                    'put_gamma_exposure': gex_data['put_gamma_exposure'],
                    'gamma_ratio': gex_data['gamma_ratio'],
                    'gamma_imbalance': gex_data['gamma_imbalance']
                })
            
            # Convert to DataFrame and merge
            gamma_df = pd.DataFrame(gamma_features).set_index('timestamp')
            df = df.join(gamma_df, how='left')
            
            # Add derived gamma features
            df['gamma_exposure_normalized'] = df['total_gamma_exposure'] / (current_price ** 2 + 1e-8)
            df['gamma_pressure'] = np.where(df['total_gamma_exposure'] > 0, 1, -1)
            
            # Options volume features (if available)
            if options_data:
                total_call_volume = sum(opt.get('volume', 0) for opt in options_data if opt.get('contract_type') == 'CALL')
                total_put_volume = sum(opt.get('volume', 0) for opt in options_data if opt.get('contract_type') == 'PUT')
                
                df['options_call_volume'] = total_call_volume
                df['options_put_volume'] = total_put_volume
                df['options_put_call_ratio'] = total_put_volume / (total_call_volume + 1e-8)
            
            logger.info(f"Added {len(gamma_df.columns)} options features")
            
        except Exception as e:
            logger.error(f"Error adding options features: {e}")
            # Add empty options features to maintain consistency
            df['total_gamma_exposure'] = 0
            df['call_gamma_exposure'] = 0
            df['put_gamma_exposure'] = 0
            df['gamma_ratio'] = 0
            df['gamma_imbalance'] = 0
        
        return df
    
    def _add_future_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add forward-looking labels for supervised learning"""
        # Future returns at different horizons
        for horizon in [1, 5, 10, 20]:
            df[f'future_return_{horizon}'] = df['c'].shift(-horizon) / df['c'] - 1
            
            # Binary classification labels
            df[f'future_up_{horizon}'] = (df[f'future_return_{horizon}'] > 0).astype(int)
            
            # Strong move labels
            df[f'strong_up_{horizon}'] = (df[f'future_return_{horizon}'] > 0.01).astype(int)
            df[f'strong_down_{horizon}'] = (df[f'future_return_{horizon}'] < -0.01).astype(int)
        
        # Volatility prediction labels
        df['future_volatility_5'] = df['c'].pct_change().shift(-5).rolling(5).std()
        df['high_volatility_ahead'] = (df['future_volatility_5'] > 
                                      df['future_volatility_5'].rolling(20).mean()).astype(int)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI manually"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))


# Example usage
if __name__ == "__main__":
    # Test feature engineering
    logging.basicConfig(level=logging.INFO)
    
    # Sample data for testing
    sample_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-15 10:00', periods=100, freq='1min'),
        'o': np.random.randn(100).cumsum() + 100,
        'h': np.random.randn(100).cumsum() + 101,
        'l': np.random.randn(100).cumsum() + 99,
        'c': np.random.randn(100).cumsum() + 100,
        'v': np.random.randint(1000, 5000, 100),
        'typical_price': np.random.randn(100).cumsum() + 100,
        'vwap_20': np.random.randn(100).cumsum() + 100
    })
    
    # Sample options data
    sample_options = [
        {'strike_price': 100, 'open_interest': 1000, 'contract_type': 'CALL', 
         'expiration_date': '2024-01-19', 'implied_volatility': 0.25},
        {'strike_price': 105, 'open_interest': 800, 'contract_type': 'CALL', 
         'expiration_date': '2024-01-19', 'implied_volatility': 0.30},
        {'strike_price': 95, 'open_interest': 1200, 'contract_type': 'PUT', 
         'expiration_date': '2024-01-19', 'implied_volatility': 0.28}
    ]
    
    engineer = AdvancedFeatureEngineer()
    features_df = engineer.create_comprehensive_features(sample_df, 'TEST', sample_options)
    
    print(f"Created features: {features_df.shape}")
    print(f"Feature columns: {len(features_df.columns)}")
    print(f"Options features included: {'total_gamma_exposure' in features_df.columns}")