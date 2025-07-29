#!/usr/bin/env python3
"""
Options-Stock Feature Integration for Athena Trading System
Combines options Greeks, gamma exposure, and flow data with stock features
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class OptionsStockIntegrator:
    """
    Integrates options data with stock data for comprehensive feature engineering
    """
    
    def __init__(self):
        self.options_cache = {}
        self.gamma_exposure_history = {}
        
    def merge_options_with_stock_data(self, 
                                    stock_data: pl.DataFrame, 
                                    options_data: Dict[str, Dict]) -> pl.DataFrame:
        """
        Merge options features with stock data
        
        Args:
            stock_data: Stock data with OHLCV + features
            options_data: Options data from EnhancedPolygonClient
            
        Returns:
            Combined DataFrame with options and stock features
        """
        # Convert stock data to pandas for easier manipulation
        stock_df = stock_data.to_pandas() if isinstance(stock_data, pl.DataFrame) else stock_data
        
        # Add options features for each symbol
        options_features = []
        
        for symbol in stock_df['symbol'].unique():
            symbol_stock_data = stock_df[stock_df['symbol'] == symbol].copy()
            
            if symbol in options_data and 'error' not in options_data[symbol]:
                # Get options features for this symbol
                symbol_options = options_data[symbol]
                gamma_data = symbol_options.get('gamma_exposure', {})
                flow_data = symbol_options.get('flow_metrics', {})
                
                # Add options features to stock data
                symbol_stock_data['gamma_exposure'] = gamma_data.get('total_gamma_exposure', 0)
                symbol_stock_data['call_gamma'] = gamma_data.get('call_gamma_exposure', 0) 
                symbol_stock_data['put_gamma'] = gamma_data.get('put_gamma_exposure', 0)
                symbol_stock_data['net_gamma'] = gamma_data.get('net_gamma', 0)
                
                symbol_stock_data['call_volume'] = flow_data.get('call_volume', 0)
                symbol_stock_data['put_volume'] = flow_data.get('put_volume', 0)
                symbol_stock_data['call_put_ratio'] = flow_data.get('call_put_ratio', 1.0)
                
                symbol_stock_data['avg_iv_calls'] = flow_data.get('avg_iv_calls', 0)
                symbol_stock_data['avg_iv_puts'] = flow_data.get('avg_iv_puts', 0)
                symbol_stock_data['iv_skew'] = flow_data.get('iv_skew', 0)
                
                symbol_stock_data['total_call_delta'] = flow_data.get('total_call_delta', 0)
                symbol_stock_data['total_put_delta'] = flow_data.get('total_put_delta', 0)
                symbol_stock_data['net_delta'] = flow_data.get('net_delta', 0)
                
                # Calculate additional derived features
                symbol_stock_data = self._add_options_derived_features(symbol_stock_data)
                
            else:
                # Fill with zeros if no options data available
                options_columns = [
                    'gamma_exposure', 'call_gamma', 'put_gamma', 'net_gamma',
                    'call_volume', 'put_volume', 'call_put_ratio',
                    'avg_iv_calls', 'avg_iv_puts', 'iv_skew',
                    'total_call_delta', 'total_put_delta', 'net_delta'
                ]
                for col in options_columns:
                    symbol_stock_data[col] = 0
                    
                symbol_stock_data = self._add_options_derived_features(symbol_stock_data)
            
            options_features.append(symbol_stock_data)
        
        # Combine all symbols
        combined_df = pd.concat(options_features, ignore_index=True)
        
        return pl.from_pandas(combined_df)
    
    def _add_options_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived options features
        """
        # Gamma exposure relative to stock price
        df['gamma_exposure_ratio'] = df['gamma_exposure'] / (df['c'] * df['v']).replace(0, 1)
        
        # Options activity intensity
        total_options_volume = df['call_volume'] + df['put_volume']
        df['options_volume_ratio'] = total_options_volume / df['v'].replace(0, 1)
        
        # IV percentile (simplified - in production, use rolling percentiles)
        df['iv_calls_zscore'] = (df['avg_iv_calls'] - df['avg_iv_calls'].mean()) / (df['avg_iv_calls'].std() + 0.001)
        df['iv_puts_zscore'] = (df['avg_iv_puts'] - df['avg_iv_puts'].mean()) / (df['avg_iv_puts'].std() + 0.001)
        
        # Delta-adjusted volume
        df['delta_adj_call_volume'] = df['call_volume'] * abs(df['total_call_delta'])
        df['delta_adj_put_volume'] = df['put_volume'] * abs(df['total_put_delta'])
        
        # Gamma pressure indicator
        df['gamma_pressure'] = np.where(
            df['gamma_exposure'] > 0, 
            1,  # Positive gamma (supportive)
            -1  # Negative gamma (amplifying)
        )
        
        return df
    
    def calculate_gamma_levels(self, options_snapshot: pl.DataFrame, 
                             underlying_price: float,
                             price_range: float = 0.1) -> Dict:
        """
        Calculate key gamma levels and support/resistance from options
        
        Args:
            options_snapshot: Options snapshot data
            underlying_price: Current stock price
            price_range: Price range to analyze (10% default)
            
        Returns:
            Dict with gamma levels and analysis
        """
        if options_snapshot.is_empty():
            return {}
        
        # Filter options near current price
        price_min = underlying_price * (1 - price_range)
        price_max = underlying_price * (1 + price_range)
        
        nearby_options = options_snapshot.filter(
            (pl.col('strike_price') >= price_min) & 
            (pl.col('strike_price') <= price_max)
        )
        
        if nearby_options.is_empty():
            return {}
        
        # Calculate gamma exposure by strike
        gamma_by_strike = {}
        
        for row in nearby_options.iter_rows(named=True):
            strike = row['strike_price']
            gamma = row.get('gamma', 0)
            open_interest = row.get('open_interest', 0)
            option_type = row.get('option_type', 'call')
            
            if strike not in gamma_by_strike:
                gamma_by_strike[strike] = {'call_gamma': 0, 'put_gamma': 0, 'net_gamma': 0}
            
            exposure = gamma * open_interest * 100  # 100 shares per contract
            
            if option_type == 'call':
                gamma_by_strike[strike]['call_gamma'] += exposure
            else:
                gamma_by_strike[strike]['put_gamma'] += exposure
            
            gamma_by_strike[strike]['net_gamma'] = (
                gamma_by_strike[strike]['call_gamma'] + gamma_by_strike[strike]['put_gamma']
            )
        
        # Find key levels
        sorted_strikes = sorted(gamma_by_strike.keys())
        max_gamma_strike = max(sorted_strikes, key=lambda x: abs(gamma_by_strike[x]['net_gamma']))
        
        # Support and resistance levels
        resistance_levels = [
            strike for strike in sorted_strikes 
            if strike > underlying_price and gamma_by_strike[strike]['net_gamma'] > 0
        ]
        
        support_levels = [
            strike for strike in sorted_strikes
            if strike < underlying_price and gamma_by_strike[strike]['net_gamma'] > 0  
        ]
        
        return {
            'gamma_by_strike': gamma_by_strike,
            'max_gamma_strike': max_gamma_strike,
            'max_gamma_exposure': gamma_by_strike[max_gamma_strike]['net_gamma'],
            'resistance_levels': resistance_levels[:3],  # Top 3
            'support_levels': support_levels[-3:],       # Top 3
            'current_price': underlying_price,
            'analysis_range': (price_min, price_max)
        }
    
    def create_options_features_for_ml(self, 
                                     combined_data: pl.DataFrame,
                                     lookback_periods: List[int] = [5, 10, 20]) -> pl.DataFrame:
        """
        Create ML-ready options features with rolling calculations
        
        Args:
            combined_data: Combined stock + options data
            lookback_periods: Periods for rolling calculations
            
        Returns:
            DataFrame with additional options features for ML
        """
        df = combined_data.to_pandas() if isinstance(combined_data, pl.DataFrame) else combined_data
        
        # Sort by symbol and timestamp for rolling calculations
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        # Rolling options features
        for period in lookback_periods:
            # Rolling gamma exposure metrics
            df[f'gamma_exposure_sma_{period}'] = df.groupby('symbol')['gamma_exposure'].rolling(period).mean().values
            df[f'gamma_exposure_std_{period}'] = df.groupby('symbol')['gamma_exposure'].rolling(period).std().values
            
            # Rolling call/put ratios
            df[f'call_put_ratio_sma_{period}'] = df.groupby('symbol')['call_put_ratio'].rolling(period).mean().values
            
            # Rolling IV metrics
            df[f'iv_skew_sma_{period}'] = df.groupby('symbol')['iv_skew'].rolling(period).mean().values
            df[f'avg_iv_calls_sma_{period}'] = df.groupby('symbol')['avg_iv_calls'].rolling(period).mean().values
            
            # Rolling volume ratios
            df[f'options_volume_ratio_sma_{period}'] = df.groupby('symbol')['options_volume_ratio'].rolling(period).mean().values
        
        # Gamma regime indicators
        df['gamma_regime'] = np.select([
            df['gamma_exposure'] > df['gamma_exposure'].quantile(0.8),
            df['gamma_exposure'] < df['gamma_exposure'].quantile(0.2)
        ], [1, -1], default=0)  # 1: High gamma, -1: Low gamma, 0: Normal
        
        # Options flow divergence
        df['flow_divergence'] = (df['call_volume'] - df['put_volume']) / (df['call_volume'] + df['put_volume']).replace(0, 1)
        
        # Delta-hedging pressure
        df['delta_hedging_pressure'] = df['net_delta'] / df['v'].replace(0, 1)
        
        return pl.from_pandas(df)
    
    def generate_options_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generate options-based trading signals
        
        Args:
            data: Combined data with options features
            
        Returns:
            DataFrame with options signals
        """
        df = data.to_pandas() if isinstance(data, pl.DataFrame) else data
        
        # Initialize signal columns
        df['gamma_signal'] = 0
        df['flow_signal'] = 0
        df['iv_signal'] = 0
        df['combined_options_signal'] = 0
        
        # Gamma exposure signals
        df['gamma_signal'] = np.where(
            df['gamma_exposure'] > df['gamma_exposure'].quantile(0.7), 1,  # Positive gamma support
            np.where(df['gamma_exposure'] < df['gamma_exposure'].quantile(0.3), -1, 0)  # Negative gamma risk
        )
        
        # Flow signals
        df['flow_signal'] = np.where(
            df['call_put_ratio'] > 1.5, 1,  # Bullish flow
            np.where(df['call_put_ratio'] < 0.7, -1, 0)  # Bearish flow
        )
        
        # IV signals (contrarian)
        df['iv_signal'] = np.where(
            df['avg_iv_calls'] > df['avg_iv_calls'].quantile(0.8), -1,  # High IV - sell signal
            np.where(df['avg_iv_calls'] < df['avg_iv_calls'].quantile(0.2), 1, 0)  # Low IV - buy signal
        )
        
        # Combined options signal
        df['combined_options_signal'] = (df['gamma_signal'] + df['flow_signal'] + df['iv_signal']) / 3
        
        # Options signal strength
        df['options_signal_strength'] = abs(df['combined_options_signal'])
        
        return pl.from_pandas(df)

# Integration testing function
async def test_options_integration_pipeline():
    """
    Test the complete options integration pipeline
    """
    from enhanced_polygon_options import EnhancedPolygonClient
    import os
    
    # Sample stock data (using your working format)
    sample_stock_data = pl.DataFrame({
        'symbol': ['AAPL'] * 10 + ['MSFT'] * 10,
        'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(10)] * 2,
        'o': [150.0 + i for i in range(10)] + [300.0 + i for i in range(10)],
        'h': [151.0 + i for i in range(10)] + [301.0 + i for i in range(10)],
        'l': [149.0 + i for i in range(10)] + [299.0 + i for i in range(10)],
        'c': [150.5 + i for i in range(10)] + [300.5 + i for i in range(10)],
        'v': [1000000 + i * 1000 for i in range(10)] * 2,
        'typical_price': [150.17 + i for i in range(10)] + [300.17 + i for i in range(10)],
    })
    
    # Initialize integrator
    integrator = OptionsStockIntegrator()
    
    # Mock options data (in production, this comes from EnhancedPolygonClient)
    mock_options_data = {
        'AAPL': {
            'gamma_exposure': {
                'total_gamma_exposure': 1500000,
                'call_gamma_exposure': 1200000,
                'put_gamma_exposure': 300000,
                'net_gamma': 1500000
            },
            'flow_metrics': {
                'call_volume': 25000,
                'put_volume': 15000,
                'call_put_ratio': 1.67,
                'avg_iv_calls': 0.25,
                'avg_iv_puts': 0.28,
                'iv_skew': 0.03,
                'total_call_delta': 12500,
                'total_put_delta': -7500,
                'net_delta': 5000
            }
        },
        'MSFT': {
            'gamma_exposure': {
                'total_gamma_exposure': 800000,
                'call_gamma_exposure': 600000,
                'put_gamma_exposure': 200000,
                'net_gamma': 800000
            },
            'flow_metrics': {
                'call_volume': 18000,
                'put_volume': 12000,
                'call_put_ratio': 1.5,
                'avg_iv_calls': 0.22,
                'avg_iv_puts': 0.24,
                'iv_skew': 0.02,
                'total_call_delta': 9000,
                'total_put_delta': -6000,
                'net_delta': 3000
            }
        }
    }
    
    print("🧪 Testing Options Integration Pipeline...")
    
    # Step 1: Merge options with stock data
    print("📊 Step 1: Merging options with stock data...")
    combined_data = integrator.merge_options_with_stock_data(sample_stock_data, mock_options_data)
    print(f"   ✅ Combined data shape: {combined_data.shape}")
    print(f"   📈 Columns: {list(combined_data.columns)}")
    
    # Step 2: Create ML features
    print("🤖 Step 2: Creating ML-ready options features...")
    ml_features = integrator.create_options_features_for_ml(combined_data, lookback_periods=[3, 5])
    print(f"   ✅ ML features shape: {ml_features.shape}")
    print(f"   🔢 Feature count: {len(ml_features.columns)}")
    
    # Step 3: Generate options signals
    print("📡 Step 3: Generating options signals...")
    signals_data = integrator.generate_options_signals(ml_features)
    print(f"   ✅ Signals generated for {len(signals_data)} records")
    
    # Show sample results
    print("\n📋 SAMPLE RESULTS (AAPL):")
    aapl_sample = signals_data.filter(pl.col('symbol') == 'AAPL').head(1)
    if len(aapl_sample) > 0:
        sample_row = aapl_sample.to_pandas().iloc[0]
        key_features = [
            'symbol', 'c', 'gamma_exposure', 'call_put_ratio', 'iv_skew',
            'gamma_signal', 'flow_signal', 'iv_signal', 'combined_options_signal'
        ]
        for feature in key_features:
            if feature in sample_row:
                value = sample_row[feature]
                if isinstance(value, (int, float)) and feature != 'symbol':
                    print(f"   {feature}: {value:.4f}")
                else:
                    print(f"   {feature}: {value}")
    
    print("\n🎉 Options Integration Pipeline Test Complete!")
    return signals_data

# Real-world usage example
class ProductionOptionsIntegrator:
    """
    Production-ready options integrator for live trading
    """
    
    def __init__(self, polygon_api_key: str):
        self.polygon_client = EnhancedPolygonClient(polygon_api_key)
        self.integrator = OptionsStockIntegrator()
        
    async def get_live_options_enhanced_data(self, symbols: List[str]) -> pl.DataFrame:
        """
        Get live stock + options data for production trading
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with combined stock and options features
        """
        async with self.polygon_client as client:
            # Get stock data (you'll integrate with your existing stock data pipeline)
            # For now, this is a placeholder - replace with your actual stock data fetch
            
            # Get comprehensive options data
            options_data = await client.get_multi_symbol_options_data(symbols)
            
            # Mock stock data - replace with your actual pipeline
            stock_data = self._get_current_stock_data(symbols)
            
            # Merge and create features
            combined_data = self.integrator.merge_options_with_stock_data(stock_data, options_data)
            ml_features = self.integrator.create_options_features_for_ml(combined_data)
            final_data = self.integrator.generate_options_signals(ml_features)
            
            return final_data
    
    def _get_current_stock_data(self, symbols: List[str]) -> pl.DataFrame:
        """
        Placeholder for your existing stock data pipeline
        Replace this with your actual stock data fetching logic
        """
        # This would integrate with your existing PolygonClient and feature engineering
        # For now, return mock data
        pass

if __name__ == "__main__":
    # Run integration test
    import asyncio
    asyncio.run(test_options_integration_pipeline())