#!/usr/bin/env python3
"""
Options-Stock Data Integrator for Athena
Combines options flow with stock data for ML model feeding
"""

import logging
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptionsFlowMetrics:
    """Container for calculated options flow metrics"""
    total_call_volume: float
    total_put_volume: float
    call_put_ratio: float
    net_gamma_exposure: float
    dealer_gamma_exposure: float
    flow_sentiment: str  # 'bullish', 'bearish', 'neutral'
    unusual_activity_score: float
    
    
class BlackScholesCalculator:
    """Simplified Black-Scholes for Greeks calculation"""
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calculate d1 parameter"""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2 parameter"""
        return BlackScholesCalculator.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def norm_cdf(x):
        """Standard normal cumulative distribution function"""
        return 0.5 * (1 + math.erf(x / np.sqrt(2)))
    
    @staticmethod
    def norm_pdf(x):
        """Standard normal probability density function"""
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    
    @classmethod
    def calculate_gamma(cls, S, K, T, r, sigma):
        """Calculate gamma for option"""
        if T <= 0 or sigma <= 0:
            return 0
        
        d1_val = cls.d1(S, K, T, r, sigma)
        gamma = cls.norm_pdf(d1_val) / (S * sigma * np.sqrt(T))
        return gamma
    
    @classmethod
    def calculate_delta(cls, S, K, T, r, sigma, option_type='call'):
        """Calculate delta for option"""
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1_val = cls.d1(S, K, T, r, sigma)
        
        if option_type == 'call':
            return cls.norm_cdf(d1_val)
        else:
            return cls.norm_cdf(d1_val) - 1


class OptionsStockIntegrator:
    """Integrates options and stock data for ML model input"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.risk_free_rate = 0.05  # 5% risk-free rate
        self.implied_volatility = 0.25  # Default 25% IV
        
    def load_latest_stock_data(self, symbol: str, lookback_hours: int = 24) -> pl.DataFrame:
        """Load latest stock minute data"""
        try:
            # Look for stock data files
            stock_pattern = f"{symbol}_minute_*.parquet"
            stock_files = list(self.data_dir.glob(f"**/stocks/**/{stock_pattern}"))
            
            if not stock_files:
                logger.warning(f"No stock data found for {symbol}")
                return pl.DataFrame()
            
            # Load most recent file
            latest_file = max(stock_files, key=lambda x: x.stat().st_mtime)
            stock_df = pl.read_parquet(latest_file)
            
            # Filter to recent data
            cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
            stock_df = stock_df.filter(
                pl.col('timestamp') >= cutoff_time
            ).sort('timestamp')
            
            logger.info(f"Loaded {stock_df.shape[0]} stock records for {symbol}")
            return stock_df
            
        except Exception as e:
            logger.error(f"Failed to load stock data for {symbol}: {e}")
            return pl.DataFrame()
    
    def load_latest_options_data(self, symbol: str, lookback_hours: int = 24) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load latest options contracts and data"""
        try:
            # Load options chain
            chain_pattern = f"{symbol}_chain_*.parquet"
            chain_files = list(self.data_dir.glob(f"**/options/chains/{chain_pattern}"))
            
            # Load options data
            data_pattern = f"{symbol}_options_*.parquet"
            data_files = list(self.data_dir.glob(f"**/options/data/{data_pattern}"))
            
            contracts_df = pl.DataFrame()
            options_df = pl.DataFrame()
            
            if chain_files:
                latest_chain = max(chain_files, key=lambda x: x.stat().st_mtime)
                contracts_df = pl.read_parquet(latest_chain)
                logger.info(f"Loaded {contracts_df.shape[0]} option contracts for {symbol}")
            
            if data_files:
                latest_data = max(data_files, key=lambda x: x.stat().st_mtime)
                options_df = pl.read_parquet(latest_data)
                
                # Filter to recent data
                cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
                options_df = options_df.filter(
                    pl.col('timestamp_utc') >= cutoff_time
                ).sort('timestamp_utc')
                
                logger.info(f"Loaded {options_df.shape[0]} options records for {symbol}")
            
            return contracts_df, options_df
            
        except Exception as e:
            logger.error(f"Failed to load options data for {symbol}: {e}")
            return pl.DataFrame(), pl.DataFrame()
    
    def calculate_options_greeks(self, contracts_df: pl.DataFrame, current_price: float) -> pl.DataFrame:
        """Calculate Greeks for options contracts"""
        if contracts_df.is_empty():
            return contracts_df
        
        logger.info(f"Calculating Greeks for {contracts_df.shape[0]} contracts at price ${current_price:.2f}")
        
        greeks_data = []
        
        for contract in contracts_df.iter_rows(named=True):
            try:
                # Time to expiry in years
                dte = contract['trading_dte']
                T = max(dte / 365.0, 1/365.0)  # Minimum 1 day
                
                K = contract['strike_price']
                S = current_price
                r = self.risk_free_rate
                sigma = self.implied_volatility
                option_type = contract['contract_type']
                
                # Calculate Greeks
                gamma = BlackScholesCalculator.calculate_gamma(S, K, T, r, sigma)
                delta = BlackScholesCalculator.calculate_delta(S, K, T, r, sigma, option_type)
                
                greeks_data.append({
                    'ticker': contract['ticker'],
                    'strike_price': K,
                    'contract_type': option_type,
                    'trading_dte': dte,
                    'delta': delta,
                    'gamma': gamma,
                    'moneyness': contract.get('moneyness', 'Unknown')
                })
                
            except Exception as e:
                logger.warning(f"Failed to calculate Greeks for {contract['ticker']}: {e}")
                continue
        
        if greeks_data:
            return pl.DataFrame(greeks_data)
        else:
            return pl.DataFrame()
    
    def calculate_gamma_exposure(self, options_df: pl.DataFrame, greeks_df: pl.DataFrame) -> Dict:
        """Calculate market maker gamma exposure"""
        if options_df.is_empty() or greeks_df.is_empty():
            return {'net_gamma_exposure': 0, 'dealer_gamma_exposure': 0}
        
        try:
            # Join options data with Greeks
            combined = options_df.join(
                greeks_df, 
                on=['contract_ticker'], 
                how='left'
            ).filter(pl.col('gamma').is_not_null())
            
            if combined.is_empty():
                return {'net_gamma_exposure': 0, 'dealer_gamma_exposure': 0}
            
            # Calculate gamma exposure per contract
            # GEX = Gamma * Open Interest * 100 * Spot^2 * 0.01
            # Simplified: using volume as proxy for OI
            combined = combined.with_columns([
                (pl.col('gamma') * pl.col('volume') * 100 * 0.01).alias('gamma_exposure_per_share')
            ])
            
            # Aggregate by call/put
            gamma_summary = combined.group_by(['contract_type']).agg([
                pl.col('gamma_exposure_per_share').sum().alias('total_gamma_exposure'),
                pl.col('volume').sum().alias('total_volume')
            ])
            
            # Calculate net gamma exposure
            call_gamma = 0
            put_gamma = 0
            
            for row in gamma_summary.iter_rows(named=True):
                if row['contract_type'] == 'call':
                    call_gamma = row['total_gamma_exposure']
                elif row['contract_type'] == 'put':
                    put_gamma = row['total_gamma_exposure']
            
            # Net gamma exposure (market maker perspective)
            # Dealers are short gamma, so positive GEX = resistance level
            net_gex = call_gamma - put_gamma
            dealer_gex = -net_gex  # Dealer perspective (opposite of customer)
            
            logger.info(f"Gamma Exposure - Net: {net_gex:.2f}, Dealer: {dealer_gex:.2f}")
            
            return {
                'net_gamma_exposure': net_gex,
                'dealer_gamma_exposure': dealer_gex,
                'call_gamma': call_gamma,
                'put_gamma': put_gamma
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate gamma exposure: {e}")
            return {'net_gamma_exposure': 0, 'dealer_gamma_exposure': 0}
    
    def calculate_options_flow_metrics(self, options_df: pl.DataFrame, greeks_df: pl.DataFrame, 
                                     current_price: float) -> OptionsFlowMetrics:
        """Calculate comprehensive options flow metrics"""
        if options_df.is_empty():
            return OptionsFlowMetrics(0, 0, 1.0, 0, 0, 'neutral', 0)
        
        try:
            # Basic volume metrics
            call_volume = options_df.filter(pl.col('contract_type') == 'call')['volume'].sum()
            put_volume = options_df.filter(pl.col('contract_type') == 'put')['volume'].sum()
            
            call_put_ratio = call_volume / max(put_volume, 1)
            
            # Gamma exposure
            gex_metrics = self.calculate_gamma_exposure(options_df, greeks_df)
            
            # Flow sentiment
            if call_put_ratio > 1.5:
                sentiment = 'bullish'
            elif call_put_ratio < 0.67:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            # Unusual activity score (simplified)
            total_volume = call_volume + put_volume
            avg_volume = options_df['volume'].mean()
            unusual_score = min(total_volume / max(avg_volume * 10, 1), 10.0)
            
            return OptionsFlowMetrics(
                total_call_volume=float(call_volume),
                total_put_volume=float(put_volume),
                call_put_ratio=float(call_put_ratio),
                net_gamma_exposure=gex_metrics['net_gamma_exposure'],
                dealer_gamma_exposure=gex_metrics['dealer_gamma_exposure'],
                flow_sentiment=sentiment,
                unusual_activity_score=float(unusual_score)
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate flow metrics: {e}")
            return OptionsFlowMetrics(0, 0, 1.0, 0, 0, 'neutral', 0)
    
    def create_ml_features(self, symbol: str, lookback_hours: int = 6) -> pl.DataFrame:
        """Create ML-ready features combining stock and options data"""
        logger.info(f"Creating ML features for {symbol}")
        
        # Load data
        stock_df = self.load_latest_stock_data(symbol, lookback_hours)
        contracts_df, options_df = self.load_latest_options_data(symbol, lookback_hours)
        
        if stock_df.is_empty():
            logger.warning(f"No stock data available for {symbol}")
            return pl.DataFrame()
        
        # Get current price
        current_price = float(stock_df.select('close').tail(1).item())
        
        # Calculate Greeks if we have options data
        options_features = {}
        if not contracts_df.is_empty() and not options_df.is_empty():
            greeks_df = self.calculate_options_greeks(contracts_df, current_price)
            flow_metrics = self.calculate_options_flow_metrics(options_df, greeks_df, current_price)
            
            options_features = {
                'call_volume': flow_metrics.total_call_volume,
                'put_volume': flow_metrics.total_put_volume,
                'call_put_ratio': flow_metrics.call_put_ratio,
                'net_gamma_exposure': flow_metrics.net_gamma_exposure,
                'dealer_gamma_exposure': flow_metrics.dealer_gamma_exposure,
                'flow_sentiment_bullish': 1.0 if flow_metrics.flow_sentiment == 'bullish' else 0.0,
                'flow_sentiment_bearish': 1.0 if flow_metrics.flow_sentiment == 'bearish' else 0.0,
                'unusual_activity_score': flow_metrics.unusual_activity_score,
                'options_data_available': 1.0
            }
        else:
            # Default values when no options data
            options_features = {
                'call_volume': 0,
                'put_volume': 0,
                'call_put_ratio': 1.0,
                'net_gamma_exposure': 0,
                'dealer_gamma_exposure': 0,
                'flow_sentiment_bullish': 0.0,
                'flow_sentiment_bearish': 0.0,
                'unusual_activity_score': 0.0,
                'options_data_available': 0.0
            }
        
        # Create base stock features (simplified)
        if stock_df.shape[0] >= 2:
            price_change = float(stock_df.select('close').tail(1).item() - stock_df.select('close').tail(2).head(1).item())
            price_change_pct = price_change / float(stock_df.select('close').tail(2).head(1).item()) * 100
            avg_volume = float(stock_df.select('volume').mean())
            recent_volume = float(stock_df.select('volume').tail(1).item())
            volume_ratio = recent_volume / max(avg_volume, 1)
        else:
            price_change = 0
            price_change_pct = 0
            volume_ratio = 1.0
        
        # Combine all features
        ml_features = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volume_ratio': volume_ratio,
            **options_features
        }
        
        # Create DataFrame
        features_df = pl.DataFrame([ml_features])
        
        logger.info(f"Created ML features for {symbol}:")
        logger.info(f"  Current Price: ${current_price:.2f}")
        logger.info(f"  Price Change: {price_change_pct:.2f}%")
        logger.info(f"  C/P Ratio: {options_features['call_put_ratio']:.2f}")
        logger.info(f"  Gamma Exposure: {options_features['net_gamma_exposure']:.2f}")
        
        return features_df
    
    def create_batch_ml_features(self, symbols: List[str], lookback_hours: int = 6) -> pl.DataFrame:
        """Create ML features for multiple symbols"""
        logger.info(f"Creating batch ML features for {len(symbols)} symbols")
        
        all_features = []
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Processing {i+1}/{len(symbols)}: {symbol}")
            
            try:
                features_df = self.create_ml_features(symbol, lookback_hours)
                if not features_df.is_empty():
                    all_features.append(features_df)
            except Exception as e:
                logger.error(f"Failed to create features for {symbol}: {e}")
                continue
        
        if all_features:
            combined_df = pl.concat(all_features)
            logger.info(f"Created features for {combined_df.shape[0]} symbols")
            return combined_df
        else:
            logger.warning("No features created for any symbol")
            return pl.DataFrame()
    
    def save_ml_features(self, features_df: pl.DataFrame, output_dir: str = "data/ml_ready"):
        """Save ML-ready features to file"""
        if features_df.is_empty():
            logger.warning("No features to save")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = Path(output_dir) / f"ml_features_{timestamp}.parquet"
        
        features_df.write_parquet(output_file)
        logger.info(f"Saved ML features: {output_file}")
        
        return output_file


def main():
    """Test the integrator"""
    integrator = OptionsStockIntegrator()
    
    # Test symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Create features
    features_df = integrator.create_batch_ml_features(test_symbols, lookback_hours=12)
    
    if not features_df.is_empty():
        # Save features
        output_file = integrator.save_ml_features(features_df)
        
        # Show sample
        print("\n📊 Sample ML Features:")
        sample_cols = ['symbol', 'current_price', 'price_change_pct', 'call_put_ratio', 'net_gamma_exposure']
        print(features_df.select(sample_cols))
        
        print(f"\n✅ ML features saved to: {output_file}")
    else:
        print("\n❌ No features created")


if __name__ == "__main__":
    main()