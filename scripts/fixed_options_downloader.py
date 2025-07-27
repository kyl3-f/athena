#!/usr/bin/env python3
"""
Fixed Athena Options Downloader
Based on actual Polygon API structure
"""

import sys
import argparse
import logging
import os
import requests
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/download_options_fixed.log')
    ]
)
logger = logging.getLogger(__name__)


class FixedOptionsManager:
    """
    Options data manager based on actual Polygon API responses
    """
    
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable required")
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'Athena-Options/1.0'
        })
        
        self.base_url = "https://api.polygon.io"
    
    def calculate_trading_days_to_expiry(self, current_date: str, expiration_date: str) -> int:
        """Calculate trading days to expiry (simplified - weekdays only)"""
        try:
            current = datetime.strptime(current_date, '%Y-%m-%d')
            expiry = datetime.strptime(expiration_date, '%Y-%m-%d')
            
            # Count weekdays between current and expiration
            dte = 0
            current_check = current + timedelta(days=1)  # Start from next day
            
            while current_check <= expiry:
                if current_check.weekday() < 5:  # Monday=0, Friday=4
                    dte += 1
                current_check += timedelta(days=1)
            
            return max(0, dte)
        except Exception as e:
            logger.error(f"Failed to calculate DTE: {e}")
            return 0
    
    def get_options_contracts(self, underlying: str, max_dte: int = 60) -> pl.DataFrame:
        """Get options contracts for underlying"""
        logger.info(f"Getting options contracts for {underlying}")
        
        try:
            url = f"{self.base_url}/v3/reference/options/contracts"
            params = {
                'underlying_ticker': underlying,
                'limit': 1000
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' not in data or not data['results']:
                logger.warning(f"No contracts found for {underlying}")
                return pl.DataFrame()
            
            contracts = data['results']
            logger.info(f"Found {len(contracts)} contracts for {underlying}")
            
            # Convert to DataFrame
            df = pl.DataFrame(contracts)
            
            # Add calculated fields
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # Calculate DTE for each contract
            dte_values = []
            for exp_date in df['expiration_date'].to_list():
                dte = self.calculate_trading_days_to_expiry(current_date, exp_date)
                dte_values.append(dte)
            
            df = df.with_columns([
                pl.Series('trading_dte', dte_values),
                pl.lit(current_date).alias('analysis_date')
            ])
            
            # Filter by DTE
            df = df.filter(
                (pl.col('trading_dte') >= 1) & 
                (pl.col('trading_dte') <= max_dte)
            )
            
            logger.info(f"After DTE filter (1-{max_dte} days): {df.shape[0]} contracts")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get contracts for {underlying}: {e}")
            return pl.DataFrame()
    
    def get_current_stock_price(self, underlying: str) -> float:
        """Get current stock price from existing data or API"""
        try:
            # Try to get from existing stock data files
            stock_files = list(Path(f"data/raw/stocks/aggregates/minute").glob(f"{underlying}_minute_*.parquet"))
            
            if stock_files:
                stock_df = pl.read_parquet(stock_files[-1])  # Most recent file
                if not stock_df.is_empty():
                    latest_price = stock_df.select('close').tail(1).item()
                    logger.info(f"Got {underlying} price from local data: ${latest_price:.2f}")
                    return float(latest_price)
            
            # Fallback: Get from API
            logger.info(f"Getting {underlying} price from API...")
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            today = datetime.now().strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/v2/aggs/ticker/{underlying}/range/1/day/{yesterday}/{today}"
            params = {'adjusted': 'true', 'sort': 'desc', 'limit': 1}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if 'results' in data and data['results']:
                price = data['results'][0]['c']  # Close price
                logger.info(f"Got {underlying} price from API: ${price:.2f}")
                return float(price)
            
        except Exception as e:
            logger.warning(f"Could not get stock price for {underlying}: {e}")
        
        # Ultimate fallback - get from contract strikes
        return 200.0  # Reasonable default for AAPL-like stocks
    
    def enhance_contracts_with_moneyness(self, contracts_df: pl.DataFrame, underlying: str) -> pl.DataFrame:
        """Add moneyness and other calculated fields"""
        if contracts_df.is_empty():
            return contracts_df
        
        stock_price = self.get_current_stock_price(underlying)
        
        enhanced_df = contracts_df.with_columns([
            pl.lit(stock_price).alias('current_stock_price'),
            
            # Moneyness ratio
            (pl.lit(stock_price) / pl.col('strike_price')).alias('moneyness_ratio'),
            
            # ITM/OTM classification
            pl.when(
                (pl.col('contract_type') == 'call') & (pl.lit(stock_price) > pl.col('strike_price'))
            ).then('ITM')
            .when(
                (pl.col('contract_type') == 'put') & (pl.lit(stock_price) < pl.col('strike_price'))
            ).then('ITM')
            .when(
                ((pl.lit(stock_price) - pl.col('strike_price')).abs() / pl.lit(stock_price)) < 0.02
            ).then('ATM')
            .otherwise('OTM').alias('moneyness'),
            
            # Distance from current price
            ((pl.lit(stock_price) - pl.col('strike_price')).abs() / pl.lit(stock_price)).alias('distance_from_current'),
        ])
        
        return enhanced_df
    
    def download_options_data(self, contracts_df: pl.DataFrame, start_date: str, end_date: str) -> pl.DataFrame:
        """Download options aggregates data for contracts"""
        if contracts_df.is_empty():
            return pl.DataFrame()
        
        logger.info(f"Downloading options data for {contracts_df.shape[0]} contracts")
        
        all_data = []
        
        # Limit to most liquid contracts to avoid too many API calls
        liquid_contracts = contracts_df.filter(
            (pl.col('distance_from_current') <= 0.2) &  # Within 20% of current price
            (pl.col('trading_dte') <= 45)  # Within 45 days
        ).head(50)  # Max 50 contracts
        
        logger.info(f"Processing {liquid_contracts.shape[0]} liquid contracts")
        
        for i, contract_row in enumerate(liquid_contracts.iter_rows(named=True)):
            ticker = contract_row['ticker']
            logger.info(f"  {i+1}/{liquid_contracts.shape[0]}: {ticker}")
            
            try:
                url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{end_date}"
                params = {
                    'adjusted': 'true',
                    'sort': 'asc',
                    'limit': 5000
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if 'results' in data and data['results']:
                    results = data['results']
                    
                    # Convert to DataFrame with proper column names
                    options_df = pl.DataFrame(results).rename({
                        'v': 'volume',
                        'vw': 'vwap', 
                        'o': 'open',
                        'c': 'close',
                        'h': 'high',
                        'l': 'low',
                        't': 'timestamp_ms',
                        'n': 'transactions'
                    })
                    
                    # Add contract metadata
                    options_df = options_df.with_columns([
                        pl.from_epoch(pl.col('timestamp_ms'), time_unit='ms').alias('timestamp_utc'),
                        pl.lit(ticker).alias('contract_ticker'),
                        pl.lit(contract_row['underlying_ticker']).alias('underlying_symbol'),
                        pl.lit(contract_row['contract_type']).alias('contract_type'),
                        pl.lit(contract_row['strike_price']).alias('strike_price'),
                        pl.lit(contract_row['expiration_date']).alias('expiration_date'),
                        pl.lit(contract_row['trading_dte']).alias('trading_dte'),
                        pl.lit(contract_row['moneyness']).alias('moneyness'),
                        pl.lit(contract_row['current_stock_price']).alias('stock_price_at_analysis'),
                    ])
                    
                    all_data.append(options_df)
                    logger.info(f"    Downloaded {options_df.shape[0]} minute bars")
                else:
                    logger.warning(f"    No data for {ticker}")
                    
            except Exception as e:
                logger.warning(f"    Failed to download {ticker}: {e}")
                continue
        
        if all_data:
            combined_df = pl.concat(all_data)
            logger.info(f"Combined data: {combined_df.shape[0]} total records")
            return combined_df
        else:
            logger.warning("No options data downloaded")
            return pl.DataFrame()
    
    def create_options_summary(self, contracts_df: pl.DataFrame, underlying: str) -> Dict:
        """Create summary of options chain"""
        if contracts_df.is_empty():
            return {}
        
        summary = {
            'underlying': underlying,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'total_contracts': contracts_df.shape[0],
            'current_stock_price': contracts_df.select('current_stock_price').head(1).item(),
        }
        
        # Breakdown by type and moneyness
        breakdown = contracts_df.group_by(['contract_type', 'moneyness']).agg([
            pl.count().alias('count')
        ])
        
        summary['breakdown'] = breakdown.to_dicts()
        
        # DTE distribution
        dte_stats = contracts_df.select([
            pl.col('trading_dte').min().alias('min_dte'),
            pl.col('trading_dte').max().alias('max_dte'),
            pl.col('trading_dte').mean().alias('avg_dte')
        ]).to_dicts()[0]
        
        summary['dte_stats'] = dte_stats
        
        # Strike range
        strike_stats = contracts_df.select([
            pl.col('strike_price').min().alias('min_strike'),
            pl.col('strike_price').max().alias('max_strike')
        ]).to_dicts()[0]
        
        summary['strike_range'] = strike_stats
        
        return summary


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fixed Options Data Downloader")
    
    parser.add_argument('--symbols', type=str, default='AAPL', help='Comma-separated symbols')
    parser.add_argument('--days', type=int, default=5, help='Days of options data to download')
    parser.add_argument('--max-dte', type=int, default=60, help='Maximum days to expiry')
    
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    # Create output directories
    Path('data/raw/options/chains').mkdir(parents=True, exist_ok=True)
    Path('data/raw/options/data').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Fixed Athena Options Downloader")
    logger.info("=" * 50)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Max DTE: {args.max_dte}")
    
    try:
        manager = FixedOptionsManager()
        
        for symbol in symbols:
            logger.info(f"\n📈 Processing {symbol}...")
            
            try:
                # Get options contracts
                contracts_df = manager.get_options_contracts(symbol, args.max_dte)
                
                if contracts_df.is_empty():
                    logger.warning(f"No contracts found for {symbol}")
                    continue
                
                # Enhance with moneyness
                contracts_df = manager.enhance_contracts_with_moneyness(contracts_df, symbol)
                
                # Create summary
                summary = manager.create_options_summary(contracts_df, symbol)
                logger.info(f"📊 {symbol} Summary:")
                logger.info(f"   Total contracts: {summary['total_contracts']}")
                logger.info(f"   Current price: ${summary['current_stock_price']:.2f}")
                logger.info(f"   DTE range: {summary['dte_stats']['min_dte']}-{summary['dte_stats']['max_dte']} days")
                
                # Save contracts chain
                chain_file = Path(f"data/raw/options/chains/{symbol}_chain_{datetime.now().strftime('%Y%m%d')}.parquet")
                contracts_df.write_parquet(chain_file)
                logger.info(f"💾 Saved contracts: {chain_file}")
                
                # Download options data
                options_data = manager.download_options_data(contracts_df, start_date, end_date)
                
                if not options_data.is_empty():
                    data_file = Path(f"data/raw/options/data/{symbol}_options_{start_date}_{end_date}.parquet")
                    options_data.write_parquet(data_file)
                    logger.info(f"💾 Saved options data: {data_file}")
                    
                    # Show sample
                    logger.info(f"📊 Sample options data:")
                    sample = options_data.head(3).select([
                        'timestamp_utc', 'contract_ticker', 'contract_type', 'strike_price', 
                        'close', 'volume', 'moneyness'
                    ])
                    print(sample)
                else:
                    logger.warning(f"No options data downloaded for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                continue
        
        logger.info("\n" + "=" * 50)
        logger.info("✅ Fixed options download completed!")
        logger.info("\nNext steps:")
        logger.info("1. Check data/raw/options/ for downloaded data")
        logger.info("2. Build gamma exposure analysis")
        logger.info("3. Create options flow features")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()