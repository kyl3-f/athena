#!/usr/bin/env python3
"""
Athena Historical Data Downloader
Downloads historical stock and options data from Polygon.io for model training
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Now import our modules
from config.polygon_config import PolygonConfig
from ingestion.polygon_client import PolygonClient, DataType
from config.settings import trading_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/download_historical.log')
    ]
)
logger = logging.getLogger(__name__)


def download_stock_data(
    client: PolygonClient,
    symbols: List[str],
    start_date: str,
    end_date: str,
    timeframes: List[str] = None
) -> None:
    """Download historical stock data for multiple timeframes"""
    
    if timeframes is None:
        timeframes = ['minute', 'hour', 'day']
    
    logger.info(f"Downloading stock data for {len(symbols)} symbols")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Timeframes: {timeframes}")
    
    for symbol in symbols:
        logger.info(f"\n📊 Processing {symbol}...")
        
        for timeframe in timeframes:
            try:
                logger.info(f"  Downloading {timeframe} data...")
                
                # Download aggregates
                df = client.get_stock_aggregates(
                    symbol=symbol,
                    multiplier=1,
                    timespan=timeframe,
                    from_date=start_date,
                    to_date=end_date,
                    limit=50000
                )
                
                if not df.is_empty():
                    # Save to parquet file
                    output_dir = Path(f"data/raw/stocks/aggregates/{timeframe}")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.parquet"
                    file_path = output_dir / filename
                    
                    client.save_to_parquet(df, file_path)
                    logger.info(f"    ✅ Saved {df.shape[0]} {timeframe} bars to {file_path}")
                else:
                    logger.warning(f"    ⚠️  No {timeframe} data for {symbol}")
                    
            except Exception as e:
                logger.error(f"    ❌ Failed to download {timeframe} data for {symbol}: {e}")


def download_options_data(
    client: PolygonClient,
    symbols: List[str],
    start_date: str,
    end_date: str,
    timeframes: List[str] = None
) -> None:
    """Download historical options data"""
    
    if timeframes is None:
        timeframes = ['minute', 'hour', 'day']
    
    logger.info(f"\n🎯 Downloading options data for {len(symbols)} symbols")
    
    for symbol in symbols:
        logger.info(f"\n📈 Processing options for {symbol}...")
        
        try:
            # First, get available options contracts
            logger.info(f"  Finding options contracts...")
            
            contracts_df = client.get_options_contracts(
                underlying_ticker=symbol,
                limit=100  # Limit to avoid too many API calls
            )
            
            if contracts_df.is_empty():
                logger.warning(f"    ⚠️  No options contracts found for {symbol}")
                continue
            
            logger.info(f"    Found {contracts_df.shape[0]} options contracts")
            
            # Sample a few contracts to download (to avoid hitting rate limits)
            # Focus on ATM calls and puts with near-term expirations
            sample_contracts = contracts_df.head(10)  # Just get first 10 for now
            
            for timeframe in timeframes:
                logger.info(f"  Downloading {timeframe} options data...")
                
                for row in sample_contracts.iter_rows(named=True):
                    contract_ticker = row.get('ticker', '')
                    
                    if not contract_ticker:
                        continue
                    
                    try:
                        # Download options aggregates
                        options_df = client.get_options_aggregates(
                            options_ticker=contract_ticker,
                            multiplier=1,
                            timespan=timeframe,
                            from_date=start_date,
                            to_date=end_date,
                            limit=10000
                        )
                        
                        if not options_df.is_empty():
                            # Save to parquet file
                            output_dir = Path(f"data/raw/options/aggregates/{timeframe}")
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Clean filename
                            safe_ticker = contract_ticker.replace(':', '_')
                            filename = f"{safe_ticker}_{timeframe}_{start_date}_{end_date}.parquet"
                            file_path = output_dir / filename
                            
                            client.save_to_parquet(options_df, file_path)
                            logger.info(f"      ✅ Saved {options_df.shape[0]} bars for {contract_ticker}")
                        
                    except Exception as e:
                        logger.warning(f"      ⚠️  Failed to download {contract_ticker}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"  ❌ Failed to process options for {symbol}: {e}")


def get_date_range(days_back: int) -> tuple:
    """Calculate start and end dates based on days back"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def main():
    """Main download function"""
    parser = argparse.ArgumentParser(description="Download historical market data for Athena")
    
    parser.add_argument(
        '--symbols',
        type=str,
        default=','.join(trading_config.stock_symbols),
        help=f'Comma-separated list of symbols (default: {",".join(trading_config.stock_symbols)})'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days of historical data to download (default: 30)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD) - overrides --days'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD) - overrides --days'
    )
    
    parser.add_argument(
        '--timeframes',
        type=str,
        default='minute,hour,day',
        help='Comma-separated timeframes to download (default: minute,hour,day)'
    )
    
    parser.add_argument(
        '--stocks-only',
        action='store_true',
        help='Download only stock data (skip options)'
    )
    
    parser.add_argument(
        '--options-only',
        action='store_true',
        help='Download only options data (skip stocks)'
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    timeframes = [t.strip() for t in args.timeframes.split(',')]
    
    # Calculate date range
    if args.start_date and args.end_date:
        start_date, end_date = args.start_date, args.end_date
    else:
        start_date, end_date = get_date_range(args.days)
    
    # Create output directories
    Path('data/raw/stocks/aggregates').mkdir(parents=True, exist_ok=True)
    Path('data/raw/options/aggregates').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Athena Historical Data Downloader")
    logger.info("=" * 50)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Stocks only: {args.stocks_only}")
    logger.info(f"Options only: {args.options_only}")
    
    try:
        # Initialize Polygon client
        config = PolygonConfig()
        
        with PolygonClient(config) as client:
            logger.info("✅ Connected to Polygon API")
            
            # Download stock data
            if not args.options_only:
                download_stock_data(client, symbols, start_date, end_date, timeframes)
            
            # Download options data
            if not args.stocks_only:
                logger.info("\n" + "="*30)
                download_options_data(client, symbols, start_date, end_date, timeframes)
            
            logger.info("\n" + "="*50)
            logger.info("🎉 Download completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Check your data in the data/raw/ directory")
            logger.info("2. Connect DBeaver to view data in SQLite")
            logger.info("3. Run feature engineering pipeline")
            logger.info("4. Train your first trading model")
            
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()