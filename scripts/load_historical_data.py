# scripts/load_historical_data.py
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
import os
from typing import List, Dict, Tuple
import yfinance as yf  # For getting listing dates
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.polygon_client import PolygonClient, RateLimitConfig, BatchProcessor
from config.settings import trading_config, DATA_DIR, LOGS_DIR

logger = logging.getLogger(__name__)

class HistoricalDataLoader:
    """
    Load 5 years of historical data (or since listing) for all stocks
    Optimized for Advanced Polygon tier subscriptions
    """
    
    def __init__(self, polygon_api_key: str, test_mode: bool = True):
        self.polygon_api_key = polygon_api_key
        self.test_mode = test_mode
        
        # Extended test symbols for comprehensive testing
        if test_mode:
            self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ", "NVDA", "AMZN"]
            logger.info(f"TEST MODE: Loading historical data for {len(self.symbols)} symbols")
        else:
            # In production, load from your full symbol list
            self.symbols = self._load_full_symbol_list()
            logger.info(f"PRODUCTION MODE: Loading historical data for {len(self.symbols)} symbols")
        
        # Advanced tier configuration
        self.rate_limit_config = RateLimitConfig(
            requests_per_minute=1000,
            concurrent_requests=30,  # Higher for historical batch loading
            retry_attempts=3
        )
        
        self.batch_processor = BatchProcessor(batch_size=75, max_workers=10)
    
    def _load_full_symbol_list(self) -> List[str]:
        """Load full symbol list for production use"""
        # This would typically load from a file or database
        # For now, use Russell 3000 or S&P 500 + additional symbols
        symbol_file = Path("config/symbols.txt")
        if symbol_file.exists():
            with open(symbol_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        else:
            logger.warning("No symbol file found, using default list")
            return trading_config.stock_symbols
    
    def _get_listing_date(self, symbol: str) -> datetime:
        """Get the listing date for a symbol using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="max", interval="1d")
            if not hist.empty:
                listing_date = hist.index[0].to_pydatetime().replace(tzinfo=None)
                logger.info(f"{symbol} listing date: {listing_date.strftime('%Y-%m-%d')}")
                return listing_date
            else:
                # Fallback to 5 years ago if no data
                return datetime.now() - timedelta(days=5*365)
        except Exception as e:
            logger.warning(f"Could not get listing date for {symbol}: {e}")
            return datetime.now() - timedelta(days=5*365)
    
    def _calculate_date_ranges(self, symbol: str) -> List[Tuple[str, str]]:
        """
        Calculate optimal date ranges for historical data collection
        Polygon has limits on data per request, so we split into chunks
        """
        end_date = datetime.now()
        
        # Get listing date or 5 years, whichever is more recent
        five_years_ago = end_date - timedelta(days=5*365)
        listing_date = self._get_listing_date(symbol)
        start_date = max(listing_date, five_years_ago)
        
        logger.info(f"{symbol}: Loading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Split into 6-month chunks to avoid API limits
        date_ranges = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=180), end_date)
            date_ranges.append((
                current_start.strftime('%Y-%m-%d'),
                current_end.strftime('%Y-%m-%d')
            ))
            current_start = current_end + timedelta(days=1)
        
        logger.info(f"{symbol}: Split into {len(date_ranges)} date ranges")
        return date_ranges
    
    async def load_historical_stock_data(self, symbol: str) -> Dict:
        """Load complete historical stock data for a symbol"""
        try:
            date_ranges = self._calculate_date_ranges(symbol)
            all_minute_data = []
            all_daily_data = []
            
            async with PolygonClient(self.polygon_api_key, self.rate_limit_config) as client:
                for from_date, to_date in date_ranges:
                    logger.info(f"Loading {symbol} data: {from_date} to {to_date}")
                    
                    # Get minute data
                    minute_data = await client.get_stock_bars(
                        symbol=symbol,
                        from_date=from_date,
                        to_date=to_date,
                        timespan="minute",
                        multiplier=1
                    )
                    all_minute_data.extend(minute_data)
                    
                    # Get daily data
                    daily_data = await client.get_stock_bars(
                        symbol=symbol,
                        from_date=from_date,
                        to_date=to_date,
                        timespan="day",
                        multiplier=1
                    )
                    all_daily_data.extend(daily_data)
                    
                    # Small delay between ranges to be respectful
                    await asyncio.sleep(0.5)
            
            logger.info(f"{symbol}: Loaded {len(all_minute_data)} minute bars, {len(all_daily_data)} daily bars")
            
            return {
                'symbol': symbol,
                'minute_data': all_minute_data,
                'daily_data': all_daily_data,
                'date_ranges': date_ranges,
                'total_records': len(all_minute_data) + len(all_daily_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return None
    
    async def load_historical_options_data(self, symbol: str, days_back: int = 30) -> Dict:
        """
        Load historical options data for a symbol
        Note: Historical options data is more limited and expensive
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            all_contracts = []
            all_quotes = []
            
            async with PolygonClient(self.polygon_api_key, self.rate_limit_config) as client:
                # Get options contracts for upcoming Fridays (typical expiration dates)
                current_date = end_date
                
                # Look ahead for the next 8 weeks of Friday expirations
                for weeks_ahead in range(8):
                    # Calculate next Friday
                    days_ahead = (4 - current_date.weekday()) % 7  # 4 = Friday
                    if days_ahead == 0 and current_date.weekday() == 4:  # If today is Friday
                        days_ahead = 7  # Get next Friday
                    
                    target_date = current_date + timedelta(days=days_ahead + (weeks_ahead * 7))
                    date_str = target_date.strftime('%Y-%m-%d')
                    
                    logger.info(f"Checking {symbol} options for expiration: {date_str}")
                    
                    # Get contracts for this expiration date
                    contracts = await client.get_options_contracts(
                        underlying=symbol,
                        expiration_date=date_str
                    )
                    
                    if contracts:
                        logger.info(f"Found {len(contracts)} contracts for {symbol} exp {date_str}")
                        all_contracts.extend(contracts)
                    
                    await asyncio.sleep(0.2)  # Small delay between requests
            
            logger.info(f"{symbol}: Loaded {len(all_contracts)} options contracts")
            
            return {
                'symbol': symbol,
                'contracts': all_contracts,
                'quotes': all_quotes,
                'period': f"Next 8 weeks from {end_date.strftime('%Y-%m-%d')}",
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error loading options data for {symbol}: {e}")
            return None
    
    async def run_full_historical_load(self):
        """Run complete historical data loading for all symbols"""
        logger.info("=== Starting Full Historical Data Load ===")
        logger.info(f"Symbols to process: {len(self.symbols)}")
        
        # Load stock data
        logger.info("Loading historical stock data...")
        stock_results = await self.batch_processor.process_symbols_batch(
            symbols=self.symbols,
            process_func=self.load_historical_stock_data
        )
        
        # Save stock data
        await self._save_historical_data(stock_results, 'historical_stock_data')
        
        # Load options data (smaller timeframe due to data volume/cost)
        logger.info("Loading recent options data...")
        options_results = await self.batch_processor.process_symbols_batch(
            symbols=self.symbols,
            process_func=self.load_historical_options_data,
            days_back=30  # Last 30 days of options data
        )
        
        # Save options data
        await self._save_historical_data(options_results, 'historical_options_data')
        
        # Generate summary report
        self._generate_load_report(stock_results, options_results)
        
        return stock_results, options_results
    
    async def _save_historical_data(self, data: Dict, data_type: str):
        """Save historical data with compression for large datasets"""
        bronze_dir = DATA_DIR / "bronze" / data_type
        bronze_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save each symbol's data separately for easier processing
        for symbol, symbol_data in data.items():
            if symbol_data is not None:
                filename = f"{symbol}_{data_type}_{timestamp}.json"
                filepath = bronze_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(symbol_data, f, indent=2, default=str)
                
                logger.info(f"Saved {symbol} data to {filepath}")
    
    def _generate_load_report(self, stock_results: Dict, options_results: Dict):
        """Generate a summary report of the data loading process"""
        stock_success = len([r for r in stock_results.values() if r is not None])
        options_success = len([r for r in options_results.values() if r is not None])
        
        total_minute_bars = sum(
            len(r.get('minute_data', [])) for r in stock_results.values() if r
        )
        total_daily_bars = sum(
            len(r.get('daily_data', [])) for r in stock_results.values() if r
        )
        total_options_contracts = sum(
            len(r.get('contracts', [])) for r in options_results.values() if r
        )
        
        report = {
            'load_timestamp': datetime.now().isoformat(),
            'symbols_processed': len(self.symbols),
            'stock_data': {
                'successful_symbols': stock_success,
                'failed_symbols': len(stock_results) - stock_success,
                'total_minute_bars': total_minute_bars,
                'total_daily_bars': total_daily_bars
            },
            'options_data': {
                'successful_symbols': options_success,
                'failed_symbols': len(options_results) - options_success,
                'total_contracts': total_options_contracts
            }
        }
        
        # Save report
        report_file = DATA_DIR / "bronze" / f"load_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log summary
        logger.info("=== Historical Data Load Complete ===")
        logger.info(f"Stock data: {stock_success}/{len(self.symbols)} symbols successful")
        logger.info(f"Total minute bars loaded: {total_minute_bars:,}")
        logger.info(f"Total daily bars loaded: {total_daily_bars:,}")
        logger.info(f"Options data: {options_success}/{len(self.symbols)} symbols successful")
        logger.info(f"Total options contracts: {total_options_contracts:,}")
        logger.info(f"Report saved to: {report_file}")


# scripts/market_hours_scheduler.py
import asyncio
import schedule
import time
from datetime import datetime, time as dt_time
import pytz
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class MarketHoursScheduler:
    """
    Schedule data collection during market hours
    Run every 10-15 minutes when market is open
    """
    
    def __init__(self, polygon_api_key: str):
        self.polygon_api_key = polygon_api_key
        self.eastern = pytz.timezone('US/Eastern')
        self.is_running = False
        
        # Market hours (Eastern Time)
        self.market_open = dt_time(9, 30)  # 9:30 AM
        self.market_close = dt_time(16, 0)  # 4:00 PM
        
        # Weekend days (Saturday=5, Sunday=6)
        self.weekend_days = [5, 6]
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open"""
        now_eastern = datetime.now(self.eastern)
        current_time = now_eastern.time()
        current_weekday = now_eastern.weekday()
        
        # Check if it's a weekend
        if current_weekday in self.weekend_days:
            return False
        
        # Check if it's within market hours
        return self.market_open <= current_time <= self.market_close
    
    async def run_market_data_collection(self):
        """Run the market data collection if market is open"""
        if not self.is_market_open():
            logger.info("Market is closed, skipping data collection")
            return
        
        logger.info("Market is open, starting data collection...")
        
        try:
            from scripts.ingest_market_data import MarketDataIngester
            
            # Run data collection
            ingester = MarketDataIngester(
                polygon_api_key=self.polygon_api_key,
                test_mode=True  # Set to False for production
            )
            
            # Collect recent data (last hour)
            stock_results = await ingester.run_stock_data_ingestion(days_back=1)
            options_results = await ingester.run_options_data_ingestion()
            
            logger.info("Market data collection completed successfully")
            
        except Exception as e:
            logger.error(f"Error during market data collection: {e}")
    
    def schedule_market_hours_collection(self):
        """Set up the market hours collection schedule"""
        # Schedule every 15 minutes during potential market hours
        schedule.every(15).minutes.do(
            lambda: asyncio.run(self.run_market_data_collection())
        )
        
        logger.info("Market hours data collection scheduled (every 15 minutes)")
        logger.info(f"Market hours: {self.market_open} - {self.market_close} Eastern Time")
    
    def run_scheduler(self):
        """Run the scheduler loop"""
        self.is_running = True
        logger.info("Starting market hours scheduler...")
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.is_running = False
        logger.info("Market hours scheduler stopped")


async def main():
    """Main function for historical data loading"""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        logger.error("POLYGON_API_KEY environment variable not set")
        return
    
    # Create historical data loader
    loader = HistoricalDataLoader(
        polygon_api_key=api_key,
        test_mode=True  # Set to False for full production load
    )
    
    try:
        # Run full historical data load
        stock_results, options_results = await loader.run_full_historical_load()
        
        logger.info("Historical data loading complete!")
        logger.info("Ready to start live market data collection")
        
    except Exception as e:
        logger.error(f"Historical data loading failed: {e}")
        raise


def start_live_collection():
    """Start the live market data collection scheduler"""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        logger.error("POLYGON_API_KEY environment variable not set")
        return
    
    scheduler = MarketHoursScheduler(api_key)
    scheduler.schedule_market_hours_collection()
    scheduler.run_scheduler()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "live":
        # Run live collection
        start_live_collection()
    else:
        # Run historical data load
        asyncio.run(main())