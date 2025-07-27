# scripts/collect_live_data.py
import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, time as dt_time
import pytz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.polygon_client import PolygonClient, RateLimitConfig, BatchProcessor
from config.settings import trading_config, DATA_DIR, LOGS_DIR

logger = logging.getLogger(__name__)

class LiveMarketDataCollector:
    """
    Dedicated live market data collection for production use
    Handles 5000+ tickers with proper rate limiting
    """
    
    def __init__(self, polygon_api_key: str, production_mode: bool = True):
        self.polygon_api_key = polygon_api_key
        self.production_mode = production_mode
        self.eastern = pytz.timezone('US/Eastern')
        
        # Load symbol list based on mode
        if production_mode:
            self.symbols = self._load_production_symbols()
            logger.info(f"PRODUCTION MODE: Loaded {len(self.symbols)} symbols")
        else:
            self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ", "NVDA", "AMZN"]
            logger.info(f"TEST MODE: Using {len(self.symbols)} test symbols")
        
        # Advanced tier configuration for 5000+ symbols
        self.rate_limit_config = RateLimitConfig(
            requests_per_minute=1000,  # Advanced tier limit
            concurrent_requests=50,    # Higher for live collection
            retry_attempts=3,
            backoff_factor=1.5
        )
        
        # Larger batches for production
        batch_size = 100 if production_mode else 25
        max_workers = 15 if production_mode else 8
        
        self.batch_processor = BatchProcessor(
            batch_size=batch_size, 
            max_workers=max_workers
        )
    
    def _load_production_symbols(self) -> list:
        """Load production symbol list (5000+ tickers)"""
        # Check multiple sources for symbols
        symbol_sources = [
            Path("config/symbols.txt"),
            Path("config/russell_3000.txt"),
            Path("config/sp500.txt"),
            Path("data/symbols/all_symbols.txt")
        ]
        
        symbols = set()
        
        for source in symbol_sources:
            if source.exists():
                logger.info(f"Loading symbols from {source}")
                try:
                    with open(source, 'r') as f:
                        file_symbols = [line.strip().upper() for line in f if line.strip()]
                        symbols.update(file_symbols)
                        logger.info(f"Added {len(file_symbols)} symbols from {source}")
                except Exception as e:
                    logger.warning(f"Error reading {source}: {e}")
        
        if not symbols:
            logger.warning("No symbol files found, using default large cap list")
            symbols = self._get_default_symbol_list()
        
        symbol_list = sorted(list(symbols))
        logger.info(f"Total unique symbols loaded: {len(symbol_list)}")
        
        return symbol_list
    
    def _get_default_symbol_list(self) -> set:
        """Default symbol list if no files found (major indices + large caps)"""
        # Major ETFs and large cap stocks
        default_symbols = {
            # Major ETFs
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VEA', 'VWO', 'AGG', 'LQD', 'HYG',
            'GLD', 'SLV', 'USO', 'UNG', 'TLT', 'SHY', 'XLF', 'XLK', 'XLE', 'XLV',
            'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE', 'XLC',
            
            # FAANG + Major Tech
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA',
            'NFLX', 'CRM', 'ORCL', 'ADBE', 'INTC', 'AMD', 'QCOM', 'AVGO',
            
            # Major Financial
            'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'AXP', 'V', 'MA',
            'PYPL', 'SQ', 'COF', 'USB', 'PNC', 'TFC', 'SCHW',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'DHR', 'ABT', 'LLY',
            'BMY', 'AMGN', 'GILD', 'CVS', 'CI', 'ANTM', 'HUM',
            
            # Consumer & Retail
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'SBUX', 'NKE', 'DIS',
            'AMGN', 'COST', 'TGT', 'LOW', 'BKNG', 'EBAY',
            
            # Industrial & Energy
            'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX',
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC',
            
            # Communication & Media
            'T', 'VZ', 'TMUS', 'CHTR', 'CMCSA', 'NFLX', 'DIS', 'ROKU'
        }
        
        # Add more symbols to reach ~1000 for testing
        # In production, you'd load Russell 3000 or similar
        additional_symbols = {
            f'SYMBOL_{i:04d}' for i in range(len(default_symbols), 1000)
        }
        
        logger.info(f"Using default symbol list with {len(default_symbols)} real symbols")
        return default_symbols
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now_et = datetime.now(self.eastern)
        current_time = now_et.time()
        current_weekday = now_et.weekday()
        
        # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        
        is_weekday = current_weekday < 5  # Monday=0, Friday=4
        is_market_hours = market_open <= current_time <= market_close
        
        return is_weekday and is_market_hours
    
    async def collect_stock_data_for_symbol(self, symbol: str) -> dict:
        """Collect recent stock data for a single symbol"""
        try:
            async with PolygonClient(self.polygon_api_key, self.rate_limit_config) as client:
                # Get last 2 hours of minute data
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d')
                
                minute_data = await client.get_stock_bars(
                    symbol=symbol,
                    from_date=start_date,
                    to_date=end_date,
                    timespan="minute",
                    multiplier=1
                )
                
                return {
                    'symbol': symbol,
                    'minute_data': minute_data,
                    'timestamp': datetime.now().isoformat(),
                    'data_count': len(minute_data)
                }
                
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {e}")
            return None
    
    async def collect_options_data_for_symbol(self, symbol: str) -> dict:
        """Collect current options data for a single symbol"""
        try:
            async with PolygonClient(self.polygon_api_key, self.rate_limit_config) as client:
                # Get options contracts expiring in next 2 weeks
                contracts = await client.get_options_contracts(
                    underlying=symbol
                )
                
                return {
                    'symbol': symbol,
                    'contracts': contracts,
                    'timestamp': datetime.now().isoformat(),
                    'contract_count': len(contracts)
                }
                
        except Exception as e:
            logger.error(f"Error collecting options for {symbol}: {e}")
            return None
    
    async def run_live_collection(self):
        """Run live data collection for all symbols"""
        if not self.is_market_open():
            logger.info("Market is closed - skipping data collection")
            return {'skipped': True, 'reason': 'market_closed'}
        
        logger.info(f"🚀 Starting live data collection for {len(self.symbols)} symbols")
        
        # Collect stock data
        logger.info("Collecting stock data...")
        stock_results = await self.batch_processor.process_symbols_batch(
            symbols=self.symbols,
            process_func=self.collect_stock_data_for_symbol
        )
        
        # Collect options data (for subset of symbols to manage API usage)
        # Options data is more expensive, so collect for top symbols only
        priority_symbols = self.symbols[:100] if self.production_mode else self.symbols[:5]
        
        logger.info(f"Collecting options data for {len(priority_symbols)} priority symbols...")
        options_results = await self.batch_processor.process_symbols_batch(
            symbols=priority_symbols,
            process_func=self.collect_options_data_for_symbol
        )
        
        # Save to Bronze layer
        await self._save_live_data(stock_results, 'live_stock_data')
        await self._save_live_data(options_results, 'live_options_data')
        
        # Generate collection summary
        summary = self._generate_collection_summary(stock_results, options_results)
        
        logger.info("✅ Live data collection complete")
        return summary
    
    async def _save_live_data(self, data: dict, data_type: str):
        """Save live data to Bronze layer"""
        bronze_dir = DATA_DIR / "bronze" / data_type
        bronze_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save successful results only
        successful_data = {k: v for k, v in data.items() if v is not None}
        
        if successful_data:
            filename = f"{data_type}_{timestamp}.json"
            filepath = bronze_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(successful_data, f, indent=2, default=str)
            
            logger.info(f"💾 Saved {len(successful_data)} {data_type} records to {filepath}")
    
    def _generate_collection_summary(self, stock_results: dict, options_results: dict) -> dict:
        """Generate collection summary"""
        stock_success = len([r for r in stock_results.values() if r is not None])
        options_success = len([r for r in options_results.values() if r is not None])
        
        total_minute_bars = sum(
            r.get('data_count', 0) for r in stock_results.values() if r
        )
        total_options_contracts = sum(
            r.get('contract_count', 0) for r in options_results.values() if r
        )
        
        return {
            'collection_timestamp': datetime.now().isoformat(),
            'market_open': self.is_market_open(),
            'symbols_attempted': len(self.symbols),
            'stock_data': {
                'successful_symbols': stock_success,
                'failed_symbols': len(stock_results) - stock_success,
                'total_minute_bars': total_minute_bars,
                'success_rate_pct': round(stock_success / len(stock_results) * 100, 2)
            },
            'options_data': {
                'symbols_attempted': len(options_results),
                'successful_symbols': options_success,
                'failed_symbols': len(options_results) - options_success,
                'total_contracts': total_options_contracts,
                'success_rate_pct': round(options_success / len(options_results) * 100, 2) if options_results else 0
            }
        }


async def main():
    """Main execution function"""
    import os
    
    # Setup logging
    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / 'live_data_collection.log'),
            logging.StreamHandler()
        ]
    )
    
    # Get API key
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        logger.error("POLYGON_API_KEY environment variable not set")
        return
    
    # Create collector
    collector = LiveMarketDataCollector(
        polygon_api_key=api_key,
        production_mode=True  # Set to True for 5000+ symbols
    )
    
    try:
        # Run collection
        summary = await collector.run_live_collection()
        
        if summary.get('skipped'):
            logger.info(f"Collection skipped: {summary['reason']}")
        else:
            logger.info("=" * 50)
            logger.info("📊 COLLECTION SUMMARY")
            logger.info("=" * 50)
            
            stock_data = summary['stock_data']
            options_data = summary['options_data']
            
            logger.info(f"Stock data: {stock_data['successful_symbols']}/{summary['symbols_attempted']} symbols")
            logger.info(f"Stock success rate: {stock_data['success_rate_pct']}%")
            logger.info(f"Total minute bars: {stock_data['total_minute_bars']:,}")
            
            logger.info(f"Options data: {options_data['successful_symbols']}/{options_data['symbols_attempted']} symbols")
            logger.info(f"Options success rate: {options_data['success_rate_pct']}%")
            logger.info(f"Total contracts: {options_data['total_contracts']:,}")
        
    except Exception as e:
        logger.error(f"Live data collection failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())