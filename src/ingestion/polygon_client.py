# src/ingestion/polygon_client.py
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
import json
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Rate limiting configuration for Polygon.io API"""
    requests_per_minute: int = 100  # Adjust based on your Polygon plan
    concurrent_requests: int = 10   # Max concurrent requests
    retry_attempts: int = 3
    backoff_factor: float = 1.5

class PolygonClient:
    """
    Scalable Polygon.io client designed to handle 5000+ tickers efficiently
    with proper rate limiting, error handling, and batch processing
    """
    
    def __init__(self, api_key: str, rate_limit_config: RateLimitConfig = None):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.rate_limit = rate_limit_config or RateLimitConfig()
        self.session = None
        self._request_times = []
        self._semaphore = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=50,  # Total connection limit
            limit_per_host=20  # Per-host connection limit
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Athena-Trading-System/1.0'}
        )
        
        # Semaphore for controlling concurrent requests
        self._semaphore = asyncio.Semaphore(self.rate_limit.concurrent_requests)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit_check(self):
        """Check and enforce rate limiting"""
        now = time.time()
        
        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        # If we're at the rate limit, wait
        if len(self._request_times) >= self.rate_limit.requests_per_minute:
            sleep_time = 61 - (now - self._request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        self._request_times.append(now)
    
    async def _make_request(self, url: str, params: Dict) -> Optional[Dict]:
        """Make a rate-limited API request with retries"""
        params['apikey'] = self.api_key
        
        async with self._semaphore:
            await self._rate_limit_check()
            
            for attempt in range(self.rate_limit.retry_attempts):
                try:
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:  # Rate limited
                            wait_time = self.rate_limit.backoff_factor ** attempt
                            logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"API error {response.status} for {url}")
                            return None
                            
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
                    await asyncio.sleep(self.rate_limit.backoff_factor ** attempt)
                except Exception as e:
                    logger.error(f"Request error for {url}: {e}")
                    await asyncio.sleep(self.rate_limit.backoff_factor ** attempt)
            
            logger.error(f"Failed to fetch {url} after {self.rate_limit.retry_attempts} attempts")
            return None
    
    async def get_stock_bars(self, symbol: str, from_date: str, to_date: str, 
                           timespan: str = "minute", multiplier: int = 1) -> List[Dict]:
        """
        Get stock price bars for a symbol
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            timespan: minute, hour, day, week, month, quarter, year
            multiplier: Size of the timespan (e.g., 5 for 5-minute bars)
        """
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000
        }
        
        data = await self._make_request(url, params)
        if data and data.get('status') == 'OK':
            return data.get('results', [])
        return []
    
    async def get_options_contracts(self, underlying: str, 
                                  contract_type: str = None,
                                  expiration_date: str = None,
                                  strike_price: float = None) -> List[Dict]:
        """
        Get options contracts for an underlying symbol
        """
        url = f"{self.base_url}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "limit": 1000
        }
        
        if contract_type:
            params["contract_type"] = contract_type
        if expiration_date:
            params["expiration_date"] = expiration_date
        if strike_price:
            params["strike_price"] = strike_price
            
        data = await self._make_request(url, params)
        if data and data.get('status') == 'OK':
            return data.get('results', [])
        return []
    
    async def get_options_quotes(self, options_ticker: str, date: str) -> List[Dict]:
        """Get quotes for a specific options contract"""
        url = f"{self.base_url}/v3/quotes/{options_ticker}"
        params = {
            "timestamp.gte": f"{date}T00:00:00Z",
            "timestamp.lt": f"{date}T23:59:59Z",
            "limit": 50000
        }
        
        data = await self._make_request(url, params)
        if data and data.get('status') == 'OK':
            return data.get('results', [])
        return []


class BatchProcessor:
    """
    Process large batches of tickers efficiently
    Designed to handle 5000+ symbols with proper resource management
    """
    
    def __init__(self, batch_size: int = 50, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        
    async def process_symbols_batch(self, 
                                  symbols: List[str], 
                                  process_func: Callable,
                                  *args, **kwargs) -> Dict[str, any]:
        """
        Process a large list of symbols in batches
        """
        results = {}
        total_symbols = len(symbols)
        
        logger.info(f"Processing {total_symbols} symbols in batches of {self.batch_size}")
        
        # Split symbols into batches
        batches = [symbols[i:i + self.batch_size] 
                  for i in range(0, len(symbols), self.batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} symbols)")
            
            # Process batch concurrently
            tasks = [process_func(symbol, *args, **kwargs) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for symbol, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing {symbol}: {result}")
                    results[symbol] = None
                else:
                    results[symbol] = result
            
            # Brief pause between batches to be respectful to the API
            if batch_idx < len(batches) - 1:
                await asyncio.sleep(1)
        
        success_count = sum(1 for r in results.values() if r is not None)
        logger.info(f"Batch processing complete: {success_count}/{total_symbols} successful")
        
        return results


# scripts/ingest_market_data.py
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.polygon_client import PolygonClient, RateLimitConfig, BatchProcessor
from config.settings import trading_config, db_config, LOGS_DIR

# Setup logging
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'data_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketDataIngester:
    """
    Main data ingestion orchestrator
    Coordinates the fetching and storage of market data for all symbols
    """
    
    def __init__(self, polygon_api_key: str, test_mode: bool = True):
        self.polygon_api_key = polygon_api_key
        self.test_mode = test_mode
        
        # For testing, use a subset of symbols
        if test_mode:
            self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]
            logger.info(f"TEST MODE: Using {len(self.symbols)} symbols")
        else:
            # In production, this would load from a file or database
            self.symbols = trading_config.stock_symbols
            logger.info(f"PRODUCTION MODE: Using {len(self.symbols)} symbols")
        
        # Configure rate limiting for Advanced tier subscriptions
        # Advanced tier: Much higher limits, can be more aggressive
        self.rate_limit_config = RateLimitConfig(
            requests_per_minute=1000,  # Advanced tier limit
            concurrent_requests=25,    # Higher concurrency for faster processing
            retry_attempts=3
        )
        
        self.batch_processor = BatchProcessor(batch_size=50, max_workers=8)  # Larger batches for advanced tier
    
    async def fetch_stock_data_for_symbol(self, symbol: str, from_date: str, to_date: str) -> Dict:
        """Fetch stock data for a single symbol"""
        try:
            async with PolygonClient(self.polygon_api_key, self.rate_limit_config) as client:
                # Get minute-level data
                minute_data = await client.get_stock_bars(
                    symbol=symbol,
                    from_date=from_date,
                    to_date=to_date,
                    timespan="minute",
                    multiplier=1
                )
                
                # Get daily data for longer-term features
                daily_data = await client.get_stock_bars(
                    symbol=symbol,
                    from_date=from_date,
                    to_date=to_date,
                    timespan="day",
                    multiplier=1
                )
                
                return {
                    'symbol': symbol,
                    'minute_data': minute_data,
                    'daily_data': daily_data,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def fetch_options_data_for_symbol(self, symbol: str, date: str) -> Dict:
        """Fetch options data for a single symbol"""
        try:
            async with PolygonClient(self.polygon_api_key, self.rate_limit_config) as client:
                # Get options contracts
                contracts = await client.get_options_contracts(
                    underlying=symbol,
                    expiration_date=date
                )
                
                return {
                    'symbol': symbol,
                    'contracts': contracts,
                    'date': date,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error fetching options data for {symbol}: {e}")
            return None
    
    async def run_stock_data_ingestion(self, days_back: int = 7):
        """Run stock data ingestion for all symbols"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Starting stock data ingestion from {from_date} to {to_date}")
        
        # Process all symbols in batches
        results = await self.batch_processor.process_symbols_batch(
            symbols=self.symbols,
            process_func=self.fetch_stock_data_for_symbol,
            from_date=from_date,
            to_date=to_date
        )
        
        # Save results (Bronze layer - raw data)
        await self._save_raw_data(results, 'stock_data', from_date, to_date)
        
        return results
    
    async def run_options_data_ingestion(self, target_date: str = None):
        """Run options data ingestion for all symbols"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Starting options data ingestion for {target_date}")
        
        results = await self.batch_processor.process_symbols_batch(
            symbols=self.symbols,
            process_func=self.fetch_options_data_for_symbol,
            date=target_date
        )
        
        # Save results (Bronze layer - raw data)
        await self._save_raw_data(results, 'options_data', target_date)
        
        return results
    
    async def _save_raw_data(self, data: Dict, data_type: str, *date_parts):
        """Save raw data to Bronze layer (JSON files for now)"""
        from config.settings import DATA_DIR
        
        bronze_dir = DATA_DIR / "bronze" / data_type
        bronze_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{data_type}_{'-'.join(date_parts)}_{timestamp}.json"
        filepath = bronze_dir / filename
        
        # Filter out None results and save
        clean_data = {k: v for k, v in data.items() if v is not None}
        
        with open(filepath, 'w') as f:
            json.dump(clean_data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(clean_data)} records to {filepath}")

async def main():
    """Main execution function"""
    import os
    
    # Get API key from environment
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        logger.error("POLYGON_API_KEY environment variable not set")
        return
    
    # Create ingester
    ingester = MarketDataIngester(
        polygon_api_key=api_key,
        test_mode=True  # Set to False for production
    )
    
    try:
        # Run stock data ingestion
        logger.info("=== Starting Stock Data Ingestion ===")
        stock_results = await ingester.run_stock_data_ingestion(days_back=3)
        
        # Run options data ingestion
        logger.info("=== Starting Options Data Ingestion ===")
        options_results = await ingester.run_options_data_ingestion()
        
        logger.info("=== Data Ingestion Complete ===")
        logger.info(f"Stock data: {len([r for r in stock_results.values() if r])}/{len(stock_results)} successful")
        logger.info(f"Options data: {len([r for r in options_results.values() if r])}/{len(options_results)} successful")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())