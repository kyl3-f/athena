"""
Athena Trading Signal Detection System
Polygon.io API Client Wrapper

This module provides a comprehensive wrapper for Polygon.io API interactions,
handling both historical data ingestion and real-time streaming for options and stocks.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, AsyncGenerator, Any
from pathlib import Path
import logging

import requests
import websockets
import pandas as pd
import polars as pl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataType(Enum):
    """Supported data types from Polygon"""
    STOCK_TRADES = "stock_trades"
    STOCK_AGGREGATES = "stock_aggregates"
    OPTIONS_TRADES = "options_trades"
    OPTIONS_AGGREGATES = "options_aggregates"
    MARKET_STATUS = "market_status"


@dataclass
class PolygonConfig:
    """Configuration for Polygon API client"""
    api_key: str
    base_url: str = "https://api.polygon.io"
    websocket_url: str = "wss://socket.polygon.io"
    max_retries: int = 3
    backoff_factor: float = 1.0
    timeout: int = 30
    rate_limit_per_minute: int = 100  # Adjust based on your plan


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        # Remove calls outside the time window
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0]) + 1
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
        
        self.calls.append(now)


class PolygonClient:
    """
    Comprehensive Polygon.io API client for Athena trading system
    
    Features:
    - Rate limiting and retry logic
    - Both REST and WebSocket support  
    - Automatic data validation and cleaning
    - Export to multiple formats (parquet, CSV, JSON)
    - Market calendar integration
    - Error handling and logging
    """
    
    def __init__(self, config: PolygonConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)
        
        # Setup HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=config.max_retries,
            backoff_factor=config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'Authorization': f'Bearer {config.api_key}',
            'User-Agent': 'Athena-Trading-System/1.0'
        })
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a rate-limited HTTP request to Polygon API"""
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.config.base_url}{endpoint}"
        
        try:
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_market_status(self) -> Dict:
        """Get current market status"""
        return self._make_request("/v1/marketstatus/now")
    
    def get_market_calendar(self, start_date: str, end_date: str) -> List[Dict]:
        """Get market calendar for date range"""
        params = {'start': start_date, 'end': end_date}
        response = self._make_request("/v1/marketstatus/upcoming", params)
        
        # Handle different response formats
        if isinstance(response, list):
            return response
        elif isinstance(response, dict):
            return response.get('results', [])
        else:
            logger.warning(f"Unexpected calendar response format: {type(response)}")
            return []
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            status = self.get_market_status()
            return status.get('market', 'closed') == 'open'
        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False
    
    # STOCK DATA METHODS
    
    def get_stock_trades(
        self, 
        symbol: str, 
        date: str,
        timestamp_gte: Optional[str] = None,
        timestamp_lte: Optional[str] = None,
        limit: int = 50000
    ) -> pl.DataFrame:
        """
        Get stock trades for a specific symbol and date
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            date: Date in YYYY-MM-DD format
            timestamp_gte: Minimum timestamp (inclusive)
            timestamp_lte: Maximum timestamp (inclusive)  
            limit: Max number of results
        
        Returns:
            Polars DataFrame with trade data
        """
        endpoint = f"/v3/trades/{symbol}"
        params = {
            'timestamp.date': date,
            'limit': limit
        }
        
        if timestamp_gte:
            params['timestamp.gte'] = timestamp_gte
        if timestamp_lte:
            params['timestamp.lte'] = timestamp_lte
        
        response = self._make_request(endpoint, params)
        trades = response.get('results', [])
        
        if not trades:
            logger.warning(f"No trades found for {symbol} on {date}")
            return pl.DataFrame()
        
        # Convert to DataFrame with proper schema
        df = pl.DataFrame(trades)
        
        # Standardize column names and types
        if not df.is_empty():
            df = df.rename({
                't': 'timestamp_ns',
                'y': 'timestamp_utc', 
                'p': 'price',
                's': 'size',
                'x': 'exchange',
                'c': 'conditions',
                'i': 'id',
                'z': 'tape'
            })
            
            # Convert timestamp to datetime
            df = df.with_columns([
                pl.from_epoch(pl.col('timestamp_ns'), time_unit='ns').alias('timestamp_utc'),
                pl.lit(symbol).alias('symbol')
            ])
        
        return df
    
    def get_stock_aggregates(
        self,
        symbol: str,
        multiplier: int = 1,
        timespan: str = 'minute',
        from_date: str = None,
        to_date: str = None,
        limit: int = 50000
    ) -> pl.DataFrame:
        """
        Get stock aggregate bars
        
        Args:
            symbol: Stock symbol
            multiplier: Size of timespan multiplier  
            timespan: Size of time window ('minute', 'hour', 'day')
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Max results
        
        Returns:
            Polars DataFrame with aggregate data
        """
        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {'limit': limit, 'adjusted': 'true', 'sort': 'asc'}
        
        response = self._make_request(endpoint, params)
        results = response.get('results', [])
        
        if not results:
            logger.warning(f"No aggregates found for {symbol}")
            return pl.DataFrame()
        
        df = pl.DataFrame(results)
        
        if not df.is_empty():
            df = df.rename({
                't': 'timestamp_ms',
                'o': 'open',
                'h': 'high', 
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vw': 'vwap',
                'n': 'transactions'
            })
            
            # Convert timestamp
            df = df.with_columns([
                pl.from_epoch(pl.col('timestamp_ms'), time_unit='ms').alias('timestamp_utc'),
                pl.lit(symbol).alias('symbol')
            ])
        
        return df
    
    # OPTIONS DATA METHODS
    
    def get_options_contracts(
        self,
        underlying_ticker: str,
        contract_type: Optional[str] = None,
        expiration_date: Optional[str] = None,
        strike_price: Optional[float] = None,
        limit: int = 1000
    ) -> pl.DataFrame:
        """
        Get options contracts for underlying ticker
        
        Args:
            underlying_ticker: Underlying stock symbol
            contract_type: 'call' or 'put'
            expiration_date: Contract expiration date
            strike_price: Strike price
            limit: Max results
        
        Returns:
            DataFrame with contract details
        """
        endpoint = "/v3/reference/options/contracts"
        params = {
            'underlying_ticker': underlying_ticker,
            'limit': limit
        }
        
        if contract_type:
            params['contract_type'] = contract_type
        if expiration_date:
            params['expiration_date'] = expiration_date
        if strike_price:
            params['strike_price'] = strike_price
        
        response = self._make_request(endpoint, params)
        results = response.get('results', [])
        
        if not results:
            return pl.DataFrame()
        
        return pl.DataFrame(results)
    
    def get_options_trades(
        self,
        options_ticker: str,
        date: str,
        timestamp_gte: Optional[str] = None,
        timestamp_lte: Optional[str] = None,
        limit: int = 50000
    ) -> pl.DataFrame:
        """Get options trades for specific contract and date"""
        endpoint = f"/v3/trades/{options_ticker}"
        params = {
            'timestamp.date': date,
            'limit': limit
        }
        
        if timestamp_gte:
            params['timestamp.gte'] = timestamp_gte
        if timestamp_lte:
            params['timestamp.lte'] = timestamp_lte
        
        response = self._make_request(endpoint, params)
        trades = response.get('results', [])
        
        if not trades:
            return pl.DataFrame()
        
        df = pl.DataFrame(trades)
        
        if not df.is_empty():
            # Parse options ticker to extract contract details
            contract_info = self._parse_options_ticker(options_ticker)
            
            df = df.rename({
                't': 'timestamp_ns',
                'p': 'price',
                's': 'size',
                'x': 'exchange',
                'c': 'conditions'
            })
            
            df = df.with_columns([
                pl.from_epoch(pl.col('timestamp_ns'), time_unit='ns').alias('timestamp_utc'),
                pl.lit(options_ticker).alias('symbol'),
                pl.lit(contract_info['underlying']).alias('underlying_symbol'),
                pl.lit(contract_info['type']).alias('contract_type'),
                pl.lit(contract_info['strike']).alias('strike_price'),
                pl.lit(contract_info['expiration']).alias('expiration_date')
            ])
        
        return df
    
    def get_options_aggregates(
        self,
        options_ticker: str,
        multiplier: int = 1,
        timespan: str = 'minute',
        from_date: str = None,
        to_date: str = None,
        limit: int = 50000
    ) -> pl.DataFrame:
        """Get options aggregate bars"""
        endpoint = f"/v2/aggs/ticker/{options_ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {'limit': limit, 'adjusted': 'true', 'sort': 'asc'}
        
        response = self._make_request(endpoint, params)
        results = response.get('results', [])
        
        if not results:
            return pl.DataFrame()
        
        df = pl.DataFrame(results)
        
        if not df.is_empty():
            contract_info = self._parse_options_ticker(options_ticker)
            
            df = df.rename({
                't': 'timestamp_ms',
                'o': 'open',
                'h': 'high',
                'l': 'low', 
                'c': 'close',
                'v': 'volume',
                'vw': 'vwap',
                'n': 'transactions'
            })
            
            df = df.with_columns([
                pl.from_epoch(pl.col('timestamp_ms'), time_unit='ms').alias('timestamp_utc'),
                pl.lit(options_ticker).alias('symbol'),
                pl.lit(contract_info['underlying']).alias('underlying_symbol'),
                pl.lit(contract_info['type']).alias('contract_type'),
                pl.lit(contract_info['strike']).alias('strike_price'),
                pl.lit(contract_info['expiration']).alias('expiration_date')
            ])
        
        return df
    
    def _parse_options_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Parse options ticker to extract contract details
        Example: 'AAPL250117C00150000' -> {underlying: 'AAPL', expiration: '2025-01-17', type: 'call', strike: 150.0}
        """
        try:
            # Standard format: SYMBOL + YYMMDD + C/P + Strike*1000
            underlying = ''
            i = 0
            
            # Extract underlying symbol (letters at start)
            while i < len(ticker) and ticker[i].isalpha():
                underlying += ticker[i]
                i += 1
            
            # Extract date (6 digits)
            date_str = ticker[i:i+6]
            exp_date = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
            i += 6
            
            # Extract call/put
            contract_type = 'call' if ticker[i] == 'C' else 'put'
            i += 1
            
            # Extract strike price
            strike_str = ticker[i:]
            strike_price = float(strike_str) / 1000
            
            return {
                'underlying': underlying,
                'expiration': exp_date,
                'type': contract_type,
                'strike': strike_price
            }
            
        except Exception as e:
            logger.error(f"Failed to parse options ticker {ticker}: {e}")
            return {
                'underlying': ticker,
                'expiration': None,
                'type': None,
                'strike': None
            }
    
    # WEBSOCKET STREAMING METHODS
    
    async def stream_market_data(
        self,
        symbols: List[str],
        data_types: List[str] = None
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream real-time market data via WebSocket
        
        Args:
            symbols: List of symbols to subscribe to
            data_types: List of data types ('T' for trades, 'A' for aggregates)
        
        Yields:
            Real-time market data messages
        """
        if data_types is None:
            data_types = ['T', 'A']  # Trades and aggregates
        
        uri = f"{self.config.websocket_url}/stocks"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Authenticate
                auth_message = {
                    "action": "auth",
                    "params": self.config.api_key
                }
                await websocket.send(json.dumps(auth_message))
                
                # Subscribe to symbols
                for data_type in data_types:
                    subscribe_message = {
                        "action": "subscribe",
                        "params": f"{data_type}.{','.join(symbols)}"
                    }
                    await websocket.send(json.dumps(subscribe_message))
                
                # Listen for messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if isinstance(data, list):
                            for item in data:
                                yield item
                        else:
                            yield data
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode WebSocket message: {e}")
                        
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    # DATA EXPORT METHODS
    
    def save_to_parquet(
        self,
        df: pl.DataFrame,
        file_path: Union[str, Path],
        partition_cols: List[str] = None
    ) -> None:
        """Save DataFrame to parquet file with optional partitioning"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if partition_cols:
                df.write_parquet(file_path, partition_by=partition_cols)
            else:
                df.write_parquet(file_path)
            logger.info(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save parquet file: {e}")
            raise
    
    def save_to_csv(self, df: pl.DataFrame, file_path: Union[str, Path]) -> None:
        """Save DataFrame to CSV file"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df.write_csv(file_path)
            logger.info(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV file: {e}")
            raise
    
    # UTILITY METHODS
    
    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """Get list of trading days between start and end date"""
        try:
            calendar = self.get_market_calendar(start_date, end_date)
            
            if not calendar:
                logger.warning("No market calendar data returned")
                return []
            
            # Handle different calendar data structures
            trading_days = []
            for day in calendar:
                if isinstance(day, dict):
                    # Look for date field in various possible keys
                    date_value = day.get('date') or day.get('day') or day.get('trading_date')
                    status = day.get('status', 'unknown')
                    
                    if date_value and status in ['open', 'early_close']:
                        trading_days.append(date_value)
                elif isinstance(day, str):
                    # Sometimes the response might just be a list of date strings
                    trading_days.append(day)
            
            return trading_days
            
        except Exception as e:
            logger.error(f"Failed to get trading days: {e}")
            # Fallback: generate weekdays (excluding weekends)
            from datetime import datetime, timedelta
            
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            trading_days = []
            current = start
            while current <= end:
                # Skip weekends (Monday=0, Sunday=6)
                if current.weekday() < 5:  # Monday to Friday
                    trading_days.append(current.strftime('%Y-%m-%d'))
                current += timedelta(days=1)
            
            logger.info(f"Using fallback weekday calculation: {len(trading_days)} days")
            return trading_days
    
    def batch_download_historical(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        data_type: DataType,
        output_dir: Union[str, Path]
    ) -> None:
        """
        Batch download historical data for multiple symbols
        
        Args:
            symbols: List of symbols to download
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)  
            data_type: Type of data to download
            output_dir: Directory to save files
        """
        output_dir = Path(output_dir)
        
        trading_days = self.get_trading_days(start_date, end_date)
        
        for symbol in symbols:
            logger.info(f"Downloading {data_type.value} for {symbol}")
            
            for date in trading_days:
                try:
                    if data_type == DataType.STOCK_TRADES:
                        df = self.get_stock_trades(symbol, date)
                    elif data_type == DataType.STOCK_AGGREGATES:
                        df = self.get_stock_aggregates(
                            symbol, from_date=date, to_date=date
                        )
                    elif data_type == DataType.OPTIONS_TRADES:
                        df = self.get_options_trades(symbol, date)
                    elif data_type == DataType.OPTIONS_AGGREGATES:
                        df = self.get_options_aggregates(
                            symbol, from_date=date, to_date=date
                        )
                    else:
                        logger.warning(f"Unsupported data type: {data_type}")
                        continue
                    
                    if not df.is_empty():
                        filename = f"{data_type.value}_{symbol}_{date.replace('-', '')}.parquet"
                        file_path = output_dir / f"date={date}" / filename
                        self.save_to_parquet(df, file_path)
                    
                except Exception as e:
                    logger.error(f"Failed to download {symbol} for {date}: {e}")
                    continue
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()


# Example usage and testing
if __name__ == "__main__":
    # Example configuration (use environment variables in production)
    config = PolygonConfig(
        api_key="YOUR_API_KEY_HERE",  # Replace with actual API key
        rate_limit_per_minute=100
    )
    
    with PolygonClient(config) as client:
        # Test market status
        print("Market Status:", client.get_market_status())
        
        # Test stock data
        stock_df = client.get_stock_aggregates(
            "AAPL", 
            from_date="2024-01-01", 
            to_date="2024-01-01"
        )
        print(f"Stock data shape: {stock_df.shape}")
        
        # Test options contracts
        contracts_df = client.get_options_contracts("AAPL", limit=10)
        print(f"Options contracts shape: {contracts_df.shape}")