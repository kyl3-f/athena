# scripts/production_pipeline.py
import asyncio
import sys
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta, date
import pytz
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.polygon_client import PolygonClient, RateLimitConfig, BatchProcessor
from src.processing.data_cleaner import DataCleaner
from src.processing.feature_engineer import AdvancedFeatureEngineer
from config.settings import DATA_DIR, LOGS_DIR

logger = logging.getLogger(__name__)

class ProductionMarketPipeline:
    """
    Full production pipeline for automated market data collection and processing
    Features:
    - Polygon Calendar API integration
    - Stock + Options chain collection
    - Custom Greeks calculation (no Polygon Greeks needed)
    - 15-minute snapshots during market hours
    - Automated feature engineering for ML
    """
    
    def __init__(self, polygon_api_key: str):
        self.polygon_api_key = polygon_api_key
        self.eastern = pytz.timezone('US/Eastern')
        
        # Load symbols for production
        self.symbols = self._load_production_symbols()
        logger.info(f"Loaded {len(self.symbols)} symbols for production pipeline")
        
        # Advanced tier configuration
        self.rate_limit_config = RateLimitConfig(
            requests_per_minute=1000,
            concurrent_requests=30,  # Conservative for stability
            retry_attempts=3,
            backoff_factor=1.5
        )
        
        self.batch_processor = BatchProcessor(batch_size=50, max_workers=10)
        
        # Processing components
        self.data_cleaner = DataCleaner()
        self.feature_engineer = AdvancedFeatureEngineer()
        
        # Pipeline state
        self.last_snapshot_time = None
        self.pipeline_stats = {
            'snapshots_today': 0,
            'symbols_processed': 0,
            'options_chains_collected': 0,
            'features_generated': 0,
            'errors_encountered': 0
        }
    
    def _load_production_symbols(self) -> List[str]:
        """Load symbols from production symbol list"""
        symbol_file = Path("config/symbols.txt")
        
        if symbol_file.exists():
            with open(symbol_file, 'r') as f:
                symbols = [line.strip().upper() for line in f if line.strip()]
            logger.info(f"Loaded {len(symbols)} symbols from {symbol_file}")
            return symbols
        else:
            # Fallback to test symbols if no production list
            test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ", "NVDA", "AMZN"]
            logger.warning(f"No production symbol list found, using {len(test_symbols)} test symbols")
            return test_symbols
    
    async def check_market_status(self) -> Dict:
        """Check if market is open using Polygon Calendar API"""
        try:
            async with PolygonClient(self.polygon_api_key, self.rate_limit_config) as client:
                today = date.today().strftime('%Y-%m-%d')
                
                # Get market status from Polygon Calendar API
                url = f"{client.base_url}/v1/marketstatus/upcoming"
                params = {"apikey": client.api_key}
                
                async with client.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Parse market status
                        now_et = datetime.now(self.eastern)
                        
                        # Check if today is a trading day
                        is_trading_day = True  # Default assumption
                        market_open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                        market_close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
                        
                        # Parse Polygon response for holidays/closures
                        if 'results' in data:
                            for event in data['results']:
                                event_date = event.get('date', '')
                                if event_date == today and event.get('status') == 'closed':
                                    is_trading_day = False
                                    break
                        
                        is_market_hours = market_open_time <= now_et <= market_close_time
                        is_weekday = now_et.weekday() < 5
                        is_market_open = is_trading_day and is_weekday and is_market_hours
                        
                        return {
                            'is_market_open': is_market_open,
                            'is_trading_day': is_trading_day,
                            'is_weekday': is_weekday,
                            'is_market_hours': is_market_hours,
                            'current_time_et': now_et.isoformat(),
                            'market_open_time': market_open_time.isoformat(),
                            'market_close_time': market_close_time.isoformat(),
                            'next_check_in_minutes': 15
                        }
                    else:
                        logger.warning(f"Could not get market status from Polygon: {response.status}")
                        # Fallback to basic time-based check
                        return self._fallback_market_check()
                        
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return self._fallback_market_check()
    
    def _fallback_market_check(self) -> Dict:
        """Fallback market check without API"""
        now_et = datetime.now(self.eastern)
        market_open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        is_weekday = now_et.weekday() < 5
        is_market_hours = market_open_time <= now_et <= market_close_time
        is_market_open = is_weekday and is_market_hours
        
        return {
            'is_market_open': is_market_open,
            'is_trading_day': is_weekday,  # Assume weekdays are trading days
            'is_weekday': is_weekday,
            'is_market_hours': is_market_hours,
            'current_time_et': now_et.isoformat(),
            'market_open_time': market_open_time.isoformat(),
            'market_close_time': market_close_time.isoformat(),
            'next_check_in_minutes': 15,
            'note': 'Fallback check - no API verification'
        }
    
    async def collect_stock_snapshot(self, symbol: str) -> Optional[Dict]:
        """Collect current stock data snapshot"""
        try:
            async with PolygonClient(self.polygon_api_key, self.rate_limit_config) as client:
                # Get last 30 minutes of data for context
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d')
                
                minute_data = await client.get_stock_bars(
                    symbol=symbol,
                    from_date=start_date,
                    to_date=end_date,
                    timespan="minute",
                    multiplier=1
                )
                
                if minute_data:
                    return {
                        'symbol': symbol,
                        'snapshot_time': datetime.now().isoformat(),
                        'minute_data': minute_data,
                        'latest_price': minute_data[-1].get('c') if minute_data else None,
                        'data_points': len(minute_data)
                    }
                else:
                    logger.warning(f"No stock data received for {symbol}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error collecting stock snapshot for {symbol}: {e}")
            return None
    
    async def collect_options_chain(self, symbol: str) -> Optional[Dict]:
        """Collect complete options chain within 60 days"""
        try:
            async with PolygonClient(self.polygon_api_key, self.rate_limit_config) as client:
                # Calculate date range (next 60 days)
                start_date = datetime.now().date()
                end_date = start_date + timedelta(days=60)
                
                all_contracts = []
                
                # Get options contracts for the next 8 weeks (covers 60 days)
                current_date = start_date
                while current_date <= end_date:
                    # Skip weekends for efficiency
                    if current_date.weekday() < 5:
                        date_str = current_date.strftime('%Y-%m-%d')
                        
                        # Get contracts expiring on this date
                        contracts = await client.get_options_contracts(
                            underlying=symbol,
                            expiration_date=date_str
                        )
                        
                        if contracts:
                            # Add expiration date to each contract for reference
                            for contract in contracts:
                                contract['collection_time'] = datetime.now().isoformat()
                                contract['days_to_expiry'] = (current_date - start_date).days
                            
                            all_contracts.extend(contracts)
                    
                    current_date += timedelta(days=1)
                    
                    # Small delay to be respectful to API
                    await asyncio.sleep(0.1)
                
                if all_contracts:
                    # Organize contracts by type and expiration
                    calls = [c for c in all_contracts if c.get('contract_type') == 'call']
                    puts = [c for c in all_contracts if c.get('contract_type') == 'put']
                    
                    return {
                        'symbol': symbol,
                        'collection_time': datetime.now().isoformat(),
                        'total_contracts': len(all_contracts),
                        'calls_count': len(calls),
                        'puts_count': len(puts),
                        'all_contracts': all_contracts,
                        'date_range': {
                            'start': start_date.isoformat(),
                            'end': end_date.isoformat()
                        }
                    }
                else:
                    logger.warning(f"No options contracts found for {symbol}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error collecting options chain for {symbol}: {e}")
            return None
    
    async def take_market_snapshot(self) -> Dict:
        """Take complete market snapshot (stocks + options)"""
        snapshot_time = datetime.now()
        logger.info(f"📸 Taking market snapshot at {snapshot_time.strftime('%H:%M:%S')}")
        
        # Collect stock data for all symbols
        logger.info(f"Collecting stock data for {len(self.symbols)} symbols...")
        stock_results = await self.batch_processor.process_symbols_batch(
            symbols=self.symbols,
            process_func=self.collect_stock_snapshot
        )
        
        # Collect options data for high-priority symbols (liquid stocks)
        priority_symbols = self.symbols[:100]  # Adjust based on your needs
        logger.info(f"Collecting options chains for {len(priority_symbols)} priority symbols...")
        
        options_results = await self.batch_processor.process_symbols_batch(
            symbols=priority_symbols,
            process_func=self.collect_options_chain
        )
        
        # Calculate snapshot statistics
        successful_stocks = len([r for r in stock_results.values() if r is not None])
        successful_options = len([r for r in options_results.values() if r is not None])
        
        total_minute_bars = sum(
            r.get('data_points', 0) for r in stock_results.values() if r
        )
        total_options_contracts = sum(
            r.get('total_contracts', 0) for r in options_results.values() if r
        )
        
        snapshot_data = {
            'snapshot_id': snapshot_time.strftime('%Y%m%d_%H%M%S'),
            'timestamp': snapshot_time.isoformat(),
            'market_snapshot': {
                'symbols_attempted': len(self.symbols),
                'successful_stocks': successful_stocks,
                'successful_options': successful_options,
                'total_minute_bars': total_minute_bars,
                'total_options_contracts': total_options_contracts,
                'success_rate_stocks': round(successful_stocks / len(self.symbols) * 100, 2),
                'success_rate_options': round(successful_options / len(priority_symbols) * 100, 2)
            },
            'stock_data': stock_results,
            'options_data': options_results
        }
        
        # Save snapshot
        await self._save_snapshot(snapshot_data)
        
        # Update pipeline stats
        self.pipeline_stats['snapshots_today'] += 1
        self.pipeline_stats['symbols_processed'] += successful_stocks
        self.pipeline_stats['options_chains_collected'] += successful_options
        
        logger.info(f"✅ Snapshot complete: {successful_stocks}/{len(self.symbols)} stocks, {successful_options}/{len(priority_symbols)} options")
        
        return snapshot_data
    
    async def _save_snapshot(self, snapshot_data: Dict):
        """Save market snapshot to Bronze layer"""
        try:
            # Create timestamp-based directory structure
            timestamp = datetime.now()
            date_str = timestamp.strftime('%Y%m%d')
            time_str = timestamp.strftime('%H%M%S')
            
            # Save to Bronze layer with organized structure
            bronze_dir = DATA_DIR / "bronze" / "market_snapshots" / date_str
            bronze_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_file = bronze_dir / f"snapshot_{time_str}.json"
            
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot_data, f, indent=2, default=str)
            
            logger.info(f"💾 Snapshot saved: {snapshot_file}")
            
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
    
    async def process_snapshot_to_features(self, snapshot_data: Dict) -> Dict:
        """Process snapshot data through feature engineering pipeline"""
        try:
            logger.info("🔄 Processing snapshot through feature engineering...")
            
            processed_symbols = 0
            feature_results = {}
            
            for symbol, stock_data in snapshot_data['stock_data'].items():
                if stock_data is None:
                    continue
                
                try:
                    # Get corresponding options data
                    options_data = snapshot_data['options_data'].get(symbol)
                    options_contracts = options_data.get('all_contracts', []) if options_data else []
                    
                    # Clean stock data
                    minute_data = stock_data.get('minute_data', [])
                    if not minute_data:
                        continue
                    
                    clean_df = self.data_cleaner.clean_stock_data(minute_data)
                    if clean_df.empty:
                        continue
                    
                    # Create features (including custom Greeks calculation)
                    features_df = self.feature_engineer.create_comprehensive_features(
                        clean_df, symbol, options_contracts
                    )
                    
                    if not features_df.empty:
                        # Save to Silver layer
                        await self._save_features(features_df, symbol, snapshot_data['snapshot_id'])
                        
                        feature_results[symbol] = {
                            'success': True,
                            'feature_count': len(features_df.columns),
                            'data_points': len(features_df),
                            'has_options_features': 'total_gamma_exposure' in features_df.columns
                        }
                        processed_symbols += 1
                    
                except Exception as e:
                    logger.error(f"Error processing features for {symbol}: {e}")
                    feature_results[symbol] = {'success': False, 'error': str(e)}
            
            self.pipeline_stats['features_generated'] += processed_symbols
            
            logger.info(f"✅ Feature processing complete: {processed_symbols} symbols processed")
            
            return {
                'processed_symbols': processed_symbols,
                'results': feature_results,
                'snapshot_id': snapshot_data['snapshot_id']
            }
            
        except Exception as e:
            logger.error(f"Error in feature processing: {e}")
            return {'processed_symbols': 0, 'error': str(e)}
    
    async def _save_features(self, features_df, symbol: str, snapshot_id: str):
        """Save processed features to Silver layer"""
        try:
            silver_dir = DATA_DIR / "silver" / "snapshots" / datetime.now().strftime('%Y%m%d')
            silver_dir.mkdir(parents=True, exist_ok=True)
            
            features_file = silver_dir / f"{symbol}_{snapshot_id}.parquet"
            features_df.to_parquet(features_file, compression='snappy')
            
            # Also save metadata
            metadata = {
                'symbol': symbol,
                'snapshot_id': snapshot_id,
                'feature_count': len(features_df.columns),
                'data_points': len(features_df),
                'processing_time': datetime.now().isoformat()
            }
            
            metadata_file = silver_dir / f"{symbol}_{snapshot_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving features for {symbol}: {e}")
    
    async def run_pipeline_cycle(self) -> Dict:
        """Run one complete pipeline cycle"""
        cycle_start = datetime.now()
        
        try:
            # 1. Check market status
            market_status = await self.check_market_status()
            
            if not market_status['is_market_open']:
                logger.info(f"Market is closed. Next check in {market_status['next_check_in_minutes']} minutes.")
                return {
                    'cycle_completed': False,
                    'reason': 'market_closed',
                    'market_status': market_status,
                    'next_check_minutes': market_status['next_check_in_minutes']
                }
            
            # 2. Take market snapshot
            snapshot_data = await self.take_market_snapshot()
            
            # 3. Process through feature engineering
            feature_results = await self.process_snapshot_to_features(snapshot_data)
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            
            cycle_result = {
                'cycle_completed': True,
                'cycle_duration_seconds': cycle_duration,
                'snapshot_data': snapshot_data['market_snapshot'],
                'feature_results': feature_results,
                'pipeline_stats': self.pipeline_stats.copy(),
                'next_cycle_minutes': 15
            }
            
            logger.info(f"🎯 Pipeline cycle completed in {cycle_duration:.1f}s")
            return cycle_result
            
        except Exception as e:
            logger.error(f"Error in pipeline cycle: {e}")
            self.pipeline_stats['errors_encountered'] += 1
            return {
                'cycle_completed': False,
                'error': str(e),
                'next_check_minutes': 5  # Retry sooner on error
            }
    
    async def run_continuous_pipeline(self):
        """Run continuous pipeline during market hours"""
        logger.info("🚀 Starting continuous market pipeline")
        
        while True:
            try:
                # Run pipeline cycle
                cycle_result = await self.run_pipeline_cycle()
                
                # Determine wait time for next cycle
                if cycle_result['cycle_completed']:
                    wait_minutes = cycle_result.get('next_cycle_minutes', 15)
                else:
                    wait_minutes = cycle_result.get('next_check_minutes', 15)
                
                logger.info(f"⏰ Next cycle in {wait_minutes} minutes")
                
                # Wait for next cycle
                await asyncio.sleep(wait_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Pipeline stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in continuous pipeline: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on unexpected error


async def main():
    """Main execution function"""
    import os
    
    # Setup logging
    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / 'production_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    # Get API key
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        logger.error("POLYGON_API_KEY environment variable not set")
        return
    
    # Create and run pipeline
    pipeline = ProductionMarketPipeline(api_key)
    
    try:
        # Check if we should run continuously or just one cycle
        if len(sys.argv) > 1 and sys.argv[1] == '--once':
            # Run single cycle for testing
            result = await pipeline.run_pipeline_cycle()
            print(json.dumps(result, indent=2, default=str))
        else:
            # Run continuous pipeline
            await pipeline.run_continuous_pipeline()
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())