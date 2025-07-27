# scripts/process_to_silver.py
import asyncio
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.processing.data_cleaner import DataCleaner
from src.processing.feature_engineer import AdvancedFeatureEngineer
from config.settings import DATA_DIR, LOGS_DIR

logger = logging.getLogger(__name__)

class ProductionSilverProcessor:
    """
    Production-ready Silver layer processor
    Integrates data cleaning, feature engineering, and options data
    """
    
    def __init__(self):
        self.data_cleaner = DataCleaner()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.bronze_dir = DATA_DIR / "bronze"
        self.silver_dir = DATA_DIR / "silver"
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        
    def load_bronze_data(self, symbol: str) -> tuple:
        """Load both stock and options bronze data for a symbol"""
        # Load stock data
        stock_data = self._load_bronze_file(symbol, "historical_stock_data")
        
        # Load options data
        options_data = self._load_bronze_file(symbol, "historical_options_data")
        
        return stock_data, options_data
    
    def _load_bronze_file(self, symbol: str, data_type: str) -> dict:
        """Load the most recent bronze file for a symbol and data type"""
        bronze_path = self.bronze_dir / data_type
        
        if not bronze_path.exists():
            logger.warning(f"No bronze directory found for {data_type}")
            return None
        
        # Find the most recent file for this symbol
        pattern = f"{symbol}_{data_type}_*.json"
        files = list(bronze_path.glob(pattern))
        
        if not files:
            logger.warning(f"No bronze data found for {symbol} in {data_type}")
            return None
        
        # Get the most recent file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading {data_type} from {latest_file}")
        
        try:
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {latest_file}: {e}")
            return None
    
    def process_symbol_complete_pipeline(self, symbol: str) -> dict:
        """Run complete pipeline: load → clean → engineer features → save"""
        logger.info(f"=== Processing {symbol} through complete pipeline ===")
        
        try:
            # 1. Load Bronze data
            stock_data, options_data = self.load_bronze_data(symbol)
            
            if not stock_data:
                logger.error(f"No stock data found for {symbol}")
                return {'success': False, 'error': 'No stock data'}
            
            # Extract minute data
            minute_data = stock_data.get('minute_data', [])
            if not minute_data:
                logger.error(f"No minute data found for {symbol}")
                return {'success': False, 'error': 'No minute data'}
            
            logger.info(f"{symbol}: Loaded {len(minute_data)} minute bars")
            
            # 2. Clean the data
            clean_df = self.data_cleaner.clean_stock_data(minute_data)
            if clean_df.empty:
                logger.error(f"No clean data remaining for {symbol}")
                return {'success': False, 'error': 'No clean data after cleaning'}
            
            logger.info(f"{symbol}: {len(clean_df)} clean bars after cleaning")
            
            # 3. Extract options data if available
            options_contracts = None
            if options_data and options_data.get('contracts'):
                options_contracts = options_data['contracts']
                logger.info(f"{symbol}: Loaded {len(options_contracts)} options contracts")
            else:
                logger.info(f"{symbol}: No options data available")
            
            # 4. Feature engineering
            features_df = self.feature_engineer.create_comprehensive_features(
                clean_df, symbol, options_contracts
            )
            
            # Add metadata
            features_df['symbol'] = symbol
            features_df['processing_timestamp'] = datetime.now()
            
            logger.info(f"{symbol}: Created {len(features_df.columns)} features")
            
            # 5. Save to Silver layer
            save_result = self.save_silver_data(features_df, symbol)
            
            # 6. Generate processing summary
            summary = {
                'success': True,
                'symbol': symbol,
                'original_records': len(minute_data),
                'clean_records': len(clean_df),
                'final_features': len(features_df),
                'feature_count': len(features_df.columns),
                'options_contracts': len(options_contracts) if options_contracts else 0,
                'data_retention_pct': round(len(clean_df) / len(minute_data) * 100, 2),
                'date_range': {
                    'start': str(features_df.index.min()),
                    'end': str(features_df.index.max())
                },
                'files_saved': save_result
            }
            
            logger.info(f"✅ {symbol}: Pipeline complete - {summary['final_features']} records, {summary['feature_count']} features")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            return {'success': False, 'symbol': symbol, 'error': str(e)}
    
    def save_silver_data(self, df, symbol: str) -> dict:
        """Save processed data to Silver layer with metadata"""
        if df is None or df.empty:
            logger.warning(f"No data to save for {symbol}")
            return {'success': False}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Save main features as Parquet (efficient for large datasets)
            features_file = self.silver_dir / f"{symbol}_features_{timestamp}.parquet"
            df.to_parquet(features_file, compression='snappy')
            
            # Save metadata as JSON
            metadata = {
                'symbol': symbol,
                'processing_timestamp': timestamp,
                'rows': len(df),
                'columns': len(df.columns),
                'features': list(df.columns),
                'date_range': {
                    'start': str(df.index.min()),
                    'end': str(df.index.max())
                },
                'data_quality': {
                    'has_options_features': 'total_gamma_exposure' in df.columns,
                    'missing_values_pct': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
                    'infinite_values': int((df == float('inf')).sum().sum() + (df == float('-inf')).sum().sum())
                }
            }
            
            metadata_file = self.silver_dir / f"{symbol}_metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Save feature summary for ML model reference
            feature_summary = {
                'symbol': symbol,
                'timestamp': timestamp,
                'feature_categories': self._categorize_features(df.columns),
                'target_variables': [col for col in df.columns if col.startswith('future_')],
                'options_features': [col for col in df.columns if 'gamma' in col or 'options' in col]
            }
            
            summary_file = self.silver_dir / f"{symbol}_feature_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(feature_summary, f, indent=2)
            
            logger.info(f"💾 Saved {symbol} silver data:")
            logger.info(f"   Features: {features_file}")
            logger.info(f"   Metadata: {metadata_file}")
            logger.info(f"   Summary: {summary_file}")
            
            return {
                'success': True,
                'features_file': str(features_file),
                'metadata_file': str(metadata_file),
                'summary_file': str(summary_file)
            }
            
        except Exception as e:
            logger.error(f"Error saving {symbol} data: {e}")
            return {'success': False, 'error': str(e)}
    
    def _categorize_features(self, columns) -> dict:
        """Categorize features for better understanding"""
        categories = {
            'price_features': [],
            'technical_indicators': [],
            'volume_features': [],
            'statistical_features': [],
            'time_features': [],
            'microstructure_features': [],
            'options_features': [],
            'target_variables': [],
            'metadata': []
        }
        
        for col in columns:
            col_lower = col.lower()
            
            if col.startswith('future_'):
                categories['target_variables'].append(col)
            elif any(word in col_lower for word in ['gamma', 'options', 'call', 'put']):
                categories['options_features'].append(col)
            elif any(word in col_lower for word in ['return', 'momentum', 'position', 'acceleration']):
                categories['price_features'].append(col)
            elif any(word in col_lower for word in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'stoch', 'williams', 'cci', 'atr']):
                categories['technical_indicators'].append(col)
            elif any(word in col_lower for word in ['volume', 'vwap', 'obv', 'mfi']):
                categories['volume_features'].append(col)
            elif any(word in col_lower for word in ['volatility', 'skewness', 'kurtosis', 'autocorr']):
                categories['statistical_features'].append(col)
            elif any(word in col_lower for word in ['hour', 'minute', 'day', 'month', 'session', 'market']):
                categories['time_features'].append(col)
            elif any(word in col_lower for word in ['spread', 'impact', 'tick', 'illiquidity', 'kyle']):
                categories['microstructure_features'].append(col)
            else:
                categories['metadata'].append(col)
        
        return categories
    
    async def process_all_available_symbols(self) -> dict:
        """Process all symbols available in Bronze layer"""
        # Auto-detect symbols from bronze data
        symbols = self._detect_available_symbols()
        
        if not symbols:
            logger.error("No symbols found in bronze data")
            return {'success': False, 'error': 'No symbols found'}
        
        logger.info(f"🚀 Starting Silver processing for {len(symbols)} symbols: {symbols}")
        
        results = {}
        successful_count = 0
        failed_symbols = []
        
        for symbol in symbols:
            try:
                result = self.process_symbol_complete_pipeline(symbol)
                results[symbol] = result
                
                if result.get('success'):
                    successful_count += 1
                else:
                    failed_symbols.append(symbol)
                    
            except Exception as e:
                logger.error(f"Fatal error processing {symbol}: {e}")
                results[symbol] = {'success': False, 'error': str(e)}
                failed_symbols.append(symbol)
        
        # Generate final report
        final_report = self._generate_processing_report(results, successful_count, failed_symbols)
        
        logger.info("🎉 Silver layer processing complete!")
        logger.info(f"✅ Successful: {successful_count}/{len(symbols)} symbols")
        if failed_symbols:
            logger.warning(f"❌ Failed: {failed_symbols}")
        
        return final_report
    
    def _detect_available_symbols(self) -> list:
        """Auto-detect available symbols from bronze data"""
        bronze_stock_dir = self.bronze_dir / "historical_stock_data"
        
        if not bronze_stock_dir.exists():
            logger.error("No bronze stock data directory found")
            return []
        
        symbols = set()
        for file in bronze_stock_dir.glob("*_historical_stock_data_*.json"):
            # Extract symbol from filename
            symbol = file.stem.split('_')[0]
            symbols.add(symbol)
        
        symbol_list = sorted(list(symbols))
        logger.info(f"Detected {len(symbol_list)} symbols: {symbol_list}")
        return symbol_list
    
    def _generate_processing_report(self, results: dict, successful_count: int, failed_symbols: list) -> dict:
        """Generate comprehensive processing report"""
        total_records = sum(r.get('final_features', 0) for r in results.values() if r.get('success'))
        total_features = sum(r.get('feature_count', 0) for r in results.values() if r.get('success'))
        avg_retention = sum(r.get('data_retention_pct', 0) for r in results.values() if r.get('success')) / max(successful_count, 1)
        
        # Calculate options coverage
        symbols_with_options = sum(1 for r in results.values() if r.get('success') and r.get('options_contracts', 0) > 0)
        
        report = {
            'processing_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_symbols_processed': len(results),
                'successful_symbols': successful_count,
                'failed_symbols': len(failed_symbols),
                'success_rate_pct': round(successful_count / len(results) * 100, 2) if results else 0
            },
            'data_statistics': {
                'total_records_processed': total_records,
                'average_features_per_symbol': round(total_features / max(successful_count, 1), 0),
                'average_data_retention_pct': round(avg_retention, 2),
                'symbols_with_options_data': symbols_with_options,
                'options_coverage_pct': round(symbols_with_options / max(successful_count, 1) * 100, 2)
            },
            'failed_symbols': failed_symbols,
            'detailed_results': results,
            'next_steps': [
                "Review failed symbols and fix data issues",
                "Validate feature quality in Silver layer",
                "Proceed to model training (Gold layer)",
                "Set up live data pipeline for market hours"
            ]
        }
        
        # Save report
        report_file = self.silver_dir / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Processing report saved: {report_file}")
        return report


async def main():
    """Main execution function for Silver layer processing"""
    # Setup logging
    LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / 'silver_processing.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🚀 Starting Production Silver Layer Processing")
    logger.info("=" * 60)
    
    try:
        # Create processor
        processor = ProductionSilverProcessor()
        
        # Process all available symbols
        final_report = await processor.process_all_available_symbols()
        
        # Print summary
        if final_report.get('summary'):
            summary = final_report['summary']
            data_stats = final_report['data_statistics']
            
            print("\n" + "=" * 60)
            print("🎯 SILVER LAYER PROCESSING COMPLETE")
            print("=" * 60)
            print(f"✅ Successful symbols: {summary['successful_symbols']}")
            print(f"❌ Failed symbols: {summary['failed_symbols']}")
            print(f"📊 Success rate: {summary['success_rate_pct']}%")
            print(f"📈 Total records: {data_stats['total_records_processed']:,}")
            print(f"🔧 Avg features per symbol: {data_stats['average_features_per_symbol']}")
            print(f"📊 Data retention: {data_stats['average_data_retention_pct']}%")
            print(f"📱 Options coverage: {data_stats['options_coverage_pct']}%")
            print("=" * 60)
            
            if final_report.get('failed_symbols'):
                print(f"⚠️  Failed symbols: {', '.join(final_report['failed_symbols'])}")
            
            print("🎉 Ready for model training!")
        
    except Exception as e:
        logger.error(f"Silver layer processing failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())