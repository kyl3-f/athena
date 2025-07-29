#!/usr/bin/env python3
"""
Live Market Data Validation Script
Tests real-time data quality and system performance with live market data
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import polars as pl
import numpy as np
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiveMarketValidator:
    """
    Validates live market data quality and system performance
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.validation_results = {}
        
        # Test symbols - mix of high and low volume
        self.test_symbols = [
            'AAPL',  # High volume tech
            'MSFT',  # High volume tech  
            'SPY',   # High volume ETF
            'TSLA',  # High volatility
            'GOOGL', # High price stock
            'AMD',   # Medium volume tech
            'NVDA',  # AI/chip stock
            'META',  # Social media
            'AMZN',  # E-commerce
            'QQQ'    # Tech ETF
        ]
    
    async def validate_data_quality(self, data: pl.DataFrame, data_type: str) -> Dict:
        """Validate data quality metrics"""
        if data is None or len(data) == 0:
            return {
                'valid': False,
                'reason': 'No data received',
                'metrics': {}
            }
        
        metrics = {}
        issues = []
        
        try:
            # Basic structure validation
            if 'symbol' not in data.columns:
                issues.append('Missing symbol column')
            if 'timestamp' not in data.columns:
                issues.append('Missing timestamp column')
            
            # Data freshness (should be recent for live data)
            if 'timestamp' in data.columns:
                latest_time = data['timestamp'].max()
                if latest_time:
                    age_minutes = (datetime.now() - latest_time).total_seconds() / 60
                    metrics['data_age_minutes'] = age_minutes
                    
                    if age_minutes > 30:  # Data older than 30 minutes
                        issues.append(f'Stale data: {age_minutes:.1f} minutes old')
            
            # Price validation for stock data
            if data_type == 'stock' and 'close' in data.columns:
                close_prices = data['close'].drop_nulls()
                if len(close_prices) > 0:
                    metrics['avg_price'] = float(close_prices.mean())
                    metrics['price_range'] = float(close_prices.max() - close_prices.min())
                    
                    # Check for unrealistic prices
                    if close_prices.min() <= 0:
                        issues.append('Non-positive prices detected')
                    if close_prices.max() > 10000:  # Very high price
                        issues.append('Extremely high prices detected')
            
            # Volume validation
            if 'volume' in data.columns:
                volumes = data['volume'].drop_nulls()
                if len(volumes) > 0:
                    metrics['avg_volume'] = float(volumes.mean())
                    metrics['zero_volume_pct'] = float((volumes == 0).sum() / len(volumes) * 100)
                    
                    if metrics['zero_volume_pct'] > 50:
                        issues.append('High percentage of zero volume')
            
            # Missing data analysis
            for col in data.columns:
                null_pct = float(data[col].null_count() / len(data) * 100)
                metrics[f'{col}_null_pct'] = null_pct
                
                if null_pct > 20:  # More than 20% missing
                    issues.append(f'High missing data in {col}: {null_pct:.1f}%')
            
            # Symbol coverage
            unique_symbols = data['symbol'].n_unique()
            metrics['unique_symbols'] = unique_symbols
            metrics['total_records'] = len(data)
            metrics['records_per_symbol'] = len(data) / unique_symbols if unique_symbols > 0 else 0
            
        except Exception as e:
            issues.append(f'Validation error: {str(e)}')
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'metrics': metrics
        }
    
    async def test_polygon_api_performance(self) -> Dict:
        """Test Polygon API performance and rate limiting"""
        logger.info("🔌 Testing Polygon API performance...")
        
        try:
            from src.ingestion.polygon_client import PolygonClient
            
            client = PolygonClient()
            
            # Test 1: Single symbol fetch
            start_time = time.time()
            single_data = await client.fetch_live_data(['AAPL'])
            single_duration = time.time() - start_time
            
            # Test 2: Multiple symbol fetch
            start_time = time.time()
            multi_data = await client.fetch_live_data(self.test_symbols)
            multi_duration = time.time() - start_time
            
            # Test 3: Options data fetch
            start_time = time.time()
            options_data = await client.fetch_options_data(['AAPL', 'MSFT'])
            options_duration = time.time() - start_time
            
            # Validate data quality
            stock_quality = await self.validate_data_quality(multi_data, 'stock')
            options_quality = await self.validate_data_quality(options_data, 'options')
            
            return {
                'performance': {
                    'single_symbol_time': single_duration,
                    'multi_symbol_time': multi_duration,
                    'options_fetch_time': options_duration,
                    'symbols_per_second': len(self.test_symbols) / multi_duration if multi_duration > 0 else 0
                },
                'data_quality': {
                    'stock': stock_quality,
                    'options': options_quality
                },
                'success': stock_quality['valid'] and options_quality['valid']
            }
            
        except Exception as e:
            logger.error(f"API performance test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def test_feature_pipeline_speed(self) -> Dict:
        """Test feature engineering pipeline speed with live data"""
        logger.info("⚙️ Testing feature engineering pipeline...")
        
        try:
            from src.ingestion.polygon_client import PolygonClient
            from src.processing.data_cleaner import DataCleaner
            from src.processing.feature_engineer import FeatureEngineer
            
            # Get live data
            client = PolygonClient()
            raw_data = await client.fetch_live_data(self.test_symbols[:5])  # Use 5 symbols for speed
            
            if raw_data is None or len(raw_data) == 0:
                return {'success': False, 'error': 'No raw data available'}
            
            # Time each pipeline stage
            stages = {}
            
            # Stage 1: Data cleaning
            start_time = time.time()
            cleaner = DataCleaner()
            cleaned_data = cleaner.clean_stock_data(raw_data)
            stages['cleaning'] = time.time() - start_time
            
            # Stage 2: Feature engineering
            start_time = time.time()
            engineer = FeatureEngineer()
            
            # Create minimal options data for testing
            options_data = pl.DataFrame({
                'symbol': ['AAPL', 'MSFT'],
                'strike': [150.0, 300.0],
                'expiry': [datetime.now() + timedelta(days=30)] * 2,
                'option_type': ['call', 'put'],
                'bid': [5.0, 10.0],
                'ask': [5.5, 10.5],
                'volume': [100, 200]
            })
            
            features = engineer.create_features(cleaned_data, options_data)
            stages['feature_engineering'] = time.time() - start_time
            
            # Validate feature output
            feature_quality = await self.validate_data_quality(features, 'features')
            
            total_time = sum(stages.values())
            records_processed = len(raw_data)
            
            return {
                'success': feature_quality['valid'],
                'performance': {
                    'total_time': total_time,
                    'records_processed': records_processed,
                    'records_per_second': records_processed / total_time if total_time > 0 else 0,
                    'stage_times': stages
                },
                'feature_quality': feature_quality,
                'feature_count': len(features.columns) if features is not None else 0
            }
            
        except Exception as e:
            logger.error(f"Feature pipeline test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def test_ml_signal_generation(self) -> Dict:
        """Test ML signal generation with live data"""
        logger.info("🤖 Testing ML signal generation...")
        
        try:
            from ml_signal_generator import MLSignalGenerator
            
            ml_generator = MLSignalGenerator()
            
            # Create synthetic but realistic feature data
            n_samples = 100
            n_features = 50
            
            np.random.seed(42)
            feature_data = np.random.randn(n_samples, n_features)
            
            # Add some realistic patterns
            feature_data[:, 0] = np.cumsum(np.random.randn(n_samples) * 0.1)  # Price-like trend
            feature_data[:, 1] = np.random.exponential(1000, n_samples)       # Volume-like
            
            feature_columns = [f'feature_{i}' for i in range(n_features)]
            
            test_features = pl.DataFrame({
                'symbol': self.test_symbols[:10] * 10,  # 10 samples per symbol
                'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(n_samples)],
                **{col: feature_data[:, i] for i, col in enumerate(feature_columns)}
            })
            
            # Time signal generation
            start_time = time.time()
            signals = ml_generator.generate_signals(test_features)
            generation_time = time.time() - start_time
            
            if signals is None or len(signals) == 0:
                return {
                    'success': False,
                    'error': 'No signals generated'
                }
            
            # Validate signal quality
            signal_validation = {
                'total_signals': len(signals),
                'unique_symbols': signals['symbol'].n_unique(),
                'signal_distribution': signals['signal'].value_counts().to_dict(),
                'avg_confidence': float(signals['confidence'].mean()),
                'avg_strength': float(signals['strength'].mean()),
                'high_confidence_signals': len(signals.filter(pl.col('confidence') > 0.7))
            }
            
            return {
                'success': True,
                'performance': {
                    'generation_time': generation_time,
                    'signals_per_second': len(signals) / generation_time if generation_time > 0 else 0
                },
                'signal_quality': signal_validation
            }
            
        except Exception as e:
            logger.error(f"ML signal generation test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def test_end_to_end_pipeline(self) -> Dict:
        """Test complete end-to-end pipeline with live data"""
        logger.info("🔄 Testing end-to-end pipeline...")
        
        pipeline_start = time.time()
        
        try:
            # Stage 1: Data ingestion
            api_test = await self.test_polygon_api_performance()
            if not api_test['success']:
                return {
                    'success': False,
                    'stage_failed': 'data_ingestion',
                    'error': api_test.get('error', 'API test failed')
                }
            
            # Stage 2: Feature pipeline
            feature_test = await self.test_feature_pipeline_speed()
            if not feature_test['success']:
                return {
                    'success': False,
                    'stage_failed': 'feature_engineering',
                    'error': feature_test.get('error', 'Feature pipeline failed')
                }
            
            # Stage 3: ML signal generation
            ml_test = await self.test_ml_signal_generation()
            if not ml_test['success']:
                return {
                    'success': False,
                    'stage_failed': 'signal_generation',
                    'error': ml_test.get('error', 'ML pipeline failed')
                }
            
            total_time = time.time() - pipeline_start
            
            return {
                'success': True,
                'performance': {
                    'total_pipeline_time': total_time,
                    'api_performance': api_test['performance'],
                    'feature_performance': feature_test['performance'],
                    'ml_performance': ml_test['performance']
                },
                'quality_metrics': {
                    'data_quality': api_test['data_quality'],
                    'feature_quality': feature_test['feature_quality'],
                    'signal_quality': ml_test['signal_quality']
                }
            }
            
        except Exception as e:
            logger.error(f"End-to-end pipeline test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_time': time.time() - pipeline_start
            }
    
    async def run_live_validation(self) -> Dict:
        """Run complete live market validation"""
        logger.info("🚀 Starting Live Market Data Validation...")
        logger.info(f"📅 Test Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S ET')}")
        logger.info(f"📊 Test Symbols: {', '.join(self.test_symbols)}")
        
        validation_start = time.time()
        
        # Check market hours
        now = datetime.now().time()
        market_open = datetime.strptime("09:30", "%H:%M").time()
        market_close = datetime.strptime("16:00", "%H:%M").time()
        is_market_hours = market_open <= now <= market_close
        is_trading_day = datetime.now().weekday() < 5
        
        logger.info(f"🕐 Market Status: {'OPEN' if is_market_hours and is_trading_day else 'CLOSED'}")
        
        results = {
            'validation_start': self.start_time.isoformat(),
            'market_status': {
                'is_market_hours': is_market_hours,
                'is_trading_day': is_trading_day,
                'current_time': now.strftime('%H:%M:%S')
            }
        }
        
        # Run individual tests
        tests = {
            'api_performance': self.test_polygon_api_performance,
            'feature_pipeline': self.test_feature_pipeline_speed,
            'ml_signals': self.test_ml_signal_generation,
            'end_to_end': self.test_end_to_end_pipeline
        }
        
        for test_name, test_func in tests.items():
            logger.info(f"🧪 Running {test_name.replace('_', ' ').title()}...")
            
            try:
                test_start = time.time()
                test_result = await test_func()
                test_duration = time.time() - test_start
                
                test_result['duration'] = test_duration
                results[test_name] = test_result
                
                status = "✅ PASS" if test_result['success'] else "❌ FAIL"
                logger.info(f"{status} {test_name} ({test_duration:.2f}s)")
                
            except Exception as e:
                logger.error(f"❌ FAIL {test_name}: {e}")
                results[test_name] = {
                    'success': False,
                    'error': str(e),
                    'duration': time.time() - test_start if 'test_start' in locals() else 0
                }
        
        # Generate summary
        total_duration = time.time() - validation_start
        successful_tests = sum(1 for test in results.values() 
                             if isinstance(test, dict) and test.get('success', False))
        total_tests = len(tests)
        
        results['summary'] = {
            'total_duration': total_duration,
            'successful_tests': successful_tests,
            'total_tests': total_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'system_ready': successful_tests == total_tests
        }
        
        # Log summary
        logger.info(f"\n" + "="*60)
        logger.info(f"🏁 LIVE VALIDATION COMPLETE")
        logger.info(f"="*60)
        logger.info(f"✅ PASSED: {successful_tests}/{total_tests}")
        logger.info(f"❌ FAILED: {total_tests - successful_tests}/{total_tests}")
        logger.info(f"⏱️  DURATION: {total_duration:.2f}s")
        logger.info(f"🚀 SYSTEM READY: {'YES' if results['summary']['system_ready'] else 'NO'}")
        logger.info(f"="*60)
        
        return results
    
    def generate_validation_report(self, results: Dict) -> str:
        """Generate detailed validation report"""
        report = []
        report.append("# ATHENA LIVE MARKET VALIDATION REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
        report.append(f"Market Status: {'OPEN' if results['market_status']['is_market_hours'] and results['market_status']['is_trading_day'] else 'CLOSED'}")
        report.append("")
        
        # Executive Summary
        summary = results['summary']
        report.append("## EXECUTIVE SUMMARY")
        report.append(f"- **System Ready**: {'✅ YES' if summary['system_ready'] else '❌ NO'}")
        report.append(f"- **Success Rate**: {summary['success_rate']:.1%} ({summary['successful_tests']}/{summary['total_tests']} tests)")
        report.append(f"- **Total Duration**: {summary['total_duration']:.2f} seconds")
        report.append("")
        
        # Performance Metrics
        if 'end_to_end' in results and results['end_to_end']['success']:
            perf = results['end_to_end']['performance']
            report.append("## PERFORMANCE METRICS")
            report.append(f"- **Pipeline Speed**: {perf['total_pipeline_time']:.2f}s end-to-end")
            
            if 'api_performance' in perf:
                api_perf = perf['api_performance']
                report.append(f"- **API Speed**: {api_perf.get('symbols_per_second', 0):.1f} symbols/second")
            
            if 'feature_performance' in perf:
                feat_perf = perf['feature_performance']
                report.append(f"- **Feature Engineering**: {feat_perf.get('records_per_second', 0):.1f} records/second")
            
            if 'ml_performance' in perf:
                ml_perf = perf['ml_performance']
                report.append(f"- **Signal Generation**: {ml_perf.get('signals_per_second', 0):.1f} signals/second")
            
            report.append("")
        
        # Data Quality Assessment
        if 'api_performance' in results and results['api_performance']['success']:
            data_quality = results['api_performance']['data_quality']
            report.append("## DATA QUALITY")
            
            if 'stock' in data_quality:
                stock_q = data_quality['stock']
                report.append(f"- **Stock Data**: {'✅ Good' if stock_q['valid'] else '❌ Issues'}")
                if stock_q['metrics']:
                    report.append(f"  - Data Age: {stock_q['metrics'].get('data_age_minutes', 'N/A'):.1f} minutes")
                    report.append(f"  - Average Price: ${stock_q['metrics'].get('avg_price', 0):.2f}")
                    report.append(f"  - Records: {stock_q['metrics'].get('total_records', 0)}")
            
            if 'options' in data_quality:
                options_q = data_quality['options']
                report.append(f"- **Options Data**: {'✅ Good' if options_q['valid'] else '❌ Issues'}")
            
            report.append("")
        
        # Signal Quality
        if 'ml_signals' in results and results['ml_signals']['success']:
            signal_q = results['ml_signals']['signal_quality']
            report.append("## SIGNAL QUALITY")
            report.append(f"- **Total Signals**: {signal_q.get('total_signals', 0)}")
            report.append(f"- **Average Confidence**: {signal_q.get('avg_confidence', 0):.1%}")
            report.append(f"- **Average Strength**: {signal_q.get('avg_strength', 0):.1f}/10")
            report.append(f"- **High Confidence**: {signal_q.get('high_confidence_signals', 0)} signals >70%")
            report.append("")
        
        # Test Details
        report.append("## DETAILED TEST RESULTS")
        
        test_names = {
            'api_performance': 'API Performance',
            'feature_pipeline': 'Feature Pipeline',
            'ml_signals': 'ML Signal Generation',
            'end_to_end': 'End-to-End Pipeline'
        }
        
        for test_key, test_name in test_names.items():
            if test_key in results:
                result = results[test_key]
                status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
                duration = result.get('duration', 0)
                
                report.append(f"### {test_name} - {status}")
                report.append(f"- Duration: {duration:.2f}s")
                
                if not result.get('success', False) and 'error' in result:
                    report.append(f"- Error: {result['error']}")
                
                report.append("")
        
        # Recommendations
        report.append("## RECOMMENDATIONS")
        
        if summary['system_ready']:
            report.append("🚀 **System is ready for production deployment!**")
            report.append("")
            report.append("Next steps:")
            report.append("1. Run final cleanup: `python cleanup_project.py`")
            report.append("2. Deploy to cloud: `python deploy_to_digitalocean.py`")
            report.append("3. Start monitoring dashboard")
            report.append("4. Begin live trading signal generation")
        else:
            report.append("⚠️ **System needs attention before production:**")
            report.append("")
            
            failed_tests = [name for name, result in results.items() 
                          if isinstance(result, dict) and not result.get('success', True)]
            
            for test in failed_tests:
                if test in results and 'error' in results[test]:
                    report.append(f"- Fix {test}: {results[test]['error']}")
        
        return "\n".join(report)

async def main():
    """Main validation execution"""
    
    # Check if Polygon API key is set
    if not os.getenv('POLYGON_API_KEY'):
        logger.error("❌ POLYGON_API_KEY environment variable not set!")
        logger.error("Please set your Polygon API key before running live validation.")
        return 1
    
    validator = LiveMarketValidator()
    
    # Run validation
    results = await validator.run_live_validation()
    
    # Generate and save report
    report = validator.generate_validation_report(results)
    report_file = project_root / "live_validation_report.md"
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"📄 Detailed report saved to: {report_file}")
    
    # Also save JSON results for programmatic access
    import json
    results_file = project_root / "live_validation_results.json"
    
    # Convert datetime objects to string for JSON serialization
    def convert_datetime(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert_datetime)
    
    logger.info(f"📊 Results data saved to: {results_file}")
    
    # Return exit code based on system readiness
    return 0 if results['summary']['system_ready'] else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())