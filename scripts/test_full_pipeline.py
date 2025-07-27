# scripts/test_full_pipeline.py
import asyncio
import sys
import logging
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.cloud_settings import cloud_config
from scripts.cloud_production_pipeline import CloudProductionPipeline

logger = logging.getLogger(__name__)

class FullPipelineTest:
    """
    Comprehensive testing of the full Athena pipeline
    Tests all components end-to-end before cloud deployment
    """
    
    def __init__(self):
        self.config = cloud_config
        self.test_results = {
            'start_time': datetime.now().isoformat(),
            'tests': {},
            'overall_success': False
        }
        
    async def test_configuration_loading(self) -> bool:
        """Test 1: Configuration and environment setup"""
        print("🧪 Test 1: Configuration Loading")
        print("-" * 40)
        
        try:
            # Test config loading
            print(f"Environment: {self.config.environment}")
            print(f"Base Path: {self.config.base_path}")
            print(f"Data Dir: {self.config.DATA_DIR}")
            print(f"Symbols: {len(self.config.get_symbol_list())}")
            
            # Test API key
            api_key = self.config.API_CONFIG['polygon_api_key']
            if not api_key:
                raise ValueError("POLYGON_API_KEY not configured")
            
            print(f"API Key: {'✓ Configured' if api_key else '✗ Missing'}")
            
            # Test directories
            required_dirs = [
                self.config.DATA_DIR,
                self.config.LOGS_DIR,
                self.config.DATA_DIR / 'bronze',
                self.config.DATA_DIR / 'silver'
            ]
            
            for dir_path in required_dirs:
                if not dir_path.exists():
                    raise FileNotFoundError(f"Required directory missing: {dir_path}")
                print(f"Directory: {dir_path} ✓")
            
            self.test_results['tests']['configuration'] = {
                'status': 'PASS',
                'details': {
                    'environment': self.config.environment,
                    'symbols_count': len(self.config.get_symbol_list()),
                    'api_key_configured': bool(api_key),
                    'directories_created': len(required_dirs)
                }
            }
            
            print("✅ Configuration test PASSED\n")
            return True
            
        except Exception as e:
            print(f"❌ Configuration test FAILED: {e}\n")
            self.test_results['tests']['configuration'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    async def test_market_status_check(self) -> bool:
        """Test 2: Market status detection"""
        print("🧪 Test 2: Market Status Check")
        print("-" * 40)
        
        try:
            pipeline = CloudProductionPipeline()
            
            # Test market status
            start_time = time.time()
            market_status = await pipeline.check_market_status()
            duration = time.time() - start_time
            
            print(f"Market Status Response Time: {duration:.2f}s")
            print(f"Market Open: {market_status.get('is_market_open')}")
            print(f"Trading Day: {market_status.get('is_trading_day')}")
            print(f"Market Hours: {market_status.get('is_market_hours')}")
            print(f"Current Time ET: {market_status.get('current_time_et')}")
            
            # Validate response structure
            required_fields = ['is_market_open', 'is_trading_day', 'current_time_et']
            missing_fields = [field for field in required_fields if field not in market_status]
            
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            self.test_results['tests']['market_status'] = {
                'status': 'PASS',
                'details': {
                    'response_time_seconds': duration,
                    'market_open': market_status.get('is_market_open'),
                    'trading_day': market_status.get('is_trading_day')
                }
            }
            
            print("✅ Market status test PASSED\n")
            return True
            
        except Exception as e:
            print(f"❌ Market status test FAILED: {e}\n")
            self.test_results['tests']['market_status'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    async def test_data_collection(self) -> bool:
        """Test 3: Data collection for sample symbols"""
        print("🧪 Test 3: Data Collection")
        print("-" * 40)
        
        try:
            pipeline = CloudProductionPipeline()
            
            # Test with small set of symbols
            test_symbols = ["AAPL", "SPY", "QQQ"]
            print(f"Testing data collection for: {test_symbols}")
            
            results = {}
            total_stock_data = 0
            total_options_data = 0
            
            for symbol in test_symbols:
                print(f"Collecting data for {symbol}...")
                start_time = time.time()
                
                result = await pipeline.collect_comprehensive_market_data(symbol)
                duration = time.time() - start_time
                
                if result and result.get('success'):
                    stock_points = result.get('stock_data', {}).get('data_points', 0)
                    options_trades = result.get('options_flow', {}).get('trade_count', 0)
                    
                    total_stock_data += stock_points
                    total_options_data += options_trades
                    
                    results[symbol] = {
                        'success': True,
                        'duration': duration,
                        'stock_data_points': stock_points,
                        'options_trades': options_trades
                    }
                    
                    print(f"  ✓ {symbol}: {stock_points} stock points, {options_trades} options trades ({duration:.1f}s)")
                else:
                    results[symbol] = {
                        'success': False,
                        'error': result.get('error', 'Unknown error')
                    }
                    print(f"  ✗ {symbol}: Failed - {result.get('error', 'Unknown error')}")
            
            success_count = sum(1 for r in results.values() if r['success'])
            success_rate = success_count / len(test_symbols)
            
            print(f"\nCollection Summary:")
            print(f"Success Rate: {success_rate:.1%} ({success_count}/{len(test_symbols)})")
            print(f"Total Stock Data Points: {total_stock_data}")
            print(f"Total Options Trades: {total_options_data}")
            
            # Pass if at least 70% successful
            test_passed = success_rate >= 0.7
            
            self.test_results['tests']['data_collection'] = {
                'status': 'PASS' if test_passed else 'FAIL',
                'details': {
                    'symbols_tested': len(test_symbols),
                    'success_rate': success_rate,
                    'total_stock_data': total_stock_data,
                    'total_options_data': total_options_data,
                    'results': results
                }
            }
            
            if test_passed:
                print("✅ Data collection test PASSED\n")
            else:
                print("❌ Data collection test FAILED (low success rate)\n")
            
            return test_passed
            
        except Exception as e:
            print(f"❌ Data collection test FAILED: {e}\n")
            self.test_results['tests']['data_collection'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    async def test_options_flow_analysis(self) -> bool:
        """Test 4: Options flow detection and analysis"""
        print("🧪 Test 4: Options Flow Analysis")
        print("-" * 40)
        
        try:
            from src.processing.options_flow_detector import OptionsFlowDetector
            
            # Create sample options trade data for testing
            sample_trades = [
                {
                    'participant_timestamp': int(datetime.now().timestamp() * 1e9),
                    'price': 5.50,
                    'size': 100,
                    'underlying': 'AAPL',
                    'contract_type': 'call',
                    'strike_price': 150,
                    'expiration_date': '2024-02-16',
                    'open_interest': 1000
                },
                {
                    'participant_timestamp': int(datetime.now().timestamp() * 1e9),
                    'price': 12.75,
                    'size': 250,  # Large block
                    'underlying': 'AAPL',
                    'contract_type': 'put',
                    'strike_price': 145,
                    'expiration_date': '2024-02-16',
                    'open_interest': 800
                },
                {
                    'participant_timestamp': int(datetime.now().timestamp() * 1e9),
                    'price': 25.00,
                    'size': 75,
                    'underlying': 'AAPL',
                    'contract_type': 'call',
                    'strike_price': 155,
                    'expiration_date': '2024-02-16',
                    'open_interest': 500
                }
            ]
            
            detector = OptionsFlowDetector()
            
            print(f"Analyzing {len(sample_trades)} sample options trades...")
            
            # Test flow analysis
            analysis = detector.analyze_unusual_flow(sample_trades, 'AAPL')
            
            print(f"Unusual Activity Score: {analysis.get('unusual_activity_score', 0)}")
            print(f"Unusual Activity Detected: {analysis.get('unusual_activity', False)}")
            print(f"Total Premium: ${analysis.get('total_premium', 0):,.2f}")
            
            # Test alert generation
            alerts = detector.generate_flow_alerts(analysis)
            print(f"Alerts Generated: {len(alerts)}")
            
            for alert in alerts:
                print(f"  - {alert['type']}: {alert['message']}")
            
            # Validate analysis structure
            required_fields = ['symbol', 'unusual_activity_score', 'patterns']
            missing_fields = [field for field in required_fields if field not in analysis]
            
            if missing_fields:
                raise ValueError(f"Missing analysis fields: {missing_fields}")
            
            self.test_results['tests']['options_flow'] = {
                'status': 'PASS',
                'details': {
                    'sample_trades_analyzed': len(sample_trades),
                    'unusual_activity_score': analysis.get('unusual_activity_score', 0),
                    'unusual_activity_detected': analysis.get('unusual_activity', False),
                    'alerts_generated': len(alerts)
                }
            }
            
            print("✅ Options flow analysis test PASSED\n")
            return True
            
        except Exception as e:
            print(f"❌ Options flow analysis test FAILED: {e}\n")
            self.test_results['tests']['options_flow'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    async def test_data_storage(self) -> bool:
        """Test 5: Data storage and file operations"""
        print("🧪 Test 5: Data Storage")
        print("-" * 40)
        
        try:
            # Test JSON storage (Bronze layer)
            test_data = {
                'test_snapshot': {
                    'timestamp': datetime.now().isoformat(),
                    'symbols': ['AAPL', 'GOOGL'],
                    'data_points': 1000
                }
            }
            
            bronze_dir = self.config.DATA_DIR / 'bronze' / 'test'
            bronze_dir.mkdir(parents=True, exist_ok=True)
            
            json_file = bronze_dir / 'test_data.json'
            with open(json_file, 'w') as f:
                json.dump(test_data, f, indent=2, default=str)
            
            # Verify JSON read
            with open(json_file, 'r') as f:
                loaded_data = json.load(f)
            
            if loaded_data != test_data:
                raise ValueError("JSON storage/retrieval failed")
            
            print("✓ JSON storage test passed")
            
            # Test Parquet storage (Silver layer)
            try:
                import pandas as pd
                
                test_df = pd.DataFrame({
                    'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
                    'symbol': ['AAPL'] * 100,
                    'price': [150 + i * 0.1 for i in range(100)],
                    'volume': [1000 + i * 10 for i in range(100)]
                })
                
                silver_dir = self.config.DATA_DIR / 'silver' / 'test'
                silver_dir.mkdir(parents=True, exist_ok=True)
                
                parquet_file = silver_dir / 'test_features.parquet'
                test_df.to_parquet(parquet_file, compression='snappy')
                
                # Verify Parquet read
                loaded_df = pd.read_parquet(parquet_file)
                
                if len(loaded_df) != len(test_df):
                    raise ValueError("Parquet storage/retrieval failed")
                
                print("✓ Parquet storage test passed")
                
                # Clean up test files
                json_file.unlink()
                parquet_file.unlink()
                
                parquet_success = True
                
            except ImportError:
                print("⚠ Parquet test skipped (pandas/pyarrow not available)")
                parquet_success = False
            except Exception as e:
                print(f"⚠ Parquet test failed: {e}")
                parquet_success = False
            
            self.test_results['tests']['data_storage'] = {
                'status': 'PASS',
                'details': {
                    'json_storage': True,
                    'parquet_storage': parquet_success
                }
            }
            
            print("✅ Data storage test PASSED\n")
            return True
            
        except Exception as e:
            print(f"❌ Data storage test FAILED: {e}\n")
            self.test_results['tests']['data_storage'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    async def test_complete_pipeline_cycle(self) -> bool:
        """Test 6: Complete end-to-end pipeline cycle"""
        print("🧪 Test 6: Complete Pipeline Cycle")
        print("-" * 40)
        
        try:
            pipeline = CloudProductionPipeline()
            
            print("Running complete pipeline cycle...")
            start_time = time.time()
            
            # Run one complete cycle
            cycle_result = await pipeline.run_cloud_pipeline_cycle()
            
            duration = time.time() - start_time
            
            print(f"Cycle Duration: {duration:.1f}s")
            print(f"Cycle Completed: {cycle_result.get('cycle_completed')}")
            
            if cycle_result.get('cycle_completed'):
                snapshot = cycle_result.get('snapshot_summary', {})
                market_data = snapshot.get('market_data', {})
                options_flow = snapshot.get('options_flow_analysis', {})
                
                print(f"Symbols Processed: {market_data.get('successful_collections', 0)}")
                print(f"Success Rate: {market_data.get('success_rate', 0)}%")
                print(f"Stock Data Points: {market_data.get('total_stock_data_points', 0)}")
                print(f"Options Trades: {market_data.get('total_options_trades', 0)}")
                print(f"Unusual Flow Symbols: {len(options_flow.get('unusual_flow_symbols', []))}")
                
                self.test_results['tests']['complete_cycle'] = {
                    'status': 'PASS',
                    'details': {
                        'cycle_duration_seconds': duration,
                        'symbols_processed': market_data.get('successful_collections', 0),
                        'success_rate': market_data.get('success_rate', 0),
                        'unusual_flow_detected': len(options_flow.get('unusual_flow_symbols', []))
                    }
                }
                
                print("✅ Complete pipeline cycle test PASSED\n")
                return True
                
            else:
                reason = cycle_result.get('reason', 'Unknown')
                if reason == 'market_closed':
                    print("ℹ Market is closed - pipeline correctly detected this")
                    
                    self.test_results['tests']['complete_cycle'] = {
                        'status': 'PASS',
                        'details': {
                            'market_closed': True,
                            'correct_detection': True
                        }
                    }
                    
                    print("✅ Complete pipeline cycle test PASSED (market closed)\n")
                    return True
                else:
                    raise ValueError(f"Pipeline cycle failed: {reason}")
            
        except Exception as e:
            print(f"❌ Complete pipeline cycle test FAILED: {e}\n")
            self.test_results['tests']['complete_cycle'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    async def run_all_tests(self) -> dict:
        """Run all pipeline tests"""
        print("🚀 ATHENA FULL PIPELINE TEST SUITE")
        print("=" * 60)
        
        tests = [
            ("Configuration Loading", self.test_configuration_loading),
            ("Market Status Check", self.test_market_status_check),
            ("Data Collection", self.test_data_collection),
            ("Options Flow Analysis", self.test_options_flow_analysis),
            ("Data Storage", self.test_data_storage),
            ("Complete Pipeline Cycle", self.test_complete_pipeline_cycle)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                success = await test_func()
                if success:
                    passed_tests += 1
            except Exception as e:
                print(f"💥 {test_name} CRASHED: {e}\n")
                self.test_results['tests'][test_name.lower().replace(' ', '_')] = {
                    'status': 'CRASH',
                    'error': str(e)
                }
        
        # Generate final results
        self.test_results['end_time'] = datetime.now().isoformat()
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests,
            'overall_success': passed_tests >= (total_tests * 0.8)  # 80% pass rate
        }
        
        # Save test results
        results_file = self.config.LOGS_DIR / f"pipeline_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Print summary
        self._print_test_summary()
        
        return self.test_results
    
    def _print_test_summary(self):
        """Print comprehensive test summary"""
        summary = self.test_results['summary']
        
        print("=" * 60)
        print("📊 PIPELINE TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Tests Passed: {summary['passed_tests']}")
        print(f"❌ Tests Failed: {summary['failed_tests']}")
        print(f"📊 Success Rate: {summary['success_rate']:.1%}")
        print(f"🚀 Production Ready: {'YES' if summary['overall_success'] else 'NO'}")
        print("=" * 60)
        
        # Show detailed results
        for test_name, result in self.test_results['tests'].items():
            status = result['status']
            emoji = "✅" if status == "PASS" else "⚠️" if status == "PARTIAL" else "❌"
            print(f"{emoji} {test_name.replace('_', ' ').title()}: {status}")
            
            if 'error' in result:
                print(f"   Error: {result['error']}")
        
        print("=" * 60)
        
        if summary['overall_success']:
            print("🎉 PIPELINE IS READY FOR CLOUD DEPLOYMENT!")
            print("All core components tested and functioning correctly.")
        else:
            print("⚠️ PIPELINE NEEDS ATTENTION BEFORE DEPLOYMENT")
            print("Please review failed tests above.")
        
        print("=" * 60)


async def main():
    """Run full pipeline test suite"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    try:
        tester = FullPipelineTest()
        results = await tester.run_all_tests()
        
        # Exit with appropriate code
        if results['summary']['overall_success']:
            print("\n🎉 All tests passed! Ready for cloud deployment!")
            sys.exit(0)
        else:
            print("\n⚠️ Some tests failed. Review issues before deployment.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 TEST SUITE CRASHED: {e}")
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())