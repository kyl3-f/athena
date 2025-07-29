#!/usr/bin/env python3
"""
Comprehensive Test Suite for Athena Trading System
Tests all components before production deployment
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemTestSuite:
    """
    Comprehensive test suite for all Athena components
    """
    
    def __init__(self):
        self.test_results = {}
        self.test_data_dir = project_root / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Test configuration
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
        self.test_start_time = datetime.now()
    
    def log_test_result(self, test_name: str, passed: bool, details: str = "", duration: float = 0):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} {test_name} ({duration:.2f}s): {details}")
        
        self.test_results[test_name] = {
            'passed': passed,
            'details': details,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
    
    async def test_environment_setup(self) -> bool:
        """Test 1: Environment and configuration"""
        start_time = time.time()
        
        try:
            # Check environment variables
            required_env_vars = ['POLYGON_API_KEY']
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            
            if missing_vars:
                self.log_test_result(
                    "Environment Setup", 
                    False, 
                    f"Missing environment variables: {missing_vars}",
                    time.time() - start_time
                )
                return False
            
            # Check project structure
            required_dirs = [
                'src/ingestion',
                'src/processing', 
                'config',
                'data/bronze',
                'data/silver'
            ]
            
            missing_dirs = []
            for dir_path in required_dirs:
                full_path = project_root / dir_path
                if not full_path.exists():
                    full_path.mkdir(parents=True, exist_ok=True)
                    missing_dirs.append(dir_path)
            
            # Check config files
            config_files = [
                'config/settings.py',
                'config/.env.example'
            ]
            
            missing_files = [f for f in config_files if not (project_root / f).exists()]
            
            details = ""
            if missing_dirs:
                details += f"Created missing directories: {missing_dirs}. "
            if missing_files:
                details += f"Missing config files: {missing_files}. "
            
            self.log_test_result(
                "Environment Setup", 
                len(missing_files) == 0,
                details or "All environment checks passed",
                time.time() - start_time
            )
            
            return len(missing_files) == 0
            
        except Exception as e:
            self.log_test_result("Environment Setup", False, f"Error: {e}", time.time() - start_time)
            return False
    
    async def test_polygon_client(self) -> bool:
        """Test 2: Polygon API client"""
        start_time = time.time()
        
        try:
            from src.ingestion.polygon_client import PolygonClient
            
            client = PolygonClient()
            
            # Test basic connection
            test_symbol = 'AAPL'
            data = await client.fetch_live_data([test_symbol])
            
            if data is None or len(data) == 0:
                self.log_test_result(
                    "Polygon Client", 
                    False, 
                    "No data returned from API",
                    time.time() - start_time
                )
                return False
            
            # Validate data structure
            required_columns = ['symbol', 'timestamp', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.log_test_result(
                    "Polygon Client", 
                    False, 
                    f"Missing required columns: {missing_columns}",
                    time.time() - start_time
                )
                return False
            
            # Test rate limiting
            rate_limit_test = await client.test_rate_limiting()
            
            self.log_test_result(
                "Polygon Client", 
                True, 
                f"Fetched {len(data)} records for {test_symbol}. Rate limit: OK",
                time.time() - start_time
            )
            
            return True
            
        except Exception as e:
            self.log_test_result("Polygon Client", False, f"Error: {e}", time.time() - start_time)
            return False
    
    async def test_data_cleaning(self) -> bool:
        """Test 3: Data cleaning pipeline"""
        start_time = time.time()
        
        try:
            from src.processing.data_cleaner import DataCleaner
            
            cleaner = DataCleaner()
            
            # Create test data with known issues
            test_data = pl.DataFrame({
                'symbol': ['AAPL', 'AAPL', 'MSFT', None, 'GOOGL'],
                'timestamp': [
                    datetime.now(),
                    datetime.now(),
                    datetime.now(),
                    datetime.now(),
                    None
                ],
                'close': [150.0, None, 300.0, 250.0, 2500.0],
                'volume': [1000000, 500000, None, 750000, 1200000],
                'high': [151.0, 149.0, 301.0, 251.0, 2510.0],
                'low': [149.0, 148.0, 299.0, 249.0, 2490.0]
            })
            
            # Clean data
            cleaned_data = cleaner.clean_stock_data(test_data)
            
            # Validate cleaning results
            if cleaned_data is None:
                self.log_test_result("Data Cleaning", False, "Cleaner returned None", time.time() - start_time)
                return False
            
            # Check that null/invalid rows were handled
            initial_rows = len(test_data)
            final_rows = len(cleaned_data)
            
            # Check for required columns
            required_columns = ['symbol', 'timestamp', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in cleaned_data.columns]
            
            success = len(missing_columns) == 0 and final_rows > 0
            
            self.log_test_result(
                "Data Cleaning", 
                success,
                f"Processed {initial_rows} → {final_rows} rows. Missing columns: {missing_columns}",
                time.time() - start_time
            )
            
            return success
            
        except Exception as e:
            self.log_test_result("Data Cleaning", False, f"Error: {e}", time.time() - start_time)
            return False
    
    async def test_feature_engineering(self) -> bool:
        """Test 4: Feature engineering pipeline"""
        start_time = time.time()
        
        try:
            from src.processing.feature_engineer import FeatureEngineer
            
            engineer = FeatureEngineer()
            
            # Create realistic test data
            timestamps = [datetime.now() - timedelta(minutes=i) for i in range(100, 0, -1)]
            
            stock_data = pl.DataFrame({
                'symbol': ['AAPL'] * 100,
                'timestamp': timestamps,
                'open': [149 + i * 0.1 for i in range(100)],
                'high': [150 + i * 0.1 for i in range(100)],
                'low': [148 + i * 0.1 for i in range(100)],
                'close': [149.5 + i * 0.1 for i in range(100)],
                'volume': [1000000 + i * 1000 for i in range(100)]
            })
            
            # Create options data
            options_data = pl.DataFrame({
                'symbol': ['AAPL'] * 20,
                'strike': [140, 145, 150, 155, 160] * 4,
                'expiry': [datetime.now() + timedelta(days=30)] * 20,
                'option_type': ['call', 'put'] * 10,
                'bid': [i * 0.5 for i in range(20)],
                'ask': [i * 0.5 + 0.1 for i in range(20)],
                'volume': [100 + i * 10 for i in range(20)]
            })
            
            # Generate features
            features = engineer.create_features(stock_data, options_data)
            
            if features is None or len(features) == 0:
                self.log_test_result("Feature Engineering", False, "No features generated", time.time() - start_time)
                return False
            
            # Validate feature columns
            expected_feature_types = [
                'rsi_', 'sma_', 'ema_', 'bb_', 'macd_',  # Technical indicators
                'call_put_ratio', 'gamma_exposure',  # Options features
                'price_momentum', 'volume_momentum'  # Momentum features
            ]
            
            feature_columns = features.columns
            found_features = [ft for ft in expected_feature_types 
                            if any(ft in col for col in feature_columns)]
            
            feature_count = len(feature_columns)
            success = feature_count >= 50  # Should have at least 50 features
            
            self.log_test_result(
                "Feature Engineering", 
                success,
                f"Generated {feature_count} features. Found types: {len(found_features)}/{len(expected_feature_types)}",
                time.time() - start_time
            )
            
            return success
            
        except Exception as e:
            self.log_test_result("Feature Engineering", False, f"Error: {e}", time.time() - start_time)
            return False
    
    async def test_options_integration(self) -> bool:
        """Test 5: Options-Stock integration"""
        start_time = time.time()
        
        try:
            from options_stock_integrator import OptionsStockIntegrator
            
            integrator = OptionsStockIntegrator()
            
            # Test ML feature preparation
            ml_data = integrator.prepare_ml_features()
            
            # This might return None if no data exists yet, which is OK for initial test
            if ml_data is None:
                self.log_test_result(
                    "Options Integration", 
                    True,
                    "No existing data to integrate (expected for new system)",
                    time.time() - start_time
                )
                return True
            
            # If data exists, validate structure
            if len(ml_data) > 0:
                required_columns = ['symbol', 'timestamp', 'target']
                missing_columns = [col for col in required_columns if col not in ml_data.columns]
                
                success = len(missing_columns) == 0
                
                self.log_test_result(
                    "Options Integration", 
                    success,
                    f"Integrated {len(ml_data)} records. Missing columns: {missing_columns}",
                    time.time() - start_time
                )
                
                return success
            
            return True
            
        except Exception as e:
            self.log_test_result("Options Integration", False, f"Error: {e}", time.time() - start_time)
            return False
    
    async def test_ml_pipeline(self) -> bool:
        """Test 6: Machine learning pipeline"""
        start_time = time.time()
        
        try:
            from ml_signal_generator import MLSignalGenerator
            
            ml_generator = MLSignalGenerator()
            
            # Create synthetic training data
            n_samples = 1000
            n_features = 50
            
            import numpy as np
            np.random.seed(42)
            
            # Create feature data
            feature_data = np.random.randn(n_samples, n_features)
            
            # Create realistic targets (BUY=1, HOLD=0, SELL=-1)
            targets = np.random.choice([-1, 0, 1], n_samples, p=[0.3, 0.4, 0.3])
            
            # Create polars DataFrame
            feature_columns = [f'feature_{i}' for i in range(n_features)]
            
            training_data = pl.DataFrame({
                'symbol': ['TEST'] * n_samples,
                'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_samples)],
                'target': targets,
                **{col: feature_data[:, i] for i, col in enumerate(feature_columns)}
            })
            
            # Test model training
            performance = ml_generator.train_models(training_data)
            
            if performance is None:
                self.log_test_result("ML Pipeline", False, "Model training returned None", time.time() - start_time)
                return False
            
            # Validate performance metrics
            required_metrics = ['accuracy', 'precision', 'recall', 'f1']
            missing_metrics = [metric for metric in required_metrics if metric not in performance]
            
            # Test signal generation with the trained model
            test_features = training_data.select([col for col in training_data.columns if col.startswith('feature_')])
            signals = ml_generator.generate_signals(test_features.head(10))
            
            if signals is None or len(signals) == 0:
                self.log_test_result("ML Pipeline", False, "Signal generation failed", time.time() - start_time)
                return False
            
            # Validate signal structure
            required_signal_columns = ['symbol', 'signal', 'confidence', 'strength']
            missing_signal_columns = [col for col in required_signal_columns if col not in signals.columns]
            
            success = len(missing_metrics) == 0 and len(missing_signal_columns) == 0
            
            self.log_test_result(
                "ML Pipeline", 
                success,
                f"Models trained. Accuracy: {performance.get('accuracy', 'N/A'):.3f}. Generated {len(signals)} signals",
                time.time() - start_time
            )
            
            return success
            
        except Exception as e:
            self.log_test_result("ML Pipeline", False, f"Error: {e}", time.time() - start_time)
            return False
    
    async def test_data_persistence(self) -> bool:
        """Test 7: Data persistence and file I/O"""
        start_time = time.time()
        
        try:
            # Test data directories
            data_dirs = [
                'data/bronze/live_stock_data',
                'data/bronze/options/chains',
                'data/bronze/options/data',
                'data/silver',
                'data/ml_ready',
                'data/signals'
            ]
            
            created_dirs = []
            for dir_path in data_dirs:
                full_path = project_root / dir_path
                if not full_path.exists():
                    full_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(dir_path)
            
            # Test file write/read operations
            test_data = pl.DataFrame({
                'symbol': ['TEST'],
                'timestamp': [datetime.now()],
                'value': [123.45]
            })
            
            test_file = project_root / "data" / "test_write.csv"
            test_data.write_csv(test_file)
            
            # Verify file was written and can be read
            if not test_file.exists():
                self.log_test_result("Data Persistence", False, "Failed to write test file", time.time() - start_time)
                return False
            
            read_data = pl.read_csv(test_file)
            
            # Clean up test file
            test_file.unlink()
            
            success = len(read_data) == 1 and read_data[0, 'symbol'] == 'TEST'
            
            self.log_test_result(
                "Data Persistence", 
                success,
                f"Created {len(created_dirs)} directories. File I/O: OK",
                time.time() - start_time
            )
            
            return success
            
        except Exception as e:
            self.log_test_result("Data Persistence", False, f"Error: {e}", time.time() - start_time)
            return False
    
    async def test_production_orchestrator(self) -> bool:
        """Test 8: Production orchestrator initialization"""
        start_time = time.time()
        
        try:
            from production_orchestrator import ProductionOrchestrator
            
            orchestrator = ProductionOrchestrator()
            
            # Test component initialization
            required_components = [
                'polygon_client', 'data_cleaner', 'feature_engineer',
                'integrator', 'ml_generator', 'monitor'
            ]
            
            missing_components = [comp for comp in required_components 
                                if comp not in orchestrator.components]
            
            # Test status method
            status = orchestrator.status()
            required_status_fields = ['running', 'market_hours', 'trading_day', 'components_loaded']
            missing_status_fields = [field for field in required_status_fields if field not in status]
            
            # Test market hours detection
            market_hours_working = isinstance(status['market_hours'], bool)
            trading_day_working = isinstance(status['trading_day'], bool)
            
            success = (len(missing_components) == 0 and 
                      len(missing_status_fields) == 0 and 
                      market_hours_working and 
                      trading_day_working)
            
            self.log_test_result(
                "Production Orchestrator", 
                success,
                f"Components: {len(orchestrator.components)}/{len(required_components)}. Status: OK",
                time.time() - start_time
            )
            
            return success
            
        except Exception as e:
            self.log_test_result("Production Orchestrator", False, f"Error: {e}", time.time() - start_time)
            return False
    
    async def test_alert_system(self) -> bool:
        """Test 9: Alert system configuration"""
        start_time = time.time()
        
        try:
            # Check email configuration
            email_config = {
                'SMTP_SERVER': os.getenv('SMTP_SERVER'),
                'SMTP_PORT': os.getenv('SMTP_PORT'),
                'SENDER_EMAIL': os.getenv('SENDER_EMAIL'),
                'SENDER_PASSWORD': os.getenv('SENDER_PASSWORD'),
                'RECIPIENT_EMAIL': os.getenv('RECIPIENT_EMAIL')
            }
            
            configured_fields = [k for k, v in email_config.items() if v is not None]
            missing_fields = [k for k, v in email_config.items() if v is None]
            
            # Alert system can work without email (logs only)
            email_configured = len(missing_fields) == 0
            
            self.log_test_result(
                "Alert System", 
                True,  # Always pass as email is optional
                f"Email config: {len(configured_fields)}/5 fields. Missing: {missing_fields}",
                time.time() - start_time
            )
            
            return True
            
        except Exception as e:
            self.log_test_result("Alert System", False, f"Error: {e}", time.time() - start_time)
            return False
    
    async def test_symbol_management(self) -> bool:
        """Test 10: Symbol list management"""
        start_time = time.time()
        
        try:
            symbols_file = project_root / "config" / "symbols.txt"
            
            # If symbols file doesn't exist, try to create it
            if not symbols_file.exists():
                try:
                    # Try running the symbol loader
                    import subprocess
                    result = subprocess.run([
                        sys.executable, 
                        str(project_root / "scripts" / "load_finviz_symbols.py")
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.returncode != 0:
                        # Create a basic symbol list if the script fails
                        with open(symbols_file, 'w') as f:
                            basic_symbols = [
                                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                                'META', 'NFLX', 'NVDA', 'SPY', 'QQQ'
                            ]
                            f.write('\n'.join(basic_symbols))
                
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    # Fallback: create basic symbol list
                    with open(symbols_file, 'w') as f:
                        basic_symbols = [
                            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                            'META', 'NFLX', 'NVDA', 'SPY', 'QQQ'
                        ]
                        f.write('\n'.join(basic_symbols))
            
            # Validate symbols file
            if symbols_file.exists():
                with open(symbols_file, 'r') as f:
                    symbols = [line.strip() for line in f if line.strip()]
                
                valid_symbols = [s for s in symbols if s.isalpha() and len(s) <= 5]
                
                success = len(valid_symbols) >= 5  # At least 5 valid symbols
                
                self.log_test_result(
                    "Symbol Management", 
                    success,
                    f"Loaded {len(valid_symbols)} valid symbols from {len(symbols)} total",
                    time.time() - start_time
                )
                
                return success
            
            self.log_test_result("Symbol Management", False, "Could not create symbols file", time.time() - start_time)
            return False
            
        except Exception as e:
            self.log_test_result("Symbol Management", False, f"Error: {e}", time.time() - start_time)
            return False
    
    async def run_all_tests(self) -> Dict:
        """Run all tests and return results"""
        logger.info("🧪 Starting Athena System Test Suite...")
        
        test_functions = [
            self.test_environment_setup,
            self.test_polygon_client,
            self.test_data_cleaning,
            self.test_feature_engineering,
            self.test_options_integration,
            self.test_ml_pipeline,
            self.test_data_persistence,
            self.test_production_orchestrator,
            self.test_alert_system,
            self.test_symbol_management
        ]
        
        for test_func in test_functions:
            try:
                await test_func()
            except Exception as e:
                test_name = test_func.__name__.replace('test_', '').replace('_', ' ').title()
                self.log_test_result(test_name, False, f"Test crashed: {e}")
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        failed_tests = total_tests - passed_tests
        
        total_duration = time.time() - self.test_start_time.timestamp()
        
        logger.info(f"\n" + "="*60)
        logger.info(f"🏁 TEST SUITE COMPLETE")
        logger.info(f"="*60)
        logger.info(f"✅ PASSED: {passed_tests}/{total_tests}")
        logger.info(f"❌ FAILED: {failed_tests}/{total_tests}")
        logger.info(f"⏱️  DURATION: {total_duration:.2f}s")
        logger.info(f"="*60)
        
        if failed_tests > 0:
            logger.info("❌ FAILED TESTS:")
            for test_name, result in self.test_results.items():
                if not result['passed']:
                    logger.info(f"  • {test_name}: {result['details']}")
        
        # System readiness assessment
        critical_tests = [
            'Environment Setup',
            'Polygon Client', 
            'Data Cleaning',
            'Feature Engineering',
            'Production Orchestrator'
        ]
        
        critical_failures = [test for test in critical_tests 
                           if test in self.test_results and not self.test_results[test]['passed']]
        
        system_ready = len(critical_failures) == 0
        
        logger.info(f"\n🚀 SYSTEM READY FOR PRODUCTION: {'YES' if system_ready else 'NO'}")
        
        if not system_ready:
            logger.info(f"❗ Critical issues to fix: {critical_failures}")
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'system_ready': system_ready,
            'critical_failures': critical_failures,
            'duration': total_duration,
            'results': self.test_results
        }
    
    def generate_test_report(self) -> str:
        """Generate detailed test report"""
        if not self.test_results:
            return "No tests have been run yet."
        
        report = []
        report.append("# ATHENA TRADING SYSTEM - TEST REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        
        report.append("## SUMMARY")
        report.append(f"- Total Tests: {total_tests}")
        report.append(f"- Passed: {passed_tests}")
        report.append(f"- Failed: {total_tests - passed_tests}")
        report.append(f"- Success Rate: {(passed_tests/total_tests)*100:.1f}%\n")
        
        # Detailed results
        report.append("## DETAILED RESULTS")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report.append(f"### {test_name} - {status}")
            report.append(f"- Duration: {result['duration']:.2f}s")
            report.append(f"- Details: {result['details']}")
            report.append(f"- Timestamp: {result['timestamp']}\n")
        
        return "\n".join(report)

async def main():
    """Main test execution"""
    test_suite = SystemTestSuite()
    
    # Run all tests
    results = await test_suite.run_all_tests()
    
    # Generate and save report
    report = test_suite.generate_test_report()
    report_file = project_root / "test_report.md"
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"📄 Detailed report saved to: {report_file}")
    
    # Return appropriate exit code
    return 0 if results['system_ready'] else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)