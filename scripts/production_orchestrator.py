#!/usr/bin/env python3
"""
Production Orchestrator for Athena Trading System
Coordinates all system components for live trading signal generation
"""

import asyncio
import logging
import os
import sys
import time
import signal
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional
import threading
import queue
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.ingestion.polygon_client import PolygonClient
from src.processing.data_cleaner import DataCleaner
from src.processing.feature_engineer import FeatureEngineer
from options_stock_integrator import OptionsStockIntegrator
from ml_signal_generator import MLSignalGenerator
from monitoring_dashboard import MonitoringDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionOrchestrator:
    """
    Orchestrates the entire Athena trading system production workflow
    """
    
    def __init__(self):
        self.running = False
        self.components = {}
        self.signal_queue = queue.Queue()
        self.alert_queue = queue.Queue()
        self.last_signal_time = None
        
        # Market hours (Eastern Time)
        self.market_open = dt_time(9, 30)  # 9:30 AM
        self.market_close = dt_time(16, 0)  # 4:00 PM
        
        # Timing configuration
        self.data_collection_interval = 15 * 60  # 15 minutes
        self.signal_generation_interval = 20 * 60  # 20 minutes
        self.model_retrain_hour = 6  # 6:00 AM daily
        
        # Alert thresholds
        self.confidence_threshold = 0.60
        self.strength_threshold = 7.0
        
        self._initialize_components()
        self._setup_signal_handlers()
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing system components...")
            
            # Data ingestion
            self.components['polygon_client'] = PolygonClient()
            self.components['data_cleaner'] = DataCleaner()
            self.components['feature_engineer'] = FeatureEngineer()
            
            # Integration and ML
            self.components['integrator'] = OptionsStockIntegrator()
            self.components['ml_generator'] = MLSignalGenerator()
            
            # Monitoring
            self.components['monitor'] = MonitoringDashboard()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
    
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours"""
        now = datetime.now().time()
        return self.market_open <= now <= self.market_close
    
    def is_trading_day(self) -> bool:
        """Check if today is a trading day (weekday)"""
        return datetime.now().weekday() < 5  # Monday = 0, Friday = 4
    
    async def collect_data_cycle(self):
        """Execute one data collection cycle"""
        try:
            logger.info("Starting data collection cycle...")
            
            # Load symbol list
            symbols_file = project_root / "config" / "symbols.txt"
            if not symbols_file.exists():
                logger.warning("Symbols file not found, running symbol loader...")
                os.system(f"python {project_root}/scripts/load_finviz_symbols.py")
            
            with open(symbols_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Collecting data for {len(symbols)} symbols")
            
            # Collect live stock data
            stock_data = await self.components['polygon_client'].fetch_live_data(symbols)
            
            # Collect options data
            options_data = await self.components['polygon_client'].fetch_options_data(symbols[:100])  # Top 100 for options
            
            # Clean and process data
            cleaned_stock = self.components['data_cleaner'].clean_stock_data(stock_data)
            cleaned_options = self.components['data_cleaner'].clean_options_data(options_data)
            
            # Feature engineering
            features = self.components['feature_engineer'].create_features(cleaned_stock, cleaned_options)
            
            logger.info(f"Data collection cycle completed. Features: {len(features)} rows")
            return features
            
        except Exception as e:
            logger.error(f"Data collection cycle failed: {e}")
            raise
    
    async def generate_signals_cycle(self):
        """Execute one signal generation cycle"""
        try:
            logger.info("Starting signal generation cycle...")
            
            # Load latest feature data
            integrator = self.components['integrator']
            ml_ready_data = integrator.prepare_ml_features()
            
            if ml_ready_data is None or len(ml_ready_data) == 0:
                logger.warning("No ML-ready data available for signal generation")
                return
            
            # Generate signals
            ml_generator = self.components['ml_generator']
            signals = ml_generator.generate_signals(ml_ready_data)
            
            if signals is not None and len(signals) > 0:
                # Filter high-confidence signals
                strong_signals = signals[
                    (signals['confidence'] >= self.confidence_threshold) &
                    (signals['strength'] >= self.strength_threshold)
                ]
                
                logger.info(f"Generated {len(signals)} signals, {len(strong_signals)} strong signals")
                
                # Queue strong signals for alerts
                for _, signal in strong_signals.iterrows():
                    self.alert_queue.put(signal.to_dict())
                
                # Save signals
                self._save_signals(signals)
                self.last_signal_time = datetime.now()
                
                return signals
            
        except Exception as e:
            logger.error(f"Signal generation cycle failed: {e}")
            raise
    
    def _save_signals(self, signals):
        """Save signals to file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        signals_dir = project_root / "data" / "signals"
        signals_dir.mkdir(parents=True, exist_ok=True)
        
        signals_file = signals_dir / f"signals_{timestamp}.csv"
        signals.write_csv(signals_file)
        
        # Also save as latest
        latest_file = signals_dir / "latest_signals.csv"
        signals.write_csv(latest_file)
        
        logger.info(f"Signals saved to {signals_file}")
    
    async def retrain_models(self):
        """Retrain ML models with latest data"""
        try:
            logger.info("Starting model retraining...")
            
            ml_generator = self.components['ml_generator']
            integrator = self.components['integrator']
            
            # Prepare training data
            training_data = integrator.prepare_training_data()
            
            if training_data is not None and len(training_data) > 1000:  # Minimum data requirement
                # Retrain models
                performance = ml_generator.train_models(training_data)
                logger.info(f"Model retraining completed. Performance: {performance}")
                
                # Save model performance
                self._save_model_performance(performance)
            else:
                logger.warning("Insufficient data for model retraining")
                
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    def _save_model_performance(self, performance):
        """Save model performance metrics"""
        perf_dir = project_root / "data" / "model_performance"
        perf_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        perf_file = perf_dir / f"performance_{timestamp}.json"
        
        import json
        with open(perf_file, 'w') as f:
            json.dump(performance, f, indent=2)
        
        logger.info(f"Model performance saved to {perf_file}")
    
    def send_alert_email(self, signal: Dict):
        """Send email alert for strong signals"""
        try:
            # Email configuration from environment
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            sender_email = os.getenv('SENDER_EMAIL')
            sender_password = os.getenv('SENDER_PASSWORD')
            recipient_email = os.getenv('RECIPIENT_EMAIL')
            
            if not all([sender_email, sender_password, recipient_email]):
                logger.warning("Email configuration incomplete, skipping alert")
                return
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = f"🚨 Athena Alert: {signal['signal']} {signal['symbol']}"
            
            # Email body
            body = f"""
            Strong Trading Signal Detected!
            
            Symbol: {signal['symbol']}
            Signal: {signal['signal']}
            Confidence: {signal['confidence']:.1%}
            Strength: {signal['strength']:.1f}/10
            
            Current Price: ${signal.get('current_price', 'N/A')}
            Target Price: ${signal.get('target_price', 'N/A')}
            
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}
            
            This is an automated alert from the Athena Trading System.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            logger.info(f"Alert email sent for {signal['symbol']} {signal['signal']}")
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
    
    async def alert_processor(self):
        """Process alerts in background"""
        while self.running:
            try:
                if not self.alert_queue.empty():
                    signal = self.alert_queue.get(timeout=1)
                    self.send_alert_email(signal)
                    self.alert_queue.task_done()
                else:
                    await asyncio.sleep(1)
            except queue.Empty:
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Alert processor error: {e}")
                await asyncio.sleep(5)
    
    async def main_loop(self):
        """Main production loop"""
        logger.info("Starting main production loop...")
        
        data_last_run = 0
        signal_last_run = 0
        retrain_last_day = 0
        
        # Start alert processor
        alert_task = asyncio.create_task(self.alert_processor())
        
        while self.running:
            try:
                current_time = time.time()
                current_hour = datetime.now().hour
                current_day = datetime.now().day
                
                # Check if we should retrain models (daily at 6 AM)
                if current_hour == self.model_retrain_hour and current_day != retrain_last_day:
                    await self.retrain_models()
                    retrain_last_day = current_day
                
                # Only run during market hours on trading days
                if not (self.is_trading_day() and self.is_market_hours()):
                    logger.info("Outside market hours, sleeping...")
                    await asyncio.sleep(60)  # Check every minute
                    continue
                
                # Data collection cycle
                if current_time - data_last_run >= self.data_collection_interval:
                    await self.collect_data_cycle()
                    data_last_run = current_time
                
                # Signal generation cycle
                if current_time - signal_last_run >= self.signal_generation_interval:
                    await self.generate_signals_cycle()
                    signal_last_run = current_time
                
                # Short sleep between checks
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
        
        # Cleanup alert processor
        alert_task.cancel()
        logger.info("Main loop stopped")
    
    def start(self):
        """Start the production system"""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("🚀 Starting Athena Trading System...")
        self.running = True
        
        # Create logs directory
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        try:
            # Run main loop
            asyncio.run(self.main_loop())
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the production system"""
        if not self.running:
            return
        
        logger.info("🛑 Stopping Athena Trading System...")
        self.running = False
        
        # Wait for alert queue to empty
        if not self.alert_queue.empty():
            logger.info("Waiting for alert queue to empty...")
            self.alert_queue.join()
        
        logger.info("System stopped successfully")
    
    def status(self) -> Dict:
        """Get system status"""
        return {
            'running': self.running,
            'market_hours': self.is_market_hours(),
            'trading_day': self.is_trading_day(),
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'alert_queue_size': self.alert_queue.qsize(),
            'components_loaded': len(self.components)
        }

def main():
    """Main entry point"""
    orchestrator = ProductionOrchestrator()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'status':
            status = orchestrator.status()
            print(f"System Status: {status}")
        elif command == 'test':
            print("Running system test...")
            # Add test functionality here
        else:
            print(f"Unknown command: {command}")
            print("Usage: python production_orchestrator.py [start|status|test]")
    else:
        # Default: start the system
        orchestrator.start()

if __name__ == "__main__":
    main()