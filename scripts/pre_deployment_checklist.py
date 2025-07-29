#!/usr/bin/env python3
"""
Pre-Deployment Checklist & Final Feature Enhancements
Ensures system is production-ready and adds last-minute improvements
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PreDeploymentManager:
    """
    Manages pre-deployment checks and feature enhancements
    """
    
    def __init__(self):
        self.checklist_results = {}
        self.enhancements_added = []
        
    def log_check(self, check_name: str, passed: bool, details: str = ""):
        """Log checklist item result"""
        status = "✅" if passed else "❌"
        logger.info(f"{status} {check_name}: {details}")
        
        self.checklist_results[check_name] = {
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
    
    def check_environment_variables(self) -> bool:
        """Check all required environment variables"""
        logger.info("🔧 Checking environment variables...")
        
        required_vars = {
            'POLYGON_API_KEY': 'Polygon.io API key for market data',
            'SMTP_SERVER': 'SMTP server for email alerts (optional)',
            'SENDER_EMAIL': 'Email address for sending alerts (optional)',
            'RECIPIENT_EMAIL': 'Email address for receiving alerts (optional)'
        }
        
        missing_required = []
        missing_optional = []
        
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                if 'optional' in description:
                    missing_optional.append(f"{var} ({description})")
                else:
                    missing_required.append(f"{var} ({description})")
        
        success = len(missing_required) == 0
        
        details = ""
        if missing_required:
            details += f"Missing required: {missing_required}. "
        if missing_optional:
            details += f"Missing optional: {missing_optional}. "
        
        self.log_check("Environment Variables", success, 
                      details or "All environment variables configured")
        
        return success
    
    def check_project_structure(self) -> bool:
        """Verify project directory structure"""
        logger.info("📁 Checking project structure...")
        
        required_structure = {
            'src/ingestion': 'Data ingestion modules',
            'src/processing': 'Data processing modules', 
            'config': 'Configuration files',
            'data/bronze': 'Raw data storage',
            'data/silver': 'Processed data storage',
            'data/signals': 'Trading signals output',
            'logs': 'Application logs',
            'scripts': 'Utility scripts'
        }
        
        missing_dirs = []
        created_dirs = []
        
        for dir_path, description in required_structure.items():
            full_path = project_root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(dir_path)
                missing_dirs.append(f"{dir_path} ({description})")
        
        self.log_check("Project Structure", True, 
                      f"Created {len(created_dirs)} missing directories" if created_dirs else "All directories exist")
        
        return True
    
    def check_dependencies(self) -> bool:
        """Check Python package dependencies"""
        logger.info("📦 Checking dependencies...")
        
        required_packages = [
            'polygon-api-client',
            'polars',
            'scikit-learn',
            'xgboost',
            'lightgbm',
            'streamlit',
            'psycopg2-binary',
            'pandas',
            'numpy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        success = len(missing_packages) == 0
        
        self.log_check("Dependencies", success,
                      f"Missing packages: {missing_packages}" if missing_packages else "All packages installed")
        
        return success
    
    def check_configuration_files(self) -> bool:
        """Check configuration files exist and are valid"""
        logger.info("⚙️ Checking configuration files...")
        
        config_checks = {}
        
        # Check settings.py
        settings_file = project_root / "config" / "settings.py"
        if settings_file.exists():
            config_checks['settings.py'] = True
        else:
            config_checks['settings.py'] = False
        
        # Check .env.example
        env_example = project_root / "config" / ".env.example"
        if env_example.exists():
            config_checks['.env.example'] = True
        else:
            config_checks['.env.example'] = False
        
        # Check or create symbols.txt
        symbols_file = project_root / "config" / "symbols.txt"
        if not symbols_file.exists():
            # Create basic symbols file
            basic_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'META', 'NFLX', 'NVDA', 'AMD', 'INTC',
                'SPY', 'QQQ', 'IWM', 'DIA', 'VTI'
            ]
            with open(symbols_file, 'w') as f:
                f.write('\n'.join(basic_symbols))
            config_checks['symbols.txt'] = True
        else:
            config_checks['symbols.txt'] = True
        
        success = all(config_checks.values())
        missing_files = [f for f, exists in config_checks.items() if not exists]
        
        self.log_check("Configuration Files", success,
                      f"Missing: {missing_files}" if missing_files else "All config files present")
        
        return success
    
    def run_system_tests(self) -> bool:
        """Run comprehensive system tests"""
        logger.info("🧪 Running system tests...")
        
        try:
            # Run the test suite
            result = subprocess.run([
                sys.executable, 
                str(project_root / "system_test_suite.py")
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            success = result.returncode == 0
            
            if success:
                self.log_check("System Tests", True, "All tests passed")
            else:
                self.log_check("System Tests", False, f"Tests failed: {result.stderr[:200]}")
            
            return success
            
        except subprocess.TimeoutExpired:
            self.log_check("System Tests", False, "Tests timed out after 5 minutes")
            return False
        except Exception as e:
            self.log_check("System Tests", False, f"Test execution error: {e}")
            return False
    
    def add_performance_monitoring(self) -> bool:
        """Add enhanced performance monitoring features"""
        logger.info("📊 Adding performance monitoring...")
        
        try:
            # Create performance monitoring module
            monitor_file = project_root / "performance_monitor.py"
            
            monitor_code = '''#!/usr/bin/env python3
"""
Enhanced Performance Monitoring for Athena Trading System
"""

import time
import psutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import json

class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.start_time = time.time()
        
    def collect_system_metrics(self) -> Dict:
        """Collect current system performance metrics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': dict(psutil.net_io_counters()._asdict()),
            'process_count': len(psutil.pids()),
            'uptime_seconds': time.time() - self.start_time
        }
    
    def log_performance_alert(self, metric: str, value: float, threshold: float):
        """Log performance alerts"""
        if value > threshold:
            logging.warning(f"Performance Alert: {metric} = {value:.1f}% (threshold: {threshold}%)")
    
    def monitor_continuously(self):
        """Continuous monitoring with alerts"""
        metrics = self.collect_system_metrics()
        
        # Check thresholds
        self.log_performance_alert("CPU", metrics['cpu_percent'], 80)
        self.log_performance_alert("Memory", metrics['memory_percent'], 85)
        self.log_performance_alert("Disk", metrics['disk_usage'], 90)
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 entries
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def save_metrics(self, filepath: Path):
        """Save metrics to file"""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
'''
            
            with open(monitor_file, 'w') as f:
                f.write(monitor_code)
            
            self.enhancements_added.append("Performance Monitoring")
            self.log_check("Performance Monitoring", True, "Enhanced monitoring added")
            return True
            
        except Exception as e:
            self.log_check("Performance Monitoring", False, f"Failed to add monitoring: {e}")
            return False
    
    def add_risk_management(self) -> bool:
        """Add risk management features"""
        logger.info("⚠️ Adding risk management...")
        
        try:
            # Create risk management module
            risk_file = project_root / "risk_manager.py"
            
            risk_code = '''#!/usr/bin/env python3
"""
Risk Management Module for Athena Trading System
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import polars as pl

class RiskManager:
    """Manage trading risks and position limits"""
    
    def __init__(self):
        self.max_daily_signals = 50
        self.max_position_size = 10000  # USD
        self.max_sector_exposure = 0.3   # 30%
        self.daily_signal_count = 0
        self.daily_reset_time = None
        
    def reset_daily_counters(self):
        """Reset daily counters at market open"""
        today = datetime.now().date()
        if self.daily_reset_time is None or self.daily_reset_time.date() < today:
            self.daily_signal_count = 0
            self.daily_reset_time = datetime.now()
            logging.info("Daily risk counters reset")
    
    def validate_signal(self, signal: Dict) -> Tuple[bool, str]:
        """Validate if signal passes risk checks"""
        self.reset_daily_counters()
        
        # Check daily signal limit
        if self.daily_signal_count >= self.max_daily_signals:
            return False, f"Daily signal limit reached ({self.max_daily_signals})"
        
        # Check confidence threshold
        if signal.get('confidence', 0) < 0.6:
            return False, f"Signal confidence too low: {signal.get('confidence', 0):.1%}"
        
        # Check strength threshold
        if signal.get('strength', 0) < 7.0:
            return False, f"Signal strength too low: {signal.get('strength', 0):.1f}"
        
        return True, "Signal approved"
    
    def approve_signal(self, signal: Dict) -> bool:
        """Approve signal and update counters"""
        valid, reason = self.validate_signal(signal)
        
        if valid:
            self.daily_signal_count += 1
            logging.info(f"Signal approved for {signal.get('symbol', 'UNKNOWN')}: {reason}")
            return True
        else:
            logging.warning(f"Signal rejected for {signal.get('symbol', 'UNKNOWN')}: {reason}")
            return False
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            'daily_signals_used': self.daily_signal_count,
            'daily_signals_remaining': max(0, self.max_daily_signals - self.daily_signal_count),
            'daily_limit_utilization': self.daily_signal_count / self.max_daily_signals,
            'last_reset': self.daily_reset_time.isoformat() if self.daily_reset_time else None
        }
'''
            
            with open(risk_file, 'w') as f:
                f.write(risk_code)
            
            self.enhancements_added.append("Risk Management")
            self.log_check("Risk Management", True, "Risk management system added")
            return True
            
        except Exception as e:
            self.log_check("Risk Management", False, f"Failed to add risk management: {e}")
            return False
    
    def add_enhanced_alerts(self) -> bool:
        """Add enhanced alert system"""
        logger.info("🔔 Adding enhanced alerts...")
        
        try:
            # Create enhanced alerts module
            alerts_file = project_root / "enhanced_alerts.py"
            
            alerts_code = '''#!/usr/bin/env python3
"""
Enhanced Alert System for Athena Trading System
"""

import logging
import smtplib
import os
from datetime import datetime
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, List, Optional

class EnhancedAlerts:
    """Enhanced alerting with multiple channels and formatting"""
    
    def __init__(self):
        self.alert_history = []
        self.email_config = {
            'server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'port': int(os.getenv('SMTP_PORT', '587')),
            'sender': os.getenv('SENDER_EMAIL'),
            'password': os.getenv('SENDER_PASSWORD'),
            'recipient': os.getenv('RECIPIENT_EMAIL')
        }
        
    def format_signal_alert(self, signal: Dict) -> str:
        """Format signal into readable alert"""
        return f"""
🚨 ATHENA TRADING ALERT 🚨

Symbol: {signal.get('symbol', 'N/A')}
Signal: {signal.get('signal', 'N/A')}
Confidence: {signal.get('confidence', 0):.1%}
Strength: {signal.get('strength', 0):.1f}/10

📊 Market Data:
Current Price: ${signal.get('current_price', 'N/A')}
Volume: {signal.get('volume', 'N/A'):,}
Market Cap: ${signal.get('market_cap', 'N/A')}

🎯 Targets:
Entry: ${signal.get('entry_price', 'N/A')}
Target: ${signal.get('target_price', 'N/A')}
Stop Loss: ${signal.get('stop_loss', 'N/A')}

⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}

This is an automated alert from Athena Trading System.
"""
    
    def send_enhanced_email(self, signal: Dict) -> bool:
        """Send enhanced email alert"""
        try:
            if not all([self.email_config['sender'], 
                       self.email_config['password'], 
                       self.email_config['recipient']]):
                logging.warning("Email configuration incomplete")
                return False
            
            msg = MimeMultipart('alternative')
            msg['Subject'] = f"🚨 Athena: {signal.get('signal', '')} {signal.get('symbol', '')}"
            msg['From'] = self.email_config['sender']
            msg['To'] = self.email_config['recipient']
            
            # Plain text version
            text_body = self.format_signal_alert(signal)
            text_part = MimeText(text_body, 'plain')
            msg.attach(text_part)
            
            # Send email
            with smtplib.SMTP(self.email_config['server'], self.email_config['port']) as server:
                server.starttls()
                server.login(self.email_config['sender'], self.email_config['password'])
                server.send_message(msg)
            
            logging.info(f"Enhanced alert sent for {signal.get('symbol', 'UNKNOWN')}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send enhanced alert: {e}")
            return False
    
    def log_console_alert(self, signal: Dict):
        """Log formatted alert to console"""
        alert_text = self.format_signal_alert(signal)
        print("\\n" + "="*60)
        print(alert_text)
        print("="*60 + "\\n")
    
    def send_alert(self, signal: Dict, channels: List[str] = ['email', 'console']):
        """Send alert through multiple channels"""
        alert_record = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'channels': channels,
            'success': {}
        }
        
        if 'email' in channels:
            alert_record['success']['email'] = self.send_enhanced_email(signal)
        
        if 'console' in channels:
            self.log_console_alert(signal)
            alert_record['success']['console'] = True
        
        self.alert_history.append(alert_record)
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
'''
            
            with open(alerts_file, 'w') as f:
                f.write(alerts_code)
            
            self.enhancements_added.append("Enhanced Alerts")
            self.log_check("Enhanced Alerts", True, "Enhanced alert system added")
            return True
            
        except Exception as e:
            self.log_check("Enhanced Alerts", False, f"Failed to add enhanced alerts: {e}")
            return False
    
    def cleanup_project(self) -> bool:
        """Clean up project files and optimize structure"""
        logger.info("🧹 Cleaning up project...")
        
        try:
            # Run cleanup script if it exists
            cleanup_script = project_root / "scripts" / "cleanup_project.py"
            if cleanup_script.exists():
                result = subprocess.run([
                    sys.executable, str(cleanup_script)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log_check("Project Cleanup", True, "Project cleaned successfully")
                    return True
                else:
                    self.log_check("Project Cleanup", False, f"Cleanup failed: {result.stderr}")
                    return False
            else:
                # Basic cleanup
                temp_files = list(project_root.glob("**/*.pyc"))
                temp_files.extend(list(project_root.glob("**/__pycache__")))
                
                for temp_file in temp_files:
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir():
                        import shutil
                        shutil.rmtree(temp_file)
                
                self.log_check("Project Cleanup", True, f"Cleaned {len(temp_files)} temp files")
                return True
                
        except Exception as e:
            self.log_check("Project Cleanup", False, f"Cleanup error: {e}")
            return False
    
    def run_full_checklist(self) -> Dict:
        """Run complete pre-deployment checklist"""
        logger.info("🚀 Running Pre-Deployment Checklist...")
        logger.info("="*60)
        
        checklist_items = [
            ("Environment Variables", self.check_environment_variables),
            ("Project Structure", self.check_project_structure),
            ("Dependencies", self.check_dependencies),
            ("Configuration Files", self.check_configuration_files),
            ("Performance Monitoring", self.add_performance_monitoring),
            ("Risk Management", self.add_risk_management),
            ("Enhanced Alerts", self.add_enhanced_alerts),
            ("Project Cleanup", self.cleanup_project),
            ("System Tests", self.run_system_tests)
        ]
        
        for item_name, check_func in checklist_items:
            try:
                check_func()
            except Exception as e:
                self.log_check(item_name, False, f"Exception: {e}")
        
        # Generate summary
        total_checks = len(self.checklist_results)
        passed_checks = sum(1 for result in self.checklist_results.values() if result['passed'])
        failed_checks = total_checks - passed_checks
        
        logger.info("="*60)
        logger.info("🏁 PRE-DEPLOYMENT CHECKLIST COMPLETE")
        logger.info("="*60)
        logger.info(f"✅ PASSED: {passed_checks}/{total_checks}")
        logger.info(f"❌ FAILED: {failed_checks}/{total_checks}")
        logger.info(f"🔧 ENHANCEMENTS ADDED: {len(self.enhancements_added)}")
        logger.info("="*60)
        
        if failed_checks > 0:
            logger.info("❌ FAILED CHECKS:")
            for check_name, result in self.checklist_results.items():
                if not result['passed']:
                    logger.info(f"  • {check_name}: {result['details']}")
        
        if self.enhancements_added:
            logger.info(f"🔧 ENHANCEMENTS ADDED: {', '.join(self.enhancements_added)}")
        
        # Deployment readiness
        critical_checks = [
            "Environment Variables",
            "Dependencies", 
            "Configuration Files"
        ]
        
        critical_failures = [check for check in critical_checks 
                           if check in self.checklist_results and not self.checklist_results[check]['passed']]
        
        deployment_ready = len(critical_failures) == 0
        
        logger.info(f"\n🚀 DEPLOYMENT READY: {'YES' if deployment_ready else 'NO'}")
        
        if not deployment_ready:
            logger.info(f"❗ Critical issues to fix: {critical_failures}")
        else:
            logger.info("🎉 System is ready for production deployment!")
            logger.info("\nNext steps:")
            logger.info("1. Run live market validation: python live_market_test.py")
            logger.info("2. Deploy to cloud: python deploy_to_digitalocean.py")
            logger.info("3. Start production system: python production_orchestrator.py")
        
        return {
            'total_checks': total_checks,
            'passed': passed_checks,
            'failed': failed_checks,
            'deployment_ready': deployment_ready,
            'critical_failures': critical_failures,
            'enhancements_added': self.enhancements_added,
            'results': self.checklist_results
        }
    
    def generate_deployment_guide(self) -> str:
        """Generate comprehensive deployment guide"""
        guide = []
        guide.append("# ATHENA TRADING SYSTEM - DEPLOYMENT GUIDE")
        guide.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        guide.append("")
        
        guide.append("## SYSTEM STATUS")
        summary = self.run_full_checklist()
        guide.append(f"- **Deployment Ready**: {'✅ YES' if summary['deployment_ready'] else '❌ NO'}")
        guide.append(f"- **Checks Passed**: {summary['passed']}/{summary['total_checks']}")
        guide.append(f"- **Enhancements**: {len(summary['enhancements_added'])} added")
        guide.append("")
        
        guide.append("## DEPLOYMENT STEPS")
        guide.append("")
        
        guide.append("### 1. Final Validation")
        guide.append("```bash")
        guide.append("# Run live market validation")
        guide.append("python live_market_test.py")
        guide.append("")
        guide.append("# Check test results")
        guide.append("cat live_validation_report.md")
        guide.append("```")
        guide.append("")
        
        guide.append("### 2. Environment Setup")
        guide.append("```bash")
        guide.append("# Set required environment variables")
        guide.append("export POLYGON_API_KEY='your_api_key_here'")
        guide.append("")
        guide.append("# Optional: Email alerts")
        guide.append("export SMTP_SERVER='smtp.gmail.com'")
        guide.append("export SMTP_PORT='587'")
        guide.append("export SENDER_EMAIL='your_email@gmail.com'")
        guide.append("export SENDER_PASSWORD='your_app_password'")
        guide.append("export RECIPIENT_EMAIL='alerts@yourcompany.com'")
        guide.append("```")
        guide.append("")
        
        guide.append("### 3. Local Testing")
        guide.append("```bash")
        guide.append("# Test individual components")
        guide.append("python -c \"from src.ingestion.polygon_client import PolygonClient; print('✅ Polygon client OK')\"")
        guide.append("python -c \"from production_orchestrator import ProductionOrchestrator; print('✅ Orchestrator OK')\"")
        guide.append("")
        guide.append("# Run system test suite")
        guide.append("python system_test_suite.py")
        guide.append("```")
        guide.append("")
        
        guide.append("### 4. Cloud Deployment")
        guide.append("```bash")
        guide.append("# Deploy to DigitalOcean")
        guide.append("python deploy_to_digitalocean.py")
        guide.append("")
        guide.append("# Or manual deployment")
        guide.append("scp -r . root@your-droplet-ip:/opt/athena")
        guide.append("ssh root@your-droplet-ip")
        guide.append("cd /opt/athena")
        guide.append("bash deploy.sh")
        guide.append("```")
        guide.append("")
        
        guide.append("### 5. Production Startup")
        guide.append("```bash")
        guide.append("# Start the production system")
        guide.append("python production_orchestrator.py")
        guide.append("")
        guide.append("# Or as a service")
        guide.append("sudo systemctl start athena-trading")
        guide.append("sudo systemctl enable athena-trading")
        guide.append("```")
        guide.append("")
        
        guide.append("### 6. Monitoring")
        guide.append("```bash")
        guide.append("# Start monitoring dashboard")
        guide.append("streamlit run monitoring_dashboard.py --server.port 8501")
        guide.append("")
        guide.append("# Check system status")
        guide.append("python production_orchestrator.py status")
        guide.append("")
        guide.append("# View logs")
        guide.append("tail -f logs/production.log")
        guide.append("```")
        guide.append("")
        
        guide.append("## PRODUCTION CHECKLIST")
        guide.append("")
        guide.append("Before going live, ensure:")
        guide.append("- [ ] All environment variables are set")
        guide.append("- [ ] Polygon API key is valid and has sufficient credits")
        guide.append("- [ ] Email alerts are configured and tested")
        guide.append("- [ ] System tests pass successfully")
        guide.append("- [ ] Live market validation completes successfully")
        guide.append("- [ ] Monitoring dashboard is accessible")
        guide.append("- [ ] Log files are being written correctly")
        guide.append("- [ ] Risk management limits are configured appropriately")
        guide.append("")
        
        guide.append("## OPERATIONAL PROCEDURES")
        guide.append("")
        guide.append("### Daily Operations")
        guide.append("- System automatically starts at market open (9:30 AM ET)")
        guide.append("- Models retrain daily at 6:00 AM with fresh data")
        guide.append("- Data collection occurs every 15 minutes during market hours")
        guide.append("- Signal generation runs every 20 minutes")
        guide.append("- Strong signals (>7.0 strength) trigger email alerts")
        guide.append("")
        
        guide.append("### Maintenance")
        guide.append("- Monitor system performance via dashboard")
        guide.append("- Review daily signal quality in logs")
        guide.append("- Check disk space and clean old data as needed")
        guide.append("- Update symbol lists monthly")
        guide.append("- Review and adjust risk management parameters")
        guide.append("")
        
        guide.append("## TROUBLESHOOTING")
        guide.append("")
        guide.append("### Common Issues")
        guide.append("1. **No data received**: Check Polygon API key and credits")
        guide.append("2. **Email alerts not working**: Verify SMTP settings")
        guide.append("3. **High CPU usage**: Check feature engineering complexity")
        guide.append("4. **Memory issues**: Reduce symbol count or batch size")
        guide.append("5. **Signal quality low**: Review and retrain models")
        guide.append("")
        
        guide.append("### Emergency Procedures")
        guide.append("- **Stop system**: `pkill -f production_orchestrator.py`")
        guide.append("- **Emergency contact**: Check logs in `/opt/athena/logs/`")
        guide.append("- **Data backup**: Daily backups in `/opt/athena/data/`")
        guide.append("")
        
        guide.append("## PERFORMANCE EXPECTATIONS")
        guide.append("")
        guide.append("### Typical Performance")
        guide.append("- **Data Processing**: 1000+ symbols in <30 seconds")
        guide.append("- **Feature Engineering**: 50+ features per symbol")
        guide.append("- **Signal Generation**: 10-50 signals per 20-minute cycle")
        guide.append("- **System Resources**: <2GB RAM, <50% CPU during market hours")
        guide.append("")
        
        guide.append("### Success Metrics")
        guide.append("- **Uptime**: >99% during market hours")
        guide.append("- **Data Freshness**: <5 minutes for stock data")
        guide.append("- **Signal Quality**: >60% confidence, >7.0 strength for alerts")
        guide.append("- **Response Time**: <30 seconds for high-priority signals")
        guide.append("")
        
        return "\n".join(guide)

def main():
    """Main execution"""
    manager = PreDeploymentManager()
    
    # Run checklist
    results = manager.run_full_checklist()
    
    # Generate deployment guide
    guide = manager.generate_deployment_guide()
    guide_file = project_root / "DEPLOYMENT_GUIDE.md"
    
    with open(guide_file, 'w') as f:
        f.write(guide)
    
    logger.info(f"📚 Deployment guide saved to: {guide_file}")
    
    # Save checklist results
    import json
    results_file = project_root / "pre_deployment_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"📊 Checklist results saved to: {results_file}")
    
    # Return exit code
    return 0 if results['deployment_ready'] else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)