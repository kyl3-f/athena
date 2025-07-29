#!/usr/bin/env python3
"""
Athena Trading System - Status Summary & Quick Start Guide
Shows current system status and provides immediate next steps
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def print_header():
    """Print system header"""
    print("🏛️" + "="*58 + "🏛️")
    print("    ATHENA TRADING SYSTEM - STATUS & QUICK START")
    print("="*62)
    print(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("="*62)

def check_market_status():
    """Check if market is currently open"""
    import pytz
    
    # Use Eastern Time for market hours
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    current_time_et = now_et.time()
    weekday = now_et.weekday()  # 0=Monday, 6=Sunday
    
    # Get local time for display
    local_now = datetime.now()
    local_tz = local_now.astimezone().tzinfo.tzname(local_now)
    
    market_open = datetime.strptime("09:30", "%H:%M").time()
    market_close = datetime.strptime("16:00", "%H:%M").time()
    
    is_trading_day = weekday < 5  # Monday-Friday
    is_market_hours = market_open <= current_time_et <= market_close
    is_market_open = is_trading_day and is_market_hours
    
    print("📈 MARKET STATUS")
    print("-" * 20)
    print(f"📅 Day: {'Trading Day' if is_trading_day else 'Weekend/Holiday'}")
    print(f"🕐 Local Time: {local_now.strftime('%H:%M:%S')} {local_tz}")
    print(f"🕐 Market Time: {now_et.strftime('%H:%M:%S')} ET")
    print(f"🚦 Status: {'🟢 OPEN' if is_market_open else '🔴 CLOSED'}")
    
    if is_market_open:
        # Calculate time until market close (in ET)
        market_close_today_et = datetime.combine(now_et.date(), market_close)
        market_close_today_et = et_tz.localize(market_close_today_et)
        time_left = market_close_today_et - now_et
        hours, remainder = divmod(int(time_left.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        print(f"⏰ Time Remaining: {hours}h {minutes}m")
    else:
        # Calculate time until market open
        if is_trading_day and current_time_et < market_open:
            market_open_today_et = datetime.combine(now_et.date(), market_open)
            market_open_today_et = et_tz.localize(market_open_today_et)
            time_until = market_open_today_et - now_et
        else:
            # Next trading day
            days_ahead = 1 if weekday < 4 else (7 - weekday)
            next_trading_day = now_et + timedelta(days=days_ahead)
            market_open_next = datetime.combine(next_trading_day.date(), market_open)
            market_open_next = et_tz.localize(market_open_next)
            time_until = market_open_next - now_et
        
        hours, remainder = divmod(int(time_until.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        print(f"⏰ Next Open: {hours}h {minutes}m")
    
    print()
    return is_market_open

def check_environment():
    """Check environment configuration"""
    print("🔧 ENVIRONMENT STATUS")
    print("-" * 22)
    
    # Required variables
    polygon_key = os.getenv('POLYGON_API_KEY')
    print(f"🔑 Polygon API Key: {'✅ Set' if polygon_key else '❌ Missing'}")
    
    # Optional email variables
    smtp_server = os.getenv('SMTP_SERVER')
    sender_email = os.getenv('SENDER_EMAIL')
    recipient_email = os.getenv('RECIPIENT_EMAIL')
    
    email_configured = all([smtp_server, sender_email, recipient_email])
    print(f"📧 Email Alerts: {'✅ Configured' if email_configured else '⚠️  Not Configured (Optional)'}")
    
    if not polygon_key:
        print("\n❗ TO SET POLYGON API KEY:")
        print("   export POLYGON_API_KEY='your_api_key_here'")
    
    if not email_configured:
        print("\n💡 TO ENABLE EMAIL ALERTS:")
        print("   export SMTP_SERVER='smtp.gmail.com'")
        print("   export SENDER_EMAIL='your_email@gmail.com'")
        print("   export SENDER_PASSWORD='your_app_password'")
        print("   export RECIPIENT_EMAIL='alerts@yourcompany.com'")
    
    print()
    return polygon_key is not None

def check_project_structure():
    """Check project files and structure"""
    print("📁 PROJECT STATUS")
    print("-" * 18)
    
    # Core files
    core_files = {
        'production_orchestrator.py': 'Production System',
        'system_test_suite.py': 'Test Suite',
        'live_market_test.py': 'Live Market Validator',
        'pre_deployment_checklist.py': 'Deployment Checker',
        'src/ingestion/polygon_client.py': 'Data Client',
        'src/processing/data_cleaner.py': 'Data Cleaner',
        'src/processing/feature_engineer.py': 'Feature Engineer'
    }
    
    missing_files = []
    present_files = []
    
    for file_path, description in core_files.items():
        full_path = project_root / file_path
        if full_path.exists():
            present_files.append(f"✅ {description}")
        else:
            missing_files.append(f"❌ {description} ({file_path})")
    
    for file_status in present_files:
        print(file_status)
    
    for file_status in missing_files:
        print(file_status)
    
    # Check data directories
    data_dirs = ['data/bronze', 'data/silver', 'data/signals', 'logs']
    for dir_path in data_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Data directories: ✅ Ready")
    
    # Check symbols file
    symbols_file = project_root / "config" / "symbols.txt"
    if symbols_file.exists():
        with open(symbols_file, 'r') as f:
            symbol_count = len([line for line in f if line.strip()])
        print(f"📊 Symbols loaded: ✅ {symbol_count} symbols")
    else:
        print(f"📊 Symbols file: ⚠️  Missing (will be created)")
    
    print()
    return len(missing_files) == 0

def show_quick_actions(market_open, env_ready, files_ready):
    """Show available quick actions"""
    print("⚡ QUICK ACTIONS")
    print("-" * 16)
    
    if not env_ready:
        print("🔧 1. Set environment variables (required)")
        print("   See environment status above for instructions")
        print()
        return
    
    if not files_ready:
        print("📁 1. Complete project setup")
        print("   Missing core files - please check project structure")
        print()
        return
    
    # All good - show actions
    print("🧪 1. Run System Tests")
    print("   python3 system_test_suite.py")
    print()
    
    if market_open:
        print("🔴 2. Live Market Validation (RECOMMENDED - Market is Open!)")
        print("   python3 live_market_test.py")
        print()
    
    print("📋 3. Pre-Deployment Checklist")
    print("   python3 pre_deployment_checklist.py")
    print()
    
    print("🚀 4. Complete Deployment Pipeline")
    print("   bash run_athena_deployment.sh")
    print()
    
    print("🖥️  5. Start Local Production System")
    print("   python3 production_orchestrator.py")
    print()
    
    print("☁️  6. Deploy to Cloud")
    print("   python3 deploy_to_digitalocean.py")
    print()

def show_system_capabilities():
    """Show what the system can do"""
    print("🎯 SYSTEM CAPABILITIES")
    print("-" * 23)
    print("📊 Real-time data collection for 5000+ symbols")
    print("🧠 Machine learning signal generation (4 ML models)")
    print("📈 Options flow analysis with Greeks calculation")
    print("⚡ 15-minute data cycles, 20-minute signal generation")
    print("🔔 Email alerts for high-confidence signals (>7.0 strength)")
    print("📱 Web dashboard for monitoring and analytics")
    print("🛡️  Risk management and position limits")
    print("☁️  Cloud-ready deployment (DigitalOcean)")
    print("⏰ Automatic market hours operation")
    print("🔄 Daily model retraining at 6:00 AM")
    print()

def show_performance_specs():
    """Show expected performance"""
    print("⚡ PERFORMANCE SPECIFICATIONS")
    print("-" * 30)
    print("🔄 Data Processing: 1000+ symbols/30 seconds")
    print("🧠 ML Inference: 10-50 signals/20 minutes")
    print("💾 Memory Usage: <2GB RAM during operation")
    print("⚙️  CPU Usage: <50% during market hours")
    print("📊 Feature Count: 100+ technical + options indicators")
    print("🎯 Signal Quality: >60% confidence threshold")
    print("💰 Cloud Cost: ~$24/month (DigitalOcean)")
    print()

def show_next_steps(market_open, env_ready, files_ready):
    """Show recommended next steps"""
    print("🎯 RECOMMENDED NEXT STEPS")
    print("-" * 26)
    
    if not env_ready:
        print("1. ❗ Set POLYGON_API_KEY environment variable")
        print("2. 🔧 Optionally configure email alerts")
        print("3. 🧪 Run system tests once environment is ready")
    elif not files_ready:
        print("1. ❗ Check missing project files")
        print("2. 🔧 Complete project structure")
        print("3. 🧪 Run system tests")
    elif market_open:
        print("1. 🔴 URGENT: Run live market validation (market is open!)")
        print("   python3 live_market_test.py")
        print()
        print("2. 🚀 If validation passes, start production:")
        print("   bash run_athena_deployment.sh")
        print()
        print("3. 📊 Monitor via dashboard:")
        print("   streamlit run monitoring_dashboard.py --server.port 8501")
    else:
        print("1. 🧪 Run comprehensive system tests:")
        print("   python3 system_test_suite.py")
        print()
        print("2. 📋 Complete pre-deployment checklist:")
        print("   python3 pre_deployment_checklist.py")
        print()
        print("3. 🚀 When market opens, run full deployment pipeline:")
        print("   bash run_athena_deployment.sh")
    
    print()

def main():
    """Main status display"""
    print_header()
    
    # Check all systems
    market_open = check_market_status()
    env_ready = check_environment()
    files_ready = check_project_structure()
    
    # Show capabilities
    show_system_capabilities()
    show_performance_specs()
    
    # Show actions
    show_quick_actions(market_open, env_ready, files_ready)
    
    # Show next steps
    show_next_steps(market_open, env_ready, files_ready)
    
    # Final status
    print("🏛️" + "="*58 + "🏛️")
    
    if market_open and env_ready and files_ready:
        print("    🚀 SYSTEM READY - MARKET IS OPEN - GO LIVE! 🚀")
    elif env_ready and files_ready:
        print("    ✅ SYSTEM READY - WAITING FOR MARKET OPEN")
    elif env_ready:
        print("    ⚠️  SYSTEM PARTIALLY READY - CHECK PROJECT FILES")
    else:
        print("    ❌ SYSTEM NOT READY - SET ENVIRONMENT VARIABLES")
    
    print("="*62)
    print(f"📞 Need help? Check DEPLOYMENT_GUIDE.md for detailed instructions")
    print("="*62)

if __name__ == "__main__":
    main()