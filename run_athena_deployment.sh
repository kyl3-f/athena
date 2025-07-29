#!/bin/bash

# Athena Trading System - Complete Deployment Runner
# Runs all necessary steps to get the system production-ready

set -e  # Exit on any error

echo "🚀 ATHENA TRADING SYSTEM - DEPLOYMENT PIPELINE"
echo "=============================================="
echo "Starting deployment at $(date)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

print_success "Python 3 found: $(python3 --version)"

# Check if required environment variables are set
print_status "Checking environment variables..."

if [ -z "$POLYGON_API_KEY" ]; then
    print_error "POLYGON_API_KEY environment variable is not set!"
    echo "Please set your Polygon API key:"
    echo "export POLYGON_API_KEY='your_api_key_here'"
    exit 1
fi

print_success "Polygon API key is set"

# Optional email configuration check
if [ -z "$SENDER_EMAIL" ] || [ -z "$RECIPIENT_EMAIL" ]; then
    print_warning "Email configuration not complete - alerts will be console-only"
else
    print_success "Email configuration found"
fi

echo ""
print_status "=== STEP 1: PRE-DEPLOYMENT CHECKLIST ==="

if python3 pre_deployment_checklist.py; then
    print_success "Pre-deployment checklist passed"
else
    print_error "Pre-deployment checklist failed"
    exit 1
fi

echo ""
print_status "=== STEP 2: SYSTEM TESTS ==="

if python3 system_test_suite.py; then
    print_success "System tests passed"
else
    print_error "System tests failed"
    exit 1
fi

echo ""
print_status "=== STEP 3: LIVE MARKET VALIDATION ==="

# Check if market is open (basic check)
current_hour=$(date +%H)
current_day=$(date +%u)  # 1-7, Monday is 1

if [ "$current_day" -le 5 ] && [ "$current_hour" -ge 9 ] && [ "$current_hour" -lt 16 ]; then
    print_status "Market appears to be open - running live validation"
    
    if python3 live_market_test.py; then
        print_success "Live market validation passed"
    else
        print_warning "Live market validation had issues - check live_validation_report.md"
    fi
else
    print_warning "Market appears to be closed - skipping live validation"
    print_status "You can run 'python3 live_market_test.py' manually when market is open"
fi

echo ""
print_status "=== STEP 4: FINAL PREPARATION ==="

# Create necessary directories
mkdir -p logs data/bronze data/silver data/signals

# Generate symbols list if not exists
if [ ! -f "config/symbols.txt" ]; then
    print_status "Generating symbols list..."
    if [ -f "scripts/load_finviz_symbols.py" ]; then
        python3 scripts/load_finviz_symbols.py || print_warning "Symbol generation failed - using defaults"
    fi
fi

print_success "Project structure ready"

echo ""
print_status "=== STEP 5: PRODUCTION READINESS CHECK ==="

# Test import of main components
python3 -c "
try:
    from production_orchestrator import ProductionOrchestrator
    from src.ingestion.polygon_client import PolygonClient
    from src.processing.data_cleaner import DataCleaner
    from src.processing.feature_engineer import FeatureEngineer
    print('✅ All main components import successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    print_success "All components ready"
else
    print_error "Component import failed"
    exit 1
fi

echo ""
echo "=============================================="
print_success "🎉 ATHENA DEPLOYMENT PIPELINE COMPLETE!"
echo "=============================================="
echo ""

print_status "DEPLOYMENT OPTIONS:"
echo ""

echo "🖥️  LOCAL TESTING:"
echo "   python3 production_orchestrator.py"
echo ""

echo "☁️  CLOUD DEPLOYMENT:"
echo "   python3 deploy_to_digitalocean.py"
echo ""

echo "📊 MONITORING:"
echo "   streamlit run monitoring_dashboard.py --server.port 8501"
echo ""

echo "📋 STATUS CHECK:"
echo "   python3 production_orchestrator.py status"
echo ""

print_status "IMPORTANT FILES GENERATED:"
echo "   📄 DEPLOYMENT_GUIDE.md - Complete deployment guide"
echo "   📄 live_validation_report.md - Live market test results"
echo "   📄 test_report.md - System test results"
echo ""

print_warning "NEXT STEPS:"
echo "1. Review all generated reports"
echo "2. Choose deployment method (local or cloud)"
echo "3. Start the monitoring dashboard"
echo "4. Begin live trading signal generation!"
echo ""

print_status "🚀 Ready for production at $(date)"

# Ask user what they want to do next
echo ""
read -p "Do you want to start the system locally now? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Starting Athena Trading System locally..."
    echo ""
    print_warning "Press Ctrl+C to stop the system"
    echo ""
    python3 production_orchestrator.py
else
    print_success "Deployment pipeline complete. Start the system when ready!"
fi