#!/usr/bin/env python3
"""
Athena Trading Signal Detection System
Directory Structure Setup Script

This script creates the complete directory structure for the Athena trading system,
including all necessary folders, configuration files, and starter files.

Usage:
    python setup_athena_structure.py [--project-name athena_trading]
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


def create_directory_structure(base_path: Path):
    """Create the complete directory structure for Athena"""
    
    directories = [
        # Configuration
        "config",
        
        # Data directories (local storage)
        "data/raw/options/trades",
        "data/raw/options/aggregates", 
        "data/raw/stocks/trades",
        "data/raw/stocks/aggregates",
        "data/processed/features",
        "data/processed/signals",
        "data/models/trained_models",
        "data/models/model_metrics",
        "data/models/backtest_results",
        "data/temp",
        
        # Source code
        "src/ingestion",
        "src/processing",
        "src/models",
        "src/alerts",
        "src/monitoring",
        "src/utils",
        
        # Scripts
        "scripts",
        
        # Tests
        "tests/test_ingestion",
        "tests/test_processing", 
        "tests/test_models",
        "tests/test_alerts",
        
        # Notebooks
        "notebooks",
        
        # Dashboard
        "dashboard/components",
        "dashboard/static",
        "dashboard/templates",
        
        # Docker
        "docker",
        
        # Deployment
        "deployment/digitalocean/terraform",
        "deployment/digitalocean/kubernetes",
        "deployment/digitalocean/docker-swarm",
        "deployment/monitoring/grafana",
        
        # Logs
        "logs"
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    return directories


def create_init_files(base_path: Path):
    """Create __init__.py files for Python packages"""
    
    init_locations = [
        "config/__init__.py",
        "src/__init__.py",
        "src/ingestion/__init__.py",
        "src/processing/__init__.py",
        "src/models/__init__.py",
        "src/alerts/__init__.py",
        "src/monitoring/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py",
        "tests/test_ingestion/__init__.py",
        "tests/test_processing/__init__.py",
        "tests/test_models/__init__.py",
        "tests/test_alerts/__init__.py"
    ]
    
    print("\nCreating Python package files...")
    for init_file in init_locations:
        file_path = base_path / init_file
        file_path.write_text('"""Athena Trading Signal Detection System"""\n', encoding='utf-8')
        print(f"   Created: {init_file}")


def create_config_files(base_path: Path):
    """Create configuration files"""
    
    print("\nCreating configuration files...")
    
    # Main settings file
    settings_content = '''"""
Athena Trading Signal Detection System
Main Configuration Settings
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Environment
ENVIRONMENT = os.getenv("ATHENA_ENV", "local")  # local, development, production

@dataclass
class DatabaseConfig:
    """Database configuration"""
    # Local SQLite for development
    local_db_path: str = str(DATA_DIR / "athena.db")
    
    # Cloud database settings (populate for production)
    postgres_host: Optional[str] = os.getenv("POSTGRES_HOST")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "athena")
    postgres_user: str = os.getenv("POSTGRES_USER", "athena")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")

@dataclass
class AlertConfig:
    """Alert system configuration"""
    # Email settings
    smtp_server: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    email_user: str = os.getenv("EMAIL_USER", "")
    email_password: str = os.getenv("EMAIL_PASSWORD", "")
    email_recipients: List[str] = os.getenv("EMAIL_RECIPIENTS", "").split(",")

@dataclass
class TradingConfig:
    """Trading and model configuration"""
    # Symbols to monitor
    stock_symbols: List[str] = None
    
    # Model settings
    model_retrain_frequency: str = "daily"  # daily, weekly, monthly
    signal_threshold: float = 0.7  # Minimum confidence for alerts
    max_positions: int = 10
    
    def __post_init__(self):
        if self.stock_symbols is None:
            self.stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ"]

# Global configuration instances
db_config = DatabaseConfig()
alert_config = AlertConfig()
trading_config = TradingConfig()
'''
    
    (base_path / "config/settings.py").write_text(settings_content, encoding='utf-8')
    print("   Created: config/settings.py")
    
    # Polygon configuration
    polygon_config_content = '''"""
Polygon.io API Configuration for Athena
"""

import os
from dataclasses import dataclass

@dataclass
class PolygonConfig:
    """Polygon.io API configuration"""
    api_key: str = os.getenv("POLYGON_API_KEY", "")
    base_url: str = "https://api.polygon.io"
    websocket_url: str = "wss://socket.polygon.io"
    
    # Rate limiting (adjust based on your subscription)
    max_retries: int = 3
    backoff_factor: float = 1.0
    timeout: int = 30
    rate_limit_per_minute: int = 100
    
    def __post_init__(self):
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable is required")

# Create global instance
polygon_config = PolygonConfig()
'''
    
    (base_path / "config/polygon_config.py").write_text(polygon_config_content, encoding='utf-8')
    print("   Created: config/polygon_config.py")


def create_environment_files(base_path: Path):
    """Create environment and configuration files"""
    
    print("\nCreating environment files...")
    
    # .env.example file
    env_example_content = '''# Athena Trading Signal Detection System
# Environment Configuration

# Environment
ATHENA_ENV=local

# Polygon.io API
POLYGON_API_KEY=your_polygon_api_key_here

# Database (PostgreSQL for production)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=athena
POSTGRES_USER=athena
POSTGRES_PASSWORD=your_secure_password

# Email Alerts
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_RECIPIENTS=trader1@example.com,trader2@example.com
'''
    
    (base_path / ".env.example").write_text(env_example_content, encoding='utf-8')
    print("   Created: .env.example")
    
    # .gitignore file
    gitignore_content = '''# Athena Trading System - Git Ignore

# Environment variables
.env
.env.local
.env.production

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Data directories (exclude from git)
data/
!data/.gitkeep
logs/
*.log

# Jupyter Notebooks
.ipynb_checkpoints

# Model artifacts
*.pkl
*.joblib
*.h5
*.pb

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Docker
.dockerignore

# Temporary files
*.tmp
*.temp
temp/
tmp/

# API keys and secrets
secrets/
credentials/
keys/

# Database
*.db
*.sqlite3

# Coverage reports
htmlcov/
.coverage
.pytest_cache/
.coverage.*

# MyPy
.mypy_cache/
.dmypy.json
dmypy.json
'''
    
    (base_path / ".gitignore").write_text(gitignore_content, encoding='utf-8')
    print("   Created: .gitignore")


def create_requirements_files(base_path: Path):
    """Create requirements files"""
    
    print("\nCreating requirements files...")
    
    # Main requirements
    requirements_content = '''# Athena Trading Signal Detection System
# Main Requirements

# Data processing
polars>=0.20.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=10.0.0

# API clients
requests>=2.28.0
websockets>=11.0.0
aiohttp>=3.8.0

# Machine learning
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=3.3.0

# Technical analysis
ta-lib>=0.4.25

# Database
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0

# Monitoring and logging
loguru>=0.7.0

# Alerts
twilio>=8.0.0

# Web framework (for dashboard)
streamlit>=1.28.0
plotly>=5.14.0

# Utilities
python-dotenv>=1.0.0
click>=8.1.0
pydantic>=2.0.0

# Date/time handling
pendulum>=2.1.0
pytz>=2023.3

# Configuration
pyyaml>=6.0

# HTTP client enhancements
httpx>=0.24.0
'''
    
    (base_path / "requirements.txt").write_text(requirements_content, encoding='utf-8')
    print("   Created: requirements.txt")


def create_main_files(base_path: Path):
    """Create main application files"""
    
    print("\nCreating main application files...")
    
    # README.md
    readme_content = f'''# Athena Trading Signal Detection System

**Athena** - An intelligent options and stock trading signal detection system using advanced machine learning and real-time market data analysis.

## Overview

Athena combines options flow analysis with underlying stock movements to detect profitable trading opportunities. The system ingests real-time data from Polygon.io, performs feature engineering, and uses machine learning models to identify high-probability trading signals.

## Features

- Real-time Data Ingestion: Options and stock data from Polygon.io
- ML-Powered Signals: Advanced feature engineering and model training
- Cross-Asset Analysis: Options flow correlation with underlying movements
- Smart Alerts: Email and SMS notifications for trading opportunities
- Live Dashboard: Real-time monitoring and signal visualization
- Auto-Retraining: Dynamic model updates based on performance
- Cloud-Ready: Scalable deployment on DigitalOcean

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configure API Keys

Edit `.env` file with your credentials:
- Polygon.io API key
- Email settings for alerts
- Database credentials (for production)

### 3. Download Historical Data

```bash
python scripts/download_historical.py --symbols AAPL,GOOGL,MSFT --days 30
```

### 4. Train Initial Model

```bash
python scripts/train_model.py
```

### 5. Start the System

```bash
# Local development
python scripts/run_live_system.py
```

## Directory Structure

```
athena_trading/
├── config/              # Configuration files
├── data/                # Data storage (Bronze/Silver/Gold)
├── src/                 # Source code
├── scripts/             # Operational scripts
├── dashboard/           # Web dashboard
├── docker/              # Containerization
└── deployment/          # Cloud deployment
```

## Configuration

### Trading Parameters

Edit `config/settings.py` to adjust:
- Symbols to monitor
- Signal thresholds  
- Risk management rules
- Model retraining frequency

## License

Private trading system - All rights reserved.

---

**Disclaimer**: This system is for educational and research purposes. Always perform due diligence and risk management when trading financial instruments.

*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
'''
    
    (base_path / "README.md").write_text(readme_content, encoding='utf-8')
    print("   Created: README.md")


def create_data_gitkeep_files(base_path: Path):
    """Create .gitkeep files in data directories"""
    
    print("\nCreating .gitkeep files for data directories...")
    
    data_dirs = [
        "data/raw/options/trades",
        "data/raw/options/aggregates",
        "data/raw/stocks/trades", 
        "data/raw/stocks/aggregates",
        "data/processed/features",
        "data/processed/signals",
        "data/models/trained_models",
        "data/models/model_metrics",
        "data/models/backtest_results",
        "data/temp",
        "logs"
    ]
    
    for data_dir in data_dirs:
        gitkeep_path = base_path / data_dir / ".gitkeep"
        gitkeep_path.write_text("# Keep this directory in git\n", encoding='utf-8')
        print(f"   Created: {data_dir}/.gitkeep")


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Athena Trading System directory structure")
    parser.add_argument(
        "--project-name", 
        default="athena_trading",
        help="Name of the project directory (default: athena_trading)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force creation even if directory exists"
    )
    
    args = parser.parse_args()
    
    # Create base project directory
    base_path = Path(args.project_name)
    
    if base_path.exists() and not args.force:
        print(f"Directory '{args.project_name}' already exists!")
        print("Use --force to overwrite or choose a different name.")
        sys.exit(1)
    
    print(f"Setting up Athena Trading Signal Detection System")
    print(f"Project directory: {base_path.absolute()}")
    print("=" * 60)
    
    try:
        # Create all directories
        create_directory_structure(base_path)
        
        # Create Python package files
        create_init_files(base_path)
        
        # Create configuration files
        create_config_files(base_path)
        
        # Create environment files
        create_environment_files(base_path)
        
        # Create requirements files
        create_requirements_files(base_path)
        
        # Create main application files
        create_main_files(base_path)
        
        # Create .gitkeep files for data directories
        create_data_gitkeep_files(base_path)
        
        print("\n" + "=" * 60)
        print("Athena Trading System setup completed successfully!")
        print("\nNext Steps:")
        print(f"   1. cd {args.project_name}")
        print("   2. cp .env.example .env")
        print("   3. Edit .env with your API keys and settings")
        print("   4. pip install -r requirements.txt")
        print("   5. python scripts/download_historical.py --help")
        print("\nDocumentation:")
        print("   • README.md - Full setup guide")
        print("   • config/ - Configuration options")
        print("   • .env.example - Environment variables")
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()