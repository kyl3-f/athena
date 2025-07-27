# Athena Trading Signal Detection System

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

*Generated on 2025-07-26 20:57:52*
