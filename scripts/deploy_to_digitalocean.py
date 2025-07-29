#!/usr/bin/env python3
"""
DigitalOcean Deployment Script for Athena Trading System
Prepares and deploys the system to DigitalOcean droplets
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DigitalOceanDeployer:
    """Handles deployment to DigitalOcean"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent  # Go up from scripts/ to project root.parent  # Go up from scripts/ to project root.parent  # Go up from scripts/ to project root.parent  # Go up from scripts/ to project root
        self.deployment_config = {
            'droplet_size': 's-2vcpu-4gb',  # $24/month
            'region': 'nyc1',  # Close to NYSE
            'image': 'ubuntu-22-04-x64',
            'ssh_keys': [],  # Add your SSH key IDs
            'tags': ['athena', 'trading', 'production']
        }
        
    def create_dockerfile(self):
        """Create optimized Dockerfile for production"""
        dockerfile_content = '''FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw/stocks data/raw/options data/ml_ready data/signals logs models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports (for dashboard if needed)
EXPOSE 8050 8501

# Health check
HEALTHCHECK --interval=5m --timeout=30s --start-period=5m --retries=3 \\
    CMD python -c "from production_orchestrator import ProductionOrchestrator; p = ProductionOrchestrator(); print(p.health_check())"

# Default command
CMD ["python", "production_orchestrator.py", "--mode", "schedule"]
'''
        
        dockerfile_path = self.project_root / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        logger.info(f"Created Dockerfile: {dockerfile_path}")
        return dockerfile_path
    
    def create_docker_compose(self):
        """Create docker-compose for local testing and production"""
        compose_content = '''version: '3.8'

services:
  athena-trading:
    build: .
    container_name: athena-production
    restart: unless-stopped
    environment:
      - POLYGON_API_KEY=${POLYGON_API_KEY}
      - EMAIL_USER=${EMAIL_USER}
      - EMAIL_PASSWORD=${EMAIL_PASSWORD}
      - ALERT_RECIPIENTS=${ALERT_RECIPIENTS}
      - SMTP_SERVER=${SMTP_SERVER:-smtp.gmail.com}
      - SMTP_PORT=${SMTP_PORT:-587}
      - DATABASE_URL=${DATABASE_URL}
      - PRODUCTION_MODE=true
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
      - ./config:/app/config
    networks:
      - athena-network
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Optional: PostgreSQL database
  athena-db:
    image: postgres:15
    container_name: athena-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=athena
      - POSTGRES_USER=athena
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - athena-network
    ports:
      - "5432:5432"

  # Optional: Redis for caching
  athena-redis:
    image: redis:7-alpine
    container_name: athena-redis
    restart: unless-stopped
    networks:
      - athena-network
    ports:
      - "6379:6379"

networks:
  athena-network:
    driver: bridge

volumes:
  postgres_data:
'''
        
        compose_path = self.project_root / "docker-compose.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        logger.info(f"Created docker-compose.yml: {compose_path}")
        return compose_path
    
    def create_env_template(self):
        """Create environment template for production"""
        env_content = '''# Athena Trading System - Production Environment

# API Keys
POLYGON_API_KEY=your_polygon_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Email Alerts
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password_here
ALERT_RECIPIENTS=recipient1@gmail.com,recipient2@gmail.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# Database
DATABASE_URL=postgresql://athena:password@athena-db:5432/athena
DB_PASSWORD=secure_database_password_here

# Trading Configuration
PRODUCTION_MODE=true
MAX_SYMBOLS=100
SIGNAL_CONFIDENCE_THRESHOLD=0.7
ALERT_STRENGTH_THRESHOLD=7.0

# Risk Management
MAX_POSITION_SIZE=10000
DAILY_LOSS_LIMIT=5000
MAX_TRADES_PER_DAY=20

# System Configuration
LOG_LEVEL=INFO
DATA_RETENTION_DAYS=90
MODEL_RETRAIN_FREQUENCY=24h

# DigitalOcean Specific
DO_TOKEN=your_digitalocean_token_here
DO_REGION=nyc1
DO_SIZE=s-2vcpu-4gb
'''
        
        env_path = self.project_root / ".env.production"
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        logger.info(f"Created .env.production template: {env_path}")
        return env_path
    
    def create_systemd_service(self):
        """Create systemd service file for production"""
        service_content = '''[Unit]
Description=Athena Trading System
After=network.target docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/athena
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
User=athena
Group=athena

[Install]
WantedBy=multi-user.target
'''
        
        service_path = self.project_root / "athena.service"
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        logger.info(f"Created systemd service: {service_path}")
        return service_path
    
    def create_deployment_script(self):
        """Create deployment script for DigitalOcean"""
        deploy_script = '''#!/bin/bash
set -e

echo "🚀 Deploying Athena Trading System to DigitalOcean"

# Configuration
PROJECT_NAME="athena-trading"
DEPLOY_USER="athena"
DEPLOY_DIR="/opt/athena"

# Update system
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
echo "📚 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create deployment user
echo "👤 Creating deployment user..."
sudo useradd -m -s /bin/bash $DEPLOY_USER || true
sudo usermod -aG docker $DEPLOY_USER

# Create deployment directory
echo "📁 Setting up deployment directory..."
sudo mkdir -p $DEPLOY_DIR
sudo chown $DEPLOY_USER:$DEPLOY_USER $DEPLOY_DIR

# Copy application files
echo "📋 Copying application files..."
sudo cp -r . $DEPLOY_DIR/
sudo chown -R $DEPLOY_USER:$DEPLOY_USER $DEPLOY_DIR

# Set up environment
echo "⚙️ Setting up environment..."
cd $DEPLOY_DIR
sudo -u $DEPLOY_USER cp .env.production .env

# Create data directories
echo "📊 Creating data directories..."
sudo -u $DEPLOY_USER mkdir -p data/raw/stocks data/raw/options data/ml_ready data/signals logs models config

# Install systemd service
echo "🔧 Installing systemd service..."
sudo cp athena.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable athena

# Build and start containers
echo "🏗️ Building and starting containers..."
cd $DEPLOY_DIR
sudo -u $DEPLOY_USER docker-compose build
sudo -u $DEPLOY_USER docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check status
echo "✅ Checking service status..."
sudo systemctl status athena --no-pager
sudo -u $DEPLOY_USER docker-compose ps

echo "🎉 Deployment completed!"
echo ""
echo "📊 Next steps:"
echo "1. Configure your .env file: sudo -u $DEPLOY_USER nano $DEPLOY_DIR/.env"
echo "2. Add your symbols: sudo -u $DEPLOY_USER nano $DEPLOY_DIR/config/symbols.txt"
echo "3. Check logs: sudo -u $DEPLOY_USER docker-compose logs -f"
echo "4. Monitor system: sudo systemctl status athena"
echo ""
echo "🔍 Useful commands:"
echo "- Restart: sudo systemctl restart athena"
echo "- View logs: sudo -u $DEPLOY_USER docker-compose logs -f athena-trading"
echo "- Update: cd $DEPLOY_DIR && sudo -u $DEPLOY_USER git pull && sudo -u $DEPLOY_USER docker-compose up --build -d"
'''
        
        script_path = self.project_root / "deploy.sh"
        with open(script_path, 'w') as f:
            f.write(deploy_script)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Created deployment script: {script_path}")
        return script_path
    
    def create_monitoring_dashboard(self):
        """Create simple monitoring dashboard"""
        dashboard_content = '''#!/usr/bin/env python3
"""
Simple monitoring dashboard for Athena system
"""

import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import json

st.set_page_config(page_title="Athena Trading Monitor", layout="wide")

st.title("🏛️ Athena Trading System Monitor")

# Sidebar
st.sidebar.header("System Status")

# Load latest signals
@st.cache_data(ttl=60)
def load_latest_signals():
    signal_files = list(Path("data/signals").glob("*.parquet"))
    if signal_files:
        latest_file = max(signal_files, key=lambda x: x.stat().st_mtime)
        return pl.read_parquet(latest_file)
    return pl.DataFrame()

# Load system health
def get_system_health():
    try:
        # You would implement actual health check here
        return "HEALTHY"
    except:
        return "UNKNOWN"

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("System Status", get_system_health())

with col2:
    signals_df = load_latest_signals()
    st.metric("Active Signals", len(signals_df) if not signals_df.is_empty() else 0)

with col3:
    if not signals_df.is_empty():
        strong_signals = signals_df.filter(pl.col("signal_strength") > 7.0)
        st.metric("Strong Signals", len(strong_signals))
    else:
        st.metric("Strong Signals", 0)

with col4:
    # Market hours indicator
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    is_market_hours = market_open <= now <= market_close and now.weekday() < 5
    
    st.metric("Market Status", "OPEN" if is_market_hours else "CLOSED")

# Recent Signals
if not signals_df.is_empty():
    st.header("📈 Recent Trading Signals")
    
    # Filter and display
    display_df = signals_df.head(20).select([
        'symbol', 'signal', 'confidence', 'signal_strength', 'reasoning'
    ])
    
    st.dataframe(display_df, use_container_width=True)
    
    # Signal distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        signal_counts = signals_df.group_by('signal').agg(pl.count().alias('count'))
        fig = px.pie(signal_counts.to_pandas(), values='count', names='signal', 
                    title="Signal Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Signal strength histogram
        fig = px.histogram(signals_df.to_pandas(), x='signal_strength', 
                          title="Signal Strength Distribution")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No recent signals found")

# System logs (last 50 lines)
st.header("📋 System Logs")
try:
    log_file = Path("logs/production_orchestrator.log")
    if log_file.exists():
        with open(log_file, 'r') as f:
            logs = f.readlines()[-50:]  # Last 50 lines
        
        st.code("".join(logs), language="text")
    else:
        st.info("No log file found")
except Exception as e:
    st.error(f"Error reading logs: {e}")

# Auto-refresh
if st.checkbox("Auto-refresh (60s)"):
    import time
    time.sleep(60)
    st.experimental_rerun()
'''
        
        dashboard_path = self.project_root / "monitoring_dashboard.py"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_content)
        
        logger.info(f"Created monitoring dashboard: {dashboard_path}")
        return dashboard_path
    
    def prepare_deployment(self):
        """Prepare all deployment files"""
        logger.info("🚀 Preparing Athena for DigitalOcean deployment...")
        
        files_created = []
        
        # Create deployment files
        files_created.append(self.create_dockerfile())
        files_created.append(self.create_docker_compose())
        files_created.append(self.create_env_template())
        files_created.append(self.create_systemd_service())
        files_created.append(self.create_deployment_script())
        files_created.append(self.create_monitoring_dashboard())
        
        logger.info("✅ Deployment preparation completed!")
        logger.info("\nFiles created:")
        for file_path in files_created:
            logger.info(f"  - {file_path}")
        
        logger.info("\n🎯 Next steps:")
        logger.info("1. Configure your .env.production file with API keys")
        logger.info("2. Add your SSH keys to DigitalOcean")
        logger.info("3. Create a DigitalOcean droplet")
        logger.info("4. Copy files to droplet and run: bash deploy.sh")
        logger.info("5. Access monitoring: streamlit run monitoring_dashboard.py")
        
        return files_created


def main():
    """Main deployment preparation"""
    deployer = DigitalOceanDeployer()
    deployer.prepare_deployment()


if __name__ == "__main__":
    main()