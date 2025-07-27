# deploy/cloud_setup.py
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List
import json

logger = logging.getLogger(__name__)

class CloudDeployment:
    """
    Prepare Athena system for DigitalOcean cloud deployment
    Handles path portability, environment setup, and cloud-specific configurations
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.deployment_config = {}
        
    def detect_environment(self) -> str:
        """Detect if running on cloud or local"""
        # Common cloud environment indicators
        cloud_indicators = [
            os.getenv('DIGITALOCEAN_APP_ID'),
            os.getenv('KUBERNETES_SERVICE_HOST'),
            os.getenv('DYNO'),  # Heroku
            os.getenv('AWS_EXECUTION_ENV'),
            os.path.exists('/etc/kubernetes'),
            os.path.exists('/.dockerenv')
        ]
        
        if any(cloud_indicators):
            return 'cloud'
        elif os.name == 'nt':  # Windows
            return 'windows'
        else:
            return 'unix'
    
    def create_portable_paths(self) -> Dict:
        """Create OS-agnostic path configurations"""
        
        # Use environment variables for data directories (cloud-friendly)
        base_data_dir = os.getenv('ATHENA_DATA_DIR', str(self.project_root / 'data'))
        base_logs_dir = os.getenv('ATHENA_LOGS_DIR', str(self.project_root / 'logs'))
        
        paths = {
            'project_root': str(self.project_root),
            'data_dir': base_data_dir,
            'logs_dir': base_logs_dir,
            'bronze_dir': os.path.join(base_data_dir, 'bronze'),
            'silver_dir': os.path.join(base_data_dir, 'silver'),
            'gold_dir': os.path.join(base_data_dir, 'gold'),
            'config_dir': str(self.project_root / 'config'),
            'scripts_dir': str(self.project_root / 'scripts'),
            'src_dir': str(self.project_root / 'src')
        }
        
        # Ensure all directories exist
        for dir_path in [paths['data_dir'], paths['logs_dir'], paths['bronze_dir'], 
                        paths['silver_dir'], paths['gold_dir']]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        return paths
    
    def create_cloud_config(self) -> Dict:
        """Create cloud-specific configuration"""
        environment = self.detect_environment()
        
        config = {
            'environment': environment,
            'paths': self.create_portable_paths(),
            'database': {
                'type': os.getenv('DATABASE_TYPE', 'sqlite'),
                'host': os.getenv('DATABASE_HOST', 'localhost'),
                'port': int(os.getenv('DATABASE_PORT', 5432)),
                'name': os.getenv('DATABASE_NAME', 'athena'),
                'user': os.getenv('DATABASE_USER', 'athena'),
                'password': os.getenv('DATABASE_PASSWORD', '')
            },
            'api': {
                'polygon_key': os.getenv('POLYGON_API_KEY'),
                'rate_limits': {
                    'requests_per_minute': int(os.getenv('POLYGON_RATE_LIMIT', 1000)),
                    'concurrent_requests': int(os.getenv('POLYGON_CONCURRENT', 30))
                }
            },
            'storage': {
                'use_s3': os.getenv('USE_S3_STORAGE', 'false').lower() == 'true',
                's3_bucket': os.getenv('S3_BUCKET_NAME'),
                's3_region': os.getenv('S3_REGION', 'us-east-1')
            },
            'monitoring': {
                'enable_alerts': os.getenv('ENABLE_ALERTS', 'true').lower() == 'true',
                'webhook_url': os.getenv('SLACK_WEBHOOK_URL'),
                'email_alerts': os.getenv('EMAIL_ALERTS', 'true').lower() == 'true'
            }
        }
        
        return config
    
    def generate_dockerfile(self) -> str:
        """Generate Dockerfile for containerized deployment"""
        dockerfile_content = '''# Athena Trading System - Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data and logs directories
RUN mkdir -p /app/data/bronze /app/data/silver /app/data/gold /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV ATHENA_DATA_DIR=/app/data
ENV ATHENA_LOGS_DIR=/app/logs

# Create non-root user for security
RUN useradd -m -u 1000 athena && chown -R athena:athena /app
USER athena

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Default command
CMD ["python", "scripts/production_pipeline.py"]
'''
        
        dockerfile_path = self.project_root / 'Dockerfile'
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
        
        return str(dockerfile_path)
    
    def generate_docker_compose(self) -> str:
        """Generate docker-compose.yml for local testing"""
        compose_content = '''version: '3.8'

services:
  athena-app:
    build: .
    container_name: athena-trading
    environment:
      - POLYGON_API_KEY=${POLYGON_API_KEY}
      - DATABASE_TYPE=postgresql
      - DATABASE_HOST=postgres
      - DATABASE_NAME=athena
      - DATABASE_USER=athena
      - DATABASE_PASSWORD=athena_password
      - ATHENA_DATA_DIR=/app/data
      - ATHENA_LOGS_DIR=/app/logs
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - postgres
    restart: unless-stopped
    
  postgres:
    image: postgres:15
    container_name: athena-db
    environment:
      - POSTGRES_DB=athena
      - POSTGRES_USER=athena
      - POSTGRES_PASSWORD=athena_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: athena-cache
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  postgres_data:
'''
        
        compose_path = self.project_root / 'docker-compose.yml'
        with open(compose_path, 'w') as f:
            f.write(compose_content)
        
        return str(compose_path)
    
    def create_deployment_scripts(self) -> List[str]:
        """Create deployment automation scripts"""
        
        # DigitalOcean App Platform deployment script
        deploy_script = '''#!/bin/bash
# DigitalOcean Deployment Script for Athena Trading System

echo "🚀 Deploying Athena to DigitalOcean..."

# Install doctl if not present
if ! command -v doctl &> /dev/null; then
    echo "Installing DigitalOcean CLI..."
    wget https://github.com/digitalocean/doctl/releases/download/v1.98.0/doctl-1.98.0-linux-amd64.tar.gz
    tar xf doctl-1.98.0-linux-amd64.tar.gz
    sudo mv doctl /usr/local/bin
fi

# Authenticate (requires DIGITALOCEAN_ACCESS_TOKEN environment variable)
doctl auth init --access-token $DIGITALOCEAN_ACCESS_TOKEN

# Create app
echo "Creating DigitalOcean App..."
doctl apps create app-spec.yaml

echo "✅ Deployment initiated. Check DigitalOcean dashboard for status."
'''
        
        deploy_path = self.project_root / 'deploy.sh'
        with open(deploy_path, 'w') as f:
            f.write(deploy_script)
        
        # Make executable
        import stat
        deploy_path.chmod(deploy_path.stat().st_mode | stat.S_IEXEC)
        
        # App specification for DigitalOcean
        app_spec = {
            "name": "athena-trading",
            "services": [
                {
                    "name": "athena-app",
                    "source_dir": "/",
                    "github": {
                        "repo": "your-username/athena-trading",
                        "branch": "main"
                    },
                    "run_command": "python scripts/production_pipeline.py",
                    "environment_slug": "python",
                    "instance_count": 1,
                    "instance_size_slug": "professional-xs",
                    "env": [
                        {
                            "key": "POLYGON_API_KEY",
                            "value": "${POLYGON_API_KEY}",
                            "type": "SECRET"
                        },
                        {
                            "key": "DATABASE_TYPE",
                            "value": "postgresql"
                        },
                        {
                            "key": "PYTHONPATH",
                            "value": "/app"
                        }
                    ]
                }
            ],
            "databases": [
                {
                    "name": "athena-db",
                    "engine": "PG",
                    "version": "15",
                    "size": "db-s-1vcpu-1gb"
                }
            ]
        }
        
        app_spec_path = self.project_root / 'app-spec.yaml'
        import yaml
        with open(app_spec_path, 'w') as f:
            yaml.dump(app_spec, f, default_flow_style=False)
        
        return [str(deploy_path), str(app_spec_path)]


# src/processing/options_flow_detector.py
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class OptionsFlowDetector:
    """
    Detect unusual options activity and flow patterns
    Captures large block trades, unusual volume, and institutional flow
    """
    
    def __init__(self):
        # Thresholds for unusual activity detection
        self.volume_threshold_multiplier = 3.0  # 3x average volume
        self.size_threshold = 50  # Minimum 50 contracts for "unusual" size
        self.premium_threshold = 50000  # $50k+ premium for large trades
        self.oi_ratio_threshold = 0.10  # 10% of open interest
        
    async def collect_options_trades(self, client, symbol: str, date: str) -> List[Dict]:
        """
        Collect real-time options trade data for unusual flow detection
        """
        try:
            all_trades = []
            
            # Get options contracts for today's expirations (most active)
            contracts = await client.get_options_contracts(
                underlying=symbol,
                expiration_date=date
            )
            
            if not contracts:
                return []
            
            # For each contract, get recent trade data
            for contract in contracts[:20]:  # Limit to top 20 contracts for performance
                contract_symbol = contract.get('ticker')
                if not contract_symbol:
                    continue
                
                try:
                    # Get trades for this options contract
                    url = f"{client.base_url}/v3/trades/{contract_symbol}"
                    params = {
                        "timestamp.gte": f"{date}T09:30:00Z",
                        "timestamp.lt": f"{date}T16:00:00Z",
                        "limit": 1000,
                        "apikey": client.api_key
                    }
                    
                    async with client.session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            trades = data.get('results', [])
                            
                            # Add contract metadata to each trade
                            for trade in trades:
                                trade.update({
                                    'underlying': symbol,
                                    'contract_type': contract.get('contract_type'),
                                    'strike_price': contract.get('strike_price'),
                                    'expiration_date': contract.get('expiration_date'),
                                    'open_interest': contract.get('open_interest', 0),
                                    'collection_time': datetime.now().isoformat()
                                })
                            
                            all_trades.extend(trades)
                            
                except Exception as e:
                    logger.warning(f"Error getting trades for {contract_symbol}: {e}")
                    continue
                
                # Small delay to respect rate limits
                await asyncio.sleep(0.1)
            
            logger.info(f"Collected {len(all_trades)} options trades for {symbol}")
            return all_trades
            
        except Exception as e:
            logger.error(f"Error collecting options trades for {symbol}: {e}")
            return []
    
    def analyze_unusual_flow(self, trades: List[Dict], symbol: str) -> Dict:
        """
        Analyze trades for unusual options flow patterns
        """
        if not trades:
            return {'symbol': symbol, 'unusual_activity': False}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(trades)
        
        # Calculate trade metrics
        df['premium'] = df['price'] * df['size'] * 100  # Options premium in $
        df['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns')
        
        # Detect unusual activity patterns
        unusual_patterns = {}
        
        # 1. Large Block Trades
        large_blocks = df[df['size'] >= self.size_threshold]
        unusual_patterns['large_blocks'] = {
            'count': len(large_blocks),
            'total_premium': large_blocks['premium'].sum(),
            'avg_size': large_blocks['size'].mean() if len(large_blocks) > 0 else 0,
            'trades': large_blocks.to_dict('records')[:10]  # Top 10 for storage
        }
        
        # 2. High Premium Trades
        high_premium = df[df['premium'] >= self.premium_threshold]
        unusual_patterns['high_premium_trades'] = {
            'count': len(high_premium),
            'total_premium': high_premium['premium'].sum(),
            'avg_premium': high_premium['premium'].mean() if len(high_premium) > 0 else 0,
            'trades': high_premium.to_dict('records')[:10]
        }
        
        # 3. Volume vs Open Interest Analysis
        contract_analysis = []
        for contract_group in df.groupby(['strike_price', 'contract_type', 'expiration_date']):
            group_data = contract_group[1]
            total_volume = group_data['size'].sum()
            open_interest = group_data['open_interest'].iloc[0] if len(group_data) > 0 else 1
            
            volume_oi_ratio = total_volume / max(open_interest, 1)
            
            if volume_oi_ratio >= self.oi_ratio_threshold:
                contract_analysis.append({
                    'strike_price': contract_group[0][0],
                    'contract_type': contract_group[0][1],
                    'expiration_date': contract_group[0][2],
                    'total_volume': total_volume,
                    'open_interest': open_interest,
                    'volume_oi_ratio': volume_oi_ratio,
                    'trade_count': len(group_data),
                    'avg_price': group_data['price'].mean(),
                    'total_premium': group_data['premium'].sum()
                })
        
        unusual_patterns['high_volume_contracts'] = sorted(
            contract_analysis, 
            key=lambda x: x['volume_oi_ratio'], 
            reverse=True
        )[:10]
        
        # 4. Call vs Put Flow Analysis
        call_trades = df[df['contract_type'] == 'call']
        put_trades = df[df['contract_type'] == 'put']
        
        call_volume = call_trades['size'].sum()
        put_volume = put_trades['size'].sum()
        call_premium = call_trades['premium'].sum()
        put_premium = put_trades['premium'].sum()
        
        unusual_patterns['call_put_flow'] = {
            'call_volume': call_volume,
            'put_volume': put_volume,
            'call_premium': call_premium,
            'put_premium': put_premium,
            'put_call_volume_ratio': put_volume / max(call_volume, 1),
            'put_call_premium_ratio': put_premium / max(call_premium, 1)
        }
        
        # 5. Time-based Flow Analysis
        hourly_flow = df.groupby(df['timestamp'].dt.hour).agg({
            'size': 'sum',
            'premium': 'sum',
            'price': 'mean'
        }).to_dict('index')
        
        unusual_patterns['hourly_flow'] = hourly_flow
        
        # 6. Overall Unusual Activity Score
        unusual_score = 0
        
        # Score based on large blocks
        if unusual_patterns['large_blocks']['count'] > 5:
            unusual_score += 2
        
        # Score based on high premium
        if unusual_patterns['high_premium_trades']['total_premium'] > 500000:  # $500k+
            unusual_score += 3
        
        # Score based on volume/OI ratio
        high_ratio_contracts = len([c for c in unusual_patterns['high_volume_contracts'] 
                                  if c['volume_oi_ratio'] > 0.2])
        if high_ratio_contracts > 2:
            unusual_score += 2
        
        # Score based on put/call imbalance
        put_call_ratio = unusual_patterns['call_put_flow']['put_call_premium_ratio']
        if put_call_ratio > 2.0 or put_call_ratio < 0.5:  # Significant imbalance
            unusual_score += 1
        
        return {
            'symbol': symbol,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_trades': len(df),
            'total_volume': df['size'].sum(),
            'total_premium': df['premium'].sum(),
            'unusual_activity_score': unusual_score,
            'unusual_activity': unusual_score >= 3,  # Threshold for "unusual"
            'patterns': unusual_patterns,
            'summary': {
                'large_block_count': unusual_patterns['large_blocks']['count'],
                'high_premium_count': unusual_patterns['high_premium_trades']['count'],
                'unusual_contracts': len(unusual_patterns['high_volume_contracts']),
                'put_call_ratio': unusual_patterns['call_put_flow']['put_call_premium_ratio']
            }
        }
    
    def generate_flow_alerts(self, analysis: Dict) -> List[Dict]:
        """
        Generate alerts for significant unusual flow
        """
        alerts = []
        symbol = analysis['symbol']
        score = analysis['unusual_activity_score']
        
        if score >= 5:
            alerts.append({
                'type': 'CRITICAL_FLOW',
                'symbol': symbol,
                'message': f"CRITICAL unusual options flow detected in {symbol} (Score: {score})",
                'priority': 'HIGH'
            })
        elif score >= 3:
            alerts.append({
                'type': 'UNUSUAL_FLOW',
                'symbol': symbol,
                'message': f"Unusual options activity in {symbol} (Score: {score})",
                'priority': 'MEDIUM'
            })
        
        # Specific pattern alerts
        patterns = analysis['patterns']
        
        # Large block alert
        if patterns['large_blocks']['count'] > 10:
            alerts.append({
                'type': 'LARGE_BLOCKS',
                'symbol': symbol,
                'message': f"{patterns['large_blocks']['count']} large block trades in {symbol}",
                'priority': 'MEDIUM'
            })
        
        # High premium alert
        if patterns['high_premium_trades']['total_premium'] > 1000000:  # $1M+
            alerts.append({
                'type': 'HIGH_PREMIUM',
                'symbol': symbol,
                'message': f"${patterns['high_premium_trades']['total_premium']:,.0f} in high premium trades for {symbol}",
                'priority': 'HIGH'
            })
        
        return alerts


# Update production_pipeline.py to include options flow detection
# Add this method to ProductionMarketPipeline class:

async def collect_options_flow_data(self, symbol: str) -> Optional[Dict]:
    """Collect options trade flow data for unusual activity detection"""
    try:
        async with PolygonClient(self.polygon_api_key, self.rate_limit_config) as client:
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Get options trade data
            flow_detector = OptionsFlowDetector()
            trades = await flow_detector.collect_options_trades(client, symbol, today)
            
            if trades:
                # Analyze for unusual patterns
                flow_analysis = flow_detector.analyze_unusual_flow(trades, symbol)
                
                # Generate alerts if needed
                alerts = flow_detector.generate_flow_alerts(flow_analysis)
                
                return {
                    'symbol': symbol,
                    'collection_time': datetime.now().isoformat(),
                    'trade_count': len(trades),
                    'flow_analysis': flow_analysis,
                    'alerts': alerts,
                    'raw_trades': trades  # Store for further analysis
                }
            else:
                return {
                    'symbol': symbol,
                    'collection_time': datetime.now().isoformat(),
                    'trade_count': 0,
                    'flow_analysis': None,
                    'alerts': [],
                    'raw_trades': []
                }
                
    except Exception as e:
        logger.error(f"Error collecting options flow for {symbol}: {e}")
        return None