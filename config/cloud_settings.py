# config/cloud_settings.py
import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass 
class CloudConfig:
    """Cloud-portable configuration that works across environments"""
    
    def __init__(self):
        self.environment = self._detect_environment()
        self.base_path = self._get_base_path()
        self._setup_paths()
        
    def _detect_environment(self) -> str:
        """Detect runtime environment"""
        if any([
            os.getenv('DIGITALOCEAN_APP_ID'),
            os.getenv('KUBERNETES_SERVICE_HOST'), 
            os.getenv('DYNO'),
            os.path.exists('/.dockerenv')
        ]):
            return 'cloud'
        elif os.name == 'nt':
            return 'windows'
        else:
            return 'unix'
    
    def _get_base_path(self) -> Path:
        """Get base path that works in any environment"""
        # Try environment variable first
        if base_dir := os.getenv('ATHENA_BASE_DIR'):
            return Path(base_dir)
        
        # Try current working directory
        if (Path.cwd() / 'config').exists():
            return Path.cwd()
        
        # Try script location
        if hasattr(sys, '_getframe'):
            try:
                frame = sys._getframe(1)
                script_dir = Path(frame.f_globals['__file__']).parent
                while script_dir != script_dir.parent:
                    if (script_dir / 'config').exists():
                        return script_dir
                    script_dir = script_dir.parent
            except:
                pass
        
        # Fallback to current directory
        return Path.cwd()
    
    def _setup_paths(self):
        """Setup all paths with cloud portability"""
        # Data directories (environment configurable)
        self.DATA_DIR = Path(os.getenv('ATHENA_DATA_DIR', self.base_path / 'data'))
        self.LOGS_DIR = Path(os.getenv('ATHENA_LOGS_DIR', self.base_path / 'logs'))
        self.CONFIG_DIR = Path(os.getenv('ATHENA_CONFIG_DIR', self.base_path / 'config'))
        
        # Create all required directories
        directories = [
            self.DATA_DIR,
            self.LOGS_DIR,
            self.DATA_DIR / 'bronze',
            self.DATA_DIR / 'silver', 
            self.DATA_DIR / 'gold',
            self.DATA_DIR / 'bronze' / 'market_snapshots',
            self.DATA_DIR / 'bronze' / 'options_flow',
            self.DATA_DIR / 'silver' / 'snapshots',
            self.DATA_DIR / 'silver' / 'features'
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not create directory {directory}: {e}")
    
    # Database configuration
    @property
    def DATABASE_CONFIG(self) -> Dict:
        return {
            'type': os.getenv('DATABASE_TYPE', 'sqlite'),
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': int(os.getenv('DATABASE_PORT', 5432)),
            'name': os.getenv('DATABASE_NAME', 'athena'),
            'user': os.getenv('DATABASE_USER', 'athena'),
            'password': os.getenv('DATABASE_PASSWORD', ''),
            'sqlite_path': os.getenv('SQLITE_DB_PATH', str(self.DATA_DIR / 'athena.db'))
        }
    
    # API configuration
    @property 
    def API_CONFIG(self) -> Dict:
        return {
            'polygon_api_key': os.getenv('POLYGON_API_KEY'),
            'rate_limit': {
                'requests_per_minute': int(os.getenv('POLYGON_RATE_LIMIT', 1000)),
                'concurrent_requests': int(os.getenv('POLYGON_CONCURRENT', 30)),
                'retry_attempts': int(os.getenv('POLYGON_RETRIES', 3))
            }
        }
    
    # Trading configuration
    @property
    def TRADING_CONFIG(self) -> Dict:
        return {
            'symbols_file': os.getenv('SYMBOLS_FILE', str(self.CONFIG_DIR / 'symbols.txt')),
            'test_mode': os.getenv('ATHENA_TEST_MODE', 'false').lower() == 'true',
            'max_symbols': int(os.getenv('MAX_SYMBOLS', 10000)),
            'cycle_interval_minutes': int(os.getenv('CYCLE_INTERVAL', 15))
        }
    
    # Alert configuration
    @property
    def ALERT_CONFIG(self) -> Dict:
        return {
            'enabled': os.getenv('ALERTS_ENABLED', 'true').lower() == 'true',
            'slack_webhook': os.getenv('SLACK_WEBHOOK_URL'),
            'email_enabled': os.getenv('EMAIL_ALERTS', 'false').lower() == 'true',
            'email_recipients': os.getenv('EMAIL_RECIPIENTS', '').split(','),
            'sms_enabled': os.getenv('SMS_ALERTS', 'false').lower() == 'true'
        }
    
    def get_symbol_list(self) -> list:
        """Load symbols with cloud portability"""
        symbols_file = Path(self.TRADING_CONFIG['symbols_file'])
        
        if symbols_file.exists():
            try:
                with open(symbols_file, 'r') as f:
                    symbols = [line.strip().upper() for line in f if line.strip()]
                logger.info(f"Loaded {len(symbols)} symbols from {symbols_file}")
                return symbols
            except Exception as e:
                logger.error(f"Error loading symbols from {symbols_file}: {e}")
        
        # Fallback symbols
        fallback_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ", "NVDA", "AMZN"]
        logger.warning(f"Using fallback symbols: {fallback_symbols}")
        return fallback_symbols
    
    def save_config_snapshot(self) -> str:
        """Save current configuration for debugging"""
        config_snapshot = {
            'environment': self.environment,
            'base_path': str(self.base_path),
            'data_dir': str(self.DATA_DIR),
            'logs_dir': str(self.LOGS_DIR),
            'database_config': self.DATABASE_CONFIG,
            'api_config': {k: v for k, v in self.API_CONFIG.items() if k != 'polygon_api_key'},
            'trading_config': self.TRADING_CONFIG,
            'alert_config': {k: v for k, v in self.ALERT_CONFIG.items() if 'webhook' not in k.lower()},
            'timestamp': datetime.now().isoformat()
        }
        
        snapshot_file = self.LOGS_DIR / f"config_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(snapshot_file, 'w') as f:
                json.dump(config_snapshot, f, indent=2, default=str)
            logger.info(f"Configuration snapshot saved: {snapshot_file}")
            return str(snapshot_file)
        except Exception as e:
            logger.error(f"Error saving config snapshot: {e}")
            return ""


# Create database config class for backwards compatibility
class DatabaseConfig:
    """Database configuration wrapper for backwards compatibility"""
    
    def __init__(self, config_dict: Dict):
        self.database_type = config_dict.get('type', 'sqlite')
        self.sqlite_db_path = config_dict.get('sqlite_path', 'data/athena.db')
        self.postgres_host = config_dict.get('host', 'localhost')
        self.postgres_port = config_dict.get('port', 5432)
        self.postgres_db = config_dict.get('name', 'athena')
        self.postgres_user = config_dict.get('user', 'athena')
        self.postgres_password = config_dict.get('password', '')
    
    def get_database_url(self) -> str:
        """Get database URL for SQLAlchemy"""
        if self.database_type == 'sqlite':
            Path(self.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{self.sqlite_db_path}"
        elif self.database_type == 'postgresql':
            return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")


# Create trading config class for backwards compatibility  
class TradingConfig:
    """Trading configuration wrapper for backwards compatibility"""
    
    def __init__(self, config_dict: Dict):
        self.stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ", "NVDA", "AMZN"]
        self.model_retrain_frequency = "daily"
        self.signal_threshold = 0.7
        self.max_positions = 10
        self.max_position_size = 0.05
        self.stop_loss_pct = 0.02


# Create alert config class for backwards compatibility
class AlertConfig:
    """Alert configuration wrapper for backwards compatibility"""
    
    def __init__(self, config_dict: Dict):
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.email_user = os.getenv("EMAIL_USER", "")
        self.email_password = os.getenv("EMAIL_PASSWORD", "")
        self.email_recipients = config_dict.get('email_recipients', [])


# Global cloud config instance
cloud_config = CloudConfig()

# Backwards compatibility exports
DATA_DIR = cloud_config.DATA_DIR
LOGS_DIR = cloud_config.LOGS_DIR
db_config = DatabaseConfig(cloud_config.DATABASE_CONFIG)
trading_config = TradingConfig(cloud_config.TRADING_CONFIG)
alert_config = AlertConfig(cloud_config.ALERT_CONFIG)


# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'level': 'INFO',
            'formatter': 'detailed',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': str(LOGS_DIR / 'athena.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}


def setup_database():
    """Initialize database tables"""
    try:
        from sqlalchemy import MetaData
        
        engine_url = db_config.get_database_url()
        
        if db_config.database_type == "sqlite":
            Path(db_config.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
            print(f"SQLite database ready at: {db_config.sqlite_db_path}")
            print("You can connect to it in DBeaver using the path above")
        
        return engine_url
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return None


# Test function
def test_cloud_config():
    """Test cloud configuration"""
    print("🧪 Testing Cloud Configuration")
    print("=" * 40)
    print(f"Environment: {cloud_config.environment}")
    print(f"Base Path: {cloud_config.base_path}")
    print(f"Data Directory: {cloud_config.DATA_DIR}")
    print(f"Logs Directory: {cloud_config.LOGS_DIR}")
    print(f"Symbols: {len(cloud_config.get_symbol_list())}")
    print(f"Database Type: {cloud_config.DATABASE_CONFIG['type']}")
    print(f"API Key Set: {'Yes' if cloud_config.API_CONFIG['polygon_api_key'] else 'No'}")
    
    # Test config snapshot
    snapshot_file = cloud_config.save_config_snapshot()
    if snapshot_file:
        print(f"Config Snapshot: {snapshot_file}")
    
    print("✅ Cloud configuration test complete!")


if __name__ == "__main__":
    test_cloud_config()