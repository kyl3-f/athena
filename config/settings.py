"""
Athena Trading Signal Detection System
Updated Database Configuration for SQLite + DBeaver
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Environment
ENVIRONMENT = os.getenv("ATHENA_ENV", "local")  # local, development, production

@dataclass
class DatabaseConfig:
    """Database configuration supporting both SQLite and PostgreSQL"""
    
    # Database type
    database_type: str = os.getenv("DATABASE_TYPE", "sqlite")
    
    # SQLite configuration (for local development with DBeaver)
    sqlite_db_path: str = os.getenv("SQLITE_DB_PATH", str(DATA_DIR / "athena.db"))
    
    # PostgreSQL configuration (for cloud deployment)
    postgres_host: Optional[str] = os.getenv("POSTGRES_HOST")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "athena")
    postgres_user: str = os.getenv("POSTGRES_USER", "athena")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    
    def get_database_url(self) -> str:
        """Get the appropriate database URL based on configuration"""
        if self.database_type == "sqlite":
            # Ensure the data directory exists
            Path(self.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{self.sqlite_db_path}"
        
        elif self.database_type == "postgresql":
            return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")
    
    def create_engine(self):
        """Create SQLAlchemy engine"""
        url = self.get_database_url()
        
        if self.database_type == "sqlite":
            # SQLite-specific configurations
            return create_engine(
                url,
                echo=False,  # Set to True for SQL debugging
                connect_args={"check_same_thread": False}  # Allow multiple threads
            )
        else:
            # PostgreSQL configurations
            return create_engine(
                url,
                echo=False,
                pool_size=10,
                max_overflow=20
            )
    
    def create_session(self):
        """Create database session"""
        engine = self.create_engine()
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        return SessionLocal()

@dataclass
class AlertConfig:
    """Alert system configuration"""
    # Email settings
    smtp_server: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    email_user: str = os.getenv("EMAIL_USER", "")
    email_password: str = os.getenv("EMAIL_PASSWORD", "")
    email_recipients: list = None
    
    def __post_init__(self):
        if self.email_recipients is None:
            recipients_str = os.getenv("EMAIL_RECIPIENTS", "")
            self.email_recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]

@dataclass
class TradingConfig:
    """Trading and model configuration"""
    # Symbols to monitor
    stock_symbols: list = None
    
    # Model settings
    model_retrain_frequency: str = "daily"  # daily, weekly, monthly
    signal_threshold: float = 0.7  # Minimum confidence for alerts
    max_positions: int = 10
    
    # Risk management
    max_position_size: float = 0.05  # 5% of portfolio
    stop_loss_pct: float = 0.02  # 2% stop loss
    
    def __post_init__(self):
        if self.stock_symbols is None:
            self.stock_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY", "QQQ"]

# Global configuration instances
db_config = DatabaseConfig()
alert_config = AlertConfig()
trading_config = TradingConfig()

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
    from sqlalchemy import MetaData
    
    engine = db_config.create_engine()
    
    # Create tables here when we define our models
    # For now, just ensure the database file exists
    if db_config.database_type == "sqlite":
        Path(db_config.sqlite_db_path).parent.mkdir(parents=True, exist_ok=True)
        # Touch the database file to create it
        engine.connect().close()
        print(f"SQLite database ready at: {db_config.sqlite_db_path}")
        print("You can connect to it in DBeaver using the path above")
    
    return engine

if __name__ == "__main__":
    # Test the database configuration
    print("Testing database configuration...")
    print(f"Database type: {db_config.database_type}")
    print(f"Database URL: {db_config.get_database_url()}")
    
    try:
        engine = setup_database()
        print("✅ Database configuration successful!")
        
        if db_config.database_type == "sqlite":
            print(f"\nTo connect in DBeaver:")
            print(f"1. Create new connection → SQLite")
            print(f"2. Path: {Path(db_config.sqlite_db_path).absolute()}")
            print(f"3. Test connection")
            
    except Exception as e:
        print(f"❌ Database configuration failed: {e}")