"""
Polygon.io API Configuration for Athena
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    
    # Find and load .env file
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from: {env_path}")
    else:
        print(f"Warning: .env file not found at {env_path}")
        
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")

@dataclass
class PolygonConfig:
    """Polygon.io API configuration"""
    api_key: str = ""
    base_url: str = "https://api.polygon.io"
    websocket_url: str = "wss://socket.polygon.io"
    
    # Rate limiting (adjust based on your subscription)
    max_retries: int = 3
    backoff_factor: float = 1.0
    timeout: int = 30
    rate_limit_per_minute: int = 100
    
    # Data preferences
    adjusted: bool = True
    sort: str = "asc"
    limit: int = 50000
    
    def __post_init__(self):
        # Load API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("POLYGON_API_KEY", "")
        
        # Validate API key
        if not self.api_key or self.api_key == "your_polygon_api_key_here":
            print("\n❌ POLYGON_API_KEY not found!")
            print("Please add your API key to the .env file:")
            print("POLYGON_API_KEY=your_actual_api_key_here")
            print("\nGet your API key from: https://polygon.io/dashboard")

# Test configuration when module is imported
def test_config():
    """Test if configuration is working"""
    config = PolygonConfig()
    
    if config.api_key and config.api_key != "your_polygon_api_key_here":
        print(f"✅ Polygon API key configured (ends with: ...{config.api_key[-4:]})")
        return True
    else:
        return False

# Only create global instance for backwards compatibility
# Don't raise error on import - let calling code handle it
try:
    polygon_config = PolygonConfig()
except Exception as e:
    print(f"Warning: Could not create polygon_config: {e}")
    polygon_config = None