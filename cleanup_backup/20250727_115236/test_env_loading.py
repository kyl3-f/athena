#!/usr/bin/env python3
"""
Test environment variable loading
"""

import os
from pathlib import Path

def test_env_loading():
    """Test if .env file is loading correctly"""
    print("🔧 Testing environment variable loading...")
    
    # Check if .env file exists
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ .env file not found!")
        print("   Copy .env.example to .env and add your API key")
        return False
    
    print(f"✅ .env file found at: {env_path.absolute()}")
    
    # Try to load with python-dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        print("✅ python-dotenv loaded successfully")
    except ImportError:
        print("❌ python-dotenv not installed")
        print("   Run: pip install python-dotenv")
        return False
    
    # Check if API key is loaded
    api_key = os.getenv("POLYGON_API_KEY")
    
    if not api_key:
        print("❌ POLYGON_API_KEY not found in environment")
        print("   Check your .env file")
        return False
    
    if api_key == "your_polygon_api_key_here":
        print("❌ POLYGON_API_KEY is still the placeholder value")
        print("   Replace with your real API key from polygon.io")
        return False
    
    print(f"✅ POLYGON_API_KEY loaded (ends with: ...{api_key[-4:]})")
    
    # Test other environment variables
    env_vars = {
        "ATHENA_ENV": os.getenv("ATHENA_ENV"),
        "DATABASE_TYPE": os.getenv("DATABASE_TYPE"), 
        "SQLITE_DB_PATH": os.getenv("SQLITE_DB_PATH")
    }
    
    for var_name, var_value in env_vars.items():
        if var_value:
            print(f"✅ {var_name}: {var_value}")
        else:
            print(f"⚠️  {var_name}: not set")
    
    return True

if __name__ == "__main__":
    print("🚀 Environment Loading Test")
    print("=" * 40)
    
    success = test_env_loading()
    
    print("\n" + "=" * 40)
    
    if success:
        print("🎉 Environment loading successful!")
        print("Now try: python test_polygon.py")
    else:
        print("❌ Environment loading failed")
        print("Fix the issues above and try again")