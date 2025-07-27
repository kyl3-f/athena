#!/usr/bin/env python3
"""
Simple Polygon API test - just test basic connection
"""

import sys
import os

# Add src to path
sys.path.append('src')

def simple_polygon_test():
    """Test basic Polygon API connection without complex features"""
    print("🌐 Testing basic Polygon API connection...")
    
    try:
        from config.polygon_config import PolygonConfig
        from ingestion.polygon_client import PolygonClient
        
        # Create config
        config = PolygonConfig()
        
        if not config.api_key:
            print("❌ API key not configured")
            return False
        
        print(f"✅ API Key loaded (ends with: ...{config.api_key[-4:]})")
        
        # Test basic connection
        with PolygonClient(config) as client:
            print("🔗 Testing market status...")
            
            # Test 1: Market status (simple endpoint)
            try:
                status = client.get_market_status()
                print(f"✅ Market status: {status}")
                
                # Extract market state
                if isinstance(status, dict):
                    market_state = status.get('market', 'unknown')
                    print(f"📊 Current market state: {market_state}")
                
            except Exception as e:
                print(f"❌ Market status failed: {e}")
                return False
            
            print("🔗 Testing stock data...")
            
            # Test 2: Simple stock aggregates (yesterday's data)
            try:
                from datetime import datetime, timedelta
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                
                stock_df = client.get_stock_aggregates(
                    "AAPL",
                    multiplier=1,
                    timespan='day',  # Daily bars are more reliable
                    from_date=yesterday,
                    to_date=yesterday,
                    limit=1
                )
                
                if not stock_df.is_empty():
                    print(f"✅ Stock data: Retrieved {stock_df.shape[0]} AAPL daily bar(s)")
                    print(f"   Columns: {list(stock_df.columns)}")
                    
                    # Show sample data
                    if stock_df.shape[0] > 0:
                        first_row = stock_df.row(0, named=True)
                        print(f"   Sample: {first_row}")
                else:
                    print("⚠️  No stock data returned")
                
            except Exception as e:
                print(f"⚠️  Stock data test failed: {e}")
                print("   This might be normal if market is closed")
            
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Simple Polygon API Test")
    print("=" * 40)
    
    success = simple_polygon_test()
    
    print("\n" + "=" * 40)
    
    if success:
        print("🎉 Basic Polygon API connection works!")
        print("\nYour API connection is ready for:")
        print("   • Downloading historical data")
        print("   • Building trading signals")
        print("   • Real-time market monitoring")
        
        print("\nNext step: Download some historical data")
        print("Run: python scripts/download_historical.py")
        
    else:
        print("❌ Connection test failed")
        print("Check your API key and subscription status")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)