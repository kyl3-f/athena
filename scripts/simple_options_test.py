#!/usr/bin/env python3
"""
Standalone Options Test
Tests options data directly with Polygon API using requests
"""

import os
import requests
import polars as pl
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment
load_dotenv()

def test_polygon_options_direct():
    """Test Polygon options API directly"""
    
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("❌ POLYGON_API_KEY not found in environment")
        return
    
    print("🚀 Direct Polygon Options API Test")
    print("=" * 40)
    print(f"✅ API Key loaded: ...{api_key[-4:]}")
    
    # Setup session
    session = requests.Session()
    session.headers.update({
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'Athena-Options-Test/1.0'
    })
    
    base_url = "https://api.polygon.io"
    
    try:
        # Test 1: Get options contracts for AAPL
        print("\n📋 Testing options contracts for AAPL...")
        
        contracts_url = f"{base_url}/v3/reference/options/contracts"
        params = {
            'underlying_ticker': 'AAPL',
            'limit': 20
        }
        
        response = session.get(contracts_url, params=params)
        response.raise_for_status()
        
        contracts_data = response.json()
        print(f"✅ API Response Status: {response.status_code}")
        print(f"   Response keys: {list(contracts_data.keys())}")
        
        if 'results' in contracts_data and contracts_data['results']:
            contracts = contracts_data['results']
            print(f"✅ Found {len(contracts)} AAPL contracts")
            
            # Show first contract structure
            first_contract = contracts[0]
            print(f"\n📊 First contract structure:")
            print(f"   Keys: {list(first_contract.keys())}")
            print(f"   Sample contract: {first_contract}")
            
            # Test 2: Try to get data for one contract
            if 'ticker' in first_contract:
                contract_ticker = first_contract['ticker']
                print(f"\n📈 Testing options data for: {contract_ticker}")
                
                # Get last 3 days
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
                
                # Options aggregates endpoint
                aggs_url = f"{base_url}/v2/aggs/ticker/{contract_ticker}/range/1/minute/{start_date}/{end_date}"
                aggs_params = {
                    'adjusted': 'true',
                    'sort': 'asc',
                    'limit': 50
                }
                
                try:
                    aggs_response = session.get(aggs_url, params=aggs_params)
                    aggs_response.raise_for_status()
                    
                    aggs_data = aggs_response.json()
                    print(f"✅ Options data response: {aggs_response.status_code}")
                    print(f"   Response keys: {list(aggs_data.keys())}")
                    
                    if 'results' in aggs_data and aggs_data['results']:
                        results = aggs_data['results']
                        print(f"✅ Found {len(results)} minute bars")
                        print(f"   Sample bar: {results[0]}")
                        
                        # Convert to DataFrame
                        df = pl.DataFrame(results)
                        print(f"   DataFrame shape: {df.shape}")
                        print(f"   Columns: {list(df.columns)}")
                        
                    else:
                        print("⚠️  No options data in results")
                        
                except Exception as e:
                    print(f"⚠️  Options data request failed: {e}")
        else:
            print("❌ No contracts in response results")
        
        # Test 3: Market calendar
        print("\n📅 Testing market calendar...")
        
        today = datetime.now().strftime('%Y-%m-%d')
        future = (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
        
        calendar_url = f"{base_url}/v1/marketstatus/upcoming"
        calendar_params = {
            'start': today,
            'end': future
        }
        
        try:
            cal_response = session.get(calendar_url, params=calendar_params)
            cal_response.raise_for_status()
            
            cal_data = cal_response.json()
            print(f"✅ Calendar response: {cal_response.status_code}")
            print(f"   Calendar data type: {type(cal_data)}")
            print(f"   Calendar sample: {cal_data}")
            
        except Exception as e:
            print(f"⚠️  Calendar request failed: {e}")
        
        print("\n" + "=" * 40)
        print("✅ Direct API test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_polygon_options_direct()