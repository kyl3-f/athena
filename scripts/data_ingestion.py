# scripts/data_ingestion.py
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict
import logging

class PolygonDataIngester:
    def __init__(self, api_key: str, base_url: str = "https://api.polygon.io"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_stock_data(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """Fetch stock price data from Polygon"""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/minute/{from_date}/{to_date}"
        params = {"apikey": self.api_key}
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            return data.get('results', [])
    
    async def get_options_data(self, underlying: str, date: str) -> List[Dict]:
        """Fetch options data from Polygon"""
        url = f"{self.base_url}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "contract_type": "call",  # You'll need separate calls for puts
            "expiration_date": date,
            "apikey": self.api_key
        }
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            return data.get('results', [])