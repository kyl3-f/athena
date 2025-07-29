#!/usr/bin/env python3
"""
Enhanced PolygonClient with Real-time Options Greeks and Chain Analysis
Leverages Polygon's advanced subscription for complete options data
"""

import asyncio
import aiohttp
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

class OptionsChainAnalyzer:
    """
    Analyzes complete options chains for gamma exposure and flow analysis
    """
    
    def __init__(self):
        self.current_chains = {}
        self.gamma_exposure_cache = {}
    
    def calculate_gamma_exposure(self, options_data: pl.DataFrame, underlying_price: float) -> Dict:
        """
        Calculate total gamma exposure from options chain data
        
        Args:
            options_data: Options snapshot data with Greeks
            underlying_price: Current underlying stock price
            
        Returns:
            Dict with gamma exposure metrics
        """
        if options_data.is_empty():
            return {'total_gamma_exposure': 0, 'call_gamma': 0, 'put_gamma': 0}
        
        # Calculate gamma exposure by strike
        gamma_exposure = []
        
        for row in options_data.iter_rows(named=True):
            strike = row.get('strike_price', 0)
            gamma = row.get('gamma', 0)
            open_interest = row.get('open_interest', 0)
            option_type = row.get('option_type', 'call')
            
            # Market maker gamma exposure (opposite of retail)
            mm_gamma = -gamma * open_interest * 100  # 100 shares per contract
            
            if option_type == 'call':
                # Calls: negative gamma exposure below strike, positive above
                exposure = mm_gamma if underlying_price > strike else -mm_gamma
            else:
                # Puts: positive gamma exposure below strike, negative above  
                exposure = -mm_gamma if underlying_price < strike else mm_gamma
            
            gamma_exposure.append({
                'strike': strike,
                'gamma_exposure': exposure,
                'option_type': option_type,
                'gamma': gamma,
                'open_interest': open_interest
            })
        
        total_exposure = sum(item['gamma_exposure'] for item in gamma_exposure)
        call_gamma = sum(item['gamma_exposure'] for item in gamma_exposure if item['option_type'] == 'call')
        put_gamma = sum(item['gamma_exposure'] for item in gamma_exposure if item['option_type'] == 'put')
        
        return {
            'total_gamma_exposure': total_exposure,
            'call_gamma_exposure': call_gamma,
            'put_gamma_exposure': put_gamma,
            'gamma_by_strike': gamma_exposure,
            'net_gamma': call_gamma + put_gamma
        }
    
    def analyze_flow_metrics(self, options_data: pl.DataFrame) -> Dict:
        """
        Analyze options flow metrics from snapshot data
        """
        if options_data.is_empty():
            return {}
        
        # Separate calls and puts
        calls = options_data.filter(pl.col('option_type') == 'call')
        puts = options_data.filter(pl.col('option_type') == 'put')
        
        # Calculate flow metrics
        total_call_volume = calls['volume'].sum() if len(calls) > 0 else 0
        total_put_volume = puts['volume'].sum() if len(puts) > 0 else 0
        
        call_put_ratio = total_call_volume / total_put_volume if total_put_volume > 0 else float('inf')
        
        # Implied volatility analysis
        avg_iv_calls = calls['implied_volatility'].mean() if len(calls) > 0 else 0
        avg_iv_puts = puts['implied_volatility'].mean() if len(puts) > 0 else 0
        
        # Delta analysis
        total_call_delta = (calls['delta'] * calls['open_interest']).sum() if len(calls) > 0 else 0
        total_put_delta = (puts['delta'] * puts['open_interest']).sum() if len(puts) > 0 else 0
        
        return {
            'call_volume': total_call_volume,
            'put_volume': total_put_volume,
            'call_put_ratio': call_put_ratio,
            'avg_iv_calls': avg_iv_calls,
            'avg_iv_puts': avg_iv_puts,
            'iv_skew': avg_iv_puts - avg_iv_calls,
            'total_call_delta': total_call_delta,
            'total_put_delta': total_put_delta,
            'net_delta': total_call_delta + total_put_delta
        }

class EnhancedPolygonClient:
    """
    Enhanced Polygon API client for real-time options data with Greeks
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = None
        self.analyzer = OptionsChainAnalyzer()
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated request to Polygon API"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        params['apikey'] = self.api_key
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"API request failed: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {}
    
    async def get_options_contracts(self, underlying_ticker: str, 
                                  expiration_date: str = None,
                                  strike_price: float = None,
                                  contract_type: str = None) -> pl.DataFrame:
        """
        Get all options contracts for an underlying ticker
        
        Args:
            underlying_ticker: Stock symbol (e.g., 'AAPL')
            expiration_date: Optional expiration filter (YYYY-MM-DD)
            strike_price: Optional strike price filter
            contract_type: Optional 'call' or 'put' filter
            
        Returns:
            Polars DataFrame with options contracts
        """
        endpoint = f"/v3/reference/options/contracts"
        params = {
            'underlying_ticker': underlying_ticker,
            'limit': 1000  # Get comprehensive data
        }
        
        if expiration_date:
            params['expiration_date'] = expiration_date
        if strike_price:
            params['strike_price'] = strike_price
        if contract_type:
            params['contract_type'] = contract_type
        
        response = await self._make_request(endpoint, params)
        
        if 'results' not in response:
            logger.warning(f"No options contracts found for {underlying_ticker}")
            return pl.DataFrame()
        
        contracts = []
        for contract in response['results']:
            contracts.append({
                'ticker': contract.get('ticker', ''),
                'underlying_ticker': contract.get('underlying_ticker', ''),
                'contract_type': contract.get('contract_type', ''),
                'strike_price': contract.get('strike_price', 0),
                'expiration_date': contract.get('expiration_date', ''),
                'shares_per_contract': contract.get('shares_per_contract', 100),
                'primary_exchange': contract.get('primary_exchange', '')
            })
        
        return pl.DataFrame(contracts)
    
    async def get_options_snapshot(self, underlying_ticker: str) -> pl.DataFrame:
        """
        Get real-time options snapshot with Greeks for all contracts
        
        Args:
            underlying_ticker: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Polars DataFrame with options snapshot including Greeks
        """
        endpoint = f"/v3/snapshot/options/{underlying_ticker}"
        
        response = await self._make_request(endpoint)
        
        if 'results' not in response:
            logger.warning(f"No options snapshot found for {underlying_ticker}")
            return pl.DataFrame()
        
        snapshots = []
        for result in response['results']:
            details = result.get('details', {})
            greeks = result.get('greeks', {})
            last_quote = result.get('last_quote', {})
            last_trade = result.get('last_trade', {})
            
            snapshots.append({
                'ticker': result.get('value', ''),
                'underlying_ticker': underlying_ticker,
                'option_type': details.get('contract_type', ''),
                'strike_price': details.get('strike_price', 0),
                'expiration_date': details.get('expiration_date', ''),
                
                # Greeks (from Polygon's calculations)
                'delta': greeks.get('delta', 0),
                'gamma': greeks.get('gamma', 0),
                'theta': greeks.get('theta', 0),
                'vega': greeks.get('vega', 0),
                'implied_volatility': result.get('implied_volatility', 0),
                
                # Market data
                'last_price': last_trade.get('price', 0),
                'bid': last_quote.get('bid', 0),
                'ask': last_quote.get('ask', 0),
                'bid_size': last_quote.get('bid_size', 0),
                'ask_size': last_quote.get('ask_size', 0),
                'volume': result.get('session', {}).get('volume', 0),
                'open_interest': result.get('open_interest', 0),
                
                # Timestamps
                'updated': result.get('updated', 0),
                'timestamp': datetime.now()
            })
        
        return pl.DataFrame(snapshots)
    
    async def get_options_trades(self, option_ticker: str, 
                               timestamp_gte: datetime = None,
                               timestamp_lt: datetime = None,
                               limit: int = 1000) -> pl.DataFrame:
        """
        Get recent options trades for flow analysis
        
        Args:
            option_ticker: Options contract ticker (e.g., 'O:AAPL231117C00150000')
            timestamp_gte: Start time filter
            timestamp_lt: End time filter  
            limit: Maximum number of trades
            
        Returns:
            Polars DataFrame with options trades
        """
        endpoint = f"/v3/trades/{option_ticker}"
        params = {'limit': limit}
        
        if timestamp_gte:
            params['timestamp.gte'] = int(timestamp_gte.timestamp() * 1000)
        if timestamp_lt:
            params['timestamp.lt'] = int(timestamp_lt.timestamp() * 1000)
        
        response = await self._make_request(endpoint, params)
        
        if 'results' not in response:
            return pl.DataFrame()
        
        trades = []
        for trade in response['results']:
            trades.append({
                'ticker': option_ticker,
                'price': trade.get('price', 0),
                'size': trade.get('size', 0),
                'timestamp': datetime.fromtimestamp(trade.get('participant_timestamp', 0) / 1000),
                'exchange': trade.get('exchange', ''),
                'conditions': trade.get('conditions', []),
                'sip_timestamp': datetime.fromtimestamp(trade.get('sip_timestamp', 0) / 1000)
            })
        
        return pl.DataFrame(trades)
    
    async def get_comprehensive_options_data(self, underlying_ticker: str) -> Dict[str, pl.DataFrame]:
        """
        Get comprehensive options data including contracts, snapshots, and analysis
        
        Args:
            underlying_ticker: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Dict with contracts, snapshots, and analysis data
        """
        logger.info(f"Fetching comprehensive options data for {underlying_ticker}")
        
        # Get all data concurrently
        contracts_task = self.get_options_contracts(underlying_ticker)
        snapshot_task = self.get_options_snapshot(underlying_ticker)
        
        contracts, snapshot = await asyncio.gather(contracts_task, snapshot_task)
        
        # Get current stock price for gamma exposure calculation
        stock_endpoint = f"/v2/aggs/ticker/{underlying_ticker}/prev"
        stock_response = await self._make_request(stock_endpoint)
        stock_price = stock_response.get('results', [{}])[0].get('c', 0) if stock_response.get('results') else 0
        
        # Analyze options data
        gamma_analysis = self.analyzer.calculate_gamma_exposure(snapshot, stock_price)
        flow_analysis = self.analyzer.analyze_flow_metrics(snapshot)
        
        return {
            'contracts': contracts,
            'snapshot': snapshot,
            'gamma_exposure': gamma_analysis,
            'flow_metrics': flow_analysis,
            'underlying_price': stock_price,
            'timestamp': datetime.now()
        }
    
    async def get_multi_symbol_options_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get options data for multiple symbols efficiently
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict with options data for each symbol
        """
        logger.info(f"Fetching options data for {len(symbols)} symbols")
        
        # Create tasks for all symbols
        tasks = {
            symbol: self.get_comprehensive_options_data(symbol) 
            for symbol in symbols
        }
        
        # Execute all requests concurrently
        results = {}
        for symbol, task in tasks.items():
            try:
                results[symbol] = await task
            except Exception as e:
                logger.error(f"Failed to get options data for {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def create_options_features_dataframe(self, options_data: Dict) -> pl.DataFrame:
        """
        Create a DataFrame with options-based features for ML
        
        Args:
            options_data: Output from get_comprehensive_options_data
            
        Returns:
            DataFrame with options features
        """
        if 'error' in options_data:
            return pl.DataFrame()
        
        gamma_data = options_data.get('gamma_exposure', {})
        flow_data = options_data.get('flow_metrics', {})
        
        features = {
            'timestamp': [options_data.get('timestamp', datetime.now())],
            'underlying_price': [options_data.get('underlying_price', 0)],
            
            # Gamma exposure features
            'total_gamma_exposure': [gamma_data.get('total_gamma_exposure', 0)],
            'call_gamma_exposure': [gamma_data.get('call_gamma_exposure', 0)],
            'put_gamma_exposure': [gamma_data.get('put_gamma_exposure', 0)],
            'net_gamma': [gamma_data.get('net_gamma', 0)],
            
            # Flow features
            'call_volume': [flow_data.get('call_volume', 0)],
            'put_volume': [flow_data.get('put_volume', 0)],
            'call_put_ratio': [flow_data.get('call_put_ratio', 0)],
            'avg_iv_calls': [flow_data.get('avg_iv_calls', 0)],
            'avg_iv_puts': [flow_data.get('avg_iv_puts', 0)],
            'iv_skew': [flow_data.get('iv_skew', 0)],
            'total_call_delta': [flow_data.get('total_call_delta', 0)],
            'total_put_delta': [flow_data.get('total_put_delta', 0)],
            'net_delta': [flow_data.get('net_delta', 0)]
        }
        
        return pl.DataFrame(features)

# Usage example and testing functions
async def test_options_integration():
    """Test the enhanced options functionality"""
    api_key = "your_polygon_api_key"  # Replace with actual key
    
    async with EnhancedPolygonClient(api_key) as client:
        # Test with AAPL
        symbol = "AAPL"
        
        print(f"Testing options integration for {symbol}...")
        
        # Get comprehensive options data
        options_data = await client.get_comprehensive_options_data(symbol)
        
        print(f"Contracts found: {len(options_data['contracts'])}")
        print(f"Snapshot records: {len(options_data['snapshot'])}")
        print(f"Gamma exposure: {options_data['gamma_exposure']['total_gamma_exposure']:,.0f}")
        print(f"Call/Put ratio: {options_data['flow_metrics']['call_put_ratio']:.2f}")
        
        # Create features DataFrame
        features_df = client.create_options_features_dataframe(options_data)
        print(f"Options features: {features_df.columns}")
        
        return options_data

if __name__ == "__main__":
    # Run test
    asyncio.run(test_options_integration())