#!/usr/bin/env python3
"""
Athena Options Greeks Calculator & Historical Logger
Calculates and maintains historical Greeks for options flow analysis and model training
"""

import numpy as np
import polars as pl
from scipy.stats import norm
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import math

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GreeksSnapshot:
    """Container for Greeks at a specific point in time"""
    timestamp: datetime
    underlying_symbol: str
    contract_ticker: str
    strike_price: float
    expiration_date: str
    contract_type: str
    spot_price: float
    option_price: float
    time_to_expiry: float
    implied_volatility: float
    risk_free_rate: float
    dividend_yield: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    vanna: float
    charm: float
    volga: float
    speed: float
    zomma: float


class BlackScholesCalculator:
    """
    Black-Scholes model for options pricing and Greeks calculation
    """
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate d1 parameter"""
        if T <= 0 or sigma <= 0:
            return 0
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate d2 parameter"""
        if T <= 0 or sigma <= 0:
            return 0
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        return d1_val - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Black-Scholes call option price"""
        if T <= 0:
            return max(S - K, 0)
        
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholesCalculator.d2(S, K, T, r, sigma, q)
        
        call_value = (S * np.exp(-q * T) * norm.cdf(d1_val) - 
                     K * np.exp(-r * T) * norm.cdf(d2_val))
        return max(call_value, 0)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Black-Scholes put option price"""
        if T <= 0:
            return max(K - S, 0)
        
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholesCalculator.d2(S, K, T, r, sigma, q)
        
        put_value = (K * np.exp(-r * T) * norm.cdf(-d2_val) - 
                    S * np.exp(-q * T) * norm.cdf(-d1_val))
        return max(put_value, 0)
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0) -> float:
        """Calculate Delta"""
        if T <= 0:
            if option_type.lower() == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        
        if option_type.lower() == 'call':
            return np.exp(-q * T) * norm.cdf(d1_val)
        else:
            return -np.exp(-q * T) * norm.cdf(-d1_val)
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate Gamma"""
        if T <= 0 or sigma <= 0:
            return 0
        
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        gamma_val = (np.exp(-q * T) * norm.pdf(d1_val)) / (S * sigma * np.sqrt(T))
        return gamma_val
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0) -> float:
        """Calculate Theta (per day)"""
        if T <= 0:
            return 0
        
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholesCalculator.d2(S, K, T, r, sigma, q)
        
        term1 = -(S * norm.pdf(d1_val) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
        
        if option_type.lower() == 'call':
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2_val)
            term3 = q * S * np.exp(-q * T) * norm.cdf(d1_val)
            theta_val = term1 + term2 + term3
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2_val)
            term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1_val)
            theta_val = term1 + term2 + term3
        
        return theta_val / 365  # Convert to per day
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate Vega"""
        if T <= 0:
            return 0
        
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        vega_val = S * np.exp(-q * T) * norm.pdf(d1_val) * np.sqrt(T)
        return vega_val / 100  # Convert to per 1% volatility change
    
    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0) -> float:
        """Calculate Rho"""
        if T <= 0:
            return 0
        
        d2_val = BlackScholesCalculator.d2(S, K, T, r, sigma, q)
        
        if option_type.lower() == 'call':
            rho_val = K * T * np.exp(-r * T) * norm.cdf(d2_val)
        else:
            rho_val = -K * T * np.exp(-r * T) * norm.cdf(-d2_val)
        
        return rho_val / 100  # Convert to per 1% interest rate change
    
    @staticmethod
    def vanna(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate Vanna (dDelta/dIV)"""
        if T <= 0 or sigma <= 0:
            return 0
        
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholesCalculator.d2(S, K, T, r, sigma, q)
        
        vanna_val = -np.exp(-q * T) * norm.pdf(d1_val) * (d2_val / sigma)
        return vanna_val / 100  # Per 1% volatility change
    
    @staticmethod
    def charm(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0) -> float:
        """Calculate Charm (dDelta/dTime)"""
        if T <= 0:
            return 0
        
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholesCalculator.d2(S, K, T, r, sigma, q)
        
        if option_type.lower() == 'call':
            charm_val = -np.exp(-q * T) * norm.pdf(d1_val) * (
                (r - q) / (sigma * np.sqrt(T)) - d2_val / (2 * T)
            ) - q * np.exp(-q * T) * norm.cdf(d1_val)
        else:
            charm_val = -np.exp(-q * T) * norm.pdf(d1_val) * (
                (r - q) / (sigma * np.sqrt(T)) - d2_val / (2 * T)
            ) + q * np.exp(-q * T) * norm.cdf(-d1_val)
        
        return charm_val / 365  # Per day
    
    @staticmethod
    def volga(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate Volga (dVega/dIV)"""
        if T <= 0 or sigma <= 0:
            return 0
        
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholesCalculator.d2(S, K, T, r, sigma, q)
        
        volga_val = S * np.exp(-q * T) * norm.pdf(d1_val) * np.sqrt(T) * (d1_val * d2_val / sigma)
        return volga_val / 10000  # Per (1% volatility)^2
    
    @staticmethod
    def speed(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate Speed (dGamma/dSpot)"""
        if T <= 0 or sigma <= 0:
            return 0
        
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        gamma_val = BlackScholesCalculator.gamma(S, K, T, r, sigma, q)
        
        speed_val = -gamma_val * (d1_val / (S * sigma * np.sqrt(T)) + 1 / S)
        return speed_val
    
    @staticmethod
    def zomma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
        """Calculate Zomma (dGamma/dIV)"""
        if T <= 0 or sigma <= 0:
            return 0
        
        d1_val = BlackScholesCalculator.d1(S, K, T, r, sigma, q)
        d2_val = BlackScholesCalculator.d2(S, K, T, r, sigma, q)
        gamma_val = BlackScholesCalculator.gamma(S, K, T, r, sigma, q)
        
        zomma_val = gamma_val * ((d1_val * d2_val - 1) / sigma)
        return zomma_val / 100  # Per 1% volatility change


class ImpliedVolatilityCalculator:
    """
    Calculate implied volatility using Newton-Raphson method
    """
    
    @staticmethod
    def calculate_iv(market_price: float, S: float, K: float, T: float, r: float, 
                    option_type: str, q: float = 0, max_iterations: int = 100, 
                    tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        """
        if T <= 0 or market_price <= 0:
            return 0.2  # Default 20% volatility
        
        # Initial guess
        sigma = 0.2
        
        for i in range(max_iterations):
            if option_type.lower() == 'call':
                bs_price = BlackScholesCalculator.call_price(S, K, T, r, sigma, q)
            else:
                bs_price = BlackScholesCalculator.put_price(S, K, T, r, sigma, q)
            
            price_diff = bs_price - market_price
            
            if abs(price_diff) < tolerance:
                return max(sigma, 0.01)  # Minimum 1% volatility
            
            # Calculate vega for Newton-Raphson
            vega = BlackScholesCalculator.vega(S, K, T, r, sigma, q) * 100  # Convert back
            
            if vega == 0:
                return max(sigma, 0.01)
            
            # Newton-Raphson update
            sigma = sigma - price_diff / vega
            sigma = max(sigma, 0.01)  # Keep positive
            sigma = min(sigma, 5.0)   # Cap at 500%
        
        return max(sigma, 0.01)


class GreeksCalculator:
    """
    Main Greeks calculator and historical logger
    """
    
    def __init__(self, risk_free_rate: float = 0.05, dividend_yields: Dict[str, float] = None):
        self.risk_free_rate = risk_free_rate
        self.dividend_yields = dividend_yields or {}
        self.bs_calc = BlackScholesCalculator()
        self.iv_calc = ImpliedVolatilityCalculator()
    
    def calculate_time_to_expiry(self, expiration_date: str, current_date: str = None) -> float:
        """Calculate time to expiry in years"""
        if current_date is None:
            current_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            current = datetime.strptime(current_date, '%Y-%m-%d')
            expiry = datetime.strptime(expiration_date, '%Y-%m-%d')
            
            # Calculate business days (approximate)
            total_days = (expiry - current).days
            business_days = total_days * (5/7)  # Approximate business days
            
            return max(business_days / 365, 1/365)  # Minimum 1 day
        except:
            return 1/365  # Default to 1 day
    
    def calculate_greeks_for_option(
        self,
        contract_ticker: str,
        underlying_symbol: str, 
        strike_price: float,
        expiration_date: str,
        contract_type: str,
        spot_price: float,
        option_price: float,
        timestamp: datetime = None,
        use_market_iv: bool = True
    ) -> GreeksSnapshot:
        """
        Calculate all Greeks for a single option
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate time to expiry
        T = self.calculate_time_to_expiry(expiration_date, timestamp.strftime('%Y-%m-%d'))
        
        # Get dividend yield
        q = self.dividend_yields.get(underlying_symbol, 0.0)
        
        # Calculate or use provided implied volatility
        if use_market_iv:
            sigma = self.iv_calc.calculate_iv(
                option_price, spot_price, strike_price, T, 
                self.risk_free_rate, contract_type, q
            )
        else:
            sigma = 0.2  # Default 20%
        
        # Calculate all Greeks
        delta = self.bs_calc.delta(spot_price, strike_price, T, self.risk_free_rate, sigma, contract_type, q)
        gamma = self.bs_calc.gamma(spot_price, strike_price, T, self.risk_free_rate, sigma, q)
        theta = self.bs_calc.theta(spot_price, strike_price, T, self.risk_free_rate, sigma, contract_type, q)
        vega = self.bs_calc.vega(spot_price, strike_price, T, self.risk_free_rate, sigma, q)
        rho = self.bs_calc.rho(spot_price, strike_price, T, self.risk_free_rate, sigma, contract_type, q)
        vanna = self.bs_calc.vanna(spot_price, strike_price, T, self.risk_free_rate, sigma, q)
        charm = self.bs_calc.charm(spot_price, strike_price, T, self.risk_free_rate, sigma, contract_type, q)
        volga = self.bs_calc.volga(spot_price, strike_price, T, self.risk_free_rate, sigma, q)
        speed = self.bs_calc.speed(spot_price, strike_price, T, self.risk_free_rate, sigma, q)
        zomma = self.bs_calc.zomma(spot_price, strike_price, T, self.risk_free_rate, sigma, q)
        
        return GreeksSnapshot(
            timestamp=timestamp,
            underlying_symbol=underlying_symbol,
            contract_ticker=contract_ticker,
            strike_price=strike_price,
            expiration_date=expiration_date,
            contract_type=contract_type,
            spot_price=spot_price,
            option_price=option_price,
            time_to_expiry=T,
            implied_volatility=sigma,
            risk_free_rate=self.risk_free_rate,
            dividend_yield=q,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            vanna=vanna,
            charm=charm,
            volga=volga,
            speed=speed,
            zomma=zomma
        )
    
    def calculate_greeks_for_chain(self, options_data: pl.DataFrame, stock_price: float) -> pl.DataFrame:
        """
        Calculate Greeks for entire options chain
        """
        if options_data.is_empty():
            return pl.DataFrame()
        
        logger.info(f"Calculating Greeks for {options_data.shape[0]} options")
        
        greeks_data = []
        
        for row in options_data.iter_rows(named=True):
            try:
                # Get option data
                contract_ticker = row.get('contract_ticker', '')
                underlying_symbol = row.get('underlying_symbol', '')
                strike_price = row.get('strike_price', 0)
                expiration_date = row.get('expiration_date', '')
                contract_type = row.get('contract_type', 'call')
                option_price = row.get('close', 0)
                timestamp = row.get('timestamp_utc', datetime.now())
                
                if option_price <= 0 or strike_price <= 0:
                    continue
                
                # Calculate Greeks
                greeks = self.calculate_greeks_for_option(
                    contract_ticker, underlying_symbol, strike_price,
                    expiration_date, contract_type, stock_price, option_price, timestamp
                )
                
                # Convert to dictionary
                greeks_dict = {
                    'timestamp_utc': greeks.timestamp,
                    'underlying_symbol': greeks.underlying_symbol,
                    'contract_ticker': greeks.contract_ticker,
                    'strike_price': greeks.strike_price,
                    'expiration_date': greeks.expiration_date,
                    'contract_type': greeks.contract_type,
                    'spot_price': greeks.spot_price,
                    'option_price': greeks.option_price,
                    'time_to_expiry': greeks.time_to_expiry,
                    'implied_volatility': greeks.implied_volatility,
                    'delta': greeks.delta,
                    'gamma': greeks.gamma,
                    'theta': greeks.theta,
                    'vega': greeks.vega,
                    'rho': greeks.rho,
                    'vanna': greeks.vanna,
                    'charm': greeks.charm,
                    'volga': greeks.volga,
                    'speed': greeks.speed,
                    'zomma': greeks.zomma,
                    # Add original option data
                    'volume': row.get('volume', 0),
                    'open_interest': row.get('open_interest', 0) if 'open_interest' in row else 0,
                }
                
                greeks_data.append(greeks_dict)
                
            except Exception as e:
                logger.warning(f"Failed to calculate Greeks for row: {e}")
                continue
        
        if greeks_data:
            greeks_df = pl.DataFrame(greeks_data)
            logger.info(f"Calculated Greeks for {greeks_df.shape[0]} options")
            return greeks_df
        else:
            logger.warning("No Greeks calculated")
            return pl.DataFrame()
    
    def save_greeks_history(self, greeks_df: pl.DataFrame, output_dir: Path, underlying: str, date: str):
        """Save Greeks to historical log"""
        if greeks_df.is_empty():
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save daily snapshot
        daily_file = output_dir / f"{underlying}_greeks_{date}.parquet"
        greeks_df.write_parquet(daily_file)
        logger.info(f"💾 Saved Greeks snapshot: {daily_file}")
        
        # Append to historical log
        historical_file = output_dir / f"{underlying}_greeks_historical.parquet"
        
        if historical_file.exists():
            # Append to existing file
            existing_df = pl.read_parquet(historical_file)
            combined_df = pl.concat([existing_df, greeks_df]).unique(['timestamp_utc', 'contract_ticker'])
            combined_df.write_parquet(historical_file)
        else:
            # Create new file
            greeks_df.write_parquet(historical_file)
        
        logger.info(f"📊 Updated historical Greeks log: {historical_file}")


def main():
    """Test the Greeks calculator"""
    
    # Example usage
    calculator = GreeksCalculator(
        risk_free_rate=0.05,  # 5% risk-free rate
        dividend_yields={'AAPL': 0.005, 'SPY': 0.015}  # Dividend yields
    )
    
    # Test single option
    greeks = calculator.calculate_greeks_for_option(
        contract_ticker='O:AAPL250801C00200000',
        underlying_symbol='AAPL',
        strike_price=200.0,
        expiration_date='2025-08-01',
        contract_type='call',
        spot_price=210.0,
        option_price=15.0
    )
    
    print("🧮 Sample Greeks Calculation:")
    print(f"Contract: {greeks.contract_ticker}")
    print(f"Spot: ${greeks.spot_price:.2f}, Strike: ${greeks.strike_price:.2f}")
    print(f"Time to Expiry: {greeks.time_to_expiry:.4f} years")
    print(f"Implied Vol: {greeks.implied_volatility:.1%}")
    print(f"Delta: {greeks.delta:.4f}")
    print(f"Gamma: {greeks.gamma:.6f}")
    print(f"Theta: {greeks.theta:.4f}")
    print(f"Vega: {greeks.vega:.4f}")
    print(f"Vanna: {greeks.vanna:.6f}")
    print(f"Charm: {greeks.charm:.6f}")


if __name__ == "__main__":
    main()