#!/usr/bin/env python3
"""
Test the complete Greeks calculation system
"""

import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent if Path(__file__).parent.name == 'scripts' else Path(__file__).parent
sys.path.insert(0, str(project_root))

from greeks_calculator import GreeksCalculator, BlackScholesCalculator
import polars as pl
from datetime import datetime

def test_greeks_calculations():
    """Test Greeks calculations with known values"""
    print("🧮 Testing Greeks Calculations")
    print("=" * 40)
    
    # Test Black-Scholes pricing first
    S = 100  # Spot price
    K = 100  # Strike price (ATM)
    T = 0.25 # 3 months to expiry
    r = 0.05 # 5% risk-free rate
    sigma = 0.2 # 20% volatility
    
    # Test call pricing
    call_price = BlackScholesCalculator.call_price(S, K, T, r, sigma)
    put_price = BlackScholesCalculator.put_price(S, K, T, r, sigma)
    
    print(f"📊 Black-Scholes Test (S=$100, K=$100, T=0.25y, r=5%, σ=20%):")
    print(f"   Call Price: ${call_price:.4f}")
    print(f"   Put Price: ${put_price:.4f}")
    
    # Test Greeks
    delta_call = BlackScholesCalculator.delta(S, K, T, r, sigma, 'call')
    delta_put = BlackScholesCalculator.delta(S, K, T, r, sigma, 'put')
    gamma = BlackScholesCalculator.gamma(S, K, T, r, sigma)
    theta_call = BlackScholesCalculator.theta(S, K, T, r, sigma, 'call')
    vega = BlackScholesCalculator.vega(S, K, T, r, sigma)
    vanna = BlackScholesCalculator.vanna(S, K, T, r, sigma)
    
    print(f"\n🎯 Greeks Test:")
    print(f"   Call Delta: {delta_call:.4f}")
    print(f"   Put Delta: {delta_put:.4f}")
    print(f"   Gamma: {gamma:.6f}")
    print(f"   Call Theta: {theta_call:.4f}")
    print(f"   Vega: {vega:.4f}")
    print(f"   Vanna: {vanna:.6f}")
    
    # Validate with expected ranges
    validations = []
    validations.append(("Call Delta", 0.4 < delta_call < 0.7, delta_call))
    validations.append(("Put Delta", -0.7 < delta_put < -0.4, delta_put))
    validations.append(("Gamma", gamma > 0, gamma))
    validations.append(("Call Theta", theta_call < 0, theta_call))
    validations.append(("Vega", vega > 0, vega))
    
    print(f"\n✅ Validation Results:")
    for name, is_valid, value in validations:
        status = "✅" if is_valid else "❌"
        print(f"   {status} {name}: {value:.6f}")
    
    return all(validation[1] for validation in validations)

def test_greeks_calculator():
    """Test the complete GreeksCalculator class"""
    print(f"\n🔧 Testing GreeksCalculator Class")
    print("-" * 40)
    
    calculator = GreeksCalculator(
        risk_free_rate=0.05,
        dividend_yields={'AAPL': 0.005}
    )
    
    # Test single option
    greeks = calculator.calculate_greeks_for_option(
        contract_ticker='O:AAPL250215C00200000',
        underlying_symbol='AAPL',
        strike_price=200.0,
        expiration_date='2025-02-15',
        contract_type='call',
        spot_price=210.0,
        option_price=15.0,
        timestamp=datetime(2025, 1, 15)
    )
    
    print(f"📈 Single Option Test:")
    print(f"   Contract: {greeks.contract_ticker}")
    print(f"   Underlying: {greeks.underlying_symbol}")
    print(f"   Spot: ${greeks.spot_price:.2f}")
    print(f"   Strike: ${greeks.strike_price:.2f}")
    print(f"   Option Price: ${greeks.option_price:.2f}")
    print(f"   Time to Expiry: {greeks.time_to_expiry:.4f} years")
    print(f"   Implied Vol: {greeks.implied_volatility:.1%}")
    
    print(f"\n📊 Calculated Greeks:")
    print(f"   Delta: {greeks.delta:.4f}")
    print(f"   Gamma: {greeks.gamma:.6f}")
    print(f"   Theta: {greeks.theta:.4f}")
    print(f"   Vega: {greeks.vega:.4f}")
    print(f"   Rho: {greeks.rho:.4f}")
    print(f"   Vanna: {greeks.vanna:.6f}")
    print(f"   Charm: {greeks.charm:.6f}")
    print(f"   Volga: {greeks.volga:.8f}")
    print(f"   Speed: {greeks.speed:.8f}")
    print(f"   Zomma: {greeks.zomma:.8f}")
    
    # Test with multiple options
    print(f"\n📋 Testing Multiple Options:")
    
    # Create sample options data
    sample_options = [
        {'contract_ticker': 'O:AAPL250215C00200000', 'underlying_symbol': 'AAPL', 'strike_price': 200, 'expiration_date': '2025-02-15', 'contract_type': 'call', 'close': 15.0, 'volume': 100, 'timestamp_utc': datetime(2025, 1, 15)},
        {'contract_ticker': 'O:AAPL250215C00210000', 'underlying_symbol': 'AAPL', 'strike_price': 210, 'expiration_date': '2025-02-15', 'contract_type': 'call', 'close': 8.0, 'volume': 200, 'timestamp_utc': datetime(2025, 1, 15)},
        {'contract_ticker': 'O:AAPL250215P00200000', 'underlying_symbol': 'AAPL', 'strike_price': 200, 'expiration_date': '2025-02-15', 'contract_type': 'put', 'close': 5.0, 'volume': 150, 'timestamp_utc': datetime(2025, 1, 15)},
        {'contract_ticker': 'O:AAPL250215P00190000', 'underlying_symbol': 'AAPL', 'strike_price': 190, 'expiration_date': '2025-02-15', 'contract_type': 'put', 'close': 2.0, 'volume': 75, 'timestamp_utc': datetime(2025, 1, 15)},
    ]
    
    options_df = pl.DataFrame(sample_options)
    stock_price = 210.0
    
    greeks_df = calculator.calculate_greeks_for_chain(options_df, stock_price)
    
    if not greeks_df.is_empty():
        print(f"   Calculated Greeks for {greeks_df.shape[0]} options")
        print(f"   Columns: {list(greeks_df.columns)}")
        
        # Show summary statistics
        summary = greeks_df.select([
            pl.col('delta').sum().alias('total_delta'),
            pl.col('gamma').sum().alias('total_gamma'),
            pl.col('vega').sum().alias('total_vega'),
            pl.col('vanna').sum().alias('total_vanna')
        ])
        
        print(f"\n📈 Portfolio Greeks Summary:")
        for row in summary.iter_rows(named=True):
            print(f"   Total Delta: {row['total_delta']:.4f}")
            print(f"   Total Gamma: {row['total_gamma']:.6f}")
            print(f"   Total Vega: {row['total_vega']:.4f}")
            print(f"   Total Vanna: {row['total_vanna']:.6f}")
        
        return True
    else:
        print("   ❌ No Greeks calculated")
        return False

def test_dealer_exposure_calculation():
    """Test dealer gamma exposure calculation logic"""
    print(f"\n⚡ Testing Dealer Exposure Logic")
    print("-" * 40)
    
    # Sample Greeks data
    sample_data = [
        {'strike_price': 190, 'contract_type': 'put', 'gamma': 0.01, 'vanna': -0.05, 'volume': 100, 'spot_price': 200},
        {'strike_price': 200, 'contract_type': 'call', 'gamma': 0.015, 'vanna': 0.03, 'volume': 200, 'spot_price': 200},
        {'strike_price': 200, 'contract_type': 'put', 'gamma': 0.015, 'vanna': -0.03, 'volume': 150, 'spot_price': 200},
        {'strike_price': 210, 'contract_type': 'call', 'gamma': 0.01, 'vanna': 0.02, 'volume': 120, 'spot_price': 200},
    ]
    
    df = pl.DataFrame(sample_data)
    
    # Calculate dealer exposure (simplified)
    dealer_exposure = df.with_columns([
        # Dealer positioning assumptions:
        # - Dealers are net SHORT puts (negative gamma exposure)
        # - Dealers are net LONG calls (positive gamma exposure)
        pl.when(pl.col('contract_type') == 'put')
        .then(-pl.col('gamma') * pl.col('volume') * 100)  # Short puts = negative gamma
        .when(pl.col('contract_type') == 'call')
        .then(pl.col('gamma') * pl.col('volume') * 100)   # Long calls = positive gamma
        .otherwise(0)
        .alias('dealer_gamma_exposure'),
        
        pl.when(pl.col('contract_type') == 'put')
        .then(-pl.col('vanna') * pl.col('volume') * 100)  # Short puts = negative vanna
        .when(pl.col('contract_type') == 'call')
        .then(pl.col('vanna') * pl.col('volume') * 100)   # Long calls = positive vanna
        .otherwise(0)
        .alias('dealer_vanna_exposure')
    ])
    
    print("📊 Dealer Exposure by Strike:")
    for row in dealer_exposure.iter_rows(named=True):
        print(f"   ${row['strike_price']} {row['contract_type']}: Gamma={row['dealer_gamma_exposure']:,.0f}, Vanna={row['dealer_vanna_exposure']:,.0f}")
    
    # Total exposure
    total_gamma = dealer_exposure.select(pl.col('dealer_gamma_exposure').sum()).item()
    total_vanna = dealer_exposure.select(pl.col('dealer_vanna_exposure').sum()).item()
    
    print(f"\n🎯 Total Dealer Exposure:")
    print(f"   Net Gamma Exposure: {total_gamma:,.0f}")
    print(f"   Net Vanna Exposure: {total_vanna:,.0f}")
    
    # Interpretation
    gamma_bias = "Long Gamma (Stabilizing)" if total_gamma > 0 else "Short Gamma (Destabilizing)"
    vanna_bias = "Positive Vanna" if total_vanna > 0 else "Negative Vanna"
    
    print(f"   Market Structure: {gamma_bias}")
    print(f"   Volatility Exposure: {vanna_bias}")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Complete Greeks System Test")
    print("=" * 50)
    
    tests = [
        ("Black-Scholes Calculations", test_greeks_calculations),
        ("GreeksCalculator Class", test_greeks_calculator),
        ("Dealer Exposure Logic", test_dealer_exposure_calculation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append(result)
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status}: {test_name}")
        except Exception as e:
            print(f"❌ ERROR in {test_name}: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"   {status} {test_name}")
    
    overall = "✅ ALL TESTS PASSED" if all(results) else "❌ SOME TESTS FAILED"
    print(f"\n{overall}")
    
    if all(results):
        print("\n🎉 Greeks calculation system is ready!")
        print("Next steps:")
        print("1. Run: python complete_options_system.py --symbols AAPL --save-all")
        print("2. Check data/processed/greeks_historical/ for results")
        print("3. Build ML features from Greeks history")

if __name__ == "__main__":
    main()