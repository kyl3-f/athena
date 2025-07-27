#!/usr/bin/env python3
"""
Complete Athena Options System with Greeks Calculation
Downloads options data and calculates comprehensive Greeks for model training
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import polars as pl

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from fixed_options_downloader import FixedOptionsManager
from calculation.greeks_calculator import GreeksCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/complete_options_system.log')
    ]
)
logger = logging.getLogger(__name__)


class CompleteOptionsSystem:
    """
    Complete options analysis system with Greeks calculation and historical logging
    """
    
    def __init__(self):
        self.options_manager = FixedOptionsManager()
        self.greeks_calculator = GreeksCalculator(
            risk_free_rate=0.045,  # Current ~4.5%
            dividend_yields={
                'AAPL': 0.005,   # 0.5%
                'GOOGL': 0.0,    # No dividend
                'MSFT': 0.007,   # 0.7%
                'TSLA': 0.0,     # No dividend
                'SPY': 0.015,    # ~1.5%
                'QQQ': 0.005     # ~0.5%
            }
        )
    
    def calculate_dealer_gamma_exposure(self, greeks_df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate dealer gamma exposure using our calculated Greeks
        """
        if greeks_df.is_empty():
            return pl.DataFrame()
        
        logger.info("Calculating dealer gamma exposure from Greeks...")
        
        # Estimate open interest (placeholder - would need real data)
        # For now, use volume as proxy for activity
        estimated_oi = greeks_df.with_columns([
            (pl.col('volume') * 10).alias('estimated_open_interest')  # Rough estimate
        ])
        
        # Calculate dealer positioning
        dealer_exposure = estimated_oi.with_columns([
            # Dealer gamma exposure (dealers typically short puts, long calls)
            pl.when(pl.col('contract_type') == 'put')
            .then(-pl.col('gamma') * pl.col('estimated_open_interest') * 100)  # Short puts = negative gamma
            .when(pl.col('contract_type') == 'call')
            .then(pl.col('gamma') * pl.col('estimated_open_interest') * 100)   # Long calls = positive gamma
            .otherwise(0)
            .alias('dealer_gamma_exposure'),
            
            # Vanna exposure (sensitivity to volatility changes)
            pl.when(pl.col('contract_type') == 'put')
            .then(-pl.col('vanna') * pl.col('estimated_open_interest') * 100)
            .when(pl.col('contract_type') == 'call')
            .then(pl.col('vanna') * pl.col('estimated_open_interest') * 100)
            .otherwise(0)
            .alias('dealer_vanna_exposure'),
            
            # Charm exposure (delta decay)
            pl.when(pl.col('contract_type') == 'put')
            .then(-pl.col('charm') * pl.col('estimated_open_interest') * 100)
            .when(pl.col('contract_type') == 'call')
            .then(pl.col('charm') * pl.col('estimated_open_interest') * 100)
            .otherwise(0)
            .alias('dealer_charm_exposure'),
            
            # Notional exposure
            (pl.col('estimated_open_interest') * pl.col('option_price') * 100).alias('notional_exposure')
        ])
        
        # Aggregate by strike
        gamma_by_strike = dealer_exposure.group_by('strike_price').agg([
            pl.col('dealer_gamma_exposure').sum().alias('total_gamma_exposure'),
            pl.col('dealer_vanna_exposure').sum().alias('total_vanna_exposure'),
            pl.col('dealer_charm_exposure').sum().alias('total_charm_exposure'),
            pl.col('notional_exposure').sum().alias('total_notional'),
            pl.col('spot_price').first().alias('current_price'),
            pl.col('underlying_symbol').first().alias('underlying'),
            pl.col('timestamp_utc').first().alias('analysis_timestamp')
        ]).sort('strike_price')
        
        # Add analysis fields
        gamma_by_strike = gamma_by_strike.with_columns([
            # Distance from current price
            (pl.col('strike_price') - pl.col('current_price')).alias('strike_distance'),
            ((pl.col('strike_price') - pl.col('current_price')).abs() / pl.col('current_price')).alias('strike_distance_pct'),
            
            # Exposure magnitudes
            pl.col('total_gamma_exposure').abs().alias('gamma_magnitude'),
            pl.col('total_vanna_exposure').abs().alias('vanna_magnitude'),
            pl.col('total_charm_exposure').abs().alias('charm_magnitude'),
        ])
        
        return gamma_by_strike
    
    def identify_gamma_flip_levels(self, gamma_exposure_df: pl.DataFrame) -> Dict:
        """
        Identify price levels where gamma exposure flips sign
        """
        if gamma_exposure_df.is_empty():
            return {}
        
        # Sort by strike and calculate cumulative gamma
        sorted_gamma = gamma_exposure_df.sort('strike_price')
        sorted_gamma = sorted_gamma.with_columns([
            pl.col('total_gamma_exposure').cumsum().alias('cumulative_gamma')
        ])
        
        current_price = sorted_gamma.select('current_price').head(1).item()
        flip_levels = []
        
        # Find zero crossings
        for i in range(1, sorted_gamma.shape[0]):
            current_row = sorted_gamma.row(i, named=True)
            prev_row = sorted_gamma.row(i-1, named=True)
            
            current_gamma = current_row['cumulative_gamma']
            prev_gamma = prev_row['cumulative_gamma']
            
            # Check for sign change
            if (current_gamma > 0 and prev_gamma < 0) or (current_gamma < 0 and prev_gamma > 0):
                # Linear interpolation for flip level
                strike1 = prev_row['strike_price']
                strike2 = current_row['strike_price']
                
                if current_gamma != prev_gamma:
                    flip_strike = strike1 + (strike2 - strike1) * (-prev_gamma) / (current_gamma - prev_gamma)
                    flip_distance = (flip_strike - current_price) / current_price
                    
                    flip_levels.append({
                        'flip_strike': flip_strike,
                        'flip_distance_pct': flip_distance,
                        'gamma_before': prev_gamma,
                        'gamma_after': current_gamma,
                        'direction': 'positive_to_negative' if prev_gamma > 0 else 'negative_to_positive'
                    })
        
        # Find nearest flip level
        nearest_flip = {}
        if flip_levels:
            nearest_flip = min(flip_levels, key=lambda x: abs(x['flip_distance_pct']))
        
        return {
            'all_flip_levels': flip_levels,
            'nearest_flip_level': nearest_flip,
            'flip_count': len(flip_levels)
        }
    
    def analyze_options_flow_signals(self, greeks_df: pl.DataFrame) -> Dict:
        """
        Analyze options flow for trading signals
        """
        if greeks_df.is_empty():
            return {}
        
        logger.info("Analyzing options flow signals...")
        
        signals = {}
        current_time = datetime.now()
        
        # 1. Unusual Volume Analysis
        avg_volume = greeks_df.select(pl.col('volume').mean()).item()
        high_volume_options = greeks_df.filter(pl.col('volume') > avg_volume * 3)
        
        signals['unusual_volume'] = {
            'count': high_volume_options.shape[0],
            'threshold': avg_volume * 3,
            'contracts': high_volume_options.select(['contract_ticker', 'volume', 'strike_price', 'contract_type']).to_dicts()
        }
        
        # 2. High Gamma Concentration
        high_gamma = greeks_df.filter(pl.col('gamma').abs() > greeks_df.select(pl.col('gamma').abs().quantile(0.9)).item())
        
        signals['high_gamma_activity'] = {
            'count': high_gamma.shape[0],
            'total_gamma': high_gamma.select(pl.col('gamma').sum()).item(),
            'avg_gamma': high_gamma.select(pl.col('gamma').mean()).item()
        }
        
        # 3. Vanna Exposure Analysis
        total_vanna = greeks_df.select(pl.col('vanna').sum()).item()
        
        signals['vanna_exposure'] = {
            'total_vanna': total_vanna,
            'vanna_bias': 'positive' if total_vanna > 0 else 'negative',
            'volatility_sensitivity': abs(total_vanna)
        }
        
        # 4. Near-term Expiration Gamma
        near_term = greeks_df.filter(pl.col('time_to_expiry') <= 0.1)  # ~5 weeks
        
        if not near_term.is_empty():
            signals['near_term_gamma'] = {
                'total_gamma': near_term.select(pl.col('gamma').sum()).item(),
                'contract_count': near_term.shape[0],
                'avg_time_to_expiry': near_term.select(pl.col('time_to_expiry').mean()).item()
            }
        
        # 5. Put/Call Ratio Analysis
        puts = greeks_df.filter(pl.col('contract_type') == 'put')
        calls = greeks_df.filter(pl.col('contract_type') == 'call')
        
        put_volume = puts.select(pl.col('volume').sum()).item() if not puts.is_empty() else 0
        call_volume = calls.select(pl.col('volume').sum()).item() if not calls.is_empty() else 0
        
        signals['put_call_analysis'] = {
            'put_volume': put_volume,
            'call_volume': call_volume,
            'put_call_ratio': put_volume / call_volume if call_volume > 0 else 0,
            'put_gamma': puts.select(pl.col('gamma').sum()).item() if not puts.is_empty() else 0,
            'call_gamma': calls.select(pl.col('gamma').sum()).item() if not calls.is_empty() else 0
        }
        
        return signals
    
    def process_complete_options_analysis(
        self, 
        underlying: str, 
        days_back: int = 5, 
        max_dte: int = 60
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict, Dict]:
        """
        Complete options analysis pipeline
        
        Returns:
            Tuple of (options_data, greeks_data, gamma_exposure, flow_signals)
        """
        logger.info(f"Processing complete options analysis for {underlying}")
        
        # 1. Get options contracts and data
        contracts_df = self.options_manager.get_options_contracts(underlying, max_dte)
        
        if contracts_df.is_empty():
            logger.warning(f"No contracts found for {underlying}")
            return pl.DataFrame(), pl.DataFrame(), {}, {}
        
        # Enhance with moneyness
        contracts_df = self.options_manager.enhance_contracts_with_moneyness(contracts_df, underlying)
        
        # Download options data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        options_data = self.options_manager.download_options_data(contracts_df, start_date, end_date)
        
        if options_data.is_empty():
            logger.warning(f"No options data downloaded for {underlying}")
            return contracts_df, pl.DataFrame(), {}, {}
        
        # 2. Calculate Greeks
        current_stock_price = self.options_manager.get_current_stock_price(underlying)
        greeks_df = self.greeks_calculator.calculate_greeks_for_chain(options_data, current_stock_price)
        
        if greeks_df.is_empty():
            logger.warning(f"No Greeks calculated for {underlying}")
            return contracts_df, options_data, {}, {}
        
        # 3. Calculate dealer gamma exposure
        gamma_exposure_df = self.calculate_dealer_gamma_exposure(greeks_df)
        
        # 4. Identify gamma flip levels
        flip_analysis = self.identify_gamma_flip_levels(gamma_exposure_df)
        
        # 5. Analyze options flow signals
        flow_signals = self.analyze_options_flow_signals(greeks_df)
        
        # 6. Combine results
        exposure_analysis = {
            'gamma_exposure_by_strike': gamma_exposure_df.to_dicts() if not gamma_exposure_df.is_empty() else [],
            'flip_levels': flip_analysis,
            'summary': {
                'total_gamma_exposure': gamma_exposure_df.select(pl.col('total_gamma_exposure').sum()).item() if not gamma_exposure_df.is_empty() else 0,
                'total_vanna_exposure': gamma_exposure_df.select(pl.col('total_vanna_exposure').sum()).item() if not gamma_exposure_df.is_empty() else 0,
                'current_stock_price': current_stock_price,
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        # 7. Save historical Greeks
        greeks_output_dir = Path('data/processed/greeks_historical')
        self.greeks_calculator.save_greeks_history(
            greeks_df, 
            greeks_output_dir, 
            underlying, 
            datetime.now().strftime('%Y-%m-%d')
        )
        
        return contracts_df, greeks_df, exposure_analysis, flow_signals


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Complete Options Analysis System")
    
    parser.add_argument('--symbols', type=str, default='AAPL', help='Comma-separated symbols')
    parser.add_argument('--days', type=int, default=5, help='Days of options data')
    parser.add_argument('--max-dte', type=int, default=45, help='Maximum days to expiry')
    parser.add_argument('--save-all', action='store_true', help='Save all analysis results')
    
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Create output directories
    Path('data/processed/greeks_historical').mkdir(parents=True, exist_ok=True)
    Path('data/processed/gamma_exposure').mkdir(parents=True, exist_ok=True)
    Path('data/processed/flow_signals').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Complete Athena Options Analysis System")
    logger.info("=" * 60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Days back: {args.days}")
    logger.info(f"Max DTE: {args.max_dte}")
    
    try:
        system = CompleteOptionsSystem()
        
        for symbol in symbols:
            logger.info(f"\n📈 Processing {symbol}...")
            
            try:
                # Run complete analysis
                contracts_df, greeks_df, exposure_analysis, flow_signals = system.process_complete_options_analysis(
                    symbol, args.days, args.max_dte
                )
                
                # Display results
                if not greeks_df.is_empty():
                    logger.info(f"✅ Analysis complete for {symbol}")
                    logger.info(f"   Contracts analyzed: {contracts_df.shape[0]}")
                    logger.info(f"   Greeks calculated: {greeks_df.shape[0]}")
                    
                    # Gamma exposure summary
                    summary = exposure_analysis.get('summary', {})
                    logger.info(f"   Total Gamma Exposure: {summary.get('total_gamma_exposure', 0):,.0f}")
                    logger.info(f"   Total Vanna Exposure: {summary.get('total_vanna_exposure', 0):,.0f}")
                    logger.info(f"   Current Price: ${summary.get('current_stock_price', 0):.2f}")
                    
                    # Flow signals
                    unusual_vol = flow_signals.get('unusual_volume', {})
                    logger.info(f"   Unusual Volume Contracts: {unusual_vol.get('count', 0)}")
                    
                    put_call = flow_signals.get('put_call_analysis', {})
                    logger.info(f"   Put/Call Ratio: {put_call.get('put_call_ratio', 0):.2f}")
                    
                    # Gamma flip levels
                    flip_info = exposure_analysis.get('flip_levels', {})
                    nearest_flip = flip_info.get('nearest_flip_level', {})
                    if nearest_flip:
                        logger.info(f"   Nearest Gamma Flip: ${nearest_flip.get('flip_strike', 0):.2f} ({nearest_flip.get('flip_distance_pct', 0):.1%})")
                    
                    # Save detailed results if requested
                    if args.save_all:
                        import json
                        
                        # Save exposure analysis
                        exposure_file = Path(f"data/processed/gamma_exposure/{symbol}_exposure_{datetime.now().strftime('%Y%m%d')}.json")
                        with open(exposure_file, 'w') as f:
                            json.dump(exposure_analysis, f, indent=2, default=str)
                        
                        # Save flow signals
                        signals_file = Path(f"data/processed/flow_signals/{symbol}_signals_{datetime.now().strftime('%Y%m%d')}.json")
                        with open(signals_file, 'w') as f:
                            json.dump(flow_signals, f, indent=2, default=str)
                        
                        logger.info(f"💾 Saved detailed analysis to files")
                
                else:
                    logger.warning(f"⚠️  No analysis results for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Complete options analysis finished!")
        logger.info("\nData saved to:")
        logger.info("• data/processed/greeks_historical/ - Historical Greeks for model training")
        logger.info("• data/processed/gamma_exposure/ - Dealer exposure analysis")
        logger.info("• data/processed/flow_signals/ - Options flow signals")
        logger.info("\nNext steps:")
        logger.info("1. Build ML features from Greeks history")
        logger.info("2. Train gamma flip prediction models")
        logger.info("3. Create real-time signal detection")
        logger.info("4. Develop volatility surface analysis")
        
    except Exception as e:
        logger.error(f"System failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()