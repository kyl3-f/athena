#!/usr/bin/env python3
"""
Athena Dealer Gamma Exposure Analyzer
Advanced analysis of dealer gamma exposure and options flow patterns
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class GammaExposureAnalyzer:
    """
    Advanced gamma exposure analysis for market structure insights
    """
    
    def __init__(self):
        self.analysis_results = {}
    
    def load_gamma_data(self, symbol: str, date: str = None) -> pl.DataFrame:
        """Load dealer gamma exposure data"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        gamma_file = Path(f"data/processed/dealer_gamma_exposure/{symbol}_gamma_{date}.parquet")
        
        if gamma_file.exists():
            df = pl.read_parquet(gamma_file)
            logger.info(f"Loaded gamma data for {symbol}: {df.shape[0]} strikes")
            return df
        else:
            logger.warning(f"No gamma data found for {symbol} on {date}")
            return pl.DataFrame()
    
    def identify_gamma_walls(self, gamma_df: pl.DataFrame, threshold_percentile: float = 80) -> pl.DataFrame:
        """
        Identify significant gamma walls (strikes with high gamma concentration)
        
        Args:
            gamma_df: Dealer gamma exposure DataFrame
            threshold_percentile: Percentile threshold for significant gamma
        
        Returns:
            DataFrame of significant gamma walls
        """
        if gamma_df.is_empty():
            return pl.DataFrame()
        
        # Calculate gamma magnitude threshold
        threshold = gamma_df.select(
            pl.col('gamma_magnitude').quantile(threshold_percentile / 100)
        ).item()
        
        # Identify gamma walls
        gamma_walls = gamma_df.filter(
            pl.col('gamma_magnitude') >= threshold
        ).with_columns([
            # Classify wall type
            pl.when(pl.col('total_gamma_exposure') > 0)
            .then('positive_gamma_wall')
            .otherwise('negative_gamma_wall')
            .alias('wall_type'),
            
            # Distance categories
            pl.when(pl.col('strike_distance_pct') <= 0.02)
            .then('very_close')
            .when(pl.col('strike_distance_pct') <= 0.05)
            .then('close')
            .when(pl.col('strike_distance_pct') <= 0.10)
            .then('moderate')
            .otherwise('distant')
            .alias('distance_category'),
            
            # Wall strength
            (pl.col('gamma_magnitude') / pl.col('total_underlying_gamma').abs()).alias('relative_strength')
        ]).sort('gamma_magnitude', descending=True)
        
        logger.info(f"Identified {gamma_walls.shape[0]} gamma walls above {threshold:.0f} threshold")
        return gamma_walls
    
    def analyze_gamma_profile(self, gamma_df: pl.DataFrame) -> Dict:
        """Analyze overall gamma exposure profile"""
        if gamma_df.is_empty():
            return {}
        
        current_price = gamma_df.select('stock_price').head(1).item()
        
        # Calculate key metrics
        analysis = {
            'current_stock_price': current_price,
            'total_gamma_exposure': gamma_df.select(pl.col('total_gamma_exposure').sum()).item(),
            'net_gamma_exposure': gamma_df.select(pl.col('total_gamma_exposure').sum()).item(),
            
            # Gamma distribution
            'positive_gamma_strikes': gamma_df.filter(pl.col('total_gamma_exposure') > 0).shape[0],
            'negative_gamma_strikes': gamma_df.filter(pl.col('total_gamma_exposure') < 0).shape[0],
            
            # Key levels
            'max_positive_gamma': gamma_df.filter(pl.col('total_gamma_exposure') > 0).select(pl.col('total_gamma_exposure').max()).item() or 0,
            'max_negative_gamma': gamma_df.filter(pl.col('total_gamma_exposure') < 0).select(pl.col('total_gamma_exposure').min()).item() or 0,
            
            # Proximity analysis
            'nearby_gamma_5pct': gamma_df.filter(pl.col('strike_distance_pct') <= 0.05).select(pl.col('total_gamma_exposure').sum()).item(),
            'nearby_gamma_10pct': gamma_df.filter(pl.col('strike_distance_pct') <= 0.10).select(pl.col('total_gamma_exposure').sum()).item(),
        }
        
        # Find key strikes
        max_gamma_strike = gamma_df.sort('gamma_magnitude', descending=True).head(1)
        if not max_gamma_strike.is_empty():
            analysis['max_gamma_strike'] = max_gamma_strike.select('strike_price').item()
            analysis['max_gamma_exposure'] = max_gamma_strike.select('total_gamma_exposure').item()
        
        # Zero gamma level (equilibrium)
        above_current = gamma_df.filter(pl.col('strike_price') > current_price)
        below_current = gamma_df.filter(pl.col('strike_price') < current_price)
        
        analysis['gamma_above_current'] = above_current.select(pl.col('total_gamma_exposure').sum()).item() if not above_current.is_empty() else 0
        analysis['gamma_below_current'] = below_current.select(pl.col('total_gamma_exposure').sum()).item() if not below_current.is_empty() else 0
        
        # Gamma bias (positive = dealers long gamma, negative = dealers short gamma)
        analysis['gamma_bias'] = analysis['gamma_above_current'] + analysis['gamma_below_current']
        
        return analysis
    
    def calculate_gamma_impact_zones(self, gamma_df: pl.DataFrame) -> pl.DataFrame:
        """Calculate price zones where gamma effects are strongest"""
        if gamma_df.is_empty():
            return pl.DataFrame()
        
        current_price = gamma_df.select('stock_price').head(1).item()
        
        # Create price zones around current level
        price_zones = []
        zone_width = 0.01  # 1% zones
        
        for i in range(-10, 11):  # -10% to +10% in 1% increments
            zone_center = current_price * (1 + i * zone_width)
            zone_lower = zone_center * (1 - zone_width/2)
            zone_upper = zone_center * (1 + zone_width/2)
            
            # Calculate gamma exposure in this zone
            zone_gamma = gamma_df.filter(
                (pl.col('strike_price') >= zone_lower) & 
                (pl.col('strike_price') < zone_upper)
            ).select(pl.col('total_gamma_exposure').sum()).item() or 0
            
            price_zones.append({
                'zone_center': zone_center,
                'zone_lower': zone_lower,
                'zone_upper': zone_upper,
                'zone_pct_from_current': i * zone_width,
                'gamma_exposure': zone_gamma,
                'gamma_density': zone_gamma / (zone_upper - zone_lower) if zone_upper > zone_lower else 0
            })
        
        zones_df = pl.DataFrame(price_zones)
        
        # Add impact classification
        zones_df = zones_df.with_columns([
            pl.when(pl.col('gamma_exposure').abs() > zones_df.select(pl.col('gamma_exposure').abs().quantile(0.8)).item())
            .then('high_impact')
            .when(pl.col('gamma_exposure').abs() > zones_df.select(pl.col('gamma_exposure').abs().quantile(0.6)).item())
            .then('medium_impact')
            .otherwise('low_impact')
            .alias('impact_level')
        ])
        
        return zones_df
    
    def predict_gamma_flip_levels(self, gamma_df: pl.DataFrame) -> Dict:
        """Predict price levels where gamma exposure flips sign"""
        if gamma_df.is_empty():
            return {}
        
        current_price = gamma_df.select('stock_price').head(1).item()
        
        # Sort by strike price
        sorted_gamma = gamma_df.sort('strike_price')
        
        # Calculate cumulative gamma exposure
        sorted_gamma = sorted_gamma.with_columns([
            pl.col('total_gamma_exposure').cumsum().alias('cumulative_gamma')
        ])
        
        # Find zero-crossing points (gamma flip levels)
        flip_levels = []
        
        for i in range(1, sorted_gamma.shape[0]):
            current_row = sorted_gamma.row(i, named=True)
            prev_row = sorted_gamma.row(i-1, named=True)
            
            current_gamma = current_row['cumulative_gamma']
            prev_gamma = prev_row['cumulative_gamma']
            
            # Check for sign change
            if (current_gamma > 0 and prev_gamma < 0) or (current_gamma < 0 and prev_gamma > 0):
                # Interpolate the exact flip level
                strike1 = prev_row['strike_price']
                strike2 = current_row['strike_price']
                
                # Linear interpolation to find zero crossing
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
        
        # Find the most significant flip levels (closest to current price)
        if flip_levels:
            flip_df = pl.DataFrame(flip_levels)
            flip_df = flip_df.with_columns([
                pl.col('flip_distance_pct').abs().alias('abs_distance')
            ]).sort('abs_distance')
            
            nearest_flip = flip_df.head(1).to_dicts()[0] if not flip_df.is_empty() else {}
            
            return {
                'all_flip_levels': flip_levels,
                'nearest_flip_level': nearest_flip,
                'flip_count': len(flip_levels)
            }
        
        return {'all_flip_levels': [], 'nearest_flip_level': {}, 'flip_count': 0}
    
    def generate_gamma_report(self, symbol: str, date: str = None) -> Dict:
        """Generate comprehensive gamma exposure report"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Generating gamma exposure report for {symbol} on {date}")
        
        # Load data
        gamma_df = self.load_gamma_data(symbol, date)
        
        if gamma_df.is_empty():
            return {'error': f'No gamma data available for {symbol} on {date}'}
        
        # Perform analyses
        report = {
            'symbol': symbol,
            'analysis_date': date,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Overall gamma profile
        report['gamma_profile'] = self.analyze_gamma_profile(gamma_df)
        
        # Gamma walls
        gamma_walls = self.identify_gamma_walls(gamma_df)
        report['gamma_walls'] = {
            'count': gamma_walls.shape[0],
            'top_walls': gamma_walls.head(5).to_dicts() if not gamma_walls.is_empty() else []
        }
        
        # Impact zones
        impact_zones = self.calculate_gamma_impact_zones(gamma_df)
        report['impact_zones'] = {
            'high_impact_zones': impact_zones.filter(pl.col('impact_level') == 'high_impact').to_dicts(),
            'zone_analysis': impact_zones.to_dicts()
        }
        
        # Flip levels
        flip_analysis = self.predict_gamma_flip_levels(gamma_df)
        report['gamma_flip_levels'] = flip_analysis
        
        # Trading implications
        report['trading_implications'] = self._generate_trading_implications(report)
        
        return report
    
    def _generate_trading_implications(self, report: Dict) -> Dict:
        """Generate trading implications from gamma analysis"""
        implications = {}
        
        gamma_profile = report.get('gamma_profile', {})
        gamma_bias = gamma_profile.get('gamma_bias', 0)
        current_price = gamma_profile.get('current_stock_price', 0)
        
        # Overall market structure
        if gamma_bias > 0:
            implications['market_structure'] = 'dealers_long_gamma'
            implications['expected_behavior'] = 'price_stabilization'
            implications['volatility_expectation'] = 'suppressed'
        else:
            implications['market_structure'] = 'dealers_short_gamma'
            implications['expected_behavior'] = 'price_acceleration'
            implications['volatility_expectation'] = 'amplified'
        
        # Key levels to watch
        flip_levels = report.get('gamma_flip_levels', {})
        nearest_flip = flip_levels.get('nearest_flip_level', {})
        
        if nearest_flip:
            implications['key_level'] = nearest_flip.get('flip_strike', current_price)
            implications['key_level_distance'] = nearest_flip.get('flip_distance_pct', 0)
            implications['level_significance'] = 'gamma_flip_point'
        
        # Gamma walls
        walls = report.get('gamma_walls', {}).get('top_walls', [])
        if walls:
            closest_wall = min(walls, key=lambda x: abs(x.get('strike_distance_pct', 1)))
            implications['nearest_gamma_wall'] = closest_wall.get('strike_price', current_price)
            implications['wall_type'] = closest_wall.get('wall_type', 'unknown')
        
        return implications


def main():
    """Main gamma analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze dealer gamma exposure")
    parser.add_argument('--symbol', type=str, default='AAPL', help='Symbol to analyze')
    parser.add_argument('--date', type=str, help='Analysis date (YYYY-MM-DD), default: today')
    parser.add_argument('--save-report', action='store_true', help='Save detailed report to file')
    
    args = parser.parse_args()
    
    analyzer = GammaExposureAnalyzer()
    
    # Generate report
    report = analyzer.generate_gamma_report(args.symbol, args.date)
    
    if 'error' in report:
        logger.error(report['error'])
        return
    
    # Display key findings
    logger.info(f"\n🎯 GAMMA EXPOSURE ANALYSIS - {args.symbol}")
    logger.info("=" * 50)
    
    gamma_profile = report['gamma_profile']
    logger.info(f"Current Price: ${gamma_profile['current_stock_price']:.2f}")
    logger.info(f"Total Gamma Exposure: {gamma_profile['total_gamma_exposure']:,.0f}")
    logger.info(f"Gamma Bias: {gamma_profile['gamma_bias']:,.0f}")
    
    # Market structure implications
    implications = report['trading_implications']
    logger.info(f"\nMarket Structure: {implications.get('market_structure', 'unknown')}")
    logger.info(f"Expected Behavior: {implications.get('expected_behavior', 'unknown')}")
    logger.info(f"Volatility Expectation: {implications.get('volatility_expectation', 'unknown')}")
    
    # Key levels
    if 'key_level' in implications:
        logger.info(f"\nKey Level: ${implications['key_level']:.2f}")
        logger.info(f"Distance: {implications['key_level_distance']:.1%}")
    
    # Gamma walls
    walls = report['gamma_walls']['top_walls']
    if walls:
        logger.info(f"\nTop Gamma Walls:")
        for i, wall in enumerate(walls[:3], 1):
            logger.info(f"  {i}. ${wall['strike_price']:.2f} - {wall['total_gamma_exposure']:,.0f} gamma ({wall['wall_type']})")
    
    # Save detailed report if requested
    if args.save_report:
        import json
        report_file = Path(f"data/processed/dealer_gamma_exposure/{args.symbol}_gamma_report_{args.date or datetime.now().strftime('%Y-%m-%d')}.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n📊 Detailed report saved to: {report_file}")


if __name__ == "__main__":
    main()