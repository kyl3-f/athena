#!/usr/bin/env python3
"""
Athena Enhanced Options Historical Downloader
Downloads and organizes options data with proper underlying stock linking
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import polars as pl

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from config.polygon_config import PolygonConfig
from ingestion.polygon_client import PolygonClient
from config.settings import trading_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/download_options.log')
    ]
)
logger = logging.getLogger(__name__)


class OptionsDataManager:
    """
    Enhanced options data manager for scalable options flow analysis
    Includes dealer gamma exposure (DEX/GEX) analysis and full option chains
    """
    
    def __init__(self, client: PolygonClient):
        self.client = client
        self.contracts_cache = {}
        self.trading_calendar = None
        
    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        """Get trading days from Polygon for accurate DTE calculation"""
        if self.trading_calendar is None:
            try:
                calendar_data = self.client.get_market_calendar(start_date, end_date)
                if calendar_data:
                    # Extract trading days
                    trading_days = []
                    for day in calendar_data:
                        if isinstance(day, dict):
                            date_value = day.get('date') or day.get('day') or day.get('trading_date')
                            status = day.get('status', 'unknown')
                            if date_value and status in ['open', 'early_close']:
                                trading_days.append(date_value)
                    self.trading_calendar = sorted(trading_days)
                    logger.info(f"Loaded {len(self.trading_calendar)} trading days")
                else:
                    # Fallback: generate weekdays
                    logger.warning("Using weekday fallback for trading calendar")
                    self.trading_calendar = self._generate_weekdays(start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to get trading calendar: {e}")
                self.trading_calendar = self._generate_weekdays(start_date, end_date)
        
        return self.trading_calendar
    
    def _generate_weekdays(self, start_date: str, end_date: str) -> List[str]:
        """Generate weekdays as fallback trading calendar"""
        from datetime import datetime, timedelta
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        weekdays = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # Monday=0, Friday=4
                weekdays.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        return weekdays
    
    def calculate_trading_days_to_expiry(self, current_date: str, expiration_date: str) -> int:
        """Calculate trading days to expiry using market calendar"""
        try:
            # Get trading calendar from current date to expiration
            extended_end = (datetime.strptime(expiration_date, '%Y-%m-%d') + timedelta(days=5)).strftime('%Y-%m-%d')
            trading_days = self.get_trading_calendar(current_date, extended_end)
            
            # Count trading days between current and expiration (inclusive)
            dte = 0
            for day in trading_days:
                if current_date < day <= expiration_date:
                    dte += 1
            
            return max(0, dte)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Failed to calculate trading DTE: {e}")
            # Fallback to calendar days / 7 * 5 approximation
            try:
                current = datetime.strptime(current_date, '%Y-%m-%d')
                expiry = datetime.strptime(expiration_date, '%Y-%m-%d')
                calendar_days = (expiry - current).days
                return max(0, int(calendar_days * 5 / 7))  # Approximate trading days
            except:
                return 0
    
    def get_full_options_chain(
        self, 
        underlying: str,
        reference_date: str = None,
        max_dte: int = 60,
        include_expired: bool = False
    ) -> pl.DataFrame:
        """
        Get complete options chain for underlying with proper DTE calculation
        
        Args:
            underlying: Stock symbol
            reference_date: Date for DTE calculation (default: today)
            max_dte: Maximum trading days to expiry
            include_expired: Include expired options
        
        Returns:
            Complete options chain with calculated DTE and Greeks
        """
        if reference_date is None:
            reference_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Getting full options chain for {underlying} (ref date: {reference_date})")
        
        try:
            # Get all contracts for underlying (multiple API calls may be needed)
            all_contracts = []
            limit = 1000
            
            # For now, get contracts in one call and filter
            contracts_df = self.client.get_options_contracts(
                underlying_ticker=underlying,
                limit=limit
            )
            
            if contracts_df.is_empty():
                logger.warning(f"No contracts found for {underlying}")
                return pl.DataFrame()
            
            all_contracts.append(contracts_df)
            
            # Combine all contract data
            contracts_df = pl.concat(all_contracts)
            logger.info(f"Found {contracts_df.shape[0]} total contracts for {underlying}")
            
            # Calculate trading days to expiry
            contracts_df = self._add_trading_dte(contracts_df, reference_date)
            
            # Filter by DTE if specified
            if max_dte is not None:
                if not include_expired:
                    contracts_df = contracts_df.filter(
                        (pl.col('trading_dte') >= 0) & (pl.col('trading_dte') <= max_dte)
                    )
                else:
                    contracts_df = contracts_df.filter(pl.col('trading_dte') <= max_dte)
            
            # Add moneyness and other calculated fields
            contracts_df = self._enhance_contracts_data(contracts_df, underlying, reference_date)
            
            logger.info(f"Final chain: {contracts_df.shape[0]} contracts within {max_dte} trading days")
            return contracts_df
            
        except Exception as e:
            logger.error(f"Failed to get options chain for {underlying}: {e}")
            return pl.DataFrame()
    
    def _add_trading_dte(self, contracts_df: pl.DataFrame, reference_date: str) -> pl.DataFrame:
        """Add trading days to expiry calculation"""
        if contracts_df.is_empty():
            return contracts_df
        
        # Calculate trading DTE for each contract
        dte_values = []
        
        for exp_date in contracts_df['expiration_date'].to_list():
            if exp_date:
                dte = self.calculate_trading_days_to_expiry(reference_date, str(exp_date))
                dte_values.append(dte)
            else:
                dte_values.append(0)
        
        # Add trading DTE column
        contracts_df = contracts_df.with_columns([
            pl.Series('trading_dte', dte_values)
        ])
        
        return contracts_df
    
    def _enhance_contracts_data(self, contracts_df: pl.DataFrame, underlying: str, reference_date: str) -> pl.DataFrame:
        """Enhance contracts with additional calculated fields"""
        if contracts_df.is_empty():
            return contracts_df
        
        # Get current stock price (from recent stock data)
        stock_price = self._get_current_stock_price(underlying, reference_date)
        
        contracts_df = contracts_df.with_columns([
            # Current stock price
            pl.lit(stock_price).alias('current_stock_price'),
            
            # Moneyness calculations
            (pl.lit(stock_price) / pl.col('strike_price')).alias('moneyness_ratio'),
            
            # ITM/OTM classification
            pl.when(
                (pl.col('contract_type') == 'call') & (pl.lit(stock_price) > pl.col('strike_price'))
            ).then('ITM')
            .when(
                (pl.col('contract_type') == 'put') & (pl.lit(stock_price) < pl.col('strike_price'))
            ).then('ITM')
            .when(
                (pl.lit(stock_price) - pl.col('strike_price')).abs() / pl.lit(stock_price) < 0.02
            ).then('ATM')
            .otherwise('OTM').alias('moneyness'),
            
            # Distance from ATM
            ((pl.lit(stock_price) - pl.col('strike_price')).abs() / pl.lit(stock_price)).alias('distance_from_atm'),
            
            # Contract notional value
            (pl.col('last_price').fill_null(0) * 100).alias('contract_value'),  # Assuming 100 multiplier
            
            # Volume metrics
            (pl.col('volume').fill_null(0) / pl.col('open_interest').fill_null(1)).alias('volume_oi_ratio'),
        ])
        
        return contracts_df
    
    def _get_current_stock_price(self, underlying: str, reference_date: str) -> float:
        """Get current/recent stock price for calculations"""
        try:
            # Try to get from existing stock data files
            stock_files = list(Path(f"data/raw/stocks/aggregates/minute").glob(f"{underlying}_minute_*.parquet"))
            
            if stock_files:
                stock_df = pl.read_parquet(stock_files[-1])  # Most recent file
                if not stock_df.is_empty():
                    # Get price closest to reference date
                    closest_price = stock_df.filter(
                        pl.col('timestamp_utc').dt.date() <= datetime.strptime(reference_date, '%Y-%m-%d').date()
                    ).select('close').tail(1)
                    
                    if not closest_price.is_empty():
                        return float(closest_price.item())
            
            # Fallback: try to get from API
            yesterday = (datetime.strptime(reference_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            stock_data = self.client.get_stock_aggregates(
                underlying, 
                timespan='day',
                from_date=yesterday,
                to_date=reference_date,
                limit=1
            )
            
            if not stock_data.is_empty():
                return float(stock_data.select('close').tail(1).item())
            
        except Exception as e:
            logger.warning(f"Could not get stock price for {underlying}: {e}")
        
        # Ultimate fallback
        return 100.0  # Placeholder value
    
    def calculate_dealer_gamma_exposure(
        self,
        options_chain: pl.DataFrame,
        underlying: str,
        reference_date: str
    ) -> pl.DataFrame:
        """
        Calculate dealer gamma exposure (DEX/GEX) assuming:
        - Dealers are net short puts (negative gamma)
        - Dealers are net long calls (positive gamma)
        
        Returns DataFrame with gamma exposure by strike and total exposure
        """
        logger.info(f"Calculating dealer gamma exposure for {underlying}")
        
        if options_chain.is_empty():
            return pl.DataFrame()
        
        # Filter for liquid options with gamma data
        liquid_options = options_chain.filter(
            (pl.col('open_interest').fill_null(0) > 0) &
            (pl.col('gamma').is_not_null()) &
            (pl.col('trading_dte') > 0)
        )
        
        if liquid_options.is_empty():
            logger.warning(f"No liquid options with gamma data for {underlying}")
            return pl.DataFrame()
        
        # Calculate dealer positioning
        dealer_exposure = liquid_options.with_columns([
            # Dealer gamma exposure calculation
            # Puts: Dealers are typically net short (negative gamma exposure)
            # Calls: Dealers are typically net long (positive gamma exposure)
            pl.when(pl.col('contract_type') == 'put')
            .then(-pl.col('gamma').fill_null(0) * pl.col('open_interest').fill_null(0) * 100)  # Short puts = negative gamma
            .when(pl.col('contract_type') == 'call')  
            .then(pl.col('gamma').fill_null(0) * pl.col('open_interest').fill_null(0) * 100)   # Long calls = positive gamma
            .otherwise(0)
            .alias('dealer_gamma_exposure'),
            
            # Net positioning by contract type
            pl.when(pl.col('contract_type') == 'put')
            .then(-pl.col('open_interest').fill_null(0))  # Dealers short puts
            .when(pl.col('contract_type') == 'call')
            .then(pl.col('open_interest').fill_null(0))   # Dealers long calls  
            .otherwise(0)
            .alias('dealer_net_position'),
            
            # Notional exposure
            (pl.col('open_interest').fill_null(0) * pl.col('last_price').fill_null(0) * 100).alias('notional_exposure')
        ])
        
        # Aggregate by strike for gamma wall analysis
        gamma_by_strike = dealer_exposure.group_by('strike_price').agg([
            pl.col('dealer_gamma_exposure').sum().alias('total_gamma_exposure'),
            pl.col('dealer_net_position').sum().alias('net_dealer_position'),
            pl.col('notional_exposure').sum().alias('total_notional'),
            pl.col('contract_type').first().alias('primary_type'),  # Dominant type at strike
            pl.col('current_stock_price').first().alias('stock_price')
        ]).sort('strike_price')
        
        # Add gamma wall analysis
        gamma_by_strike = gamma_by_strike.with_columns([
            # Distance from current price
            (pl.col('strike_price') - pl.col('stock_price')).alias('strike_distance'),
            ((pl.col('strike_price') - pl.col('stock_price')).abs() / pl.col('stock_price')).alias('strike_distance_pct'),
            
            # Gamma exposure impact
            pl.col('total_gamma_exposure').abs().alias('gamma_magnitude'),
        ])
        
        # Calculate key levels
        total_gamma_exposure = dealer_exposure.select(pl.col('dealer_gamma_exposure').sum()).item()
        max_gamma_strike = gamma_by_strike.sort('gamma_magnitude', descending=True).select('strike_price').head(1)
        
        # Add summary metrics
        summary_metrics = {
            'underlying': underlying,
            'reference_date': reference_date,
            'total_dealer_gamma_exposure': total_gamma_exposure,
            'net_gamma_exposure': gamma_by_strike.select(pl.col('total_gamma_exposure').sum()).item(),
            'max_gamma_strike': max_gamma_strike.item() if not max_gamma_strike.is_empty() else None,
            'current_stock_price': dealer_exposure.select('current_stock_price').head(1).item(),
            'options_count': dealer_exposure.shape[0]
        }
        
        logger.info(f"DEX Summary - Total Gamma: {total_gamma_exposure:,.0f}, Max Gamma Strike: {summary_metrics['max_gamma_strike']}")
        
        # Return both detailed and summary data
        gamma_by_strike = gamma_by_strike.with_columns([
            pl.lit(underlying).alias('underlying'),
            pl.lit(reference_date).alias('reference_date'),
            pl.lit(total_gamma_exposure).alias('total_underlying_gamma')
        ])
        
        return gamma_by_strike
    
    def download_complete_options_dataset(
        self,
        underlying: str,
        start_date: str,
        end_date: str,
        reference_date: str = None,
        max_dte: int = 60,
        include_gamma_analysis: bool = True
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Download complete options dataset with dealer gamma exposure analysis
        
        Returns:
            Tuple of (full_options_chain, options_trades_df, dealer_gamma_exposure)
        """
        if reference_date is None:
            reference_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Downloading complete options dataset for {underlying}")
        
        # 1. Get full options chain with proper DTE calculation
        options_chain = self.get_full_options_chain(
            underlying, 
            reference_date=reference_date,
            max_dte=max_dte
        )
        
        if options_chain.is_empty():
            return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
        
        logger.info(f"Got options chain: {options_chain.shape[0]} contracts")
        
        # 2. Calculate dealer gamma exposure
        dealer_gamma_df = pl.DataFrame()
        if include_gamma_analysis:
            dealer_gamma_df = self.calculate_dealer_gamma_exposure(
                options_chain, 
                underlying, 
                reference_date
            )
        
        # 3. Download options trades for liquid contracts (sample for performance)
        liquid_contracts = options_chain.filter(
            (pl.col('volume').fill_null(0) > 10) | 
            (pl.col('open_interest').fill_null(0) > 50)
        ).head(100)  # Limit for API efficiency
        
        options_trades_df = self.download_options_trades_for_contracts(
            liquid_contracts, 
            underlying, 
            start_date, 
            end_date
        )
        
        # 4. Load corresponding stock data and create combined dataset
        stock_files = list(Path(f"data/raw/stocks/aggregates/minute").glob(f"{underlying}_minute_*.parquet"))
        
        if stock_files and not options_trades_df.is_empty():
            stock_df = pl.read_parquet(stock_files[0])
            combined_df = self.create_options_stock_dataset(options_trades_df, stock_df, underlying)
            
            # Add dealer gamma context to combined data
            if not dealer_gamma_df.is_empty():
                # Add nearest gamma wall info
                current_price = stock_df.select('close').tail(1).item()
                nearest_gamma_wall = dealer_gamma_df.filter(
                    pl.col('strike_distance_pct') < 0.05  # Within 5% of current price
                ).sort('gamma_magnitude', descending=True).head(1)
                
                if not nearest_gamma_wall.is_empty():
                    gamma_wall_strike = nearest_gamma_wall.select('strike_price').item()
                    combined_df = combined_df.with_columns([
                        pl.lit(gamma_wall_strike).alias('nearest_gamma_wall'),
                        (pl.col('stock_close') - pl.lit(gamma_wall_strike)).alias('distance_to_gamma_wall')
                    ])
        else:
            combined_df = options_trades_df
        
        return options_chain, combined_df, dealer_gamma_df
    
    def download_options_trades_for_contracts(
        self,
        contracts_df: pl.DataFrame,
        underlying: str,
        start_date: str,
        end_date: str
    ) -> pl.DataFrame:
        """Download options trades for specific contracts"""
        
        if contracts_df.is_empty():
            return pl.DataFrame()
        
        logger.info(f"Downloading options trades for {contracts_df.shape[0]} {underlying} contracts")
        
        all_trades = []
        contract_count = 0
        
        for contract_row in contracts_df.iter_rows(named=True):
            contract_ticker = contract_row.get('ticker', '')
            
            if not contract_ticker:
                continue
            
            contract_count += 1
            logger.info(f"  Processing contract {contract_count}/{contracts_df.shape[0]}: {contract_ticker}")
            
            try:
                # Download aggregates (more reliable than trades for most use cases)
                trades_df = self.client.get_options_aggregates(
                    options_ticker=contract_ticker,
                    multiplier=1,
                    timespan='minute',
                    from_date=start_date,
                    to_date=end_date,
                    limit=10000
                )
                
                if not trades_df.is_empty():
                    # Add contract metadata
                    trades_df = trades_df.with_columns([
                        pl.lit(contract_row.get('strike_price', 0)).alias('strike_price'),
                        pl.lit(contract_row.get('contract_type', '')).alias('contract_type'),
                        pl.lit(contract_row.get('expiration_date', '')).alias('expiration_date'),
                        pl.lit(contract_row.get('open_interest', 0)).alias('open_interest'),
                    ])
                    
                    all_trades.append(trades_df)
                    logger.info(f"    Downloaded {trades_df.shape[0]} minute bars")
                else:
                    logger.warning(f"    No data for {contract_ticker}")
                    
            except Exception as e:
                logger.warning(f"    Failed to download {contract_ticker}: {e}")
                continue
        
        if all_trades:
            combined_df = pl.concat(all_trades)
            logger.info(f"Combined {len(all_trades)} contracts into {combined_df.shape[0]} total records")
            return combined_df
        else:
            logger.warning(f"No options data downloaded for {underlying}")
            return pl.DataFrame()
    
    def create_options_stock_dataset(
        self,
        options_df: pl.DataFrame,
        stock_df: pl.DataFrame,
        underlying: str
    ) -> pl.DataFrame:
        """Create time-aligned options + stock dataset"""
        
        if options_df.is_empty() or stock_df.is_empty():
            logger.warning(f"Empty data for {underlying} - cannot create combined dataset")
            return pl.DataFrame()
        
        logger.info(f"Creating combined options-stock dataset for {underlying}")
        
        # Prepare stock data for joining
        stock_join = stock_df.select([
            'timestamp_utc',
            'open', 'high', 'low', 'close', 'volume'
        ]).rename({
            'open': 'stock_open',
            'high': 'stock_high', 
            'low': 'stock_low',
            'close': 'stock_close',
            'volume': 'stock_volume'
        })
        
        # Join options with stock data on timestamp
        combined_df = options_df.join(
            stock_join,
            on='timestamp_utc',
            how='left'
        )
        
        # Calculate additional cross-asset features
        combined_df = combined_df.with_columns([
            # Moneyness
            (pl.col('stock_close') / pl.col('strike_price')).alias('moneyness'),
            
            # Options to stock volume ratio
            (pl.col('volume') / pl.col('stock_volume')).alias('options_stock_volume_ratio'),
            
            # Time value (simplified)
            pl.max_horizontal([
                pl.col('close') - pl.max_horizontal([
                    pl.when(pl.col('contract_type') == 'call')
                    .then(pl.col('stock_close') - pl.col('strike_price'))
                    .otherwise(pl.col('strike_price') - pl.col('stock_close')),
                    pl.lit(0)
                ]), 
                pl.lit(0)
            ]).alias('time_value'),
        ])
        
        logger.info(f"Created combined dataset: {combined_df.shape[0]} records with {combined_df.shape[1]} columns")
        return combined_df


def main():
    """Main options download function with enhanced gamma analysis"""
    parser = argparse.ArgumentParser(description="Download options data with dealer gamma exposure analysis")
    
    parser.add_argument(
        '--symbols',
        type=str,
        default='AAPL,SPY',
        help='Comma-separated list of underlying symbols'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=5,
        help='Number of days of historical options trades'
    )
    
    parser.add_argument(
        '--max-dte',
        type=int,
        default=60,
        help='Maximum trading days to expiry for options chain'
    )
    
    parser.add_argument(
        '--reference-date',
        type=str,
        help='Reference date for DTE calculation (YYYY-MM-DD), default: today'
    )
    
    parser.add_argument(
        '--gamma-analysis',
        action='store_true',
        default=True,
        help='Include dealer gamma exposure analysis'
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    reference_date = args.reference_date or datetime.now().strftime('%Y-%m-%d')
    
    # Calculate date range for trades
    end_date = datetime.strptime(reference_date, '%Y-%m-%d').date()
    start_date = end_date - timedelta(days=args.days)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Create output directories
    Path('data/raw/options/chains').mkdir(parents=True, exist_ok=True)
    Path('data/raw/options/trades').mkdir(parents=True, exist_ok=True)
    Path('data/processed/options_stock_combined').mkdir(parents=True, exist_ok=True)
    Path('data/processed/dealer_gamma_exposure').mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Athena Enhanced Options Data Downloader")
    logger.info("=" * 60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Reference date: {reference_date}")
    logger.info(f"Trades date range: {start_str} to {end_str}")
    logger.info(f"Max trading DTE: {args.max_dte}")
    logger.info(f"Gamma analysis: {args.gamma_analysis}")
    
    try:
        config = PolygonConfig()
        
        with PolygonClient(config) as client:
            manager = OptionsDataManager(client)
            
            for symbol in symbols:
                logger.info(f"\n📈 Processing {symbol}...")
                
                try:
                    # Download complete dataset with gamma analysis
                    options_chain, combined_df, gamma_df = manager.download_complete_options_dataset(
                        symbol,
                        start_str,
                        end_str,
                        reference_date=reference_date,
                        max_dte=args.max_dte,
                        include_gamma_analysis=args.gamma_analysis
                    )
                    
                    # Save options chain
                    if not options_chain.is_empty():
                        chain_file = Path(f"data/raw/options/chains/{symbol}_chain_{reference_date}.parquet")
                        options_chain.write_parquet(chain_file)
                        logger.info(f"💾 Saved options chain: {chain_file}")
                        
                        # Show chain summary
                        chain_summary = options_chain.group_by(['contract_type', 'moneyness']).agg([
                            pl.count().alias('contracts'),
                            pl.col('open_interest').sum().alias('total_oi'),
                            pl.col('volume').sum().alias('total_volume')
                        ])
                        logger.info(f"📊 Chain summary for {symbol}:")
                        print(chain_summary)
                    
                    # Save combined trades data
                    if not combined_df.is_empty():
                        combined_file = Path(f"data/processed/options_stock_combined/{symbol}_combined_{start_str}_{end_str}.parquet")
                        combined_df.write_parquet(combined_file)
                        logger.info(f"💾 Saved combined data: {combined_file}")
                    
                    # Save dealer gamma exposure
                    if not gamma_df.is_empty():
                        gamma_file = Path(f"data/processed/dealer_gamma_exposure/{symbol}_gamma_{reference_date}.parquet")
                        gamma_df.write_parquet(gamma_file)
                        logger.info(f"💾 Saved gamma exposure: {gamma_file}")
                        
                        # Show gamma summary
                        logger.info(f"🎯 Dealer Gamma Summary for {symbol}:")
                        total_gamma = gamma_df.select(pl.col('total_gamma_exposure').sum()).item()
                        max_gamma_strike = gamma_df.sort('gamma_magnitude', descending=True).head(1)
                        logger.info(f"   Total Dealer Gamma Exposure: {total_gamma:,.0f}")
                        
                        if not max_gamma_strike.is_empty():
                            strike = max_gamma_strike.select('strike_price').item()
                            exposure = max_gamma_strike.select('total_gamma_exposure').item()
                            logger.info(f"   Max Gamma Strike: ${strike:.2f} (Exposure: {exposure:,.0f})")
                        
                        # Show top gamma strikes
                        top_gamma = gamma_df.sort('gamma_magnitude', descending=True).head(5).select([
                            'strike_price', 'total_gamma_exposure', 'strike_distance', 'primary_type'
                        ])
                        logger.info(f"   Top Gamma Strikes:")
                        print(top_gamma)
                    
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")
                    continue
            
            logger.info("\n" + "="*60)
            logger.info("🎉 Enhanced options download completed!")
            logger.info("\nData saved to:")
            logger.info("• data/raw/options/chains/ - Full options chains with DTE")
            logger.info("• data/processed/dealer_gamma_exposure/ - DEX/GEX analysis")
            logger.info("• data/processed/options_stock_combined/ - Cross-asset datasets")
            logger.info("\nNext steps:")
            logger.info("1. Analyze dealer gamma exposure patterns")
            logger.info("2. Build options flow feature engineering")
            logger.info("3. Create gamma wall detection algorithms")
            logger.info("4. Develop cross-asset momentum signals")
            
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()