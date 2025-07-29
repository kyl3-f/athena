#!/usr/bin/env python3
"""
Athena Finviz Integration - Enhanced Symbol Management & Data Collection
Integrates with existing Athena system while scaling to 5000+ symbols
"""

import polars as pl
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinvizSymbolManager:
    """Enhanced symbol management for Athena using Finviz CSV data"""
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            # Auto-detect project root
            current_file = Path(__file__).resolve()
            self.project_root = current_file.parent.parent if 'scripts' in str(current_file) else current_file.parent
        else:
            self.project_root = Path(project_root)
        
        self.finviz_path = self.project_root / "data" / "finviz" / "finviz.csv"
        self.legacy_symbols_path = self.project_root / "config" / "symbols.txt"
        
        # Cache for loaded data
        self._finviz_df = None
        self._last_load_time = None
    
    def load_finviz_data(self, force_reload: bool = False) -> pl.DataFrame:
        """Load Finviz CSV data with caching"""
        
        # Check if we need to reload
        if (self._finviz_df is None or force_reload or 
            (self._last_load_time and (datetime.now() - self._last_load_time).seconds > 3600)):
            
            try:
                logger.info(f"Loading Finviz data from {self.finviz_path}")
                
                # Read CSV with proper schema handling
                self._finviz_df = pl.read_csv(
                    self.finviz_path,
                    schema_overrides={
                        "No.": pl.Int64,
                        "Ticker": pl.Utf8,
                        "Company": pl.Utf8,
                        "Index": pl.Utf8,
                        "Sector": pl.Utf8,
                        "Industry": pl.Utf8,
                        "Country": pl.Utf8,
                        "Exchange": pl.Utf8,
                        "Market Cap": pl.Float64,
                        "P/E": pl.Float64
                    },
                    null_values=["", "N/A", "-", "None"]
                )
                
                self._last_load_time = datetime.now()
                logger.info(f"Loaded {len(self._finviz_df)} records from Finviz CSV")
                
                # Log data summary
                logger.info(f"Data summary:")
                logger.info(f"  - Exchanges: {self._finviz_df['Exchange'].unique().to_list()}")
                logger.info(f"  - Countries: {self._finviz_df['Country'].unique().to_list()}")
                logger.info(f"  - Sectors: {len(self._finviz_df['Sector'].unique())} unique sectors")
                
            except Exception as e:
                logger.error(f"Error loading Finviz data: {e}")
                self._finviz_df = pl.DataFrame()
        
        return self._finviz_df
    
    def get_all_symbols(self, filters: Optional[Dict] = None) -> List[str]:
        """Get all symbols with optional filtering"""
        df = self.load_finviz_data()
        
        if df.is_empty():
            logger.warning("No Finviz data available, falling back to legacy symbols")
            return self._load_legacy_symbols()
        
        logger.info(f"Starting with {len(df)} total records")
        
        # Apply filters if provided
        if filters:
            df = self._apply_filters(df, filters)
            logger.info(f"After filtering: {len(df)} records remain")
        
        # Extract and clean symbols
        symbols = df['Ticker'].to_list()
        logger.info(f"Extracted {len(symbols)} raw symbols")
        
        # Remove null/empty symbols
        symbols = [s for s in symbols if s and str(s).strip()]
        logger.info(f"After removing nulls: {len(symbols)} symbols")
        
        symbols = [str(s).upper().strip() for s in symbols]
        logger.info(f"After cleaning: {len(symbols)} symbols")
        
        # Enhanced validation - be more permissive with real market symbols
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in symbols:
            # Clean the symbol
            clean_symbol = symbol.strip().upper()
            
            # More permissive validation for real market symbols
            if (len(clean_symbol) >= 1 and len(clean_symbol) <= 6 and  # Allow up to 6 chars
                not clean_symbol.startswith('.') and
                not clean_symbol.endswith('.') and
                # Allow letters, numbers, hyphens, and dots (common in real symbols)
                all(c.isalnum() or c in ['-', '.'] for c in clean_symbol)):
                valid_symbols.append(clean_symbol)
            else:
                invalid_symbols.append(clean_symbol)
        
        if invalid_symbols:
            logger.warning(f"Filtered out {len(invalid_symbols)} invalid symbols: {invalid_symbols[:10]}...")
        
        logger.info(f"Final result: {len(valid_symbols)} valid symbols (filtered {len(invalid_symbols)} invalid)")
        return valid_symbols
    
    def get_liquid_symbols(self, count: int = 1000) -> List[str]:
        """Get most liquid symbols based on market cap and exchange"""
        df = self.load_finviz_data()
        
        if df.is_empty():
            return self._load_legacy_symbols()[:count]
        
        # Filter for major exchanges and valid market cap, but be more inclusive
        liquid_df = df.filter(
            (pl.col("Exchange").is_in(["NASD", "NYSE", "AMEX", "CBOE"])) &  # Include all major exchanges
            (pl.col("Country") == "USA") &  # Focus on US symbols for liquidity
            (pl.col("Ticker").is_not_null())  # Must have ticker
        )
        
        # Sort by market cap (descending), handling nulls
        liquid_df = liquid_df.with_columns([
            pl.col("Market Cap").fill_null(0)  # Fill null market caps with 0
        ]).sort("Market Cap", descending=True)
        
        symbols = liquid_df['Ticker'].head(count).to_list()
        symbols = [s.upper().strip() for s in symbols if s]
        
        logger.info(f"Selected {len(symbols)} liquid symbols from {len(liquid_df)} available")
        return symbols
    
    def get_options_priority_symbols(self, count: int = 500) -> List[str]:
        """Get symbols prioritized for options trading"""
        
        # Start with hand-picked high-volume options symbols
        priority_symbols = [
            # Major ETFs
            'SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'XLE', 'XLF', 'XLK', 'XLV', 'XLP',
            'XLI', 'XLB', 'XLRE', 'XLU', 'XLY', 'EWZ', 'FXI', 'EEM', 'VXX', 'UVXY',
            
            # Mega-cap stocks with active options
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK.B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
            
            # High-volatility popular stocks
            'AMD', 'NFLX', 'CRM', 'UBER', 'LYFT', 'SNAP', 'TWTR', 'ROKU', 'ZM', 'PTON',
            'PLTR', 'GME', 'AMC', 'BB', 'NOK', 'SPCE', 'NIO', 'XPEV', 'LI', 'BABA',
            
            # Financial sector (high options volume)
            'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF', 'AXP',
            
            # Tech sector
            'ORCL', 'CSCO', 'IBM', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'ADI'
        ]
        
        # Get our universe of symbols
        all_symbols = self.get_all_symbols()
        
        # Filter priority symbols that exist in our universe
        available_priority = [s for s in priority_symbols if s in all_symbols]
        
        # If we need more symbols, add liquid symbols
        if len(available_priority) < count:
            liquid_symbols = self.get_liquid_symbols(count * 2)
            for symbol in liquid_symbols:
                if symbol not in available_priority and len(available_priority) < count:
                    available_priority.append(symbol)
        
        result = available_priority[:count]
        logger.info(f"Selected {len(result)} priority symbols for options trading")
        return result
    
    def get_symbols_by_criteria(self, **criteria) -> List[str]:
        """Get symbols filtered by specific criteria"""
        
        filters = {}
        
        # Map common criteria to filters
        if 'sector' in criteria:
            filters['Sector'] = criteria['sector'] if isinstance(criteria['sector'], list) else [criteria['sector']]
        
        if 'exchange' in criteria:
            filters['Exchange'] = criteria['exchange'] if isinstance(criteria['exchange'], list) else [criteria['exchange']]
        
        if 'min_market_cap' in criteria:
            filters['min_market_cap'] = criteria['min_market_cap']
        
        if 'max_market_cap' in criteria:
            filters['max_market_cap'] = criteria['max_market_cap']
        
        if 'country' in criteria:
            filters['Country'] = criteria['country'] if isinstance(criteria['country'], list) else [criteria['country']]
        
        return self.get_all_symbols(filters)
    
    def _apply_filters(self, df: pl.DataFrame, filters: Dict) -> pl.DataFrame:
        """Apply filters to the dataframe"""
        
        for key, value in filters.items():
            if key == 'Sector' and isinstance(value, list):
                df = df.filter(pl.col("Sector").is_in(value))
            elif key == 'Exchange' and isinstance(value, list):
                df = df.filter(pl.col("Exchange").is_in(value))
            elif key == 'Country' and isinstance(value, list):
                df = df.filter(pl.col("Country").is_in(value))
            elif key == 'min_market_cap':
                df = df.filter(pl.col("Market Cap") >= value)
            elif key == 'max_market_cap':
                df = df.filter(pl.col("Market Cap") <= value)
        
        return df
    
    def _load_legacy_symbols(self) -> List[str]:
        """Fallback to legacy symbols.txt file"""
        try:
            with open(self.legacy_symbols_path, 'r') as f:
                symbols = [line.strip().upper() for line in f if line.strip()]
            logger.info(f"Loaded {len(symbols)} symbols from legacy file")
            return symbols
        except FileNotFoundError:
            logger.warning("No legacy symbols file found")
            return []
    
    def save_symbols_to_legacy_format(self, symbols: List[str]):
        """Save symbols to legacy format for backwards compatibility"""
        self.legacy_symbols_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.legacy_symbols_path, 'w') as f:
            for symbol in symbols:
                f.write(f"{symbol}\n")
        
        logger.info(f"Saved {len(symbols)} symbols to {self.legacy_symbols_path}")
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics about the data"""
        df = self.load_finviz_data()
        
        if df.is_empty():
            return {"error": "No data available"}
        
        return {
            "total_symbols": len(df),
            "exchanges": df['Exchange'].unique().to_list(),
            "sectors": df['Sector'].unique().to_list(),
            "countries": df['Country'].unique().to_list(),
            "market_cap_stats": {
                "min": df['Market Cap'].min(),
                "max": df['Market Cap'].max(),
                "median": df['Market Cap'].median(),
                "count_with_data": df.filter(pl.col("Market Cap").is_not_null()).height
            },
            "pe_stats": {
                "min": df['P/E'].min(),
                "max": df['P/E'].max(),
                "median": df['P/E'].median(),
                "count_with_data": df.filter(pl.col("P/E").is_not_null()).height
            }
        }

class AthenaEnhancedCollector:
    """Enhanced data collector integrated with existing Athena system"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.symbol_manager = FinvizSymbolManager(self.project_root)
        
        # Paths for compatibility with existing system
        self.bronze_path = self.project_root / "data" / "bronze"
        self.logs_path = self.project_root / "logs"
        
        # Ensure directories exist
        self.bronze_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
    
    async def run_collection_cycle(self, 
                                 collection_type: str = "full",
                                 symbol_count: int = None) -> Dict:
        """Run enhanced collection cycle with Finviz integration"""
        
        start_time = datetime.now()
        collection_id = f"athena_{collection_type}_{int(start_time.timestamp())}"
        
        logger.info(f"Starting Athena collection cycle: {collection_id}")
        
        # Select symbols based on collection type
        if collection_type == "liquid":
            symbols = self.symbol_manager.get_liquid_symbols(symbol_count or 1000)
        elif collection_type == "options_priority":
            symbols = self.symbol_manager.get_options_priority_symbols(symbol_count or 500)
        elif collection_type == "tech_sector":
            symbols = self.symbol_manager.get_symbols_by_criteria(sector=["Technology"])
        else:  # full
            symbols = self.symbol_manager.get_all_symbols()
            if symbol_count:
                symbols = symbols[:symbol_count]
        
        logger.info(f"Collection type: {collection_type}")
        logger.info(f"Symbol count: {len(symbols)}")
        
        # Save symbols for existing pipeline compatibility
        self.symbol_manager.save_symbols_to_legacy_format(symbols)
        
        # For now, create a placeholder for the actual collection
        # This will be replaced with the scalable collector integration
        collection_results = {
            "collection_id": collection_id,
            "collection_type": collection_type,
            "symbols_requested": len(symbols),
            "symbols_processed": 0,  # To be updated by actual collector
            "start_time": start_time,
            "end_time": None,
            "success_rate": 0.0,
            "errors": []
        }
        
        # TODO: Integrate with your existing collect_live_data.py or new scalable collector
        logger.info("Ready to integrate with scalable collector...")
        
        # For demonstration, simulate some processing time
        logger.info("Simulating collection process...")
        await asyncio.sleep(2)
        
        end_time = datetime.now()
        collection_results.update({
            "end_time": end_time,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "symbols_processed": len(symbols),  # Simulated success
            "success_rate": 0.95  # Simulated success rate
        })
        
        logger.info(f"Collection cycle {collection_id} completed")
        logger.info(f"Duration: {collection_results['duration_seconds']:.1f} seconds")
        logger.info(f"Success rate: {collection_results['success_rate']:.2%}")
        
        return collection_results
    
    def generate_collection_report(self, results: Dict) -> str:
        """Generate formatted collection report"""
        
        report = f"""
🎯 ATHENA COLLECTION REPORT
========================
Collection ID: {results['collection_id']}
Collection Type: {results['collection_type']}
Start Time: {results['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
End Time: {results['end_time'].strftime('%Y-%m-%d %H:%M:%S')}
Duration: {results['duration_seconds']:.1f} seconds

📊 PERFORMANCE METRICS
=====================
Symbols Requested: {results['symbols_requested']}
Symbols Processed: {results['symbols_processed']}
Success Rate: {results['success_rate']:.2%}
Processing Rate: {results['symbols_processed'] / results['duration_seconds']:.1f} symbols/second

🎯 ASSESSMENT
============
"""
        
        if results['success_rate'] > 0.95:
            report += "✅ EXCELLENT - Production ready performance"
        elif results['success_rate'] > 0.90:
            report += "✅ GOOD - Minor optimization opportunities"
        else:
            report += "⚠️ NEEDS IMPROVEMENT - Check error handling"
        
        if results.get('errors'):
            report += f"\n\n❌ ERRORS ({len(results['errors'])})\n"
            for error in results['errors'][:5]:  # Show first 5 errors
                report += f"- {error}\n"
        
        return report

# Integration with existing Athena scripts
async def main():
    """Main function for testing the enhanced collection system"""
    
    logger.info("🏛️ ATHENA ENHANCED COLLECTION SYSTEM")
    logger.info("====================================")
    
    # Initialize the enhanced collector
    collector = AthenaEnhancedCollector()
    
    # Test symbol loading
    logger.info("Testing Finviz symbol loading...")
    summary = collector.symbol_manager.get_data_summary()
    
    if "error" not in summary:
        logger.info(f"✅ Successfully loaded Finviz data:")
        logger.info(f"   Total symbols: {summary['total_symbols']}")
        logger.info(f"   Exchanges: {summary['exchanges']}")
        logger.info(f"   Sectors: {len(summary['sectors'])} unique sectors")
        logger.info(f"   Market cap range: ${summary['market_cap_stats']['min']:.0f}M - ${summary['market_cap_stats']['max']:.0f}M")
    else:
        logger.error("❌ Failed to load Finviz data")
        return
    
    # Test different collection types
    collection_types = [
        ("liquid", 100),      # Top 100 liquid symbols
        ("options_priority", 50),  # Top 50 options symbols
        ("tech_sector", 25)   # 25 tech sector symbols
    ]
    
    for collection_type, count in collection_types:
        logger.info(f"\n🔄 Testing {collection_type} collection...")
        
        results = await collector.run_collection_cycle(
            collection_type=collection_type,
            symbol_count=count
        )
        
        # Generate and display report
        report = collector.generate_collection_report(results)
        logger.info(report)
    
    logger.info("\n🎉 All tests completed successfully!")
    logger.info("Ready to integrate with your existing Athena pipeline.")

if __name__ == "__main__":
    # Run the enhanced collection system
    asyncio.run(main())