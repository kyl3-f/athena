# scripts/load_finviz_symbols.py
import pandas as pd
import logging
from pathlib import Path
import sys
from typing import Set, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class FinvizSymbolLoader:
    """
    Load ticker universe from Finviz CSV export
    
    Instructions:
    1. Go to finviz.com/screener.ashx
    2. Set your filters (market cap, volume, etc.)
    3. Click "Export" to download CSV
    4. Place CSV in data/finviz/ directory
    5. Run this script
    """
    
    def __init__(self):
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        self.finviz_dir = Path("data/finviz")
        self.finviz_dir.mkdir(parents=True, exist_ok=True)
        
    def find_finviz_csv(self) -> Path:
        """Find the most recent Finviz CSV file"""
        # Look for CSV files in finviz directory
        csv_files = list(self.finviz_dir.glob("*.csv"))
        
        if not csv_files:
            # Also check root directory in case user put it there
            root_csv_files = list(Path(".").glob("finviz*.csv"))
            csv_files.extend(root_csv_files)
        
        if not csv_files:
            raise FileNotFoundError(
                f"No Finviz CSV file found. Please:\n"
                f"1. Go to finviz.com/screener.ashx\n"
                f"2. Set your filters\n"
                f"3. Click 'Export' to download CSV\n"
                f"4. Place the CSV file in {self.finviz_dir}/ or the project root\n"
                f"5. Run this script again"
            )
        
        # Get the most recent file
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Found Finviz CSV: {latest_file}")
        
        return latest_file
    
    def load_finviz_symbols(self, csv_file: Path) -> Set[str]:
        """Load symbols from Finviz CSV export"""
        try:
            logger.info(f"Loading symbols from {csv_file}")
            
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Log CSV structure for debugging
            logger.info(f"CSV columns: {list(df.columns)}")
            logger.info(f"CSV shape: {df.shape}")
            
            # Find ticker column (Finviz usually uses 'Ticker' or 'Symbol')
            ticker_columns = ['Ticker', 'Symbol', 'ticker', 'symbol']
            ticker_column = None
            
            for col in ticker_columns:
                if col in df.columns:
                    ticker_column = col
                    break
            
            if ticker_column is None:
                logger.error(f"Could not find ticker column. Available columns: {list(df.columns)}")
                logger.info("Please ensure your CSV has a 'Ticker' or 'Symbol' column")
                return set()
            
            # Extract symbols
            symbols = set(df[ticker_column].astype(str).str.upper().str.strip())
            
            # Remove any NaN or invalid symbols
            symbols = {s for s in symbols if s and s != 'NAN' and len(s) <= 5 and s.isalpha()}
            
            logger.info(f"Loaded {len(symbols)} valid symbols from Finviz CSV")
            
            return symbols
            
        except Exception as e:
            logger.error(f"Error loading Finviz CSV: {e}")
            return set()
    
    def filter_symbols(self, symbols: Set[str], min_market_cap: str = None, 
                      volume_filter: int = None) -> Set[str]:
        """Additional filtering if needed"""
        # Basic filtering already done in load_finviz_symbols
        # You can add more sophisticated filtering here if needed
        
        filtered_symbols = symbols.copy()
        
        # Remove penny stocks and ETFs if desired
        exclude_patterns = ['ETFM', 'FUND', 'SPDR', 'INVS']
        for pattern in exclude_patterns:
            filtered_symbols = {s for s in filtered_symbols if pattern not in s}
        
        logger.info(f"After filtering: {len(filtered_symbols)} symbols")
        return filtered_symbols
    
    def save_symbol_list(self, symbols: Set[str]) -> dict:
        """Save symbols to production files"""
        if not symbols:
            logger.error("No symbols to save")
            return {'success': False}
        
        # Save master list
        master_file = self.config_dir / "symbols.txt"
        with open(master_file, 'w') as f:
            for symbol in sorted(symbols):
                f.write(f"{symbol}\n")
        
        # Save metadata
        metadata = {
            'source': 'finviz_export',
            'total_symbols': len(symbols),
            'symbols_preview': sorted(list(symbols))[:50],  # First 50 for preview
            'load_timestamp': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = self.config_dir / "symbols_metadata.json"
        with open(metadata_file, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
        
        # Also save a backup
        backup_file = self.finviz_dir / f"symbols_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(backup_file, 'w') as f:
            for symbol in sorted(symbols):
                f.write(f"{symbol}\n")
        
        result = {
            'success': True,
            'symbol_count': len(symbols),
            'files_created': {
                'master_list': str(master_file),
                'metadata': str(metadata_file),
                'backup': str(backup_file)
            }
        }
        
        logger.info(f"✅ Saved {len(symbols)} symbols to {master_file}")
        return result
    
    def validate_symbols(self, symbols: Set[str]) -> dict:
        """Validate symbol format and quality"""
        valid_symbols = set()
        invalid_symbols = set()
        warnings = []
        
        for symbol in symbols:
            # Basic validation
            if (len(symbol) >= 1 and len(symbol) <= 5 and 
                symbol.replace('-', '').replace('.', '').isalnum() and
                symbol.isascii()):
                valid_symbols.add(symbol)
            else:
                invalid_symbols.add(symbol)
        
        # Check for common issues
        if len(valid_symbols) < 100:
            warnings.append("Symbol count seems low (< 100). Check your Finviz filters.")
        
        single_char_symbols = {s for s in valid_symbols if len(s) == 1}
        if single_char_symbols:
            warnings.append(f"Found single-character symbols: {single_char_symbols}")
        
        validation_result = {
            'total_input': len(symbols),
            'valid_symbols': len(valid_symbols),
            'invalid_symbols': len(invalid_symbols),
            'validation_rate_pct': round(len(valid_symbols) / len(symbols) * 100, 2),
            'warnings': warnings,
            'invalid_examples': list(invalid_symbols)[:10] if invalid_symbols else []
        }
        
        logger.info(f"Validation: {validation_result['valid_symbols']}/{validation_result['total_input']} valid ({validation_result['validation_rate_pct']}%)")
        
        if warnings:
            for warning in warnings:
                logger.warning(warning)
        
        return validation_result, valid_symbols
    
    def process_finviz_export(self) -> dict:
        """Complete pipeline: find CSV → load → validate → save"""
        try:
            # Find CSV file
            csv_file = self.find_finviz_csv()
            
            # Load symbols
            raw_symbols = self.load_finviz_symbols(csv_file)
            if not raw_symbols:
                return {'success': False, 'error': 'No symbols loaded from CSV'}
            
            # Validate symbols
            validation, valid_symbols = self.validate_symbols(raw_symbols)
            
            # Apply additional filters if needed
            filtered_symbols = self.filter_symbols(valid_symbols)
            
            # Save to production files
            save_result = self.save_symbol_list(filtered_symbols)
            
            # Combine results
            final_result = {
                'success': True,
                'csv_file': str(csv_file),
                'validation': validation,
                'final_symbol_count': len(filtered_symbols),
                'files_created': save_result.get('files_created', {}),
                'ready_for_production': len(filtered_symbols) > 0
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing Finviz export: {e}")
            return {'success': False, 'error': str(e)}


def main():
    """Load Finviz ticker universe for production use"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    logger.info("🚀 Loading Finviz Ticker Universe")
    logger.info("=" * 60)
    
    try:
        loader = FinvizSymbolLoader()
        result = loader.process_finviz_export()
        
        if result['success']:
            print("\n" + "=" * 60)
            print("🎉 FINVIZ SYMBOL LOADING COMPLETE")
            print("=" * 60)
            print(f"📁 CSV file: {result['csv_file']}")
            print(f"📊 Symbols loaded: {result['final_symbol_count']:,}")
            print(f"✅ Validation rate: {result['validation']['validation_rate_pct']}%")
            print("=" * 60)
            print("📁 Files created:")
            for desc, path in result['files_created'].items():
                print(f"  • {desc}: {path}")
            print("=" * 60)
            
            if result['validation']['warnings']:
                print("⚠️  Warnings:")
                for warning in result['validation']['warnings']:
                    print(f"  • {warning}")
            
            print("✅ Ready for production trading!")
            
        else:
            print(f"❌ Failed to load symbols: {result.get('error')}")
            print("\nInstructions:")
            print("1. Go to finviz.com/screener.ashx")
            print("2. Set your filters (Market Cap > $1B, Volume > 1M, etc.)")
            print("3. Click 'Export' to download CSV")
            print("4. Place CSV in data/finviz/ directory")
            print("5. Run this script again")
        
    except Exception as e:
        logger.error(f"Finviz symbol loading failed: {e}")
        raise


if __name__ == "__main__":
    main()