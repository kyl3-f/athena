#!/usr/bin/env python3
"""
Athena Data Inspector
Inspect downloaded data and show statistics
"""

import sys
from pathlib import Path
import polars as pl

def inspect_data():
    """Inspect downloaded data files"""
    print("🔍 Athena Data Inspector")
    print("=" * 40)
    
    # Use current working directory as project root
    project_root = Path.cwd()
    data_dir = project_root / "data" / "raw" / "stocks" / "aggregates"
    
    print(f"Looking in: {data_dir}")
    
    if not data_dir.exists():
        print("❌ No data directory found")
        return
    
    timeframes = ['minute', 'hour', 'day']
    
    for timeframe in timeframes:
        timeframe_dir = data_dir / timeframe
        
        if not timeframe_dir.exists():
            print(f"⚠️  No {timeframe} data found")
            continue
        
        print(f"\n📊 {timeframe.upper()} DATA:")
        print("-" * 20)
        
        # Find parquet files
        parquet_files = list(timeframe_dir.glob("*.parquet"))
        
        for file in parquet_files:
            try:
                # Load data
                df = pl.read_parquet(file)
                
                # File info
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"\n📄 {file.name}")
                print(f"   Size: {size_mb:.2f} MB")
                print(f"   Rows: {df.shape[0]:,}")
                print(f"   Columns: {df.shape[1]}")
                
                # Data summary
                if not df.is_empty():
                    # Time range
                    min_time = df['timestamp_utc'].min()
                    max_time = df['timestamp_utc'].max()
                    print(f"   Time range: {min_time} to {max_time}")
                    
                    # Price summary
                    if 'close' in df.columns:
                        min_price = df['close'].min()
                        max_price = df['close'].max()
                        avg_price = df['close'].mean()
                        print(f"   Price range: ${min_price:.2f} - ${max_price:.2f}")
                        print(f"   Average price: ${avg_price:.2f}")
                    
                    # Volume summary
                    if 'volume' in df.columns:
                        total_volume = df['volume'].sum()
                        avg_volume = df['volume'].mean()
                        print(f"   Total volume: {total_volume:,}")
                        print(f"   Average volume: {avg_volume:,.0f}")
                    
                    # Show sample rows
                    print(f"\n   📋 Sample data (first 3 rows):")
                    sample_df = df.head(3).select(['timestamp_utc', 'open', 'high', 'low', 'close', 'volume'])
                    print(sample_df)
                    
            except Exception as e:
                print(f"   ❌ Error reading {file.name}: {e}")

def analyze_trading_patterns():
    """Analyze basic trading patterns in the data"""
    print(f"\n📈 TRADING PATTERN ANALYSIS")
    print("=" * 40)
    
    # Use current working directory as project root
    project_root = Path.cwd()
    minute_file = project_root / "data" / "raw" / "stocks" / "aggregates" / "minute" / "AAPL_minute_2025-07-21_2025-07-26.parquet"
    
    if not minute_file.exists():
        print("❌ No minute data found for analysis")
        return
    
    try:
        df = pl.read_parquet(minute_file)
        
        # Calculate basic metrics
        df = df.with_columns([
            # Price change
            (pl.col('close') - pl.col('open')).alias('price_change'),
            # Price change percentage
            ((pl.col('close') - pl.col('open')) / pl.col('open') * 100).alias('price_change_pct'),
            # Extract hour from timestamp
            pl.col('timestamp_utc').dt.hour().alias('hour')
        ])
        
        print(f"📊 Minute-by-minute analysis ({df.shape[0]} bars):")
        
        # Biggest price movements
        print(f"\n🔥 Biggest price movements:")
        big_moves = df.sort('price_change_pct', descending=True).head(5)
        for row in big_moves.iter_rows(named=True):
            time = row['timestamp_utc']
            change = row['price_change_pct']
            volume = row['volume']
            print(f"   {time}: {change:+.2f}% (Volume: {volume:,})")
        
        # Volume analysis
        print(f"\n📊 Volume analysis:")
        avg_volume = df['volume'].mean()
        max_volume = df['volume'].max()
        high_volume_bars = df.filter(pl.col('volume') > avg_volume * 2).shape[0]
        print(f"   Average volume: {avg_volume:,.0f}")
        print(f"   Max volume: {max_volume:,}")
        print(f"   High volume bars (>2x avg): {high_volume_bars}")
        
        # Trading hours analysis
        print(f"\n🕒 Trading hours analysis:")
        hourly_stats = df.group_by('hour').agg([
            pl.col('volume').mean().alias('avg_volume'),
            pl.col('price_change_pct').std().alias('volatility')
        ]).sort('hour')
        
        print("   Hour | Avg Volume | Volatility")
        print("   -----|------------|----------")
        for row in hourly_stats.iter_rows(named=True):
            hour = row['hour']
            vol = row['avg_volume']
            vola = row['volatility'] if row['volatility'] is not None else 0
            print(f"   {hour:4d} | {vol:10,.0f} | {vola:8.2f}%")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")

def check_data_quality():
    """Check data quality and completeness"""
    print(f"\n✅ DATA QUALITY CHECK")
    print("=" * 40)
    
    project_root = Path.cwd()
    timeframes = ['minute', 'hour', 'day']
    
    for timeframe in timeframes:
        file_path = project_root / "data" / "raw" / "stocks" / "aggregates" / timeframe / f"AAPL_{timeframe}_2025-07-21_2025-07-26.parquet"
        
        if not file_path.exists():
            print(f"❌ {timeframe} data missing")
            continue
        
        try:
            df = pl.read_parquet(file_path)
            
            print(f"\n📋 {timeframe.upper()} data quality:")
            
            # Check for missing values
            null_counts = df.null_count()
            total_nulls = sum(null_counts.row(0))
            print(f"   Missing values: {total_nulls}")
            
            # Check for duplicate timestamps
            unique_times = df['timestamp_utc'].n_unique()
            total_rows = df.shape[0]
            print(f"   Duplicate timestamps: {total_rows - unique_times}")
            
            # Check for invalid prices (negative, zero, or extreme)
            invalid_prices = df.filter(
                (pl.col('close') <= 0) | 
                (pl.col('open') <= 0) |
                (pl.col('high') <= 0) |
                (pl.col('low') <= 0)
            ).shape[0]
            print(f"   Invalid prices: {invalid_prices}")
            
            # Check for invalid volumes
            invalid_volumes = df.filter(pl.col('volume') < 0).shape[0]
            print(f"   Invalid volumes: {invalid_volumes}")
            
            # Data completeness score
            completeness = 100 - (total_nulls + invalid_prices + invalid_volumes) / (total_rows * df.shape[1]) * 100
            print(f"   Data quality score: {completeness:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Quality check failed: {e}")

def main():
    """Main inspection function"""
    inspect_data()
    analyze_trading_patterns()
    check_data_quality()
    
    print(f"\n🎉 Data inspection complete!")
    print(f"\nYour data is ready for:")
    print(f"   • Feature engineering")
    print(f"   • Model training")
    print(f"   • Backtesting strategies")
    print(f"   • Real-time signal generation")

if __name__ == "__main__":
    main()