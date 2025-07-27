# src/processing/data_cleaner.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Clean and normalize raw market data from Bronze layer
    Handles missing values, outliers, and data quality issues
    """
    
    def __init__(self):
        self.outlier_threshold_z = 3.0  # Z-score threshold for outlier detection
        self.min_volume_threshold = 100  # Minimum volume to consider valid
        self.max_price_change_pct = 0.20  # Max 20% price change per minute (circuit breaker proxy)
        
    def clean_stock_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """
        Clean raw stock price data from Polygon.io format
        Expected format: [{'t': timestamp_ms, 'o': open, 'h': high, 'l': low, 'c': close, 'v': volume, 'n': trades}, ...]
        """
        if not raw_data:
            logger.warning("No raw data provided")
            return pd.DataFrame()
        
        logger.info(f"Starting data cleaning with {len(raw_data)} raw records")
        
        # Convert to DataFrame
        df = pd.DataFrame(raw_data)
        
        # Validate required columns exist
        required_columns = ['t', 'o', 'h', 'l', 'c', 'v']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Data cleaning pipeline
        df = self._handle_missing_values(df)
        df = self._remove_invalid_prices(df)
        df = self._validate_ohlc_relationships(df)
        df = self._remove_outliers(df)
        df = self._filter_market_hours(df)
        df = self._add_basic_derived_columns(df)
        
        logger.info(f"Data cleaning complete: {len(df)} clean records remaining")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in stock data"""
        original_count = len(df)
        
        # Remove rows with critical missing values
        critical_columns = ['t', 'o', 'h', 'l', 'c', 'v']
        df = df.dropna(subset=critical_columns)
        
        # Replace zero prices with NaN (likely data errors)
        price_columns = ['o', 'h', 'l', 'c']
        for col in price_columns:
            df.loc[df[col] <= 0, col] = np.nan
        
        # Forward fill small gaps (max 3 consecutive missing values)
        df[price_columns] = df[price_columns].ffill(limit=3)
        
        # Remove any remaining rows with NaN prices
        df = df.dropna(subset=price_columns)
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with missing/invalid critical data")
        
        return df
    
    def _remove_invalid_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with invalid price data"""
        original_count = len(df)
        
        # Remove rows where any price is negative or zero
        price_columns = ['o', 'h', 'l', 'c']
        valid_prices = (df[price_columns] > 0).all(axis=1)
        df = df[valid_prices]
        
        # Remove rows with extreme price changes (likely data errors)
        df['price_change_pct'] = df['c'].pct_change().abs()
        extreme_changes = df['price_change_pct'] > self.max_price_change_pct
        df = df[~extreme_changes]
        
        # Remove rows with zero volume (invalid for minute data)
        df = df[df['v'] > 0]
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with invalid prices")
        
        return df
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC relationships (High >= Open,Close,Low, etc.)"""
        original_count = len(df)
        
        # Create validation conditions
        valid_high = (df['h'] >= df['o']) & (df['h'] >= df['c']) & (df['h'] >= df['l'])
        valid_low = (df['l'] <= df['o']) & (df['l'] <= df['c']) & (df['l'] <= df['h'])
        valid_volume = df['v'] >= self.min_volume_threshold
        
        # Apply all validation conditions
        valid_mask = valid_high & valid_low & valid_volume
        df = df[valid_mask]
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with invalid OHLC relationships")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers using Z-score and IQR methods"""
        if len(df) < 100:  # Need sufficient data for outlier detection
            return df
        
        original_count = len(df)
        
        # Z-score based outlier removal for prices
        price_columns = ['o', 'h', 'l', 'c']
        for col in price_columns:
            if col in df.columns and df[col].std() > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < self.outlier_threshold_z]
        
        # IQR-based outlier removal for volume (more robust for skewed data)
        if 'v' in df.columns and len(df) > 50:
            Q1 = df['v'].quantile(0.25)
            Q3 = df['v'].quantile(0.75)
            IQR = Q3 - Q1
            volume_outlier_threshold = Q3 + 3 * IQR  # More lenient for volume
            df = df[df['v'] <= volume_outlier_threshold]
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} statistical outliers")
        
        return df
    
    def _filter_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to market hours only (9:30 AM - 4:00 PM ET)"""
        original_count = len(df)
        
        # Convert to Eastern Time for market hours filtering
        df['timestamp_et'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        
        # Define market hours
        market_start = 9.5  # 9:30 AM
        market_end = 16.0   # 4:00 PM
        
        # Extract hour with minutes as decimal
        df['hour_decimal'] = df['timestamp_et'].dt.hour + df['timestamp_et'].dt.minute / 60
        
        # Filter to market hours and weekdays only
        market_hours_mask = (
            (df['hour_decimal'] >= market_start) & 
            (df['hour_decimal'] <= market_end) &
            (df['timestamp_et'].dt.dayofweek < 5)  # Monday=0, Friday=4
        )
        
        df = df[market_hours_mask]
        
        # Drop temporary columns
        df = df.drop(['timestamp_et', 'hour_decimal'], axis=1)
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} records outside market hours")
        
        return df
    
    def _add_basic_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived columns for feature engineering"""
        # Price-based features
        df['price_range'] = df['h'] - df['l']
        df['price_range_pct'] = (df['price_range'] / df['c']) * 100
        
        # Candlestick body and wick analysis
        df['body_size'] = abs(df['c'] - df['o'])
        df['body_size_pct'] = (df['body_size'] / df['c']) * 100
        df['upper_wick'] = df['h'] - df[['o', 'c']].max(axis=1)
        df['lower_wick'] = df[['o', 'c']].min(axis=1) - df['l']
        
        # Price movement direction
        df['is_green'] = (df['c'] > df['o']).astype(int)
        df['is_red'] = (df['c'] < df['o']).astype(int)
        df['is_doji'] = (abs(df['c'] - df['o']) / df['c'] < 0.001).astype(int)
        
        # Volume-weighted features
        df['typical_price'] = (df['h'] + df['l'] + df['c']) / 3
        df['volume_weighted_component'] = df['typical_price'] * df['v']  # Component for VWAP calculation
        
        # Calculate simple VWAP over a rolling window (20 periods)
        df['vwap_20'] = (df['volume_weighted_component'].rolling(20).sum() / 
                        df['v'].rolling(20).sum())
        
        # For the first few periods where we don't have 20 periods, use expanding window
        df['vwap_20'] = df['vwap_20'].fillna(
            df['volume_weighted_component'].expanding().sum() / df['v'].expanding().sum()
        )
        
        # Returns
        df['return_1min'] = df['c'].pct_change()
        df['log_return_1min'] = np.log(df['c'] / df['c'].shift(1))
        
        # Volume features
        df['volume_log'] = np.log1p(df['v'])  # log(1 + volume) to handle zeros
        
        # Clean up any infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill(limit=3)
        df = df.dropna()
        
        return df
    
    def get_cleaning_summary(self, original_count: int, final_count: int) -> Dict:
        """Generate a summary of the cleaning process"""
        cleaned_pct = (final_count / original_count * 100) if original_count > 0 else 0
        
        return {
            'original_records': original_count,
            'final_records': final_count,
            'records_removed': original_count - final_count,
            'data_retention_pct': round(cleaned_pct, 2),
            'cleaning_timestamp': datetime.now().isoformat()
        }
    
    def validate_cleaned_data(self, df: pd.DataFrame) -> bool:
        """Validate that cleaned data meets quality standards"""
        if df.empty:
            logger.error("Cleaned data is empty")
            return False
        
        # Check for required columns
        required_cols = ['t', 'o', 'h', 'l', 'c', 'v', 'timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns after cleaning: {missing_cols}")
            return False
        
        # Check for invalid values
        price_cols = ['o', 'h', 'l', 'c']
        if (df[price_cols] <= 0).any().any():
            logger.error("Found non-positive prices after cleaning")
            return False
        
        if (df['v'] <= 0).any():
            logger.error("Found non-positive volume after cleaning")
            return False
        
        # Check OHLC relationships
        invalid_ohlc = (
            (df['h'] < df['o']) | 
            (df['h'] < df['c']) | 
            (df['h'] < df['l']) |
            (df['l'] > df['o']) | 
            (df['l'] > df['c'])
        ).any()
        
        if invalid_ohlc:
            logger.error("Found invalid OHLC relationships after cleaning")
            return False
        
        logger.info("Data validation passed")
        return True


# Example usage and testing
if __name__ == "__main__":
    # Test the data cleaner with sample data
    import json
    import pytz
    from pathlib import Path
    from datetime import timedelta
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # Create sample test data (with market hours timestamps)
    from datetime import datetime
    import pytz
    
    # Create timestamps during market hours (10 AM, 11 AM, 12 PM ET on a weekday)
    et = pytz.timezone('US/Eastern')
    base_date = datetime(2024, 1, 15, 10, 0, 0)  # Monday, January 15, 2024, 10:00 AM
    
    sample_data = [
        {'t': int(base_date.replace(tzinfo=et).timestamp() * 1000), 'o': 100.0, 'h': 101.0, 'l': 99.5, 'c': 100.5, 'v': 1000},
        {'t': int((base_date + timedelta(minutes=1)).replace(tzinfo=et).timestamp() * 1000), 'o': 100.5, 'h': 102.0, 'l': 100.0, 'c': 101.5, 'v': 1500},
        {'t': int((base_date + timedelta(minutes=2)).replace(tzinfo=et).timestamp() * 1000), 'o': 101.5, 'h': 101.8, 'l': 101.0, 'c': 101.2, 'v': 800},
        # Add some problematic data to test cleaning
        {'t': int((base_date + timedelta(minutes=3)).replace(tzinfo=et).timestamp() * 1000), 'o': 0, 'h': 102.0, 'l': 100.0, 'c': 101.0, 'v': 1200},  # Zero open price
        {'t': int((base_date + timedelta(minutes=4)).replace(tzinfo=et).timestamp() * 1000), 'o': 101.0, 'h': 100.0, 'l': 102.0, 'c': 101.5, 'v': 900},  # Invalid OHLC
        {'t': int((base_date + timedelta(minutes=5)).replace(tzinfo=et).timestamp() * 1000), 'o': 101.5, 'h': 102.5, 'l': 101.0, 'c': 102.0, 'v': 0},  # Zero volume
    ]
    
    # Test the cleaner
    cleaner = DataCleaner()
    original_count = len(sample_data)
    
    print(f"Testing DataCleaner with {original_count} sample records")
    
    # Clean the data
    cleaned_df = cleaner.clean_stock_data(sample_data)
    
    # Generate summary
    summary = cleaner.get_cleaning_summary(original_count, len(cleaned_df))
    print(f"Cleaning Summary: {summary}")
    
    # Validate results
    is_valid = cleaner.validate_cleaned_data(cleaned_df)
    print(f"Data validation passed: {is_valid}")
    
    if not cleaned_df.empty:
        print(f"\nCleaned data shape: {cleaned_df.shape}")
        print(f"Columns: {list(cleaned_df.columns)}")
        print(f"\nFirst few rows:")
        print(cleaned_df.head())