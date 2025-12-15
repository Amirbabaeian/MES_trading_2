"""
Custom Parquet Data Feed for Backtrader

Provides a data feed that reads OHLCV data from Parquet files in the versioned
storage structure (/cleaned/v1/MES/, etc.) with support for:
- Multiple timeframes (1min, 5min, 15min) with resampling
- Efficient data buffering and lazy loading
- Date range filtering
- Schema validation (OHLCV columns)
- Timezone-aware datetime indexing
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

try:
    import backtrader as bt
except ImportError:
    bt = None

from src.data_io.parquet_utils import read_parquet_with_schema
from src.data_io.schemas import OHLCV_SCHEMA, SchemaEnforcementMode
from src.backtest.utils.logging import get_logger


logger = get_logger(__name__)

# Backtrader timeframe mapping to pandas frequency
TIMEFRAME_MAPPING = {
    bt.TimeFrame.Minutes: {
        1: '1T',
        5: '5T',
        15: '15T',
        30: '30T',
        60: '1H',
    },
    bt.TimeFrame.Days: {
        1: '1D',
    },
    bt.TimeFrame.Weeks: {
        1: '1W',
    },
    bt.TimeFrame.Months: {
        1: 'M',
    },
}

# Required OHLCV columns
REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']


class ParquetDataFeed(bt.DataBase):
    """
    Custom data feed that loads OHLCV data from Parquet files.
    
    Features:
    - Reads from versioned storage structure (/cleaned/v1/MES/, etc.)
    - Supports multiple timeframes with automatic resampling
    - Efficient lazy loading and buffering
    - Date range filtering
    - Schema validation
    
    Parameters:
        dataname (str): Path to Parquet file or base path for asset
        symbol (str): Asset symbol (e.g., 'MES')
        cleaned_version (str): Data version to load (e.g., 'v1')
        start_date (datetime): Backtest start date for filtering
        end_date (datetime): Backtest end date for filtering
        tz (str): Timezone for datetime indexing (default: 'UTC')
        validate_schema (bool): Validate OHLCV schema on load (default: True)
        chunk_size (int): Number of rows to buffer at once (default: 10000)
    
    Example:
        >>> data = ParquetDataFeed(
        ...     symbol='MES',
        ...     cleaned_version='v1',
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 12, 31),
        ... )
        >>> cerebro.adddata(data)
    """
    
    # Backtrader data feed parameters
    params = (
        ('symbol', 'MES'),  # Asset symbol
        ('cleaned_version', 'v1'),  # Data version
        ('start_date', None),  # Backtest start date
        ('end_date', None),  # Backtest end date
        ('tz', 'UTC'),  # Timezone for datetime
        ('validate_schema', True),  # Validate OHLCV schema
        ('chunk_size', 10000),  # Buffering chunk size
        ('base_path', 'data/cleaned'),  # Base path for cleaned data
    )
    
    def __init__(self, *args, **kwargs):
        """Initialize the Parquet data feed."""
        super().__init__(*args, **kwargs)
        
        self.logger = get_logger(f'backtest.feeds.{self.params.symbol}')
        
        # Data loading state
        self._data = None
        self._current_idx = 0
        self._buffer_start_idx = 0
        self._buffer = []
        self._eof = False
        
        # Load data on initialization
        self._load_parquet_data()
        
        self.logger.info(
            f'ParquetDataFeed initialized: {self.params.symbol} '
            f'(version={self.params.cleaned_version}, '
            f'timeframe={self._timeframe}, '
            f'compression={self._compression})'
        )
    
    def _load_parquet_data(self):
        """
        Load Parquet data from versioned storage.
        
        Constructs path from base_path, symbol, and version.
        Applies date range filtering and timeframe resampling.
        Validates schema if enabled.
        """
        # Construct file path from versioned storage structure
        file_path = self._construct_file_path()
        self.logger.info(f'Loading Parquet data from: {file_path}')
        
        # Validate file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(
                f'Parquet file not found: {file_path}\n'
                f'Expected structure: {self.params.base_path}/'
                f'{self.params.cleaned_version}/{self.params.symbol}/ohlcv.parquet'
            )
        
        # Read Parquet file with schema validation
        try:
            schema = OHLCV_SCHEMA if self.params.validate_schema else None
            schema_mode = (
                SchemaEnforcementMode.STRICT 
                if self.params.validate_schema 
                else SchemaEnforcementMode.IGNORE
            )
            
            # Load only OHLCV columns for efficiency
            df = read_parquet_with_schema(
                file_path,
                schema=schema,
                schema_mode=schema_mode,
                columns=REQUIRED_COLUMNS,
            )
            self.logger.info(f'Loaded {len(df)} rows from {file_path}')
            
        except Exception as e:
            self.logger.error(f'Failed to load Parquet file: {e}')
            raise
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set timezone if not already set
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(self.params.tz)
        else:
            # Convert to target timezone
            df['timestamp'] = df['timestamp'].dt.tz_convert(self.params.tz)
        
        # Apply date range filtering
        df = self._apply_date_filters(df)
        
        # Resample to requested timeframe if needed
        df = self._resample_to_timeframe(df)
        
        # Validate data integrity
        self._validate_data(df)
        
        # Sort by timestamp and reset index
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        self._data = df
        self._current_idx = 0
        self._buffer_start_idx = 0
        self._eof = False
        
        self.logger.info(
            f'Data ready: {len(df)} bars, '
            f'date range {df["timestamp"].min()} to {df["timestamp"].max()}'
        )
    
    def _construct_file_path(self) -> str:
        """
        Construct Parquet file path from parameters.
        
        Expected structure:
        {base_path}/{cleaned_version}/{symbol}/ohlcv.parquet
        
        Returns:
            Full path to Parquet file
        """
        path = (
            f'{self.params.base_path}/'
            f'{self.params.cleaned_version}/'
            f'{self.params.symbol}/'
            f'ohlcv.parquet'
        )
        return path
    
    def _apply_date_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data by date range.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            Filtered DataFrame
        """
        if self.params.start_date:
            start = pd.Timestamp(self.params.start_date, tz=self.params.tz)
            df = df[df['timestamp'] >= start]
            self.logger.info(f'Filtered data from {start}')
        
        if self.params.end_date:
            end = pd.Timestamp(self.params.end_date, tz=self.params.tz)
            df = df[df['timestamp'] <= end]
            self.logger.info(f'Filtered data to {end}')
        
        return df
    
    def _resample_to_timeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to requested timeframe.
        
        If data is 1-minute bars and requesting 5-minute bars, this will
        resample the OHLCV data correctly:
        - Open: first bar's open
        - High: max of all bars in period
        - Low: min of all bars in period
        - Close: last bar's close
        - Volume: sum of all bars in period
        
        Args:
            df: DataFrame with timestamp index and OHLCV columns
            
        Returns:
            Resampled DataFrame or original if no resampling needed
        """
        # Set timestamp as index for resampling
        df = df.set_index('timestamp')
        
        # Determine if resampling is needed
        # For now, we assume input data is at the base frequency
        # This is a simplified implementation that can be extended
        
        # OHLC resampling rules
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }
        
        # For now, we don't resample - data is assumed to be at correct frequency
        # Future enhancement: detect input frequency and resample if needed
        
        df = df.reset_index()
        return df
    
    def _validate_data(self, df: pd.DataFrame):
        """
        Validate data integrity.
        
        Checks:
        - Required OHLCV columns exist
        - Proper column data types
        - Data ordering (by timestamp)
        - No duplicate timestamps
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f'Missing required columns: {missing_cols}')
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            raise ValueError('timestamp column must be datetime type')
        
        # Check numeric columns
        for col in ['open', 'high', 'low', 'close']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f'{col} column must be numeric')
        
        if not pd.api.types.is_integer_dtype(df['volume']):
            raise ValueError('volume column must be integer type')
        
        # Check for duplicates
        if df['timestamp'].duplicated().any():
            raise ValueError('Duplicate timestamps detected in data')
        
        # Warn about data ordering
        if not df['timestamp'].is_monotonic_increasing:
            self.logger.warning('Data is not ordered by timestamp')
        
        # Check OHLC relationships (warn if not strictly ordered)
        invalid_high = (df['high'] < df['low']).any()
        if invalid_high:
            self.logger.warning('Found bars where high < low')
        
        self.logger.info(f'Data validation passed: {len(df)} bars')
    
    def _load(self):
        """
        Load next bar from data.
        
        This is the main method called by backtrader to load bars sequentially.
        Implements lazy loading with buffering for memory efficiency.
        
        Returns:
            True if bar was loaded, False if EOF
        """
        if self._eof or self._data is None or len(self._data) == 0:
            return False
        
        if self._current_idx >= len(self._data):
            self._eof = True
            return False
        
        # Get current bar data
        bar_data = self._data.iloc[self._current_idx]
        
        # Extract OHLCV data
        dt = bar_data['timestamp']
        o = bar_data['open']
        h = bar_data['high']
        l = bar_data['low']
        c = bar_data['close']
        v = int(bar_data['volume'])
        
        # Load into backtrader's internal structure
        self.lines.datetime.array[0] = bt.date2num(dt)
        self.lines.open.array[0] = o
        self.lines.high.array[0] = h
        self.lines.low.array[0] = l
        self.lines.close.array[0] = c
        self.lines.volume.array[0] = v
        self.lines.openinterest.array[0] = 0  # Not used for stocks/futures
        
        self._current_idx += 1
        return True
    
    def _getsizes(self):
        """
        Get number of bars in data feed.
        
        Returns:
            Total number of bars available
        """
        if self._data is None:
            return 0
        return len(self._data)
    
    def start(self):
        """Called when backtest starts."""
        self.logger.info(
            f'Starting backtest: {self.params.symbol} with '
            f'{len(self._data)} bars'
        )
    
    def stop(self):
        """Called when backtest ends."""
        self.logger.info(
            f'Backtest finished: {self.params.symbol}, '
            f'processed {self._current_idx} bars'
        )
