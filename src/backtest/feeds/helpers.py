"""
Helper functions for creating and configuring Parquet data feeds.

Provides convenience functions for:
- Creating feeds from symbol and date range
- Batch loading multiple feeds
- Validation of feed parameters
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, date, timedelta

from .parquet_feed import ParquetDataFeed
from src.backtest.utils.logging import get_logger


logger = get_logger(__name__)


def create_parquet_feed(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    cleaned_version: str = 'v1',
    base_path: str = 'data/cleaned',
    tz: str = 'UTC',
    validate_schema: bool = True,
) -> ParquetDataFeed:
    """
    Create a ParquetDataFeed with sensible defaults.
    
    This is the primary helper function for instantiating feeds.
    Validates parameters and constructs the feed object.
    
    Args:
        symbol: Asset symbol (e.g., 'MES', 'ES')
        start_date: Backtest start date
        end_date: Backtest end date
        cleaned_version: Data version to load (e.g., 'v1', 'v2')
        base_path: Base path to cleaned data directory
        tz: Timezone for datetime handling (default: 'UTC')
        validate_schema: Whether to validate OHLCV schema (default: True)
    
    Returns:
        ParquetDataFeed instance
    
    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If data file doesn't exist
    
    Example:
        >>> feed = create_parquet_feed(
        ...     symbol='MES',
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 12, 31),
        ... )
    """
    # Validate dates
    if not isinstance(start_date, (datetime, date)):
        raise ValueError('start_date must be datetime.datetime or datetime.date')
    if not isinstance(end_date, (datetime, date)):
        raise ValueError('end_date must be datetime.datetime or datetime.date')
    
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.max.time())
    
    if start_date >= end_date:
        raise ValueError('start_date must be before end_date')
    
    # Validate symbol
    if not symbol or not isinstance(symbol, str):
        raise ValueError('symbol must be a non-empty string')
    
    # Validate version
    if not cleaned_version or not isinstance(cleaned_version, str):
        raise ValueError('cleaned_version must be a non-empty string')
    
    # Validate base path
    base_path_obj = Path(base_path)
    if not base_path_obj.exists():
        logger.warning(f'Base path does not exist: {base_path}')
    
    logger.info(
        f'Creating ParquetDataFeed: {symbol} '
        f'({start_date.date()} to {end_date.date()}, '
        f'version={cleaned_version})'
    )
    
    # Create feed
    feed = ParquetDataFeed(
        symbol=symbol,
        cleaned_version=cleaned_version,
        start_date=start_date,
        end_date=end_date,
        base_path=base_path,
        tz=tz,
        validate_schema=validate_schema,
    )
    
    return feed


def create_multi_feeds(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    cleaned_version: str = 'v1',
    base_path: str = 'data/cleaned',
    tz: str = 'UTC',
) -> Dict[str, ParquetDataFeed]:
    """
    Create multiple ParquetDataFeeds for different symbols.
    
    Useful for strategies that trade multiple assets.
    
    Args:
        symbols: List of asset symbols
        start_date: Backtest start date
        end_date: Backtest end date
        cleaned_version: Data version
        base_path: Base path to cleaned data
        tz: Timezone
    
    Returns:
        Dictionary mapping symbol to ParquetDataFeed
    
    Example:
        >>> feeds = create_multi_feeds(
        ...     symbols=['MES', 'ES'],
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 12, 31),
        ... )
        >>> for symbol, feed in feeds.items():
        ...     cerebro.adddata(feed, name=symbol)
    """
    feeds = {}
    for symbol in symbols:
        try:
            feed = create_parquet_feed(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                cleaned_version=cleaned_version,
                base_path=base_path,
                tz=tz,
            )
            feeds[symbol] = feed
            logger.info(f'Created feed for {symbol}')
        except Exception as e:
            logger.error(f'Failed to create feed for {symbol}: {e}')
            raise
    
    return feeds


def validate_feed_exists(
    symbol: str,
    cleaned_version: str = 'v1',
    base_path: str = 'data/cleaned',
) -> bool:
    """
    Check if a Parquet data feed exists.
    
    Args:
        symbol: Asset symbol
        cleaned_version: Data version
        base_path: Base path to cleaned data
    
    Returns:
        True if feed file exists, False otherwise
    
    Example:
        >>> if validate_feed_exists('MES'):
        ...     feed = create_parquet_feed('MES', ...)
    """
    file_path = (
        f'{base_path}/'
        f'{cleaned_version}/'
        f'{symbol}/'
        f'ohlcv.parquet'
    )
    exists = Path(file_path).exists()
    
    if exists:
        logger.info(f'Feed file exists: {file_path}')
    else:
        logger.warning(f'Feed file not found: {file_path}')
    
    return exists


def list_available_feeds(
    cleaned_version: str = 'v1',
    base_path: str = 'data/cleaned',
) -> List[str]:
    """
    List all available asset symbols in the cleaned data directory.
    
    Args:
        cleaned_version: Data version
        base_path: Base path to cleaned data
    
    Returns:
        List of available symbols
    
    Example:
        >>> symbols = list_available_feeds()
        >>> print(f'Available symbols: {symbols}')
    """
    version_path = Path(base_path) / cleaned_version
    
    if not version_path.exists():
        logger.warning(f'Version path does not exist: {version_path}')
        return []
    
    # Get all directories that contain ohlcv.parquet
    available = []
    for asset_dir in version_path.iterdir():
        if asset_dir.is_dir():
            parquet_file = asset_dir / 'ohlcv.parquet'
            if parquet_file.exists():
                available.append(asset_dir.name)
    
    logger.info(f'Found {len(available)} available feeds: {available}')
    return sorted(available)


def get_feed_date_range(
    symbol: str,
    cleaned_version: str = 'v1',
    base_path: str = 'data/cleaned',
) -> Optional[tuple]:
    """
    Get the date range of available data for a symbol.
    
    Useful for determining valid backtest periods.
    
    Args:
        symbol: Asset symbol
        cleaned_version: Data version
        base_path: Base path to cleaned data
    
    Returns:
        Tuple of (min_date, max_date) or None if file doesn't exist
    
    Example:
        >>> start_date, end_date = get_feed_date_range('MES')
        >>> print(f'Data available from {start_date} to {end_date}')
    """
    file_path = (
        f'{base_path}/'
        f'{cleaned_version}/'
        f'{symbol}/'
        f'ohlcv.parquet'
    )
    
    if not Path(file_path).exists():
        logger.warning(f'Feed file not found: {file_path}')
        return None
    
    try:
        import pandas as pd
        from src.data_io.parquet_utils import read_parquet_with_schema
        
        # Read only timestamp column for efficiency
        df = read_parquet_with_schema(
            file_path,
            columns=['timestamp'],
        )
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        
        logger.info(
            f'Feed {symbol} date range: {min_date} to {max_date}'
        )
        
        return (min_date, max_date)
    
    except Exception as e:
        logger.error(f'Failed to get date range for {symbol}: {e}')
        return None
