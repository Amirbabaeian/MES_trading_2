"""
Data Feeds Module

Contains custom data feed implementations for backtrader.
Supports various data sources including Parquet files, databases, and APIs.

Includes:
- ParquetDataFeed: Loads OHLCV data from Parquet files in versioned storage
- Helper functions for feed instantiation and validation
"""

from .parquet_feed import ParquetDataFeed
from .helpers import (
    create_parquet_feed,
    create_multi_feeds,
    validate_feed_exists,
    list_available_feeds,
    get_feed_date_range,
)

__all__ = [
    'ParquetDataFeed',
    'create_parquet_feed',
    'create_multi_feeds',
    'validate_feed_exists',
    'list_available_feeds',
    'get_feed_date_range',
]
