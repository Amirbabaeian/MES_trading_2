"""
Data ingestion module for market data providers.

This module provides:
- Abstract base class for data providers (DataProvider)
- Concrete adapter implementations for specific vendors
- Exception hierarchy for error handling
- Rate limiting and retry utilities
"""

from src.data_ingestion.base_provider import DataProvider
from src.data_ingestion.exceptions import (
    DataProviderError,
    AuthenticationError,
    DataNotAvailableError,
    RateLimitError,
    PaginationError,
    ConfigurationError,
    ValidationError,
    ConnectionError,
)
from src.data_ingestion.mock_provider import MockDataProvider
from src.data_ingestion.ib_provider import IBDataProvider
from src.data_ingestion.polygon_provider import PolygonDataProvider
from src.data_ingestion.databento_provider import DatabentoDataProvider
from src.data_ingestion.rate_limiter import (
    RateLimiter,
    ExponentialBackoff,
    retry_with_backoff,
)

__all__ = [
    # Base interface
    "DataProvider",
    # Exceptions
    "DataProviderError",
    "AuthenticationError",
    "DataNotAvailableError",
    "RateLimitError",
    "PaginationError",
    "ConfigurationError",
    "ValidationError",
    "ConnectionError",
    # Implementations
    "MockDataProvider",
    "IBDataProvider",
    "PolygonDataProvider",
    "DatabentoDataProvider",
    # Utilities
    "RateLimiter",
    "ExponentialBackoff",
    "retry_with_backoff",
]
