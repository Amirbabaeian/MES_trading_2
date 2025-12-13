"""
Data ingestion module for fetching OHLCV data from various vendors.

This module provides an abstract interface (DataProvider) for data vendors,
allowing downstream code to work with different data sources without changes.
Includes concrete implementations for multiple vendors and mock providers for testing.
"""

# Base classes and interfaces
from .base_provider import DataProvider

# Exceptions
from .exceptions import (
    DataProviderError,
    AuthenticationError,
    DataNotAvailableError,
    RateLimitError,
    ValidationError,
    SchemaError,
    ConnectionError,
    PaginationError,
    TimeoutError,
)

# Rate limiting and retry utilities
from .rate_limiter import RateLimiter, RateLimiterMixin, retry_with_backoff

# Mock implementations
from .mock_provider import MockProvider, FailingMockProvider

# Vendor-specific adapters
from .ib_provider import IBDataProvider
from .polygon_provider import PolygonDataProvider
from .databento_provider import DatabentoDataProvider

# Credential management
from .credentials import (
    CredentialLoader,
    CredentialManager,
    CredentialValidator,
    CredentialTester,
    SensitiveDataMasker,
    IBCredentialValidator,
    PolygonCredentialValidator,
    DatabentoCredentialValidator,
)

# Advanced retry and progress utilities
from .retry import (
    RetryConfig,
    retry_with_config,
    RequestRateLimiter,
    RetryableError,
)

# Progress tracking for resumable jobs
from .progress import (
    ProgressTracker,
    ProgressState,
)

# Orchestration system for coordinated data fetching
from .orchestrator import (
    Orchestrator,
    IngestionTask,
    IngestionResult,
    DataValidationError,
)

__all__ = [
    # Base class
    "DataProvider",
    # Exceptions
    "DataProviderError",
    "AuthenticationError",
    "DataNotAvailableError",
    "RateLimitError",
    "ValidationError",
    "SchemaError",
    "ConnectionError",
    "PaginationError",
    "TimeoutError",
    # Rate limiting and retry utilities
    "RateLimiter",
    "RateLimiterMixin",
    "retry_with_backoff",
    # Advanced retry utilities
    "RetryConfig",
    "retry_with_config",
    "RequestRateLimiter",
    "RetryableError",
    # Mock providers
    "MockProvider",
    "FailingMockProvider",
    # Vendor adapters
    "IBDataProvider",
    "PolygonDataProvider",
    "DatabentoDataProvider",
    # Credential management
    "CredentialLoader",
    "CredentialManager",
    "CredentialValidator",
    "CredentialTester",
    "SensitiveDataMasker",
    "IBCredentialValidator",
    "PolygonCredentialValidator",
    "DatabentoCredentialValidator",
    # Progress tracking
    "ProgressTracker",
    "ProgressState",
    # Orchestration
    "Orchestrator",
    "IngestionTask",
    "IngestionResult",
    "DataValidationError",
]
