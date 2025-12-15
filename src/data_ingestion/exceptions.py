"""
Exception hierarchy for data provider errors.

This module defines custom exceptions raised by data providers when
encountering vendor-agnostic error conditions. Adapters may wrap
vendor-specific errors into these standard exception types.
"""


class DataProviderError(Exception):
    """Base exception for all data provider errors."""

    pass


class AuthenticationError(DataProviderError):
    """
    Raised when authentication with the provider fails.

    Reasons may include:
    - Invalid credentials (API key, secret, token)
    - Expired authentication token
    - Insufficient permissions for requested data
    - Connection failure during authentication handshake
    """

    pass


class DataNotAvailableError(DataProviderError):
    """
    Raised when requested data is not available from the provider.

    Reasons may include:
    - Symbol not supported by the provider
    - Data outside the provider's historical range
    - Date range contains no trading data (e.g., weekends, holidays)
    - Timeframe not supported for the symbol
    """

    pass


class RateLimitError(DataProviderError):
    """
    Raised when API rate limits are exceeded.

    Indicates that the client should implement exponential backoff
    or wait before retrying the request. The provider may include
    a retry-after interval in the error message or via headers.

    This is a transient error and the request may succeed if retried later.
    """

    pass


class PaginationError(DataProviderError):
    """
    Raised when pagination of large data requests fails.

    Reasons may include:
    - Invalid pagination parameters (page number, offset)
    - Data corruption in pagination metadata
    - Provider-side pagination service unavailable
    """

    pass


class ConfigurationError(DataProviderError):
    """
    Raised when the provider is misconfigured.

    Reasons may include:
    - Missing required configuration parameters
    - Invalid configuration values
    - Incompatible settings combination
    """

    pass


class ValidationError(DataProviderError):
    """
    Raised when input parameters fail validation.

    Reasons may include:
    - Invalid symbol format
    - Invalid date range (end_date before start_date)
    - Unsupported timeframe
    - Negative volume in returned data
    """

    pass


class ConnectionError(DataProviderError):
    """
    Raised when connection to the provider fails.

    This is distinct from AuthenticationError and may be transient.
    Reasons may include:
    - Network timeout
    - Provider service unavailable
    - DNS resolution failure
    """

    pass
