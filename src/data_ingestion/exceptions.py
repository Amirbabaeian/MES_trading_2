"""
Custom exception classes for data provider operations.

This module defines the exception hierarchy for data provider errors,
including authentication, data availability, rate limiting, and validation errors.
"""


class DataProviderError(Exception):
    """Base exception for all data provider errors."""
    pass


class AuthenticationError(DataProviderError):
    """Raised when authentication with the data provider fails."""
    
    def __init__(self, message: str, provider: str = None):
        """
        Initialize AuthenticationError.
        
        Args:
            message: Error message describing the authentication failure.
            provider: Name of the provider that failed (optional).
        """
        self.message = message
        self.provider = provider
        full_message = f"[{provider}] {message}" if provider else message
        super().__init__(full_message)


class DataNotAvailableError(DataProviderError):
    """Raised when requested data is not available from the provider."""
    
    def __init__(self, message: str, symbol: str = None, start_date = None, end_date = None):
        """
        Initialize DataNotAvailableError.
        
        Args:
            message: Error message describing the unavailable data.
            symbol: The symbol that was requested (optional).
            start_date: The start date of the requested range (optional).
            end_date: The end date of the requested range (optional).
        """
        self.message = message
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        details = []
        if symbol:
            details.append(f"symbol={symbol}")
        if start_date:
            details.append(f"start={start_date}")
        if end_date:
            details.append(f"end={end_date}")
        detail_str = f" ({', '.join(details)})" if details else ""
        full_message = f"{message}{detail_str}"
        super().__init__(full_message)


class RateLimitError(DataProviderError):
    """Raised when the provider's rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: int = None):
        """
        Initialize RateLimitError.
        
        Args:
            message: Error message describing the rate limit violation.
            retry_after: Number of seconds to wait before retrying (optional).
        """
        self.message = message
        self.retry_after = retry_after
        if retry_after:
            full_message = f"{message} (retry after {retry_after} seconds)"
        else:
            full_message = message
        super().__init__(full_message)


class ValidationError(DataProviderError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field: str = None, value = None):
        """
        Initialize ValidationError.
        
        Args:
            message: Error message describing the validation failure.
            field: The field that failed validation (optional).
            value: The invalid value (optional).
        """
        self.message = message
        self.field = field
        self.value = value
        details = []
        if field:
            details.append(f"field={field}")
        if value is not None:
            details.append(f"value={value}")
        detail_str = f" ({', '.join(details)})" if details else ""
        full_message = f"{message}{detail_str}"
        super().__init__(full_message)


class SchemaError(DataProviderError):
    """Raised when returned data doesn't match the expected schema."""
    
    def __init__(self, message: str, expected_columns = None, actual_columns = None):
        """
        Initialize SchemaError.
        
        Args:
            message: Error message describing the schema mismatch.
            expected_columns: List of expected columns (optional).
            actual_columns: List of actual columns in returned data (optional).
        """
        self.message = message
        self.expected_columns = expected_columns or []
        self.actual_columns = actual_columns or []
        super().__init__(message)
    
    def get_missing_columns(self):
        """Get columns that are expected but missing."""
        return set(self.expected_columns) - set(self.actual_columns)
    
    def get_extra_columns(self):
        """Get columns that exist but weren't expected."""
        return set(self.actual_columns) - set(self.expected_columns)


class ConnectionError(DataProviderError):
    """Raised when unable to connect to the data provider."""
    
    def __init__(self, message: str, provider: str = None):
        """
        Initialize ConnectionError.
        
        Args:
            message: Error message describing the connection failure.
            provider: Name of the provider (optional).
        """
        self.message = message
        self.provider = provider
        full_message = f"[{provider}] {message}" if provider else message
        super().__init__(full_message)


class PaginationError(DataProviderError):
    """Raised when pagination handling fails."""
    
    def __init__(self, message: str, page: int = None):
        """
        Initialize PaginationError.
        
        Args:
            message: Error message describing the pagination failure.
            page: The page number where the error occurred (optional).
        """
        self.message = message
        self.page = page
        if page is not None:
            full_message = f"{message} (page {page})"
        else:
            full_message = message
        super().__init__(full_message)


class TimeoutError(DataProviderError):
    """Raised when a data provider request times out."""
    
    def __init__(self, message: str, timeout_seconds: float = None):
        """
        Initialize TimeoutError.
        
        Args:
            message: Error message describing the timeout.
            timeout_seconds: The timeout duration in seconds (optional).
        """
        self.message = message
        self.timeout_seconds = timeout_seconds
        if timeout_seconds is not None:
            full_message = f"{message} (timeout after {timeout_seconds}s)"
        else:
            full_message = message
        super().__init__(full_message)
