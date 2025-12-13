"""
Abstract base class for vendor-agnostic data providers.

This module defines the DataProvider interface that all concrete implementations
must follow. The interface abstracts away vendor-specific details and ensures
consistent data retrieval across different market data sources.

Key Design Principles:
- Vendor-agnostic: No vendor-specific details in this base class
- Standardized output: All providers return data in the same schema
- Timezone contract: Providers return UTC timestamps; consumers convert as needed
- Extensibility: Support optional methods for extended functionality
- Error clarity: Well-defined exception hierarchy for error handling
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd

from src.data_ingestion.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    PaginationError,
)


class DataProvider(ABC):
    """
    Abstract base class for market data providers.

    All concrete implementations (e.g., IQFeed, Polygon.io, etc.) must
    inherit from this class and implement all abstract methods.

    Standardized Output Schema
    --------------------------
    All OHLCV data must be returned as a pandas DataFrame with:
    - Columns: timestamp, open, high, low, close, volume
    - Index: DatetimeIndex with name 'timestamp'
    - Dtypes:
      - timestamp: datetime64[ns, UTC]
      - open, high, low, close: float64
      - volume: int64
    - All timestamps in UTC (conversion to other timezones happens downstream)
    - Sorted chronologically (oldest to newest)
    - No gaps or NaN values

    Thread Safety
    ~~~~~~~~~~~~~~
    Implementations should document thread-safety guarantees:
    - authenticate() should be thread-safe
    - fetch_ohlcv() may not be thread-safe for the same symbol
    - Consider connection pooling for concurrent requests
    """

    def __init__(self) -> None:
        """
        Initialize the data provider.

        Subclasses may override to set up configuration, validate parameters, etc.
        """
        self._authenticated = False

    def __enter__(self):
        """
        Context manager entry.

        Automatically authenticates the provider when used with 'with' statement.
        """
        self.authenticate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.

        Subclasses may override to clean up resources.
        """
        self.disconnect()

    @abstractmethod
    def authenticate(self) -> None:
        """
        Establish connection and authenticate with the provider.

        This method must be called before calling fetch_ohlcv() or
        get_available_symbols(). It may establish API connections, validate
        credentials, or initialize session state.

        Raises
        ------
        AuthenticationError
            If authentication fails due to invalid credentials, expired tokens,
            network issues, or insufficient permissions.
        ConfigurationError
            If required configuration parameters are missing.
        ConnectionError
            If the connection cannot be established.

        Examples
        --------
        >>> provider = ConcreteDataProvider(api_key="...", api_secret="...")
        >>> provider.authenticate()
        >>> # Now safe to call fetch_ohlcv()
        """
        pass

    def disconnect(self) -> None:
        """
        Close the connection to the provider.

        This method is called automatically when the provider is used
        as a context manager. Subclasses may override to clean up
        resources (e.g., close WebSocket connections, flush buffers).

        This is a non-abstract optional method with a default no-op implementation.
        """
        pass

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV (Open, High, Low, Close, Volume) bars for a symbol.

        This is the core method for data retrieval. Implementations must handle
        pagination internally if the date range is large. Return data must conform
        to the standardized schema documented in the class docstring.

        Parameters
        ----------
        symbol : str
            The asset symbol to fetch data for. Format is vendor-specific but
            should be uppercase (e.g., "ES", "MES", "VIX", "AAPL").
            Symbols should be validated against get_available_symbols().
        start_date : datetime
            Start of the date range (inclusive). Should be a datetime object.
            Time-of-day is typically ignored (market open time assumed).
        end_date : datetime
            End of the date range (inclusive). Should be after start_date.
        timeframe : str
            Aggregation interval for bars. Vendor-agnostic formats:
            - "1m", "5m", "15m", "30m", "60m" (intraday minutes)
            - "D" or "1D" (daily)
            - "W" or "1W" (weekly)
            - "M" or "1M" (monthly)
            Other formats may be supported by specific vendors.

        Returns
        -------
        pd.DataFrame
            OHLCV data in standardized schema:
            - Index: DatetimeIndex named 'timestamp' (UTC)
            - Columns: ['open', 'high', 'low', 'close', 'volume']
            - Dtypes: float64 (prices), int64 (volume)
            - Sorted chronologically (oldest to newest)
            - No gaps or NaN values
            - Empty DataFrame with correct schema if no data available

        Raises
        ------
        DataNotAvailableError
            If the symbol is not supported, the timeframe is not available,
            or the date range contains no trading data.
        RateLimitError
            If API rate limits are exceeded (may be retried).
        AuthenticationError
            If the session is no longer authenticated.
        ValidationError
            If input parameters are invalid (e.g., end_date before start_date).
        ConnectionError
            If the connection to the provider is lost.

        Notes
        -----
        Pagination: Implementations must handle pagination internally. If a vendor
        has data limits per request, the implementation should split the date range
        and combine results transparently.

        Timezone: All returned timestamps are in UTC. Consumers must convert to
        their desired timezone (e.g., US/Eastern for market hours).

        Data Quality: Providers should validate returned data:
        - No negative volumes
        - High >= Low, High >= Close, Open, Low <= Close, Open
        - No duplicate timestamps
        - No gaps in trading hours (weekends/holidays are expected)

        Examples
        --------
        >>> provider = ConcreteDataProvider(api_key="...")
        >>> provider.authenticate()
        >>> df = provider.fetch_ohlcv(
        ...     symbol="ES",
        ...     start_date=datetime(2023, 1, 1),
        ...     end_date=datetime(2023, 1, 31),
        ...     timeframe="1D"
        ... )
        >>> df.head()
                                open      high       low     close  volume
        timestamp
        2023-01-01 17:00:00+00:00  3800.0  3850.0  3790.0  3810.0  1000000
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Retrieve the list of symbols (assets) available from this provider.

        This list should be used to validate symbols before calling fetch_ohlcv().
        The returned list may be cached to avoid repeated API calls.

        Returns
        -------
        List[str]
            A list of supported symbols in uppercase. Format is vendor-specific
            (e.g., ["ES", "MES", "NQ", "VIX", "AAPL", ...]).

        Raises
        ------
        AuthenticationError
            If the session is no longer authenticated.
        ConnectionError
            If the connection to the provider is lost.
        RateLimitError
            If API rate limits are exceeded (may be retried).

        Notes
        -----
        Caching: Implementations may cache this list and only refresh it
        when authenticate() is called or on explicit request.

        Returns: If the provider supports thousands of symbols, consider
        implementing a separate method for filtered symbol lookup
        (not in this interface).

        Examples
        --------
        >>> provider = ConcreteDataProvider(api_key="...")
        >>> provider.authenticate()
        >>> symbols = provider.get_available_symbols()
        >>> if "ES" in symbols:
        ...     df = provider.fetch_ohlcv("ES", ...)
        """
        pass

    def handle_pagination(
        self,
        request_func,
        max_records_per_request: int,
        total_records: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Handle pagination for large data requests.

        This method is a utility for implementations that need to paginate
        through large result sets. It is not abstract; implementations may use it
        or override it with vendor-specific pagination logic.

        Parameters
        ----------
        request_func : callable
            A function that accepts pagination parameters and returns
            a DataFrame. Signature should be compatible with:
            request_func(offset=..., limit=..., **kwargs)
        max_records_per_request : int
            Maximum number of records the vendor allows per request.
        total_records : Optional[int]
            Total number of records expected. If None, will paginate until
            request_func returns fewer records than max_records_per_request.
        **kwargs
            Additional keyword arguments to pass to request_func.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame from all paginated requests, in the standardized
            OHLCV schema.

        Raises
        ------
        PaginationError
            If pagination metadata is invalid or corrupt.

        Notes
        -----
        This is a helper method for implementations. Concrete providers may
        handle pagination differently or not use this method at all.

        Example Implementation (for a hypothetical adapter):
        >>> def fetch_ohlcv(self, symbol, start_date, end_date, timeframe):
        ...     def request_page(offset, limit):
        ...         return self.api.get_bars(
        ...             symbol=symbol,
        ...             start=start_date,
        ...             end=end_date,
        ...             timeframe=timeframe,
        ...             offset=offset,
        ...             limit=limit,
        ...         )
        ...     return self.handle_pagination(
        ...         request_page,
        ...         max_records_per_request=1000,
        ...     )
        """
        all_data = []
        offset = 0

        while True:
            batch = request_func(offset=offset, limit=max_records_per_request, **kwargs)

            if batch.empty:
                break

            all_data.append(batch)
            offset += len(batch)

            if total_records is not None and offset >= total_records:
                break

            if len(batch) < max_records_per_request:
                break

        if not all_data:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([], name="timestamp"),
            )

        result = pd.concat(all_data, ignore_index=False)
        return result

    def validate_ohlcv_data(self, df: pd.DataFrame) -> None:
        """
        Validate that a DataFrame conforms to the standardized OHLCV schema.

        This is a helper method that implementations can call to validate
        their data before returning. Raises ValidationError if data is invalid.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Raises
        ------
        ValidationError
            If the DataFrame doesn't conform to the schema.
        """
        from src.data_ingestion.exceptions import ValidationError

        # Check columns
        required_columns = {"open", "high", "low", "close", "volume"}
        if set(df.columns) != required_columns:
            raise ValidationError(
                f"DataFrame columns must be {required_columns}, got {set(df.columns)}"
            )

        # Check index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValidationError("DataFrame index must be DatetimeIndex")
        if df.index.name != "timestamp":
            raise ValidationError("DatetimeIndex name must be 'timestamp'")
        if df.index.tz is None or str(df.index.tz) != "UTC":
            raise ValidationError("DatetimeIndex must be timezone-aware (UTC)")

        # Check dtypes
        if df["volume"].dtype != "int64":
            raise ValidationError(f"Volume dtype must be int64, got {df['volume'].dtype}")
        for col in ["open", "high", "low", "close"]:
            if df[col].dtype != "float64":
                raise ValidationError(f"{col} dtype must be float64, got {df[col].dtype}")

        # Check data validity
        if (df["volume"] < 0).any():
            raise ValidationError("Volume must be non-negative")
        if (df["high"] < df["low"]).any():
            raise ValidationError("High must be >= Low")

    def get_contract_details(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieve contract details for a symbol (optional, vendor-specific).

        This is an optional method that providers may implement to return
        additional metadata about a symbol (e.g., contract multiplier, tick size,
        trading hours, currency).

        Parameters
        ----------
        symbol : str
            The symbol to retrieve details for.

        Returns
        -------
        Dict[str, Any]
            A dictionary with vendor-specific contract details. Common keys:
            - 'multiplier': Contract multiplier (e.g., 50 for ES)
            - 'tick_size': Minimum price movement
            - 'currency': Currency in which prices are quoted
            - 'trading_hours': Trading hours in a standard format
            - 'exchange': Primary exchange for the symbol

        Raises
        ------
        DataNotAvailableError
            If the symbol is not found.
        NotImplementedError
            If this provider doesn't support contract details.

        Notes
        -----
        This method is optional. Providers that don't support it may raise
        NotImplementedError or return an empty dictionary.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support contract details"
        )
