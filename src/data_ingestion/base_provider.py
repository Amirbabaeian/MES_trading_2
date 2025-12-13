"""
Abstract base class for vendor-agnostic data providers.

This module defines the DataProvider interface that all concrete data provider
implementations must follow. The interface enables swapping data vendors without
downstream code changes and ensures consistent data schemas across providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import pandas as pd


class DataProvider(ABC):
    """
    Abstract base class for data providers.
    
    This class defines the interface that all data provider implementations must follow.
    Providers are responsible for:
    - Authentication and session management
    - Fetching OHLCV (Open, High, Low, Close, Volume) data
    - Managing rate limits and pagination
    - Returning data in a standardized schema
    
    All providers must return data with the following schema:
    - Columns: timestamp, open, high, low, close, volume
    - Data types: timestamp (datetime64[ns, UTC]), prices (float64), volume (int64)
    - Index: timestamp as DatetimeIndex (UTC)
    - No missing values in core OHLCV columns
    
    Timezone Handling Contract:
    - All timestamps returned by providers MUST be in UTC
    - Consumers may convert to other timezones (e.g., NY time) in downstream layers
    - Providers should NOT convert to consumer timezones
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the data provider.
        
        Args:
            name: Human-readable name of the provider (e.g., "AlphaVantage", "IB", "Polygon").
        """
        self.name = name or self.__class__.__name__
        self._authenticated = False
    
    @property
    def is_authenticated(self) -> bool:
        """Check if the provider is currently authenticated."""
        return self._authenticated
    
    @abstractmethod
    def authenticate(self) -> None:
        """
        Establish connection/session with the data provider.
        
        This method should:
        - Validate API credentials/configuration
        - Establish network connection
        - Initialize any required session objects
        - Set internal state indicating successful authentication
        
        Returns:
            None
        
        Raises:
            AuthenticationError: If authentication credentials are invalid or missing.
            ConnectionError: If unable to establish connection to provider.
        
        Example:
            >>> provider = AlphaVantageProvider(api_key='YOUR_KEY')
            >>> provider.authenticate()  # Validates API key and establishes session
            >>> print(provider.is_authenticated)
            True
        """
        pass
    
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data for a symbol in a date range.
        
        This is the core method for retrieving market data. Implementations must:
        - Validate input parameters (symbol, dates, timeframe)
        - Handle pagination automatically for large date ranges
        - Return data in standardized schema (see class docstring)
        - Convert all timestamps to UTC
        - Handle vendor-specific data quirks (gaps, halts, etc.)
        
        Parameters:
            symbol: Asset symbol or ticker.
                Examples: "MES" (E-mini S&P 500 futures),
                          "ES" (E-mini S&P 500 futures),
                          "VIX" (Volatility Index futures),
                          "AAPL" (Apple stock),
                          "BTC/USD" (Bitcoin/USD pair)
            
            start_date: Start of date range (inclusive).
                Must be timezone-aware or will be assumed UTC.
                Example: datetime(2024, 1, 1, tzinfo=pytz.UTC)
            
            end_date: End of date range (inclusive).
                Must be timezone-aware or will be assumed UTC.
                Must be >= start_date.
            
            timeframe: Candlestick period as string.
                Supported values: "1M" (1 minute), "5M", "15M", "30M", "1H", "1D", "1W", "1MO"
                Default: "1D" (daily data)
        
        Returns:
            pd.DataFrame: OHLCV data with structure:
                Columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                Index: DatetimeIndex with name 'timestamp' (UTC timezone)
                Shape: (n_bars, 6) where n_bars is number of candlesticks
                Example:
                                        open     high      low    close    volume
                    timestamp
                    2024-01-01 00:00:00  100.0   101.0    99.0   100.5  1000000
                    2024-01-02 00:00:00  100.5   102.0    99.5   101.0  1200000
                    ...
        
        Raises:
            AuthenticationError: If provider requires authentication before fetching data.
            DataNotAvailableError: If data for symbol/timeframe/date range not available.
            ValidationError: If input parameters are invalid.
            RateLimitError: If provider rate limit is exceeded.
            TimeoutError: If request times out.
        
        Notes:
            - Data is expected to be continuous with no gaps (except market closures).
            - Duplicate timestamps should not appear in results.
            - Data should be sorted by timestamp in ascending order.
            - Volume should be in contracts (for futures) or shares (for stocks).
            - Prices should be adjusted for splits/dividends if applicable.
            - Returns empty DataFrame if no data available for date range.
            - Some vendors may only support certain timeframes; document in adapter.
        
        Example:
            >>> provider = AlphaVantageProvider(api_key='KEY')
            >>> provider.authenticate()
            >>> df = provider.fetch_ohlcv(
            ...     symbol='MES',
            ...     start_date=datetime(2024, 1, 1),
            ...     end_date=datetime(2024, 12, 31),
            ...     timeframe='1D'
            ... )
            >>> print(df.head())
            >>> print(df.shape)
            (252, 6)  # Approximately 252 trading days in 2024
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Retrieve list of symbols supported by this provider.
        
        This method returns all symbols that can be fetched from this provider.
        The list may be filtered or paginated depending on the provider's capabilities.
        
        Returns:
            List[str]: List of available symbols/tickers.
                Examples: ["ES", "MES", "NQ", "VIX", "AAPL", "MSFT", ...]
        
        Raises:
            AuthenticationError: If authentication is required but not completed.
            ConnectionError: If unable to connect to provider.
            TimeoutError: If request times out.
        
        Notes:
            - Some providers may have thousands of symbols; implementations should cache.
            - Symbol format is provider-specific; adapters must normalize.
            - The list may be filtered (e.g., only US equities, only futures, etc.).
        
        Example:
            >>> provider = AlphaVantageProvider(api_key='KEY')
            >>> provider.authenticate()
            >>> symbols = provider.get_available_symbols()
            >>> print(len(symbols))
            2000
            >>> print(symbols[:5])
            ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
        """
        pass
    
    def handle_pagination(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        page_size: int = 1000
    ) -> List[pd.DataFrame]:
        """
        Handle pagination for large data requests.
        
        For providers with strict data limits per request, this method breaks large
        date ranges into smaller chunks and fetches each chunk separately.
        
        This is a convenience method that automatically handles pagination logic.
        Default implementation is provided but can be overridden for vendor-specific
        pagination strategies.
        
        Parameters:
            symbol: Asset symbol to fetch.
            start_date: Start of date range (inclusive).
            end_date: End of date range (inclusive).
            timeframe: Candlestick period (e.g., "1D", "1H").
            page_size: Maximum number of bars per request (default: 1000).
                Some vendors may have stricter limits; check adapter documentation.
        
        Returns:
            List[pd.DataFrame]: List of DataFrames, one per page.
                Each DataFrame follows the standard OHLCV schema.
                Example: [df_page1, df_page2, df_page3]
        
        Raises:
            AuthenticationError: If authentication required but not completed.
            DataNotAvailableError: If data for symbol not available.
            ValidationError: If parameters are invalid.
            RateLimitError: If rate limit exceeded during pagination.
            TimeoutError: If any request times out.
        
        Notes:
            - Default implementation chunks by date range; override for custom logic.
            - Pagination may be necessary due to:
              - API limits on records per request
              - Memory constraints for large date ranges
              - Rate limiting on concurrent requests
            - Caller should combine results: pd.concat(pages, ignore_index=False)
        
        Example:
            >>> provider = AlphaVantageProvider(api_key='KEY')
            >>> provider.authenticate()
            >>> pages = provider.handle_pagination(
            ...     symbol='ES',
            ...     start_date=datetime(2020, 1, 1),
            ...     end_date=datetime(2024, 12, 31),
            ...     timeframe='1D'
            ... )
            >>> df = pd.concat(pages, ignore_index=False).sort_index()
            >>> print(df.shape)
            (1200, 6)  # 5 years of daily data
        """
        # Default implementation: estimate business days and chunk
        # Providers can override for more sophisticated pagination
        business_days = pd.bdate_range(start=start_date, end=end_date).size
        num_pages = max(1, (business_days + page_size - 1) // page_size)
        
        if num_pages == 1:
            return [self.fetch_ohlcv(symbol, start_date, end_date, timeframe)]
        
        pages = []
        current_date = start_date
        
        for page_num in range(num_pages):
            # Calculate page end date
            page_end = min(
                current_date + pd.Timedelta(days=page_size),
                end_date
            )
            
            # Fetch this page
            page_data = self.fetch_ohlcv(symbol, current_date, page_end, timeframe)
            if not page_data.empty:
                pages.append(page_data)
            
            # Move to next page
            current_date = page_end + pd.Timedelta(days=1)
            
            if current_date > end_date:
                break
        
        return pages
    
    def get_contract_details(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieve detailed information about a contract/instrument.
        
        This is an optional method for retrieving metadata about a specific symbol.
        Implementation is not required but recommended for futures and derivatives.
        
        Parameters:
            symbol: Asset symbol (e.g., "MES", "ES", "VIX").
        
        Returns:
            Dict[str, Any]: Contract details with possible keys:
                - name: Human-readable name (e.g., "E-mini S&P 500 December 2024")
                - exchange: Trading exchange (e.g., "CME", "CBOE")
                - contract_type: "STOCK", "FUTURE", "OPTION", "INDEX", etc.
                - underlying: Underlying asset symbol (for derivatives)
                - multiplier: Contract multiplier (e.g., 50 for MES, 100 for ES)
                - min_tick: Minimum price movement
                - expiration: Expiration date for futures (datetime or string)
                - active: Whether contract is currently trading (bool)
        
        Raises:
            AuthenticationError: If authentication required but not completed.
            DataNotAvailableError: If symbol not found.
            ValidationError: If symbol format invalid.
        
        Notes:
            - This method is optional; return empty dict if not implemented.
            - Different providers have different contract metadata.
            - For stocks, many fields may be None or not applicable.
        
        Example:
            >>> provider = InteractiveBrokersProvider()
            >>> provider.authenticate()
            >>> details = provider.get_contract_details('MES')
            >>> print(details)
            {
                'name': 'E-mini S&P 500 December 2024',
                'exchange': 'CME',
                'contract_type': 'FUTURE',
                'multiplier': 50,
                'expiration': datetime(2024, 12, 20)
            }
        """
        # Default: empty dict (optional method)
        return {}
    
    def __repr__(self) -> str:
        """Return string representation of provider."""
        auth_status = "authenticated" if self._authenticated else "not authenticated"
        return f"{self.__class__.__name__}(name='{self.name}', {auth_status})"
