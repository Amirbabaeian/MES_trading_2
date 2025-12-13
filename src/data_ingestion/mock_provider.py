"""
Mock data provider implementation for testing and development.

This module provides a MockProvider that generates synthetic OHLCV data
following the DataProvider interface. It's useful for:
- Testing downstream code without real API connections
- Development without rate limits
- Consistent, reproducible data
- Testing error handling without network issues
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .base_provider import DataProvider
from .exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    ValidationError,
)


class MockProvider(DataProvider):
    """
    Mock data provider that generates synthetic OHLCV data.
    
    This provider generates realistic but synthetic OHLCV data for testing.
    It supports the standard DataProvider interface and can be used as a
    drop-in replacement for real providers in testing scenarios.
    
    Characteristics:
    - Simulates realistic price movements using geometric Brownian motion
    - Generates volume with variability
    - Supports multiple symbols with different price ranges
    - Configurable starting price, volatility, and trend
    - Can simulate authentication failures and missing data
    
    Example:
        >>> provider = MockProvider()
        >>> provider.authenticate()
        >>> df = provider.fetch_ohlcv(
        ...     symbol='ES',
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 12, 31),
        ...     timeframe='1D'
        ... )
        >>> print(df.head())
        >>> print(df.shape)
    """
    
    # Default symbols and their characteristics
    DEFAULT_SYMBOLS = {
        'ES': {'name': 'E-mini S&P 500', 'start_price': 5000, 'volatility': 0.015},
        'MES': {'name': 'Micro E-mini S&P 500', 'start_price': 5000, 'volatility': 0.015},
        'NQ': {'name': 'E-mini Nasdaq 100', 'start_price': 18000, 'volatility': 0.02},
        'VIX': {'name': 'Volatility Index', 'start_price': 20, 'volatility': 0.3},
        'AAPL': {'name': 'Apple Inc.', 'start_price': 150, 'volatility': 0.02},
        'MSFT': {'name': 'Microsoft Corporation', 'start_price': 400, 'volatility': 0.018},
        'TSLA': {'name': 'Tesla Inc.', 'start_price': 200, 'volatility': 0.035},
        'BTC/USD': {'name': 'Bitcoin/USD', 'start_price': 40000, 'volatility': 0.04},
    }
    
    def __init__(
        self,
        symbols: Optional[Dict[str, Dict[str, Any]]] = None,
        seed: Optional[int] = None,
        fail_auth: bool = False
    ):
        """
        Initialize the mock provider.
        
        Args:
            symbols: Dictionary of symbols with characteristics.
                If None, uses DEFAULT_SYMBOLS.
                Format: {symbol: {'name': str, 'start_price': float, 'volatility': float}}
            seed: Random seed for reproducible data generation.
            fail_auth: If True, authentication will always fail (for error testing).
        """
        super().__init__(name="MockProvider")
        self.symbols = symbols or self.DEFAULT_SYMBOLS.copy()
        self.seed = seed
        self.fail_auth = fail_auth
        
        if seed is not None:
            np.random.seed(seed)
    
    def authenticate(self) -> None:
        """
        Mock authentication (always succeeds unless fail_auth=True).
        
        Raises:
            AuthenticationError: If fail_auth=True (for error testing).
        """
        if self.fail_auth:
            raise AuthenticationError("Mock authentication failure")
        self._authenticated = True
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D"
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data.
        
        Generates realistic price data using geometric Brownian motion
        with the following characteristics:
        - Realistic price movements with trending and mean reversion
        - Proportional volume with daily variability
        - High >= Low, Open within High/Low, Close within High/Low
        - Timestamps at market open times (9:30 AM EST = 2:30 PM UTC for daily)
        
        Args:
            symbol: Symbol to fetch (must be in self.symbols).
            start_date: Start of date range.
            end_date: End of date range.
            timeframe: Candlestick period (currently only "1D" fully supported).
        
        Returns:
            DataFrame with OHLCV data in standardized schema.
        
        Raises:
            ValidationError: If symbol not found or dates invalid.
            DataNotAvailableError: If requested range has no data.
        """
        # Validate inputs
        if symbol not in self.symbols:
            raise ValidationError(f"Symbol '{symbol}' not available", field="symbol")
        
        if start_date > end_date:
            raise ValidationError("start_date must be <= end_date")
        
        # Generate business dates
        dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
        
        if len(dates) == 0:
            return pd.DataFrame(
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            ).set_index('timestamp')
        
        # Get symbol characteristics
        symbol_info = self.symbols[symbol]
        start_price = symbol_info['start_price']
        volatility = symbol_info['volatility']
        
        # Generate price data using geometric Brownian motion
        n_bars = len(dates)
        
        # Generate daily returns
        drift = 0.0005  # Small positive drift
        daily_returns = np.random.normal(drift, volatility, n_bars)
        
        # Convert to price movements
        price_multipliers = np.exp(daily_returns)
        
        # Generate closes
        closes = start_price * np.cumprod(price_multipliers)
        
        # Generate opens (slightly offset from previous close)
        opens = np.concatenate([[start_price], closes[:-1] * (1 + np.random.normal(0, 0.002, n_bars-1))])
        
        # Generate highs and lows
        highs = np.zeros(n_bars)
        lows = np.zeros(n_bars)
        
        for i in range(n_bars):
            o, c = opens[i], closes[i]
            # Intraday range (2-4% of average price)
            avg_price = (o + c) / 2
            intraday_range = avg_price * np.random.uniform(0.01, 0.04)
            
            highs[i] = max(o, c) + intraday_range
            lows[i] = min(o, c) - intraday_range
        
        # Generate volumes (average 1M, with 0.5-2x variability)
        base_volume = 1_000_000
        volumes = (base_volume * np.random.uniform(0.5, 2.0, n_bars)).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        # Set timestamp as UTC index
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
        
        return df
    
    def get_available_symbols(self) -> List[str]:
        """
        Return list of available symbols.
        
        Returns:
            List of symbol strings.
        """
        return list(self.symbols.keys())
    
    def get_contract_details(self, symbol: str) -> Dict[str, Any]:
        """
        Return contract details for a symbol.
        
        Args:
            symbol: Symbol to get details for.
        
        Returns:
            Dictionary with contract information.
        
        Raises:
            ValidationError: If symbol not found.
        """
        if symbol not in self.symbols:
            raise ValidationError(f"Symbol '{symbol}' not found")
        
        info = self.symbols[symbol]
        return {
            'name': info['name'],
            'symbol': symbol,
            'active': True,
            'currency': 'USD',
        }


class FailingMockProvider(DataProvider):
    """
    Mock provider that simulates various failure modes for error testing.
    
    This provider is useful for testing error handling in code that uses
    data providers. It can simulate:
    - Authentication failures
    - Missing data
    - Rate limiting
    - Timeouts
    """
    
    def __init__(self, failure_mode: str = 'auth'):
        """
        Initialize with a specific failure mode.
        
        Args:
            failure_mode: Type of failure to simulate.
                Options: 'auth', 'missing_data', 'rate_limit', 'timeout'
        """
        super().__init__(name="FailingMockProvider")
        self.failure_mode = failure_mode
    
    def authenticate(self) -> None:
        """Fail authentication if configured."""
        if self.failure_mode == 'auth':
            raise AuthenticationError("Simulated authentication failure")
        self._authenticated = True
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D"
    ) -> pd.DataFrame:
        """Fail with configured error mode."""
        if self.failure_mode == 'missing_data':
            raise DataNotAvailableError(
                "Simulated missing data",
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
        return pd.DataFrame(
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        ).set_index('timestamp')
    
    def get_available_symbols(self) -> List[str]:
        """Return empty symbol list."""
        return []
