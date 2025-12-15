"""
Mock data provider for testing.

This module provides a mock implementation of the DataProvider interface
that generates synthetic OHLCV data. It's useful for testing downstream
code without connecting to real market data providers.

The mock provider simulates realistic market behavior:
- Generates consistent data for a fixed set of symbols
- Creates random walk-based price movements
- Produces varying volume patterns
- Respects market hours (no data on weekends)
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any

import pandas as pd
import numpy as np

from src.data_ingestion.base_provider import DataProvider
from src.data_ingestion.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    ValidationError,
)


class MockDataProvider(DataProvider):
    """
    Mock implementation of DataProvider for testing.

    Generates synthetic OHLCV data with realistic properties:
    - Supports common equity index futures: ES, MES, NQ, YM
    - Supports VIX volatility index
    - Returns data in the standardized schema
    - Can be configured with specific seed for reproducibility

    Parameters
    ----------
    api_key : str, optional
        Dummy API key. Authentication always succeeds.
    seed : int, optional
        Random seed for reproducible synthetic data. Default is 42.
    supported_symbols : List[str], optional
        Override the default supported symbols.

    Examples
    --------
    >>> mock = MockDataProvider(seed=42)
    >>> mock.authenticate()
    >>> df = mock.fetch_ohlcv("ES", datetime(2023, 1, 1), datetime(2023, 1, 31), "1D")
    >>> print(df.head())
    """

    # Default supported symbols
    DEFAULT_SYMBOLS = ["ES", "MES", "NQ", "YM", "VIX", "AAPL", "MSFT"]

    def __init__(
        self,
        api_key: str = "mock_key",
        seed: int = 42,
        supported_symbols: List[str] = None,
    ) -> None:
        """Initialize the mock provider."""
        super().__init__()
        self.api_key = api_key
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.supported_symbols = supported_symbols or self.DEFAULT_SYMBOLS
        self._authenticated = False

    def authenticate(self) -> None:
        """
        Dummy authentication that always succeeds.

        Can be configured to fail by setting api_key to "invalid".
        """
        if self.api_key == "invalid":
            raise AuthenticationError("Invalid API key")
        self._authenticated = True

    def disconnect(self) -> None:
        """Dummy disconnect method."""
        self._authenticated = False

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for the given parameters.

        Parameters
        ----------
        symbol : str
            Symbol to fetch data for.
        start_date : datetime
            Start date (inclusive).
        end_date : datetime
            End date (inclusive).
        timeframe : str
            Timeframe (e.g., "1D", "1H", "5m").

        Returns
        -------
        pd.DataFrame
            Synthetic OHLCV data in standardized schema.

        Raises
        ------
        AuthenticationError
            If not authenticated.
        DataNotAvailableError
            If symbol is not supported or timeframe is invalid.
        ValidationError
            If date range is invalid.
        """
        if not self._authenticated:
            raise AuthenticationError("Not authenticated. Call authenticate() first.")

        if symbol not in self.supported_symbols:
            raise DataNotAvailableError(f"Symbol {symbol} not supported")

        if end_date < start_date:
            raise ValidationError("end_date must be after start_date")

        # Generate trading dates (exclude weekends)
        all_dates = pd.bdate_range(start=start_date, end=end_date, freq="B")

        if len(all_dates) == 0:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([], name="timestamp", tz="UTC"),
            )

        # Generate synthetic data
        n = len(all_dates)

        # Starting price depends on symbol
        price_map = {
            "ES": 4000.0,
            "MES": 4000.0,
            "NQ": 12000.0,
            "YM": 33000.0,
            "VIX": 18.0,
            "AAPL": 150.0,
            "MSFT": 300.0,
        }
        start_price = price_map.get(symbol, 100.0)

        # Generate random walk price movements
        returns = self.rng.normal(0.0005, 0.015, n)
        close_prices = start_price * np.exp(np.cumsum(returns))

        # Generate OHLC around close
        opens = close_prices + self.rng.normal(0, 5, n)
        highs = np.maximum(opens, close_prices) + np.abs(
            self.rng.normal(0, 10, n)
        )
        lows = np.minimum(opens, close_prices) - np.abs(
            self.rng.normal(0, 10, n)
        )

        # Generate volume (varies by symbol)
        volume_base = 500000 if symbol in ["ES", "MES"] else 100000
        volumes = volume_base + self.rng.randint(-100000, 100000, n)
        volumes = np.maximum(volumes, 1000).astype("int64")

        # Create DataFrame
        df = pd.DataFrame(
            {
                "open": opens.astype("float64"),
                "high": highs.astype("float64"),
                "low": lows.astype("float64"),
                "close": close_prices.astype("float64"),
                "volume": volumes,
            },
            index=pd.DatetimeIndex(all_dates, name="timestamp"),
        )

        # Add timezone info
        df.index = df.index.tz_localize("UTC")

        return df

    def get_available_symbols(self) -> List[str]:
        """
        Return the list of supported symbols.

        Returns
        -------
        List[str]
            List of supported symbol strings.

        Raises
        ------
        AuthenticationError
            If not authenticated.
        """
        if not self._authenticated:
            raise AuthenticationError("Not authenticated. Call authenticate() first.")
        return self.supported_symbols.copy()

    def get_contract_details(self, symbol: str) -> Dict[str, Any]:
        """
        Return mock contract details for a symbol.

        Parameters
        ----------
        symbol : str
            Symbol to retrieve details for.

        Returns
        -------
        Dict[str, Any]
            Mock contract details.

        Raises
        ------
        DataNotAvailableError
            If symbol is not supported.
        """
        if symbol not in self.supported_symbols:
            raise DataNotAvailableError(f"Symbol {symbol} not supported")

        contract_details = {
            "ES": {
                "multiplier": 50,
                "tick_size": 0.25,
                "currency": "USD",
                "exchange": "GLOBEX",
                "description": "E-mini S&P 500 Futures",
            },
            "MES": {
                "multiplier": 5,
                "tick_size": 0.25,
                "currency": "USD",
                "exchange": "GLOBEX",
                "description": "Micro E-mini S&P 500 Futures",
            },
            "NQ": {
                "multiplier": 20,
                "tick_size": 0.25,
                "currency": "USD",
                "exchange": "GLOBEX",
                "description": "E-mini Nasdaq-100 Futures",
            },
            "YM": {
                "multiplier": 5,
                "tick_size": 1.0,
                "currency": "USD",
                "exchange": "GLOBEX",
                "description": "E-mini Dow Jones Futures",
            },
            "VIX": {
                "multiplier": 100,
                "tick_size": 0.05,
                "currency": "USD",
                "exchange": "CBOE",
                "description": "Volatility Index",
            },
            "AAPL": {
                "multiplier": 1,
                "tick_size": 0.01,
                "currency": "USD",
                "exchange": "NASDAQ",
                "description": "Apple Inc. Stock",
            },
            "MSFT": {
                "multiplier": 1,
                "tick_size": 0.01,
                "currency": "USD",
                "exchange": "NASDAQ",
                "description": "Microsoft Corporation Stock",
            },
        }

        return contract_details.get(symbol, {})
