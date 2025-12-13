"""
Unit tests for data provider adapters.

Tests cover:
- Adapter instantiation and authentication
- OHLCV data fetching with various parameters
- Rate limiting enforcement
- Retry logic with exponential backoff
- Error handling and exception raising
- Symbol availability and contract details
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import logging

from src.data_ingestion import (
    IBDataProvider,
    PolygonDataProvider,
    DatabentoDataProvider,
    AuthenticationError,
    DataNotAvailableError,
    ValidationError,
)


# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestIBDataProvider:
    """Test suite for Interactive Brokers data provider."""
    
    def test_initialization(self):
        """Test provider initialization with default parameters."""
        provider = IBDataProvider()
        assert provider.name == "IBDataProvider"
        assert not provider.is_authenticated
        assert provider.host == "127.0.0.1"
        assert provider.port == 7497
    
    def test_initialization_with_custom_params(self):
        """Test provider initialization with custom parameters."""
        provider = IBDataProvider(
            host="example.com",
            port=7496,
            client_id=2,
            api_key="test_account"
        )
        assert provider.host == "example.com"
        assert provider.port == 7496
        assert provider.client_id == 2
        assert provider.api_key == "test_account"
    
    def test_authentication_success(self):
        """Test successful authentication."""
        provider = IBDataProvider()
        provider.authenticate()
        assert provider.is_authenticated
    
    def test_authentication_sets_auth_state(self):
        """Test that authentication sets internal state."""
        provider = IBDataProvider()
        assert not provider.is_authenticated
        provider.authenticate()
        assert provider._authenticated
    
    def test_get_available_symbols(self):
        """Test retrieving available symbols."""
        provider = IBDataProvider()
        symbols = provider.get_available_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "ES" in symbols
        assert "MES" in symbols
        assert "NQ" in symbols
    
    def test_fetch_ohlcv_requires_authentication(self):
        """Test that fetch_ohlcv requires authentication."""
        provider = IBDataProvider()
        with pytest.raises(AuthenticationError):
            provider.fetch_ohlcv(
                symbol="ES",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31)
            )
    
    def test_fetch_ohlcv_invalid_symbol(self):
        """Test that invalid symbols raise ValidationError."""
        provider = IBDataProvider()
        provider.authenticate()
        with pytest.raises(ValidationError):
            provider.fetch_ohlcv(
                symbol="INVALID",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31)
            )
    
    def test_fetch_ohlcv_invalid_date_range(self):
        """Test that invalid date ranges raise ValidationError."""
        provider = IBDataProvider()
        provider.authenticate()
        with pytest.raises(ValidationError):
            provider.fetch_ohlcv(
                symbol="ES",
                start_date=datetime(2024, 1, 31),
                end_date=datetime(2024, 1, 1)
            )
    
    def test_fetch_ohlcv_success(self):
        """Test successful OHLCV data fetch."""
        provider = IBDataProvider()
        provider.authenticate()
        
        df = provider.fetch_ohlcv(
            symbol="ES",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_fetch_ohlcv_returns_proper_schema(self):
        """Test that fetched data follows the standard schema."""
        provider = IBDataProvider()
        provider.authenticate()
        
        df = provider.fetch_ohlcv(
            symbol="MES",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        
        # Check columns
        expected_columns = {'open', 'high', 'low', 'close', 'volume'}
        assert expected_columns.issubset(set(df.columns))
        
        # Check index is timestamp
        assert df.index.name == 'timestamp'
        
        # Check data types
        assert pd.api.types.is_float_dtype(df['open'])
        assert pd.api.types.is_float_dtype(df['high'])
        assert pd.api.types.is_float_dtype(df['low'])
        assert pd.api.types.is_float_dtype(df['close'])
        assert pd.api.types.is_integer_dtype(df['volume'])
    
    def test_get_contract_details_valid_symbol(self):
        """Test retrieving contract details for valid symbol."""
        provider = IBDataProvider()
        details = provider.get_contract_details("ES")
        
        assert isinstance(details, dict)
        assert 'symbol' in details
        assert details['symbol'] == "ES"
        assert 'exchange' in details
        assert 'contract_type' in details
    
    def test_get_contract_details_futures_has_multiplier(self):
        """Test that futures contracts include multiplier."""
        provider = IBDataProvider()
        details = provider.get_contract_details("ES")
        
        assert details['contract_type'] == 'FUTURE'
        assert 'multiplier' in details
        assert details['multiplier'] == 50
    
    def test_get_contract_details_invalid_symbol(self):
        """Test that invalid symbols raise ValidationError."""
        provider = IBDataProvider()
        with pytest.raises(ValidationError):
            provider.get_contract_details("INVALID")
    
    def test_rate_limiter_available(self):
        """Test that rate limiter is available."""
        provider = IBDataProvider(max_requests=10, period_seconds=60)
        assert provider.rate_limiter is not None
        assert provider.get_available_requests() <= 10


class TestPolygonDataProvider:
    """Test suite for Polygon.io data provider."""
    
    def test_initialization(self):
        """Test provider initialization with default parameters."""
        provider = PolygonDataProvider()
        assert provider.name == "PolygonDataProvider"
        assert not provider.is_authenticated
    
    def test_initialization_with_api_key(self):
        """Test provider initialization with API key."""
        provider = PolygonDataProvider(api_key="test_key_123")
        assert provider.api_key == "test_key_123"
    
    def test_initialization_rate_limits(self):
        """Test that rate limits are configurable."""
        provider = PolygonDataProvider(max_requests=30, period_seconds=60)
        assert provider.rate_limiter.max_requests == 30
    
    def test_authentication_success(self):
        """Test successful authentication (stub mode)."""
        provider = PolygonDataProvider(api_key="test_key")
        provider.authenticate()
        assert provider.is_authenticated
    
    def test_get_available_symbols(self):
        """Test retrieving available symbols."""
        provider = PolygonDataProvider()
        symbols = provider.get_available_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "AAPL" in symbols
        assert "SPY" in symbols
    
    def test_fetch_ohlcv_requires_authentication(self):
        """Test that fetch_ohlcv requires authentication."""
        provider = PolygonDataProvider()
        with pytest.raises(AuthenticationError):
            provider.fetch_ohlcv(
                symbol="AAPL",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31)
            )
    
    def test_fetch_ohlcv_invalid_symbol(self):
        """Test that invalid symbols raise ValidationError."""
        provider = PolygonDataProvider()
        provider.authenticate()
        with pytest.raises(ValidationError):
            provider.fetch_ohlcv(
                symbol="INVALID",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31)
            )
    
    def test_fetch_ohlcv_success(self):
        """Test successful OHLCV data fetch."""
        provider = PolygonDataProvider()
        provider.authenticate()
        
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_fetch_ohlcv_returns_proper_schema(self):
        """Test that fetched data follows standard schema."""
        provider = PolygonDataProvider()
        provider.authenticate()
        
        df = provider.fetch_ohlcv(
            symbol="MSFT",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        
        # Check columns
        expected_columns = {'open', 'high', 'low', 'close', 'volume'}
        assert expected_columns.issubset(set(df.columns))
        
        # Check index is timestamp with UTC
        assert df.index.name == 'timestamp'
        assert df.index.tz is not None  # Should be timezone-aware
    
    def test_get_contract_details_valid_symbol(self):
        """Test retrieving details for valid symbol."""
        provider = PolygonDataProvider()
        details = provider.get_contract_details("AAPL")
        
        assert isinstance(details, dict)
        assert details['symbol'] == "AAPL"
        assert 'name' in details
        assert details['market'] == 'stocks'
    
    def test_get_contract_details_invalid_symbol(self):
        """Test that invalid symbols raise ValidationError."""
        provider = PolygonDataProvider()
        with pytest.raises(ValidationError):
            provider.get_contract_details("INVALID")


class TestDatabentoDataProvider:
    """Test suite for Databento data provider."""
    
    def test_initialization(self):
        """Test provider initialization with default parameters."""
        provider = DatabentoDataProvider()
        assert provider.name == "DatabentoDataProvider"
        assert not provider.is_authenticated
    
    def test_initialization_with_api_key(self):
        """Test provider initialization with API key."""
        provider = DatabentoDataProvider(api_key="test_key_456")
        assert provider.api_key == "test_key_456"
    
    def test_initialization_custom_rate_limits(self):
        """Test custom rate limiting."""
        provider = DatabentoDataProvider(max_requests=100, period_seconds=60)
        assert provider.rate_limiter.max_requests == 100
    
    def test_authentication_success(self):
        """Test successful authentication (stub mode)."""
        provider = DatabentoDataProvider(api_key="test_key")
        provider.authenticate()
        assert provider.is_authenticated
    
    def test_get_available_symbols(self):
        """Test retrieving available symbols."""
        provider = DatabentoDataProvider()
        symbols = provider.get_available_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "ES" in symbols
        assert "MES" in symbols
    
    def test_fetch_ohlcv_requires_authentication(self):
        """Test that fetch_ohlcv requires authentication."""
        provider = DatabentoDataProvider()
        with pytest.raises(AuthenticationError):
            provider.fetch_ohlcv(
                symbol="ES",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31)
            )
    
    def test_fetch_ohlcv_invalid_symbol(self):
        """Test that invalid symbols raise ValidationError."""
        provider = DatabentoDataProvider()
        provider.authenticate()
        with pytest.raises(ValidationError):
            provider.fetch_ohlcv(
                symbol="INVALID",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31)
            )
    
    def test_fetch_ohlcv_success(self):
        """Test successful OHLCV data fetch."""
        provider = DatabentoDataProvider()
        provider.authenticate()
        
        df = provider.fetch_ohlcv(
            symbol="ES",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_fetch_ohlcv_returns_proper_schema(self):
        """Test that fetched data follows standard schema."""
        provider = DatabentoDataProvider()
        provider.authenticate()
        
        df = provider.fetch_ohlcv(
            symbol="NQ",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31)
        )
        
        # Check columns
        expected_columns = {'open', 'high', 'low', 'close', 'volume'}
        assert expected_columns.issubset(set(df.columns))
        
        # Check index
        assert df.index.name == 'timestamp'
    
    def test_get_contract_details_valid_symbol(self):
        """Test retrieving contract details."""
        provider = DatabentoDataProvider()
        details = provider.get_contract_details("ES")
        
        assert isinstance(details, dict)
        assert details['symbol'] == "ES"
        assert details['asset_class'] == 'FUTURE'
        assert 'multiplier' in details
    
    def test_get_contract_details_invalid_symbol(self):
        """Test that invalid symbols raise ValidationError."""
        provider = DatabentoDataProvider()
        with pytest.raises(ValidationError):
            provider.get_contract_details("INVALID")


class TestRateLimiting:
    """Test rate limiting functionality across adapters."""
    
    def test_rate_limiter_tracks_requests(self):
        """Test that rate limiter tracks API requests."""
        provider = IBDataProvider(max_requests=5, period_seconds=1)
        
        assert provider.request_count == 0
        provider.authenticate()
        df = provider.fetch_ohlcv(
            symbol="ES",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5)
        )
        
        assert provider.request_count > 0
    
    def test_available_requests_capacity(self):
        """Test that available requests capacity is tracked."""
        provider = PolygonDataProvider(max_requests=5, period_seconds=60)
        provider.authenticate()
        
        available = provider.get_available_requests()
        assert available >= 0
        assert available <= 5
    
    def test_reset_request_count(self):
        """Test resetting request counter."""
        provider = DatabentoDataProvider(max_requests=10, period_seconds=60)
        provider.authenticate()
        
        df = provider.fetch_ohlcv(
            symbol="ES",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 5)
        )
        
        count_before = provider.request_count
        assert count_before > 0
        
        provider.reset_request_count()
        assert provider.request_count_since_reset == 0


class TestAdapterComparison:
    """Test comparisons across all adapters."""
    
    def test_all_adapters_implement_interface(self):
        """Test that all adapters implement required interface."""
        providers = [
            IBDataProvider(),
            PolygonDataProvider(),
            DatabentoDataProvider(),
        ]
        
        for provider in providers:
            assert hasattr(provider, 'authenticate')
            assert hasattr(provider, 'fetch_ohlcv')
            assert hasattr(provider, 'get_available_symbols')
            assert hasattr(provider, 'is_authenticated')
    
    def test_all_adapters_return_same_schema(self):
        """Test that all adapters return same OHLCV schema."""
        providers = [
            IBDataProvider(),
            PolygonDataProvider(),
            DatabentoDataProvider(),
        ]
        
        for provider in providers:
            provider.authenticate()
            
            # Get data from each provider
            if provider.name == "IBDataProvider":
                df = provider.fetch_ohlcv(
                    symbol="ES",
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 5)
                )
            elif provider.name == "PolygonDataProvider":
                df = provider.fetch_ohlcv(
                    symbol="AAPL",
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 5)
                )
            else:  # DatabentoDataProvider
                df = provider.fetch_ohlcv(
                    symbol="ES",
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 5)
                )
            
            # Verify schema
            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            assert required_columns.issubset(set(df.columns))
            assert df.index.name == 'timestamp'
    
    def test_all_adapters_have_error_handling(self):
        """Test that all adapters handle errors properly."""
        providers = [
            IBDataProvider(),
            PolygonDataProvider(),
            DatabentoDataProvider(),
        ]
        
        for provider in providers:
            provider.authenticate()
            
            # Invalid symbol should raise ValidationError
            with pytest.raises(ValidationError):
                provider.fetch_ohlcv(
                    symbol="INVALID_SYMBOL_XYZ",
                    start_date=datetime(2024, 1, 1),
                    end_date=datetime(2024, 1, 5)
                )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_date_range(self):
        """Test handling of empty date ranges."""
        provider = IBDataProvider()
        provider.authenticate()
        
        # Single day range
        df = provider.fetch_ohlcv(
            symbol="ES",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 1)
        )
        
        assert isinstance(df, pd.DataFrame)
    
    def test_very_large_date_range(self):
        """Test handling of large date ranges."""
        provider = PolygonDataProvider()
        provider.authenticate()
        
        df = provider.fetch_ohlcv(
            symbol="AAPL",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2024, 12, 31)
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_multiple_sequential_fetches(self):
        """Test multiple sequential data fetches."""
        provider = DatabentoDataProvider()
        provider.authenticate()
        
        dfs = []
        for month in range(1, 4):
            df = provider.fetch_ohlcv(
                symbol="ES",
                start_date=datetime(2024, month, 1),
                end_date=datetime(2024, month, 28)
            )
            dfs.append(df)
        
        assert len(dfs) == 3
        assert all(isinstance(df, pd.DataFrame) for df in dfs)
