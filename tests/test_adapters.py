"""
Unit tests for data provider adapters.

Tests cover:
- Authentication flows
- OHLCV data fetching
- Symbol listing
- Rate limiting
- Error handling
- Data schema validation
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import logging

from src.data_ingestion.ib_provider import IBDataProvider
from src.data_ingestion.polygon_provider import PolygonDataProvider
from src.data_ingestion.databento_provider import DatabentoDataProvider
from src.data_ingestion.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    ValidationError,
    ConfigurationError,
    RateLimitError,
)
from src.data_ingestion.rate_limiter import RateLimiter, ExponentialBackoff

logger = logging.getLogger(__name__)


# Fixtures for common test data
@pytest.fixture
def start_date():
    """Standard test start date."""
    return datetime(2023, 1, 1)


@pytest.fixture
def end_date():
    """Standard test end date."""
    return datetime(2023, 1, 31)


@pytest.fixture
def test_symbol():
    """Standard test symbol."""
    return "ES"


@pytest.fixture
def test_timeframe():
    """Standard test timeframe."""
    return "1D"


# ============================================================================
# Interactive Brokers Adapter Tests
# ============================================================================


class TestIBDataProvider:
    """Tests for Interactive Brokers adapter."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        provider = IBDataProvider()
        assert provider.host == "127.0.0.1"
        assert provider.port == 7497
        assert provider.client_id == 1
        assert not provider._authenticated
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        provider = IBDataProvider(
            account_id="DU123456",
            host="192.168.1.1",
            port=7498,
            client_id=2,
            requests_per_second=3.0,
        )
        assert provider.account_id == "DU123456"
        assert provider.host == "192.168.1.1"
        assert provider.port == 7498
        assert provider.client_id == 2
        assert provider.rate_limiter.requests_per_second == 3.0
    
    def test_authenticate_success(self):
        """Test successful authentication."""
        provider = IBDataProvider(account_id="DU123456")
        provider.authenticate()
        assert provider._authenticated
    
    def test_authenticate_no_account_id(self):
        """Test authentication fails without account_id."""
        provider = IBDataProvider()
        with pytest.raises(ConfigurationError):
            provider.authenticate()
    
    def test_authenticate_idempotent(self):
        """Test that authenticate can be called multiple times."""
        provider = IBDataProvider(account_id="DU123456")
        provider.authenticate()
        provider.authenticate()  # Should not fail
        assert provider._authenticated
    
    def test_fetch_ohlcv_not_authenticated(self, start_date, end_date, test_symbol, test_timeframe):
        """Test fetch_ohlcv fails if not authenticated."""
        provider = IBDataProvider(account_id="DU123456")
        with pytest.raises(AuthenticationError):
            provider.fetch_ohlcv(test_symbol, start_date, end_date, test_timeframe)
    
    def test_fetch_ohlcv_success(self, start_date, end_date, test_symbol, test_timeframe):
        """Test successful OHLCV data fetch."""
        provider = IBDataProvider(account_id="DU123456")
        provider.authenticate()
        
        df = provider.fetch_ohlcv(test_symbol, start_date, end_date, test_timeframe)
        
        # Validate schema
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"
        assert str(df.index.tz) == "UTC"
    
    def test_fetch_ohlcv_unsupported_symbol(self, start_date, end_date, test_timeframe):
        """Test fetch_ohlcv fails for unsupported symbol."""
        provider = IBDataProvider(account_id="DU123456")
        provider.authenticate()
        
        with pytest.raises(DataNotAvailableError):
            provider.fetch_ohlcv("INVALID", start_date, end_date, test_timeframe)
    
    def test_fetch_ohlcv_invalid_date_range(self, test_symbol, test_timeframe):
        """Test fetch_ohlcv fails with invalid date range."""
        provider = IBDataProvider(account_id="DU123456")
        provider.authenticate()
        
        start_date = datetime(2023, 1, 31)
        end_date = datetime(2023, 1, 1)  # End before start
        
        with pytest.raises(ValidationError):
            provider.fetch_ohlcv(test_symbol, start_date, end_date, test_timeframe)
    
    def test_fetch_ohlcv_data_quality(self, start_date, end_date, test_symbol, test_timeframe):
        """Test that returned data meets schema requirements."""
        provider = IBDataProvider(account_id="DU123456")
        provider.authenticate()
        
        df = provider.fetch_ohlcv(test_symbol, start_date, end_date, test_timeframe)
        
        if len(df) > 0:
            # Check data types
            assert df["open"].dtype == "float64"
            assert df["high"].dtype == "float64"
            assert df["low"].dtype == "float64"
            assert df["close"].dtype == "float64"
            assert df["volume"].dtype == "int64"
            
            # Check value constraints
            assert (df["high"] >= df["low"]).all()
            assert (df["volume"] >= 0).all()
    
    def test_get_available_symbols_not_authenticated(self):
        """Test get_available_symbols fails if not authenticated."""
        provider = IBDataProvider(account_id="DU123456")
        with pytest.raises(AuthenticationError):
            provider.get_available_symbols()
    
    def test_get_available_symbols_success(self):
        """Test successful symbol listing."""
        provider = IBDataProvider(account_id="DU123456")
        provider.authenticate()
        
        symbols = provider.get_available_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "ES" in symbols
        assert "MES" in symbols
    
    def test_disconnect(self):
        """Test disconnect method."""
        provider = IBDataProvider(account_id="DU123456")
        provider.authenticate()
        assert provider._authenticated
        
        provider.disconnect()
        assert not provider._authenticated
    
    def test_context_manager(self, start_date, end_date, test_symbol, test_timeframe):
        """Test provider as context manager."""
        provider = IBDataProvider(account_id="DU123456")
        
        with provider:
            assert provider._authenticated
            df = provider.fetch_ohlcv(test_symbol, start_date, end_date, test_timeframe)
            assert isinstance(df, pd.DataFrame)
        
        assert not provider._authenticated


# ============================================================================
# Polygon Data Provider Tests
# ============================================================================


class TestPolygonDataProvider:
    """Tests for Polygon.io adapter."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        provider = PolygonDataProvider()
        assert provider.base_url == "https://api.polygon.io"
        assert provider.timeout == 30.0
        assert not provider._authenticated
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        provider = PolygonDataProvider(
            api_key="test_key_123",
            base_url="https://custom.polygon.io",
            timeout=60.0,
            requests_per_second=2.0,
        )
        assert provider.api_key == "test_key_123"
        assert provider.base_url == "https://custom.polygon.io"
        assert provider.timeout == 60.0
    
    def test_authenticate_success(self):
        """Test successful authentication."""
        provider = PolygonDataProvider(api_key="test_key_123")
        provider.authenticate()
        assert provider._authenticated
    
    def test_authenticate_no_api_key(self):
        """Test authentication fails without API key."""
        provider = PolygonDataProvider()
        with pytest.raises(ConfigurationError):
            provider.authenticate()
    
    def test_fetch_ohlcv_not_authenticated(self, start_date, end_date, test_symbol, test_timeframe):
        """Test fetch_ohlcv fails if not authenticated."""
        provider = PolygonDataProvider(api_key="test_key_123")
        with pytest.raises(AuthenticationError):
            provider.fetch_ohlcv(test_symbol, start_date, end_date, test_timeframe)
    
    def test_fetch_ohlcv_success(self, start_date, end_date, test_timeframe):
        """Test successful OHLCV data fetch."""
        provider = PolygonDataProvider(api_key="test_key_123")
        provider.authenticate()
        
        # Polygon should support SPY as well as ES
        df = provider.fetch_ohlcv("SPY", start_date, end_date, test_timeframe)
        
        # Validate schema
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"
        assert str(df.index.tz) == "UTC"
    
    def test_fetch_ohlcv_unsupported_symbol(self, start_date, end_date, test_timeframe):
        """Test fetch_ohlcv fails for unsupported symbol."""
        provider = PolygonDataProvider(api_key="test_key_123")
        provider.authenticate()
        
        with pytest.raises(DataNotAvailableError):
            provider.fetch_ohlcv("INVALID", start_date, end_date, test_timeframe)
    
    def test_get_available_symbols_success(self):
        """Test successful symbol listing."""
        provider = PolygonDataProvider(api_key="test_key_123")
        provider.authenticate()
        
        symbols = provider.get_available_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "SPY" in symbols
        assert "AAPL" in symbols
    
    def test_parse_timeframe(self):
        """Test timeframe parsing."""
        provider = PolygonDataProvider()
        
        assert provider._parse_timeframe("1m") == (1, "minute")
        assert provider._parse_timeframe("5m") == (5, "minute")
        assert provider._parse_timeframe("1H") == (1, "hour")
        assert provider._parse_timeframe("1D") == (1, "day")
        assert provider._parse_timeframe("1W") == (1, "week")
        assert provider._parse_timeframe("1M") == (1, "month")
        # Test default
        assert provider._parse_timeframe("INVALID") == (1, "day")


# ============================================================================
# Databento Data Provider Tests
# ============================================================================


class TestDatabentoDataProvider:
    """Tests for Databento adapter."""
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        provider = DatabentoDataProvider()
        assert provider.base_url == "https://api.databento.com"
        assert provider.dataset == "GLBX"
        assert not provider._authenticated
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        provider = DatabentoDataProvider(
            api_key="test_key_123",
            client="test_client",
            dataset="XNAS",
            requests_per_second=5.0,
        )
        assert provider.api_key == "test_key_123"
        assert provider.client == "test_client"
        assert provider.dataset == "XNAS"
    
    def test_authenticate_success(self):
        """Test successful authentication."""
        provider = DatabentoDataProvider(
            api_key="test_key_123",
            client="test_client",
        )
        provider.authenticate()
        assert provider._authenticated
    
    def test_authenticate_no_api_key(self):
        """Test authentication fails without API key."""
        provider = DatabentoDataProvider(client="test_client")
        with pytest.raises(ConfigurationError):
            provider.authenticate()
    
    def test_authenticate_no_client(self):
        """Test authentication fails without client."""
        provider = DatabentoDataProvider(api_key="test_key_123")
        with pytest.raises(ConfigurationError):
            provider.authenticate()
    
    def test_fetch_ohlcv_not_authenticated(self, start_date, end_date, test_symbol, test_timeframe):
        """Test fetch_ohlcv fails if not authenticated."""
        provider = DatabentoDataProvider(
            api_key="test_key_123",
            client="test_client",
        )
        with pytest.raises(AuthenticationError):
            provider.fetch_ohlcv(test_symbol, start_date, end_date, test_timeframe)
    
    def test_fetch_ohlcv_success(self, start_date, end_date, test_symbol, test_timeframe):
        """Test successful OHLCV data fetch."""
        provider = DatabentoDataProvider(
            api_key="test_key_123",
            client="test_client",
        )
        provider.authenticate()
        
        df = provider.fetch_ohlcv(test_symbol, start_date, end_date, test_timeframe)
        
        # Validate schema
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"
        assert str(df.index.tz) == "UTC"
    
    def test_fetch_ohlcv_unsupported_symbol(self, start_date, end_date, test_timeframe):
        """Test fetch_ohlcv fails for unsupported symbol."""
        provider = DatabentoDataProvider(
            api_key="test_key_123",
            client="test_client",
        )
        provider.authenticate()
        
        with pytest.raises(DataNotAvailableError):
            provider.fetch_ohlcv("INVALID", start_date, end_date, test_timeframe)
    
    def test_get_available_symbols_success(self):
        """Test successful symbol listing."""
        provider = DatabentoDataProvider(
            api_key="test_key_123",
            client="test_client",
        )
        provider.authenticate()
        
        symbols = provider.get_available_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert "ES" in symbols
        assert "BTC" in symbols
    
    def test_timeframe_to_nanoseconds(self):
        """Test timeframe to nanosecond conversion."""
        provider = DatabentoDataProvider()
        
        assert provider._timeframe_to_nanoseconds("1m") == 60 * 10**9
        assert provider._timeframe_to_nanoseconds("1H") == 60 * 60 * 10**9
        assert provider._timeframe_to_nanoseconds("1D") == 24 * 60 * 60 * 10**9
        assert provider._timeframe_to_nanoseconds("1W") == 7 * 24 * 60 * 60 * 10**9


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiter:
    """Tests for rate limiting utility."""
    
    def test_init_default_parameters(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter()
        assert limiter.requests_per_second == 10.0
        assert limiter.burst_size == 5
    
    def test_wait_if_needed_no_delay(self):
        """Test that wait_if_needed doesn't delay on first request."""
        import time
        limiter = RateLimiter(requests_per_second=1.0)
        
        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start
        
        # Should be nearly instantaneous
        assert elapsed < 0.1
    
    def test_reset(self):
        """Test rate limiter reset."""
        limiter = RateLimiter()
        limiter.wait_if_needed()
        assert len(limiter.request_times) == 1
        
        limiter.reset()
        assert len(limiter.request_times) == 0


class TestExponentialBackoff:
    """Tests for exponential backoff strategy."""
    
    def test_init_default_parameters(self):
        """Test ExponentialBackoff initialization."""
        backoff = ExponentialBackoff()
        assert backoff.initial_delay == 1.0
        assert backoff.max_delay == 300.0
        assert backoff.max_retries == 5
    
    def test_get_delay(self):
        """Test delay calculation."""
        backoff = ExponentialBackoff(
            initial_delay=1.0,
            max_delay=100.0,
            exponential_base=2.0,
        )
        
        # Delays should increase exponentially
        assert backoff.get_delay(0) == 1.0
        assert backoff.get_delay(1) == 2.0
        assert backoff.get_delay(2) == 4.0
        assert backoff.get_delay(3) == 8.0
        
        # Should cap at max_delay
        assert backoff.get_delay(10) == 100.0
    
    def test_get_delay_negative_attempt(self):
        """Test get_delay with negative attempt."""
        backoff = ExponentialBackoff()
        assert backoff.get_delay(-1) == 0.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestAdapterIntegration:
    """Integration tests across multiple adapters."""
    
    def test_all_adapters_return_same_schema(self, start_date, end_date, test_timeframe):
        """Test that all adapters return data with the same schema."""
        adapters = [
            IBDataProvider(account_id="DU123456"),
            PolygonDataProvider(api_key="test_key"),
            DatabentoDataProvider(api_key="test_key", client="test"),
        ]
        
        for adapter in adapters:
            adapter.authenticate()
            
            # Use a symbol supported by all
            df = adapter.fetch_ohlcv("ES", start_date, end_date, test_timeframe)
            
            # Check schema consistency
            assert list(df.columns) == ["open", "high", "low", "close", "volume"]
            assert df.index.name == "timestamp"
            assert str(df.index.tz) == "UTC"
    
    def test_empty_date_range(self):
        """Test handling of empty date range."""
        provider = IBDataProvider(account_id="DU123456")
        provider.authenticate()
        
        # Request data for a single date (weekend)
        start_date = datetime(2023, 1, 7)  # Saturday
        end_date = datetime(2023, 1, 8)    # Sunday
        
        df = provider.fetch_ohlcv("ES", start_date, end_date, "1D")
        
        # Should return empty DataFrame with correct schema
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
