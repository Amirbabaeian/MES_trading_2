# Data Provider Adapter Implementation Guide

## Overview

This guide documents the concrete adapter implementations for the data provider interface. All three adapters (Interactive Brokers, Polygon, Databento) are currently **stub implementations** that return synthetic mock data matching the standardized schema. They are production-ready for testing the downstream data ingestion pipeline and can be completed once a vendor is selected.

## Architecture

All adapters follow the same architecture:

```
DataProvider (abstract base class)
├── IBDataProvider (Interactive Brokers stub)
├── PolygonDataProvider (Polygon.io stub)
└── DatabentoDataProvider (Databento stub)
```

### Key Components

1. **Rate Limiting** (`rate_limiter.py`)
   - `RateLimiter`: Tracks requests and enforces configurable rate limits
   - `ExponentialBackoff`: Implements exponential backoff for retry logic
   - `@retry_with_backoff`: Decorator for automatic retries

2. **Adapters** (one file per vendor)
   - Implements all abstract methods from `DataProvider`
   - Returns synthetic OHLCV data matching standardized schema
   - Includes authenticated state tracking
   - Configuration placeholders for future implementation

3. **Configuration Templates** (`config/provider_templates/`)
   - YAML files documenting required and optional configuration
   - Credential structure and rate limit settings
   - Asset classes and timeframes supported by each vendor

## Adapters Summary

### 1. Interactive Brokers (`ib_provider.py`)

**Status**: Stub implementation with synthetic data

**Purpose**: Provide adapter for Interactive Brokers TWS/Gateway API

**Key Features**:
- Supports futures (ES, NQ, YM, etc.), stocks, and indices
- Local connection to TWS/Gateway on configurable host:port
- Credential structure: `account_id`, `host`, `port`, `client_id`
- Rate limit: 5 requests/second (conservative)
- Supports: 1m, 5m, 15m, 30m, 60m, 1D, 1W, 1M timeframes

**Default Symbols**: ES, MES, NQ, YM, RTY, VIX, AAPL, MSFT, GOOGL, AMZN

**Configuration File**: `config/provider_templates/ib_config.yaml`

**Usage Example**:
```python
from src.data_ingestion.ib_provider import IBDataProvider
from datetime import datetime

provider = IBDataProvider(
    account_id="DU123456",
    host="127.0.0.1",
    port=7497,
)
provider.authenticate()

# Fetch ES data for January 2023, daily bars
df = provider.fetch_ohlcv(
    symbol="ES",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 1, 31),
    timeframe="1D"
)

# Get available symbols
symbols = provider.get_available_symbols()
provider.disconnect()
```

**Future Implementation Checklist**:
- [ ] Install `ib_insync` package
- [ ] Implement TWS/Gateway connection via `ib_insync.IB()`
- [ ] Add contract lookup and qualification
- [ ] Implement historical data requests with pagination
- [ ] Add support for extended trading hours
- [ ] Implement tick data aggregation for sub-minute timeframes

---

### 2. Polygon.io (`polygon_provider.py`)

**Status**: Stub implementation with synthetic data

**Purpose**: Provide adapter for Polygon.io REST API

**Key Features**:
- Supports stocks, options, forex, crypto, indices
- REST API based (no local connection required)
- Credential structure: `api_key`
- Rate limit: 4 requests/second (Starter tier, adjustable)
- Supports: 1m, 5m, 15m, 30m, 60m, 1D, 1W, 1M, 1Q, 1Y timeframes

**Default Symbols**: SPY, QQQ, DIA, AAPL, MSFT, GOOGL, AMZN, TSLA, META

**Configuration File**: `config/provider_templates/polygon_config.yaml`

**Usage Example**:
```python
from src.data_ingestion.polygon_provider import PolygonDataProvider
from datetime import datetime

provider = PolygonDataProvider(api_key="your_api_key_here")
provider.authenticate()

# Fetch SPY data
df = provider.fetch_ohlcv(
    symbol="SPY",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 1, 31),
    timeframe="1D"
)

symbols = provider.get_available_symbols()
```

**Future Implementation Checklist**:
- [ ] Install `requests` or `polygon` SDK
- [ ] Implement API key validation
- [ ] Add actual API calls to `/v1/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}`
- [ ] Implement pagination for large date ranges
- [ ] Add response parsing (convert Polygon fields: o, h, l, c, v, vw)
- [ ] Implement 429 rate limit handling with longer backoff
- [ ] Add support for split/dividend adjustments
- [ ] Cache available symbols with TTL

---

### 3. Databento (`databento_provider.py`)

**Status**: Stub implementation with synthetic data

**Purpose**: Provide adapter for Databento Historical API

**Key Features**:
- Supports futures, stocks, options, crypto
- REST API with optional streaming
- Credential structure: `api_key`, `client` (typically email)
- Rate limit: 10 requests/second (generous, adjustable)
- Supports: 1m, 5m, 15m, 30m, 60m, 1H, 1D, 1W, 1M timeframes

**Default Symbols**: ES, MES, NQ, YM, RTY, GC, CL, NG, VIX, AAPL, MSFT, GOOGL, BTC, ETH

**Configuration File**: `config/provider_templates/databento_config.yaml`

**Usage Example**:
```python
from src.data_ingestion.databento_provider import DatabentoDataProvider
from datetime import datetime

provider = DatabentoDataProvider(
    api_key="your_api_key",
    client="your_email@example.com"
)
provider.authenticate()

# Fetch ES futures data
df = provider.fetch_ohlcv(
    symbol="ES",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 1, 31),
    timeframe="1D"
)

symbols = provider.get_available_symbols()
```

**Future Implementation Checklist**:
- [ ] Install `databento` SDK
- [ ] Implement API key and client validation
- [ ] Add Historical API calls via `Historical.get_range()`
- [ ] Implement dataset selection (GLBX for futures, XNAS/XNYS for equities)
- [ ] Add nanosecond interval conversion for timeframes
- [ ] Parse Databento response format (CSV or AVRO)
- [ ] Implement pagination for large date ranges
- [ ] Add support for tick data and quote data
- [ ] Cache symbol lists per dataset

---

## Rate Limiting Framework

### RateLimiter Class

Controls request frequency to respect API rate limits:

```python
from src.data_ingestion.rate_limiter import RateLimiter

limiter = RateLimiter(
    requests_per_second=5.0,
    burst_size=3
)

# Before making an API call:
limiter.wait_if_needed()
# ... make API call ...
```

**Features**:
- Tracks request timestamps in sliding window
- Allows configurable burst size before enforcing delays
- Minimal overhead for requests within limits

### ExponentialBackoff Class

Implements exponential backoff for retry logic:

```python
from src.data_ingestion.rate_limiter import ExponentialBackoff

backoff = ExponentialBackoff(
    initial_delay=1.0,
    max_delay=300.0,
    exponential_base=2.0,
    max_retries=5
)

# Calculate delay for attempt 0, 1, 2, etc.
delay = backoff.get_delay(attempt=0)  # 1.0 second
delay = backoff.get_delay(attempt=1)  # 2.0 seconds
delay = backoff.get_delay(attempt=2)  # 4.0 seconds

# Sleep for calculated delay
backoff.sleep(attempt=0)
```

**Features**:
- Calculates exponential delays: initial_delay * (base ^ attempt)
- Caps at max_delay to prevent extremely long waits
- Suitable for 429 (rate limit) and 503 (service unavailable) errors

### @retry_with_backoff Decorator

Automatically retries functions with exponential backoff:

```python
from src.data_ingestion.rate_limiter import retry_with_backoff

@retry_with_backoff(max_retries=3)
def fetch_data():
    # This will be retried up to 3 times if it raises an exception
    return api_call()

# With custom callback
def on_retry(attempt, exception):
    logger.warning(f"Retry {attempt}: {exception}")

@retry_with_backoff(
    max_retries=5,
    on_retry=on_retry
)
def fetch_with_logging():
    return api_call()
```

---

## Data Schema

All adapters return OHLCV data in the standardized format:

```python
import pandas as pd
import numpy as np

# Example return value from any adapter.fetch_ohlcv()
df = pd.DataFrame({
    'open': [100.0, 101.0, ...],
    'high': [102.0, 103.0, ...],
    'low': [99.5, 100.5, ...],
    'close': [101.5, 102.5, ...],
    'volume': [1000000, 1100000, ...]
})

# Index properties
df.index = pd.DatetimeIndex(
    [...],
    name='timestamp',
    tz='UTC'  # All timestamps MUST be UTC
)

# Data types
assert df['open'].dtype == 'float64'
assert df['high'].dtype == 'float64'
assert df['low'].dtype == 'float64'
assert df['close'].dtype == 'float64'
assert df['volume'].dtype == 'int64'
```

---

## Configuration Files

Each adapter has a corresponding YAML configuration template:

### `config/provider_templates/ib_config.yaml`
```yaml
provider:
  type: "interactive_brokers"

connection:
  host: "127.0.0.1"
  port: 7497
  client_id: 1
  account_id: "DU123456"  # Required

rate_limiting:
  requests_per_second: 5
  burst_size: 3
```

### `config/provider_templates/polygon_config.yaml`
```yaml
provider:
  type: "polygon"

connection:
  base_url: "https://api.polygon.io"
  timeout_seconds: 30

authentication:
  method: "api_key"
  api_key: "${POLYGON_API_KEY}"  # Set via environment

rate_limiting:
  requests_per_second: 4
  burst_size: 3
```

### `config/provider_templates/databento_config.yaml`
```yaml
provider:
  type: "databento"

authentication:
  method: "api_key"
  api_key: "${DATABENTO_API_KEY}"
  client: "${DATABENTO_CLIENT}"

dataset:
  default: "GLBX"  # CME futures
  enabled:
    - "GLBX"
    - "XNAS"
    - "XNYS"

rate_limiting:
  requests_per_second: 10
  burst_size: 5
```

---

## Testing

Comprehensive unit tests are provided in `tests/test_adapters.py`:

```bash
# Run all adapter tests
pytest tests/test_adapters.py -v

# Run tests for a specific adapter
pytest tests/test_adapters.py::TestIBDataProvider -v

# Run tests with logging
pytest tests/test_adapters.py -v --log-cli-level=DEBUG
```

### Test Coverage

- **Initialization**: Default and custom parameters
- **Authentication**: Success, failure, idempotency
- **Data Fetching**: Success, unsupported symbols, invalid date ranges
- **Data Quality**: Schema validation, data types, value constraints
- **Symbol Listing**: Success, error handling
- **Rate Limiting**: Request tracking, backoff calculation
- **Integration**: Schema consistency across adapters

---

## Design Patterns

### 1. Stub Implementation Pattern

Each adapter currently implements a **stub pattern**:
- Implements all required interface methods
- Returns synthetic data matching the schema
- Includes TODO comments for actual API integration
- Configuration and credential structures documented

This allows:
- Testing downstream code without real API connections
- Clear implementation roadmap
- Easy completion once a vendor is selected

### 2. Context Manager Pattern

All adapters support context manager syntax for automatic resource cleanup:

```python
# Automatic authenticate() and disconnect()
with IBDataProvider(account_id="DU123456") as provider:
    df = provider.fetch_ohlcv(...)
# disconnect() called automatically
```

### 3. Rate Limiting Pattern

Integrates rate limiting into fetch operations:

```python
def fetch_ohlcv(self, ...):
    # Apply rate limiting before API call
    self.rate_limiter.wait_if_needed()
    # ... API call ...
```

---

## Migration Path: Stub to Production

When a vendor is selected (e.g., Interactive Brokers):

1. **Install SDK**: `pip install ib_insync`
2. **Update Credentials**: Replace stub initialization with real API client
3. **Implement Authentication**: Real TWS/Gateway connection validation
4. **Add API Calls**: Replace `_generate_synthetic_data()` with actual API calls
5. **Add Pagination**: Handle large date ranges with pagination
6. **Add Error Handling**: Map vendor-specific errors to standard exceptions
7. **Run Tests**: Verify tests pass with real data
8. **Deploy**: Update credentials in deployment configuration

### Example: IBDataProvider Completion

```python
# From stub:
def authenticate(self) -> None:
    logger.warning("Currently using stub with synthetic data.")
    self._authenticated = True

# To production:
def authenticate(self) -> None:
    import ib_insync
    try:
        self._ib_client = ib_insync.IB()
        self._ib_client.connect(self.host, self.port, self.client_id)
        account_info = self._ib_client.accountSummary(self.account_id)
        self._authenticated = True
    except Exception as e:
        raise AuthenticationError(f"Failed to connect: {e}")
```

---

## Troubleshooting

### Common Issues

**Issue**: `ConfigurationError: api_key is required`
- **Solution**: Ensure required credentials are provided in constructor

**Issue**: `AuthenticationError: Not authenticated. Call authenticate() first.`
- **Solution**: Call `provider.authenticate()` before `fetch_ohlcv()`

**Issue**: `DataNotAvailableError: Symbol X not supported`
- **Solution**: Call `provider.get_available_symbols()` to see supported symbols

**Issue**: Synthetic data doesn't match my expectations
- **Solution**: These are stub implementations. Real data will be provided by the actual API once implementation is complete.

---

## Dependencies

### Required
- `pandas >= 1.0`
- `numpy >= 1.19`

### Optional (for production)
- `ib_insync` - Interactive Brokers API
- `requests` or `polygon-sdk` - Polygon.io API
- `databento` - Databento API

---

## Future Enhancements

1. **Streaming Support**: Add WebSocket/streaming methods for real-time data
2. **Symbol Search**: Implement symbol search across all adapters
3. **Contract Details**: Add method to fetch contract specifications
4. **Options Data**: Support options chains and implied volatility
5. **Tick Data**: Support raw tick data in addition to OHLCV
6. **Caching**: Add optional caching layer for frequently accessed data
7. **Validation**: Enhance data validation with market-specific rules
8. **Metrics**: Add performance metrics (latency, requests/sec, errors)

---

## References

- [Data Provider Interface Specification](data_provider_interface.md)
- [Interactive Brokers TWS API](https://interactivebrokers.com/api)
- [Polygon.io API Documentation](https://polygon.io/docs)
- [Databento API Documentation](https://databento.com/docs)
