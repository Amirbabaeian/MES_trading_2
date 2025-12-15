# Data Provider Interface Specification

## Overview

The Data Provider interface defines a vendor-agnostic contract for retrieving OHLCV (Open, High, Low, Close, Volume) market data from different sources. This abstraction enables:

- **Vendor independence**: Swap data sources without modifying downstream code
- **Consistent schema**: All providers return data in the same standardized format
- **Clear error handling**: Well-defined exception hierarchy for error conditions
- **Extensibility**: Optional methods for vendor-specific functionality

## Design Principles

### 1. Vendor Agnosticism
The base class (`DataProvider`) contains no vendor-specific details. All concrete implementations are adapters that translate vendor-specific APIs into the standardized interface.

### 2. Standardized Schema
All OHLCV data is returned as a pandas DataFrame with:
- **Columns**: `timestamp` (index), `open`, `high`, `low`, `close`, `volume`
- **Data types**:
  - `timestamp`: `datetime64[ns, UTC]` (DatetimeIndex)
  - `open`, `high`, `low`, `close`: `float64`
  - `volume`: `int64`
- **Index**: `DatetimeIndex` with name `'timestamp'`, timezone-aware (UTC)
- **Ordering**: Chronological (oldest to newest)
- **Gaps**: None; data is continuous within trading hours

### 3. Timezone Contract
**All providers return UTC timestamps.** Timezone conversion to trading hours (e.g., US/Eastern) happens in the data cleaning layer, not in the provider.

This design:
- Eliminates timezone ambiguity at the source
- Centralizes timezone handling (single location for changes)
- Simplifies testing (UTC is unambiguous)
- Matches industry standards (databases typically use UTC)

### 4. Error Clarity
Errors are categorized by type, enabling appropriate error handling in consumers:
- **`AuthenticationError`**: Fix credentials and retry
- **`DataNotAvailableError`**: Check symbol validity or date range
- **`RateLimitError`**: Implement backoff and retry
- **`ValidationError`**: Fix input parameters
- **`ConnectionError`**: Check network connectivity and retry

---

## Core Interface

### Abstract Base Class: `DataProvider`

All concrete implementations must inherit from `DataProvider` and implement the following abstract methods.

#### Method: `authenticate()`

Establish a connection and authenticate with the provider.

```python
def authenticate(self) -> None:
    """
    Establish connection and authenticate with the provider.
    
    Must be called before fetch_ohlcv() or get_available_symbols().
    """
```

**Behavior**:
- Validates credentials (API keys, tokens, etc.)
- Establishes connection (network socket, HTTP session, etc.)
- Initializes session state
- Sets internal flag `_authenticated = True`

**Raises**:
- `AuthenticationError`: Invalid credentials, expired tokens, insufficient permissions
- `ConfigurationError`: Missing required configuration
- `ConnectionError`: Network issues

**Example**:
```python
provider = IQFeedProvider(api_key="...", api_secret="...")
provider.authenticate()  # Validates credentials and connects
```

#### Method: `fetch_ohlcv(symbol, start_date, end_date, timeframe)`

Retrieve OHLCV bars for a symbol in the specified date range and timeframe.

```python
def fetch_ohlcv(
    self,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str,
) -> pd.DataFrame:
    """
    Retrieve OHLCV bars for a symbol.
    
    Returns data in standardized schema (see below).
    """
```

**Parameters**:
- `symbol` (str): Asset symbol, typically uppercase (e.g., "ES", "MES", "VIX")
- `start_date` (datetime): Start of range, inclusive
- `end_date` (datetime): End of range, inclusive
- `timeframe` (str): Aggregation interval:
  - Intraday: `"1m"`, `"5m"`, `"15m"`, `"30m"`, `"60m"`
  - Daily: `"1D"` or `"D"`
  - Weekly: `"1W"` or `"W"`
  - Monthly: `"1M"` or `"M"`

**Returns**:
- `pd.DataFrame` with standardized schema (see below)
- Empty DataFrame with correct schema if no data available

**Raises**:
- `DataNotAvailableError`: Symbol unsupported, timeframe unavailable, no data in range
- `RateLimitError`: API limits exceeded (transient, may retry)
- `AuthenticationError`: Not authenticated
- `ValidationError`: Invalid parameters
- `ConnectionError`: Network issue

**Key behaviors**:
- **Pagination**: Handled internally. If a vendor limits data per request, the implementation must split requests and combine results transparently.
- **Timezone**: All returned timestamps are UTC.
- **Data quality**: No gaps, duplicates, or NaN values.

**Example**:
```python
provider = IQFeedProvider(...)
provider.authenticate()

df = provider.fetch_ohlcv(
    symbol="ES",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 1, 31),
    timeframe="1D"
)

print(df)
#                                open     high      low    close   volume
# timestamp                                                               
# 2023-01-01 17:00:00+00:00  3800.0  3850.0  3790.0  3810.0  1000000
# 2023-01-02 17:00:00+00:00  3810.0  3820.0  3800.0  3815.0  1100000
```

#### Method: `get_available_symbols()`

Retrieve the list of supported symbols.

```python
def get_available_symbols(self) -> List[str]:
    """
    Return the list of supported symbols.
    
    Format is vendor-specific (uppercase). May be cached.
    """
```

**Returns**:
- `List[str]`: Supported symbol names in uppercase

**Raises**:
- `AuthenticationError`: Not authenticated
- `ConnectionError`: Network issue
- `RateLimitError`: API limits exceeded

**Example**:
```python
provider = IQFeedProvider(...)
provider.authenticate()

symbols = provider.get_available_symbols()
print(symbols)  # ["ES", "MES", "NQ", "YM", "VIX", ...]

if "ES" in symbols:
    df = provider.fetch_ohlcv("ES", ...)
```

### Non-Abstract (Optional) Methods

#### Method: `disconnect()`

Close the connection to the provider.

```python
def disconnect(self) -> None:
    """
    Close the connection and clean up resources.
    
    Called automatically when used as a context manager.
    Subclasses may override to clean up (e.g., close WebSockets).
    """
```

This is called automatically by the context manager. Implementations may override to clean up resources.

#### Method: `handle_pagination(request_func, max_records_per_request, total_records, **kwargs)`

Helper method for paginating large requests. Useful for vendors that limit records per API call.

```python
def handle_pagination(
    self,
    request_func,
    max_records_per_request: int,
    total_records: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Paginate through large result sets.
    
    Repeatedly calls request_func with pagination params,
    combines results into a single DataFrame.
    """
```

**Example usage** (in a vendor adapter):
```python
def fetch_ohlcv(self, symbol, start_date, end_date, timeframe):
    def request_page(offset, limit):
        return self.api.get_bars(
            symbol=symbol,
            start=start_date,
            end=end_date,
            timeframe=timeframe,
            offset=offset,
            limit=limit,
        )
    
    return self.handle_pagination(
        request_page,
        max_records_per_request=1000,
    )
```

#### Method: `validate_ohlcv_data(df)`

Validate that a DataFrame conforms to the standardized schema.

```python
def validate_ohlcv_data(self, df: pd.DataFrame) -> None:
    """
    Validate OHLCV data against schema.
    
    Raises ValidationError if invalid.
    """
```

**Validation checks**:
- Correct columns and column order
- Correct dtypes
- DatetimeIndex with UTC timezone
- No negative volumes
- High >= Low, etc.

#### Method: `get_contract_details(symbol)` (Optional)

Retrieve vendor-specific contract metadata (optional).

```python
def get_contract_details(self, symbol: str) -> Dict[str, Any]:
    """
    Return contract details (optional).
    
    Raises NotImplementedError if not supported.
    """
```

**Returns** (example):
```python
{
    "multiplier": 50,
    "tick_size": 0.25,
    "currency": "USD",
    "exchange": "GLOBEX",
    "description": "E-mini S&P 500 Futures"
}
```

---

## Standardized OHLCV Schema

All providers must return data in this exact format:

### DataFrame Structure

```python
# Example: ES daily data
               open     high      low    close   volume
timestamp                                               
2023-01-01  3800.0  3850.0  3790.0  3810.0  1000000
2023-01-02  3810.0  3820.0  3800.0  3815.0  1100000
```

### Columns

| Column    | Data Type | Description |
|-----------|-----------|-------------|
| timestamp | (index)   | UTC timestamp as DatetimeIndex |
| open      | float64   | Opening price |
| high      | float64   | Highest price in bar |
| low       | float64   | Lowest price in bar |
| close     | float64   | Closing price |
| volume    | int64     | Number of shares/contracts traded |

### Data Types (Python/NumPy)

```python
df.dtypes
# open             float64
# high             float64
# low              float64
# close            float64
# volume            int64
# dtype: object

df.index
# DatetimeIndex(['2023-01-01', '2023-01-02', ...], dtype='datetime64[ns, UTC]', name='timestamp', freq=None)
```

### Verification Code

```python
def is_valid_ohlcv(df: pd.DataFrame) -> bool:
    """Check if DataFrame is valid OHLCV."""
    # Check columns
    if set(df.columns) != {"open", "high", "low", "close", "volume"}:
        return False
    
    # Check dtypes
    if df["volume"].dtype != "int64":
        return False
    if any(df[col].dtype != "float64" for col in ["open", "high", "low", "close"]):
        return False
    
    # Check index
    if not isinstance(df.index, pd.DatetimeIndex):
        return False
    if df.index.name != "timestamp":
        return False
    if df.index.tz is None or str(df.index.tz) != "UTC":
        return False
    
    # Check data validity
    if (df["volume"] < 0).any():
        return False
    if (df["high"] < df["low"]).any():
        return False
    
    return True
```

---

## Timezone Handling Contract

### Provider Responsibility
Providers must return **all timestamps in UTC**. This is non-negotiable.

### Consumer Responsibility
Downstream code must convert UTC timestamps to the desired timezone (typically US/Eastern for market hours).

### Rationale
1. **Unambiguous**: UTC has no daylight saving time ambiguity
2. **Standard**: Matches database and industry conventions
3. **Centralized**: Single layer (cleaning) handles all timezone conversions
4. **Testable**: UTC data is easier to test (no clock changes)

### Conversion Example

```python
# Provider returns UTC timestamps
df = provider.fetch_ohlcv("ES", ...)
# df.index.tz == "UTC"

# Consumer converts to market timezone
df_ny = df.copy()
df_ny.index = df_ny.index.tz_convert("US/Eastern")

# Use df_ny for market hour calculations
```

---

## Error Handling

### Exception Hierarchy

```
DataProviderError (base)
├── AuthenticationError
├── DataNotAvailableError
├── RateLimitError
├── PaginationError
├── ConfigurationError
├── ValidationError
└── ConnectionError
```

### Error Handling Patterns

#### Transient Errors (Retry)

These errors may be temporary; retry with backoff:
- `RateLimitError`: API rate limits exceeded
- `ConnectionError`: Network issues

```python
import time

max_retries = 3
for attempt in range(max_retries):
    try:
        df = provider.fetch_ohlcv(...)
        break
    except (RateLimitError, ConnectionError) as e:
        if attempt == max_retries - 1:
            raise
        wait_time = 2 ** attempt  # Exponential backoff
        time.sleep(wait_time)
```

#### Non-Transient Errors (Fix and Retry)

These require human intervention:
- `AuthenticationError`: Fix credentials
- `DataNotAvailableError`: Check symbol or date range
- `ValidationError`: Fix input parameters

```python
try:
    df = provider.fetch_ohlcv(symbol="ES", ...)
except DataNotAvailableError as e:
    print(f"Error: {e}")
    print(f"Available symbols: {provider.get_available_symbols()}")
except ValidationError as e:
    print(f"Invalid input: {e}")
    # Fix parameters and retry manually
```

---

## Usage Examples

### Basic Usage (Context Manager)

```python
from datetime import datetime
from src.data_ingestion.mock_provider import MockDataProvider

# Automatically authenticates on enter, disconnects on exit
with MockDataProvider(seed=42) as provider:
    df = provider.fetch_ohlcv(
        symbol="ES",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        timeframe="1D"
    )
    print(df.head())
```

### Manual Authentication/Disconnection

```python
provider = MockDataProvider(seed=42)
provider.authenticate()

try:
    symbols = provider.get_available_symbols()
    df = provider.fetch_ohlcv("ES", datetime(2023, 1, 1), datetime(2023, 1, 31), "1D")
finally:
    provider.disconnect()
```

### Fetching Multiple Symbols

```python
with MockDataProvider() as provider:
    symbols = ["ES", "NQ", "VIX"]
    data = {}
    
    for symbol in symbols:
        data[symbol] = provider.fetch_ohlcv(
            symbol=symbol,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            timeframe="1D"
        )
    
    print(data["ES"].head())
```

### Error Handling

```python
from src.data_ingestion.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    ValidationError
)

provider = MockDataProvider()

try:
    provider.authenticate()
    df = provider.fetch_ohlcv("INVALID", datetime(2023, 1, 1), datetime(2023, 1, 31), "1D")
except AuthenticationError as e:
    print(f"Auth failed: {e}")
except DataNotAvailableError as e:
    print(f"Symbol not found. Available: {provider.get_available_symbols()}")
except ValidationError as e:
    print(f"Invalid input: {e}")
```

---

## Implementing a New Adapter

To implement a new data provider adapter, follow this template:

```python
from datetime import datetime
from typing import List
import pandas as pd
from src.data_ingestion.base_provider import DataProvider
from src.data_ingestion.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
)

class MyDataProvider(DataProvider):
    """
    Concrete implementation for MyData vendor.
    
    Adapts vendor-specific API to DataProvider interface.
    """
    
    def __init__(self, api_key: str, api_secret: str):
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = None  # Initialize vendor-specific client
    
    def authenticate(self) -> None:
        """Connect and authenticate with vendor API."""
        # Vendor-specific authentication logic
        if not self.api_key:
            raise AuthenticationError("API key required")
        
        # Create session, validate credentials, etc.
        self.session = MyVendorClient(self.api_key, self.api_secret)
        self._authenticated = True
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Fetch and normalize vendor data."""
        if not self._authenticated:
            raise AuthenticationError("Not authenticated")
        
        # Call vendor API
        vendor_data = self.session.get_bars(
            symbol=symbol,
            start=start_date,
            end=end_date,
            interval=self._normalize_timeframe(timeframe),
        )
        
        # Transform to standardized schema
        df = self._normalize_dataframe(vendor_data)
        
        # Validate before returning
        self.validate_ohlcv_data(df)
        
        return df
    
    def get_available_symbols(self) -> List[str]:
        """Fetch list of supported symbols."""
        if not self._authenticated:
            raise AuthenticationError("Not authenticated")
        
        symbols = self.session.list_symbols()
        return sorted(symbols)
    
    def _normalize_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to vendor format."""
        mapping = {
            "1m": "minute",
            "1D": "daily",
            # ... etc
        }
        return mapping.get(timeframe, timeframe)
    
    def _normalize_dataframe(self, vendor_data) -> pd.DataFrame:
        """Convert vendor DataFrame to standardized schema."""
        df = pd.DataFrame(vendor_data)
        
        # Rename columns to standard names
        df.rename(columns={
            "open_price": "open",
            "high_price": "high",
            # ... etc
        }, inplace=True)
        
        # Ensure timestamp is DatetimeIndex with UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        df.index = df.index.tz_localize("UTC")
        
        # Ensure dtypes
        df["volume"] = df["volume"].astype("int64")
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype("float64")
        
        return df[["open", "high", "low", "close", "volume"]]
```

---

## Thread Safety

Implementations should document thread-safety guarantees:

- **`authenticate()`**: Should be thread-safe. May establish a connection pool.
- **`fetch_ohlcv()`**: May not be thread-safe for the same symbol. Different symbols may be safe.
- **Connection pooling**: Recommended for high-concurrency use cases.

Example (thread-safe implementation):

```python
import threading

class ThreadSafeDataProvider(DataProvider):
    """Vendor adapter with thread-safe data fetching."""
    
    def __init__(self, max_workers=5):
        super().__init__()
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
    
    def fetch_ohlcv(self, symbol, start_date, end_date, timeframe):
        # Thread-safe fetch for different symbols
        return self.pool.submit(
            self._fetch_impl,
            symbol,
            start_date,
            end_date,
            timeframe
        ).result()
    
    def _fetch_impl(self, symbol, start_date, end_date, timeframe):
        # Actual implementation
        pass
```

---

## Testing

Use `MockDataProvider` for testing downstream code:

```python
from src.data_ingestion.mock_provider import MockDataProvider
import pytest

@pytest.fixture
def data_provider():
    return MockDataProvider(seed=42)

def test_data_processing(data_provider):
    data_provider.authenticate()
    df = data_provider.fetch_ohlcv("ES", datetime(2023, 1, 1), datetime(2023, 1, 31), "1D")
    
    # Test processing logic
    assert len(df) > 0
    assert df.index.tz.zone == "UTC"
    assert set(df.columns) == {"open", "high", "low", "close", "volume"}
```

---

## FAQ

**Q: Can providers return data in a different timezone?**
A: No. All providers must return UTC timestamps. Timezone conversion happens in downstream code.

**Q: What if a symbol is not supported?**
A: Raise `DataNotAvailableError` with a clear message. Consumers can check `get_available_symbols()`.

**Q: How should pagination be handled?**
A: Transparently, within `fetch_ohlcv()`. Use `handle_pagination()` helper or implement vendor-specific pagination.

**Q: Can I return NaN values?**
A: No. All data must be continuous and valid. Return an empty DataFrame if no data is available.

**Q: Should I cache `get_available_symbols()`?**
A: Yes. The list is typically static. Refresh on `authenticate()` if necessary.

**Q: Are methods thread-safe?**
A: Document in your implementation. Use locking or connection pools for concurrent access.

---

## Summary

The `DataProvider` interface enables vendor-independent data retrieval through:

1. **Consistent schema**: All data in the same format (DataFrame with UTC timestamps)
2. **Clear contracts**: Well-defined methods with specific behavior
3. **Error clarity**: Categorized exceptions for appropriate error handling
4. **Extensibility**: Optional methods for vendor-specific features
5. **Testability**: Mock provider for testing without real data

Implement the abstract methods, document vendor-specific quirks, and follow the interface contract.
