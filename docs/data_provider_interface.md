# Data Provider Interface Documentation

## Overview

The Data Provider Interface is a vendor-agnostic abstraction for fetching market data (OHLCV candlesticks) from any data source. It enables:

- **Vendor independence**: Swap data providers without downstream code changes
- **Standardized schema**: All providers return identical DataFrame structure
- **Error handling**: Consistent exception hierarchy for provider errors
- **Testing**: Mock provider for development without real APIs
- **Extensibility**: Easy to implement custom providers

## Architecture

```
┌─────────────────────────────────────────────┐
│        Downstream Code (Features,           │
│        Training, Analysis)                  │
└──────────────────┬──────────────────────────┘
                   │ uses interface
                   ▼
        ┌──────────────────────┐
        │  DataProvider (ABC)  │  Abstract base class
        │  - authenticate()    │  Defines contract
        │  - fetch_ohlcv()     │
        │  - get_available_... │
        └──────────────────────┘
              ▲         ▲        ▲         ▲
              │         │        │         │
    ┌─────────┴──┐  ┌───┴────┐ ┌─┴──────┐ └──────────┐
    │             │  │        │  │        │            │
 [AlphaVantage] [IB] [Polygon] [Yahoo] [Crypto]   [MockProvider]
 concrete impl  concrete    concrete  concrete    for testing
```

## DataProvider Interface

### Abstract Base Class

```python
from src.data_ingestion import DataProvider
from datetime import datetime
import pandas as pd

class MyCustomProvider(DataProvider):
    """Implement your data provider here."""
    
    def authenticate(self) -> None:
        """Establish connection/session."""
        pass
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D"
    ) -> pd.DataFrame:
        """Fetch OHLCV data."""
        pass
    
    def get_available_symbols(self) -> list[str]:
        """Return list of supported symbols."""
        pass
```

### Required Methods

#### 1. `authenticate() -> None`

**Purpose**: Establish connection and validate credentials.

**Behavior**:
- Validates API credentials/configuration
- Establishes network connection if needed
- Initializes session objects
- Sets `_authenticated = True` on success

**Raises**:
- `AuthenticationError`: Invalid credentials
- `ConnectionError`: Cannot reach provider

**Example**:
```python
provider = AlphaVantageProvider(api_key='YOUR_KEY')
provider.authenticate()  # Validates key, connects
assert provider.is_authenticated
```

---

#### 2. `fetch_ohlcv(symbol, start_date, end_date, timeframe='1D') -> DataFrame`

**Purpose**: Core method for retrieving market data.

**Parameters**:
- `symbol` (str): Asset identifier
  - Examples: `"ES"`, `"MES"`, `"VIX"`, `"AAPL"`, `"BTC/USD"`
  
- `start_date` (datetime): Inclusive range start
  - Must be timezone-aware or assumed UTC
  - Example: `datetime(2024, 1, 1, tzinfo=pytz.UTC)`
  
- `end_date` (datetime): Inclusive range end
  - Must be >= start_date
  
- `timeframe` (str): Candlestick period
  - Options: `"1M"`, `"5M"`, `"15M"`, `"30M"`, `"1H"`, `"1D"`, `"1W"`, `"1MO"`
  - Default: `"1D"` (daily)

**Returns**: DataFrame with standardized schema:

```
                        open     high      low    close    volume
timestamp (UTC)
2024-01-01 00:00:00  5000.0  5020.0  4995.0  5010.0  1500000
2024-01-02 00:00:00  5010.0  5030.0  5005.0  5025.0  1700000
2024-01-03 00:00:00  5025.0  5050.0  5020.0  5040.0  1600000
```

**Schema Details**:
- **Index**: `DatetimeIndex` with name `'timestamp'`, timezone UTC
- **Columns**: `['timestamp', 'open', 'high', 'low', 'close', 'volume']`
- **Data types**:
  - `open, high, low, close`: `float64`
  - `volume`: `int64`
  - `timestamp`: `datetime64[ns, UTC]`
- **Constraints**:
  - `high >= max(open, close)`
  - `low <= min(open, close)`
  - No NaN/null values in core columns
  - Sorted ascending by timestamp
  - No duplicate timestamps
  - Continuous (no artificial gaps except market closures)

**Raises**:
- `AuthenticationError`: Not authenticated
- `ValidationError`: Invalid parameters
- `DataNotAvailableError`: Symbol/timeframe/date range not available
- `RateLimitError`: Rate limit exceeded
- `TimeoutError`: Request times out
- `ConnectionError`: Cannot connect to provider

**Vendor-Specific Quirks** (adapters must handle):
- **Market closures**: No data on weekends/holidays
- **Pre/After hours**: Some providers include, others don't
- **Splits/Dividends**: Stocks may be adjusted; futures/crypto are not
- **Delisted symbols**: Old tickers may not have recent data
- **Futures expiration**: Contract rolls on specific dates
- **Data quality**: Some providers have gaps or late arrivals

**Example**:
```python
provider = AlphaVantageProvider(api_key='KEY')
provider.authenticate()

df = provider.fetch_ohlcv(
    symbol='MES',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    timeframe='1D'
)

# Verify schema
assert isinstance(df.index, pd.DatetimeIndex)
assert df.index.name == 'timestamp'
assert df.index.tz.zone == 'UTC'
assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
assert df.shape[0] > 0  # Has data
assert df['high'].min() >= df['low'].max() is False  # proper ordering
```

---

#### 3. `get_available_symbols() -> List[str]`

**Purpose**: Retrieve supported symbols.

**Returns**: List of symbol strings.

**Example Output**:
```python
['ES', 'MES', 'NQ', 'VIX', 'AAPL', 'MSFT', 'GOOGL', ...]
```

**Raises**:
- `AuthenticationError`: Authentication required but not done
- `ConnectionError`: Cannot reach provider
- `TimeoutError`: Request times out

**Implementation Notes**:
- Symbol format is provider-specific; normalize in adapters
- Some providers have thousands of symbols; consider caching
- List may be filtered (e.g., only US equities)
- Should be called after `authenticate()`

**Example**:
```python
provider = InteractiveBrokersProvider()
provider.authenticate()
symbols = provider.get_available_symbols()
assert 'ES' in symbols
assert 'AAPL' in symbols
```

---

### Optional Methods

#### `handle_pagination(symbol, start_date, end_date, timeframe, page_size=1000) -> List[DataFrame]`

**Purpose**: Handle large data requests with provider limits.

**Behavior**:
- Breaks large date ranges into chunks
- Fetches each chunk separately
- Returns list of DataFrames
- Default implementation provided (can override)

**Returns**: List of DataFrames, each following standard schema.

**Example**:
```python
provider = AlphaVantageProvider(...)
provider.authenticate()

# Fetch 10 years of data in pages
pages = provider.handle_pagination(
    symbol='ES',
    start_date=datetime(2015, 1, 1),
    end_date=datetime(2024, 12, 31),
    timeframe='1D'
)

# Combine pages
df = pd.concat(pages, ignore_index=False).sort_index()
```

---

#### `get_contract_details(symbol) -> Dict[str, Any]`

**Purpose**: Retrieve metadata about a symbol.

**Returns**: Dictionary with possible keys:
- `name`: Human-readable name
- `exchange`: Trading exchange
- `contract_type`: "STOCK", "FUTURE", "OPTION", "INDEX", etc.
- `underlying`: Underlying asset (for derivatives)
- `multiplier`: Contract multiplier
- `min_tick`: Minimum price movement
- `expiration`: Expiration date
- `active`: Is contract currently trading

**Example**:
```python
provider = InteractiveBrokersProvider()
provider.authenticate()

details = provider.get_contract_details('MES')
print(details)
# {
#     'name': 'E-mini S&P 500 Dec 2024',
#     'exchange': 'CME',
#     'contract_type': 'FUTURE',
#     'multiplier': 50,
#     'min_tick': 0.25,
#     'expiration': datetime(2024, 12, 20)
# }
```

---

## Exception Hierarchy

```
DataProviderError (base)
├── AuthenticationError          # Credentials invalid
├── DataNotAvailableError        # Symbol/timeframe/date not available
├── ValidationError              # Invalid parameters
├── SchemaError                  # Data doesn't match expected schema
├── ConnectionError              # Cannot connect to provider
├── RateLimitError               # Rate limit exceeded
├── PaginationError              # Pagination handling failed
└── TimeoutError                 # Request times out
```

### Common Exception Scenarios

**AuthenticationError**:
```python
try:
    provider = AlphaVantageProvider(api_key='INVALID')
    provider.authenticate()  # Raises AuthenticationError
except AuthenticationError as e:
    print(f"Auth failed: {e}")
    print(f"Provider: {e.provider}")
```

**DataNotAvailableError**:
```python
try:
    df = provider.fetch_ohlcv(
        symbol='INVALID_SYMBOL',
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31)
    )
except DataNotAvailableError as e:
    print(f"Data not available: {e}")
    print(f"Symbol: {e.symbol}")
```

**RateLimitError**:
```python
try:
    for symbol in symbols:
        df = provider.fetch_ohlcv(symbol, start, end)
except RateLimitError as e:
    print(f"Rate limit: {e}")
    if e.retry_after:
        time.sleep(e.retry_after)
```

---

## Timezone Handling Contract

### Key Principle
**Providers return UTC. Consumers convert.**

### Details

1. **Provider Returns**:
   - All timestamps in UTC
   - Example: `2024-01-15 14:30:00+00:00` (2:30 PM UTC)

2. **Consumer Converts** (in downstream code):
   - Convert to desired timezone
   - Example: Convert 14:30 UTC → 9:30 EST (trading open)
   - Use: `df.index = df.index.tz_convert('America/New_York')`

3. **Why**:
   - UTC is universal; unambiguous
   - Providers may serve multiple timezones
   - Cleaning layer handles timezone conversions
   - Prevents confusion with daylight saving time

### Example

```python
# Provider returns UTC
df = provider.fetch_ohlcv('ES', start, end)
print(df.index.tz)  # UTC

# Consumer converts
df_ny = df.copy()
df_ny.index = df_ny.index.tz_convert('America/New_York')
print(df_ny.index.tz)  # America/New_York
```

---

## Usage Patterns

### Basic Usage

```python
from src.data_ingestion import DataProvider
from src.data_ingestion.adapters import AlphaVantageProvider
from datetime import datetime

# Initialize
provider = AlphaVantageProvider(api_key='YOUR_KEY')

# Authenticate
provider.authenticate()

# Fetch data
df = provider.fetch_ohlcv(
    symbol='MES',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    timeframe='1D'
)

# Use data
print(df.head())
print(df.shape)
```

### Provider Swapping

```python
# Swap providers without changing client code
def get_data(provider: DataProvider, symbol: str):
    """Works with any DataProvider implementation."""
    provider.authenticate()
    df = provider.fetch_ohlcv(
        symbol=symbol,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31)
    )
    return df

# Use different providers
alpha_provider = AlphaVantageProvider(api_key='KEY1')
data1 = get_data(alpha_provider, 'AAPL')

polygon_provider = PolygonProvider(api_key='KEY2')
data2 = get_data(polygon_provider, 'AAPL')

# Same code, different sources!
```

### Error Handling

```python
from src.data_ingestion import (
    DataProvider,
    AuthenticationError,
    DataNotAvailableError,
    RateLimitError
)

try:
    provider = SomeProvider()
    provider.authenticate()
    df = provider.fetch_ohlcv('INVALID', datetime.now(), datetime.now())
except AuthenticationError:
    print("Auth failed")
except DataNotAvailableError as e:
    print(f"No data for {e.symbol}")
except RateLimitError as e:
    if e.retry_after:
        print(f"Wait {e.retry_after} seconds")
```

### Pagination for Large Ranges

```python
# Automatic pagination
pages = provider.handle_pagination(
    symbol='ES',
    start_date=datetime(2010, 1, 1),
    end_date=datetime(2024, 12, 31),
    timeframe='1D'
)

# Combine results
df = pd.concat(pages, ignore_index=False).sort_index()
print(f"Total bars: {len(df)}")
```

### Testing with MockProvider

```python
from src.data_ingestion import MockProvider

# Use mock for testing
provider = MockProvider(seed=42)
provider.authenticate()

df = provider.fetch_ohlcv(
    symbol='ES',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    timeframe='1D'
)

# Reproducible data (same seed = same data)
assert len(df) == 252  # Trading days in 2024
```

---

## Implementing a Custom Provider

### Template

```python
from src.data_ingestion import DataProvider
from src.data_ingestion.exceptions import (
    AuthenticationError,
    DataNotAvailableError,
    ValidationError
)
from datetime import datetime
import pandas as pd

class MyDataProvider(DataProvider):
    """Custom provider implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(name="MyProvider")
        self.api_key = api_key
        self.session = None  # Your connection object
    
    def authenticate(self) -> None:
        """Validate credentials and establish connection."""
        if not self.api_key:
            raise AuthenticationError("API key required")
        
        # Validate key with provider
        try:
            # Your validation logic here
            self.session = self._create_session()
            self._authenticated = True
        except Exception as e:
            raise AuthenticationError(str(e), provider=self.name)
    
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1D"
    ) -> pd.DataFrame:
        """Fetch OHLCV data."""
        if not self._authenticated:
            raise AuthenticationError("Must call authenticate() first")
        
        # Validate inputs
        if start_date > end_date:
            raise ValidationError("start_date must be <= end_date")
        
        # Fetch from provider
        try:
            data = self._fetch_from_api(symbol, start_date, end_date, timeframe)
        except KeyError:
            raise DataNotAvailableError(
                f"Symbol {symbol} not available",
                symbol=symbol
            )
        
        # Convert to standard schema
        df = self._normalize_schema(data)
        
        return df
    
    def get_available_symbols(self) -> list:
        """Return supported symbols."""
        return self._load_symbol_list()
    
    # Private methods
    def _create_session(self):
        """Create provider session."""
        pass
    
    def _fetch_from_api(self, symbol, start, end, timeframe):
        """Fetch raw data from provider API."""
        pass
    
    def _normalize_schema(self, data):
        """Convert to standard schema."""
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
        return df[['open', 'high', 'low', 'close', 'volume']]
```

---

## Testing Considerations

### Unit Testing with MockProvider

```python
import pytest
from src.data_ingestion import MockProvider, DataProvider

def test_fetch_ohlcv_schema():
    """Test that returned data has correct schema."""
    provider = MockProvider()
    provider.authenticate()
    
    df = provider.fetch_ohlcv(
        symbol='ES',
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        timeframe='1D'
    )
    
    # Verify schema
    assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert df.index.name == 'timestamp'
    assert df.index.tz.zone == 'UTC'
    
    # Verify data integrity
    assert (df['high'] >= df['low']).all()
    assert (df['high'] >= df['open']).all()
    assert (df['high'] >= df['close']).all()
    assert df.index.is_monotonic_increasing
```

### Integration Testing with Real Provider

```python
@pytest.mark.integration
def test_alphavantage_provider():
    """Test real provider (requires API key)."""
    provider = AlphaVantageProvider(api_key=os.getenv('ALPHA_KEY'))
    provider.authenticate()
    
    df = provider.fetch_ohlcv('AAPL', datetime(2024, 1, 1), datetime(2024, 1, 31))
    
    assert len(df) > 0
    assert not df.isnull().any().any()
```

### Error Testing with FailingMockProvider

```python
from src.data_ingestion import FailingMockProvider, AuthenticationError

def test_error_handling():
    """Test error handling."""
    provider = FailingMockProvider(failure_mode='auth')
    
    with pytest.raises(AuthenticationError):
        provider.authenticate()
```

---

## Best Practices

### 1. Always Call authenticate() First
```python
provider = AlphaVantageProvider(api_key='KEY')
provider.authenticate()  # Required!
df = provider.fetch_ohlcv(...)
```

### 2. Check Symbol Availability
```python
provider.authenticate()
available = provider.get_available_symbols()
if symbol not in available:
    print(f"Symbol {symbol} not available")
```

### 3. Handle Pagination
```python
# For large date ranges
if (end_date - start_date).days > 365:
    pages = provider.handle_pagination(...)
    df = pd.concat(pages).sort_index()
```

### 4. Catch Provider-Specific Errors
```python
try:
    df = provider.fetch_ohlcv(...)
except (AuthenticationError, DataNotAvailableError) as e:
    log.error(f"Provider error: {e}")
    raise
```

### 5. Use Context Managers (if implementing)
```python
# Provider can implement cleanup
with AlphaVantageProvider(api_key='KEY') as provider:
    provider.authenticate()
    df = provider.fetch_ohlcv(...)
    # Auto-cleanup on exit
```

### 6. Cache Symbol Lists
```python
# Don't call get_available_symbols() repeatedly
symbols = provider.get_available_symbols()  # Once
for symbol in symbols:
    if has_data(symbol):
        df = provider.fetch_ohlcv(symbol, ...)
```

---

## FAQ

**Q: Why does every provider return UTC timestamps?**
A: UTC is universal and unambiguous. Converting to specific timezones happens in the cleaning layer, not at the provider level. This keeps providers simple and prevents timezone confusion.

**Q: Can I implement pagination differently?**
A: Yes! The default `handle_pagination()` is provided for convenience, but override it for vendor-specific pagination logic.

**Q: What if a provider doesn't have all symbols?**
A: Call `get_available_symbols()` to check. If a symbol isn't available, `fetch_ohlcv()` raises `DataNotAvailableError`.

**Q: How do I handle rate limits?**
A: Catch `RateLimitError`; it includes `retry_after` (seconds to wait). Implement backoff/retry logic in your adapter.

**Q: Can I fetch intraday data?**
A: Yes! Use timeframe parameters: "1M", "5M", "15M", "30M", "1H". Provider support varies; document in adapter.

**Q: What about options and other derivatives?**
A: The interface is for OHLCV data. Options and derivatives may need specialized providers with different schemas.

---

## Summary

The DataProvider interface is:
- ✅ **Vendor-agnostic**: No vendor details in base class
- ✅ **Standardized**: All providers return identical schema
- ✅ **Well-documented**: Each method has clear contracts
- ✅ **Extensible**: Easy to implement custom providers
- ✅ **Testable**: Mock provider for development
- ✅ **Error-safe**: Comprehensive exception hierarchy

Use it to decouple your code from any single data vendor!
