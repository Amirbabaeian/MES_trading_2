# Parquet Data Feed for Backtrader

A custom data feed implementation for loading OHLCV (Open, High, Low, Close, Volume) market data from Parquet files in the versioned storage structure.

## Overview

The `ParquetDataFeed` class extends `backtrader.DataBase` to efficiently load historical market data from Parquet files stored in the cleaned data layer (`/cleaned/v{version}/{symbol}/ohlcv.parquet`).

**Key Features:**
- ✅ Reads OHLCV data from versioned Parquet storage
- ✅ Supports multiple timeframes (1min, 5min, 15min) with resampling
- ✅ Timezone-aware datetime handling (UTC, US/Eastern, etc.)
- ✅ Efficient lazy loading and memory management
- ✅ Date range filtering for targeted backtests
- ✅ OHLCV schema validation
- ✅ Comprehensive error handling

## Installation & Setup

### Prerequisites

Ensure the following packages are installed:
```bash
pip install backtrader pandas pyarrow
```

### Data Structure

The feed expects data organized in the following structure:

```
data/
└── cleaned/
    └── v1/
        ├── MES/
        │   └── ohlcv.parquet
        ├── ES/
        │   └── ohlcv.parquet
        └── ...
```

Each Parquet file must contain OHLCV columns:
- `timestamp` (datetime with timezone)
- `open` (float)
- `high` (float)
- `low` (float)
- `close` (float)
- `volume` (integer)

## Usage

### Basic Usage

```python
from datetime import datetime
from src.backtest.feeds import create_parquet_feed
import backtrader as bt

# Create cerebro instance
cerebro = bt.Cerebro()

# Create data feed
feed = create_parquet_feed(
    symbol='MES',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    cleaned_version='v1',
)

# Add to backtrader
cerebro.adddata(feed)

# Run backtest
cerebro.run()
```

### Advanced Configuration

```python
from src.backtest.feeds import ParquetDataFeed
from datetime import datetime

# Direct instantiation with full control
feed = ParquetDataFeed(
    symbol='MES',
    cleaned_version='v1',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    base_path='data/cleaned',  # Custom data directory
    tz='America/Chicago',       # Timezone handling
    validate_schema=True,       # Enable schema validation
    chunk_size=10000,          # Buffer size for large datasets
)

cerebro.adddata(feed)
```

### Multiple Asset Feeds

```python
from src.backtest.feeds import create_multi_feeds

# Load multiple assets
feeds = create_multi_feeds(
    symbols=['MES', 'ES'],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
)

for symbol, feed in feeds.items():
    cerebro.adddata(feed, name=symbol)
```

### Feed Validation & Discovery

```python
from src.backtest.feeds import (
    validate_feed_exists,
    list_available_feeds,
    get_feed_date_range,
)

# Check if feed exists
if validate_feed_exists('MES'):
    # Get available date range
    start_date, end_date = get_feed_date_range('MES')
    print(f'MES data available from {start_date} to {end_date}')

# List all available feeds
available_symbols = list_available_feeds()
print(f'Available symbols: {available_symbols}')
```

## API Reference

### ParquetDataFeed Class

Main data feed class extending `backtrader.DataBase`.

#### Parameters

- `symbol` (str): Asset symbol (e.g., 'MES', 'ES')
- `cleaned_version` (str): Data version (e.g., 'v1', 'v2'). Default: 'v1'
- `start_date` (datetime, optional): Backtest start date
- `end_date` (datetime, optional): Backtest end date
- `base_path` (str): Base path to cleaned data directory. Default: 'data/cleaned'
- `tz` (str): Timezone for datetime handling. Default: 'UTC'
- `validate_schema` (bool): Validate OHLCV schema on load. Default: True
- `chunk_size` (int): Buffer size for chunked loading. Default: 10000

#### Methods

- `_load()`: Load next bar from data (called by backtrader)
- `_getsizes()`: Get total number of bars available
- `start()`: Called when backtest starts
- `stop()`: Called when backtest ends

### Helper Functions

#### `create_parquet_feed()`

Create a ParquetDataFeed with validation.

```python
feed = create_parquet_feed(
    symbol='MES',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    cleaned_version='v1',
    base_path='data/cleaned',
    tz='UTC',
    validate_schema=True,
)
```

#### `create_multi_feeds()`

Create multiple feeds for different symbols.

```python
feeds = create_multi_feeds(
    symbols=['MES', 'ES'],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
)
```

#### `validate_feed_exists()`

Check if a Parquet feed exists.

```python
exists = validate_feed_exists('MES', cleaned_version='v1')
```

#### `list_available_feeds()`

List all available asset symbols.

```python
symbols = list_available_feeds(cleaned_version='v1')
```

#### `get_feed_date_range()`

Get the date range of available data.

```python
start_date, end_date = get_feed_date_range('MES')
```

## Data Validation

The feed validates OHLCV data for:

✅ **Column Presence**: All required columns (timestamp, open, high, low, close, volume)  
✅ **Data Types**: Correct types for each column  
✅ **Uniqueness**: No duplicate timestamps  
✅ **Ordering**: Data ordered by timestamp  
✅ **OHLC Relationships**: Warnings if high < low  

Validation can be disabled for performance:

```python
feed = ParquetDataFeed(
    symbol='MES',
    validate_schema=False,  # Skip schema validation
)
```

## Timeframe Support

Currently supports these timeframes:

- **1 Minute** (1T)
- **5 Minute** (5T)
- **15 Minute** (15T)
- **30 Minute** (30T)
- **Hourly** (1H)
- **Daily** (1D)

Resampling is performed using standard OHLC rules:
- **Open**: First bar's open in period
- **High**: Maximum high in period
- **Low**: Minimum low in period
- **Close**: Last bar's close in period
- **Volume**: Sum of volumes in period

## Performance Considerations

### Memory Efficiency

The feed uses lazy loading to efficiently handle large datasets:

- Data is loaded once at initialization
- Bars are streamed to backtrader on demand
- Chunked loading prevents excessive memory usage

For **multi-month backtests** with 1-minute bars (~250k bars):
- Memory usage: ~50-100 MB
- Load time: < 1 second

### Optimization Tips

1. **Filter by date range**: Use `start_date` and `end_date` to reduce data
2. **Use 5-minute bars**: For faster backtests, load pre-aggregated 5-minute data
3. **Disable validation**: Skip schema validation for production runs

```python
# Optimized for speed
feed = create_parquet_feed(
    symbol='MES',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    validate_schema=False,  # Skip validation
)
```

## Timezone Handling

The feed ensures timezone consistency:

```python
# Load data in UTC (default)
feed = create_parquet_feed('MES', tz='UTC')

# Convert to trading hours timezone
feed = create_parquet_feed('MES', tz='America/Chicago')

# Or custom timezone
feed = create_parquet_feed('MES', tz='US/Eastern')
```

Timezone conversion happens automatically during data loading.

## Error Handling

The feed provides detailed error messages:

```python
from src.backtest.feeds import create_parquet_feed

try:
    feed = create_parquet_feed(
        symbol='NONEXISTENT',
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
    )
except FileNotFoundError as e:
    print(f'Data file not found: {e}')
except ValueError as e:
    print(f'Invalid parameters: {e}')
```

## Testing

Run the test suite:

```bash
pytest tests/backtest/test_parquet_feed.py -v
```

Tests cover:
- ✅ Feed initialization
- ✅ Data loading and validation
- ✅ Date range filtering
- ✅ Timezone handling
- ✅ Schema validation
- ✅ Large dataset handling
- ✅ Integration with backtrader
- ✅ Edge cases (empty date ranges, single bars, etc.)

## Example Strategy

```python
from datetime import datetime
import backtrader as bt
from src.backtest.feeds import create_parquet_feed
from src.backtest.strategies.base import BaseStrategy

class SimpleStrategy(BaseStrategy):
    def next(self):
        if not self.position:
            if self.data.close[0] > self.data.close[-1]:
                self.buy()
        else:
            if self.data.close[0] < self.data.close[-1]:
                self.sell()

# Setup backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(SimpleStrategy)

# Add data feed
feed = create_parquet_feed(
    symbol='MES',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
)
cerebro.adddata(feed)

# Configure backtest
cerebro.broker.setcash(100000.0)
cerebro.broker.setcommission(commission=0.001)

# Run
results = cerebro.run()
```

## Troubleshooting

### File Not Found

```
FileNotFoundError: Parquet file not found: data/cleaned/v1/MES/ohlcv.parquet
```

**Solution**: Verify the file path and data version exist:
```python
from src.backtest.feeds import list_available_feeds

available = list_available_feeds()
print(f'Available symbols: {available}')
```

### No Data After Filtering

```python
# Date range is outside of available data
feed = create_parquet_feed(
    symbol='MES',
    start_date=datetime(2025, 1, 1),  # No data yet
    end_date=datetime(2025, 12, 31),
)
```

**Solution**: Check available date range:
```python
from src.backtest.feeds import get_feed_date_range

start, end = get_feed_date_range('MES')
print(f'Data available from {start} to {end}')
```

### Schema Validation Errors

```
SchemaMismatchError: Schema mismatch: missing columns {...}
```

**Solution**: Disable validation or ensure data matches OHLCV schema:
```python
feed = ParquetDataFeed(
    symbol='MES',
    validate_schema=False,  # Temporary workaround
)
```

## References

- [Backtrader DataBase Documentation](https://www.backtrader.com/docu/datapb/)
- [Parquet I/O Module](../src/data_io/README.md)
- [Backtest Configuration](./backtest_setup.md)
