# Data I/O Module: Parquet File Utilities

This module provides comprehensive utilities for reading and writing Parquet files with schema validation, compression support, metadata management, and partitioning capabilities.

## Features

- **Parquet I/O**: Read and write Parquet files using pandas/pyarrow backend
- **Schema Validation**: Validate data against predefined schemas in three modes:
  - **STRICT**: Fail on any schema mismatch
  - **COERCE**: Attempt to coerce data types
  - **IGNORE**: Warn but don't fail
- **Compression Support**: snappy (default), gzip, zstd, uncompressed
- **Metadata Management**: Embed and extract custom metadata, track data lineage
- **Partitioning**: Support for date-based and asset-based partitioned datasets
- **Error Handling**: Detailed error reporting for schema mismatches and I/O failures
- **Logging**: Comprehensive logging for all operations

## Module Structure

```
data_io/
├── __init__.py          # Public API exports
├── parquet_utils.py     # Core Parquet I/O functions
├── schemas.py           # Schema definitions and registry
├── metadata.py          # Metadata management utilities
├── errors.py            # Custom exception classes
└── README.md            # This file
```

## Quick Start

### Basic Read/Write

```python
import pandas as pd
from src.data_io import read_parquet_with_schema, write_parquet_validated

# Write data
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100),
    'value': range(100)
})
write_parquet_validated(df, 'data.parquet')

# Read data
df = read_parquet_with_schema('data.parquet')
```

### Schema Validation

```python
from src.data_io import OHLCV_SCHEMA, write_parquet_validated

# Write with schema validation
write_parquet_validated(df, 'ohlcv.parquet', schema=OHLCV_SCHEMA)

# Read with validation
df = read_parquet_with_schema('ohlcv.parquet', schema=OHLCV_SCHEMA)
```

### Compression

```python
# Write with compression
write_parquet_validated(df, 'data.parquet', compression='snappy')
write_parquet_validated(df, 'data.parquet', compression='gzip')
write_parquet_validated(df, 'data.parquet', compression='zstd')
write_parquet_validated(df, 'data.parquet', compression='uncompressed')
```

### Layer-Specific Wrappers

```python
from src.data_io import read_raw_data, write_cleaned_data, read_features

# Raw data
df = read_raw_data('raw/MES/2024-01-15/ohlcv.parquet')

# Cleaned data
write_cleaned_data(df, 'cleaned/v1/MES/ohlcv.parquet', schema=OHLCV_SCHEMA)

# Features
df_features = read_features('features/v1/price_volatility/MES.parquet')
```

### Metadata Management

```python
from src.data_io import MetadataManager

# Create metadata
metadata = MetadataManager.create_metadata(
    layer='raw',
    asset='MES',
    record_count=1000,
    file_size_bytes=1024*1024,
    timestamp_range=('2024-01-01T00:00:00Z', '2024-01-31T23:59:59Z'),
)

# Add file info
MetadataManager.add_file_info(metadata, 'ohlcv.parquet', 1024*1024)

# Write metadata
MetadataManager.write_metadata_file(metadata, 'metadata.json')

# Read metadata
metadata = MetadataManager.read_metadata_file('metadata.json')
```

### Partitioned Datasets

```python
from src.data_io import write_parquet_validated, read_partitioned_dataset

# Write partitioned by date
df['date'] = df['timestamp'].dt.date
write_parquet_validated(df, 'dataset/', partition_cols=['date'])

# Read entire partitioned dataset
df = read_partitioned_dataset('dataset/')

# Read with filters
df = read_partitioned_dataset(
    'dataset/',
    filters=[('date', '==', '2024-01-15')]
)
```

## API Reference

### Core Functions

#### `read_parquet_with_schema(file_path, schema=None, schema_mode=STRICT, columns=None, filters=None)`

Read a Parquet file with optional schema validation.

**Parameters:**
- `file_path`: Path to Parquet file
- `schema`: Optional DataSchema for validation
- `schema_mode`: SchemaEnforcementMode (STRICT, COERCE, IGNORE)
- `columns`: List of columns to read (optional)
- `filters`: Partition filters for partitioned datasets

**Returns:** pandas DataFrame

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `CorruptedFileError`: If file is corrupted
- `SchemaMismatchError`: If schema validation fails (strict mode)

#### `write_parquet_validated(df, file_path, schema=None, schema_mode=STRICT, compression='snappy', metadata=None, partition_cols=None)`

Write a DataFrame to Parquet with optional schema validation.

**Parameters:**
- `df`: DataFrame to write
- `file_path`: Output file path
- `schema`: Optional DataSchema for validation
- `schema_mode`: SchemaEnforcementMode (STRICT, COERCE, IGNORE)
- `compression`: Compression codec (snappy, gzip, zstd, uncompressed)
- `metadata`: Custom metadata key-value pairs
- `partition_cols`: Columns to partition by

**Raises:**
- `SchemaMismatchError`: If schema validation fails
- `CompressionError`: If compression codec is invalid
- `ParquetIOError`: If write fails

#### `read_partitioned_dataset(dataset_path, schema=None, schema_mode=STRICT, columns=None, filters=None)`

Read a partitioned Parquet dataset.

**Parameters:**
- `dataset_path`: Path to root of partitioned dataset
- `schema`: Optional DataSchema for validation
- `schema_mode`: SchemaEnforcementMode
- `columns`: Columns to read (optional)
- `filters`: Partition filters

**Returns:** pandas DataFrame

**Raises:**
- `PartitioningError`: If dataset is not properly partitioned

#### `get_parquet_metadata(file_path)`

Extract metadata from a Parquet file.

**Parameters:**
- `file_path`: Path to Parquet file

**Returns:** Dictionary with metadata (num_rows, num_columns, schema, etc.)

#### `get_parquet_schema(file_path)`

Get schema information from a Parquet file.

**Parameters:**
- `file_path`: Path to Parquet file

**Returns:** Dictionary mapping column names to data types

### Layer-Specific Wrappers

#### `read_raw_data(file_path, schema=None, columns=None)`

Read raw data from Parquet. Convenience wrapper with STRICT validation.

#### `write_cleaned_data(df, file_path, schema=None, compression='snappy', metadata=None)`

Write cleaned data to Parquet with STRICT validation and snappy compression.

#### `read_features(file_path, schema=None, columns=None)`

Read feature data from Parquet with STRICT validation.

#### `write_features(df, file_path, schema=None, compression='snappy', metadata=None)`

Write feature data to Parquet with STRICT validation.

### Schema Functions

#### `get_schema(schema_name)`

Retrieve a schema from the registry.

**Parameters:**
- `schema_name`: Schema name (ohlcv, price_volatility, momentum_indicators, volume_profile)

**Returns:** DataSchema object

**Raises:**
- `KeyError`: If schema not found in registry

#### `register_schema(schema)`

Register a new schema in the registry.

**Parameters:**
- `schema`: DataSchema object to register

### MetadataManager

#### `MetadataManager.create_metadata(...)`

Create a metadata dictionary with all required fields.

**Parameters:**
- `layer`: Data layer (raw, cleaned, features)
- `asset`: Asset identifier (MES, ES, VIX)
- `record_count`: Number of records
- `file_size_bytes`: Total size in bytes
- `timestamp_range`: Optional tuple of (start, end) timestamps
- `custom_metadata`: Optional custom key-value pairs
- `version_info`: Optional version information
- `dependencies`: Optional dependency information

**Returns:** Metadata dictionary

#### `MetadataManager.write_metadata_file(metadata, output_path)`

Write metadata to a JSON file.

#### `MetadataManager.read_metadata_file(metadata_path)`

Read metadata from a JSON file.

#### `MetadataManager.add_file_info(metadata, filename, size_bytes, ...)`

Add file information to metadata.

#### `MetadataManager.add_quality_metrics(metadata, null_counts=None, quality_flags=None, completeness_percentage=None)`

Add data quality metrics to metadata.

#### `MetadataManager.add_dependencies(metadata, raw_sources=None, cleaned_version=None, processing_script=None, processing_script_hash=None)`

Add dependency information to metadata.

## Predefined Schemas

### OHLCV_SCHEMA

Open, High, Low, Close, Volume candlestick data schema.

**Columns:**
- `timestamp` (timestamp[ns], required)
- `open` (double, required)
- `high` (double, required)
- `low` (double, required)
- `close` (double, required)
- `volume` (int64, required)

### PRICE_VOLATILITY_SCHEMA

Price volatility indicators schema.

**Columns:**
- `timestamp` (timestamp[ns], required)
- `volatility_30d` (double, optional)
- `volatility_60d` (double, optional)
- `volatility_ratio` (double, optional)
- `price_range_pct` (double, optional)

### MOMENTUM_INDICATORS_SCHEMA

Momentum and trend indicators schema.

**Columns:**
- `timestamp` (timestamp[ns], required)
- `rsi` (double, optional)
- `macd` (double, optional)
- `macd_signal` (double, optional)
- `momentum` (double, optional)

### VOLUME_PROFILE_SCHEMA

Volume profile and distribution metrics schema.

**Columns:**
- `timestamp` (timestamp[ns], required)
- `volume_profile_high` (double, optional)
- `volume_at_price_high` (int64, optional)
- `volume_concentration` (double, optional)

## Error Classes

### ParquetIOError

Base exception for all I/O errors.

### SchemaMismatchError

Raised when Parquet file schema doesn't match expected schema.

**Properties:**
- `expected_columns`: List of expected column names
- `actual_columns`: List of actual column names
- `get_missing_columns()`: Columns expected but missing
- `get_extra_columns()`: Columns not expected

### MissingRequiredColumnError

Raised when required columns are missing from data.

**Properties:**
- `missing_columns`: List of missing column names

### DataTypeValidationError

Raised when data types don't match schema.

**Properties:**
- `column_name`: Column with type mismatch
- `expected_type`: Expected data type
- `actual_type`: Actual data type

### CompressionError

Raised when compression configuration is invalid.

**Properties:**
- `compression`: Invalid compression codec
- `valid_options`: List of valid compression options

### CorruptedFileError

Raised when Parquet file is corrupted or unreadable.

### PartitioningError

Raised when partitioned dataset operations fail.

### MetadataError

Raised when metadata operations fail.

## Examples

### Example 1: Raw Data Pipeline

```python
from src.data_io import read_raw_data, write_cleaned_data, OHLCV_SCHEMA, MetadataManager
from pathlib import Path

# Read raw data
raw_df = read_raw_data('raw/MES/2024-01-15/ohlcv.parquet')

# Clean and validate
cleaned_df = raw_df.dropna()

# Write cleaned data with metadata
output_path = Path('cleaned/v1/MES/ohlcv.parquet')
write_cleaned_data(cleaned_df, output_path, schema=OHLCV_SCHEMA)

# Create and write metadata
metadata = MetadataManager.create_metadata(
    layer='cleaned',
    asset='MES',
    record_count=len(cleaned_df),
    file_size_bytes=output_path.stat().st_size,
)
MetadataManager.write_metadata_file(metadata, output_path.parent / 'metadata.json')
```

### Example 2: Feature Engineering

```python
from src.data_io import read_cleaned_data, write_features, PRICE_VOLATILITY_SCHEMA
import pandas as pd

# Read cleaned data
df = read_cleaned_data('cleaned/v1/MES/ohlcv.parquet', schema=OHLCV_SCHEMA)

# Compute features
df['volatility_30d'] = df['close'].rolling(30).std()
df['volatility_60d'] = df['close'].rolling(60).std()
df['volatility_ratio'] = df['volatility_30d'] / df['volatility_60d']
df['price_range_pct'] = (df['high'] - df['low']) / df['close'] * 100

# Keep only feature columns
features = df[['timestamp', 'volatility_30d', 'volatility_60d', 'volatility_ratio', 'price_range_pct']]

# Write features with schema validation
write_features(
    features,
    'features/v1/price_volatility/MES.parquet',
    schema=PRICE_VOLATILITY_SCHEMA,
)
```

### Example 3: Large Dataset with Partitioning

```python
from src.data_io import write_parquet_validated, read_partitioned_dataset

# Write partitioned by date and asset
df['date'] = df['timestamp'].dt.date
df['asset'] = 'MES'

write_parquet_validated(
    df,
    'features/v1/price_volatility/',
    partition_cols=['date', 'asset'],
    compression='zstd',
)

# Read specific partition
df_2024_01_15 = read_partitioned_dataset(
    'features/v1/price_volatility/',
    filters=[('date', '==', '2024-01-15')],
)
```

## Compression Performance

Compression codec recommendations:

- **snappy** (default): Good balance of compression and speed. ~50% reduction for typical financial data.
- **gzip**: Better compression (~60-70%) but slower. Use for archive/storage.
- **zstd**: Excellent compression (~65-75%) with good speed. Use for large datasets.
- **uncompressed**: Use only for temporary/staging data.

## Logging

The module logs all major operations:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# All I/O operations will log information about file paths, row counts, and schemas
```

## Testing

Run the test suite:

```bash
pytest tests/test_parquet_io.py -v
```

Run specific test class:

```bash
pytest tests/test_parquet_io.py::TestSchemaValidation -v
```

Run with coverage:

```bash
pytest tests/test_parquet_io.py --cov=src.data_io --cov-report=html
```

## Performance Considerations

1. **Schema Validation**: Validation adds overhead. Use `SchemaEnforcementMode.IGNORE` for trusted data sources.

2. **Compression**: Default snappy provides good balance. For millions of rows:
   - Use zstd for long-term storage
   - Use snappy for frequent access
   - Use uncompressed for temporary staging

3. **Partitioning**: Use for datasets > 1GB:
   - Partition by date for time-series data
   - Partition by asset for multi-asset datasets
   - Use small partition sizes (< 100MB per partition)

4. **Memory**: Partitioned reads load all matching partitions into memory. Use filters to limit data.

## Dependencies

- pandas
- pyarrow >= 1.0
- Python >= 3.8
