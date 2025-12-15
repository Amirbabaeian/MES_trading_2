# MES Trading System 2

A comprehensive hybrid versioning data storage system for trading data with Parquet-based I/O, schema validation, and metadata management.

## Project Structure

```
.
├── README.md                          # This file
├── docs/                              # Documentation
│   ├── data_storage_structure.md      # Directory structure and naming conventions
│   └── versioning_semantics.md        # Versioning strategy documentation
├── schema/                            # Schema definitions
│   └── metadata_schema.json          # Metadata JSON schema specification
├── src/                               # Source code
│   └── data_io/                       # Data I/O module
│       ├── __init__.py               # Public API exports
│       ├── parquet_utils.py          # Core Parquet I/O functions
│       ├── schemas.py                # Schema definitions and registry
│       ├── metadata.py               # Metadata management utilities
│       ├── errors.py                 # Custom exception classes
│       └── README.md                 # Data I/O module documentation
└── tests/                             # Test suite
    └── test_parquet_io.py            # Comprehensive I/O tests

```

## Core Features

### 1. Hybrid Data Versioning

- **Raw Layer**: Date-based immutable snapshots (`/raw/{asset}/{YYYY-MM-DD}/`)
- **Cleaned Layer**: Semantic versioning (`/cleaned/v{X}/{asset}/`)
- **Features Layer**: Independent feature sets with versioning (`/features/v{Y}/{feature_set}/`)

### 2. Parquet I/O with Schema Validation

- Read/write Parquet files with pyarrow/pandas backend
- Three schema enforcement modes: STRICT, COERCE, IGNORE
- Automatic schema inference and validation
- Detailed error reporting for schema mismatches

### 3. Compression Support

- Default snappy compression (~50% reduction)
- Optional gzip (60-70% reduction)
- Optional zstd (65-75% reduction)
- Uncompressed mode for staging

### 4. Metadata Management

- Embed custom key-value pairs in Parquet files
- Track data lineage and dependencies
- Schema information and data quality metrics
- Version information and changelog support

### 5. Partitioning Strategies

- Date-based partitioning for time-series data
- Asset-based partitioning for multi-asset datasets
- Efficient filtering and selective reading

### 6. Comprehensive Error Handling

- `SchemaMismatchError`: Detailed schema mismatch reporting
- `MissingRequiredColumnError`: Required column detection
- `DataTypeValidationError`: Type mismatch and coercion options
- `CorruptedFileError`: File corruption detection

## Quick Start

### Basic Usage

```python
from src.data_io import read_raw_data, write_cleaned_data, OHLCV_SCHEMA
import pandas as pd

# Read raw data
df = read_raw_data('raw/MES/2024-01-15/ohlcv.parquet')

# Clean the data
df_cleaned = df.dropna()

# Write cleaned data with schema validation
write_cleaned_data(df_cleaned, 'cleaned/v1/MES/ohlcv.parquet', schema=OHLCV_SCHEMA)
```

### Schema Validation

```python
from src.data_io import read_parquet_with_schema, write_parquet_validated, OHLCV_SCHEMA

# Write with strict schema validation
write_parquet_validated(df, 'data.parquet', schema=OHLCV_SCHEMA)

# Read with validation
df = read_parquet_with_schema('data.parquet', schema=OHLCV_SCHEMA)
```

### Compression

```python
# Write with different compression codecs
write_parquet_validated(df, 'data.parquet', compression='snappy')   # Default
write_parquet_validated(df, 'data.parquet', compression='gzip')     # Better compression
write_parquet_validated(df, 'data.parquet', compression='zstd')     # Best compression
```

### Metadata Management

```python
from src.data_io import MetadataManager

# Create metadata
metadata = MetadataManager.create_metadata(
    layer='cleaned',
    asset='MES',
    record_count=1000,
    file_size_bytes=1024*1024,
)

# Add quality metrics
MetadataManager.add_quality_metrics(
    metadata,
    null_counts={'volume': 0},
    quality_flags=[],
    completeness_percentage=100.0,
)

# Write metadata file
MetadataManager.write_metadata_file(metadata, 'metadata.json')
```

## Predefined Schemas

- **OHLCV_SCHEMA**: Open, High, Low, Close, Volume candlestick data
- **PRICE_VOLATILITY_SCHEMA**: Volatility indicators (30d, 60d, ratio)
- **MOMENTUM_INDICATORS_SCHEMA**: RSI, MACD, and momentum indicators
- **VOLUME_PROFILE_SCHEMA**: Volume distribution metrics

## API Reference

### Core Functions

- `read_parquet_with_schema()`: Read Parquet with schema validation
- `write_parquet_validated()`: Write Parquet with schema validation
- `read_partitioned_dataset()`: Read partitioned Parquet datasets
- `get_parquet_metadata()`: Extract metadata from Parquet files
- `get_parquet_schema()`: Get schema information from files

### Layer-Specific Wrappers

- `read_raw_data()`: Read raw layer data
- `write_cleaned_data()`: Write cleaned layer data
- `read_features()`: Read features layer data
- `write_features()`: Write features layer data

### Schema Functions

- `get_schema()`: Retrieve schema from registry
- `register_schema()`: Register custom schema

### Metadata Functions

- `MetadataManager.create_metadata()`: Create metadata dictionary
- `MetadataManager.write_metadata_file()`: Write metadata JSON
- `MetadataManager.read_metadata_file()`: Read metadata JSON
- `MetadataManager.add_file_info()`: Add file information
- `MetadataManager.add_quality_metrics()`: Add quality metrics
- `MetadataManager.add_dependencies()`: Add dependency information

## Documentation

- [Data Storage Structure](docs/data_storage_structure.md): Directory organization and naming conventions
- [Versioning Semantics](docs/versioning_semantics.md): Versioning strategy and best practices
- [Data I/O Module](src/data_io/README.md): Comprehensive module documentation
- [Metadata Schema](schema/metadata_schema.json): JSON Schema specification

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/test_parquet_io.py -v

# Run specific test class
pytest tests/test_parquet_io.py::TestSchemaValidation -v

# Run with coverage
pytest tests/test_parquet_io.py --cov=src.data_io --cov-report=html
```

Test coverage includes:

- ✅ Basic read/write operations
- ✅ Schema validation (strict, coerce, ignore modes)
- ✅ Compression support (snappy, gzip, zstd, uncompressed)
- ✅ Partitioned datasets
- ✅ Metadata management
- ✅ Layer-specific wrappers
- ✅ Error handling and edge cases
- ✅ Data type coercion

## Architecture

### Module Organization

```
src/data_io/
├── parquet_utils.py    # Core I/O with pyarrow/pandas
│   ├── SchemaValidator  # Schema validation logic
│   ├── read_parquet_with_schema()
│   ├── write_parquet_validated()
│   └── Compression utilities
│
├── schemas.py           # Schema definitions
│   ├── DataSchema        # Schema class
│   ├── ColumnSchema      # Column definition
│   ├── OHLCV_SCHEMA
│   ├── PRICE_VOLATILITY_SCHEMA
│   └── SCHEMA_REGISTRY
│
├── metadata.py          # Metadata management
│   └── MetadataManager   # Metadata operations
│
└── errors.py            # Custom exceptions
    ├── SchemaMismatchError
    ├── MissingRequiredColumnError
    ├── DataTypeValidationError
    └── ...
```

### Data Flow

```
Raw Data Input
     ↓
read_parquet_with_schema() [Schema: OHLCV]
     ↓
Data Validation & Coercion
     ↓
DataFrame Processing (cleaning, transformation)
     ↓
write_parquet_validated() [Schema: OHLCV, Compression: snappy]
     ↓
Metadata Management
     ↓
Cleaned Data Output → Features Computation → Feature Layer Output
```

## Performance Characteristics

### Compression Performance

| Codec | Compression | Speed | Use Case |
|-------|------------|-------|----------|
| snappy | ~50% | Fast | Default, frequent access |
| gzip | ~60-70% | Slow | Archive, long-term storage |
| zstd | ~65-75% | Good | Large datasets |
| uncompressed | 0% | Instant | Temporary/staging |

### Scaling Characteristics

- **Small datasets** (<100MB): Single file, snappy compression
- **Medium datasets** (100MB-1GB): Partitioned by date, snappy compression
- **Large datasets** (>1GB): Partitioned by date+asset, zstd compression

### Memory Usage

- Read entire dataset: ~3-5x uncompressed size
- Partitioned reads: ~3-5x partition size
- Streaming not currently supported

## Dependencies

- Python >= 3.8
- pandas >= 1.0
- pyarrow >= 1.0
- pytest (for testing)

## Implementation Notes

### Schema Validation Strategy

1. **STRICT Mode**: Enforces exact schema match
   - All required columns must be present
   - Data types must match (with coercion for timestamps/numerics)
   - Extra columns trigger warnings

2. **COERCE Mode**: Attempts to fit data to schema
   - Missing required columns still raise error
   - Attempts automatic type conversion
   - Missing optional columns are ignored

3. **IGNORE Mode**: Minimal validation
   - Logs warnings only
   - No type coercion
   - Useful for trusted data sources

### Compression Selection

- **Default (snappy)**: Recommended for most use cases
  - Good balance of compression and speed
  - ~50% size reduction for financial data
  - Fast read/write operations

- **For archive**: Use gzip
  - Best compression (60-70%)
  - Slower read/write
  - Good for long-term storage

- **For large datasets**: Use zstd
  - Excellent compression (65-75%)
  - Good speed
  - Better than gzip overall

## Future Enhancements

Potential additions for future versions:

- [ ] Streaming I/O for very large datasets
- [ ] Delta Lake integration for ACID transactions
- [ ] Iceberg format support for time-travel queries
- [ ] S3/GCS/Azure Blob Storage backends
- [ ] Query pushdown and column pruning optimization
- [ ] Incremental writing and append operations
- [ ] Data validation rules and constraints
- [ ] Automatic schema evolution

## Contributing

When adding new features:

1. Add corresponding schema to `schemas.py`
2. Add tests to `tests/test_parquet_io.py`
3. Update `src/data_io/README.md` with examples
4. Ensure all tests pass

## License

[License information to be added]

## Contact

[Contact information to be added]
