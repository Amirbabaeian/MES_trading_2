"""
Data I/O module for Parquet file operations with schema validation.

This module provides utilities for reading, writing, and managing Parquet files
with support for schema validation, compression, metadata management, and
partitioning.
"""

# Core I/O functions
from .parquet_utils import (
    read_parquet_with_schema,
    write_parquet_validated,
    read_partitioned_dataset,
    read_raw_data,
    write_cleaned_data,
    read_features,
    write_features,
    get_parquet_metadata,
    get_parquet_schema,
    SchemaValidator,
    validate_compression,
)

# Schema definitions
from .schemas import (
    DataSchema,
    ColumnSchema,
    SchemaEnforcementMode,
    OHLCV_SCHEMA,
    PRICE_VOLATILITY_SCHEMA,
    MOMENTUM_INDICATORS_SCHEMA,
    VOLUME_PROFILE_SCHEMA,
    SCHEMA_REGISTRY,
    get_schema,
    register_schema,
)

# Data validation
from .validation import (
    DataValidator,
    Violation,
    ValidationResult,
    validate_ohlcv_data,
    validate_multiple_datasets,
    MissingBarsValidator,
    MissingBarsReport,
    GapInfo,
)

# Trading calendar utilities
from .calendar_utils import (
    TradingCalendar,
    MarketHours,
    MarketType,
    get_calendar,
    create_custom_calendar,
)

# Metadata management
from .metadata import MetadataManager

# Error classes
from .errors import (
    ParquetIOError,
    SchemaMismatchError,
    MissingRequiredColumnError,
    DataTypeValidationError,
    CorruptedFileError,
    PartitioningError,
    CompressionError,
    MetadataError,
)

__all__ = [
    # I/O functions
    "read_parquet_with_schema",
    "write_parquet_validated",
    "read_partitioned_dataset",
    "read_raw_data",
    "write_cleaned_data",
    "read_features",
    "write_features",
    "get_parquet_metadata",
    "get_parquet_schema",
    "SchemaValidator",
    "validate_compression",
    # Schema
    "DataSchema",
    "ColumnSchema",
    "SchemaEnforcementMode",
    "OHLCV_SCHEMA",
    "PRICE_VOLATILITY_SCHEMA",
    "MOMENTUM_INDICATORS_SCHEMA",
    "VOLUME_PROFILE_SCHEMA",
    "SCHEMA_REGISTRY",
    "get_schema",
    "register_schema",
    # Validation
    "DataValidator",
    "Violation",
    "ValidationResult",
    "validate_ohlcv_data",
    "validate_multiple_datasets",
    "MissingBarsValidator",
    "MissingBarsReport",
    "GapInfo",
    # Trading Calendar
    "TradingCalendar",
    "MarketHours",
    "MarketType",
    "get_calendar",
    "create_custom_calendar",
    # Metadata
    "MetadataManager",
    # Errors
    "ParquetIOError",
    "SchemaMismatchError",
    "MissingRequiredColumnError",
    "DataTypeValidationError",
    "CorruptedFileError",
    "PartitioningError",
    "CompressionError",
    "MetadataError",
]
