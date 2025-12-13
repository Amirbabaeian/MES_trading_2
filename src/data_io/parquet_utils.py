"""
Parquet file I/O utilities with schema validation and compression support.

Provides:
- Low-level read/write functions with pyarrow/pandas backend
- Schema validation on read/write operations
- Compression support (snappy, gzip, zstd)
- Metadata embedding and extraction
- Partitioning strategies (by date, by asset)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schemas import DataSchema, SchemaEnforcementMode
from .errors import (
    ParquetIOError,
    SchemaMismatchError,
    MissingRequiredColumnError,
    DataTypeValidationError,
    CorruptedFileError,
    PartitioningError,
    CompressionError,
)

logger = logging.getLogger(__name__)

# Valid compression codecs
VALID_COMPRESSION_CODECS = {"snappy", "gzip", "zstd", "uncompressed"}
DEFAULT_COMPRESSION = "snappy"


# ============================================================================
# Schema Validation
# ============================================================================

class SchemaValidator:
    """Validates data against DataSchema definitions."""
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        schema: DataSchema,
        mode: SchemaEnforcementMode = SchemaEnforcementMode.STRICT,
    ) -> pd.DataFrame:
        """
        Validate a DataFrame against a schema.
        
        Args:
            df: DataFrame to validate
            schema: DataSchema to validate against
            mode: Enforcement mode (STRICT, COERCE, IGNORE)
        
        Returns:
            Validated (and possibly coerced) DataFrame
        
        Raises:
            SchemaMismatchError: If schema doesn't match (strict mode)
            MissingRequiredColumnError: If required columns are missing
            DataTypeValidationError: If data types don't match
        """
        required_cols = schema.get_required_columns()
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            raise MissingRequiredColumnError(list(missing_cols))
        
        # Check for unexpected columns
        extra_cols = set(df.columns) - set(schema.get_column_names())
        if extra_cols and mode == SchemaEnforcementMode.STRICT:
            logger.warning(f"Extra columns found: {extra_cols}")
        
        # Validate column names
        expected_cols = set(schema.get_column_names())
        actual_cols = set(df.columns)
        
        if not expected_cols.issubset(actual_cols) and mode == SchemaEnforcementMode.STRICT:
            missing = expected_cols - actual_cols
            raise SchemaMismatchError(
                f"Schema mismatch: missing columns {missing}",
                expected_columns=list(expected_cols),
                actual_columns=list(actual_cols)
            )
        
        # Validate and coerce types
        for col_schema in schema.columns:
            col_name = col_schema.name
            
            if col_name not in df.columns:
                if not col_schema.nullable:
                    raise MissingRequiredColumnError([col_name])
                continue
            
            # Type validation and coercion
            df = _coerce_column_type(df, col_schema, mode)
        
        return df
    
    @staticmethod
    def validate_arrow_table(
        table: pa.Table,
        schema: DataSchema,
        mode: SchemaEnforcementMode = SchemaEnforcementMode.STRICT,
    ) -> pa.Table:
        """
        Validate an Arrow Table against a schema.
        
        Args:
            table: Arrow Table to validate
            schema: DataSchema to validate against
            mode: Enforcement mode (STRICT, COERCE, IGNORE)
        
        Returns:
            Validated Arrow Table
        
        Raises:
            SchemaMismatchError: If schema doesn't match (strict mode)
            MissingRequiredColumnError: If required columns are missing
        """
        # Convert to DataFrame for validation
        df = table.to_pandas()
        validated_df = SchemaValidator.validate_dataframe(df, schema, mode)
        return pa.Table.from_pandas(validated_df)


def _coerce_column_type(
    df: pd.DataFrame,
    col_schema,
    mode: SchemaEnforcementMode,
) -> pd.DataFrame:
    """Coerce a column to the expected type."""
    col_name = col_schema.name
    expected_type = col_schema.data_type
    
    # Map Parquet types to pandas types
    type_mapping = {
        "int32": "int32",
        "int64": "int64",
        "float": "float32",
        "double": "float64",
        "string": "object",
        "timestamp[ns]": "datetime64[ns]",
        "timestamp[us]": "datetime64[us]",
        "bool": "bool",
    }
    
    if mode == SchemaEnforcementMode.IGNORE:
        return df
    
    target_type = type_mapping.get(expected_type, expected_type)
    
    try:
        if expected_type.startswith("timestamp"):
            df[col_name] = pd.to_datetime(df[col_name])
        else:
            df[col_name] = df[col_name].astype(target_type)
    except (ValueError, TypeError) as e:
        if mode == SchemaEnforcementMode.STRICT:
            raise DataTypeValidationError(
                col_name,
                expected_type,
                str(df[col_name].dtype)
            )
        else:
            logger.warning(f"Failed to coerce {col_name} to {expected_type}: {e}")
    
    return df


# ============================================================================
# Compression Utilities
# ============================================================================

def validate_compression(compression: str) -> str:
    """
    Validate compression codec.
    
    Args:
        compression: Compression codec name
    
    Returns:
        Valid compression codec
    
    Raises:
        CompressionError: If compression codec is invalid
    """
    if compression not in VALID_COMPRESSION_CODECS:
        raise CompressionError(compression, list(VALID_COMPRESSION_CODECS))
    return compression


# ============================================================================
# Core Read/Write Functions
# ============================================================================

def read_parquet_with_schema(
    file_path: Union[str, Path],
    schema: Optional[DataSchema] = None,
    schema_mode: SchemaEnforcementMode = SchemaEnforcementMode.STRICT,
    columns: Optional[List[str]] = None,
    filters: Optional[List] = None,
) -> pd.DataFrame:
    """
    Read a Parquet file with optional schema validation.
    
    Args:
        file_path: Path to Parquet file
        schema: DataSchema to validate against (optional)
        schema_mode: Schema enforcement mode
        columns: List of columns to read (optional)
        filters: Partition filters for partitioned datasets
    
    Returns:
        pandas DataFrame
    
    Raises:
        FileNotFoundError: If file doesn't exist
        CorruptedFileError: If file is corrupted
        SchemaMismatchError: If schema validation fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    try:
        logger.info(f"Reading Parquet file: {file_path}")
        df = pd.read_parquet(file_path, columns=columns)
        logger.info(f"Read {len(df)} rows from {file_path}")
        
        if schema:
            df = SchemaValidator.validate_dataframe(df, schema, schema_mode)
            logger.info(f"Validated schema for {file_path}")
        
        return df
    
    except pa.ArrowException as e:
        raise CorruptedFileError(f"Failed to read Parquet file {file_path}: {e}")
    except Exception as e:
        raise ParquetIOError(f"Error reading Parquet file {file_path}: {e}")


def write_parquet_validated(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    schema: Optional[DataSchema] = None,
    schema_mode: SchemaEnforcementMode = SchemaEnforcementMode.STRICT,
    compression: str = DEFAULT_COMPRESSION,
    compression_level: Optional[int] = None,
    metadata: Optional[Dict[str, str]] = None,
    index: bool = False,
    partition_cols: Optional[List[str]] = None,
) -> None:
    """
    Write a DataFrame to Parquet with optional schema validation.
    
    Args:
        df: DataFrame to write
        file_path: Output file path
        schema: DataSchema to validate against (optional)
        schema_mode: Schema enforcement mode
        compression: Compression codec (snappy, gzip, zstd, uncompressed)
        compression_level: Compression level (codec-specific)
        metadata: Custom metadata key-value pairs
        index: Whether to write DataFrame index
        partition_cols: Columns to partition by (creates partitioned dataset)
    
    Raises:
        SchemaMismatchError: If schema validation fails
        CompressionError: If compression codec is invalid
        ParquetIOError: If write fails
    """
    file_path = Path(file_path)
    
    # Validate compression
    compression = validate_compression(compression)
    
    # Validate schema if provided
    if schema:
        df = SchemaValidator.validate_dataframe(df, schema, schema_mode)
    
    # Prepare metadata
    parquet_metadata = {}
    if metadata:
        parquet_metadata.update(metadata)
    
    try:
        logger.info(f"Writing Parquet file: {file_path} (compression: {compression})")
        
        # Create parent directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if partition_cols:
            # Write partitioned dataset
            table = pa.Table.from_pandas(df, preserve_index=index)
            pq.write_to_dataset(
                table,
                root_path=str(file_path),
                partition_cols=partition_cols,
                compression=compression,
                compression_level=compression_level,
            )
            logger.info(f"Wrote partitioned dataset to {file_path}")
        else:
            # Write single file
            df.to_parquet(
                file_path,
                compression=compression,
                index=index,
                engine="pyarrow",
            )
            logger.info(f"Wrote {len(df)} rows to {file_path}")
    
    except Exception as e:
        raise ParquetIOError(f"Failed to write Parquet file {file_path}: {e}")


def read_partitioned_dataset(
    dataset_path: Union[str, Path],
    schema: Optional[DataSchema] = None,
    schema_mode: SchemaEnforcementMode = SchemaEnforcementMode.STRICT,
    columns: Optional[List[str]] = None,
    filters: Optional[List] = None,
) -> pd.DataFrame:
    """
    Read a partitioned Parquet dataset.
    
    Args:
        dataset_path: Path to root of partitioned dataset
        schema: DataSchema to validate against (optional)
        schema_mode: Schema enforcement mode
        columns: Columns to read (optional)
        filters: Partition filters (e.g., [('year', '==', 2024)])
    
    Returns:
        pandas DataFrame with all partitions combined
    
    Raises:
        PartitioningError: If dataset is not properly partitioned
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    try:
        logger.info(f"Reading partitioned dataset: {dataset_path}")
        
        # Use pyarrow's parquet.read_table for partitioned datasets
        table = pq.read_table(
            str(dataset_path),
            columns=columns,
            filters=filters,
        )
        
        df = table.to_pandas()
        logger.info(f"Read {len(df)} rows from partitioned dataset")
        
        if schema:
            df = SchemaValidator.validate_dataframe(df, schema, schema_mode)
        
        return df
    
    except Exception as e:
        raise PartitioningError(f"Failed to read partitioned dataset {dataset_path}: {e}")


def get_parquet_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract metadata from a Parquet file.
    
    Args:
        file_path: Path to Parquet file
    
    Returns:
        Dictionary with metadata information
    
    Raises:
        FileNotFoundError: If file doesn't exist
        CorruptedFileError: If file is corrupted
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    try:
        parquet_file = pq.ParquetFile(file_path)
        
        metadata = {
            "num_rows": parquet_file.metadata.num_rows,
            "num_columns": parquet_file.metadata.num_columns,
            "schema": str(parquet_file.schema_arrow),
            "columns": parquet_file.schema_arrow.names,
            "created_by": parquet_file.metadata.created_by,
            "num_row_groups": parquet_file.metadata.num_row_groups,
        }
        
        # Try to extract custom metadata
        if parquet_file.metadata.metadata:
            custom_metadata = {}
            for key, value in parquet_file.metadata.metadata.items():
                try:
                    custom_metadata[key.decode() if isinstance(key, bytes) else key] = \
                        value.decode() if isinstance(value, bytes) else value
                except Exception:
                    pass
            if custom_metadata:
                metadata["custom_metadata"] = custom_metadata
        
        return metadata
    
    except pa.ArrowException as e:
        raise CorruptedFileError(f"Failed to read Parquet metadata {file_path}: {e}")
    except Exception as e:
        raise ParquetIOError(f"Error reading Parquet metadata {file_path}: {e}")


def get_parquet_schema(file_path: Union[str, Path]) -> Dict[str, str]:
    """
    Get schema information from a Parquet file.
    
    Args:
        file_path: Path to Parquet file
    
    Returns:
        Dictionary mapping column names to data types
    
    Raises:
        FileNotFoundError: If file doesn't exist
        CorruptedFileError: If file is corrupted
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    try:
        parquet_file = pq.ParquetFile(file_path)
        schema_dict = {}
        
        for i in range(parquet_file.schema_arrow.num_fields):
            field = parquet_file.schema_arrow.field(i)
            schema_dict[field.name] = str(field.type)
        
        return schema_dict
    
    except Exception as e:
        raise ParquetIOError(f"Error reading Parquet schema {file_path}: {e}")


# ============================================================================
# Layer-Specific Wrapper Functions
# ============================================================================

def read_raw_data(
    file_path: Union[str, Path],
    schema: Optional[DataSchema] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read raw data from Parquet with optional schema validation.
    
    Convenience wrapper for reading raw layer data.
    
    Args:
        file_path: Path to Parquet file
        schema: DataSchema to validate against (optional)
        columns: Columns to read (optional)
    
    Returns:
        pandas DataFrame
    """
    return read_parquet_with_schema(
        file_path,
        schema=schema,
        schema_mode=SchemaEnforcementMode.STRICT,
        columns=columns,
    )


def write_cleaned_data(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    schema: Optional[DataSchema] = None,
    compression: str = DEFAULT_COMPRESSION,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write cleaned data to Parquet with schema validation.
    
    Convenience wrapper for writing cleaned layer data.
    
    Args:
        df: DataFrame to write
        file_path: Output file path
        schema: DataSchema to validate against (optional)
        compression: Compression codec (default: snappy)
        metadata: Custom metadata key-value pairs
    """
    write_parquet_validated(
        df,
        file_path,
        schema=schema,
        schema_mode=SchemaEnforcementMode.STRICT,
        compression=compression,
        metadata=metadata,
    )


def read_features(
    file_path: Union[str, Path],
    schema: Optional[DataSchema] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Read feature data from Parquet.
    
    Convenience wrapper for reading features layer data.
    
    Args:
        file_path: Path to Parquet file
        schema: DataSchema to validate against (optional)
        columns: Columns to read (optional)
    
    Returns:
        pandas DataFrame
    """
    return read_parquet_with_schema(
        file_path,
        schema=schema,
        schema_mode=SchemaEnforcementMode.STRICT,
        columns=columns,
    )


def write_features(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    schema: Optional[DataSchema] = None,
    compression: str = DEFAULT_COMPRESSION,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write feature data to Parquet.
    
    Convenience wrapper for writing features layer data.
    
    Args:
        df: DataFrame to write
        file_path: Output file path
        schema: DataSchema to validate against (optional)
        compression: Compression codec (default: snappy)
        metadata: Custom metadata key-value pairs
    """
    write_parquet_validated(
        df,
        file_path,
        schema=schema,
        schema_mode=SchemaEnforcementMode.STRICT,
        compression=compression,
        metadata=metadata,
    )
