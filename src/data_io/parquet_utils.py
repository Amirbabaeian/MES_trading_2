"""
Core Parquet I/O utilities for the hybrid versioning system.

Provides low-level and high-level APIs for reading and writing Parquet files
with schema validation, compression, metadata management, and partitioning.
"""

from typing import Optional, Dict, Any, Union, List, Literal
from pathlib import Path
import logging
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schemas import SchemaRegistry, get_schema
from .metadata import MetadataManager
from .errors import (
    ParquetIOError,
    SchemaValidationError,
    SchemaMismatchError,
    MissingColumnsError,
    DataTypeError,
    ParquetReadError,
    ParquetWriteError,
    CorruptedFileError,
    CompressionError,
    PartitioningError,
)

logger = logging.getLogger(__name__)

# Type aliases
CompressionType = Literal["snappy", "gzip", "zstd", "none"]
EnforcementMode = Literal["strict", "coerce", "ignore"]


class ParquetIOHandler:
    """
    Low-level handler for Parquet read/write operations.
    
    Provides core functionality for reading and writing Parquet files
    with schema validation and error handling.
    """
    
    SUPPORTED_COMPRESSIONS = {"snappy", "gzip", "zstd", "none"}
    DEFAULT_COMPRESSION = "snappy"
    
    def __init__(self, schema_registry: Optional[SchemaRegistry] = None):
        """
        Initialize Parquet I/O handler.
        
        Args:
            schema_registry: Optional SchemaRegistry instance (uses default if not provided)
        """
        self.schema_registry = schema_registry or SchemaRegistry()
        self.metadata_manager = MetadataManager()
    
    @staticmethod
    def _validate_compression(compression: CompressionType) -> None:
        """
        Validate compression type.
        
        Args:
            compression: Compression codec name
            
        Raises:
            CompressionError: If compression type is not supported
        """
        if compression not in ParquetIOHandler.SUPPORTED_COMPRESSIONS:
            raise CompressionError(
                compression,
                f"Supported compressions: {ParquetIOHandler.SUPPORTED_COMPRESSIONS}"
            )
    
    def read_parquet_with_schema(
        self,
        file_path: Union[str, Path],
        schema: Optional[pa.Schema] = None,
        schema_name: Optional[str] = None,
        enforcement_mode: EnforcementMode = "strict",
        columns: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None,
    ) -> pd.DataFrame:
        """
        Read Parquet file with schema validation.
        
        Args:
            file_path: Path to Parquet file
            schema: PyArrow schema for validation. If not provided, uses schema_name
            schema_name: Name of registered schema to use for validation
            enforcement_mode: How to handle schema mismatches:
                - "strict": Raise error on any mismatch
                - "coerce": Try to coerce data to schema types
                - "ignore": Skip schema validation
            columns: Optional list of columns to read
            filters: Optional row group filters
            
        Returns:
            pandas DataFrame with data
            
        Raises:
            ParquetReadError: If file cannot be read
            SchemaValidationError: If schema validation fails
            CorruptedFileError: If file is corrupted
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ParquetReadError(str(file_path), "File does not exist")
        
        try:
            logger.info(f"Reading Parquet file: {file_path}")
            
            # Read Parquet file
            parquet_file = pq.ParquetFile(str(file_path))
            table = parquet_file.read(columns=columns, filters=filters)
            
            # Extract metadata
            file_metadata = self.metadata_manager.extract_metadata(
                parquet_file.schema_arrow.metadata
            )
            row_count = table.num_rows
            
            # Determine target schema
            if schema is None and schema_name is not None:
                schema = self.schema_registry.get(schema_name)
            
            # Validate schema if provided
            if schema is not None and enforcement_mode != "ignore":
                self._validate_schema(
                    table.schema,
                    schema,
                    enforcement_mode,
                    str(file_path)
                )
            
            # Convert to DataFrame
            df = table.to_pandas()
            
            # Attach metadata to DataFrame
            if file_metadata:
                df.attrs["parquet_metadata"] = file_metadata
            
            logger.info(
                f"Successfully read {row_count} rows from {file_path}. "
                f"Columns: {list(df.columns)}"
            )
            
            return df
        
        except pa.ArrowException as e:
            if "parquet" in str(e).lower() or "corrupt" in str(e).lower():
                raise CorruptedFileError(str(file_path), str(e))
            raise ParquetReadError(str(file_path), e)
        except SchemaValidationError:
            # Re-raise schema validation errors as-is
            raise
        except Exception as e:
            logger.error(f"Error reading Parquet file {file_path}: {e}")
            raise ParquetReadError(str(file_path), e)
    
    def write_parquet_validated(
        self,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        schema: Optional[pa.Schema] = None,
        schema_name: Optional[str] = None,
        compression: CompressionType = DEFAULT_COMPRESSION,
        enforcement_mode: EnforcementMode = "strict",
        metadata: Optional[Dict[str, str]] = None,
        partition_cols: Optional[List[str]] = None,
        coerce_types: bool = False,
    ) -> None:
        """
        Write DataFrame to Parquet file with schema validation.
        
        Args:
            df: DataFrame to write
            file_path: Output file path
            schema: PyArrow schema for validation. If not provided, uses schema_name
            schema_name: Name of registered schema to use for validation
            compression: Compression codec (default: snappy)
            enforcement_mode: How to handle schema mismatches:
                - "strict": Raise error on any mismatch
                - "coerce": Try to coerce DataFrame to schema
                - "ignore": Skip schema validation
            metadata: Custom metadata key-value pairs to embed
            partition_cols: Columns to partition by (for partitioned datasets)
            coerce_types: Whether to coerce DataFrame dtypes to match schema
            
        Raises:
            ParquetWriteError: If write fails
            SchemaValidationError: If schema validation fails
            CompressionError: If compression type is invalid
        """
        file_path = Path(file_path)
        self._validate_compression(compression)
        
        try:
            logger.info(
                f"Writing Parquet file: {file_path} "
                f"({len(df)} rows, {len(df.columns)} columns)"
            )
            
            # Determine target schema
            if schema is None and schema_name is not None:
                schema = self.schema_registry.get(schema_name)
            
            # Convert DataFrame to PyArrow table
            table = pa.Table.from_pandas(df)
            
            # Validate schema if provided
            if schema is not None and enforcement_mode != "ignore":
                self._validate_schema(
                    table.schema,
                    schema,
                    enforcement_mode,
                    str(file_path)
                )
                
                # Coerce types if requested
                if coerce_types and enforcement_mode == "coerce":
                    table = self._coerce_table_schema(table, schema)
            
            # Prepare metadata
            file_metadata = None
            if metadata is not None or schema is not None:
                file_metadata = metadata or {}
                
                # Add standard metadata
                if MetadataManager.ROW_COUNT not in file_metadata:
                    file_metadata = self.metadata_manager.create_metadata(
                        row_count=len(df),
                        **file_metadata
                    )
                
                # Convert to bytes for PyArrow
                file_metadata_bytes = {
                    k.encode("utf-8"): v.encode("utf-8")
                    for k, v in file_metadata.items()
                }
            else:
                file_metadata_bytes = None
            
            # Create output directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write Parquet file
            pq.write_table(
                table,
                str(file_path),
                compression=compression,
                metadata=file_metadata_bytes,
            )
            
            logger.info(
                f"Successfully wrote Parquet file: {file_path} "
                f"(compression: {compression})"
            )
        
        except CompressionError:
            raise
        except SchemaValidationError:
            raise
        except Exception as e:
            logger.error(f"Error writing Parquet file {file_path}: {e}")
            raise ParquetWriteError(str(file_path), e)
    
    def write_partitioned_dataset(
        self,
        df: pd.DataFrame,
        root_path: Union[str, Path],
        partition_cols: List[str],
        schema: Optional[pa.Schema] = None,
        schema_name: Optional[str] = None,
        compression: CompressionType = DEFAULT_COMPRESSION,
        enforcement_mode: EnforcementMode = "strict",
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Write partitioned Parquet dataset (directory with multiple files).
        
        Args:
            df: DataFrame to write
            root_path: Root directory for partitioned dataset
            partition_cols: Columns to partition by
            schema: PyArrow schema for validation
            schema_name: Name of registered schema
            compression: Compression codec
            enforcement_mode: Schema enforcement mode
            metadata: Custom metadata
            
        Raises:
            PartitioningError: If partitioning fails
            ParquetWriteError: If write fails
        """
        root_path = Path(root_path)
        self._validate_compression(compression)
        
        try:
            # Validate partition columns exist
            missing_cols = set(partition_cols) - set(df.columns)
            if missing_cols:
                raise PartitioningError(
                    f"Partition columns not found in DataFrame: {missing_cols}",
                    ", ".join(missing_cols)
                )
            
            logger.info(
                f"Writing partitioned dataset to: {root_path} "
                f"(partition columns: {partition_cols})"
            )
            
            # Convert to PyArrow table
            table = pa.Table.from_pandas(df)
            
            # Validate schema if provided
            if schema is None and schema_name is not None:
                schema = self.schema_registry.get(schema_name)
            
            if schema is not None and enforcement_mode != "ignore":
                self._validate_schema(table.schema, schema, enforcement_mode, str(root_path))
            
            # Create root directory
            root_path.mkdir(parents=True, exist_ok=True)
            
            # Write partitioned dataset
            pq.write_to_dataset(
                table,
                root_path=str(root_path),
                partition_cols=partition_cols,
                compression=compression,
            )
            
            # Write metadata manifest if provided
            if metadata:
                manifest_path = root_path / "_metadata.json"
                import json
                with open(manifest_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully wrote partitioned dataset to: {root_path}")
        
        except (PartitioningError, ParquetWriteError):
            raise
        except Exception as e:
            logger.error(f"Error writing partitioned dataset to {root_path}: {e}")
            raise PartitioningError(str(e), "/".join(partition_cols))
    
    def read_partitioned_dataset(
        self,
        root_path: Union[str, Path],
        schema: Optional[pa.Schema] = None,
        schema_name: Optional[str] = None,
        enforcement_mode: EnforcementMode = "strict",
        filters: Optional[List[tuple]] = None,
    ) -> pd.DataFrame:
        """
        Read partitioned Parquet dataset.
        
        Args:
            root_path: Root directory of partitioned dataset
            schema: PyArrow schema for validation
            schema_name: Name of registered schema
            enforcement_mode: Schema enforcement mode
            filters: Row group filters
            
        Returns:
            Combined DataFrame from all partitions
            
        Raises:
            ParquetReadError: If read fails
            SchemaValidationError: If schema validation fails
        """
        root_path = Path(root_path)
        
        try:
            logger.info(f"Reading partitioned dataset from: {root_path}")
            
            # Read partitioned dataset
            table = pq.read_table(str(root_path), filters=filters)
            
            # Determine target schema
            if schema is None and schema_name is not None:
                schema = self.schema_registry.get(schema_name)
            
            # Validate schema if provided
            if schema is not None and enforcement_mode != "strict":
                self._validate_schema(table.schema, schema, enforcement_mode, str(root_path))
            
            df = table.to_pandas()
            
            logger.info(
                f"Successfully read {len(df)} rows from partitioned dataset"
            )
            
            return df
        
        except SchemaValidationError:
            raise
        except Exception as e:
            logger.error(f"Error reading partitioned dataset from {root_path}: {e}")
            raise ParquetReadError(str(root_path), e)
    
    def _validate_schema(
        self,
        actual_schema: pa.Schema,
        expected_schema: pa.Schema,
        enforcement_mode: EnforcementMode,
        file_path: str,
    ) -> None:
        """
        Validate actual schema against expected schema.
        
        Args:
            actual_schema: Schema of actual data
            expected_schema: Expected schema
            enforcement_mode: How to handle mismatches
            file_path: File path for error reporting
            
        Raises:
            SchemaValidationError: If schema validation fails (in strict mode)
        """
        # Check if schemas match
        is_valid, error_msg = self.schema_registry.validate_data_against_schema(
            actual_schema,
            expected_schema,
            strict=(enforcement_mode == "strict")
        )
        
        if not is_valid:
            if enforcement_mode == "strict":
                raise SchemaMismatchError(
                    expected=self.schema_registry.schema_to_dict(expected_schema),
                    actual=self.schema_registry.schema_to_dict(actual_schema),
                    file_path=file_path,
                    mismatches=[error_msg]
                )
            elif enforcement_mode == "coerce":
                logger.warning(
                    f"Schema mismatch in {file_path}: {error_msg}. "
                    f"Will attempt to coerce types."
                )
            else:  # ignore
                logger.warning(
                    f"Schema mismatch in {file_path}: {error_msg}. "
                    f"Ignoring validation."
                )
    
    @staticmethod
    def _coerce_table_schema(table: pa.Table, target_schema: pa.Schema) -> pa.Table:
        """
        Attempt to coerce table to target schema.
        
        Args:
            table: PyArrow table with current schema
            target_schema: Target schema to coerce to
            
        Returns:
            Table with coerced schema
            
        Raises:
            SchemaValidationError: If coercion fails
        """
        try:
            # Cast columns to target types
            cast_columns = []
            for field in target_schema:
                if field.name in table.column_names:
                    col = table[field.name]
                    if col.type != field.type:
                        col = col.cast(field.type)
                    cast_columns.append(col)
            
            if cast_columns:
                return pa.table({field.name: col for field, col in 
                               zip(target_schema, cast_columns)})
            return table
        
        except pa.ArrowException as e:
            raise SchemaValidationError(
                f"Failed to coerce schema: {e}",
                schema_error=str(e)
            )
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get metadata and info about a Parquet file.
        
        Args:
            file_path: Path to Parquet file
            
        Returns:
            Dictionary with file information
            
        Raises:
            ParquetReadError: If file cannot be read
        """
        file_path = Path(file_path)
        
        try:
            parquet_file = pq.ParquetFile(str(file_path))
            
            return {
                "file_path": str(file_path),
                "num_rows": parquet_file.num_rows,
                "num_columns": len(parquet_file.schema),
                "columns": parquet_file.schema.names,
                "schema": str(parquet_file.schema),
                "metadata": self.metadata_manager.extract_metadata(
                    parquet_file.schema_arrow.metadata
                ),
                "file_size_bytes": os.path.getsize(file_path),
                "num_row_groups": parquet_file.num_row_groups,
            }
        
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            raise ParquetReadError(str(file_path), e)


# High-level wrapper functions for common operations

def read_raw_data(
    file_path: Union[str, Path],
    compression_auto_detect: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Read raw OHLCV data from Parquet file.
    
    Args:
        file_path: Path to Parquet file
        compression_auto_detect: Auto-detect compression (default: True)
        **kwargs: Additional arguments for read_parquet_with_schema
        
    Returns:
        DataFrame with raw OHLCV data
    """
    handler = ParquetIOHandler()
    return handler.read_parquet_with_schema(
        file_path,
        schema_name="ohlcv",
        **kwargs
    )


def write_cleaned_data(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
    compression: CompressionType = "snappy",
    **kwargs
) -> None:
    """
    Write cleaned OHLCV data to Parquet file.
    
    Args:
        df: DataFrame with cleaned data
        file_path: Output file path
        metadata: Optional metadata dictionary
        compression: Compression codec (default: snappy)
        **kwargs: Additional arguments for write_parquet_validated
    """
    handler = ParquetIOHandler()
    
    if metadata is None:
        metadata = {}
    
    # Add default metadata for cleaned data
    if "version" not in metadata:
        metadata["version"] = "v1"
    
    handler.write_parquet_validated(
        df,
        file_path,
        schema_name="ohlcv",
        compression=compression,
        metadata=metadata,
        **kwargs
    )


def read_features(
    file_path: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """
    Read feature data from Parquet file.
    
    Args:
        file_path: Path to Parquet file
        **kwargs: Additional arguments for read_parquet_with_schema
        
    Returns:
        DataFrame with features
    """
    handler = ParquetIOHandler()
    return handler.read_parquet_with_schema(
        file_path,
        schema_name="features",
        **kwargs
    )


def write_features(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
    compression: CompressionType = "snappy",
    **kwargs
) -> None:
    """
    Write feature data to Parquet file.
    
    Args:
        df: DataFrame with features
        file_path: Output file path
        metadata: Optional metadata dictionary
        compression: Compression codec (default: snappy)
        **kwargs: Additional arguments for write_parquet_validated
    """
    handler = ParquetIOHandler()
    
    if metadata is None:
        metadata = {}
    
    # Add default metadata for features
    if "version" not in metadata:
        metadata["version"] = "v1"
    
    handler.write_parquet_validated(
        df,
        file_path,
        schema_name="features",
        compression=compression,
        metadata=metadata,
        **kwargs
    )
