"""
Comprehensive test suite for Parquet I/O utilities.

Tests cover:
- Schema validation (strict, coerce, ignore modes)
- Read/write operations with various configurations
- Compression support
- Partitioning
- Metadata management
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import json

from src.data_io.parquet_utils import (
    read_parquet_with_schema,
    write_parquet_validated,
    read_partitioned_dataset,
    write_features,
    read_features,
    read_raw_data,
    write_cleaned_data,
    get_parquet_metadata,
    get_parquet_schema,
    SchemaValidator,
    validate_compression,
)
from src.data_io.schemas import (
    DataSchema,
    ColumnSchema,
    OHLCV_SCHEMA,
    PRICE_VOLATILITY_SCHEMA,
    get_schema,
)
from src.data_io.metadata import MetadataManager
from src.data_io.errors import (
    SchemaMismatchError,
    MissingRequiredColumnError,
    DataTypeValidationError,
    CompressionError,
    CorruptedFileError,
    ParquetIOError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1H")
    data = {
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 100),
        "high": np.random.uniform(110, 120, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(100, 110, 100),
        "volume": np.random.randint(1000, 10000, 100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_features_data():
    """Create sample features data."""
    dates = pd.date_range("2024-01-01", periods=50, freq="1D")
    data = {
        "timestamp": dates,
        "volatility_30d": np.random.uniform(0, 1, 50),
        "volatility_60d": np.random.uniform(0, 1, 50),
        "volatility_ratio": np.random.uniform(0.5, 2, 50),
        "price_range_pct": np.random.uniform(0, 5, 50),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_schema():
    """Create a sample schema for testing."""
    return DataSchema(
        name="test_schema",
        version="1.0.0",
        description="Test schema",
        columns=[
            ColumnSchema("timestamp", "timestamp[ns]", nullable=False),
            ColumnSchema("value", "double", nullable=False),
            ColumnSchema("category", "string", nullable=True),
        ],
    )


# ============================================================================
# Tests: Basic Read/Write
# ============================================================================

class TestBasicReadWrite:
    """Tests for basic read/write operations."""
    
    def test_write_and_read_parquet(self, temp_dir, sample_ohlcv_data):
        """Test basic write and read of Parquet file."""
        file_path = temp_dir / "test.parquet"
        
        # Write
        write_parquet_validated(sample_ohlcv_data, file_path)
        assert file_path.exists()
        
        # Read
        df = read_parquet_with_schema(file_path)
        assert len(df) == len(sample_ohlcv_data)
        assert list(df.columns) == list(sample_ohlcv_data.columns)
    
    def test_write_with_custom_metadata(self, temp_dir, sample_ohlcv_data):
        """Test write with custom metadata."""
        file_path = temp_dir / "test.parquet"
        metadata = {"source": "test_data", "version": "1.0"}
        
        write_parquet_validated(sample_ohlcv_data, file_path, metadata=metadata)
        
        file_metadata = get_parquet_metadata(file_path)
        assert file_metadata is not None
        assert "num_rows" in file_metadata
        assert file_metadata["num_rows"] == len(sample_ohlcv_data)
    
    def test_read_nonexistent_file(self, temp_dir):
        """Test reading a nonexistent file raises error."""
        file_path = temp_dir / "nonexistent.parquet"
        
        with pytest.raises(FileNotFoundError):
            read_parquet_with_schema(file_path)


# ============================================================================
# Tests: Schema Validation
# ============================================================================

class TestSchemaValidation:
    """Tests for schema validation."""
    
    def test_validate_with_correct_schema(self, temp_dir, sample_ohlcv_data):
        """Test validation succeeds with correct schema."""
        file_path = temp_dir / "test.parquet"
        
        write_parquet_validated(
            sample_ohlcv_data,
            file_path,
            schema=OHLCV_SCHEMA,
        )
        
        df = read_parquet_with_schema(
            file_path,
            schema=OHLCV_SCHEMA,
        )
        assert len(df) == len(sample_ohlcv_data)
    
    def test_validate_with_missing_required_column(self, temp_dir):
        """Test validation fails when required column is missing."""
        file_path = temp_dir / "test.parquet"
        
        # Create data missing a required column
        data = {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1H"),
            "open": np.random.uniform(100, 110, 10),
            # Missing: high, low, close, volume
        }
        df = pd.DataFrame(data)
        
        with pytest.raises(MissingRequiredColumnError):
            write_parquet_validated(
                df,
                file_path,
                schema=OHLCV_SCHEMA,
            )
    
    def test_read_with_column_subset(self, temp_dir, sample_ohlcv_data):
        """Test reading specific columns."""
        file_path = temp_dir / "test.parquet"
        
        write_parquet_validated(sample_ohlcv_data, file_path)
        
        columns = ["timestamp", "open", "close"]
        df = read_parquet_with_schema(file_path, columns=columns)
        
        assert set(df.columns) == set(columns)


# ============================================================================
# Tests: Compression
# ============================================================================

class TestCompression:
    """Tests for compression support."""
    
    @pytest.mark.parametrize("compression", ["snappy", "gzip", "zstd", "uncompressed"])
    def test_write_read_with_compression(self, temp_dir, sample_ohlcv_data, compression):
        """Test write and read with different compression codecs."""
        file_path = temp_dir / f"test_{compression}.parquet"
        
        write_parquet_validated(
            sample_ohlcv_data,
            file_path,
            compression=compression,
        )
        
        df = read_parquet_with_schema(file_path)
        assert len(df) == len(sample_ohlcv_data)
    
    def test_compression_validation(self):
        """Test compression codec validation."""
        assert validate_compression("snappy") == "snappy"
        assert validate_compression("gzip") == "gzip"
        
        with pytest.raises(CompressionError):
            validate_compression("invalid_codec")
    
    def test_compression_reduces_file_size(self, temp_dir, sample_ohlcv_data):
        """Test that compression reduces file size."""
        uncompressed_path = temp_dir / "uncompressed.parquet"
        compressed_path = temp_dir / "compressed.parquet"
        
        write_parquet_validated(sample_ohlcv_data, uncompressed_path, compression="uncompressed")
        write_parquet_validated(sample_ohlcv_data, compressed_path, compression="snappy")
        
        uncompressed_size = uncompressed_path.stat().st_size
        compressed_size = compressed_path.stat().st_size
        
        # Compression should reduce size
        assert compressed_size < uncompressed_size


# ============================================================================
# Tests: Partitioning
# ============================================================================

class TestPartitioning:
    """Tests for partitioned dataset operations."""
    
    def test_write_partitioned_dataset_by_date(self, temp_dir, sample_ohlcv_data):
        """Test writing partitioned dataset by date."""
        sample_ohlcv_data["date"] = sample_ohlcv_data["timestamp"].dt.date
        
        dataset_path = temp_dir / "partitioned_dataset"
        
        write_parquet_validated(
            sample_ohlcv_data,
            dataset_path,
            partition_cols=["date"],
        )
        
        # Check that partition directories were created
        assert dataset_path.exists()
    
    def test_read_partitioned_dataset(self, temp_dir, sample_ohlcv_data):
        """Test reading partitioned dataset."""
        sample_ohlcv_data["year"] = sample_ohlcv_data["timestamp"].dt.year
        
        dataset_path = temp_dir / "partitioned_dataset"
        
        write_parquet_validated(
            sample_ohlcv_data,
            dataset_path,
            partition_cols=["year"],
        )
        
        df = read_partitioned_dataset(dataset_path)
        assert len(df) > 0


# ============================================================================
# Tests: Metadata Management
# ============================================================================

class TestMetadataManagement:
    """Tests for metadata management."""
    
    def test_create_metadata(self):
        """Test metadata creation."""
        metadata = MetadataManager.create_metadata(
            layer="raw",
            asset="MES",
            record_count=1000,
            file_size_bytes=1024 * 1024,
            timestamp_range=(
                "2024-01-01T00:00:00Z",
                "2024-01-31T23:59:59Z",
            ),
        )
        
        assert metadata["layer"] == "raw"
        assert metadata["asset"] == "MES"
        assert metadata["record_count"] == 1000
    
    def test_write_and_read_metadata_file(self, temp_dir):
        """Test writing and reading metadata files."""
        metadata_path = temp_dir / "metadata.json"
        
        metadata = MetadataManager.create_metadata(
            layer="cleaned",
            asset="ES",
            record_count=500,
            file_size_bytes=512 * 1024,
            version_info={"major_version": 1, "previous_version": 0},
        )
        
        MetadataManager.write_metadata_file(metadata, metadata_path)
        assert metadata_path.exists()
        
        read_metadata = MetadataManager.read_metadata_file(metadata_path)
        assert read_metadata["layer"] == "cleaned"
        assert read_metadata["asset"] == "ES"
    
    def test_add_file_info_to_metadata(self):
        """Test adding file info to metadata."""
        metadata = MetadataManager.create_metadata(
            layer="features",
            asset="VIX",
            record_count=100,
            file_size_bytes=100 * 1024,
        )
        
        MetadataManager.add_file_info(
            metadata,
            "features.parquet",
            size_bytes=100 * 1024,
            file_format="parquet",
            row_count=100,
        )
        
        assert "file_info" in metadata
        assert "file_list" in metadata["file_info"]
        assert len(metadata["file_info"]["file_list"]) == 1
    
    def test_add_quality_metrics(self):
        """Test adding quality metrics to metadata."""
        metadata = MetadataManager.create_metadata(
            layer="raw",
            asset="MES",
            record_count=100,
            file_size_bytes=100 * 1024,
        )
        
        MetadataManager.add_quality_metrics(
            metadata,
            null_counts={"timestamp": 0, "volume": 5},
            quality_flags=["gaps_in_timestamps"],
            completeness_percentage=98.5,
        )
        
        assert "data_quality" in metadata
        assert metadata["data_quality"]["completeness_percentage"] == 98.5


# ============================================================================
# Tests: Layer-Specific Wrappers
# ============================================================================

class TestLayerWrappers:
    """Tests for layer-specific wrapper functions."""
    
    def test_read_raw_data(self, temp_dir, sample_ohlcv_data):
        """Test reading raw data."""
        file_path = temp_dir / "raw.parquet"
        
        write_parquet_validated(sample_ohlcv_data, file_path)
        df = read_raw_data(file_path)
        
        assert len(df) == len(sample_ohlcv_data)
    
    def test_write_read_cleaned_data(self, temp_dir, sample_ohlcv_data):
        """Test writing and reading cleaned data."""
        file_path = temp_dir / "cleaned.parquet"
        
        write_cleaned_data(sample_ohlcv_data, file_path)
        df = read_parquet_with_schema(file_path)
        
        assert len(df) == len(sample_ohlcv_data)
    
    def test_write_read_features(self, temp_dir, sample_features_data):
        """Test writing and reading feature data."""
        file_path = temp_dir / "features.parquet"
        
        write_features(sample_features_data, file_path)
        df = read_features(file_path)
        
        assert len(df) == len(sample_features_data)
    
    def test_features_with_schema(self, temp_dir, sample_features_data):
        """Test writing/reading features with schema validation."""
        file_path = temp_dir / "features.parquet"
        
        write_features(
            sample_features_data,
            file_path,
            schema=PRICE_VOLATILITY_SCHEMA,
        )
        
        df = read_features(file_path, schema=PRICE_VOLATILITY_SCHEMA)
        assert len(df) == len(sample_features_data)


# ============================================================================
# Tests: Schema Utilities
# ============================================================================

class TestSchemaUtilities:
    """Tests for schema utility functions."""
    
    def test_get_schema_from_registry(self):
        """Test retrieving schema from registry."""
        schema = get_schema("ohlcv")
        assert schema.name == "ohlcv"
        assert schema.version == "1.0.0"
    
    def test_get_nonexistent_schema_raises_error(self):
        """Test that requesting nonexistent schema raises error."""
        with pytest.raises(KeyError):
            get_schema("nonexistent_schema")
    
    def test_schema_column_properties(self):
        """Test schema column properties."""
        schema = OHLCV_SCHEMA
        
        required = schema.get_required_columns()
        assert "timestamp" in required
        assert "volume" in required
        
        optional = schema.get_optional_columns()
        assert len(optional) == 0  # OHLCV has no optional columns


# ============================================================================
# Tests: Parquet Metadata Extraction
# ============================================================================

class TestParquetMetadataExtraction:
    """Tests for extracting metadata from Parquet files."""
    
    def test_get_parquet_metadata(self, temp_dir, sample_ohlcv_data):
        """Test extracting metadata from Parquet file."""
        file_path = temp_dir / "test.parquet"
        write_parquet_validated(sample_ohlcv_data, file_path)
        
        metadata = get_parquet_metadata(file_path)
        
        assert metadata["num_rows"] == len(sample_ohlcv_data)
        assert metadata["num_columns"] == len(sample_ohlcv_data.columns)
        assert set(metadata["columns"]) == set(sample_ohlcv_data.columns)
    
    def test_get_parquet_schema(self, temp_dir, sample_ohlcv_data):
        """Test extracting schema from Parquet file."""
        file_path = temp_dir / "test.parquet"
        write_parquet_validated(sample_ohlcv_data, file_path)
        
        schema = get_parquet_schema(file_path)
        
        assert "timestamp" in schema
        assert "open" in schema
        assert "volume" in schema


# ============================================================================
# Tests: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_read_corrupted_file_raises_error(self, temp_dir):
        """Test that reading corrupted file raises error."""
        # Create a file that's not a valid Parquet file
        file_path = temp_dir / "corrupted.parquet"
        file_path.write_text("not a parquet file")
        
        with pytest.raises((CorruptedFileError, ParquetIOError)):
            read_parquet_with_schema(file_path)
    
    def test_invalid_compression_raises_error(self, temp_dir, sample_ohlcv_data):
        """Test that invalid compression raises error."""
        file_path = temp_dir / "test.parquet"
        
        with pytest.raises(CompressionError):
            write_parquet_validated(
                sample_ohlcv_data,
                file_path,
                compression="invalid",
            )


# ============================================================================
# Tests: Data Type Coercion
# ============================================================================

class TestDataTypeCoercion:
    """Tests for data type coercion."""
    
    def test_timestamp_coercion(self, temp_dir):
        """Test timestamp coercion."""
        schema = DataSchema(
            name="test",
            version="1.0.0",
            description="Test",
            columns=[
                ColumnSchema("timestamp", "timestamp[ns]", nullable=False),
                ColumnSchema("value", "double", nullable=False),
            ],
        )
        
        # Create data with string timestamps
        data = {
            "timestamp": ["2024-01-01 00:00:00"] * 10,
            "value": np.random.uniform(0, 100, 10),
        }
        df = pd.DataFrame(data)
        
        file_path = temp_dir / "test.parquet"
        
        # Should coerce successfully
        write_parquet_validated(df, file_path, schema=schema)
        
        df_read = read_parquet_with_schema(file_path, schema=schema)
        assert pd.api.types.is_datetime64_any_dtype(df_read["timestamp"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
