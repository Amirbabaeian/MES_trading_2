"""
Comprehensive unit tests for Parquet I/O utilities.

Tests cover schema validation, compression, metadata management,
partitioning, and error handling.
"""

import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import json
import os

import pandas as pd
import numpy as np
import pyarrow as pa

from src.data_io.parquet_utils import (
    ParquetIOHandler,
    read_raw_data,
    write_cleaned_data,
    read_features,
    write_features,
)
from src.data_io.schemas import get_schema, OHLCV_SCHEMA, FEATURES_SCHEMA
from src.data_io.metadata import MetadataManager
from src.data_io.errors import (
    SchemaValidationError,
    SchemaMismatchError,
    ParquetReadError,
    ParquetWriteError,
    CompressionError,
    PartitioningError,
)


class TestParquetIOHandler(unittest.TestCase):
    """Tests for ParquetIOHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = ParquetIOHandler()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def _create_ohlcv_df(self, rows: int = 100) -> pd.DataFrame:
        """Create a test OHLCV DataFrame."""
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(hours=i) for i in range(rows)]
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "asset": ["MES"] * rows,
            "open": np.random.uniform(4500, 4600, rows),
            "high": np.random.uniform(4600, 4700, rows),
            "low": np.random.uniform(4400, 4500, rows),
            "close": np.random.uniform(4500, 4600, rows),
            "volume": np.random.randint(1000, 10000, rows),
        })
    
    def _create_features_df(self, rows: int = 100) -> pd.DataFrame:
        """Create a test features DataFrame."""
        start_date = datetime(2024, 1, 1)
        timestamps = [start_date + timedelta(hours=i) for i in range(rows)]
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "asset": ["MES"] * rows,
            "feature_set": ["v1"] * rows,
            "features": [
                {
                    "sma_20": np.random.uniform(4500, 4600),
                    "sma_50": np.random.uniform(4500, 4600),
                    "rsi_14": np.random.uniform(30, 70),
                }
                for _ in range(rows)
            ],
        })
    
    # ===== Write Tests =====
    
    def test_write_parquet_basic(self):
        """Test basic Parquet write functionality."""
        df = self._create_ohlcv_df(50)
        output_path = self.test_dir / "test_basic.parquet"
        
        self.handler.write_parquet_validated(
            df,
            output_path,
            schema_name="ohlcv",
            enforcement_mode="ignore"
        )
        
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)
    
    def test_write_with_schema_validation(self):
        """Test write with schema validation."""
        df = self._create_ohlcv_df(50)
        output_path = self.test_dir / "test_with_schema.parquet"
        
        # Should succeed with matching schema
        self.handler.write_parquet_validated(
            df,
            output_path,
            schema_name="ohlcv",
            enforcement_mode="strict"
        )
        
        self.assertTrue(output_path.exists())
    
    def test_write_with_metadata(self):
        """Test write with custom metadata."""
        df = self._create_ohlcv_df(50)
        output_path = self.test_dir / "test_with_metadata.parquet"
        
        metadata = {
            "asset": "MES",
            "version": "v1.0",
            "changelog": "Initial version",
        }
        
        self.handler.write_parquet_validated(
            df,
            output_path,
            schema_name="ohlcv",
            metadata=metadata,
            enforcement_mode="ignore"
        )
        
        # Read back and verify metadata
        file_info = self.handler.get_file_info(output_path)
        self.assertEqual(file_info["metadata"]["asset"], "MES")
        self.assertEqual(file_info["metadata"]["version"], "v1.0")
    
    def test_write_with_different_compressions(self):
        """Test write with different compression codecs."""
        df = self._create_ohlcv_df(100)
        
        for compression in ["snappy", "gzip", "zstd", "none"]:
            output_path = self.test_dir / f"test_{compression}.parquet"
            
            self.handler.write_parquet_validated(
                df,
                output_path,
                schema_name="ohlcv",
                compression=compression,
                enforcement_mode="ignore"
            )
            
            self.assertTrue(output_path.exists())
    
    def test_write_invalid_compression(self):
        """Test write with invalid compression type."""
        df = self._create_ohlcv_df(50)
        output_path = self.test_dir / "test_invalid_compression.parquet"
        
        with self.assertRaises(CompressionError):
            self.handler.write_parquet_validated(
                df,
                output_path,
                compression="invalid",
                enforcement_mode="ignore"
            )
    
    def test_write_creates_parent_directories(self):
        """Test that write creates parent directories."""
        df = self._create_ohlcv_df(50)
        output_path = self.test_dir / "nested" / "dirs" / "test.parquet"
        
        self.handler.write_parquet_validated(
            df,
            output_path,
            schema_name="ohlcv",
            enforcement_mode="ignore"
        )
        
        self.assertTrue(output_path.exists())
    
    # ===== Read Tests =====
    
    def test_read_parquet_basic(self):
        """Test basic Parquet read functionality."""
        df_write = self._create_ohlcv_df(50)
        output_path = self.test_dir / "test_read_basic.parquet"
        
        self.handler.write_parquet_validated(
            df_write,
            output_path,
            schema_name="ohlcv",
            enforcement_mode="ignore"
        )
        
        df_read = self.handler.read_parquet_with_schema(
            output_path,
            enforcement_mode="ignore"
        )
        
        self.assertEqual(len(df_read), 50)
        self.assertListEqual(
            sorted(df_read.columns),
            sorted(["timestamp", "asset", "open", "high", "low", "close", "volume"])
        )
    
    def test_read_with_schema_validation(self):
        """Test read with schema validation."""
        df_write = self._create_ohlcv_df(50)
        output_path = self.test_dir / "test_read_with_schema.parquet"
        
        self.handler.write_parquet_validated(
            df_write,
            output_path,
            schema_name="ohlcv",
            enforcement_mode="ignore"
        )
        
        df_read = self.handler.read_parquet_with_schema(
            output_path,
            schema_name="ohlcv",
            enforcement_mode="strict"
        )
        
        self.assertEqual(len(df_read), 50)
    
    def test_read_with_column_selection(self):
        """Test read with specific columns."""
        df_write = self._create_ohlcv_df(50)
        output_path = self.test_dir / "test_read_columns.parquet"
        
        self.handler.write_parquet_validated(
            df_write,
            output_path,
            schema_name="ohlcv",
            enforcement_mode="ignore"
        )
        
        df_read = self.handler.read_parquet_with_schema(
            output_path,
            columns=["timestamp", "close"],
            enforcement_mode="ignore"
        )
        
        self.assertListEqual(sorted(df_read.columns), ["close", "timestamp"])
    
    def test_read_nonexistent_file(self):
        """Test reading nonexistent file raises error."""
        with self.assertRaises(ParquetReadError):
            self.handler.read_parquet_with_schema(
                self.test_dir / "nonexistent.parquet",
                enforcement_mode="ignore"
            )
    
    # ===== Schema Validation Tests =====
    
    def test_schema_validation_missing_columns(self):
        """Test schema validation with missing columns."""
        # Create DataFrame missing a required column
        df = pd.DataFrame({
            "timestamp": [datetime.now()],
            "asset": ["MES"],
            # Missing: open, high, low, close, volume
        })
        
        output_path = self.test_dir / "test_missing_cols.parquet"
        
        with self.assertRaises(SchemaMismatchError):
            self.handler.write_parquet_validated(
                df,
                output_path,
                schema_name="ohlcv",
                enforcement_mode="strict"
            )
    
    def test_schema_validation_extra_columns(self):
        """Test schema validation with extra columns."""
        df = self._create_ohlcv_df(50)
        df["extra_column"] = "extra_data"
        
        output_path = self.test_dir / "test_extra_cols.parquet"
        
        # Should succeed in non-strict mode
        self.handler.write_parquet_validated(
            df,
            output_path,
            schema_name="ohlcv",
            enforcement_mode="ignore"
        )
        
        self.assertTrue(output_path.exists())
    
    def test_schema_validation_type_mismatch(self):
        """Test schema validation with type mismatches."""
        df = self._create_ohlcv_df(50)
        df["close"] = df["close"].astype(str)  # Should be float64
        
        output_path = self.test_dir / "test_type_mismatch.parquet"
        
        with self.assertRaises(SchemaMismatchError):
            self.handler.write_parquet_validated(
                df,
                output_path,
                schema_name="ohlcv",
                enforcement_mode="strict"
            )
    
    def test_schema_coerce_mode(self):
        """Test schema coercion mode."""
        df = self._create_ohlcv_df(50)
        df["volume"] = df["volume"].astype(float)  # Should be uint64
        
        output_path = self.test_dir / "test_coerce.parquet"
        
        # Should warn but succeed in coerce mode
        self.handler.write_parquet_validated(
            df,
            output_path,
            schema_name="ohlcv",
            enforcement_mode="coerce"
        )
        
        self.assertTrue(output_path.exists())
    
    # ===== Metadata Tests =====
    
    def test_metadata_extraction(self):
        """Test metadata extraction from files."""
        df = self._create_ohlcv_df(50)
        output_path = self.test_dir / "test_metadata.parquet"
        
        metadata = {
            "asset": "MES",
            "version": "v1.2.3",
        }
        
        self.handler.write_parquet_validated(
            df,
            output_path,
            schema_name="ohlcv",
            metadata=metadata,
            enforcement_mode="ignore"
        )
        
        file_info = self.handler.get_file_info(output_path)
        
        self.assertIn("asset", file_info["metadata"])
        self.assertEqual(file_info["metadata"]["asset"], "MES")
    
    def test_metadata_with_dependencies(self):
        """Test metadata with complex fields like dependencies."""
        df = self._create_ohlcv_df(50)
        output_path = self.test_dir / "test_deps_metadata.parquet"
        
        metadata = MetadataManager.create_metadata(
            asset="MES",
            version="v1",
            dependencies=["raw_data_v1", "vendor_data_v2"]
        )
        
        self.handler.write_parquet_validated(
            df,
            output_path,
            metadata=metadata,
            enforcement_mode="ignore"
        )
        
        file_info = self.handler.get_file_info(output_path)
        
        # Dependencies should be present in metadata
        self.assertIn("dependencies", file_info["metadata"])
    
    # ===== Partitioning Tests =====
    
    def test_write_partitioned_dataset_by_asset(self):
        """Test writing partitioned dataset by asset."""
        # Create data with multiple assets
        df = pd.concat([
            self._create_ohlcv_df(50).assign(asset="MES"),
            self._create_ohlcv_df(50).assign(asset="ES"),
        ], ignore_index=True)
        
        output_dir = self.test_dir / "partitioned_by_asset"
        
        self.handler.write_partitioned_dataset(
            df,
            output_dir,
            partition_cols=["asset"],
            schema_name="ohlcv",
            enforcement_mode="ignore"
        )
        
        self.assertTrue(output_dir.exists())
        # Check for partition directories
        self.assertTrue((output_dir / "asset=MES").exists())
        self.assertTrue((output_dir / "asset=ES").exists())
    
    def test_write_partitioned_dataset_by_date(self):
        """Test writing partitioned dataset by date."""
        df = self._create_ohlcv_df(100)
        # Add a date column
        df["date"] = df["timestamp"].dt.date.astype(str)
        
        output_dir = self.test_dir / "partitioned_by_date"
        
        self.handler.write_partitioned_dataset(
            df,
            output_dir,
            partition_cols=["date"],
            schema_name="ohlcv",
            enforcement_mode="ignore"
        )
        
        self.assertTrue(output_dir.exists())
    
    def test_write_partitioned_missing_column(self):
        """Test partitioning by non-existent column."""
        df = self._create_ohlcv_df(50)
        output_dir = self.test_dir / "partitioned_missing"
        
        with self.assertRaises(PartitioningError):
            self.handler.write_partitioned_dataset(
                df,
                output_dir,
                partition_cols=["nonexistent"],
                schema_name="ohlcv",
                enforcement_mode="ignore"
            )
    
    def test_read_partitioned_dataset(self):
        """Test reading partitioned dataset."""
        # Create data with multiple assets
        df_write = pd.concat([
            self._create_ohlcv_df(50).assign(asset="MES"),
            self._create_ohlcv_df(50).assign(asset="ES"),
        ], ignore_index=True)
        
        output_dir = self.test_dir / "read_partitioned"
        
        self.handler.write_partitioned_dataset(
            df_write,
            output_dir,
            partition_cols=["asset"],
            schema_name="ohlcv",
            enforcement_mode="ignore"
        )
        
        df_read = self.handler.read_partitioned_dataset(
            output_dir,
            schema_name="ohlcv",
            enforcement_mode="ignore"
        )
        
        self.assertEqual(len(df_read), 100)
    
    # ===== Compression Tests =====
    
    def test_compression_reduces_file_size(self):
        """Test that compression reduces file size."""
        df = self._create_ohlcv_df(1000)
        
        # Write with no compression
        uncompressed_path = self.test_dir / "uncompressed.parquet"
        self.handler.write_parquet_validated(
            df,
            uncompressed_path,
            schema_name="ohlcv",
            compression="none",
            enforcement_mode="ignore"
        )
        
        # Write with snappy compression
        compressed_path = self.test_dir / "compressed.parquet"
        self.handler.write_parquet_validated(
            df,
            compressed_path,
            schema_name="ohlcv",
            compression="snappy",
            enforcement_mode="ignore"
        )
        
        uncompressed_size = uncompressed_path.stat().st_size
        compressed_size = compressed_path.stat().st_size
        
        # Compression should reduce file size
        self.assertLess(compressed_size, uncompressed_size)
    
    # ===== High-Level Wrapper Tests =====
    
    def test_read_raw_data_wrapper(self):
        """Test read_raw_data wrapper function."""
        df_write = self._create_ohlcv_df(50)
        output_path = self.test_dir / "raw_data.parquet"
        
        self.handler.write_parquet_validated(
            df_write,
            output_path,
            schema_name="ohlcv",
            enforcement_mode="ignore"
        )
        
        df_read = read_raw_data(output_path, enforcement_mode="ignore")
        
        self.assertEqual(len(df_read), 50)
    
    def test_write_cleaned_data_wrapper(self):
        """Test write_cleaned_data wrapper function."""
        df = self._create_ohlcv_df(50)
        output_path = self.test_dir / "cleaned_data.parquet"
        
        write_cleaned_data(
            df,
            output_path,
            metadata={"version": "v1.0"},
            enforcement_mode="ignore"
        )
        
        self.assertTrue(output_path.exists())
    
    def test_read_write_features_wrapper(self):
        """Test read_features and write_features wrapper functions."""
        df_write = self._create_features_df(50)
        output_path = self.test_dir / "features.parquet"
        
        write_features(
            df_write,
            output_path,
            metadata={"feature_set": "v1"},
            enforcement_mode="ignore"
        )
        
        df_read = read_features(output_path, enforcement_mode="ignore")
        
        self.assertEqual(len(df_read), 50)
    
    # ===== File Info Tests =====
    
    def test_get_file_info(self):
        """Test getting file information."""
        df = self._create_ohlcv_df(100)
        output_path = self.test_dir / "info_test.parquet"
        
        self.handler.write_parquet_validated(
            df,
            output_path,
            schema_name="ohlcv",
            enforcement_mode="ignore"
        )
        
        file_info = self.handler.get_file_info(output_path)
        
        self.assertEqual(file_info["num_rows"], 100)
        self.assertEqual(file_info["num_columns"], 7)  # OHLCV has 7 columns
        self.assertEqual(len(file_info["columns"]), 7)
        self.assertGreater(file_info["file_size_bytes"], 0)
    
    def test_get_file_info_nonexistent(self):
        """Test getting info for nonexistent file."""
        with self.assertRaises(ParquetReadError):
            self.handler.get_file_info(
                self.test_dir / "nonexistent.parquet"
            )


class TestMetadataManager(unittest.TestCase):
    """Tests for MetadataManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = MetadataManager()
    
    def test_create_metadata_basic(self):
        """Test basic metadata creation."""
        metadata = self.manager.create_metadata(
            asset="MES",
            version="v1.0"
        )
        
        self.assertIn("asset", metadata)
        self.assertEqual(metadata["asset"], "MES")
        self.assertEqual(metadata["version"], "v1.0")
        self.assertIn("creation_timestamp", metadata)
    
    def test_create_metadata_with_dependencies(self):
        """Test metadata creation with dependencies."""
        deps = ["raw_data_v1", "vendor_data_v2"]
        metadata = self.manager.create_metadata(
            dependencies=deps
        )
        
        # Dependencies should be JSON encoded
        self.assertIn("dependencies", metadata)
        self.assertEqual(json.loads(metadata["dependencies"]), deps)
    
    def test_validate_metadata_success(self):
        """Test successful metadata validation."""
        metadata = self.manager.create_metadata(asset="MES")
        is_valid, error = self.manager.validate_metadata(metadata)
        
        self.assertTrue(is_valid)
        self.assertIsNone(error)
    
    def test_validate_metadata_missing_required(self):
        """Test validation with missing required fields."""
        metadata = {"some_field": "value"}
        is_valid, error = self.manager.validate_metadata(
            metadata,
            required_keys=["required_field"]
        )
        
        self.assertFalse(is_valid)
        self.assertIn("Missing required metadata keys", error)
    
    def test_validate_metadata_invalid_timestamp(self):
        """Test validation with invalid timestamp."""
        metadata = {
            self.manager.CREATION_TIMESTAMP: "invalid-timestamp"
        }
        is_valid, error = self.manager.validate_metadata(metadata)
        
        self.assertFalse(is_valid)
    
    def test_merge_metadata(self):
        """Test metadata merging."""
        base = self.manager.create_metadata(asset="MES", version="v1")
        updates = {"changelog": "Updated schema"}
        
        merged = self.manager.merge_metadata(base, updates)
        
        self.assertEqual(merged["asset"], "MES")
        self.assertEqual(merged["changelog"], "Updated schema")
        self.assertIn("creation_timestamp", merged)
    
    def test_merge_preserves_timestamp(self):
        """Test that merge preserves creation timestamp."""
        original_time = "2024-01-01T12:00:00"
        base = {
            self.manager.CREATION_TIMESTAMP: original_time,
            "asset": "MES"
        }
        updates = {
            self.manager.CREATION_TIMESTAMP: "2024-01-02T12:00:00",
            "version": "v2"
        }
        
        merged = self.manager.merge_metadata(
            base,
            updates,
            preserve_creation_timestamp=True
        )
        
        self.assertEqual(
            merged[self.manager.CREATION_TIMESTAMP],
            original_time
        )


if __name__ == "__main__":
    unittest.main()
