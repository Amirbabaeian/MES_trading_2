"""
Comprehensive tests for feature storage system.

Tests cover:
- Feature metadata creation and persistence
- Feature writing with versioning
- Feature reading and retrieval
- Version freezing and comparison
- Data consistency validation
- Incremental updates
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import tempfile

from src.features.storage import (
    # Metadata
    ComputationParameter,
    FeatureDependency,
    FeatureSchema,
    FeatureComputationMetadata,
    FeatureStorageMetadata,
    FeatureMetadataPersistence,
    FeatureMetadataBuilder,
    # Writer
    FeatureWriter,
    VersionIncrementer,
    VersionFrozenError,
    DataConsistencyError,
    # Reader
    FeatureReader,
    FeatureCatalog,
    VersionNotFoundError,
    FeatureNotFoundError,
    # Versioning
    FeatureVersionManager,
    FeatureVersionComparator,
    FeatureVersionAnalyzer,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for feature storage."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_features_df():
    """Create sample feature DataFrame."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'ma_20': np.random.randn(100).cumsum() + 100,
        'volatility': np.random.uniform(0.01, 0.05, 100),
        'rsi': np.random.uniform(30, 70, 100),
    })
    return df


@pytest.fixture
def sample_feature_metadata():
    """Create sample feature computation metadata."""
    sma_meta = FeatureComputationMetadata(
        feature_name="ma_20",
        description="20-day simple moving average",
        computation_parameters=[
            ComputationParameter("window", 20, "int", "Window size in days")
        ],
        dependencies=[
            FeatureDependency("close", False, "Raw close price")
        ],
        schema=[
            FeatureSchema("ma_20", "float64", False, "20-day SMA values")
        ]
    )
    
    volatility_meta = FeatureComputationMetadata(
        feature_name="volatility",
        description="Historical volatility",
        computation_parameters=[
            ComputationParameter("window", 20, "int", "Window size in days"),
            ComputationParameter("method", "std", "string", "Volatility method")
        ],
        dependencies=[
            FeatureDependency("returns", True, "Daily returns")
        ],
        schema=[
            FeatureSchema("volatility", "float64", False, "Volatility values")
        ]
    )
    
    return [sma_meta, volatility_meta]


# ============================================================================
# Metadata Tests
# ============================================================================

class TestMetadata:
    """Test metadata management."""
    
    def test_computation_parameter_serialization(self):
        """Test ComputationParameter serialization."""
        param = ComputationParameter("window", 20, "int", "Window size")
        
        serialized = param.to_dict()
        assert serialized["name"] == "window"
        assert serialized["value"] == 20
        assert serialized["data_type"] == "int"
        
        deserialized = ComputationParameter.from_dict(serialized)
        assert deserialized.name == param.name
        assert deserialized.value == param.value
    
    def test_feature_dependency_creation(self):
        """Test FeatureDependency creation."""
        dep = FeatureDependency("returns", True, "Daily returns")
        
        assert dep.feature_name == "returns"
        assert dep.is_computed is True
        assert dep.description == "Daily returns"
    
    def test_feature_schema_creation(self):
        """Test FeatureSchema creation."""
        schema = FeatureSchema("ma_20", "float64", False, "Moving average")
        
        assert schema.column_name == "ma_20"
        assert schema.data_type == "float64"
        assert schema.nullable is False
    
    def test_feature_computation_metadata_building(self, sample_feature_metadata):
        """Test building feature computation metadata."""
        meta = sample_feature_metadata[0]
        
        assert meta.feature_name == "ma_20"
        assert len(meta.computation_parameters) == 1
        assert len(meta.dependencies) == 1
        assert len(meta.schema) == 1
    
    def test_metadata_builder(self):
        """Test FeatureMetadataBuilder."""
        builder = FeatureMetadataBuilder(
            version="1.0.0",
            asset="ES",
            feature_set_name="technical_indicators",
            source_data_version="1.2.3"
        )
        
        metadata = builder.set_record_count(100).build()
        
        assert metadata.version == "1.0.0"
        assert metadata.asset == "ES"
        assert metadata.record_count == 100
    
    def test_metadata_freeze(self):
        """Test freezing metadata."""
        metadata = FeatureStorageMetadata(
            version="1.0.0",
            asset="ES",
            feature_set_name="test",
            creation_timestamp=datetime.now(),
            source_data_version="1.0.0",
            start_date=datetime.now(),
            end_date=datetime.now(),
            record_count=100,
        )
        
        assert metadata.frozen is False
        metadata.freeze("user1")
        assert metadata.frozen is True
        
        metadata.unfreeze("user1")
        assert metadata.frozen is False
    
    def test_metadata_tags(self):
        """Test tag management."""
        metadata = FeatureStorageMetadata(
            version="1.0.0",
            asset="ES",
            feature_set_name="test",
            creation_timestamp=datetime.now(),
            source_data_version="1.0.0",
            start_date=datetime.now(),
            end_date=datetime.now(),
            record_count=100,
        )
        
        metadata.add_tag("production")
        assert "production" in metadata.tags
        
        metadata.remove_tag("production")
        assert "production" not in metadata.tags
    
    def test_metadata_persistence(self, temp_storage_dir):
        """Test saving and loading metadata."""
        metadata = FeatureStorageMetadata(
            version="1.0.0",
            asset="ES",
            feature_set_name="test",
            creation_timestamp=datetime(2024, 1, 1),
            source_data_version="1.0.0",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            record_count=100,
        )
        
        FeatureMetadataPersistence.save(metadata, temp_storage_dir)
        
        assert FeatureMetadataPersistence.exists(temp_storage_dir)
        
        loaded = FeatureMetadataPersistence.load(temp_storage_dir)
        assert loaded.version == metadata.version
        assert loaded.asset == metadata.asset


# ============================================================================
# Writer Tests
# ============================================================================

class TestFeatureWriter:
    """Test feature writing and versioning."""
    
    def test_writer_initialization(self, temp_storage_dir):
        """Test FeatureWriter initialization."""
        writer = FeatureWriter(
            base_path=temp_storage_dir,
            compression="snappy"
        )
        
        assert writer.base_path == temp_storage_dir
        assert writer.compression == "snappy"
    
    def test_write_features(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test writing features."""
        writer = FeatureWriter(base_path=temp_storage_dir)
        
        version_dir = writer.write_features(
            features_df=sample_features_df,
            asset="ES",
            version="1.0.0",
            feature_set_name="technical_indicators",
            source_data_version="1.0.0",
            feature_metadata_list=sample_feature_metadata,
            creator="test_user",
        )
        
        assert version_dir.exists()
        assert (version_dir / "data.parquet").exists()
        assert (version_dir / "metadata.json").exists()
    
    def test_write_features_validation(
        self,
        temp_storage_dir,
        sample_feature_metadata
    ):
        """Test feature writing validation."""
        writer = FeatureWriter(base_path=temp_storage_dir)
        
        # Test empty DataFrame
        with pytest.raises(Exception):
            writer.write_features(
                features_df=pd.DataFrame(),
                asset="ES",
                version="1.0.0",
                feature_set_name="test",
                source_data_version="1.0.0",
                feature_metadata_list=sample_feature_metadata,
            )
        
        # Test missing timestamp column
        df = pd.DataFrame({"col1": [1, 2, 3]})
        with pytest.raises(Exception):
            writer.write_features(
                features_df=df,
                asset="ES",
                version="1.0.0",
                feature_set_name="test",
                source_data_version="1.0.0",
                feature_metadata_list=sample_feature_metadata,
            )
    
    def test_freeze_version(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test version freezing."""
        writer = FeatureWriter(base_path=temp_storage_dir)
        
        # Write initial version
        writer.write_features(
            features_df=sample_features_df,
            asset="ES",
            version="1.0.0",
            feature_set_name="test",
            source_data_version="1.0.0",
            feature_metadata_list=sample_feature_metadata,
        )
        
        # Freeze version
        writer.freeze_version("ES", "1.0.0", "test_user")
        
        # Verify frozen
        version_dir = temp_storage_dir / "v1.0.0" / "ES"
        metadata = FeatureMetadataPersistence.load(version_dir)
        assert metadata.frozen is True
        
        # Try to write to frozen version
        with pytest.raises(VersionFrozenError):
            writer.write_features(
                features_df=sample_features_df,
                asset="ES",
                version="1.0.0",
                feature_set_name="test",
                source_data_version="1.0.0",
                feature_metadata_list=sample_feature_metadata,
            )
    
    def test_incremental_update(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test incremental feature updates."""
        writer = FeatureWriter(base_path=temp_storage_dir)
        
        # Write initial version
        writer.write_features(
            features_df=sample_features_df,
            asset="ES",
            version="1.0.0",
            feature_set_name="test",
            source_data_version="1.0.0",
            feature_metadata_list=sample_feature_metadata,
        )
        
        # Create new data
        new_dates = pd.date_range('2024-04-10', periods=20, freq='D')
        new_df = pd.DataFrame({
            'timestamp': new_dates,
            'ma_20': np.random.randn(20).cumsum() + 100,
            'volatility': np.random.uniform(0.01, 0.05, 20),
            'rsi': np.random.uniform(30, 70, 20),
        })
        
        # Incrementally update
        writer.write_incremental(
            new_features_df=new_df,
            asset="ES",
            version="1.0.0",
            feature_metadata_list=sample_feature_metadata,
            creator="test_user",
        )
        
        # Verify data was added
        version_dir = temp_storage_dir / "v1.0.0" / "ES"
        metadata = FeatureMetadataPersistence.load(version_dir)
        assert metadata.record_count == len(sample_features_df) + len(new_df)
    
    def test_version_incrementer(self):
        """Test version incrementing."""
        assert VersionIncrementer.increment_patch("1.0.0") == "1.0.1"
        assert VersionIncrementer.increment_minor("1.0.0") == "1.1.0"
        assert VersionIncrementer.increment_major("1.0.0") == "2.0.0"
        
        next_version = VersionIncrementer.get_next_version("1.0.0", "minor")
        assert next_version == "1.1.0"


# ============================================================================
# Reader Tests
# ============================================================================

class TestFeatureReader:
    """Test feature reading and retrieval."""
    
    def test_reader_initialization(self, temp_storage_dir):
        """Test FeatureReader initialization."""
        reader = FeatureReader(base_path=temp_storage_dir)
        assert reader.base_path == temp_storage_dir
    
    def test_load_features(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test loading features."""
        # First write features
        writer = FeatureWriter(base_path=temp_storage_dir)
        writer.write_features(
            features_df=sample_features_df,
            asset="ES",
            version="1.0.0",
            feature_set_name="technical_indicators",
            source_data_version="1.0.0",
            feature_metadata_list=sample_feature_metadata,
        )
        
        # Now read features
        reader = FeatureReader(base_path=temp_storage_dir)
        loaded_df = reader.load_features(
            asset="ES",
            version="1.0.0",
        )
        
        assert len(loaded_df) == len(sample_features_df)
        assert set(loaded_df.columns) == set(sample_features_df.columns)
    
    def test_load_features_with_date_range(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test loading features with date filtering."""
        writer = FeatureWriter(base_path=temp_storage_dir)
        writer.write_features(
            features_df=sample_features_df,
            asset="ES",
            version="1.0.0",
            feature_set_name="technical_indicators",
            source_data_version="1.0.0",
            feature_metadata_list=sample_feature_metadata,
        )
        
        reader = FeatureReader(base_path=temp_storage_dir)
        
        start_date = datetime(2024, 1, 10)
        end_date = datetime(2024, 1, 20)
        
        loaded_df = reader.load_features(
            asset="ES",
            version="1.0.0",
            start_date=start_date,
            end_date=end_date,
        )
        
        assert (loaded_df['timestamp'] >= start_date).all()
        assert (loaded_df['timestamp'] <= end_date).all()
    
    def test_load_specific_features(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test loading specific features."""
        writer = FeatureWriter(base_path=temp_storage_dir)
        writer.write_features(
            features_df=sample_features_df,
            asset="ES",
            version="1.0.0",
            feature_set_name="technical_indicators",
            source_data_version="1.0.0",
            feature_metadata_list=sample_feature_metadata,
        )
        
        reader = FeatureReader(base_path=temp_storage_dir)
        
        loaded_df = reader.load_features(
            asset="ES",
            version="1.0.0",
            feature_names=["ma_20"]
        )
        
        assert "ma_20" in loaded_df.columns
        assert "volatility" not in loaded_df.columns
        assert "timestamp" in loaded_df.columns
    
    def test_list_versions(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test listing available versions."""
        writer = FeatureWriter(base_path=temp_storage_dir)
        
        # Write multiple versions
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            writer.write_features(
                features_df=sample_features_df,
                asset="ES",
                version=version,
                feature_set_name="technical_indicators",
                source_data_version="1.0.0",
                feature_metadata_list=sample_feature_metadata,
            )
        
        reader = FeatureReader(base_path=temp_storage_dir)
        versions = reader.list_versions()
        
        assert "ES" in versions
        assert set(versions["ES"]) == {"2.0.0", "1.1.0", "1.0.0"}
    
    def test_get_latest_version(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test getting latest version."""
        writer = FeatureWriter(base_path=temp_storage_dir)
        
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            writer.write_features(
                features_df=sample_features_df,
                asset="ES",
                version=version,
                feature_set_name="technical_indicators",
                source_data_version="1.0.0",
                feature_metadata_list=sample_feature_metadata,
            )
        
        reader = FeatureReader(base_path=temp_storage_dir)
        latest = reader.get_latest_version("ES")
        
        assert latest == "2.0.0"
    
    def test_load_latest_version(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test loading from latest version."""
        writer = FeatureWriter(base_path=temp_storage_dir)
        
        for version in ["1.0.0", "2.0.0"]:
            writer.write_features(
                features_df=sample_features_df,
                asset="ES",
                version=version,
                feature_set_name="technical_indicators",
                source_data_version="1.0.0",
                feature_metadata_list=sample_feature_metadata,
            )
        
        reader = FeatureReader(base_path=temp_storage_dir)
        df, version = reader.load_latest_version("ES")
        
        assert version == "2.0.0"
        assert len(df) == len(sample_features_df)


# ============================================================================
# Versioning Tests
# ============================================================================

class TestFeatureVersioning:
    """Test version management and comparison."""
    
    def test_version_manager_freeze(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test version freezing via manager."""
        writer = FeatureWriter(base_path=temp_storage_dir)
        writer.write_features(
            features_df=sample_features_df,
            asset="ES",
            version="1.0.0",
            feature_set_name="test",
            source_data_version="1.0.0",
            feature_metadata_list=sample_feature_metadata,
        )
        
        manager = FeatureVersionManager(base_path=temp_storage_dir)
        manager.freeze_version("ES", "1.0.0", "test_user")
        
        assert manager.is_frozen("ES", "1.0.0")
    
    def test_version_comparator_schemas(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test schema comparison."""
        writer = FeatureWriter(base_path=temp_storage_dir)
        
        # Write two versions
        for version in ["1.0.0", "1.1.0"]:
            writer.write_features(
                features_df=sample_features_df,
                asset="ES",
                version=version,
                feature_set_name="test",
                source_data_version="1.0.0",
                feature_metadata_list=sample_feature_metadata,
            )
        
        comparator = FeatureVersionComparator(base_path=temp_storage_dir)
        comparison = comparator.compare_schemas("ES", "1.0.0", "1.1.0")
        
        assert comparison["asset"] == "ES"
        assert comparison["schema_compatible"] is True
    
    def test_version_analyzer_history(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test version history analysis."""
        writer = FeatureWriter(base_path=temp_storage_dir)
        
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            writer.write_features(
                features_df=sample_features_df,
                asset="ES",
                version=version,
                feature_set_name="test",
                source_data_version="1.0.0",
                feature_metadata_list=sample_feature_metadata,
            )
        
        analyzer = FeatureVersionAnalyzer(base_path=temp_storage_dir)
        history = analyzer.get_version_history("ES")
        
        assert len(history) == 3
        assert history[0]["version"] == "2.0.0"  # Latest first


# ============================================================================
# Integration Tests
# ============================================================================

class TestFeatureStorageIntegration:
    """Integration tests across multiple components."""
    
    def test_end_to_end_workflow(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test complete write-read-compare workflow."""
        # Write features
        writer = FeatureWriter(base_path=temp_storage_dir)
        
        for asset in ["ES", "MES"]:
            writer.write_features(
                features_df=sample_features_df,
                asset=asset,
                version="1.0.0",
                feature_set_name="technical_indicators",
                source_data_version="1.0.0",
                feature_metadata_list=sample_feature_metadata,
                creator="test_user",
            )
        
        # Read features
        reader = FeatureReader(base_path=temp_storage_dir)
        
        for asset in ["ES", "MES"]:
            df = reader.load_features(asset, "1.0.0")
            assert len(df) == len(sample_features_df)
        
        # List all versions
        versions = reader.list_versions()
        assert set(versions.keys()) == {"ES", "MES"}
        
        # Freeze versions
        manager = FeatureVersionManager(base_path=temp_storage_dir)
        manager.freeze_version("ES", "1.0.0", "test_user")
        
        assert manager.is_frozen("ES", "1.0.0")
    
    def test_multiple_asset_loading(
        self,
        temp_storage_dir,
        sample_features_df,
        sample_feature_metadata
    ):
        """Test loading features for multiple assets."""
        writer = FeatureWriter(base_path=temp_storage_dir)
        
        assets = ["ES", "MES", "VIX"]
        for asset in assets:
            writer.write_features(
                features_df=sample_features_df,
                asset=asset,
                version="1.0.0",
                feature_set_name="technical_indicators",
                source_data_version="1.0.0",
                feature_metadata_list=sample_feature_metadata,
            )
        
        reader = FeatureReader(base_path=temp_storage_dir)
        results = reader.load_multiple_assets(assets, "1.0.0")
        
        assert len(results) == 3
        for asset in assets:
            assert asset in results
            assert len(results[asset]) == len(sample_features_df)
