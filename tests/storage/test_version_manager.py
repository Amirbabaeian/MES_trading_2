"""
Comprehensive test suite for version management system.

Tests cover:
- Semantic versioning
- Version metadata persistence
- Version manager CRUD operations
- Tagging system
- Version freezing
- Version comparison and compatibility
- Validation checks
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

from src.storage.version_metadata import (
    SemanticVersion,
    DataRange,
    FileInfo,
    SchemaColumn,
    SchemaInfo,
    DataQuality,
    Lineage,
    VersionMetadata,
    ChangelogEntry,
    MetadataPersistence,
)
from src.storage.version_manager import VersionManager
from src.storage.version_utils import (
    VersionComparison,
    VersionValidator,
    CompatibilityChecker,
    VersionMigration,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_schema() -> SchemaInfo:
    """Create sample schema."""
    return SchemaInfo(
        schema_version="1.0.0",
        columns=[
            SchemaColumn("timestamp", "timestamp[ns]", False, "Market timestamp"),
            SchemaColumn("open", "double", False, "Opening price"),
            SchemaColumn("high", "double", False, "High price"),
            SchemaColumn("low", "double", False, "Low price"),
            SchemaColumn("close", "double", False, "Closing price"),
            SchemaColumn("volume", "int64", False, "Trading volume"),
        ],
    )


@pytest.fixture
def sample_data_range() -> DataRange:
    """Create sample data range."""
    return DataRange(
        start_timestamp=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        end_timestamp=datetime(2024, 1, 31, 23, 59, 59, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_metadata(sample_schema, sample_data_range) -> VersionMetadata:
    """Create sample version metadata."""
    return VersionMetadata(
        layer="cleaned",
        version=SemanticVersion(1, 0, 0),
        asset_or_feature_set="MES",
        creation_timestamp=datetime.now(timezone.utc),
        data_range=sample_data_range,
        record_count=10000,
        schema_info=sample_schema,
        files=[
            FileInfo("ohlcv.parquet", 1024*1024, "parquet", "abc123def456", 10000),
        ],
        creator="test_user",
        environment="development",
    )


@pytest.fixture
def version_manager(temp_dir) -> VersionManager:
    """Create version manager."""
    return VersionManager(temp_dir / "cleaned", layer="cleaned")


# ============================================================================
# Tests: Semantic Versioning
# ============================================================================

class TestSemanticVersion:
    """Tests for SemanticVersion."""
    
    def test_create_version(self):
        """Test creating semantic version."""
        version = SemanticVersion(1, 2, 3)
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert str(version) == "v1.2.3"
    
    def test_parse_version(self):
        """Test parsing semantic version from string."""
        version = SemanticVersion.parse("v1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
    
    def test_parse_version_without_v(self):
        """Test parsing semantic version without 'v' prefix."""
        version = SemanticVersion.parse("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
    
    def test_parse_invalid_version(self):
        """Test parsing invalid version raises error."""
        with pytest.raises(ValueError):
            SemanticVersion.parse("invalid")
    
    def test_version_comparison(self):
        """Test version comparison operators."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 0, 1)
        v3 = SemanticVersion(2, 0, 0)
        
        assert v1 < v2 < v3
        assert v3 > v2 > v1
        assert v1 == SemanticVersion(1, 0, 0)
        assert v1 <= v2
        assert v3 >= v2
    
    def test_increment_versions(self):
        """Test incrementing version components."""
        v = SemanticVersion(1, 2, 3)
        
        assert str(v.increment_major()) == "v2.0.0"
        assert str(v.increment_minor()) == "v1.3.0"
        assert str(v.increment_patch()) == "v1.2.4"
    
    def test_version_hash(self):
        """Test version can be used in sets/dicts."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 0, 0)
        v3 = SemanticVersion(1, 0, 1)
        
        version_set = {v1, v2, v3}
        assert len(version_set) == 2  # v1 and v2 are equal


# ============================================================================
# Tests: Metadata Persistence
# ============================================================================

class TestMetadataPersistence:
    """Tests for metadata persistence."""
    
    def test_save_and_load_metadata(self, temp_dir, sample_metadata):
        """Test saving and loading metadata."""
        MetadataPersistence.save(sample_metadata, temp_dir)
        
        loaded = MetadataPersistence.load(temp_dir)
        
        assert loaded.layer == sample_metadata.layer
        assert loaded.version == sample_metadata.version
        assert loaded.asset_or_feature_set == sample_metadata.asset_or_feature_set
        assert loaded.record_count == sample_metadata.record_count
    
    def test_metadata_not_found(self, temp_dir):
        """Test loading non-existent metadata raises error."""
        with pytest.raises(FileNotFoundError):
            MetadataPersistence.load(temp_dir)
    
    def test_metadata_exists(self, temp_dir, sample_metadata):
        """Test checking if metadata exists."""
        assert not MetadataPersistence.exists(temp_dir)
        
        MetadataPersistence.save(sample_metadata, temp_dir)
        
        assert MetadataPersistence.exists(temp_dir)
    
    def test_metadata_json_format(self, temp_dir, sample_metadata):
        """Test saved metadata is valid JSON."""
        MetadataPersistence.save(sample_metadata, temp_dir)
        
        metadata_file = temp_dir / "metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file) as f:
            data = json.load(f)
        
        assert data["layer"] == "cleaned"
        assert "version" in data
        assert "schema_info" in data


# ============================================================================
# Tests: Version Manager CRUD
# ============================================================================

class TestVersionManagerCRUD:
    """Tests for version manager CRUD operations."""
    
    def test_create_version(
        self,
        version_manager,
        sample_schema,
        sample_data_range,
    ):
        """Test creating a version."""
        metadata = version_manager.create_version(
            asset_or_feature_set="MES",
            version=SemanticVersion(1, 0, 0),
            data_range=sample_data_range,
            record_count=10000,
            schema_info=sample_schema,
            creator="test_user",
        )
        
        assert metadata.asset_or_feature_set == "MES"
        assert metadata.version == SemanticVersion(1, 0, 0)
        assert len(metadata.changelog) == 1
        assert metadata.changelog[0].change_type == "creation"
    
    def test_create_duplicate_version_raises_error(
        self,
        version_manager,
        sample_schema,
        sample_data_range,
    ):
        """Test creating duplicate version raises error."""
        version_manager.create_version(
            asset_or_feature_set="MES",
            version=SemanticVersion(1, 0, 0),
            data_range=sample_data_range,
            record_count=10000,
            schema_info=sample_schema,
        )
        
        with pytest.raises(ValueError):
            version_manager.create_version(
                asset_or_feature_set="MES",
                version=SemanticVersion(1, 0, 0),
                data_range=sample_data_range,
                record_count=10000,
                schema_info=sample_schema,
            )
    
    def test_get_version(self, version_manager, sample_schema, sample_data_range):
        """Test retrieving a version."""
        version_manager.create_version(
            asset_or_feature_set="MES",
            version=SemanticVersion(1, 0, 0),
            data_range=sample_data_range,
            record_count=10000,
            schema_info=sample_schema,
        )
        
        metadata = version_manager.get_version("MES", SemanticVersion(1, 0, 0))
        assert metadata.asset_or_feature_set == "MES"
        assert metadata.version == SemanticVersion(1, 0, 0)
    
    def test_get_nonexistent_version_raises_error(self, version_manager):
        """Test getting non-existent version raises error."""
        with pytest.raises(FileNotFoundError):
            version_manager.get_version("MES", SemanticVersion(1, 0, 0))
    
    def test_get_latest_version(self, version_manager, sample_schema, sample_data_range):
        """Test getting latest version."""
        # Create multiple versions
        for major in range(1, 4):
            version_manager.create_version(
                asset_or_feature_set="MES",
                version=SemanticVersion(major, 0, 0),
                data_range=sample_data_range,
                record_count=10000,
                schema_info=sample_schema,
            )
        
        latest = version_manager.get_latest_version("MES")
        assert latest.version == SemanticVersion(3, 0, 0)
    
    def test_list_versions(self, version_manager, sample_schema, sample_data_range):
        """Test listing all versions."""
        for asset in ["MES", "ES"]:
            for major in range(1, 3):
                version_manager.create_version(
                    asset_or_feature_set=asset,
                    version=SemanticVersion(major, 0, 0),
                    data_range=sample_data_range,
                    record_count=10000,
                    schema_info=sample_schema,
                )
        
        versions = version_manager.list_versions()
        assert len(versions) == 4
        
        # Check sorting
        for i in range(len(versions) - 1):
            assert (
                (versions[i].asset_or_feature_set, versions[i].version) <=
                (versions[i+1].asset_or_feature_set, versions[i+1].version)
            )
    
    def test_list_versions_for_asset(self, version_manager, sample_schema, sample_data_range):
        """Test listing versions for specific asset."""
        for asset in ["MES", "ES"]:
            for major in range(1, 3):
                version_manager.create_version(
                    asset_or_feature_set=asset,
                    version=SemanticVersion(major, 0, 0),
                    data_range=sample_data_range,
                    record_count=10000,
                    schema_info=sample_schema,
                )
        
        mes_versions = version_manager.list_versions("MES")
        assert len(mes_versions) == 2
        assert all(v.asset_or_feature_set == "MES" for v in mes_versions)
    
    def test_version_exists(self, version_manager, sample_schema, sample_data_range):
        """Test checking version existence."""
        assert not version_manager.version_exists("MES", SemanticVersion(1, 0, 0))
        
        version_manager.create_version(
            asset_or_feature_set="MES",
            version=SemanticVersion(1, 0, 0),
            data_range=sample_data_range,
            record_count=10000,
            schema_info=sample_schema,
        )
        
        assert version_manager.version_exists("MES", SemanticVersion(1, 0, 0))
    
    def test_delete_version(self, version_manager, sample_schema, sample_data_range):
        """Test deleting a version."""
        version_manager.create_version(
            asset_or_feature_set="MES",
            version=SemanticVersion(1, 0, 0),
            data_range=sample_data_range,
            record_count=10000,
            schema_info=sample_schema,
        )
        
        assert version_manager.version_exists("MES", SemanticVersion(1, 0, 0))
        
        version_manager.delete_version("MES", SemanticVersion(1, 0, 0))
        
        assert not version_manager.version_exists("MES", SemanticVersion(1, 0, 0))
    
    def test_delete_frozen_version_raises_error(
        self,
        version_manager,
        sample_schema,
        sample_data_range,
    ):
        """Test deleting frozen version raises error."""
        version_manager.create_version(
            asset_or_feature_set="MES",
            version=SemanticVersion(1, 0, 0),
            data_range=sample_data_range,
            record_count=10000,
            schema_info=sample_schema,
        )
        
        version_manager.freeze_version("MES", SemanticVersion(1, 0, 0))
        
        with pytest.raises(ValueError):
            version_manager.delete_version("MES", SemanticVersion(1, 0, 0))
    
    def test_delete_frozen_version_with_force(
        self,
        version_manager,
        sample_schema,
        sample_data_range,
    ):
        """Test deleting frozen version with force=True."""
        version_manager.create_version(
            asset_or_feature_set="MES",
            version=SemanticVersion(1, 0, 0),
            data_range=sample_data_range,
            record_count=10000,
            schema_info=sample_schema,
        )
        
        version_manager.freeze_version("MES", SemanticVersion(1, 0, 0))
        version_manager.delete_version("MES", SemanticVersion(1, 0, 0), force=True)
        
        assert not version_manager.version_exists("MES", SemanticVersion(1, 0, 0))


# ============================================================================
# Tests: Tagging
# ============================================================================

class TestVersionTagging:
    """Tests for version tagging."""
    
    def test_add_tag(self, version_manager, sample_schema, sample_data_range):
        """Test adding a tag."""
        version_manager.create_version(
            asset_or_feature_set="MES",
            version=SemanticVersion(1, 0, 0),
            data_range=sample_data_range,
            record_count=10000,
            schema_info=sample_schema,
        )
        
        version_manager.add_tag("MES", SemanticVersion(1, 0, 0), "stable")
        
        metadata = version_manager.get_version("MES", SemanticVersion(1, 0, 0))
        assert "stable" in metadata.tags
    
    def test_remove_tag(self, version_manager, sample_schema, sample_data_range):
        """Test removing a tag."""
        version_manager.create_version(
            asset_or_feature_set="MES",
            version=SemanticVersion(1, 0, 0),
            data_range=sample_data_range,
            record_count=10000,
            schema_info=sample_schema,
            tags=["stable", "production"],
        )
        
        version_manager.remove_tag("MES", SemanticVersion(1, 0, 0), "stable")
        
        metadata = version_manager.get_version("MES", SemanticVersion(1, 0, 0))
        assert "stable" not in metadata.tags
        assert "production" in metadata.tags
    
    def test_get_versions_by_tag(self, version_manager, sample_schema, sample_data_range):
        """Test finding versions by tag."""
        for major in range(1, 4):
            metadata = version_manager.create_version(
                asset_or_feature_set="MES",
                version=SemanticVersion(major, 0, 0),
                data_range=sample_data_range,
                record_count=10000,
                schema_info=sample_schema,
            )
            
            if major <= 2:
                version_manager.add_tag("MES", SemanticVersion(major, 0, 0), "production")
        
        tagged = version_manager.get_versions_by_tag("production")
        assert len(tagged) == 2


# ============================================================================
# Tests: Freezing
# ============================================================================

class TestVersionFreezing:
    """Tests for version freezing."""
    
    def test_freeze_version(self, version_manager, sample_schema, sample_data_range):
        """Test freezing a version."""
        version_manager.create_version(
            asset_or_feature_set="MES",
            version=SemanticVersion(1, 0, 0),
            data_range=sample_data_range,
            record_count=10000,
            schema_info=sample_schema,
        )
        
        version_manager.freeze_version("MES", SemanticVersion(1, 0, 0))
        
        assert version_manager.is_frozen("MES", SemanticVersion(1, 0, 0))
    
    def test_unfreeze_version(self, version_manager, sample_schema, sample_data_range):
        """Test unfreezing a version."""
        version_manager.create_version(
            asset_or_feature_set="MES",
            version=SemanticVersion(1, 0, 0),
            data_range=sample_data_range,
            record_count=10000,
            schema_info=sample_schema,
        )
        
        version_manager.freeze_version("MES", SemanticVersion(1, 0, 0))
        version_manager.unfreeze_version("MES", SemanticVersion(1, 0, 0))
        
        assert not version_manager.is_frozen("MES", SemanticVersion(1, 0, 0))
    
    def test_frozen_version_has_timestamp(self, version_manager, sample_schema, sample_data_range):
        """Test frozen version records freeze timestamp."""
        version_manager.create_version(
            asset_or_feature_set="MES",
            version=SemanticVersion(1, 0, 0),
            data_range=sample_data_range,
            record_count=10000,
            schema_info=sample_schema,
        )
        
        version_manager.freeze_version("MES", SemanticVersion(1, 0, 0))
        
        metadata = version_manager.get_version("MES", SemanticVersion(1, 0, 0))
        assert metadata.freeze_timestamp is not None


# ============================================================================
# Tests: Deprecation
# ============================================================================

class TestVersionDeprecation:
    """Tests for version deprecation."""
    
    def test_deprecate_version(self, version_manager, sample_schema, sample_data_range):
        """Test deprecating a version."""
        version_manager.create_version(
            asset_or_feature_set="MES",
            version=SemanticVersion(1, 0, 0),
            data_range=sample_data_range,
            record_count=10000,
            schema_info=sample_schema,
        )
        
        version_manager.deprecate_version("MES", SemanticVersion(1, 0, 0))
        
        metadata = version_manager.get_version("MES", SemanticVersion(1, 0, 0))
        assert "deprecated" in metadata.tags
    
    def test_get_active_versions(self, version_manager, sample_schema, sample_data_range):
        """Test getting active versions."""
        for major in range(1, 4):
            version_manager.create_version(
                asset_or_feature_set="MES",
                version=SemanticVersion(major, 0, 0),
                data_range=sample_data_range,
                record_count=10000,
                schema_info=sample_schema,
            )
        
        version_manager.deprecate_version("MES", SemanticVersion(1, 0, 0))
        
        active = version_manager.get_active_versions("MES")
        assert len(active) == 2
        assert all("deprecated" not in v.tags for v in active)


# ============================================================================
# Tests: Version Comparison
# ============================================================================

class TestVersionComparison:
    """Tests for version comparison."""
    
    def test_compare_schemas_identical(self, sample_metadata):
        """Test comparing identical schemas."""
        v2 = VersionMetadata(
            layer="cleaned",
            version=SemanticVersion(1, 1, 0),
            asset_or_feature_set="MES",
            creation_timestamp=datetime.now(timezone.utc),
            data_range=sample_metadata.data_range,
            record_count=10000,
            schema_info=sample_metadata.schema_info,
        )
        
        result = VersionComparison.compare_schemas(sample_metadata, v2)
        
        assert result["added_columns"] == []
        assert result["removed_columns"] == []
        assert result["modified_columns"] == []
        assert result["schema_compatible"] is True
    
    def test_compare_schemas_with_added_column(self, sample_metadata):
        """Test comparing schemas with added column."""
        new_schema = SchemaInfo(
            schema_version="1.1.0",
            columns=sample_metadata.schema_info.columns + [
                SchemaColumn("vwap", "double", True, "Volume weighted average price"),
            ],
        )
        
        v2 = VersionMetadata(
            layer="cleaned",
            version=SemanticVersion(1, 1, 0),
            asset_or_feature_set="MES",
            creation_timestamp=datetime.now(timezone.utc),
            data_range=sample_metadata.data_range,
            record_count=10000,
            schema_info=new_schema,
        )
        
        result = VersionComparison.compare_schemas(sample_metadata, v2)
        
        assert "vwap" in result["added_columns"]
        assert result["schema_compatible"] is True  # Adding optional column is compatible
    
    def test_compare_schemas_with_removed_column(self, sample_metadata):
        """Test comparing schemas with removed column."""
        new_schema = SchemaInfo(
            schema_version="1.1.0",
            columns=sample_metadata.schema_info.columns[:-1],  # Remove last column
        )
        
        v2 = VersionMetadata(
            layer="cleaned",
            version=SemanticVersion(1, 1, 0),
            asset_or_feature_set="MES",
            creation_timestamp=datetime.now(timezone.utc),
            data_range=sample_metadata.data_range,
            record_count=10000,
            schema_info=new_schema,
        )
        
        result = VersionComparison.compare_schemas(sample_metadata, v2)
        
        assert "volume" in result["removed_columns"]
        assert result["schema_compatible"] is False


# ============================================================================
# Tests: Version Validation
# ============================================================================

class TestVersionValidator:
    """Tests for version validation."""
    
    def test_is_complete_valid(self, sample_metadata):
        """Test checking complete valid metadata."""
        assert VersionValidator.is_complete(sample_metadata) is True
    
    def test_is_complete_missing_fields(self, sample_metadata):
        """Test checking incomplete metadata."""
        sample_metadata.schema_info.columns = []
        assert VersionValidator.is_complete(sample_metadata) is False
    
    def test_validate_integrity_valid(self, sample_metadata):
        """Test integrity validation of valid metadata."""
        result = VersionValidator.validate_integrity(sample_metadata)
        
        assert result["is_valid"] is True
        assert len(result["checks_passed"]) > 0
        assert len(result["checks_failed"]) == 0


# ============================================================================
# Tests: Compatibility Checking
# ============================================================================

class TestCompatibilityChecker:
    """Tests for compatibility checking."""
    
    def test_check_version_match_exact(self):
        """Test exact version matching."""
        metadata = VersionMetadata(
            layer="cleaned",
            version=SemanticVersion(1, 0, 0),
            asset_or_feature_set="MES",
            creation_timestamp=datetime.now(timezone.utc),
            data_range=DataRange(
                start_timestamp=datetime.now(timezone.utc),
                end_timestamp=datetime.now(timezone.utc),
            ),
            record_count=0,
            schema_info=SchemaInfo("1.0.0"),
        )
        
        assert CompatibilityChecker.check_version_match(metadata, "v1.0.0") is True
        assert CompatibilityChecker.check_version_match(metadata, "v1.0.1") is False
    
    def test_check_version_match_wildcard(self):
        """Test wildcard version matching."""
        metadata = VersionMetadata(
            layer="cleaned",
            version=SemanticVersion(1, 5, 3),
            asset_or_feature_set="MES",
            creation_timestamp=datetime.now(timezone.utc),
            data_range=DataRange(
                start_timestamp=datetime.now(timezone.utc),
                end_timestamp=datetime.now(timezone.utc),
            ),
            record_count=0,
            schema_info=SchemaInfo("1.0.0"),
        )
        
        assert CompatibilityChecker.check_version_match(metadata, "v1.x") is True
        assert CompatibilityChecker.check_version_match(metadata, "v2.x") is False
    
    def test_check_version_match_latest(self):
        """Test 'latest' version matching."""
        metadata = VersionMetadata(
            layer="cleaned",
            version=SemanticVersion(1, 0, 0),
            asset_or_feature_set="MES",
            creation_timestamp=datetime.now(timezone.utc),
            data_range=DataRange(
                start_timestamp=datetime.now(timezone.utc),
                end_timestamp=datetime.now(timezone.utc),
            ),
            record_count=0,
            schema_info=SchemaInfo("1.0.0"),
        )
        
        assert CompatibilityChecker.check_version_match(metadata, "latest") is True
