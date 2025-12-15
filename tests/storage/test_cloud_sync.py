"""
Comprehensive test suite for cloud storage synchronization utilities.

Tests cover:
- Abstract cloud storage interface
- Bidirectional sync logic
- Conflict detection and resolution
- Error handling and retry logic
- Progress reporting
- Sync state tracking
- Configuration management
"""

import pytest
import tempfile
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, MagicMock, patch, call
import hashlib

from src.storage.cloud_sync import (
    CloudStorage,
    CloudSyncEngine,
    FileMetadata,
    SyncOperation,
    SyncConflict,
    SyncProgress,
    SyncDirection,
    SyncStatus,
    ConflictStrategy,
)
from src.storage.sync_state import (
    SyncStateDatabase,
    SyncFileRecord,
    FileState,
)
from src.storage.config import (
    StorageConfig,
    SyncConfig,
    CloudStorageFactory,
    ConfigManager,
    SyncEngineBuilder,
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
def sample_files(temp_dir):
    """Create sample files for testing."""
    files = []
    
    # Create several test files
    for i in range(5):
        file_path = temp_dir / f"file_{i}.txt"
        file_path.write_text(f"Test content {i}\n" * 100)
        files.append(file_path)
    
    # Create subdirectory with files
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    for i in range(3):
        file_path = subdir / f"subfile_{i}.txt"
        file_path.write_text(f"Subdir content {i}\n" * 50)
        files.append(file_path)
    
    return files


@pytest.fixture
def mock_cloud_storage():
    """Create a mock cloud storage provider."""
    mock = Mock(spec=CloudStorage)
    mock.connect = Mock()
    mock.disconnect = Mock()
    mock.upload_file = Mock(return_value="mock_etag_123")
    mock.download_file = Mock(return_value="mock_checksum_abc")
    mock.delete_file = Mock()
    mock.list_files = Mock(return_value=[])
    mock.file_exists = Mock(return_value=True)
    mock.get_file_metadata = Mock(return_value=None)
    return mock


@pytest.fixture
def sync_engine(temp_dir, mock_cloud_storage):
    """Create a CloudSyncEngine with mock storage."""
    engine = CloudSyncEngine(
        cloud_storage=mock_cloud_storage,
        sync_state_db=temp_dir / "sync_state.db",
        conflict_strategy=ConflictStrategy.TIMESTAMP_WINS,
        max_workers=2,
        retry_attempts=2,
        retry_delay_seconds=0.1,
    )
    return engine


@pytest.fixture
def storage_config():
    """Create a storage configuration."""
    return StorageConfig(
        provider="s3",
        bucket="test-bucket",
        region="us-east-1",
        storage_class="STANDARD",
    )


@pytest.fixture
def sync_config(temp_dir, storage_config):
    """Create a sync configuration."""
    return SyncConfig(
        storage_config=storage_config,
        sync_state_db=temp_dir / "sync_state.db",
        conflict_strategy=ConflictStrategy.TIMESTAMP_WINS,
        max_workers=4,
        retry_attempts=3,
        retry_delay_seconds=5,
    )


# ============================================================================
# Tests: File Metadata
# ============================================================================

class TestFileMetadata:
    """Tests for FileMetadata."""
    
    def test_file_metadata_creation(self):
        """Test FileMetadata creation."""
        now = datetime.now(timezone.utc)
        metadata = FileMetadata(
            path="s3://bucket/key",
            size=1024,
            modified_time=now,
            checksum="abc123",
            storage_class="STANDARD",
        )
        
        assert metadata.path == "s3://bucket/key"
        assert metadata.size == 1024
        assert metadata.modified_time == now
        assert metadata.checksum == "abc123"
    
    def test_file_metadata_to_dict(self):
        """Test FileMetadata to_dict conversion."""
        now = datetime.now(timezone.utc)
        metadata = FileMetadata(
            path="test.txt",
            size=512,
            modified_time=now,
            checksum="xyz789",
        )
        
        d = metadata.to_dict()
        assert d["path"] == "test.txt"
        assert d["size"] == 512
        assert d["checksum"] == "xyz789"


# ============================================================================
# Tests: Sync Operations
# ============================================================================

class TestSyncOperation:
    """Tests for SyncOperation."""
    
    def test_sync_operation_creation(self):
        """Test SyncOperation creation."""
        op = SyncOperation(
            operation_id="op_001",
            source_path="/local/file.txt",
            dest_path="s3://bucket/file.txt",
            direction=SyncDirection.UPLOAD,
            status=SyncStatus.COMPLETED,
            file_size=1024,
            checksum="abc123",
        )
        
        assert op.operation_id == "op_001"
        assert op.direction == SyncDirection.UPLOAD
        assert op.status == SyncStatus.COMPLETED
    
    def test_sync_operation_to_dict(self):
        """Test SyncOperation to_dict conversion."""
        op = SyncOperation(
            operation_id="op_002",
            source_path="/local/file.txt",
            dest_path="s3://bucket/file.txt",
            direction=SyncDirection.DOWNLOAD,
            status=SyncStatus.FAILED,
            file_size=2048,
            error_message="Connection timeout",
        )
        
        d = op.to_dict()
        assert d["operation_id"] == "op_002"
        assert d["status"] == "failed"
        assert d["error_message"] == "Connection timeout"


# ============================================================================
# Tests: Sync Progress
# ============================================================================

class TestSyncProgress:
    """Tests for SyncProgress."""
    
    def test_progress_percent_calculation(self):
        """Test progress percentage calculation."""
        progress = SyncProgress(total_files=10, completed_files=3)
        assert progress.progress_percent == 30.0
    
    def test_bytes_percent_calculation(self):
        """Test bytes percentage calculation."""
        progress = SyncProgress(total_bytes=1000, transferred_bytes=250)
        assert progress.bytes_percent == 25.0
    
    def test_transfer_speed_calculation(self):
        """Test transfer speed calculation."""
        progress = SyncProgress(
            total_bytes=1024*1024,  # 1 MB
            transferred_bytes=1024*1024,
        )
        progress.start_time = datetime.now(timezone.utc) - timedelta(seconds=1)
        
        # Speed should be ~1 MB/s
        assert 0.9 < progress.transfer_speed_mbps < 1.1


# ============================================================================
# Tests: Upload Directory
# ============================================================================

class TestUploadDirectory:
    """Tests for upload_directory functionality."""
    
    def test_upload_directory_basic(self, sync_engine, temp_dir, sample_files):
        """Test basic directory upload."""
        mock_storage = sync_engine.cloud_storage
        mock_storage.upload_file = Mock(return_value="etag_123")
        
        progress = sync_engine.upload_directory(
            local_dir=temp_dir,
            cloud_prefix="data/raw/",
        )
        
        assert progress.completed_files == len(sample_files)
        assert progress.failed_files == 0
        assert mock_storage.upload_file.call_count == len(sample_files)
    
    def test_upload_with_pattern(self, sync_engine, temp_dir):
        """Test upload with file pattern."""
        # Create files with different extensions
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2")
        (temp_dir / "file3.csv").write_text("content3")
        
        mock_storage = sync_engine.cloud_storage
        mock_storage.upload_file = Mock(return_value="etag_123")
        
        # Only upload .txt files
        progress = sync_engine.upload_directory(
            local_dir=temp_dir,
            cloud_prefix="data/",
            pattern="**/*.txt",
        )
        
        assert progress.completed_files == 2
        assert mock_storage.upload_file.call_count == 2
    
    def test_upload_with_exclude_patterns(self, sync_engine, temp_dir):
        """Test upload with exclude patterns."""
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.txt").write_text("content2")
        (temp_dir / "temp.txt").write_text("temp")
        
        mock_storage = sync_engine.cloud_storage
        mock_storage.upload_file = Mock(return_value="etag_123")
        
        progress = sync_engine.upload_directory(
            local_dir=temp_dir,
            cloud_prefix="data/",
            exclude_patterns=["**/temp*"],
        )
        
        # Should exclude temp.txt
        assert progress.completed_files == 2


# ============================================================================
# Tests: Download Directory
# ============================================================================

class TestDownloadDirectory:
    """Tests for download_directory functionality."""
    
    def test_download_directory_basic(self, sync_engine, temp_dir):
        """Test basic directory download."""
        mock_storage = sync_engine.cloud_storage
        
        # Mock cloud files
        cloud_files = [
            FileMetadata("data/file1.txt", 100, datetime.now(timezone.utc)),
            FileMetadata("data/file2.txt", 200, datetime.now(timezone.utc)),
        ]
        mock_storage.list_files = Mock(return_value=cloud_files)
        mock_storage.download_file = Mock(return_value="checksum_123")
        
        # Create local files for download
        def mock_download(cloud_path, local_path):
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text("downloaded content")
            return "checksum_123"
        
        mock_storage.download_file = Mock(side_effect=mock_download)
        
        progress = sync_engine.download_directory(
            cloud_prefix="data/",
            local_dir=temp_dir,
        )
        
        assert progress.completed_files == 2
        assert progress.failed_files == 0


# ============================================================================
# Tests: Sync Conflicts
# ============================================================================

class TestSyncConflicts:
    """Tests for conflict detection and resolution."""
    
    def test_conflict_creation(self):
        """Test SyncConflict creation."""
        local_meta = FileMetadata("file.txt", 100, datetime.now(timezone.utc))
        cloud_meta = FileMetadata("file.txt", 150, datetime.now(timezone.utc))
        
        conflict = SyncConflict(
            file_path="file.txt",
            local_metadata=local_meta,
            cloud_metadata=cloud_meta,
            conflict_reason="both_modified",
        )
        
        assert conflict.file_path == "file.txt"
        assert conflict.conflict_reason == "both_modified"
    
    def test_conflict_resolution_timestamp_wins(self, sync_engine):
        """Test conflict resolution with timestamp strategy."""
        now = datetime.now(timezone.utc)
        local_meta = FileMetadata("file.txt", 100, now - timedelta(hours=1))
        cloud_meta = FileMetadata("file.txt", 150, now)
        
        conflict = SyncConflict(
            file_path="file.txt",
            local_metadata=local_meta,
            cloud_metadata=cloud_meta,
            conflict_reason="both_modified",
        )
        
        sync_engine._resolve_conflict(conflict)
        assert conflict.resolution == ConflictStrategy.CLOUD_WINS
    
    def test_conflict_resolution_local_wins(self):
        """Test LOCAL_WINS conflict strategy."""
        engine = CloudSyncEngine(
            cloud_storage=Mock(),
            sync_state_db=Path(".sync_state.db"),
            conflict_strategy=ConflictStrategy.LOCAL_WINS,
        )
        
        conflict = SyncConflict(
            file_path="file.txt",
            local_metadata=Mock(),
            cloud_metadata=Mock(),
            conflict_reason="both_modified",
        )
        
        engine._resolve_conflict(conflict)
        assert conflict.resolution == ConflictStrategy.LOCAL_WINS


# ============================================================================
# Tests: Sync State Database
# ============================================================================

class TestSyncStateDatabase:
    """Tests for SyncStateDatabase."""
    
    def test_database_creation(self, temp_dir):
        """Test database creation."""
        db = SyncStateDatabase(temp_dir / "sync_state.db")
        assert (temp_dir / "sync_state.db").exists()
    
    def test_record_file_sync(self, temp_dir):
        """Test recording a file sync."""
        db = SyncStateDatabase(temp_dir / "sync_state.db")
        
        record = SyncFileRecord(
            local_path="/local/file.txt",
            cloud_path="s3://bucket/file.txt",
            file_size=1024,
            local_checksum="abc123",
            cloud_checksum="abc123",
            state=FileState.COMPLETED,
            last_sync_time=datetime.now(timezone.utc),
            sync_direction="upload",
        )
        
        db.record_file_sync(record)
        
        retrieved = db.get_file_record("/local/file.txt")
        assert retrieved is not None
        assert retrieved.file_size == 1024
        assert retrieved.state == FileState.COMPLETED
    
    def test_get_failed_records(self, temp_dir):
        """Test retrieving failed records."""
        db = SyncStateDatabase(temp_dir / "sync_state.db")
        
        # Record a failed sync
        record = SyncFileRecord(
            local_path="/local/file.txt",
            cloud_path="s3://bucket/file.txt",
            file_size=1024,
            local_checksum="abc123",
            cloud_checksum="def456",
            state=FileState.FAILED,
            last_sync_time=datetime.now(timezone.utc),
            sync_direction="upload",
            error_message="Connection timeout",
        )
        db.record_file_sync(record)
        
        failed = db.get_failed_records()
        assert len(failed) == 1
        assert failed[0].error_message == "Connection timeout"
    
    def test_log_operation(self, temp_dir):
        """Test logging an operation."""
        db = SyncStateDatabase(temp_dir / "sync_state.db")
        
        db.log_operation(
            operation="upload",
            status="success",
            local_path="/local/file.txt",
            cloud_path="s3://bucket/file.txt",
            bytes_transferred=1024,
        )
        
        # Verify by checking database directly
        with sqlite3.connect(temp_dir / "sync_state.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sync_log")
            count = cursor.fetchone()[0]
            assert count == 1
    
    def test_cleanup_old_records(self, temp_dir):
        """Test cleanup of old records."""
        db = SyncStateDatabase(temp_dir / "sync_state.db")
        
        # Log some operations
        db.log_operation("upload", "success")
        
        # Cleanup should work without errors
        deleted = db.cleanup_old_records(days_old=30)
        # May be 0 or more depending on timestamps


# ============================================================================
# Tests: Configuration
# ============================================================================

class TestStorageConfig:
    """Tests for StorageConfig."""
    
    def test_config_creation(self):
        """Test StorageConfig creation."""
        config = StorageConfig(
            provider="s3",
            bucket="test-bucket",
            region="us-west-2",
            storage_class="STANDARD_IA",
        )
        
        assert config.provider == "s3"
        assert config.bucket == "test-bucket"
        assert config.region == "us-west-2"
    
    def test_config_to_dict(self):
        """Test StorageConfig to_dict."""
        config = StorageConfig(
            provider="gcs",
            bucket="gcs-bucket",
        )
        
        d = config.to_dict()
        assert d["provider"] == "gcs"
        assert d["bucket"] == "gcs-bucket"


class TestConfigManager:
    """Tests for ConfigManager."""
    
    def test_create_storage_config_from_dict(self):
        """Test creating StorageConfig from dictionary."""
        config_dict = {
            "provider": "s3",
            "bucket": "my-bucket",
            "region": "eu-west-1",
            "storage_class": "STANDARD",
        }
        
        config = ConfigManager.create_storage_config(config_dict)
        assert config.provider == "s3"
        assert config.bucket == "my-bucket"
        assert config.region == "eu-west-1"
    
    def test_env_var_substitution(self):
        """Test environment variable substitution."""
        import os
        
        os.environ["TEST_BUCKET"] = "my-test-bucket"
        config_dict = {
            "provider": "s3",
            "bucket": "${TEST_BUCKET}",
        }
        
        config_dict = ConfigManager._substitute_env_vars(config_dict)
        assert config_dict["bucket"] == "my-test-bucket"
    
    def test_env_var_with_default(self):
        """Test environment variable with default value."""
        config_dict = {
            "provider": "s3",
            "region": "${NONEXISTENT_VAR:us-east-1}",
        }
        
        config_dict = ConfigManager._substitute_env_vars(config_dict)
        assert config_dict["region"] == "us-east-1"


class TestCloudStorageFactory:
    """Tests for CloudStorageFactory."""
    
    @patch("src.storage.config.S3CloudStorage")
    def test_create_s3_provider(self, mock_s3):
        """Test creating S3 provider."""
        config = StorageConfig(
            provider="s3",
            bucket="test-bucket",
            region="us-east-1",
        )
        
        factory = CloudStorageFactory()
        provider = factory.create(config)
        
        assert mock_s3.called
    
    def test_create_unknown_provider_raises_error(self):
        """Test that unknown provider raises error."""
        config = StorageConfig(
            provider="unknown",
            bucket="bucket",
        )
        
        with pytest.raises(ValueError, match="Unknown provider"):
            CloudStorageFactory.create(config)


# ============================================================================
# Tests: Integration
# ============================================================================

class TestIntegration:
    """Integration tests for sync engine."""
    
    def test_sync_state_persistence(self, temp_dir, mock_cloud_storage):
        """Test that sync state is persisted and loaded."""
        sync_db = temp_dir / "sync_state.db"
        
        # Create first engine and record an operation
        engine1 = CloudSyncEngine(
            cloud_storage=mock_cloud_storage,
            sync_state_db=sync_db,
        )
        
        op = SyncOperation(
            operation_id="op_001",
            source_path="/local/file.txt",
            dest_path="s3://bucket/file.txt",
            direction=SyncDirection.UPLOAD,
            status=SyncStatus.COMPLETED,
            file_size=1024,
        )
        engine1.operations.append(op)
        engine1._save_sync_state()
        
        # Create second engine and verify state was loaded
        engine2 = CloudSyncEngine(
            cloud_storage=mock_cloud_storage,
            sync_state_db=sync_db,
        )
        
        assert len(engine2.operations) > 0
        assert engine2.operations[0].operation_id == "op_001"
    
    def test_hash_computation(self, temp_dir, sync_engine):
        """Test MD5 hash computation."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        hash1 = sync_engine._compute_file_hash(test_file)
        hash2 = sync_engine._compute_file_hash(test_file)
        
        # Same file should have same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex digest length


# ============================================================================
# Tests: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and retry logic."""
    
    def test_upload_retry_on_failure(self, sync_engine, temp_dir):
        """Test retry logic on upload failure."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        # First call fails, second succeeds
        sync_engine.cloud_storage.upload_file = Mock(
            side_effect=[Exception("Network error"), "etag_123"]
        )
        
        # Should retry and succeed
        result = sync_engine._upload_with_retry(test_file, "cloud/path")
        assert result == "etag_123"
        assert sync_engine.cloud_storage.upload_file.call_count == 2
    
    def test_upload_fails_after_max_retries(self, sync_engine, temp_dir):
        """Test that upload fails after max retries."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        sync_engine.cloud_storage.upload_file = Mock(
            side_effect=Exception("Persistent error")
        )
        
        with pytest.raises(Exception, match="Persistent error"):
            sync_engine._upload_with_retry(test_file, "cloud/path")
        
        # Should have tried max_retries times
        assert sync_engine.cloud_storage.upload_file.call_count == sync_engine.retry_attempts
