"""
Cloud storage synchronization utilities for hybrid architecture.

Provides:
- Abstract interface for cloud storage operations
- Bidirectional sync with conflict detection
- Incremental transfers with state tracking
- Implementations for AWS S3, GCP GCS, and Azure Blob
- Configuration management and CLI utilities
"""

from .cloud_sync import (
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
from .sync_state import (
    SyncStateDatabase,
    SyncFileRecord,
    FileState,
)
from .config import (
    StorageConfig,
    SyncConfig,
    CloudStorageFactory,
    ConfigManager,
    SyncEngineBuilder,
)
from .version_metadata import (
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
from .version_manager import VersionManager
from .version_utils import (
    VersionComparison,
    VersionValidator,
    CompatibilityChecker,
    VersionMigration,
)
from .providers.s3 import S3CloudStorage
from .providers.gcs import GCSCloudStorage
from .providers.azure import AzureCloudStorage

__all__ = [
    # Cloud sync core classes
    "CloudStorage",
    "CloudSyncEngine",
    # Cloud sync data classes
    "FileMetadata",
    "SyncOperation",
    "SyncConflict",
    "SyncProgress",
    # Cloud sync enums
    "SyncDirection",
    "SyncStatus",
    "ConflictStrategy",
    # State management
    "SyncStateDatabase",
    "SyncFileRecord",
    "FileState",
    # Configuration
    "StorageConfig",
    "SyncConfig",
    "CloudStorageFactory",
    "ConfigManager",
    "SyncEngineBuilder",
    # Version management metadata
    "SemanticVersion",
    "DataRange",
    "FileInfo",
    "SchemaColumn",
    "SchemaInfo",
    "DataQuality",
    "Lineage",
    "VersionMetadata",
    "ChangelogEntry",
    "MetadataPersistence",
    # Version management core
    "VersionManager",
    # Version utilities
    "VersionComparison",
    "VersionValidator",
    "CompatibilityChecker",
    "VersionMigration",
    # Providers
    "S3CloudStorage",
    "GCSCloudStorage",
    "AzureCloudStorage",
]
