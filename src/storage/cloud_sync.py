"""
Cloud storage synchronization utilities for hybrid architecture.

Provides:
- Abstract interface for cloud storage operations (upload, download, list, delete)
- Bidirectional sync (upload/download with smart sync)
- Incremental transfers with state tracking
- Conflict detection and resolution strategies
- Progress reporting and error handling
- Retry logic for transient failures
"""

import logging
import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class ConflictStrategy(str, Enum):
    """Strategy for handling sync conflicts."""
    TIMESTAMP_WINS = "timestamp_wins"      # Newest file wins
    LOCAL_WINS = "local_wins"              # Local file always wins
    CLOUD_WINS = "cloud_wins"              # Cloud file always wins
    MANUAL = "manual"                      # Block and require manual resolution
    VERSION_PRESERVE = "version_preserve"  # Keep both versions with suffix


class SyncDirection(str, Enum):
    """Direction of sync operation."""
    UPLOAD = "upload"
    DOWNLOAD = "download"
    BIDIRECTIONAL = "bidirectional"


class SyncStatus(str, Enum):
    """Status of a sync operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"


@dataclass
class FileMetadata:
    """Metadata for a file (local or cloud)."""
    path: str
    size: int
    modified_time: datetime
    checksum: Optional[str] = None  # MD5 or etag
    storage_class: Optional[str] = None  # For cloud storage tiers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "size": self.size,
            "modified_time": self.modified_time.isoformat(),
            "checksum": self.checksum,
            "storage_class": self.storage_class,
        }


@dataclass
class SyncOperation:
    """Represents a single sync operation."""
    operation_id: str
    source_path: str
    dest_path: str
    direction: SyncDirection
    status: SyncStatus
    file_size: int
    checksum: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    bytes_transferred: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_id": self.operation_id,
            "source_path": self.source_path,
            "dest_path": self.dest_path,
            "direction": self.direction.value,
            "status": self.status.value,
            "file_size": self.file_size,
            "checksum": self.checksum,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "timestamp": self.timestamp.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "bytes_transferred": self.bytes_transferred,
        }


@dataclass
class SyncConflict:
    """Represents a sync conflict."""
    file_path: str
    local_metadata: Optional[FileMetadata]
    cloud_metadata: Optional[FileMetadata]
    conflict_reason: str  # 'both_modified', 'missing_locally', 'missing_in_cloud', etc.
    resolution: Optional[ConflictStrategy] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "local_metadata": self.local_metadata.to_dict() if self.local_metadata else None,
            "cloud_metadata": self.cloud_metadata.to_dict() if self.cloud_metadata else None,
            "conflict_reason": self.conflict_reason,
            "resolution": self.resolution.value if self.resolution else None,
        }


@dataclass
class SyncProgress:
    """Progress information for a sync operation."""
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    total_bytes: int = 0
    transferred_bytes: int = 0
    start_time: Optional[datetime] = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def progress_percent(self) -> float:
        """Get progress percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.completed_files / self.total_files) * 100
    
    @property
    def bytes_percent(self) -> float:
        """Get bytes transferred percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.transferred_bytes / self.total_bytes) * 100
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time:
            return (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return 0.0
    
    @property
    def transfer_speed_mbps(self) -> float:
        """Get transfer speed in MB/s."""
        elapsed = self.elapsed_seconds
        if elapsed > 0:
            return (self.transferred_bytes / (1024 * 1024)) / elapsed
        return 0.0


# ============================================================================
# Abstract Cloud Storage Interface
# ============================================================================

class CloudStorage(ABC):
    """Abstract base class for cloud storage providers."""
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to cloud storage."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to cloud storage."""
        pass
    
    @abstractmethod
    def upload_file(
        self,
        local_path: Path,
        cloud_path: str,
        multipart_threshold: int = 100 * 1024 * 1024,  # 100 MB
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """
        Upload a file to cloud storage.
        
        Args:
            local_path: Local file path
            cloud_path: Cloud destination path
            multipart_threshold: Use multipart upload for files larger than this
            progress_callback: Optional callback for progress (bytes_transferred)
        
        Returns:
            Checksum or etag of uploaded file
        """
        pass
    
    @abstractmethod
    def download_file(
        self,
        cloud_path: str,
        local_path: Path,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """
        Download a file from cloud storage.
        
        Args:
            cloud_path: Cloud file path
            local_path: Local destination path
            progress_callback: Optional callback for progress (bytes_transferred)
        
        Returns:
            Checksum of downloaded file
        """
        pass
    
    @abstractmethod
    def delete_file(self, cloud_path: str) -> None:
        """Delete a file from cloud storage."""
        pass
    
    @abstractmethod
    def list_files(
        self,
        prefix: str = "",
        recursive: bool = True,
    ) -> List[FileMetadata]:
        """
        List files in cloud storage.
        
        Args:
            prefix: Optional prefix to filter files
            recursive: Whether to list recursively
        
        Returns:
            List of FileMetadata objects
        """
        pass
    
    @abstractmethod
    def file_exists(self, cloud_path: str) -> bool:
        """Check if a file exists in cloud storage."""
        pass
    
    @abstractmethod
    def get_file_metadata(self, cloud_path: str) -> Optional[FileMetadata]:
        """Get metadata for a cloud file."""
        pass


# ============================================================================
# Sync Engine
# ============================================================================

class CloudSyncEngine:
    """Orchestrates bidirectional sync between local and cloud storage."""
    
    def __init__(
        self,
        cloud_storage: CloudStorage,
        sync_state_db: Path,
        conflict_strategy: ConflictStrategy = ConflictStrategy.TIMESTAMP_WINS,
        max_workers: int = 4,
        retry_attempts: int = 3,
        retry_delay_seconds: int = 5,
    ):
        """
        Initialize sync engine.
        
        Args:
            cloud_storage: CloudStorage provider instance
            sync_state_db: Path to sync state database file
            conflict_strategy: Strategy for handling conflicts
            max_workers: Max concurrent upload/download operations
            retry_attempts: Number of retry attempts for failed operations
            retry_delay_seconds: Delay between retries
        """
        self.cloud_storage = cloud_storage
        self.sync_state_db = Path(sync_state_db)
        self.conflict_strategy = conflict_strategy
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.progress = SyncProgress()
        self.conflicts: List[SyncConflict] = []
        self.operations: List[SyncOperation] = []
        
        self._load_sync_state()
    
    def _load_sync_state(self) -> None:
        """Load sync state from database."""
        if self.sync_state_db.exists():
            try:
                with open(self.sync_state_db, "r") as f:
                    state = json.load(f)
                    self.operations = [
                        SyncOperation(**op) for op in state.get("operations", [])
                    ]
                    logger.info(f"Loaded sync state with {len(self.operations)} operations")
            except Exception as e:
                logger.warning(f"Failed to load sync state: {e}")
    
    def _save_sync_state(self) -> None:
        """Save sync state to database."""
        self.sync_state_db.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "operations": [op.to_dict() for op in self.operations],
            "conflicts": [c.to_dict() for c in self.conflicts],
        }
        with open(self.sync_state_db, "w") as f:
            json.dump(state, f, indent=2, default=str)
    
    def upload_directory(
        self,
        local_dir: Path,
        cloud_prefix: str,
        pattern: str = "**/*",
        exclude_patterns: Optional[List[str]] = None,
    ) -> SyncProgress:
        """
        Upload a directory to cloud storage.
        
        Args:
            local_dir: Local directory to upload
            cloud_prefix: Cloud storage prefix
            pattern: File pattern to match (glob-style)
            exclude_patterns: Patterns to exclude
        
        Returns:
            SyncProgress object with results
        """
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise ValueError(f"Not a directory: {local_dir}")
        
        logger.info(f"Starting upload of {local_dir} to {cloud_prefix}")
        
        # Collect files to upload
        files_to_upload = self._collect_files(local_dir, pattern, exclude_patterns)
        logger.info(f"Found {len(files_to_upload)} files to upload")
        
        self.progress = SyncProgress(total_files=len(files_to_upload))
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in files_to_upload)
        self.progress.total_bytes = total_size
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for local_file in files_to_upload:
                relative_path = local_file.relative_to(local_dir)
                cloud_path = f"{cloud_prefix}/{relative_path}".replace("\\", "/")
                
                future = executor.submit(
                    self._upload_with_retry,
                    local_file,
                    cloud_path,
                )
                futures[future] = (local_file, cloud_path)
            
            for future in as_completed(futures):
                local_file, cloud_path = futures[future]
                try:
                    checksum = future.result()
                    self.progress.completed_files += 1
                    self.progress.transferred_bytes += local_file.stat().st_size
                    logger.info(f"Uploaded {local_file} -> {cloud_path}")
                except Exception as e:
                    self.progress.failed_files += 1
                    logger.error(f"Failed to upload {local_file}: {e}")
        
        self._save_sync_state()
        return self.progress
    
    def download_directory(
        self,
        cloud_prefix: str,
        local_dir: Path,
    ) -> SyncProgress:
        """
        Download a directory from cloud storage.
        
        Args:
            cloud_prefix: Cloud storage prefix
            local_dir: Local destination directory
        
        Returns:
            SyncProgress object with results
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting download from {cloud_prefix} to {local_dir}")
        
        # List files in cloud
        cloud_files = self.cloud_storage.list_files(prefix=cloud_prefix)
        logger.info(f"Found {len(cloud_files)} files to download")
        
        self.progress = SyncProgress(total_files=len(cloud_files))
        
        # Calculate total size
        total_size = sum(f.size for f in cloud_files)
        self.progress.total_bytes = total_size
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for cloud_file in cloud_files:
                # Remove prefix from path
                relative_path = cloud_file.path.replace(cloud_prefix + "/", "")
                local_path = local_dir / relative_path
                
                future = executor.submit(
                    self._download_with_retry,
                    cloud_file.path,
                    local_path,
                )
                futures[future] = (cloud_file.path, local_path)
            
            for future in as_completed(futures):
                cloud_path, local_path = futures[future]
                try:
                    checksum = future.result()
                    self.progress.completed_files += 1
                    self.progress.transferred_bytes += local_path.stat().st_size
                    logger.info(f"Downloaded {cloud_path} -> {local_path}")
                except Exception as e:
                    self.progress.failed_files += 1
                    logger.error(f"Failed to download {cloud_path}: {e}")
        
        self._save_sync_state()
        return self.progress
    
    def sync_bidirectional(
        self,
        local_dir: Path,
        cloud_prefix: str,
        pattern: str = "**/*",
        exclude_patterns: Optional[List[str]] = None,
    ) -> Tuple[SyncProgress, List[SyncConflict]]:
        """
        Perform bidirectional sync with conflict detection.
        
        Args:
            local_dir: Local directory
            cloud_prefix: Cloud storage prefix
            pattern: File pattern to match
            exclude_patterns: Patterns to exclude
        
        Returns:
            Tuple of (SyncProgress, list of SyncConflict)
        """
        local_dir = Path(local_dir)
        logger.info(f"Starting bidirectional sync: {local_dir} <-> {cloud_prefix}")
        
        # Collect local files
        local_files = self._collect_files(local_dir, pattern, exclude_patterns)
        local_metadata = {
            f.relative_to(local_dir).as_posix(): self._get_local_metadata(f)
            for f in local_files
        }
        
        # Collect cloud files
        cloud_files = self.cloud_storage.list_files(prefix=cloud_prefix)
        cloud_metadata = {
            f.path.replace(cloud_prefix + "/", ""): f
            for f in cloud_files
        }
        
        # Detect conflicts and sync items
        self.conflicts = []
        sync_items = []
        
        # Files only in cloud -> download
        for cloud_path, cloud_meta in cloud_metadata.items():
            if cloud_path not in local_metadata:
                sync_items.append({
                    "direction": SyncDirection.DOWNLOAD,
                    "local_path": local_dir / cloud_path,
                    "cloud_path": f"{cloud_prefix}/{cloud_path}",
                    "cloud_meta": cloud_meta,
                })
        
        # Files in both -> check for conflicts
        for local_path_str, local_meta in local_metadata.items():
            if local_path_str in cloud_metadata:
                cloud_meta = cloud_metadata[local_path_str]
                
                # Compare checksums
                local_checksum = self._compute_file_hash(local_dir / local_path_str)
                
                if local_checksum != cloud_meta.checksum:
                    # Files differ
                    if local_meta.modified_time > cloud_meta.modified_time:
                        # Local is newer -> upload
                        sync_items.append({
                            "direction": SyncDirection.UPLOAD,
                            "local_path": local_dir / local_path_str,
                            "cloud_path": f"{cloud_prefix}/{local_path_str}",
                        })
                    elif cloud_meta.modified_time > local_meta.modified_time:
                        # Cloud is newer -> download
                        sync_items.append({
                            "direction": SyncDirection.DOWNLOAD,
                            "local_path": local_dir / local_path_str,
                            "cloud_path": f"{cloud_prefix}/{local_path_str}",
                            "cloud_meta": cloud_meta,
                        })
                    else:
                        # Same time -> conflict
                        conflict = SyncConflict(
                            file_path=local_path_str,
                            local_metadata=local_meta,
                            cloud_metadata=cloud_meta,
                            conflict_reason="both_modified_same_time",
                        )
                        self._resolve_conflict(conflict)
                        self.conflicts.append(conflict)
            else:
                # File only locally -> upload
                sync_items.append({
                    "direction": SyncDirection.UPLOAD,
                    "local_path": local_dir / local_path_str,
                    "cloud_path": f"{cloud_prefix}/{local_path_str}",
                })
        
        # Execute sync operations
        logger.info(f"Syncing {len(sync_items)} items ({len(self.conflicts)} conflicts)")
        
        self.progress = SyncProgress(total_files=len(sync_items))
        total_size = sum(
            item["local_path"].stat().st_size
            if item["direction"] == SyncDirection.UPLOAD and item["local_path"].exists()
            else item["cloud_meta"].size
            for item in sync_items
            if "cloud_meta" in item or item["direction"] == SyncDirection.UPLOAD
        )
        self.progress.total_bytes = total_size
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for item in sync_items:
                if item["direction"] == SyncDirection.UPLOAD:
                    future = executor.submit(
                        self._upload_with_retry,
                        item["local_path"],
                        item["cloud_path"],
                    )
                else:
                    future = executor.submit(
                        self._download_with_retry,
                        item["cloud_path"],
                        item["local_path"],
                    )
                futures[future] = item
            
            for future in as_completed(futures):
                item = futures[future]
                try:
                    future.result()
                    self.progress.completed_files += 1
                    if item["direction"] == SyncDirection.UPLOAD:
                        self.progress.transferred_bytes += item["local_path"].stat().st_size
                    else:
                        self.progress.transferred_bytes += item["local_path"].stat().st_size
                except Exception as e:
                    self.progress.failed_files += 1
                    logger.error(f"Failed to sync {item}: {e}")
        
        self._save_sync_state()
        return self.progress, self.conflicts
    
    def _upload_with_retry(self, local_path: Path, cloud_path: str) -> str:
        """Upload with retry logic."""
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(f"Upload attempt {attempt + 1}/{self.retry_attempts}: {local_path}")
                checksum = self.cloud_storage.upload_file(local_path, cloud_path)
                
                # Record operation
                op = SyncOperation(
                    operation_id=f"{cloud_path}_{datetime.now(timezone.utc).timestamp()}",
                    source_path=str(local_path),
                    dest_path=cloud_path,
                    direction=SyncDirection.UPLOAD,
                    status=SyncStatus.COMPLETED,
                    file_size=local_path.stat().st_size,
                    checksum=checksum,
                    completed_at=datetime.now(timezone.utc),
                )
                self.operations.append(op)
                
                return checksum
            except Exception as e:
                last_error = e
                logger.warning(f"Upload failed (attempt {attempt + 1}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay_seconds)
        
        raise last_error or Exception("Upload failed after all retries")
    
    def _download_with_retry(self, cloud_path: str, local_path: Path) -> str:
        """Download with retry logic."""
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                logger.debug(f"Download attempt {attempt + 1}/{self.retry_attempts}: {cloud_path}")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                checksum = self.cloud_storage.download_file(cloud_path, local_path)
                
                # Record operation
                op = SyncOperation(
                    operation_id=f"{cloud_path}_{datetime.now(timezone.utc).timestamp()}",
                    source_path=cloud_path,
                    dest_path=str(local_path),
                    direction=SyncDirection.DOWNLOAD,
                    status=SyncStatus.COMPLETED,
                    file_size=local_path.stat().st_size,
                    checksum=checksum,
                    completed_at=datetime.now(timezone.utc),
                )
                self.operations.append(op)
                
                return checksum
            except Exception as e:
                last_error = e
                logger.warning(f"Download failed (attempt {attempt + 1}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay_seconds)
        
        raise last_error or Exception("Download failed after all retries")
    
    def _collect_files(
        self,
        directory: Path,
        pattern: str,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Path]:
        """Collect files matching pattern."""
        exclude_patterns = exclude_patterns or []
        files = []
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                # Check exclude patterns
                should_exclude = False
                for exclude_pattern in exclude_patterns:
                    if file_path.match(exclude_pattern):
                        should_exclude = True
                        break
                
                if not should_exclude:
                    files.append(file_path)
        
        return files
    
    def _get_local_metadata(self, file_path: Path) -> FileMetadata:
        """Get metadata for a local file."""
        stat = file_path.stat()
        checksum = self._compute_file_hash(file_path)
        
        return FileMetadata(
            path=str(file_path),
            size=stat.st_size,
            modified_time=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            checksum=checksum,
        )
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _resolve_conflict(self, conflict: SyncConflict) -> None:
        """Resolve a conflict based on strategy."""
        if self.conflict_strategy == ConflictStrategy.TIMESTAMP_WINS:
            if conflict.local_metadata and conflict.cloud_metadata:
                if conflict.local_metadata.modified_time > conflict.cloud_metadata.modified_time:
                    conflict.resolution = ConflictStrategy.LOCAL_WINS
                else:
                    conflict.resolution = ConflictStrategy.CLOUD_WINS
        elif self.conflict_strategy == ConflictStrategy.LOCAL_WINS:
            conflict.resolution = ConflictStrategy.LOCAL_WINS
        elif self.conflict_strategy == ConflictStrategy.CLOUD_WINS:
            conflict.resolution = ConflictStrategy.CLOUD_WINS
        elif self.conflict_strategy == ConflictStrategy.MANUAL:
            conflict.resolution = ConflictStrategy.MANUAL
        elif self.conflict_strategy == ConflictStrategy.VERSION_PRESERVE:
            conflict.resolution = ConflictStrategy.VERSION_PRESERVE
    
    def retry_failed_operations(self) -> int:
        """Retry all failed operations. Returns count of retried operations."""
        failed_ops = [op for op in self.operations if op.status == SyncStatus.FAILED]
        logger.info(f"Retrying {len(failed_ops)} failed operations")
        
        retried_count = 0
        for op in failed_ops:
            try:
                if op.direction == SyncDirection.UPLOAD:
                    self._upload_with_retry(Path(op.source_path), op.dest_path)
                else:
                    self._download_with_retry(op.source_path, Path(op.dest_path))
                retried_count += 1
            except Exception as e:
                logger.error(f"Retry failed for {op.operation_id}: {e}")
        
        self._save_sync_state()
        return retried_count
    
    def get_sync_report(self) -> Dict[str, Any]:
        """Get a report of sync operations."""
        return {
            "total_operations": len(self.operations),
            "completed": len([op for op in self.operations if op.status == SyncStatus.COMPLETED]),
            "failed": len([op for op in self.operations if op.status == SyncStatus.FAILED]),
            "total_bytes_transferred": sum(op.bytes_transferred for op in self.operations),
            "conflicts_detected": len(self.conflicts),
            "elapsed_seconds": self.progress.elapsed_seconds,
            "transfer_speed_mbps": self.progress.transfer_speed_mbps,
        }
