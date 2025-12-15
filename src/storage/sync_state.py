"""
Sync state tracking system for incremental and resumable transfers.

Provides:
- Persistent tracking of synced files
- Checksum/etag comparison for smart sync
- Resume support for interrupted transfers
- Transaction log for audit purposes
"""

import logging
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class FileState(str, Enum):
    """State of a synced file."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CHECKSUM_MISMATCH = "checksum_mismatch"


@dataclass
class SyncFileRecord:
    """Record of a synced file."""
    local_path: str
    cloud_path: str
    file_size: int
    local_checksum: str
    cloud_checksum: str
    state: FileState
    last_sync_time: datetime
    sync_direction: str  # 'upload' or 'download'
    retry_count: int = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "local_path": self.local_path,
            "cloud_path": self.cloud_path,
            "file_size": self.file_size,
            "local_checksum": self.local_checksum,
            "cloud_checksum": self.cloud_checksum,
            "state": self.state.value,
            "last_sync_time": self.last_sync_time.isoformat(),
            "sync_direction": self.sync_direction,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
        }


class SyncStateDatabase:
    """Manages sync state using SQLite for efficiency."""
    
    def __init__(self, db_path: Path):
        """
        Initialize sync state database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main sync records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sync_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    local_path TEXT NOT NULL UNIQUE,
                    cloud_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    local_checksum TEXT NOT NULL,
                    cloud_checksum TEXT NOT NULL,
                    state TEXT NOT NULL,
                    last_sync_time TEXT NOT NULL,
                    sync_direction TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Transaction log for audit trail
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sync_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    local_path TEXT,
                    cloud_path TEXT,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT,
                    bytes_transferred INTEGER
                )
            """)
            
            # Conflict resolution history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conflicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    conflict_reason TEXT NOT NULL,
                    resolution TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            
            conn.commit()
            logger.info(f"Initialized sync state database at {self.db_path}")
    
    def record_file_sync(self, record: SyncFileRecord) -> None:
        """
        Record a file sync operation.
        
        Args:
            record: SyncFileRecord to store
        """
        now = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert or update
            cursor.execute("""
                INSERT OR REPLACE INTO sync_files (
                    local_path, cloud_path, file_size, local_checksum,
                    cloud_checksum, state, last_sync_time, sync_direction,
                    retry_count, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.local_path,
                record.cloud_path,
                record.file_size,
                record.local_checksum,
                record.cloud_checksum,
                record.state.value,
                record.last_sync_time.isoformat(),
                record.sync_direction,
                record.retry_count,
                record.error_message,
                now,
                now,
            ))
            
            conn.commit()
    
    def get_file_record(self, local_path: str) -> Optional[SyncFileRecord]:
        """
        Get a file record by local path.
        
        Args:
            local_path: Local file path
        
        Returns:
            SyncFileRecord if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT local_path, cloud_path, file_size, local_checksum,
                       cloud_checksum, state, last_sync_time, sync_direction,
                       retry_count, error_message
                FROM sync_files WHERE local_path = ?
            """, (local_path,))
            
            row = cursor.fetchone()
            if row:
                return SyncFileRecord(
                    local_path=row[0],
                    cloud_path=row[1],
                    file_size=row[2],
                    local_checksum=row[3],
                    cloud_checksum=row[4],
                    state=FileState(row[5]),
                    last_sync_time=datetime.fromisoformat(row[6]),
                    sync_direction=row[7],
                    retry_count=row[8],
                    error_message=row[9],
                )
            return None
    
    def get_failed_records(self) -> List[SyncFileRecord]:
        """Get all failed sync records."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT local_path, cloud_path, file_size, local_checksum,
                       cloud_checksum, state, last_sync_time, sync_direction,
                       retry_count, error_message
                FROM sync_files WHERE state = ? OR state = ?
                ORDER BY last_sync_time DESC
            """, (FileState.FAILED.value, FileState.CHECKSUM_MISMATCH.value))
            
            records = []
            for row in cursor.fetchall():
                records.append(SyncFileRecord(
                    local_path=row[0],
                    cloud_path=row[1],
                    file_size=row[2],
                    local_checksum=row[3],
                    cloud_checksum=row[4],
                    state=FileState(row[5]),
                    last_sync_time=datetime.fromisoformat(row[6]),
                    sync_direction=row[7],
                    retry_count=row[8],
                    error_message=row[9],
                ))
            return records
    
    def get_pending_records(self) -> List[SyncFileRecord]:
        """Get all pending sync records."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT local_path, cloud_path, file_size, local_checksum,
                       cloud_checksum, state, last_sync_time, sync_direction,
                       retry_count, error_message
                FROM sync_files WHERE state IN (?, ?)
                ORDER BY last_sync_time ASC
            """, (FileState.PENDING.value, FileState.IN_PROGRESS.value))
            
            records = []
            for row in cursor.fetchall():
                records.append(SyncFileRecord(
                    local_path=row[0],
                    cloud_path=row[1],
                    file_size=row[2],
                    local_checksum=row[3],
                    cloud_checksum=row[4],
                    state=FileState(row[5]),
                    last_sync_time=datetime.fromisoformat(row[6]),
                    sync_direction=row[7],
                    retry_count=row[8],
                    error_message=row[9],
                ))
            return records
    
    def log_operation(
        self,
        operation: str,
        status: str,
        local_path: Optional[str] = None,
        cloud_path: Optional[str] = None,
        message: Optional[str] = None,
        bytes_transferred: Optional[int] = None,
    ) -> None:
        """
        Log a sync operation.
        
        Args:
            operation: Operation type (upload, download, delete, etc.)
            status: Operation status (success, failure, etc.)
            local_path: Local file path
            cloud_path: Cloud file path
            message: Optional message
            bytes_transferred: Number of bytes transferred
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sync_log (
                    timestamp, local_path, cloud_path, operation,
                    status, message, bytes_transferred
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, local_path, cloud_path, operation, status, message, bytes_transferred))
            
            conn.commit()
    
    def log_conflict(
        self,
        file_path: str,
        conflict_reason: str,
        resolution: Optional[str] = None,
    ) -> None:
        """
        Log a conflict resolution.
        
        Args:
            file_path: Path to conflicted file
            conflict_reason: Reason for conflict
            resolution: How conflict was resolved
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conflicts (
                    file_path, conflict_reason, resolution, timestamp
                ) VALUES (?, ?, ?, ?)
            """, (file_path, conflict_reason, resolution, timestamp))
            
            conn.commit()
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get sync statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count by state
            cursor.execute("""
                SELECT state, COUNT(*) as count FROM sync_files GROUP BY state
            """)
            state_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Total bytes
            cursor.execute("SELECT SUM(file_size) FROM sync_files WHERE state = ?",
                          (FileState.COMPLETED.value,))
            total_bytes = cursor.fetchone()[0] or 0
            
            # Total operations
            cursor.execute("SELECT COUNT(*) FROM sync_log")
            total_ops = cursor.fetchone()[0]
            
            # Recent failures
            cursor.execute("""
                SELECT COUNT(*) FROM sync_log WHERE status = 'failure'
                AND timestamp > datetime('now', '-1 day')
            """)
            recent_failures = cursor.fetchone()[0]
            
            return {
                "state_counts": state_counts,
                "total_bytes_synced": total_bytes,
                "total_operations": total_ops,
                "recent_failures_24h": recent_failures,
            }
    
    def cleanup_old_records(self, days_old: int = 30) -> int:
        """
        Delete old sync records.
        
        Args:
            days_old: Delete records older than this many days
        
        Returns:
            Number of records deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM sync_log
                WHERE datetime(timestamp) < datetime('now', ? || ' days')
            """, (f"-{days_old}",))
            
            deleted = cursor.rowcount
            conn.commit()
            logger.info(f"Cleaned up {deleted} old sync log records")
            
            return deleted
    
    def clear_all(self) -> None:
        """Clear all sync state (use with caution)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sync_files")
            cursor.execute("DELETE FROM sync_log")
            cursor.execute("DELETE FROM conflicts")
            conn.commit()
            logger.warning("Cleared all sync state")
