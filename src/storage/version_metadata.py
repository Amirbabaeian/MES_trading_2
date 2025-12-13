"""
Version metadata schema and management for data versioning system.

Provides:
- Semantic version parsing and validation
- Version metadata dataclasses
- Schema validation for version metadata
- Changelog tracking
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import re

logger = logging.getLogger(__name__)


# ============================================================================
# Semantic Version Management
# ============================================================================

class SemanticVersion:
    """Represents a semantic version in format v{major}.{minor}.{patch}."""
    
    VERSION_PATTERN = re.compile(r'^v(\d+)\.(\d+)\.(\d+)$')
    
    def __init__(self, major: int, minor: int = 0, patch: int = 0):
        """
        Initialize semantic version.
        
        Args:
            major: Major version number (required)
            minor: Minor version number (default 0)
            patch: Patch version number (default 0)
        
        Raises:
            ValueError: If any version component is negative
        """
        if major < 0 or minor < 0 or patch < 0:
            raise ValueError("Version components must be non-negative")
        
        self.major = major
        self.minor = minor
        self.patch = patch
    
    @classmethod
    def parse(cls, version_str: str) -> 'SemanticVersion':
        """
        Parse semantic version from string.
        
        Args:
            version_str: Version string (e.g., "v1.0.0" or "1.0.0")
        
        Returns:
            SemanticVersion instance
        
        Raises:
            ValueError: If version string format is invalid
        """
        # Remove 'v' prefix if present
        normalized = version_str.lstrip('v')
        
        match = cls.VERSION_PATTERN.match(f"v{normalized}")
        if not match:
            raise ValueError(
                f"Invalid semantic version format: {version_str}. "
                "Expected v{major}.{minor}.{patch}"
            )
        
        major, minor, patch = match.groups()
        return cls(int(major), int(minor), int(patch))
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"v{self.major}.{self.minor}.{self.patch}"
    
    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"SemanticVersion(major={self.major}, minor={self.minor}, patch={self.patch})"
    
    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    def __lt__(self, other: 'SemanticVersion') -> bool:
        """Check if less than."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __le__(self, other: 'SemanticVersion') -> bool:
        """Check if less than or equal."""
        return self == other or self < other
    
    def __gt__(self, other: 'SemanticVersion') -> bool:
        """Check if greater than."""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)
    
    def __ge__(self, other: 'SemanticVersion') -> bool:
        """Check if greater than or equal."""
        return self == other or self > other
    
    def __hash__(self) -> int:
        """Return hash for use in collections."""
        return hash((self.major, self.minor, self.patch))
    
    def increment_major(self) -> 'SemanticVersion':
        """Return new version with major incremented."""
        return SemanticVersion(self.major + 1, 0, 0)
    
    def increment_minor(self) -> 'SemanticVersion':
        """Return new version with minor incremented."""
        return SemanticVersion(self.major, self.minor + 1, 0)
    
    def increment_patch(self) -> 'SemanticVersion':
        """Return new version with patch incremented."""
        return SemanticVersion(self.major, self.minor, self.patch + 1)


# ============================================================================
# Metadata Dataclasses
# ============================================================================

@dataclass
class DataRange:
    """Represents temporal range of data."""
    start_timestamp: datetime
    end_timestamp: datetime
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'DataRange':
        """Create from dictionary."""
        return cls(
            start_timestamp=datetime.fromisoformat(data["start_timestamp"]),
            end_timestamp=datetime.fromisoformat(data["end_timestamp"]),
        )


@dataclass
class FileInfo:
    """Information about a single file in the version."""
    filename: str
    size_bytes: int
    format: str  # "parquet", "json", "csv", "arrow"
    md5_checksum: str
    row_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileInfo':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SchemaColumn:
    """Definition of a single column in the schema."""
    name: str
    data_type: str  # Parquet type like "int64", "string", "timestamp[ns]"
    nullable: bool = True
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaColumn':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SchemaInfo:
    """Schema information for versioned data."""
    schema_version: str
    columns: List[SchemaColumn] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_version": self.schema_version,
            "columns": [col.to_dict() for col in self.columns],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaInfo':
        """Create from dictionary."""
        return cls(
            schema_version=data["schema_version"],
            columns=[SchemaColumn.from_dict(col) for col in data.get("columns", [])],
        )


@dataclass
class DataQuality:
    """Data quality metrics and validation results."""
    validation_status: str  # "passed", "failed", "warnings"
    quality_flags: List[str] = field(default_factory=list)  # e.g., "gaps_in_timestamps"
    completeness_percentage: Optional[float] = None
    null_counts: Dict[str, int] = field(default_factory=dict)
    validation_checks: List[str] = field(default_factory=list)  # Names of checks that passed
    validation_overrides: List[str] = field(default_factory=list)  # Overridden checks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataQuality':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Lineage:
    """Data lineage and dependencies."""
    raw_sources: List[Dict[str, Any]] = field(default_factory=list)  # [{"asset": "MES", "dates": [...]}]
    cleaned_version: Optional[int] = None  # For features layer
    feature_parameters: Dict[str, Any] = field(default_factory=dict)  # Feature computation parameters
    cleaning_parameters: Dict[str, Any] = field(default_factory=dict)  # Cleaning parameters
    processing_script: Optional[str] = None
    processing_script_hash: Optional[str] = None  # SHA-1 hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Lineage':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ChangelogEntry:
    """Single entry in version changelog."""
    version: str  # e.g., "1.0.0"
    timestamp: datetime
    author: str
    change_type: str  # "creation", "patch", "major", "deprecation"
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "author": self.author,
            "change_type": self.change_type,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChangelogEntry':
        """Create from dictionary."""
        return cls(
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            author=data["author"],
            change_type=data["change_type"],
            description=data["description"],
        )


@dataclass
class VersionMetadata:
    """Complete metadata for a versioned dataset."""
    layer: str  # "cleaned" or "features"
    version: SemanticVersion
    asset_or_feature_set: str  # Asset name (e.g., "MES") or feature set name
    creation_timestamp: datetime
    data_range: DataRange
    record_count: int
    schema_info: SchemaInfo
    files: List[FileInfo] = field(default_factory=list)
    data_quality: Optional[DataQuality] = None
    lineage: Optional[Lineage] = None
    changelog: List[ChangelogEntry] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)  # e.g., ["stable", "production"]
    frozen: bool = False
    freeze_timestamp: Optional[datetime] = None
    creator: str = ""
    environment: str = ""  # e.g., "production", "staging", "development"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer": self.layer,
            "version": str(self.version),
            "asset_or_feature_set": self.asset_or_feature_set,
            "creation_timestamp": self.creation_timestamp.isoformat(),
            "data_range": self.data_range.to_dict(),
            "record_count": self.record_count,
            "schema_info": self.schema_info.to_dict(),
            "files": [f.to_dict() for f in self.files],
            "data_quality": self.data_quality.to_dict() if self.data_quality else None,
            "lineage": self.lineage.to_dict() if self.lineage else None,
            "changelog": [entry.to_dict() for entry in self.changelog],
            "tags": self.tags,
            "frozen": self.frozen,
            "freeze_timestamp": self.freeze_timestamp.isoformat() if self.freeze_timestamp else None,
            "creator": self.creator,
            "environment": self.environment,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionMetadata':
        """Create from dictionary."""
        return cls(
            layer=data["layer"],
            version=SemanticVersion.parse(data["version"]),
            asset_or_feature_set=data["asset_or_feature_set"],
            creation_timestamp=datetime.fromisoformat(data["creation_timestamp"]),
            data_range=DataRange.from_dict(data["data_range"]),
            record_count=data["record_count"],
            schema_info=SchemaInfo.from_dict(data["schema_info"]),
            files=[FileInfo.from_dict(f) for f in data.get("files", [])],
            data_quality=DataQuality.from_dict(data["data_quality"]) if data.get("data_quality") else None,
            lineage=Lineage.from_dict(data["lineage"]) if data.get("lineage") else None,
            changelog=[ChangelogEntry.from_dict(entry) for entry in data.get("changelog", [])],
            tags=data.get("tags", []),
            frozen=data.get("frozen", False),
            freeze_timestamp=datetime.fromisoformat(data["freeze_timestamp"]) if data.get("freeze_timestamp") else None,
            creator=data.get("creator", ""),
            environment=data.get("environment", ""),
        )
    
    def add_changelog_entry(
        self,
        change_type: str,
        description: str,
        author: str = "",
    ) -> None:
        """Add entry to changelog."""
        entry = ChangelogEntry(
            version=str(self.version),
            timestamp=datetime.now(timezone.utc),
            author=author,
            change_type=change_type,
            description=description,
        )
        self.changelog.append(entry)
    
    def add_tag(self, tag: str) -> None:
        """Add tag to version."""
        if tag not in self.tags:
            self.tags.append(tag)
            logger.info(f"Added tag '{tag}' to {self.layer}/{self.version}/{self.asset_or_feature_set}")
    
    def remove_tag(self, tag: str) -> None:
        """Remove tag from version."""
        if tag in self.tags:
            self.tags.remove(tag)
            logger.info(f"Removed tag '{tag}' from {self.layer}/{self.version}/{self.asset_or_feature_set}")
    
    def freeze(self, author: str = "") -> None:
        """Freeze version to prevent modifications."""
        if not self.frozen:
            self.frozen = True
            self.freeze_timestamp = datetime.now(timezone.utc)
            self.add_changelog_entry("freeze", "Version frozen to prevent modifications", author)
            logger.info(f"Froze version {self.layer}/{self.version}/{self.asset_or_feature_set}")
    
    def unfreeze(self, author: str = "") -> None:
        """Unfreeze version to allow modifications."""
        if self.frozen:
            self.frozen = False
            self.freeze_timestamp = None
            self.add_changelog_entry("unfreeze", "Version unfrozen", author)
            logger.info(f"Unfroze version {self.layer}/{self.version}/{self.asset_or_feature_set}")
    
    def validate_frozen(self) -> None:
        """Raise error if version is frozen."""
        if self.frozen:
            raise ValueError(
                f"Cannot modify frozen version {self.layer}/{self.version}/"
                f"{self.asset_or_feature_set}"
            )
    
    def compute_total_size_bytes(self) -> int:
        """Compute total size of all files."""
        return sum(f.size_bytes for f in self.files)
    
    def format_total_size(self) -> str:
        """Return human-readable total size."""
        total_bytes = self.compute_total_size_bytes()
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if total_bytes < 1024.0:
                return f"{total_bytes:.1f} {unit}"
            total_bytes /= 1024.0
        
        return f"{total_bytes:.1f} PB"


# ============================================================================
# Metadata Persistence
# ============================================================================

class MetadataPersistence:
    """Handles saving and loading metadata from JSON files."""
    
    METADATA_FILENAME = "metadata.json"
    
    @staticmethod
    def save(
        metadata: VersionMetadata,
        directory: Path,
        pretty: bool = True,
    ) -> Path:
        """
        Save metadata to JSON file.
        
        Args:
            metadata: VersionMetadata to save
            directory: Directory to save metadata.json in
            pretty: Whether to pretty-print JSON
        
        Returns:
            Path to saved file
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        filepath = directory / MetadataPersistence.METADATA_FILENAME
        
        with open(filepath, 'w') as f:
            json.dump(
                metadata.to_dict(),
                f,
                indent=2 if pretty else None,
                default=str,
            )
        
        logger.info(f"Saved metadata to {filepath}")
        return filepath
    
    @staticmethod
    def load(directory: Path) -> VersionMetadata:
        """
        Load metadata from JSON file.
        
        Args:
            directory: Directory containing metadata.json
        
        Returns:
            VersionMetadata instance
        
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            ValueError: If metadata is invalid
        """
        directory = Path(directory)
        filepath = directory / MetadataPersistence.METADATA_FILENAME
        
        if not filepath.exists():
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded metadata from {filepath}")
        return VersionMetadata.from_dict(data)
    
    @staticmethod
    def exists(directory: Path) -> bool:
        """Check if metadata file exists."""
        directory = Path(directory)
        return (directory / MetadataPersistence.METADATA_FILENAME).exists()
