"""
Feature metadata management for computed features.

Tracks:
- Computation parameters (window sizes, thresholds, etc.)
- Dependency information (which features depend on which)
- Creation timestamp and data version used as source
- Feature schema (column names, data types)
- Version changelog documenting changes between versions
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Metadata
# ============================================================================

@dataclass
class ComputationParameter:
    """Single computation parameter used to generate features."""
    name: str
    value: Any
    data_type: str  # "int", "float", "string", "bool", etc.
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self._serialize_value(self.value),
            "data_type": self.data_type,
            "description": self.description,
        }
    
    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize value to JSON-compatible format."""
        if isinstance(value, (list, tuple)):
            return [ComputationParameter._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: ComputationParameter._serialize_value(v) for k, v in value.items()}
        elif hasattr(value, "isoformat"):  # datetime
            return value.isoformat()
        else:
            return value
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComputationParameter":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            value=data["value"],
            data_type=data["data_type"],
            description=data.get("description", ""),
        )


@dataclass
class FeatureDependency:
    """Information about a feature dependency."""
    feature_name: str
    is_computed: bool = True  # Whether it's a computed feature or raw data
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "is_computed": self.is_computed,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureDependency":
        """Create from dictionary."""
        return cls(
            feature_name=data["feature_name"],
            is_computed=data.get("is_computed", True),
            description=data.get("description", ""),
        )


@dataclass
class FeatureSchema:
    """Schema information for a feature."""
    column_name: str
    data_type: str  # "int64", "float64", "string", "bool", etc.
    nullable: bool = False
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column_name": self.column_name,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSchema":
        """Create from dictionary."""
        return cls(
            column_name=data["column_name"],
            data_type=data["data_type"],
            nullable=data.get("nullable", False),
            description=data.get("description", ""),
        )


@dataclass
class FeatureComputationMetadata:
    """Metadata for a single computed feature."""
    feature_name: str
    description: str
    computation_parameters: List[ComputationParameter] = field(default_factory=list)
    dependencies: List[FeatureDependency] = field(default_factory=list)
    schema: List[FeatureSchema] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "description": self.description,
            "computation_parameters": [p.to_dict() for p in self.computation_parameters],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "schema": [s.to_dict() for s in self.schema],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureComputationMetadata":
        """Create from dictionary."""
        return cls(
            feature_name=data["feature_name"],
            description=data["description"],
            computation_parameters=[
                ComputationParameter.from_dict(p) 
                for p in data.get("computation_parameters", [])
            ],
            dependencies=[
                FeatureDependency.from_dict(d) 
                for d in data.get("dependencies", [])
            ],
            schema=[
                FeatureSchema.from_dict(s) 
                for s in data.get("schema", [])
            ],
        )


@dataclass
class FeatureStorageMetadata:
    """
    Complete metadata for stored features.
    
    Tracks all information needed to reproduce features and understand
    their structure and dependencies.
    """
    version: str  # Semantic version (e.g., "1.0.0")
    asset: str  # Asset identifier (ES, MES, VIX, etc.)
    feature_set_name: str  # Name of the feature set
    creation_timestamp: datetime
    source_data_version: str  # Version of cleaned data used to compute features
    start_date: datetime
    end_date: datetime
    record_count: int
    features: List[FeatureComputationMetadata] = field(default_factory=list)
    compression: str = "snappy"  # Parquet compression codec
    partition_strategy: str = "none"  # "none", "by_date", "by_month", etc.
    creator: str = ""
    environment: str = "production"
    frozen: bool = False
    freeze_timestamp: Optional[datetime] = None
    changelog: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "asset": self.asset,
            "feature_set_name": self.feature_set_name,
            "creation_timestamp": self.creation_timestamp.isoformat(),
            "source_data_version": self.source_data_version,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "record_count": self.record_count,
            "features": [f.to_dict() for f in self.features],
            "compression": self.compression,
            "partition_strategy": self.partition_strategy,
            "creator": self.creator,
            "environment": self.environment,
            "frozen": self.frozen,
            "freeze_timestamp": self.freeze_timestamp.isoformat() if self.freeze_timestamp else None,
            "changelog": self.changelog,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureStorageMetadata":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            asset=data["asset"],
            feature_set_name=data["feature_set_name"],
            creation_timestamp=datetime.fromisoformat(data["creation_timestamp"]),
            source_data_version=data["source_data_version"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]),
            record_count=data["record_count"],
            features=[
                FeatureComputationMetadata.from_dict(f)
                for f in data.get("features", [])
            ],
            compression=data.get("compression", "snappy"),
            partition_strategy=data.get("partition_strategy", "none"),
            creator=data.get("creator", ""),
            environment=data.get("environment", "production"),
            frozen=data.get("frozen", False),
            freeze_timestamp=(
                datetime.fromisoformat(data["freeze_timestamp"])
                if data.get("freeze_timestamp")
                else None
            ),
            changelog=data.get("changelog", []),
            tags=data.get("tags", []),
        )
    
    def add_changelog_entry(
        self,
        change_type: str,
        description: str,
        author: str = "",
    ) -> None:
        """
        Add entry to changelog.
        
        Args:
            change_type: Type of change (creation, patch, minor, major, deprecation, freeze)
            description: Human-readable description of the change
            author: User making the change
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "change_type": change_type,
            "description": description,
            "author": author,
        }
        self.changelog.append(entry)
        logger.info(f"Added changelog entry: {change_type}")
    
    def add_tag(self, tag: str) -> None:
        """Add tag to this version."""
        if tag not in self.tags:
            self.tags.append(tag)
            logger.info(f"Added tag '{tag}' to feature version {self.version}")
    
    def remove_tag(self, tag: str) -> None:
        """Remove tag from this version."""
        if tag in self.tags:
            self.tags.remove(tag)
            logger.info(f"Removed tag '{tag}' from feature version {self.version}")
    
    def freeze(self, author: str = "") -> None:
        """Freeze this version to prevent modifications."""
        if not self.frozen:
            self.frozen = True
            self.freeze_timestamp = datetime.now(timezone.utc)
            self.add_changelog_entry("freeze", "Version frozen to prevent modifications", author)
            logger.info(f"Froze feature version {self.version}/{self.asset}")
    
    def unfreeze(self, author: str = "") -> None:
        """Unfreeze this version to allow modifications."""
        if self.frozen:
            self.frozen = False
            self.freeze_timestamp = None
            self.add_changelog_entry("unfreeze", "Version unfrozen", author)
            logger.info(f"Unfroze feature version {self.version}/{self.asset}")
    
    def validate_frozen(self) -> None:
        """Raise error if version is frozen."""
        if self.frozen:
            raise ValueError(
                f"Cannot modify frozen feature version {self.version}/{self.asset}"
            )
    
    def get_total_size_bytes(self) -> int:
        """Get approximate total size based on record count and schema."""
        # Rough estimate: 8 bytes per float64, 8 bytes per int64, varying for strings
        total_size = self.record_count * 8  # timestamp
        for feature in self.features:
            for schema_col in feature.schema:
                if "int" in schema_col.data_type:
                    total_size += self.record_count * 8
                elif "float" in schema_col.data_type:
                    total_size += self.record_count * 8
                else:
                    # String/object columns - rough estimate of 50 bytes per row
                    total_size += self.record_count * 50
        return total_size


# ============================================================================
# Metadata Persistence
# ============================================================================

class FeatureMetadataPersistence:
    """Handles saving and loading feature metadata from JSON files."""
    
    METADATA_FILENAME = "metadata.json"
    
    @staticmethod
    def save(
        metadata: FeatureStorageMetadata,
        directory: Path,
        pretty: bool = True,
    ) -> Path:
        """
        Save metadata to JSON file.
        
        Args:
            metadata: FeatureStorageMetadata to save
            directory: Directory to save metadata.json in
            pretty: Whether to pretty-print JSON
        
        Returns:
            Path to saved file
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        filepath = directory / FeatureMetadataPersistence.METADATA_FILENAME
        
        with open(filepath, 'w') as f:
            json.dump(
                metadata.to_dict(),
                f,
                indent=2 if pretty else None,
                default=str,
            )
        
        logger.info(f"Saved feature metadata to {filepath}")
        return filepath
    
    @staticmethod
    def load(directory: Path) -> FeatureStorageMetadata:
        """
        Load metadata from JSON file.
        
        Args:
            directory: Directory containing metadata.json
        
        Returns:
            FeatureStorageMetadata instance
        
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            ValueError: If metadata is invalid
        """
        directory = Path(directory)
        filepath = directory / FeatureMetadataPersistence.METADATA_FILENAME
        
        if not filepath.exists():
            raise FileNotFoundError(f"Feature metadata file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded feature metadata from {filepath}")
        return FeatureStorageMetadata.from_dict(data)
    
    @staticmethod
    def exists(directory: Path) -> bool:
        """Check if metadata file exists."""
        directory = Path(directory)
        return (directory / FeatureMetadataPersistence.METADATA_FILENAME).exists()


# ============================================================================
# Metadata Builder Helper
# ============================================================================

class FeatureMetadataBuilder:
    """Helper to build FeatureStorageMetadata incrementally."""
    
    def __init__(
        self,
        version: str,
        asset: str,
        feature_set_name: str,
        source_data_version: str,
    ):
        """
        Initialize metadata builder.
        
        Args:
            version: Semantic version string
            asset: Asset identifier
            feature_set_name: Name of the feature set
            source_data_version: Version of source data used
        """
        self.metadata = FeatureStorageMetadata(
            version=version,
            asset=asset,
            feature_set_name=feature_set_name,
            creation_timestamp=datetime.now(timezone.utc),
            source_data_version=source_data_version,
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc),
            record_count=0,
        )
    
    def set_date_range(self, start_date: datetime, end_date: datetime) -> "FeatureMetadataBuilder":
        """Set the date range of features."""
        self.metadata.start_date = start_date
        self.metadata.end_date = end_date
        return self
    
    def set_record_count(self, count: int) -> "FeatureMetadataBuilder":
        """Set the record count."""
        self.metadata.record_count = count
        return self
    
    def add_feature(self, feature_metadata: FeatureComputationMetadata) -> "FeatureMetadataBuilder":
        """Add a feature's metadata."""
        self.metadata.features.append(feature_metadata)
        return self
    
    def set_compression(self, compression: str) -> "FeatureMetadataBuilder":
        """Set the Parquet compression codec."""
        self.metadata.compression = compression
        return self
    
    def set_partition_strategy(self, strategy: str) -> "FeatureMetadataBuilder":
        """Set the partitioning strategy."""
        self.metadata.partition_strategy = strategy
        return self
    
    def set_creator(self, creator: str) -> "FeatureMetadataBuilder":
        """Set the creator."""
        self.metadata.creator = creator
        return self
    
    def set_environment(self, environment: str) -> "FeatureMetadataBuilder":
        """Set the environment."""
        self.metadata.environment = environment
        return self
    
    def build(self) -> FeatureStorageMetadata:
        """Build and return the metadata."""
        return self.metadata
