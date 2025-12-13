"""
Core version management system for cleaned and features data.

Provides:
- Version CRUD operations (create, list, retrieve, delete)
- Version tagging system
- Version deprecation and lifecycle management
- Version freezing and immutability enforcement
- Audit logging for all version operations
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timezone
from dataclasses import dataclass

from .version_metadata import (
    VersionMetadata,
    SemanticVersion,
    MetadataPersistence,
    DataRange,
    SchemaInfo,
    FileInfo,
    DataQuality,
    Lineage,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Version Manager
# ============================================================================

class VersionManager:
    """Manages versions of cleaned and features data."""
    
    def __init__(
        self,
        base_path: Path,
        layer: str = "cleaned",  # "cleaned" or "features"
    ):
        """
        Initialize version manager.
        
        Args:
            base_path: Base path for versioned data (e.g., /data/cleaned or /data/features)
            layer: Data layer being managed ("cleaned" or "features")
        
        Raises:
            ValueError: If layer is not "cleaned" or "features"
        """
        if layer not in ("cleaned", "features"):
            raise ValueError(f"Layer must be 'cleaned' or 'features', got: {layer}")
        
        self.base_path = Path(base_path)
        self.layer = layer
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {layer} VersionManager at {self.base_path}")
    
    # ========================================================================
    # Core Operations
    # ========================================================================
    
    def create_version(
        self,
        asset_or_feature_set: str,
        version: SemanticVersion,
        data_range: DataRange,
        record_count: int,
        schema_info: SchemaInfo,
        creator: str = "",
        environment: str = "",
        lineage: Optional[Lineage] = None,
        data_quality: Optional[DataQuality] = None,
        files: Optional[List[FileInfo]] = None,
    ) -> VersionMetadata:
        """
        Create a new version of data.
        
        Args:
            asset_or_feature_set: Asset name or feature set name
            version: SemanticVersion for this data
            data_range: Temporal range of data
            record_count: Total record count
            schema_info: Schema information
            creator: User/system creating version
            environment: Environment identifier
            lineage: Optional lineage information
            data_quality: Optional quality metrics
            files: Optional list of file information
        
        Returns:
            Created VersionMetadata
        
        Raises:
            ValueError: If version already exists
        """
        version_dir = self._get_version_directory(asset_or_feature_set, version)
        
        if version_dir.exists():
            raise ValueError(
                f"Version already exists: {self.layer}/{version}/{asset_or_feature_set}"
            )
        
        metadata = VersionMetadata(
            layer=self.layer,
            version=version,
            asset_or_feature_set=asset_or_feature_set,
            creation_timestamp=datetime.now(timezone.utc),
            data_range=data_range,
            record_count=record_count,
            schema_info=schema_info,
            files=files or [],
            data_quality=data_quality,
            lineage=lineage,
            creator=creator,
            environment=environment,
        )
        
        # Add initial changelog entry
        metadata.add_changelog_entry(
            change_type="creation",
            description=f"Version {version} created",
            author=creator,
        )
        
        # Save metadata
        version_dir.mkdir(parents=True, exist_ok=True)
        MetadataPersistence.save(metadata, version_dir)
        
        logger.info(
            f"Created version {self.layer}/{version}/{asset_or_feature_set} "
            f"with {record_count} records"
        )
        
        return metadata
    
    def get_version(
        self,
        asset_or_feature_set: str,
        version: SemanticVersion,
    ) -> VersionMetadata:
        """
        Retrieve metadata for specific version.
        
        Args:
            asset_or_feature_set: Asset name or feature set name
            version: SemanticVersion to retrieve
        
        Returns:
            VersionMetadata
        
        Raises:
            FileNotFoundError: If version doesn't exist
        """
        version_dir = self._get_version_directory(asset_or_feature_set, version)
        
        if not version_dir.exists():
            raise FileNotFoundError(
                f"Version not found: {self.layer}/{version}/{asset_or_feature_set}"
            )
        
        metadata = MetadataPersistence.load(version_dir)
        logger.info(f"Retrieved version {self.layer}/{version}/{asset_or_feature_set}")
        return metadata
    
    def get_latest_version(
        self,
        asset_or_feature_set: str,
    ) -> Optional[VersionMetadata]:
        """
        Get latest version for an asset.
        
        Args:
            asset_or_feature_set: Asset name or feature set name
        
        Returns:
            Latest VersionMetadata or None if no versions exist
        """
        versions = self.list_versions(asset_or_feature_set)
        
        if not versions:
            return None
        
        # Sort by version and return latest
        latest_metadata = max(versions, key=lambda m: m.version)
        logger.info(f"Retrieved latest version {latest_metadata.version} for {asset_or_feature_set}")
        return latest_metadata
    
    def version_exists(
        self,
        asset_or_feature_set: str,
        version: SemanticVersion,
    ) -> bool:
        """Check if a version exists."""
        version_dir = self._get_version_directory(asset_or_feature_set, version)
        return MetadataPersistence.exists(version_dir)
    
    def delete_version(
        self,
        asset_or_feature_set: str,
        version: SemanticVersion,
        force: bool = False,
    ) -> None:
        """
        Delete a version.
        
        Args:
            asset_or_feature_set: Asset name or feature set name
            version: SemanticVersion to delete
            force: If True, delete frozen versions; otherwise raise error
        
        Raises:
            ValueError: If version is frozen and force=False
            FileNotFoundError: If version doesn't exist
        """
        # Check if frozen
        metadata = self.get_version(asset_or_feature_set, version)
        
        if metadata.frozen and not force:
            raise ValueError(
                f"Cannot delete frozen version {self.layer}/{version}/{asset_or_feature_set}. "
                "Use force=True to override."
            )
        
        version_dir = self._get_version_directory(asset_or_feature_set, version)
        
        # Remove directory
        import shutil
        shutil.rmtree(version_dir)
        
        logger.info(
            f"Deleted version {self.layer}/{version}/{asset_or_feature_set}"
        )
    
    def list_versions(
        self,
        asset_or_feature_set: Optional[str] = None,
    ) -> List[VersionMetadata]:
        """
        List all versions.
        
        Args:
            asset_or_feature_set: If specified, only return versions for this asset.
                                 If None, return all versions.
        
        Returns:
            List of VersionMetadata sorted by version
        """
        versions = []
        
        if asset_or_feature_set:
            # Look for specific asset
            asset_dir = self.base_path / asset_or_feature_set
            if asset_dir.exists():
                for version_dir in asset_dir.glob("v*"):
                    if MetadataPersistence.exists(version_dir):
                        try:
                            metadata = MetadataPersistence.load(version_dir)
                            versions.append(metadata)
                        except Exception as e:
                            logger.warning(f"Failed to load metadata from {version_dir}: {e}")
        else:
            # Look for all assets
            for asset_dir in self.base_path.iterdir():
                if asset_dir.is_dir():
                    for version_dir in asset_dir.glob("v*"):
                        if MetadataPersistence.exists(version_dir):
                            try:
                                metadata = MetadataPersistence.load(version_dir)
                                versions.append(metadata)
                            except Exception as e:
                                logger.warning(f"Failed to load metadata from {version_dir}: {e}")
        
        # Sort by asset, then version
        versions.sort(
            key=lambda m: (m.asset_or_feature_set, m.version)
        )
        
        logger.info(f"Listed {len(versions)} versions for {self.layer} layer")
        return versions
    
    # ========================================================================
    # Tagging
    # ========================================================================
    
    def add_tag(
        self,
        asset_or_feature_set: str,
        version: SemanticVersion,
        tag: str,
    ) -> None:
        """
        Add a tag to a version.
        
        Args:
            asset_or_feature_set: Asset name or feature set name
            version: SemanticVersion to tag
            tag: Tag to add
        
        Raises:
            FileNotFoundError: If version doesn't exist
        """
        metadata = self.get_version(asset_or_feature_set, version)
        metadata.add_tag(tag)
        self._save_version_metadata(asset_or_feature_set, version, metadata)
        logger.info(f"Added tag '{tag}' to {self.layer}/{version}/{asset_or_feature_set}")
    
    def remove_tag(
        self,
        asset_or_feature_set: str,
        version: SemanticVersion,
        tag: str,
    ) -> None:
        """
        Remove a tag from a version.
        
        Args:
            asset_or_feature_set: Asset name or feature set name
            version: SemanticVersion to remove tag from
            tag: Tag to remove
        """
        metadata = self.get_version(asset_or_feature_set, version)
        metadata.remove_tag(tag)
        self._save_version_metadata(asset_or_feature_set, version, metadata)
        logger.info(f"Removed tag '{tag}' from {self.layer}/{version}/{asset_or_feature_set}")
    
    def get_versions_by_tag(self, tag: str) -> List[VersionMetadata]:
        """
        Get all versions with a specific tag.
        
        Args:
            tag: Tag to search for
        
        Returns:
            List of VersionMetadata with the tag
        """
        all_versions = self.list_versions()
        tagged = [v for v in all_versions if tag in v.tags]
        logger.info(f"Found {len(tagged)} versions with tag '{tag}'")
        return tagged
    
    # ========================================================================
    # Freezing / Immutability
    # ========================================================================
    
    def freeze_version(
        self,
        asset_or_feature_set: str,
        version: SemanticVersion,
        author: str = "",
    ) -> None:
        """
        Freeze a version to prevent modifications.
        
        Args:
            asset_or_feature_set: Asset name or feature set name
            version: SemanticVersion to freeze
            author: User freezing the version
        """
        metadata = self.get_version(asset_or_feature_set, version)
        metadata.freeze(author)
        self._save_version_metadata(asset_or_feature_set, version, metadata)
        logger.info(f"Froze version {self.layer}/{version}/{asset_or_feature_set}")
    
    def unfreeze_version(
        self,
        asset_or_feature_set: str,
        version: SemanticVersion,
        author: str = "",
    ) -> None:
        """
        Unfreeze a version to allow modifications.
        
        Args:
            asset_or_feature_set: Asset name or feature set name
            version: SemanticVersion to unfreeze
            author: User unfreezing the version
        """
        metadata = self.get_version(asset_or_feature_set, version)
        metadata.unfreeze(author)
        self._save_version_metadata(asset_or_feature_set, version, metadata)
        logger.info(f"Unfroze version {self.layer}/{version}/{asset_or_feature_set}")
    
    def is_frozen(
        self,
        asset_or_feature_set: str,
        version: SemanticVersion,
    ) -> bool:
        """Check if a version is frozen."""
        metadata = self.get_version(asset_or_feature_set, version)
        return metadata.frozen
    
    # ========================================================================
    # Deprecation
    # ========================================================================
    
    def deprecate_version(
        self,
        asset_or_feature_set: str,
        version: SemanticVersion,
        replacement_version: Optional[SemanticVersion] = None,
        author: str = "",
    ) -> None:
        """
        Deprecate a version.
        
        Args:
            asset_or_feature_set: Asset name or feature set name
            version: SemanticVersion to deprecate
            replacement_version: Optional replacement version to migrate to
            author: User deprecating the version
        """
        metadata = self.get_version(asset_or_feature_set, version)
        
        metadata.add_tag("deprecated")
        
        description = f"Version deprecated"
        if replacement_version:
            description += f", migrate to {replacement_version}"
        
        metadata.add_changelog_entry(
            change_type="deprecation",
            description=description,
            author=author,
        )
        
        self._save_version_metadata(asset_or_feature_set, version, metadata)
        logger.info(
            f"Deprecated version {self.layer}/{version}/{asset_or_feature_set}"
        )
    
    def get_active_versions(
        self,
        asset_or_feature_set: Optional[str] = None,
    ) -> List[VersionMetadata]:
        """
        Get all active (non-deprecated) versions.
        
        Args:
            asset_or_feature_set: Optional asset to filter by
        
        Returns:
            List of active VersionMetadata
        """
        all_versions = self.list_versions(asset_or_feature_set)
        active = [v for v in all_versions if "deprecated" not in v.tags]
        return active
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _get_version_directory(
        self,
        asset_or_feature_set: str,
        version: SemanticVersion,
    ) -> Path:
        """Get directory path for a version."""
        return self.base_path / asset_or_feature_set / str(version)
    
    def _save_version_metadata(
        self,
        asset_or_feature_set: str,
        version: SemanticVersion,
        metadata: VersionMetadata,
    ) -> None:
        """Save metadata to disk."""
        version_dir = self._get_version_directory(asset_or_feature_set, version)
        MetadataPersistence.save(metadata, version_dir)
