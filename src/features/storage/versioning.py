"""
Version freeze and comparison utilities for feature storage.

Handles:
- Version freezing/unfreezing
- Schema comparison between versions
- Computation parameter comparison
- Dependency analysis
- Changelog inspection
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from .metadata import FeatureStorageMetadata, FeatureMetadataPersistence
from src.storage.version_metadata import SemanticVersion

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class VersioningError(Exception):
    """Base exception for versioning errors."""
    pass


class VersionFrozenError(VersioningError):
    """Raised when attempting to modify a frozen version."""
    pass


class VersionComparisonError(VersioningError):
    """Raised when version comparison fails."""
    pass


# ============================================================================
# Version Manager
# ============================================================================

class FeatureVersionManager:
    """Manages feature version lifecycle (freeze, deprecation, etc.)."""
    
    def __init__(self, base_path: Path = Path("features")):
        """
        Initialize the version manager.
        
        Args:
            base_path: Base directory for feature storage
        """
        self.base_path = Path(base_path)
    
    def freeze_version(
        self,
        asset: str,
        version: str,
        author: str = "",
    ) -> None:
        """
        Freeze a version to prevent modifications.
        
        Args:
            asset: Asset identifier
            version: Semantic version
            author: User freezing the version
        
        Raises:
            VersioningError: If version doesn't exist
        """
        version_dir = self._get_version_directory(asset, version)
        
        if not FeatureMetadataPersistence.exists(version_dir):
            raise VersioningError(f"Version not found: {version}/{asset}")
        
        metadata = FeatureMetadataPersistence.load(version_dir)
        metadata.freeze(author)
        FeatureMetadataPersistence.save(metadata, version_dir)
        
        logger.info(f"Froze feature version {version}/{asset}")
    
    def unfreeze_version(
        self,
        asset: str,
        version: str,
        author: str = "",
    ) -> None:
        """
        Unfreeze a version to allow modifications.
        
        Args:
            asset: Asset identifier
            version: Semantic version
            author: User unfreezing the version
        
        Raises:
            VersioningError: If version doesn't exist
        """
        version_dir = self._get_version_directory(asset, version)
        
        if not FeatureMetadataPersistence.exists(version_dir):
            raise VersioningError(f"Version not found: {version}/{asset}")
        
        metadata = FeatureMetadataPersistence.load(version_dir)
        metadata.unfreeze(author)
        FeatureMetadataPersistence.save(metadata, version_dir)
        
        logger.info(f"Unfroze feature version {version}/{asset}")
    
    def is_frozen(self, asset: str, version: str) -> bool:
        """Check if a version is frozen."""
        version_dir = self._get_version_directory(asset, version)
        
        if not FeatureMetadataPersistence.exists(version_dir):
            raise VersioningError(f"Version not found: {version}/{asset}")
        
        metadata = FeatureMetadataPersistence.load(version_dir)
        return metadata.frozen
    
    def add_tag(
        self,
        asset: str,
        version: str,
        tag: str,
    ) -> None:
        """
        Add a tag to a version.
        
        Args:
            asset: Asset identifier
            version: Semantic version
            tag: Tag to add
        """
        version_dir = self._get_version_directory(asset, version)
        
        if not FeatureMetadataPersistence.exists(version_dir):
            raise VersioningError(f"Version not found: {version}/{asset}")
        
        metadata = FeatureMetadataPersistence.load(version_dir)
        metadata.add_tag(tag)
        FeatureMetadataPersistence.save(metadata, version_dir)
    
    def remove_tag(
        self,
        asset: str,
        version: str,
        tag: str,
    ) -> None:
        """
        Remove a tag from a version.
        
        Args:
            asset: Asset identifier
            version: Semantic version
            tag: Tag to remove
        """
        version_dir = self._get_version_directory(asset, version)
        
        if not FeatureMetadataPersistence.exists(version_dir):
            raise VersioningError(f"Version not found: {version}/{asset}")
        
        metadata = FeatureMetadataPersistence.load(version_dir)
        metadata.remove_tag(tag)
        FeatureMetadataPersistence.save(metadata, version_dir)
    
    def deprecate_version(
        self,
        asset: str,
        version: str,
        replacement_version: Optional[str] = None,
        author: str = "",
    ) -> None:
        """
        Deprecate a version.
        
        Args:
            asset: Asset identifier
            version: Semantic version to deprecate
            replacement_version: Optional replacement version
            author: User deprecating the version
        """
        version_dir = self._get_version_directory(asset, version)
        
        if not FeatureMetadataPersistence.exists(version_dir):
            raise VersioningError(f"Version not found: {version}/{asset}")
        
        metadata = FeatureMetadataPersistence.load(version_dir)
        metadata.add_tag("deprecated")
        
        description = f"Version deprecated"
        if replacement_version:
            description += f", migrate to {replacement_version}"
        
        metadata.add_changelog_entry(
            change_type="deprecation",
            description=description,
            author=author,
        )
        
        FeatureMetadataPersistence.save(metadata, version_dir)
        logger.info(f"Deprecated feature version {version}/{asset}")
    
    def get_changelog(
        self,
        asset: str,
        version: str,
    ) -> List[Dict[str, Any]]:
        """
        Get the changelog for a version.
        
        Args:
            asset: Asset identifier
            version: Semantic version
        
        Returns:
            List of changelog entries
        """
        version_dir = self._get_version_directory(asset, version)
        
        if not FeatureMetadataPersistence.exists(version_dir):
            raise VersioningError(f"Version not found: {version}/{asset}")
        
        metadata = FeatureMetadataPersistence.load(version_dir)
        return metadata.changelog
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _get_version_directory(self, asset: str, version: str) -> Path:
        """Get directory path for a feature version."""
        return self.base_path / f"v{version}" / asset


# ============================================================================
# Version Comparator
# ============================================================================

class FeatureVersionComparator:
    """Compare features across different versions."""
    
    def __init__(self, base_path: Path = Path("features")):
        """
        Initialize the comparator.
        
        Args:
            base_path: Base directory for feature storage
        """
        self.base_path = Path(base_path)
    
    def compare_schemas(
        self,
        asset: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """
        Compare schemas between two versions.
        
        Args:
            asset: Asset identifier
            version1: First semantic version
            version2: Second semantic version
        
        Returns:
            Dictionary with comparison results
        """
        metadata1 = self._load_metadata(asset, version1)
        metadata2 = self._load_metadata(asset, version2)
        
        features1 = {f.feature_name: f.schema for f in metadata1.features}
        features2 = {f.feature_name: f.schema for f in metadata2.features}
        
        added_features = set(features2.keys()) - set(features1.keys())
        removed_features = set(features1.keys()) - set(features2.keys())
        
        modified_features = []
        for fname in set(features1.keys()) & set(features2.keys()):
            schema1 = {s.column_name: s.data_type for s in features1[fname]}
            schema2 = {s.column_name: s.data_type for s in features2[fname]}
            
            if schema1 != schema2:
                modified_features.append({
                    "feature_name": fname,
                    "v1_schema": schema1,
                    "v2_schema": schema2,
                })
        
        return {
            "asset": asset,
            "version1": version1,
            "version2": version2,
            "added_features": sorted(list(added_features)),
            "removed_features": sorted(list(removed_features)),
            "modified_features": modified_features,
            "schema_compatible": len(removed_features) == 0 and len(modified_features) == 0,
        }
    
    def compare_computation_parameters(
        self,
        asset: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """
        Compare computation parameters between versions.
        
        Args:
            asset: Asset identifier
            version1: First semantic version
            version2: Second semantic version
        
        Returns:
            Dictionary with parameter changes
        """
        metadata1 = self._load_metadata(asset, version1)
        metadata2 = self._load_metadata(asset, version2)
        
        params1 = {
            f.feature_name: {p.name: p.value for p in f.computation_parameters}
            for f in metadata1.features
        }
        params2 = {
            f.feature_name: {p.name: p.value for p in f.computation_parameters}
            for f in metadata2.features
        }
        
        changes = []
        
        for fname in set(params1.keys()) & set(params2.keys()):
            p1 = params1[fname]
            p2 = params2[fname]
            
            for param_name in set(p1.keys()) | set(p2.keys()):
                v1 = p1.get(param_name)
                v2 = p2.get(param_name)
                
                if v1 != v2:
                    changes.append({
                        "feature_name": fname,
                        "parameter_name": param_name,
                        "v1_value": v1,
                        "v2_value": v2,
                    })
        
        return {
            "asset": asset,
            "version1": version1,
            "version2": version2,
            "parameter_changes": changes,
            "parameters_compatible": len(changes) == 0,
        }
    
    def compare_dependencies(
        self,
        asset: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """
        Compare feature dependencies between versions.
        
        Args:
            asset: Asset identifier
            version1: First semantic version
            version2: Second semantic version
        
        Returns:
            Dictionary with dependency changes
        """
        metadata1 = self._load_metadata(asset, version1)
        metadata2 = self._load_metadata(asset, version2)
        
        deps1 = {
            f.feature_name: {d.feature_name for d in f.dependencies}
            for f in metadata1.features
        }
        deps2 = {
            f.feature_name: {d.feature_name for d in f.dependencies}
            for f in metadata2.features
        }
        
        changes = []
        
        for fname in set(deps1.keys()) & set(deps2.keys()):
            added = deps2[fname] - deps1[fname]
            removed = deps1[fname] - deps2[fname]
            
            if added or removed:
                changes.append({
                    "feature_name": fname,
                    "added_dependencies": sorted(list(added)),
                    "removed_dependencies": sorted(list(removed)),
                })
        
        return {
            "asset": asset,
            "version1": version1,
            "version2": version2,
            "dependency_changes": changes,
            "dependencies_compatible": len(changes) == 0,
        }
    
    def compare_complete(
        self,
        asset: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """
        Complete comparison between two versions.
        
        Args:
            asset: Asset identifier
            version1: First semantic version
            version2: Second semantic version
        
        Returns:
            Comprehensive comparison dictionary
        """
        metadata1 = self._load_metadata(asset, version1)
        metadata2 = self._load_metadata(asset, version2)
        
        return {
            "asset": asset,
            "version1": version1,
            "version2": version2,
            "metadata": {
                "v1_created": metadata1.creation_timestamp.isoformat(),
                "v2_created": metadata2.creation_timestamp.isoformat(),
                "v1_records": metadata1.record_count,
                "v2_records": metadata2.record_count,
                "v1_source_data": metadata1.source_data_version,
                "v2_source_data": metadata2.source_data_version,
            },
            "schema_comparison": self.compare_schemas(asset, version1, version2),
            "parameters_comparison": self.compare_computation_parameters(asset, version1, version2),
            "dependencies_comparison": self.compare_dependencies(asset, version1, version2),
        }
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _load_metadata(self, asset: str, version: str) -> FeatureStorageMetadata:
        """Load metadata for a version."""
        version_dir = self.base_path / f"v{version}" / asset
        
        if not FeatureMetadataPersistence.exists(version_dir):
            raise VersionComparisonError(f"Version not found: {version}/{asset}")
        
        return FeatureMetadataPersistence.load(version_dir)


# ============================================================================
# Version Analyzer
# ============================================================================

class FeatureVersionAnalyzer:
    """Analyze version lineage and evolution."""
    
    def __init__(self, base_path: Path = Path("features")):
        """
        Initialize the analyzer.
        
        Args:
            base_path: Base directory for feature storage
        """
        self.base_path = Path(base_path)
        self.manager = FeatureVersionManager(base_path)
    
    def get_version_history(
        self,
        asset: str,
    ) -> List[Dict[str, Any]]:
        """
        Get version history for an asset.
        
        Args:
            asset: Asset identifier
        
        Returns:
            List of version information sorted chronologically
        """
        history = []
        
        for version_dir in sorted(self.base_path.glob(f"v*/{asset}")):
            try:
                metadata = FeatureMetadataPersistence.load(version_dir)
                history.append({
                    "version": metadata.version,
                    "created_at": metadata.creation_timestamp.isoformat(),
                    "source_data_version": metadata.source_data_version,
                    "record_count": metadata.record_count,
                    "frozen": metadata.frozen,
                    "tags": metadata.tags,
                    "features": len(metadata.features),
                })
            except Exception as e:
                logger.warning(f"Failed to load metadata from {version_dir}: {e}")
        
        # Sort by version
        history.sort(
            key=lambda x: SemanticVersion.parse(x["version"]),
            reverse=True
        )
        
        return history
    
    def get_breaking_changes(
        self,
        asset: str,
    ) -> List[Dict[str, Any]]:
        """
        Identify breaking changes across versions.
        
        Args:
            asset: Asset identifier
        
        Returns:
            List of breaking changes
        """
        history = self.get_version_history(asset)
        breaking_changes = []
        
        comparator = FeatureVersionComparator(self.base_path)
        
        for i in range(len(history) - 1):
            v1 = history[i + 1]["version"]
            v2 = history[i]["version"]
            
            try:
                schema_comp = comparator.compare_schemas(asset, v1, v2)
                
                if schema_comp["removed_features"] or schema_comp["modified_features"]:
                    breaking_changes.append({
                        "from_version": v1,
                        "to_version": v2,
                        "breaking_changes": {
                            "removed_features": schema_comp["removed_features"],
                            "modified_features": schema_comp["modified_features"],
                        }
                    })
            except Exception as e:
                logger.warning(f"Failed to compare {v1} and {v2}: {e}")
        
        return breaking_changes
