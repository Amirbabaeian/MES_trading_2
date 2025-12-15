"""
Feature loading and retrieval API.

Handles:
- Efficient loading by date range
- Filtering by asset and version
- Loading subsets of features
- Support for partitioned datasets
- Caching for performance
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import glob

import pandas as pd

from .metadata import FeatureStorageMetadata, FeatureMetadataPersistence
from src.storage.version_metadata import SemanticVersion

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class FeatureReaderError(Exception):
    """Base exception for feature reader errors."""
    pass


class VersionNotFoundError(FeatureReaderError):
    """Raised when requested version doesn't exist."""
    pass


class FeatureNotFoundError(FeatureReaderError):
    """Raised when requested feature doesn't exist."""
    pass


# ============================================================================
# Feature Reader
# ============================================================================

class FeatureReader:
    """
    Loads computed features from versioned storage.
    
    Supports:
    - Loading by date range
    - Filtering by asset and version
    - Loading subsets of features
    - Reading from partitioned datasets
    - Metadata inspection
    """
    
    def __init__(
        self,
        base_path: Path = Path("features"),
        cache_enabled: bool = True,
        max_cache_size_mb: int = 500,
    ):
        """
        Initialize the feature reader.
        
        Args:
            base_path: Base directory for feature storage
            cache_enabled: Whether to cache loaded features
            max_cache_size_mb: Maximum cache size in MB
        """
        self.base_path = Path(base_path)
        self.cache_enabled = cache_enabled
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_size_bytes = 0
        
        if not self.base_path.exists():
            raise FeatureReaderError(f"Feature base path not found: {self.base_path}")
        
        logger.info(f"Initialized FeatureReader with base_path: {self.base_path}")
    
    def load_features(
        self,
        asset: str,
        version: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load features for an asset and version.
        
        Args:
            asset: Asset identifier (ES, MES, VIX, etc.)
            version: Semantic version string (e.g., "1.0.0")
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            feature_names: Optional list of specific features to load
                          (if None, load all available features)
        
        Returns:
            DataFrame with loaded features
        
        Raises:
            VersionNotFoundError: If version doesn't exist
            FeatureNotFoundError: If requested features don't exist
            FeatureReaderError: If loading fails
        """
        # Get version directory
        version_dir = self._get_version_directory(asset, version)
        if not version_dir.exists():
            raise VersionNotFoundError(
                f"Feature version not found: {version}/{asset}"
            )
        
        # Load metadata
        metadata = self.get_metadata(asset, version)
        
        # Validate requested features exist
        if feature_names:
            available_features = {f.feature_name for f in metadata.features}
            requested_features = set(feature_names)
            missing_features = requested_features - available_features
            
            if missing_features:
                raise FeatureNotFoundError(
                    f"Features not found: {missing_features}"
                )
        
        # Load data from Parquet
        features_df = self._load_parquet_data(version_dir, metadata)
        
        # Filter by date range
        if start_date or end_date:
            features_df = self._filter_by_date_range(
                features_df,
                start_date,
                end_date,
            )
        
        # Filter by feature names
        if feature_names:
            cols_to_keep = ['timestamp'] + [
                f for f in feature_names if f in features_df.columns
            ]
            features_df = features_df[cols_to_keep]
        
        logger.info(
            f"Loaded {len(features_df)} records for {asset}/{version} "
            f"({len(features_df.columns)-1} features)"
        )
        
        return features_df
    
    def load_latest_version(
        self,
        asset: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Load features from the latest version of an asset.
        
        Args:
            asset: Asset identifier
            start_date: Optional start date
            end_date: Optional end date
            feature_names: Optional feature names to load
        
        Returns:
            Tuple of (DataFrame, version_string)
        """
        # Find latest version
        latest_version = self.get_latest_version(asset)
        if not latest_version:
            raise VersionNotFoundError(f"No versions found for asset: {asset}")
        
        features_df = self.load_features(
            asset=asset,
            version=latest_version,
            start_date=start_date,
            end_date=end_date,
            feature_names=feature_names,
        )
        
        return features_df, latest_version
    
    def load_multiple_assets(
        self,
        assets: List[str],
        version: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load features for multiple assets.
        
        Args:
            assets: List of asset identifiers
            version: Semantic version (same for all assets)
            start_date: Optional start date
            end_date: Optional end date
            feature_names: Optional feature names to load
        
        Returns:
            Dictionary mapping asset names to DataFrames
        """
        results = {}
        
        for asset in assets:
            try:
                df = self.load_features(
                    asset=asset,
                    version=version,
                    start_date=start_date,
                    end_date=end_date,
                    feature_names=feature_names,
                )
                results[asset] = df
                logger.info(f"Successfully loaded features for {asset}")
            except (VersionNotFoundError, FeatureNotFoundError) as e:
                logger.warning(f"Failed to load features for {asset}: {e}")
        
        return results
    
    def get_metadata(self, asset: str, version: str) -> FeatureStorageMetadata:
        """
        Get metadata for a feature version.
        
        Args:
            asset: Asset identifier
            version: Semantic version
        
        Returns:
            FeatureStorageMetadata object
        
        Raises:
            VersionNotFoundError: If version doesn't exist
        """
        version_dir = self._get_version_directory(asset, version)
        
        if not FeatureMetadataPersistence.exists(version_dir):
            raise VersionNotFoundError(
                f"Feature metadata not found: {version}/{asset}"
            )
        
        return FeatureMetadataPersistence.load(version_dir)
    
    def list_versions(
        self,
        asset: Optional[str] = None,
        include_frozen: bool = True,
        include_deprecated: bool = False,
    ) -> Dict[str, List[str]]:
        """
        List all available feature versions.
        
        Args:
            asset: Optional asset to filter by
            include_frozen: Whether to include frozen versions
            include_deprecated: Whether to include deprecated versions
        
        Returns:
            Dictionary mapping assets to lists of version strings
        """
        results = {}
        
        if asset:
            assets = [asset]
        else:
            # Discover all assets
            assets = []
            for version_dir in self.base_path.glob("v*/"):
                for asset_dir in version_dir.glob("*/"):
                    if asset_dir.is_dir() and asset_dir.name not in assets:
                        assets.append(asset_dir.name)
        
        for asset_name in sorted(assets):
            versions = []
            
            for version_dir in self.base_path.glob(f"v*/{ asset_name}"):
                if not version_dir.is_dir():
                    continue
                
                try:
                    metadata = FeatureMetadataPersistence.load(version_dir)
                    
                    # Filter by frozen status
                    if not include_frozen and metadata.frozen:
                        continue
                    
                    # Filter by deprecated status
                    is_deprecated = "deprecated" in metadata.tags
                    if not include_deprecated and is_deprecated:
                        continue
                    
                    versions.append(metadata.version)
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {version_dir}: {e}")
            
            if versions:
                results[asset_name] = sorted(
                    versions,
                    key=SemanticVersion.parse,
                    reverse=True
                )
        
        return results
    
    def get_latest_version(
        self,
        asset: str,
        include_deprecated: bool = False,
    ) -> Optional[str]:
        """
        Get the latest version for an asset.
        
        Args:
            asset: Asset identifier
            include_deprecated: Whether to consider deprecated versions
        
        Returns:
            Latest version string or None if no versions exist
        """
        versions = self.list_versions(
            asset=asset,
            include_deprecated=include_deprecated,
        )
        
        if asset in versions and versions[asset]:
            return versions[asset][0]  # Already sorted in reverse
        
        return None
    
    def list_features(
        self,
        asset: str,
        version: str,
    ) -> List[str]:
        """
        List all available features in a version.
        
        Args:
            asset: Asset identifier
            version: Semantic version
        
        Returns:
            List of feature names
        """
        metadata = self.get_metadata(asset, version)
        return [f.feature_name for f in metadata.features]
    
    def get_feature_dependencies(
        self,
        asset: str,
        version: str,
        feature_name: str,
    ) -> List[str]:
        """
        Get dependencies for a feature.
        
        Args:
            asset: Asset identifier
            version: Semantic version
            feature_name: Feature name
        
        Returns:
            List of dependency feature names
        """
        metadata = self.get_metadata(asset, version)
        
        for feature in metadata.features:
            if feature.feature_name == feature_name:
                return [d.feature_name for d in feature.dependencies]
        
        raise FeatureNotFoundError(f"Feature not found: {feature_name}")
    
    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._cache.clear()
        self._cache_size_bytes = 0
        logger.info("Cleared feature cache")
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _get_version_directory(self, asset: str, version: str) -> Path:
        """Get directory path for a feature version."""
        return self.base_path / f"v{version}" / asset
    
    def _load_parquet_data(
        self,
        version_dir: Path,
        metadata: FeatureStorageMetadata,
    ) -> pd.DataFrame:
        """Load Parquet data from a version directory."""
        # Check cache first
        cache_key = f"{version_dir.parent.name}/{version_dir.name}"
        if self.cache_enabled and cache_key in self._cache:
            logger.info(f"Loaded from cache: {cache_key}")
            return self._cache[cache_key].copy()
        
        # Try loading from single file first
        single_file = version_dir / "data.parquet"
        if single_file.exists():
            df = pd.read_parquet(single_file)
        else:
            # Try loading from partitioned data
            data_dir = version_dir / "data"
            if data_dir.exists():
                parquet_files = sorted(glob.glob(str(data_dir / "**" / "data.parquet")))
                if parquet_files:
                    dfs = [pd.read_parquet(f) for f in parquet_files]
                    df = pd.concat(dfs, ignore_index=True)
                else:
                    raise FeatureReaderError(
                        f"No Parquet files found in {version_dir}"
                    )
            else:
                raise FeatureReaderError(
                    f"No feature data found in {version_dir}"
                )
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Cache the data
        if self.cache_enabled:
            df_size = df.memory_usage(deep=True).sum()
            if self._cache_size_bytes + df_size <= self.max_cache_size_bytes:
                self._cache[cache_key] = df.copy()
                self._cache_size_bytes += df_size
                logger.info(f"Cached {cache_key} ({df_size / 1024 / 1024:.1f} MB)")
            else:
                logger.warning(f"Cache full, skipping {cache_key}")
        
        return df
    
    def _filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if start_date:
            df = df[df['timestamp'] >= pd.Timestamp(start_date)]
        
        if end_date:
            df = df[df['timestamp'] <= pd.Timestamp(end_date)]
        
        return df


# ============================================================================
# Feature Catalog
# ============================================================================

class FeatureCatalog:
    """Utility for browsing available features across versions and assets."""
    
    def __init__(self, reader: FeatureReader):
        """
        Initialize the catalog.
        
        Args:
            reader: FeatureReader instance
        """
        self.reader = reader
    
    def get_catalog(
        self,
        include_frozen: bool = False,
        include_deprecated: bool = False,
    ) -> Dict[str, Any]:
        """
        Get complete feature catalog.
        
        Args:
            include_frozen: Whether to include frozen versions
            include_deprecated: Whether to include deprecated versions
        
        Returns:
            Dictionary with catalog information
        """
        versions = self.reader.list_versions(
            include_frozen=include_frozen,
            include_deprecated=include_deprecated,
        )
        
        catalog = {}
        
        for asset, asset_versions in sorted(versions.items()):
            catalog[asset] = {}
            
            for version in asset_versions:
                try:
                    metadata = self.reader.get_metadata(asset, version)
                    features = self.reader.list_features(asset, version)
                    
                    catalog[asset][version] = {
                        "created_at": metadata.creation_timestamp.isoformat(),
                        "source_data_version": metadata.source_data_version,
                        "record_count": metadata.record_count,
                        "date_range": {
                            "start": metadata.start_date.isoformat(),
                            "end": metadata.end_date.isoformat(),
                        },
                        "features": sorted(features),
                        "tags": metadata.tags,
                        "frozen": metadata.frozen,
                        "environment": metadata.environment,
                    }
                except Exception as e:
                    logger.warning(f"Failed to get catalog for {asset}/{version}: {e}")
        
        return catalog
    
    def print_catalog(self) -> None:
        """Print a human-readable feature catalog."""
        catalog = self.get_catalog()
        
        print("\n" + "=" * 80)
        print("FEATURE CATALOG")
        print("=" * 80 + "\n")
        
        for asset, versions in sorted(catalog.items()):
            print(f"{asset}:")
            for version, info in sorted(versions.items(), reverse=True):
                tags_str = ", ".join(info["tags"]) if info["tags"] else "none"
                print(f"  v{version}")
                print(f"    Created: {info['created_at']}")
                print(f"    Records: {info['record_count']}")
                print(f"    Features: {len(info['features'])}")
                print(f"    Tags: {tags_str}")
                print(f"    Frozen: {info['frozen']}")
                print()
