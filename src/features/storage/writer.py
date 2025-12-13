"""
Feature writing and versioning logic.

Handles:
- Writing computed features to Parquet files
- Managing semantic versions for feature releases
- Metadata generation and persistence
- Validation of feature data consistency
- Version freezing for immutability
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from .metadata import (
    FeatureStorageMetadata,
    FeatureMetadataPersistence,
    FeatureMetadataBuilder,
    FeatureComputationMetadata,
    ComputationParameter,
    FeatureDependency,
    FeatureSchema,
)
from src.data_io.parquet_utils import write_parquet_validated
from src.data_io.schemas import DataSchema, ColumnSchema, SchemaEnforcementMode
from src.storage.version_metadata import SemanticVersion

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class FeatureWriterError(Exception):
    """Base exception for feature writer errors."""
    pass


class VersionFrozenError(FeatureWriterError):
    """Raised when attempting to write to a frozen version."""
    pass


class DataConsistencyError(FeatureWriterError):
    """Raised when feature data is inconsistent."""
    pass


class VersionIncrementError(FeatureWriterError):
    """Raised when version increment fails."""
    pass


# ============================================================================
# Feature Writer
# ============================================================================

class FeatureWriter:
    """
    Writes computed features to versioned Parquet storage.
    
    Manages:
    - Directory structure (features/v{X}/{asset}/)
    - Parquet file writing with validation
    - Metadata generation and storage
    - Version management and freezing
    - Data validation and consistency checks
    """
    
    def __init__(
        self,
        base_path: Path = Path("features"),
        compression: str = "snappy",
        validation_enabled: bool = True,
        auto_freeze_on_write: bool = False,
    ):
        """
        Initialize the feature writer.
        
        Args:
            base_path: Base directory for feature storage
            compression: Parquet compression codec (snappy, gzip, zstd)
            validation_enabled: Whether to validate data before writing
            auto_freeze_on_write: Whether to auto-freeze versions after writing
        """
        self.base_path = Path(base_path)
        self.compression = compression
        self.validation_enabled = validation_enabled
        self.auto_freeze_on_write = auto_freeze_on_write
        
        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized FeatureWriter with base_path: {self.base_path}")
    
    def write_features(
        self,
        features_df: pd.DataFrame,
        asset: str,
        version: str,
        feature_set_name: str,
        source_data_version: str,
        feature_metadata_list: List[FeatureComputationMetadata],
        creator: str = "",
        environment: str = "production",
        tags: Optional[List[str]] = None,
        partition_by_date: bool = False,
    ) -> Path:
        """
        Write computed features to storage.
        
        Args:
            features_df: DataFrame with computed features (must have 'timestamp' column)
            asset: Asset identifier (ES, MES, VIX, etc.)
            version: Semantic version string (e.g., "1.0.0")
            feature_set_name: Name of the feature set
            source_data_version: Version of cleaned data used as source
            feature_metadata_list: List of feature metadata objects
            creator: User creating this version
            environment: Environment (production, staging, development)
            tags: Optional list of tags for the version
            partition_by_date: Whether to partition Parquet by date
        
        Returns:
            Path to the stored features directory
        
        Raises:
            FeatureWriterError: If writing fails
            DataConsistencyError: If data validation fails
            VersionFrozenError: If version is frozen
        """
        # Validate input
        self._validate_write_input(features_df, asset, version)
        
        # Create version directory
        version_dir = self._get_version_directory(asset, version)
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if version is frozen
        metadata_path = version_dir / FeatureMetadataPersistence.METADATA_FILENAME
        if metadata_path.exists():
            existing_metadata = FeatureMetadataPersistence.load(version_dir)
            if existing_metadata.frozen:
                raise VersionFrozenError(
                    f"Cannot write to frozen version {version}/{asset}"
                )
        
        # Extract date range from timestamp column
        start_date = features_df['timestamp'].min()
        end_date = features_df['timestamp'].max()
        
        # Build metadata
        metadata_builder = FeatureMetadataBuilder(
            version=version,
            asset=asset,
            feature_set_name=feature_set_name,
            source_data_version=source_data_version,
        )
        
        metadata = (
            metadata_builder
            .set_date_range(start_date, end_date)
            .set_record_count(len(features_df))
            .set_compression(self.compression)
            .set_partition_strategy("by_date" if partition_by_date else "none")
            .set_creator(creator)
            .set_environment(environment)
        )
        
        # Add feature metadata
        for feature_meta in feature_metadata_list:
            metadata.add_feature(feature_meta)
        
        metadata_obj = metadata.build()
        
        # Add tags
        if tags:
            for tag in tags:
                metadata_obj.add_tag(tag)
        
        # Validate data consistency
        if self.validation_enabled:
            self._validate_data_consistency(features_df, feature_metadata_list)
        
        # Write Parquet file(s)
        if partition_by_date:
            self._write_partitioned(features_df, version_dir)
        else:
            self._write_single_file(features_df, version_dir)
        
        # Save metadata
        FeatureMetadataPersistence.save(metadata_obj, version_dir)
        logger.info(
            f"Wrote {len(features_df)} feature records to {version_dir}"
        )
        
        # Auto-freeze if enabled
        if self.auto_freeze_on_write:
            metadata_obj.freeze(creator)
            FeatureMetadataPersistence.save(metadata_obj, version_dir)
            logger.info(f"Auto-froze feature version {version}/{asset}")
        
        return version_dir
    
    def write_incremental(
        self,
        new_features_df: pd.DataFrame,
        asset: str,
        version: str,
        feature_metadata_list: List[FeatureComputationMetadata],
        creator: str = "",
    ) -> Path:
        """
        Incrementally add new feature data to an existing version.
        
        Args:
            new_features_df: New feature data to add
            asset: Asset identifier
            version: Semantic version to update
            feature_metadata_list: Feature metadata
            creator: User making the update
        
        Returns:
            Path to the updated version directory
        
        Raises:
            FeatureWriterError: If incremental write fails
            VersionFrozenError: If version is frozen
        """
        version_dir = self._get_version_directory(asset, version)
        
        if not version_dir.exists():
            raise FeatureWriterError(
                f"Version {version}/{asset} does not exist"
            )
        
        # Load existing metadata
        metadata = FeatureMetadataPersistence.load(version_dir)
        metadata.validate_frozen()
        
        # Load existing features
        existing_features = self._load_existing_features(version_dir)
        
        # Combine with new data
        combined_features = pd.concat(
            [existing_features, new_features_df],
            ignore_index=False
        )
        combined_features = combined_features.drop_duplicates(
            subset=['timestamp'],
            keep='last'
        ).sort_values('timestamp')
        
        # Update metadata
        metadata.record_count = len(combined_features)
        metadata.end_date = combined_features['timestamp'].max()
        metadata.add_changelog_entry(
            change_type="incremental_update",
            description=f"Added {len(new_features_df)} new records",
            author=creator,
        )
        
        # Rewrite features
        self._write_single_file(combined_features, version_dir)
        
        # Save updated metadata
        FeatureMetadataPersistence.save(metadata, version_dir)
        logger.info(
            f"Incrementally updated {version}/{asset} with {len(new_features_df)} records"
        )
        
        return version_dir
    
    def promote_version(
        self,
        asset: str,
        version: str,
        creator: str = "",
    ) -> None:
        """
        Promote a version to production by tagging it.
        
        Args:
            asset: Asset identifier
            version: Version to promote
            creator: User promoting the version
        """
        version_dir = self._get_version_directory(asset, version)
        
        if not version_dir.exists():
            raise FeatureWriterError(f"Version {version}/{asset} does not exist")
        
        metadata = FeatureMetadataPersistence.load(version_dir)
        metadata.add_tag("production")
        metadata.add_changelog_entry(
            change_type="promotion",
            description="Promoted to production",
            author=creator,
        )
        
        FeatureMetadataPersistence.save(metadata, version_dir)
        logger.info(f"Promoted {version}/{asset} to production")
    
    def freeze_version(
        self,
        asset: str,
        version: str,
        creator: str = "",
    ) -> None:
        """
        Freeze a version to prevent modifications.
        
        Args:
            asset: Asset identifier
            version: Version to freeze
            creator: User freezing the version
        """
        version_dir = self._get_version_directory(asset, version)
        
        if not version_dir.exists():
            raise FeatureWriterError(f"Version {version}/{asset} does not exist")
        
        metadata = FeatureMetadataPersistence.load(version_dir)
        metadata.freeze(creator)
        
        FeatureMetadataPersistence.save(metadata, version_dir)
        logger.info(f"Froze feature version {version}/{asset}")
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _get_version_directory(self, asset: str, version: str) -> Path:
        """Get directory path for a feature version."""
        return self.base_path / f"v{version}" / asset
    
    def _validate_write_input(
        self,
        features_df: pd.DataFrame,
        asset: str,
        version: str,
    ) -> None:
        """Validate input parameters for writing."""
        if features_df.empty:
            raise FeatureWriterError("Cannot write empty DataFrame")
        
        if 'timestamp' not in features_df.columns:
            raise FeatureWriterError("DataFrame must have 'timestamp' column")
        
        if not asset or not isinstance(asset, str):
            raise FeatureWriterError("Asset must be a non-empty string")
        
        # Validate semantic version
        try:
            SemanticVersion.parse(version)
        except ValueError as e:
            raise FeatureWriterError(f"Invalid semantic version: {version}") from e
    
    def _validate_data_consistency(
        self,
        features_df: pd.DataFrame,
        feature_metadata_list: List[FeatureComputationMetadata],
    ) -> None:
        """Validate feature data consistency."""
        # Check for missing dates
        if features_df['timestamp'].isna().any():
            raise DataConsistencyError("DataFrame contains null timestamps")
        
        # Check for NaN in feature columns
        feature_cols = [f.feature_name for f in feature_metadata_list]
        for col in feature_cols:
            if col in features_df.columns:
                nan_count = features_df[col].isna().sum()
                if nan_count > 0:
                    logger.warning(
                        f"Feature '{col}' has {nan_count} NaN values "
                        f"({nan_count/len(features_df)*100:.2f}%)"
                    )
        
        # Check for duplicates
        duplicate_count = features_df['timestamp'].duplicated().sum()
        if duplicate_count > 0:
            logger.warning(
                f"DataFrame has {duplicate_count} duplicate timestamps"
            )
        
        # Check date ordering
        if not features_df['timestamp'].is_monotonic_increasing:
            logger.warning("DataFrame is not sorted by timestamp")
    
    def _write_single_file(
        self,
        features_df: pd.DataFrame,
        version_dir: Path,
    ) -> Path:
        """Write features to a single Parquet file."""
        output_path = version_dir / "data.parquet"
        
        features_df.to_parquet(
            output_path,
            index=False,
            compression=self.compression,
            engine='pyarrow',
        )
        
        logger.info(f"Wrote features to {output_path}")
        return output_path
    
    def _write_partitioned(
        self,
        features_df: pd.DataFrame,
        version_dir: Path,
    ) -> List[Path]:
        """Write features partitioned by date."""
        # Extract year-month from timestamp for partitioning
        features_df['year_month'] = features_df['timestamp'].dt.to_period('M')
        
        written_files = []
        for period, group_df in features_df.groupby('year_month'):
            # Create partition directory
            partition_dir = version_dir / "data" / str(period).replace('-', '/')
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Remove the partition column before saving
            group_df_clean = group_df.drop('year_month', axis=1)
            
            # Write partition file
            partition_path = partition_dir / "data.parquet"
            group_df_clean.to_parquet(
                partition_path,
                index=False,
                compression=self.compression,
                engine='pyarrow',
            )
            
            written_files.append(partition_path)
            logger.info(f"Wrote partition {period} to {partition_path}")
        
        return written_files
    
    def _load_existing_features(self, version_dir: Path) -> pd.DataFrame:
        """Load existing feature data from a version directory."""
        # Try loading from single file first
        single_file = version_dir / "data.parquet"
        if single_file.exists():
            return pd.read_parquet(single_file)
        
        # Try loading from partitioned data
        data_dir = version_dir / "data"
        if data_dir.exists():
            import glob
            parquet_files = sorted(glob.glob(str(data_dir / "**" / "data.parquet")))
            if parquet_files:
                dfs = [pd.read_parquet(f) for f in parquet_files]
                return pd.concat(dfs, ignore_index=True)
        
        raise FeatureWriterError(
            f"No feature data found in {version_dir}"
        )


# ============================================================================
# Version Increment Helper
# ============================================================================

class VersionIncrementer:
    """Helper for incrementing semantic versions."""
    
    @staticmethod
    def increment_patch(version_str: str) -> str:
        """Increment patch version (e.g., 1.0.0 -> 1.0.1)."""
        v = SemanticVersion.parse(version_str)
        return str(SemanticVersion(v.major, v.minor, v.patch + 1))
    
    @staticmethod
    def increment_minor(version_str: str) -> str:
        """Increment minor version (e.g., 1.0.0 -> 1.1.0)."""
        v = SemanticVersion.parse(version_str)
        return str(SemanticVersion(v.major, v.minor + 1, 0))
    
    @staticmethod
    def increment_major(version_str: str) -> str:
        """Increment major version (e.g., 1.0.0 -> 2.0.0)."""
        v = SemanticVersion.parse(version_str)
        return str(SemanticVersion(v.major + 1, 0, 0))
    
    @staticmethod
    def get_next_version(
        current_version: str,
        change_type: str = "patch",
    ) -> str:
        """
        Get the next version based on change type.
        
        Args:
            current_version: Current semantic version
            change_type: "patch", "minor", or "major"
        
        Returns:
            Next semantic version
        """
        if change_type == "patch":
            return VersionIncrementer.increment_patch(current_version)
        elif change_type == "minor":
            return VersionIncrementer.increment_minor(current_version)
        elif change_type == "major":
            return VersionIncrementer.increment_major(current_version)
        else:
            raise ValueError(f"Unknown change type: {change_type}")
