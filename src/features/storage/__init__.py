"""
Feature storage system with versioning support.

Provides:
- Versioned storage of computed features
- Metadata tracking for reproducibility
- Efficient loading by date range and asset
- Version freezing and comparison utilities
- Changelog and dependency tracking
"""

# Metadata classes and utilities
from .metadata import (
    ComputationParameter,
    FeatureDependency,
    FeatureSchema,
    FeatureComputationMetadata,
    FeatureStorageMetadata,
    FeatureMetadataPersistence,
    FeatureMetadataBuilder,
)

# Writer classes and utilities
from .writer import (
    FeatureWriter,
    VersionIncrementer,
    FeatureWriterError,
    VersionFrozenError,
    DataConsistencyError,
    VersionIncrementError,
)

# Reader classes and utilities
from .reader import (
    FeatureReader,
    FeatureCatalog,
    FeatureReaderError,
    VersionNotFoundError,
    FeatureNotFoundError,
)

# Versioning classes and utilities
from .versioning import (
    FeatureVersionManager,
    FeatureVersionComparator,
    FeatureVersionAnalyzer,
    VersioningError,
    VersionFrozenError,
    VersionComparisonError,
)

__all__ = [
    # Metadata
    "ComputationParameter",
    "FeatureDependency",
    "FeatureSchema",
    "FeatureComputationMetadata",
    "FeatureStorageMetadata",
    "FeatureMetadataPersistence",
    "FeatureMetadataBuilder",
    # Writer
    "FeatureWriter",
    "VersionIncrementer",
    "FeatureWriterError",
    "VersionFrozenError",
    "DataConsistencyError",
    "VersionIncrementError",
    # Reader
    "FeatureReader",
    "FeatureCatalog",
    "FeatureReaderError",
    "VersionNotFoundError",
    "FeatureNotFoundError",
    # Versioning
    "FeatureVersionManager",
    "FeatureVersionComparator",
    "FeatureVersionAnalyzer",
    "VersioningError",
    "VersionComparisonError",
]
