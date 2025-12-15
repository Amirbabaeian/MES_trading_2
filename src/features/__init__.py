"""
Feature computation framework.

A deterministic, extensible framework for computing trading features.

Core Components:
- Feature: Base class for feature definition
- FeatureSet: Container for grouping features
- FeatureComputer: Orchestration engine for computation

Quick Start:
    >>> from src.features.core import Feature, FeatureComputer
    >>> from src.features.examples import SimpleReturn, RollingVolatility
    >>> 
    >>> # Create computer
    >>> computer = FeatureComputer()
    >>> computer.add_features([
    ...     SimpleReturn(),
    ...     RollingVolatility(window=20),
    ... ])
    >>> 
    >>> # Compute features
    >>> result = computer.compute(data)

For detailed documentation, see docs/feature_framework.md
"""

from .core import (
    Feature,
    FeatureSet,
    FeatureComputer,
    DependencyGraph,
)

from .storage import (
    FeatureWriter,
    FeatureReader,
    FeatureCatalog,
    FeatureVersionManager,
    FeatureVersionComparator,
    FeatureVersionAnalyzer,
    FeatureStorageMetadata,
    FeatureMetadataBuilder,
)

__all__ = [
    # Core computation
    "Feature",
    "FeatureSet",
    "FeatureComputer",
    "DependencyGraph",
    # Storage
    "FeatureWriter",
    "FeatureReader",
    "FeatureCatalog",
    "FeatureVersionManager",
    "FeatureVersionComparator",
    "FeatureVersionAnalyzer",
    "FeatureStorageMetadata",
    "FeatureMetadataBuilder",
]
