"""
Feature computation framework core module.

Exports:
- Feature: Base class for feature definition
- FeatureSet: Container for grouping features
- FeatureComputer: Orchestration engine for computation
- DependencyGraph: Dependency resolution logic

Errors:
- FeatureError: Base exception
- FeatureNotFoundError: Feature not found
- CircularDependencyError: Circular dependency detected
- DependencyError: Generic dependency error
- ParameterValidationError: Invalid parameter
- ComputationError: Feature computation failed
- IncrementalUpdateError: Incremental update failed
"""

from .base import (
    Feature,
    FeatureSet,
    FeatureComputer,
    ComputationResult,
    BatchComputationResult,
)
from .dependency import DependencyGraph
from .errors import (
    FeatureError,
    FeatureNotFoundError,
    DependencyError,
    CircularDependencyError,
    MissingDependencyError,
    ParameterValidationError,
    ComputationError,
    IncrementalUpdateError,
)

__all__ = [
    # Core classes
    "Feature",
    "FeatureSet",
    "FeatureComputer",
    "DependencyGraph",
    # Result types
    "ComputationResult",
    "BatchComputationResult",
    # Exceptions
    "FeatureError",
    "FeatureNotFoundError",
    "DependencyError",
    "CircularDependencyError",
    "MissingDependencyError",
    "ParameterValidationError",
    "ComputationError",
    "IncrementalUpdateError",
]
