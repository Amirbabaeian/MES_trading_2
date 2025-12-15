"""
Feature computation framework exceptions.

Provides custom exception classes for feature computation framework errors.
"""


class FeatureError(Exception):
    """Base exception for feature computation framework."""
    pass


class FeatureNotFoundError(FeatureError):
    """Raised when a feature is not found."""
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        super().__init__(f"Feature not found: {feature_name}")


class DependencyError(FeatureError):
    """Raised when there's an issue with feature dependencies."""
    pass


class CircularDependencyError(DependencyError):
    """Raised when circular dependency is detected."""
    def __init__(self, cycle: list):
        self.cycle = cycle
        cycle_str = " -> ".join(cycle + [cycle[0]])
        super().__init__(f"Circular dependency detected: {cycle_str}")


class MissingDependencyError(DependencyError):
    """Raised when a required dependency is missing."""
    def __init__(self, feature_name: str, missing_dep: str):
        self.feature_name = feature_name
        self.missing_dependency = missing_dep
        super().__init__(
            f"Feature '{feature_name}' depends on missing feature '{missing_dep}'"
        )


class ParameterValidationError(FeatureError):
    """Raised when feature parameters are invalid."""
    def __init__(self, feature_name: str, message: str):
        self.feature_name = feature_name
        super().__init__(f"Invalid parameter for '{feature_name}': {message}")


class ComputationError(FeatureError):
    """Raised when feature computation fails."""
    def __init__(self, feature_name: str, message: str):
        self.feature_name = feature_name
        super().__init__(f"Computation error for '{feature_name}': {message}")


class IncrementalUpdateError(FeatureError):
    """Raised when incremental update fails."""
    def __init__(self, message: str):
        super().__init__(f"Incremental update error: {message}")
