"""
Feature computation framework base classes.

Core abstractions for defining and computing features:
- Feature: Base class for single features with metadata and dependencies
- FeatureSet: Container for grouping related features
- FeatureComputer: Orchestration engine for computing features with dependency resolution

Key design principles:
- Parameterizable features with configurable computation parameters
- Dependency tracking for features that depend on other features
- Batch computation support for efficient processing
- Incremental updates for new data periods
- Strict determinism with no randomness or time-dependent logic
- Full Parquet integration for reading raw/cleaned data and writing computed features

Example:
    >>> from src.features.core.base import Feature, FeatureComputer
    >>> 
    >>> class SimpleReturn(Feature):
    ...     def __init__(self):
    ...         super().__init__(
    ...             name="simple_return",
    ...             description="Daily simple return",
    ...             parameters={},
    ...             dependencies=[],
    ...         )
    ...     
    ...     def compute(self, data, **kwargs):
    ...         return data['close'].pct_change()
    >>> 
    >>> feature = SimpleReturn()
    >>> computer = FeatureComputer([feature])
    >>> results = computer.compute(data, asset='ES')
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

import pandas as pd
import numpy as np

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
from .dependency import DependencyGraph

logger = logging.getLogger(__name__)


# ============================================================================
# Type Definitions
# ============================================================================

FeatureData = Union[pd.Series, pd.DataFrame]
ComputeFunction = Callable[[pd.DataFrame], FeatureData]


# ============================================================================
# Feature Base Class
# ============================================================================

class Feature(ABC):
    """
    Abstract base class for features.
    
    A feature is a computed quantity derived from market data. Each feature:
    - Has a name and description
    - Accepts configurable parameters
    - Declares dependencies on other features
    - Implements deterministic computation logic
    
    Subclasses must implement:
    - compute(): The actual computation logic
    - validate_parameters(): Parameter validation logic (optional)
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
    ):
        """
        Initialize a feature.
        
        Args:
            name: Unique feature identifier
            description: Human-readable description
            parameters: Dict of parameter names to values (empty if no params)
            dependencies: List of feature names this feature depends on
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.dependencies = dependencies or []
        
        # Validate parameters on initialization
        self.validate_parameters()
    
    def validate_parameters(self) -> None:
        """
        Validate feature parameters.
        
        Override this method to add custom parameter validation.
        
        Raises:
            ParameterValidationError: If parameters are invalid
        """
        pass
    
    @abstractmethod
    def compute(self, data: pd.DataFrame, **kwargs) -> FeatureData:
        """
        Compute the feature.
        
        Args:
            data: Input DataFrame with OHLCV columns (timestamp, open, high, low, close, volume)
            **kwargs: Additional context (e.g., asset name, date range)
        
        Returns:
            Computed feature values as Series or DataFrame
        
        Raises:
            ComputationError: If computation fails
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of feature."""
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        deps_str = ", ".join(self.dependencies) if self.dependencies else "none"
        return (
            f"Feature(name='{self.name}', "
            f"params=[{params_str}], "
            f"deps=[{deps_str}])"
        )


# ============================================================================
# FeatureSet Container
# ============================================================================

@dataclass
class FeatureSetMetadata:
    """Metadata for a feature set."""
    name: str
    description: str
    created_at: str
    version: str
    asset: Optional[str] = None


class FeatureSet:
    """
    Container for a collection of related features.
    
    Groups features together for:
    - Bulk computation and management
    - Consistent parameter application
    - Output organization
    
    Features within a set can depend on each other, with dependencies
    automatically resolved during computation.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        features: Optional[List[Feature]] = None,
        version: str = "1.0.0",
        asset: Optional[str] = None,
    ):
        """
        Initialize a feature set.
        
        Args:
            name: Name of the feature set
            description: Description of what the set contains
            features: List of Feature objects (can be empty)
            version: Version string for the feature set
            asset: Optional asset this set is designed for (e.g., 'ES', 'VIX')
        """
        self.name = name
        self.description = description
        self.features: Dict[str, Feature] = {}
        self.version = version
        self.asset = asset
        self.metadata = FeatureSetMetadata(
            name=name,
            description=description,
            created_at=datetime.utcnow().isoformat(),
            version=version,
            asset=asset,
        )
        
        if features:
            for feature in features:
                self.add_feature(feature)
    
    def add_feature(self, feature: Feature) -> None:
        """
        Add a feature to the set.
        
        Args:
            feature: Feature object to add
        
        Raises:
            FeatureError: If a feature with same name already exists
        """
        if feature.name in self.features:
            raise FeatureError(f"Feature '{feature.name}' already in set")
        self.features[feature.name] = feature
    
    def remove_feature(self, feature_name: str) -> None:
        """
        Remove a feature from the set.
        
        Args:
            feature_name: Name of feature to remove
        
        Raises:
            FeatureNotFoundError: If feature doesn't exist
        """
        if feature_name not in self.features:
            raise FeatureNotFoundError(feature_name)
        del self.features[feature_name]
    
    def get_feature(self, feature_name: str) -> Feature:
        """
        Get a feature by name.
        
        Args:
            feature_name: Name of feature
        
        Returns:
            Feature object
        
        Raises:
            FeatureNotFoundError: If feature doesn't exist
        """
        if feature_name not in self.features:
            raise FeatureNotFoundError(feature_name)
        return self.features[feature_name]
    
    def list_features(self) -> List[str]:
        """Get list of all feature names in the set."""
        return list(self.features.keys())
    
    def __len__(self) -> int:
        """Return number of features in the set."""
        return len(self.features)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FeatureSet(name='{self.name}', "
            f"features={len(self.features)}, "
            f"version='{self.version}')"
        )


# ============================================================================
# Feature Computer Engine
# ============================================================================

@dataclass
class ComputationResult:
    """Result of feature computation."""
    success: bool
    feature_name: str
    output: Optional[Union[pd.Series, pd.DataFrame]] = None
    error: Optional[str] = None
    computation_time: float = 0.0
    rows_computed: int = 0


@dataclass
class BatchComputationResult:
    """Result of batch feature computation."""
    success: bool
    features_computed: int = 0
    features_failed: int = 0
    results: List[ComputationResult] = field(default_factory=list)
    total_time: float = 0.0
    error_message: Optional[str] = None


class FeatureComputer:
    """
    Orchestration engine for computing features.
    
    Handles:
    - Dependency resolution and topological sorting
    - Sequential feature computation respecting dependencies
    - Batch computation across multiple date ranges
    - Incremental updates for new data periods
    - Data caching for efficient access
    - Parquet-based data I/O
    
    Computation flow:
    1. Add features or feature set
    2. Resolve dependencies (topological sort)
    3. Compute features in order
    4. Return results as DataFrame
    """
    
    def __init__(
        self,
        features: Optional[List[Feature]] = None,
        cache_intermediate: bool = True,
        strict_dependencies: bool = True,
    ):
        """
        Initialize the feature computer.
        
        Args:
            features: List of Feature objects to manage
            cache_intermediate: Whether to cache computed features for reuse
            strict_dependencies: Whether to fail on missing/circular dependencies
        """
        self.features: Dict[str, Feature] = {}
        self.cache_intermediate = cache_intermediate
        self.strict_dependencies = strict_dependencies
        self._computation_cache: Dict[str, Union[pd.Series, pd.DataFrame]] = {}
        self._dependency_graph = DependencyGraph()
        
        if features:
            for feature in features:
                self.add_feature(feature)
    
    def add_feature(self, feature: Feature) -> None:
        """
        Add a feature to the computer.
        
        Args:
            feature: Feature object to add
        
        Raises:
            FeatureError: If feature with same name already exists
            DependencyError: If dependencies are invalid (strict mode)
        """
        if feature.name in self.features:
            raise FeatureError(f"Feature '{feature.name}' already registered")
        
        self.features[feature.name] = feature
        self._dependency_graph.add_feature(feature.name, feature.dependencies)
        
        # Validate if strict mode is enabled
        if self.strict_dependencies:
            is_valid, errors = self._dependency_graph.validate()
            if not is_valid:
                cycles = self._dependency_graph.detect_cycles()
                if cycles:
                    raise CircularDependencyError(cycles[0])
                raise DependencyError(f"Invalid dependencies: {errors[0]}")
    
    def add_features(self, features: List[Feature]) -> None:
        """
        Add multiple features at once.
        
        Args:
            features: List of Feature objects
        """
        for feature in features:
            self.add_feature(feature)
    
    def add_feature_set(self, feature_set: FeatureSet) -> None:
        """
        Add all features from a FeatureSet.
        
        Args:
            feature_set: FeatureSet to add
        """
        for feature_name in feature_set.list_features():
            self.add_feature(feature_set.get_feature(feature_name))
    
    def get_computation_order(self) -> List[str]:
        """
        Get the order in which features should be computed.
        
        Returns:
            List of feature names in dependency order
        
        Raises:
            CircularDependencyError: If circular dependency is detected
            DependencyError: If dependencies are invalid
        """
        return self._dependency_graph.topological_sort()
    
    def clear_cache(self) -> None:
        """Clear the computation cache."""
        self._computation_cache.clear()
    
    def compute(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        asset: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Compute features for the given data.
        
        Args:
            data: Input DataFrame with OHLCV columns (timestamp, open, high, low, close, volume)
            features: Specific features to compute (all if None)
            asset: Asset identifier (e.g., 'ES', 'VIX')
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
        
        Returns:
            DataFrame with computed features as columns
        
        Raises:
            ComputationError: If any feature computation fails
            FeatureNotFoundError: If requested feature doesn't exist
        """
        # Filter data by date range if provided
        if start_date or end_date:
            data = self._filter_data_by_date(data, start_date, end_date)
        
        # Determine which features to compute
        if features is None:
            features_to_compute = list(self.features.keys())
        else:
            # Validate requested features exist
            for fname in features:
                if fname not in self.features:
                    raise FeatureNotFoundError(fname)
            features_to_compute = features
        
        # Get computation order
        try:
            computation_order = self._dependency_graph.topological_sort()
        except (CircularDependencyError, DependencyError) as e:
            raise ComputationError("unknown", f"Dependency resolution failed: {e}")
        
        # Filter to only features we need (and their dependencies)
        features_to_compute_set = set(features_to_compute)
        all_needed_features = set(features_to_compute)
        
        for fname in features_to_compute:
            deps = self._dependency_graph.get_all_dependencies(fname)
            all_needed_features.update(deps)
        
        # Filter computation order to only needed features
        ordered_features = [f for f in computation_order if f in all_needed_features]
        
        # Prepare output dataframe starting with timestamp
        result = data[['timestamp']].copy()
        
        # Clear cache for fresh computation
        self._computation_cache.clear()
        
        # Compute features in order
        for feature_name in ordered_features:
            feature = self.features[feature_name]
            
            try:
                logger.info(f"Computing feature: {feature_name}")
                
                # Prepare input data with already-computed features
                compute_input = data.copy()
                for cached_name, cached_data in self._computation_cache.items():
                    if isinstance(cached_data, pd.Series):
                        compute_input[cached_name] = cached_data.values
                    else:
                        compute_input = pd.concat([compute_input, cached_data], axis=1)
                
                # Compute the feature
                output = feature.compute(compute_input, asset=asset)
                
                # Ensure output is Series or DataFrame
                if isinstance(output, (pd.Series, pd.DataFrame)):
                    # Cache for later features that might depend on this
                    self._computation_cache[feature_name] = output
                    
                    # Add to result
                    if isinstance(output, pd.Series):
                        result[feature_name] = output.values
                    else:
                        result = pd.concat([result, output], axis=1)
                else:
                    raise ComputationError(
                        feature_name,
                        f"compute() returned {type(output)}, expected Series or DataFrame"
                    )
            
            except ComputationError:
                raise
            except Exception as e:
                raise ComputationError(feature_name, str(e))
        
        # Only keep requested features (plus timestamp)
        columns_to_keep = ['timestamp'] + [f for f in features_to_compute if f in result.columns]
        result = result[columns_to_keep]
        
        if not self.cache_intermediate:
            self._computation_cache.clear()
        
        return result
    
    def compute_batch(
        self,
        data: pd.DataFrame,
        date_ranges: List[Tuple[datetime, datetime]],
        features: Optional[List[str]] = None,
        asset: Optional[str] = None,
    ) -> BatchComputationResult:
        """
        Compute features across multiple date ranges (batch mode).
        
        Args:
            data: Input DataFrame with all data
            date_ranges: List of (start_date, end_date) tuples
            features: Specific features to compute (all if None)
            asset: Asset identifier
        
        Returns:
            BatchComputationResult with results for each date range
        """
        import time
        
        start_time = time.time()
        results = []
        failed_count = 0
        
        for start_date, end_date in date_ranges:
            try:
                batch_data = self.compute(
                    data,
                    features=features,
                    asset=asset,
                    start_date=start_date,
                    end_date=end_date,
                )
                
                # Create result
                result = ComputationResult(
                    success=True,
                    feature_name="batch",
                    output=batch_data,
                    rows_computed=len(batch_data),
                )
                results.append(result)
            
            except Exception as e:
                failed_count += 1
                result = ComputationResult(
                    success=False,
                    feature_name="batch",
                    error=str(e),
                )
                results.append(result)
        
        total_time = time.time() - start_time
        
        return BatchComputationResult(
            success=failed_count == 0,
            features_computed=len(results) - failed_count,
            features_failed=failed_count,
            results=results,
            total_time=total_time,
        )
    
    def compute_incremental(
        self,
        data: pd.DataFrame,
        existing_features: pd.DataFrame,
        features: Optional[List[str]] = None,
        asset: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Incrementally update features with new data.
        
        Args:
            data: New input data (typically recent data)
            existing_features: Previously computed features
            features: Specific features to update (all if None)
            asset: Asset identifier
        
        Returns:
            Combined DataFrame with old + new features
        
        Raises:
            IncrementalUpdateError: If incremental update fails
        """
        try:
            # Compute features for new data
            new_features = self.compute(
                data,
                features=features,
                asset=asset,
            )
            
            # Combine with existing features
            # Remove overlapping data and concatenate
            if not existing_features.empty:
                # Ensure timestamp column exists and is comparable
                if 'timestamp' not in existing_features.columns:
                    raise IncrementalUpdateError("Existing features missing 'timestamp' column")
                
                # Find cutoff point - last timestamp in existing that's before new data
                last_existing_time = existing_features['timestamp'].max()
                new_data_mask = new_features['timestamp'] > last_existing_time
                
                if new_data_mask.any():
                    new_features_to_add = new_features[new_data_mask]
                    combined = pd.concat(
                        [existing_features, new_features_to_add],
                        ignore_index=True,
                    )
                    return combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            return new_features
        
        except Exception as e:
            raise IncrementalUpdateError(f"Failed to update features: {e}")
    
    @staticmethod
    def _filter_data_by_date(
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Filter data by date range.
        
        Args:
            data: DataFrame with 'timestamp' column
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)
        
        Returns:
            Filtered DataFrame
        """
        if 'timestamp' not in data.columns:
            return data
        
        result = data.copy()
        
        if start_date is not None:
            result = result[result['timestamp'] >= start_date]
        
        if end_date is not None:
            result = result[result['timestamp'] <= end_date]
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FeatureComputer(features={len(self.features)}, "
            f"cache={self.cache_intermediate})"
        )
