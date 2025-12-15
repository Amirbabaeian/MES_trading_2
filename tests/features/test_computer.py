"""
Tests for FeatureComputer orchestration engine.

Tests:
- Adding features to computer
- Dependency resolution and topological sorting
- Feature computation with dependencies
- Batch computation
- Incremental updates
- Circular dependency detection
- Cache management
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.features.core.base import Feature, FeatureComputer
from src.features.core.errors import (
    FeatureError,
    FeatureNotFoundError,
    CircularDependencyError,
    DependencyError,
    ComputationError,
    IncrementalUpdateError,
)
from src.features.examples.basic import (
    SimpleReturn,
    LogReturn,
    RollingMean,
    RollingVolatility,
    PriceRange,
    VolatilityOfReturns,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    closes = 5000 + np.cumsum(np.random.randn(100) * 10)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': closes - np.abs(np.random.randn(100) * 5),
        'high': closes + np.abs(np.random.randn(100) * 5),
        'low': closes - np.abs(np.random.randn(100) * 5),
        'close': closes,
        'volume': np.random.randint(1000, 10000, 100),
    })
    
    return df


@pytest.fixture
def feature_computer():
    """Create a feature computer with sample features."""
    computer = FeatureComputer()
    computer.add_feature(SimpleReturn())
    computer.add_feature(RollingMean(window=20))
    computer.add_feature(PriceRange())
    return computer


# ============================================================================
# Tests: Feature Registration
# ============================================================================

class TestFeatureRegistration:
    """Tests for adding features to computer."""
    
    def test_add_single_feature(self):
        """Test adding a single feature."""
        computer = FeatureComputer()
        feature = SimpleReturn()
        
        computer.add_feature(feature)
        assert "simple_return" in computer.features
    
    def test_add_multiple_features(self):
        """Test adding multiple features."""
        computer = FeatureComputer()
        features = [SimpleReturn(), LogReturn(), RollingMean()]
        
        computer.add_features(features)
        assert len(computer.features) == 3
    
    def test_add_duplicate_feature(self):
        """Test that adding duplicate feature raises error."""
        computer = FeatureComputer()
        feature = SimpleReturn()
        
        computer.add_feature(feature)
        with pytest.raises(FeatureError):
            computer.add_feature(feature)
    
    def test_get_computation_order_no_dependencies(self):
        """Test computation order with no dependencies."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        computer.add_feature(LogReturn())
        computer.add_feature(RollingMean())
        
        order = computer.get_computation_order()
        assert len(order) == 3
        assert set(order) == {"simple_return", "log_return", "rolling_mean_20"}


# ============================================================================
# Tests: Dependency Resolution
# ============================================================================

class TestDependencyResolution:
    """Tests for dependency resolution and topological sorting."""
    
    def test_features_with_dependencies(self):
        """Test computation order with dependencies."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        computer.add_feature(VolatilityOfReturns(window=20))
        
        order = computer.get_computation_order()
        
        # SimpleReturn must come before VolatilityOfReturns
        simple_idx = order.index("simple_return")
        volatility_idx = order.index("volatility_of_returns_20")
        assert simple_idx < volatility_idx
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        class CircularA(Feature):
            def __init__(self):
                super().__init__("circular_a", "A", {}, ["circular_b"])
            def compute(self, data, **kwargs):
                return pd.Series([])
        
        class CircularB(Feature):
            def __init__(self):
                super().__init__("circular_b", "B", {}, ["circular_a"])
            def compute(self, data, **kwargs):
                return pd.Series([])
        
        computer = FeatureComputer()
        computer.add_feature(CircularA())
        
        with pytest.raises(CircularDependencyError):
            computer.add_feature(CircularB())
    
    def test_missing_dependency_error(self):
        """Test error when dependency is missing."""
        class DepFeature(Feature):
            def __init__(self):
                super().__init__("depends", "Dep", {}, ["nonexistent"])
            def compute(self, data, **kwargs):
                return pd.Series([])
        
        computer = FeatureComputer()
        
        with pytest.raises(DependencyError):
            computer.add_feature(DepFeature())
    
    def test_complex_dependency_chain(self):
        """Test topological sort with complex dependency chain."""
        class FeatureA(Feature):
            def __init__(self):
                super().__init__("feature_a", "A", {}, [])
            def compute(self, data, **kwargs):
                return pd.Series(np.ones(len(data)))
        
        class FeatureB(Feature):
            def __init__(self):
                super().__init__("feature_b", "B", {}, ["feature_a"])
            def compute(self, data, **kwargs):
                return data['feature_a'] * 2
        
        class FeatureC(Feature):
            def __init__(self):
                super().__init__("feature_c", "C", {}, ["feature_b"])
            def compute(self, data, **kwargs):
                return data['feature_b'] * 3
        
        computer = FeatureComputer()
        computer.add_feature(FeatureA())
        computer.add_feature(FeatureB())
        computer.add_feature(FeatureC())
        
        order = computer.get_computation_order()
        assert order == ["feature_a", "feature_b", "feature_c"]


# ============================================================================
# Tests: Feature Computation
# ============================================================================

class TestFeatureComputation:
    """Tests for computing features."""
    
    def test_single_feature_computation(self, sample_ohlcv_data):
        """Test computing a single feature."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        
        result = computer.compute(sample_ohlcv_data, features=["simple_return"])
        
        assert "timestamp" in result.columns
        assert "simple_return" in result.columns
        assert len(result) == len(sample_ohlcv_data)
    
    def test_multiple_features_computation(self, sample_ohlcv_data):
        """Test computing multiple features."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        computer.add_feature(RollingMean())
        computer.add_feature(PriceRange())
        
        result = computer.compute(sample_ohlcv_data)
        
        assert "simple_return" in result.columns
        assert "rolling_mean_20" in result.columns
        assert "price_range_pct" in result.columns
        assert len(result) == len(sample_ohlcv_data)
    
    def test_feature_with_dependencies_computation(self, sample_ohlcv_data):
        """Test computing features with dependencies."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        computer.add_feature(VolatilityOfReturns(window=20))
        
        result = computer.compute(sample_ohlcv_data)
        
        # Both features should be computed
        assert "simple_return" in result.columns
        assert "volatility_of_returns_20" in result.columns
    
    def test_compute_specific_features(self, sample_ohlcv_data):
        """Test computing only specific features."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        computer.add_feature(RollingMean())
        computer.add_feature(PriceRange())
        
        result = computer.compute(
            sample_ohlcv_data,
            features=["simple_return", "price_range_pct"]
        )
        
        assert "simple_return" in result.columns
        assert "price_range_pct" in result.columns
        assert "rolling_mean_20" not in result.columns
    
    def test_compute_nonexistent_feature(self, sample_ohlcv_data):
        """Test computing non-existent feature raises error."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        
        with pytest.raises(FeatureNotFoundError):
            computer.compute(sample_ohlcv_data, features=["nonexistent"])
    
    def test_asset_parameter_passed(self, sample_ohlcv_data):
        """Test that asset parameter is passed to compute."""
        class AssetAwareFeature(Feature):
            def __init__(self):
                super().__init__("asset_aware", "Aware", {}, [])
            
            def compute(self, data, **kwargs):
                asset = kwargs.get('asset', 'unknown')
                return pd.Series([asset] * len(data))
        
        computer = FeatureComputer()
        computer.add_feature(AssetAwareFeature())
        
        result = computer.compute(sample_ohlcv_data, asset='ES')
        assert (result['asset_aware'] == 'ES').all()


# ============================================================================
# Tests: Batch Computation
# ============================================================================

class TestBatchComputation:
    """Tests for batch feature computation."""
    
    def test_batch_computation(self, sample_ohlcv_data):
        """Test batch computation across multiple date ranges."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        
        date_ranges = [
            (sample_ohlcv_data['timestamp'].iloc[0], sample_ohlcv_data['timestamp'].iloc[24]),
            (sample_ohlcv_data['timestamp'].iloc[25], sample_ohlcv_data['timestamp'].iloc[49]),
            (sample_ohlcv_data['timestamp'].iloc[50], sample_ohlcv_data['timestamp'].iloc[-1]),
        ]
        
        result = computer.compute_batch(sample_ohlcv_data, date_ranges)
        
        assert result.success
        assert result.features_computed == 3
        assert result.features_failed == 0
    
    def test_batch_computation_with_failures(self, sample_ohlcv_data):
        """Test batch computation handles computation errors."""
        class FailFeature(Feature):
            def __init__(self):
                super().__init__("fail", "Fail", {}, [])
            
            def compute(self, data, **kwargs):
                raise ValueError("Computation failed")
        
        computer = FeatureComputer()
        computer.add_feature(FailFeature())
        
        date_ranges = [
            (sample_ohlcv_data['timestamp'].iloc[0], sample_ohlcv_data['timestamp'].iloc[24]),
            (sample_ohlcv_data['timestamp'].iloc[25], sample_ohlcv_data['timestamp'].iloc[-1]),
        ]
        
        result = computer.compute_batch(sample_ohlcv_data, date_ranges)
        
        assert not result.success
        assert result.features_failed == 2


# ============================================================================
# Tests: Incremental Updates
# ============================================================================

class TestIncrementalUpdates:
    """Tests for incremental feature updates."""
    
    def test_incremental_update_new_data(self, sample_ohlcv_data):
        """Test incremental update with new data."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        
        # Compute features for first 50 rows
        existing_data = sample_ohlcv_data.iloc[:50]
        existing_features = computer.compute(existing_data)
        
        # New data (last 50 rows)
        new_data = sample_ohlcv_data.iloc[50:]
        
        # Incremental update
        updated_features = computer.compute_incremental(
            new_data,
            existing_features,
        )
        
        assert len(updated_features) >= 50
        assert "simple_return" in updated_features.columns
    
    def test_incremental_update_overlapping_data(self, sample_ohlcv_data):
        """Test incremental update removes duplicates."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        
        # Existing data
        existing_data = sample_ohlcv_data.iloc[:60]
        existing_features = computer.compute(existing_data)
        
        # New data with overlap (last 40 existing + 40 new)
        new_data = sample_ohlcv_data.iloc[60:]
        
        updated_features = computer.compute_incremental(
            new_data,
            existing_features,
        )
        
        # Should have 100 total rows (no duplicates)
        assert len(updated_features) == 100
    
    def test_incremental_update_missing_timestamp(self):
        """Test incremental update requires timestamp column."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        
        # Existing features without timestamp
        existing_features = pd.DataFrame({
            'simple_return': [0.01, 0.02, 0.03],
        })
        
        new_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3),
            'close': [100, 101, 102],
        })
        
        with pytest.raises(IncrementalUpdateError):
            computer.compute_incremental(new_data, existing_features)


# ============================================================================
# Tests: Caching
# ============================================================================

class TestCaching:
    """Tests for computation caching."""
    
    def test_cache_intermediate_enabled(self, sample_ohlcv_data):
        """Test caching of intermediate results."""
        computer = FeatureComputer(cache_intermediate=True)
        computer.add_feature(SimpleReturn())
        computer.add_feature(VolatilityOfReturns(window=20))
        
        result = computer.compute(sample_ohlcv_data)
        
        # Both features should be present
        assert "simple_return" in result.columns
        assert "volatility_of_returns_20" in result.columns
        
        # Cache should be populated
        assert len(computer._computation_cache) > 0
    
    def test_cache_intermediate_disabled(self, sample_ohlcv_data):
        """Test that cache is cleared when disabled."""
        computer = FeatureComputer(cache_intermediate=False)
        computer.add_feature(SimpleReturn())
        computer.add_feature(VolatilityOfReturns(window=20))
        
        result = computer.compute(sample_ohlcv_data)
        
        # Features should still be computed correctly
        assert "simple_return" in result.columns
        assert "volatility_of_returns_20" in result.columns
        
        # Cache should be empty
        assert len(computer._computation_cache) == 0
    
    def test_clear_cache(self, sample_ohlcv_data, feature_computer):
        """Test clearing cache."""
        feature_computer.compute(sample_ohlcv_data)
        assert len(feature_computer._computation_cache) > 0
        
        feature_computer.clear_cache()
        assert len(feature_computer._computation_cache) == 0


# ============================================================================
# Tests: Date Filtering
# ============================================================================

class TestDateFiltering:
    """Tests for date filtering in computation."""
    
    def test_compute_with_start_date(self, sample_ohlcv_data):
        """Test computing with start date filter."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        
        start_date = sample_ohlcv_data['timestamp'].iloc[25]
        result = computer.compute(sample_ohlcv_data, start_date=start_date)
        
        assert result['timestamp'].min() >= start_date
        assert len(result) <= len(sample_ohlcv_data)
    
    def test_compute_with_end_date(self, sample_ohlcv_data):
        """Test computing with end date filter."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        
        end_date = sample_ohlcv_data['timestamp'].iloc[50]
        result = computer.compute(sample_ohlcv_data, end_date=end_date)
        
        assert result['timestamp'].max() <= end_date
        assert len(result) <= len(sample_ohlcv_data)
    
    def test_compute_with_date_range(self, sample_ohlcv_data):
        """Test computing with both start and end date."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        
        start_date = sample_ohlcv_data['timestamp'].iloc[25]
        end_date = sample_ohlcv_data['timestamp'].iloc[50]
        
        result = computer.compute(
            sample_ohlcv_data,
            start_date=start_date,
            end_date=end_date,
        )
        
        assert result['timestamp'].min() >= start_date
        assert result['timestamp'].max() <= end_date


# ============================================================================
# Tests: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in feature computer."""
    
    def test_computation_error_handling(self, sample_ohlcv_data):
        """Test that computation errors are properly reported."""
        class ErrorFeature(Feature):
            def __init__(self):
                super().__init__("error", "Error", {}, [])
            
            def compute(self, data, **kwargs):
                raise ValueError("Test error")
        
        computer = FeatureComputer()
        computer.add_feature(ErrorFeature())
        
        with pytest.raises(ComputationError):
            computer.compute(sample_ohlcv_data)
    
    def test_missing_column_error(self):
        """Test handling of missing required columns."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        
        # Data without 'close' column
        bad_data = pd.DataFrame({'timestamp': [1, 2, 3]})
        
        with pytest.raises(ComputationError):
            computer.compute(bad_data)
