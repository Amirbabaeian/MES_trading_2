"""
Integration tests for the feature framework.

Tests end-to-end scenarios and framework usage patterns.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.features.core import (
    Feature,
    FeatureSet,
    FeatureComputer,
)
from src.features.examples import (
    SimpleReturn,
    RollingMean,
    RollingVolatility,
    PriceRange,
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
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


class TestEndToEndWorkflow:
    """Test complete feature computation workflows."""
    
    def test_basic_feature_computation(self, sample_data):
        """Test basic workflow: create computer, add features, compute."""
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        computer.add_feature(RollingMean(window=20))
        
        result = computer.compute(sample_data)
        
        assert "timestamp" in result.columns
        assert "simple_return" in result.columns
        assert "rolling_mean_20" in result.columns
        assert len(result) == len(sample_data)
    
    def test_feature_set_workflow(self, sample_data):
        """Test workflow using feature sets."""
        feature_set = FeatureSet(
            name="test_set",
            description="Test features",
            features=[
                SimpleReturn(),
                RollingMean(window=20),
                RollingVolatility(window=20),
            ],
            version="1.0.0",
        )
        
        computer = FeatureComputer()
        computer.add_feature_set(feature_set)
        
        result = computer.compute(sample_data)
        
        assert len(result.columns) == 4  # timestamp + 3 features
    
    def test_multiple_feature_sets(self, sample_data):
        """Test combining multiple feature sets."""
        set1 = FeatureSet(
            name="returns",
            description="Return features",
            features=[SimpleReturn()],
        )
        
        set2 = FeatureSet(
            name="volatility",
            description="Volatility features",
            features=[RollingVolatility(window=20)],
        )
        
        computer = FeatureComputer()
        computer.add_feature_set(set1)
        computer.add_feature_set(set2)
        
        result = computer.compute(sample_data)
        assert "simple_return" in result.columns
        assert "rolling_volatility_20" in result.columns
    
    def test_selective_computation(self, sample_data):
        """Test computing only specific features."""
        computer = FeatureComputer()
        computer.add_features([
            SimpleReturn(),
            RollingMean(window=20),
            RollingVolatility(window=20),
            PriceRange(),
        ])
        
        # Compute only some features
        result = computer.compute(
            sample_data,
            features=["simple_return", "price_range_pct"],
        )
        
        assert "simple_return" in result.columns
        assert "price_range_pct" in result.columns
        assert "rolling_mean_20" not in result.columns
        assert "rolling_volatility_20" not in result.columns
    
    def test_asset_specific_features(self, sample_data):
        """Test computing with asset identifier."""
        computer = FeatureComputer()
        computer.add_features([
            SimpleReturn(),
            RollingMean(window=20),
        ])
        
        result_es = computer.compute(sample_data, asset='ES')
        result_vix = computer.compute(sample_data, asset='VIX')
        
        # Results should be identical (features are asset-agnostic)
        pd.testing.assert_frame_equal(result_es, result_vix)
    
    def test_batch_processing_workflow(self, sample_data):
        """Test batch computation workflow."""
        computer = FeatureComputer()
        computer.add_features([
            SimpleReturn(),
            RollingVolatility(window=20),
        ])
        
        # Define date ranges
        date_ranges = [
            (sample_data['timestamp'].iloc[0], sample_data['timestamp'].iloc[24]),
            (sample_data['timestamp'].iloc[25], sample_data['timestamp'].iloc[49]),
            (sample_data['timestamp'].iloc[50], sample_data['timestamp'].iloc[-1]),
        ]
        
        result = computer.compute_batch(sample_data, date_ranges)
        
        assert result.success
        assert result.features_computed == 3
        assert result.features_failed == 0
    
    def test_incremental_update_workflow(self, sample_data):
        """Test incremental update workflow."""
        computer = FeatureComputer()
        computer.add_features([
            SimpleReturn(),
            RollingMean(window=20),
        ])
        
        # Initial computation
        initial_data = sample_data.iloc[:60]
        features_v1 = computer.compute(initial_data)
        
        # New data arrives
        new_data = sample_data.iloc[60:]
        
        # Incremental update
        features_v2 = computer.compute_incremental(new_data, features_v1)
        
        # Should have all 100 rows
        assert len(features_v2) == 100
        assert "simple_return" in features_v2.columns
        assert "rolling_mean_20" in features_v2.columns
    
    def test_date_filtering_workflow(self, sample_data):
        """Test date filtering during computation."""
        computer = FeatureComputer()
        computer.add_features([SimpleReturn(), RollingMean(window=20)])
        
        # Compute for specific date range
        start_date = sample_data['timestamp'].iloc[20]
        end_date = sample_data['timestamp'].iloc[80]
        
        result = computer.compute(
            sample_data,
            start_date=start_date,
            end_date=end_date,
        )
        
        assert result['timestamp'].min() >= start_date
        assert result['timestamp'].max() <= end_date
        assert len(result) <= 61  # 80 - 20 + 1
    
    def test_parameterized_feature_set(self, sample_data):
        """Test feature set with parameterized features."""
        features = [
            RollingVolatility(window=20, annualize=False),
            RollingVolatility(window=20, annualize=True),
            RollingVolatility(window=60, annualize=True),
        ]
        
        computer = FeatureComputer()
        for feature in features:
            computer.add_feature(feature)
        
        result = computer.compute(sample_data)
        
        assert "rolling_volatility_20" in result.columns
        assert "rolling_volatility_20_annualized" in result.columns
        assert "rolling_volatility_60_annualized" in result.columns
    
    def test_framework_composition(self, sample_data):
        """Test complex composition of features."""
        from src.features.examples import VolatilityOfReturns
        
        computer = FeatureComputer()
        computer.add_features([
            SimpleReturn(),  # Base feature
            RollingMean(window=20),  # Independent feature
            VolatilityOfReturns(window=20),  # Depends on SimpleReturn
        ])
        
        # Get computation order
        order = computer.get_computation_order()
        
        # simple_return must come before volatility_of_returns
        simple_idx = order.index("simple_return")
        volatility_idx = order.index("volatility_of_returns_20")
        assert simple_idx < volatility_idx
        
        # Compute should work correctly
        result = computer.compute(sample_data)
        
        assert "simple_return" in result.columns
        assert "rolling_mean_20" in result.columns
        assert "volatility_of_returns_20" in result.columns
    
    def test_cache_management(self, sample_data):
        """Test cache management in computation."""
        from src.features.examples import VolatilityOfReturns
        
        # With caching
        computer_cached = FeatureComputer(cache_intermediate=True)
        computer_cached.add_features([
            SimpleReturn(),
            VolatilityOfReturns(window=20),
        ])
        
        result_cached = computer_cached.compute(sample_data)
        
        # Cache should have intermediate results
        assert len(computer_cached._computation_cache) > 0
        
        # Without caching
        computer_no_cache = FeatureComputer(cache_intermediate=False)
        computer_no_cache.add_features([
            SimpleReturn(),
            VolatilityOfReturns(window=20),
        ])
        
        result_no_cache = computer_no_cache.compute(sample_data)
        
        # Cache should be empty
        assert len(computer_no_cache._computation_cache) == 0
        
        # Results should be identical
        pd.testing.assert_frame_equal(result_cached, result_no_cache)


class TestErrorRecovery:
    """Test error handling in workflows."""
    
    def test_recovery_from_invalid_feature(self, sample_data):
        """Test that invalid feature is caught early."""
        from src.features.core.errors import ParameterValidationError
        
        with pytest.raises(ParameterValidationError):
            RollingMean(window=-5)
    
    def test_recovery_from_missing_data(self):
        """Test handling of missing required columns."""
        from src.features.core.errors import ComputationError
        
        computer = FeatureComputer()
        computer.add_feature(SimpleReturn())
        
        # Data without close column
        bad_data = pd.DataFrame({'timestamp': [1, 2, 3]})
        
        with pytest.raises(ComputationError):
            computer.compute(bad_data)
    
    def test_recovery_from_circular_dependency(self):
        """Test detection of circular dependencies."""
        from src.features.core.errors import CircularDependencyError
        from src.features.core.base import Feature
        
        class CircA(Feature):
            def __init__(self):
                super().__init__("circ_a", "A", {}, ["circ_b"])
            
            def compute(self, data, **kwargs):
                return pd.Series([])
        
        class CircB(Feature):
            def __init__(self):
                super().__init__("circ_b", "B", {}, ["circ_a"])
            
            def compute(self, data, **kwargs):
                return pd.Series([])
        
        computer = FeatureComputer()
        computer.add_feature(CircA())
        
        with pytest.raises(CircularDependencyError):
            computer.add_feature(CircB())
