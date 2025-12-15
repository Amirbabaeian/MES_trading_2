"""
Tests for feature framework base classes.

Tests:
- Feature base class functionality
- FeatureSet container operations
- Parameter validation
- Feature metadata
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.features.core.base import Feature, FeatureSet, FeatureComputer
from src.features.core.errors import (
    FeatureError,
    FeatureNotFoundError,
    ParameterValidationError,
    ComputationError,
)
from src.features.examples.basic import (
    SimpleReturn,
    LogReturn,
    RollingMean,
    RollingVolatility,
    PriceRange,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)  # Deterministic for testing
    
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
def simple_feature():
    """Create a simple test feature."""
    class TestFeature(Feature):
        def __init__(self):
            super().__init__(
                name="test_feature",
                description="Test feature",
                parameters={},
                dependencies=[],
            )
        
        def compute(self, data, **kwargs):
            return pd.Series(np.ones(len(data)), index=data.index)
    
    return TestFeature()


# ============================================================================
# Tests: Feature Base Class
# ============================================================================

class TestFeatureBase:
    """Tests for Feature base class."""
    
    def test_feature_initialization(self):
        """Test feature initialization."""
        feature = SimpleReturn()
        assert feature.name == "simple_return"
        assert isinstance(feature.description, str)
        assert isinstance(feature.parameters, dict)
        assert isinstance(feature.dependencies, list)
    
    def test_feature_with_parameters(self):
        """Test feature with parameters."""
        feature = RollingMean(window=50)
        assert feature.parameters["window"] == 50
        assert "50" in feature.name
    
    def test_feature_with_dependencies(self):
        """Test feature with dependencies."""
        from src.features.examples.basic import VolatilityOfReturns
        
        feature = VolatilityOfReturns(window=20)
        assert "simple_return" in feature.dependencies
    
    def test_feature_repr(self):
        """Test feature string representation."""
        feature = RollingMean(window=30)
        repr_str = repr(feature)
        assert "rolling_mean" in repr_str
        assert "window=30" in repr_str
    
    def test_invalid_parameter_validation(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ParameterValidationError):
            RollingMean(window=-1)  # Negative window
        
        with pytest.raises(ParameterValidationError):
            RollingMean(window=0)  # Zero window
        
        with pytest.raises(ParameterValidationError):
            RollingMean(window="not_an_int")  # Wrong type


# ============================================================================
# Tests: FeatureSet
# ============================================================================

class TestFeatureSet:
    """Tests for FeatureSet container."""
    
    def test_feature_set_creation(self):
        """Test creating a feature set."""
        feature_set = FeatureSet(
            name="test_set",
            description="Test feature set",
            version="1.0.0",
        )
        assert feature_set.name == "test_set"
        assert feature_set.version == "1.0.0"
        assert len(feature_set) == 0
    
    def test_add_feature_to_set(self):
        """Test adding features to a set."""
        feature_set = FeatureSet("test_set", "Test")
        feature = SimpleReturn()
        
        feature_set.add_feature(feature)
        assert len(feature_set) == 1
        assert "simple_return" in feature_set.list_features()
    
    def test_add_duplicate_feature(self):
        """Test that adding duplicate feature raises error."""
        feature_set = FeatureSet("test_set", "Test")
        feature = SimpleReturn()
        
        feature_set.add_feature(feature)
        with pytest.raises(FeatureError):
            feature_set.add_feature(feature)
    
    def test_get_feature(self):
        """Test retrieving feature from set."""
        feature_set = FeatureSet("test_set", "Test")
        feature = SimpleReturn()
        
        feature_set.add_feature(feature)
        retrieved = feature_set.get_feature("simple_return")
        assert retrieved.name == "simple_return"
    
    def test_get_missing_feature(self):
        """Test retrieving non-existent feature raises error."""
        feature_set = FeatureSet("test_set", "Test")
        
        with pytest.raises(FeatureNotFoundError):
            feature_set.get_feature("nonexistent")
    
    def test_remove_feature(self):
        """Test removing feature from set."""
        feature_set = FeatureSet("test_set", "Test")
        feature = SimpleReturn()
        
        feature_set.add_feature(feature)
        assert len(feature_set) == 1
        
        feature_set.remove_feature("simple_return")
        assert len(feature_set) == 0
    
    def test_remove_missing_feature(self):
        """Test removing non-existent feature raises error."""
        feature_set = FeatureSet("test_set", "Test")
        
        with pytest.raises(FeatureNotFoundError):
            feature_set.remove_feature("nonexistent")
    
    def test_feature_set_with_initial_features(self):
        """Test creating feature set with initial features."""
        features = [SimpleReturn(), LogReturn()]
        feature_set = FeatureSet(
            name="test_set",
            description="Test",
            features=features,
        )
        
        assert len(feature_set) == 2
        assert "simple_return" in feature_set.list_features()
        assert "log_return" in feature_set.list_features()


# ============================================================================
# Tests: Feature Computation
# ============================================================================

class TestFeatureComputation:
    """Tests for feature computation."""
    
    def test_simple_return_computation(self, sample_ohlcv_data):
        """Test computing simple returns."""
        feature = SimpleReturn()
        result = feature.compute(sample_ohlcv_data)
        
        assert len(result) == len(sample_ohlcv_data)
        assert result.isna().iloc[0]  # First return is NaN
        assert not result.isna().iloc[1:].all()  # Rest are not all NaN
    
    def test_rolling_mean_computation(self, sample_ohlcv_data):
        """Test computing rolling mean."""
        feature = RollingMean(window=20)
        result = feature.compute(sample_ohlcv_data)
        
        assert len(result) == len(sample_ohlcv_data)
        assert not result.isna().any()  # min_periods=1 means no NaN
    
    def test_rolling_volatility_computation(self, sample_ohlcv_data):
        """Test computing rolling volatility."""
        feature = RollingVolatility(window=20)
        result = feature.compute(sample_ohlcv_data)
        
        assert len(result) == len(sample_ohlcv_data)
        assert (result >= 0).all() or result.isna().all()  # Volatility is non-negative
    
    def test_price_range_computation(self, sample_ohlcv_data):
        """Test computing price range."""
        feature = PriceRange()
        result = feature.compute(sample_ohlcv_data)
        
        assert len(result) == len(sample_ohlcv_data)
        assert (result >= 0).all()  # Price range is non-negative
    
    def test_rolling_volatility_annualization(self, sample_ohlcv_data):
        """Test volatility annualization."""
        feature_daily = RollingVolatility(window=20, annualize=False)
        feature_annual = RollingVolatility(window=20, annualize=True)
        
        result_daily = feature_daily.compute(sample_ohlcv_data)
        result_annual = feature_annual.compute(sample_ohlcv_data)
        
        # Annualized should be approximately daily * sqrt(252)
        ratio = result_annual / result_daily
        expected_ratio = np.sqrt(252)
        
        # Check a few non-NaN values
        non_nan_mask = ~result_daily.isna() & ~result_annual.isna()
        if non_nan_mask.any():
            actual_ratios = ratio[non_nan_mask]
            assert np.allclose(actual_ratios, expected_ratio, rtol=0.01)


# ============================================================================
# Tests: Determinism
# ============================================================================

class TestDeterminism:
    """Tests for deterministic computation."""
    
    def test_deterministic_returns(self, sample_ohlcv_data):
        """Test that return computation is deterministic."""
        feature = SimpleReturn()
        
        result1 = feature.compute(sample_ohlcv_data)
        result2 = feature.compute(sample_ohlcv_data)
        
        pd.testing.assert_series_equal(result1, result2)
    
    def test_deterministic_rolling_mean(self, sample_ohlcv_data):
        """Test that rolling mean computation is deterministic."""
        feature = RollingMean(window=20)
        
        result1 = feature.compute(sample_ohlcv_data)
        result2 = feature.compute(sample_ohlcv_data)
        
        pd.testing.assert_series_equal(result1, result2)
    
    def test_deterministic_volatility(self, sample_ohlcv_data):
        """Test that volatility computation is deterministic."""
        feature = RollingVolatility(window=20)
        
        result1 = feature.compute(sample_ohlcv_data)
        result2 = feature.compute(sample_ohlcv_data)
        
        pd.testing.assert_series_equal(result1, result2)


# ============================================================================
# Tests: Error Handling
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_computation_error(self, simple_feature):
        """Test that computation errors are properly raised."""
        class FailingFeature(Feature):
            def __init__(self):
                super().__init__("fail", "Fails", {}, [])
            
            def compute(self, data, **kwargs):
                raise ValueError("Test error")
        
        feature = FailingFeature()
        
        with pytest.raises(ComputationError):
            feature.compute(pd.DataFrame({'test': [1, 2, 3]}))
    
    def test_invalid_window_size(self):
        """Test validation of window sizes."""
        with pytest.raises(ParameterValidationError):
            RollingMean(window=-5)
    
    def test_invalid_volatility_parameters(self):
        """Test validation of volatility parameters."""
        with pytest.raises(ParameterValidationError):
            RollingVolatility(window=20, annualize="not_bool")
