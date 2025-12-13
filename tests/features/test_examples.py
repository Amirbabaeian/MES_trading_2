"""
Tests for example feature implementations.

Tests:
- All example feature computations
- Parameter validation for parameterized features
- Correctness of mathematical formulas
- Edge cases and boundary conditions
"""

import pytest
import pandas as pd
import numpy as np

from src.features.examples.basic import (
    SimpleReturn,
    LogReturn,
    CumulativeReturn,
    RollingMean,
    RollingVolatility,
    PriceRange,
    HighLowRatio,
    CloseToOpen,
    RelativeVolume,
    VolatilityOfReturns,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_data():
    """Create simple deterministic OHLCV data."""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10),
        'open': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        'high': [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
        'low': [98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
    })
    return df


@pytest.fixture
def random_data():
    """Create random OHLCV data with seed for reproducibility."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100)
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


# ============================================================================
# Tests: Return Features
# ============================================================================

class TestReturnFeatures:
    """Tests for return calculation features."""
    
    def test_simple_return_basic(self, simple_data):
        """Test simple return calculation."""
        feature = SimpleReturn()
        result = feature.compute(simple_data)
        
        # First value should be NaN (no previous close)
        assert pd.isna(result.iloc[0])
        
        # Subsequent values should be non-NaN
        assert not result.iloc[1:].isna().any()
        
        # Check first return calculation
        # log(102/101) ≈ 0.00995
        expected_first_return = np.log(102.0 / 101.0)
        assert np.isclose(result.iloc[1], expected_first_return, rtol=1e-10)
    
    def test_log_return_basic(self, simple_data):
        """Test log return calculation."""
        feature = LogReturn()
        result = feature.compute(simple_data)
        
        assert pd.isna(result.iloc[0])
        assert not result.iloc[1:].isna().any()
        
        # Check calculation
        expected = np.log(102.0 / 101.0)
        assert np.isclose(result.iloc[1], expected, rtol=1e-10)
    
    def test_cumulative_return(self, simple_data):
        """Test cumulative return calculation."""
        feature = CumulativeReturn()
        result = feature.compute(simple_data)
        
        # All values should be non-NaN
        assert not result.isna().any()
        
        # First value should be 0
        assert np.isclose(result.iloc[0], 0.0)
        
        # Check calculation for last value
        # (110 - 101) / 101 ≈ 0.0891
        expected_last = (110.0 - 101.0) / 101.0
        assert np.isclose(result.iloc[-1], expected_last, rtol=1e-10)
    
    def test_return_consistency(self, random_data):
        """Test that SimpleReturn and LogReturn are equivalent."""
        simple_feature = SimpleReturn()
        log_feature = LogReturn()
        
        simple_result = simple_feature.compute(random_data)
        log_result = log_feature.compute(random_data)
        
        # They should be identical
        pd.testing.assert_series_equal(simple_result, log_result)


# ============================================================================
# Tests: Rolling Statistics
# ============================================================================

class TestRollingStatistics:
    """Tests for rolling window features."""
    
    def test_rolling_mean_basic(self, simple_data):
        """Test rolling mean calculation."""
        feature = RollingMean(window=2)
        result = feature.compute(simple_data)
        
        # Should have same length as input
        assert len(result) == len(simple_data)
        
        # First value should be just the first close (min_periods=1)
        assert np.isclose(result.iloc[0], 101.0)
        
        # Second value should be mean of first two closes
        expected_second = (101.0 + 102.0) / 2
        assert np.isclose(result.iloc[1], expected_second)
    
    def test_rolling_mean_larger_window(self, simple_data):
        """Test rolling mean with larger window."""
        feature = RollingMean(window=5)
        result = feature.compute(simple_data)
        
        # Check a specific value
        # Mean of closes [101, 102, 103, 104, 105] = 103
        expected = (101.0 + 102.0 + 103.0 + 104.0 + 105.0) / 5
        assert np.isclose(result.iloc[4], expected)
    
    def test_rolling_volatility_basic(self, simple_data):
        """Test rolling volatility calculation."""
        feature = RollingVolatility(window=2, annualize=False)
        result = feature.compute(simple_data)
        
        # Should have same length as input
        assert len(result) == len(simple_data)
        
        # Volatility should be non-negative
        assert (result >= 0).all() or result.isna().all()
    
    def test_rolling_volatility_annualization(self, random_data):
        """Test volatility annualization factor."""
        feature_daily = RollingVolatility(window=20, annualize=False)
        feature_annual = RollingVolatility(window=20, annualize=True)
        
        result_daily = feature_daily.compute(random_data)
        result_annual = feature_annual.compute(random_data)
        
        # Where both are non-NaN, annualized should be daily * sqrt(252)
        non_nan_mask = ~result_daily.isna() & ~result_annual.isna() & (result_daily > 0)
        if non_nan_mask.any():
            ratios = result_annual[non_nan_mask] / result_daily[non_nan_mask]
            expected = np.sqrt(252)
            assert np.allclose(ratios, expected, rtol=1e-10)


# ============================================================================
# Tests: Price-based Features
# ============================================================================

class TestPriceFeatures:
    """Tests for price-based features."""
    
    def test_price_range(self, simple_data):
        """Test price range calculation."""
        feature = PriceRange()
        result = feature.compute(simple_data)
        
        # Should have same length as input
        assert len(result) == len(simple_data)
        
        # Price range should be non-negative
        assert (result >= 0).all()
        
        # Check first value
        # (102 - 98) / 101 * 100 = 3.96...
        expected = (102.0 - 98.0) / 101.0 * 100
        assert np.isclose(result.iloc[0], expected)
    
    def test_high_low_ratio(self, simple_data):
        """Test high/low ratio calculation."""
        feature = HighLowRatio()
        result = feature.compute(simple_data)
        
        # Should have same length as input
        assert len(result) == len(simple_data)
        
        # Ratio should be >= 1 (high >= low)
        assert (result >= 1).all()
        
        # Check first value
        # 102 / 98 ≈ 1.0408
        expected = 102.0 / 98.0
        assert np.isclose(result.iloc[0], expected)
    
    def test_close_to_open(self, simple_data):
        """Test close/open ratio calculation."""
        feature = CloseToOpen()
        result = feature.compute(simple_data)
        
        # Should have same length as input
        assert len(result) == len(simple_data)
        
        # Check first value
        # 101 / 100 = 1.01
        expected = 101.0 / 100.0
        assert np.isclose(result.iloc[0], expected)


# ============================================================================
# Tests: Volume Features
# ============================================================================

class TestVolumeFeatures:
    """Tests for volume-based features."""
    
    def test_relative_volume_basic(self, simple_data):
        """Test relative volume calculation."""
        feature = RelativeVolume(window=2)
        result = feature.compute(simple_data)
        
        # Should have same length as input
        assert len(result) == len(simple_data)
        
        # Relative volume should be positive
        assert (result > 0).all()
        
        # First value should be 1.0 (volume / itself)
        assert np.isclose(result.iloc[0], 1.0)
    
    def test_relative_volume_with_constant_volume(self):
        """Test relative volume when volume is constant."""
        df = pd.DataFrame({
            'volume': [1000] * 10,
            'close': [100] * 10,
        })
        
        feature = RelativeVolume(window=5)
        result = feature.compute(df)
        
        # All values should be 1.0
        assert np.allclose(result, 1.0)


# ============================================================================
# Tests: Dependent Features
# ============================================================================

class TestDependentFeatures:
    """Tests for features with dependencies."""
    
    def test_volatility_of_returns_dependency(self, random_data):
        """Test VolatilityOfReturns requires simple_return."""
        feature = VolatilityOfReturns(window=20)
        
        # Should fail if simple_return column is missing
        with pytest.raises(Exception):  # Could be ValueError or ComputationError
            feature.compute(random_data)
    
    def test_volatility_of_returns_with_dependency(self, random_data):
        """Test VolatilityOfReturns with simple_return column."""
        # First compute simple returns
        simple_return_feature = SimpleReturn()
        returns = simple_return_feature.compute(random_data)
        
        # Add to data
        data_with_returns = random_data.copy()
        data_with_returns['simple_return'] = returns
        
        # Now compute volatility of returns
        volatility_feature = VolatilityOfReturns(window=20)
        result = volatility_feature.compute(data_with_returns)
        
        # Should have same length as input
        assert len(result) == len(random_data)
        
        # Should be non-negative
        assert (result >= 0).all() or result.isna().all()


# ============================================================================
# Tests: Parameter Validation
# ============================================================================

class TestParameterValidation:
    """Tests for parameter validation in examples."""
    
    def test_invalid_rolling_mean_window(self):
        """Test invalid window for RollingMean."""
        with pytest.raises(Exception):
            RollingMean(window=0)
        
        with pytest.raises(Exception):
            RollingMean(window=-10)
        
        with pytest.raises(Exception):
            RollingMean(window="not_int")
    
    def test_invalid_rolling_volatility_params(self):
        """Test invalid parameters for RollingVolatility."""
        with pytest.raises(Exception):
            RollingVolatility(window=-5)
        
        with pytest.raises(Exception):
            RollingVolatility(annualize="not_bool")
    
    def test_invalid_relative_volume_window(self):
        """Test invalid window for RelativeVolume."""
        with pytest.raises(Exception):
            RelativeVolume(window=-1)
        
        with pytest.raises(Exception):
            RelativeVolume(window="invalid")


# ============================================================================
# Tests: Determinism
# ============================================================================

class TestDeterminism:
    """Tests for deterministic computation."""
    
    def test_deterministic_computation_multiple_runs(self, random_data):
        """Test that all features produce identical results across runs."""
        features = [
            SimpleReturn(),
            LogReturn(),
            CumulativeReturn(),
            RollingMean(window=20),
            RollingVolatility(window=20),
            PriceRange(),
            HighLowRatio(),
            CloseToOpen(),
            RelativeVolume(window=20),
        ]
        
        # Compute each feature twice
        for feature in features:
            result1 = feature.compute(random_data)
            result2 = feature.compute(random_data)
            
            # Results should be identical
            pd.testing.assert_series_equal(result1, result2, check_names=False)
    
    def test_deterministic_with_different_data_sizes(self):
        """Test determinism with different input sizes."""
        np.random.seed(42)
        feature = RollingMean(window=10)
        
        # Compute with different data sizes
        for size in [10, 50, 100, 200]:
            df = pd.DataFrame({
                'close': np.random.randn(size).cumsum() + 100,
            })
            
            result1 = feature.compute(df)
            result2 = feature.compute(df)
            
            pd.testing.assert_series_equal(result1, result2, check_names=False)


# ============================================================================
# Tests: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_row_data(self):
        """Test features with single row of data."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1),
            'open': [100.0],
            'high': [102.0],
            'low': [98.0],
            'close': [101.0],
            'volume': [1000],
        })
        
        # Most features should handle this gracefully
        features = [
            SimpleReturn(),
            PriceRange(),
            HighLowRatio(),
        ]
        
        for feature in features:
            result = feature.compute(df)
            assert len(result) == 1
    
    def test_all_nan_data(self):
        """Test features with all NaN data."""
        df = pd.DataFrame({
            'close': [np.nan] * 10,
            'high': [np.nan] * 10,
            'low': [np.nan] * 10,
            'open': [np.nan] * 10,
            'volume': [np.nan] * 10,
        })
        
        feature = SimpleReturn()
        result = feature.compute(df)
        
        # All results should be NaN
        assert result.isna().all()
    
    def test_zero_volume(self):
        """Test features with zero volume."""
        df = pd.DataFrame({
            'volume': [0] * 10,
            'close': [100.0] * 10,
        })
        
        feature = RelativeVolume(window=5)
        result = feature.compute(df)
        
        # Should handle zero volume without errors
        assert len(result) == 10
    
    def test_constant_prices(self):
        """Test features with constant prices."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10),
            'open': [100.0] * 10,
            'high': [100.0] * 10,
            'low': [100.0] * 10,
            'close': [100.0] * 10,
            'volume': [1000] * 10,
        })
        
        # Return features should produce zeros/NaNs
        feature = SimpleReturn()
        result = feature.compute(df)
        # First is NaN, rest should be 0 or very close
        assert pd.isna(result.iloc[0])
        assert np.allclose(result.iloc[1:], 0, atol=1e-15)
        
        # Volatility should be 0
        vol_feature = RollingVolatility(window=5)
        result = vol_feature.compute(df)
        assert np.allclose(result, 0, atol=1e-15)
