"""
Comprehensive test suite for data validation.

Tests cover:
- Schema validation (columns, data types)
- Timestamp validation (ordering, timezone, format)
- OHLC relationship validation
- Violation reporting and summary statistics
- Batch validation
- Edge cases (empty data, single row, etc.)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

from src.data_io.validation import (
    DataValidator,
    Violation,
    ValidationResult,
    validate_ohlcv_data,
    validate_multiple_datasets,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def valid_ohlcv_data():
    """Create valid OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1H", tz="US/Eastern")
    np.random.seed(42)
    
    data = {
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 100),
        "high": np.random.uniform(110, 120, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(100, 110, 100),
        "volume": np.random.randint(1000, 10000, 100),
    }
    
    # Ensure OHLC relationships
    for i in range(100):
        data["high"][i] = max(data["open"][i], data["close"][i], data["high"][i])
        data["low"][i] = min(data["open"][i], data["close"][i], data["low"][i])
    
    return pd.DataFrame(data)


@pytest.fixture
def invalid_schema_data():
    """Create data with missing columns."""
    dates = pd.date_range("2024-01-01", periods=10, freq="1H", tz="US/Eastern")
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 10),
        # Missing: high, low, close, volume
    })


@pytest.fixture
def invalid_timestamp_order_data():
    """Create data with non-monotonic timestamps."""
    dates = pd.to_datetime([
        "2024-01-01 00:00:00",
        "2024-01-01 02:00:00",  # Skipped 1 hour
        "2024-01-01 01:00:00",  # Time reversal
        "2024-01-01 03:00:00",
    ], utc=True).tz_convert("US/Eastern")
    
    return pd.DataFrame({
        "timestamp": dates,
        "open": [100.0, 101.0, 102.0, 103.0],
        "high": [110.0, 111.0, 112.0, 113.0],
        "low": [90.0, 91.0, 92.0, 93.0],
        "close": [105.0, 106.0, 107.0, 108.0],
        "volume": [1000, 2000, 3000, 4000],
    })


@pytest.fixture
def invalid_ohlc_data():
    """Create data with OHLC relationship violations."""
    dates = pd.date_range("2024-01-01", periods=5, freq="1H", tz="US/Eastern")
    
    return pd.DataFrame({
        "timestamp": dates,
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [95.0, 101.0, 102.0, 103.0, 104.0],  # high < open
        "low": [90.0, 91.0, 110.0, 93.0, 94.0],      # low > close
        "close": [105.0, 106.0, 107.0, 108.0, 109.0],
        "volume": [1000, 2000, 3000, 4000, 5000],
    })


@pytest.fixture
def validator():
    """Create a DataValidator instance."""
    return DataValidator(float_tolerance=1e-9, timezone="US/Eastern")


# ============================================================================
# Tests: Basic Schema Validation
# ============================================================================

class TestSchemaValidation:
    """Tests for schema validation."""
    
    def test_valid_schema(self, valid_ohlcv_data, validator):
        """Test that valid schema passes validation."""
        result = validator.validate(valid_ohlcv_data)
        assert result.passed
        assert result.total_bars_checked == 100
    
    def test_missing_columns(self, invalid_schema_data, validator):
        """Test detection of missing required columns."""
        result = validator.validate(invalid_schema_data)
        assert not result.passed
        assert len(result.violations) > 0
        
        # Check for specific missing columns
        missing_cols = {v.column for v in result.violations if v.violation_type == "schema"}
        assert "high" in missing_cols
        assert "low" in missing_cols
        assert "close" in missing_cols
        assert "volume" in missing_cols
    
    def test_extra_columns_allowed(self, valid_ohlcv_data, validator):
        """Test that extra columns don't fail validation."""
        valid_ohlcv_data["extra_column"] = np.random.uniform(0, 1, len(valid_ohlcv_data))
        result = validator.validate(valid_ohlcv_data)
        # Should pass - extra columns are allowed
        assert result.passed
    
    def test_wrong_data_type(self, valid_ohlcv_data, validator):
        """Test detection of wrong data types."""
        # Change open to string
        valid_ohlcv_data["open"] = valid_ohlcv_data["open"].astype(str)
        result = validator.validate(valid_ohlcv_data)
        assert not result.passed
        
        # Check for type error on 'open'
        type_violations = [v for v in result.violations if v.column == "open"]
        assert len(type_violations) > 0


# ============================================================================
# Tests: Timestamp Validation
# ============================================================================

class TestTimestampValidation:
    """Tests for timestamp validation."""
    
    def test_monotonic_timestamps(self, valid_ohlcv_data, validator):
        """Test that monotonically increasing timestamps pass."""
        result = validator.validate(valid_ohlcv_data)
        timestamp_violations = [v for v in result.violations if v.violation_type == "timestamp"]
        # Should have no ordering violations
        ordering_violations = [v for v in timestamp_violations if "monotonically" in v.message]
        assert len(ordering_violations) == 0
    
    def test_non_monotonic_timestamps(self, invalid_timestamp_order_data, validator):
        """Test detection of non-monotonic timestamps."""
        result = validator.validate(invalid_timestamp_order_data)
        assert not result.passed
        
        # Should have timestamp ordering violation
        timestamp_violations = [v for v in result.violations if v.violation_type == "timestamp"]
        assert len(timestamp_violations) > 0
    
    def test_duplicate_timestamps(self, valid_ohlcv_data, validator):
        """Test detection of duplicate timestamps."""
        # Add duplicate
        valid_ohlcv_data.loc[len(valid_ohlcv_data)] = valid_ohlcv_data.iloc[0]
        valid_ohlcv_data = valid_ohlcv_data.sort_values("timestamp").reset_index(drop=True)
        
        result = validator.validate(valid_ohlcv_data)
        assert not result.passed
        
        duplicate_violations = [v for v in result.violations if "Duplicate" in v.message]
        assert len(duplicate_violations) > 0
    
    def test_allow_duplicate_timestamps(self, valid_ohlcv_data):
        """Test that duplicates are allowed when configured."""
        # Add duplicate
        valid_ohlcv_data.loc[len(valid_ohlcv_data)] = valid_ohlcv_data.iloc[0]
        valid_ohlcv_data = valid_ohlcv_data.sort_values("timestamp").reset_index(drop=True)
        
        validator = DataValidator(allow_duplicate_timestamps=True, timezone="US/Eastern")
        result = validator.validate(valid_ohlcv_data)
        # Should pass (no duplicate violations)
        duplicate_violations = [v for v in result.violations if "Duplicate" in v.message]
        assert len(duplicate_violations) == 0
    
    def test_timezone_consistency_aware(self, valid_ohlcv_data, validator):
        """Test timezone consistency for aware timestamps."""
        result = validator.validate(valid_ohlcv_data)
        # Should pass for US/Eastern timestamps
        tz_violations = [v for v in result.violations if "timezone" in v.message.lower()]
        assert len(tz_violations) == 0
    
    def test_timezone_consistency_mismatch(self, valid_ohlcv_data):
        """Test detection of timezone mismatch."""
        # Create validator expecting UTC
        validator = DataValidator(timezone="UTC")
        result = validator.validate(valid_ohlcv_data)
        
        # Should have timezone violation
        tz_violations = [v for v in result.violations if "timezone" in v.message.lower()]
        assert len(tz_violations) > 0
    
    def test_naive_timestamps(self):
        """Test validation of naive timestamps."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1H")  # No timezone
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.uniform(100, 110, 10),
            "high": np.random.uniform(110, 120, 10),
            "low": np.random.uniform(90, 100, 10),
            "close": np.random.uniform(100, 110, 10),
            "volume": np.random.randint(1000, 10000, 10),
        })
        
        validator = DataValidator(timezone="naive")
        result = validator.validate(df)
        # Should pass for naive timestamps
        tz_violations = [v for v in result.violations if "timezone" in v.message.lower()]
        assert len(tz_violations) == 0


# ============================================================================
# Tests: OHLC Validation
# ============================================================================

class TestOHLCValidation:
    """Tests for OHLC relationship validation."""
    
    def test_valid_ohlc_relationships(self, valid_ohlcv_data, validator):
        """Test that valid OHLC relationships pass."""
        result = validator.validate(valid_ohlcv_data)
        ohlc_violations = [v for v in result.violations if v.violation_type == "ohlc"]
        # Should have no OHLC violations
        assert len(ohlc_violations) == 0
    
    def test_high_less_than_open_close(self, invalid_ohlc_data, validator):
        """Test detection of high < max(open, close)."""
        result = validator.validate(invalid_ohlc_data)
        assert not result.passed
        
        # Check for high violation
        high_violations = [v for v in result.violations if v.column == "high"]
        assert len(high_violations) > 0
    
    def test_low_greater_than_open_close(self, invalid_ohlc_data, validator):
        """Test detection of low > min(open, close)."""
        result = validator.validate(invalid_ohlc_data)
        assert not result.passed
        
        # Check for low violation
        low_violations = [v for v in result.violations if v.column == "low"]
        assert len(low_violations) > 0
    
    def test_negative_prices(self, validator):
        """Test detection of negative prices."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1H", tz="US/Eastern"),
            "open": [100.0, -101.0, 102.0],
            "high": [110.0, 111.0, 112.0],
            "low": [90.0, 91.0, 92.0],
            "close": [105.0, 106.0, 107.0],
            "volume": [1000, 2000, 3000],
        })
        
        result = validator.validate(df)
        assert not result.passed
        
        negative_violations = [v for v in result.violations if "Negative" in v.message]
        assert len(negative_violations) > 0
    
    def test_zero_prices(self, validator):
        """Test detection of zero prices (except volume)."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1H", tz="US/Eastern"),
            "open": [100.0, 0.0, 102.0],  # Zero open
            "high": [110.0, 111.0, 112.0],
            "low": [90.0, 91.0, 92.0],
            "close": [105.0, 106.0, 107.0],
            "volume": [1000, 2000, 3000],
        })
        
        result = validator.validate(df)
        assert not result.passed
        
        zero_violations = [v for v in result.violations if "Zero value" in v.message]
        assert len(zero_violations) > 0
    
    def test_null_values_in_ohlc(self, validator):
        """Test detection of null values in OHLC."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1H", tz="US/Eastern"),
            "open": [100.0, np.nan, 102.0],
            "high": [110.0, 111.0, 112.0],
            "low": [90.0, 91.0, 92.0],
            "close": [105.0, 106.0, 107.0],
            "volume": [1000, 2000, 3000],
        })
        
        result = validator.validate(df)
        assert not result.passed
        
        null_violations = [v for v in result.violations if "Null value" in v.message]
        assert len(null_violations) > 0
    
    def test_zero_volume_allowed(self, validator):
        """Test that zero volume is allowed."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1H", tz="US/Eastern"),
            "open": [100.0, 101.0, 102.0],
            "high": [110.0, 111.0, 112.0],
            "low": [90.0, 91.0, 92.0],
            "close": [105.0, 106.0, 107.0],
            "volume": [0, 2000, 3000],  # Zero volume in first bar
        })
        
        result = validator.validate(df)
        # Should not have zero volume violations
        zero_volume_violations = [
            v for v in result.violations 
            if "Zero value" in v.message and v.column == "volume"
        ]
        assert len(zero_volume_violations) == 0


# ============================================================================
# Tests: Floating-Point Tolerance
# ============================================================================

class TestFloatingPointTolerance:
    """Tests for floating-point tolerance in comparisons."""
    
    def test_tolerance_in_ohlc_check(self):
        """Test that tolerance is applied in OHLC checks."""
        tolerance = 1e-6
        validator = DataValidator(float_tolerance=tolerance, timezone="US/Eastern")
        
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="1H", tz="US/Eastern"),
            "open": [100.0, 101.0],
            "high": [100.0 + tolerance/2, 101.0],  # Slightly below max(open, close)
            "low": [90.0, 91.0],
            "close": [105.0, 106.0],
            "volume": [1000, 2000],
        })
        
        result = validator.validate(df)
        # Should pass with tolerance
        high_violations = [v for v in result.violations if v.column == "high"]
        assert len(high_violations) == 0
    
    def test_violation_beyond_tolerance(self):
        """Test that violations beyond tolerance are detected."""
        tolerance = 1e-6
        validator = DataValidator(float_tolerance=tolerance, timezone="US/Eastern")
        
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=1, freq="1H", tz="US/Eastern"),
            "open": [100.0],
            "high": [99.9],  # Clearly less than max(open, close)
            "low": [90.0],
            "close": [105.0],
            "volume": [1000],
        })
        
        result = validator.validate(df)
        # Should fail
        high_violations = [v for v in result.violations if v.column == "high"]
        assert len(high_violations) > 0


# ============================================================================
# Tests: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_dataframe(self, validator):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame({
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        })
        
        result = validator.validate(df)
        assert result.passed
        assert result.total_bars_checked == 0
        assert "Empty" in result.summary.get("message", "")
    
    def test_single_row_dataframe(self, validator):
        """Test validation of single-row DataFrame."""
        dates = pd.date_range("2024-01-01", periods=1, freq="1H", tz="US/Eastern")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1000],
        })
        
        result = validator.validate(df)
        assert result.passed
        assert result.total_bars_checked == 1
    
    def test_null_timestamps(self, validator):
        """Test detection of null timestamps."""
        df = pd.DataFrame({
            "timestamp": [
                pd.Timestamp("2024-01-01", tz="US/Eastern"),
                None,
                pd.Timestamp("2024-01-01 02:00:00", tz="US/Eastern"),
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [110.0, 111.0, 112.0],
            "low": [90.0, 91.0, 92.0],
            "close": [105.0, 106.0, 107.0],
            "volume": [1000, 2000, 3000],
        })
        
        result = validator.validate(df)
        assert not result.passed
        
        null_ts_violations = [v for v in result.violations if "null" in v.message.lower()]
        assert len(null_ts_violations) > 0
    
    def test_large_dataset(self, validator):
        """Test validation performance on large dataset."""
        dates = pd.date_range("2024-01-01", periods=100000, freq="1min", tz="US/Eastern")
        np.random.seed(42)
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.uniform(100, 110, 100000),
            "high": np.random.uniform(110, 120, 100000),
            "low": np.random.uniform(90, 100, 100000),
            "close": np.random.uniform(100, 110, 100000),
            "volume": np.random.randint(1000, 10000, 100000),
        })
        
        # Ensure valid OHLC
        for i in range(len(df)):
            df.loc[i, "high"] = max(df.loc[i, "open"], df.loc[i, "close"], df.loc[i, "high"])
            df.loc[i, "low"] = min(df.loc[i, "open"], df.loc[i, "close"], df.loc[i, "low"])
        
        result = validator.validate(df)
        assert result.total_bars_checked == 100000


# ============================================================================
# Tests: Violation Reporting
# ============================================================================

class TestViolationReporting:
    """Tests for violation reporting and summary statistics."""
    
    def test_violation_details(self, invalid_ohlc_data, validator):
        """Test that violations contain actionable details."""
        result = validator.validate(invalid_ohlc_data)
        
        assert len(result.violations) > 0
        for violation in result.violations[:1]:
            assert violation.bar_index is not None
            assert violation.violation_type is not None
            assert violation.message is not None
    
    def test_violation_by_type(self, invalid_ohlc_data, validator):
        """Test violation counting by type."""
        result = validator.validate(invalid_ohlc_data)
        
        violations_by_type = result.get_violations_by_type()
        assert "ohlc" in violations_by_type
        assert violations_by_type["ohlc"] > 0
    
    def test_violation_by_column(self, invalid_ohlc_data, validator):
        """Test violation counting by column."""
        result = validator.validate(invalid_ohlc_data)
        
        violations_by_column = result.get_violations_by_column()
        assert "high" in violations_by_column or "low" in violations_by_column
    
    def test_violation_to_dict(self, invalid_ohlc_data, validator):
        """Test violation serialization to dict."""
        result = validator.validate(invalid_ohlc_data)
        
        violation_dict = result.violations[0].to_dict()
        assert "violation_type" in violation_dict
        assert "bar_index" in violation_dict
        assert "message" in violation_dict
    
    def test_result_to_dict(self, invalid_ohlc_data, validator):
        """Test result serialization to dict."""
        result = validator.validate(invalid_ohlc_data)
        
        result_dict = result.to_dict()
        assert "passed" in result_dict
        assert "total_bars_checked" in result_dict
        assert "total_violations" in result_dict
        assert "violations" in result_dict
    
    def test_result_string_representation(self, valid_ohlcv_data, invalid_ohlc_data, validator):
        """Test string representation of results."""
        valid_result = validator.validate(valid_ohlcv_data)
        invalid_result = validator.validate(invalid_ohlc_data)
        
        valid_str = str(valid_result)
        invalid_str = str(invalid_result)
        
        assert "✓" in valid_str
        assert "✗" in invalid_str


# ============================================================================
# Tests: Batch Validation
# ============================================================================

class TestBatchValidation:
    """Tests for batch validation of multiple datasets."""
    
    def test_batch_validation(self, valid_ohlcv_data, invalid_ohlc_data):
        """Test batch validation of multiple datasets."""
        datasets = {
            "valid": valid_ohlcv_data,
            "invalid": invalid_ohlc_data,
        }
        
        results = validate_multiple_datasets(datasets)
        
        assert "valid" in results
        assert "invalid" in results
        assert results["valid"].passed
        assert not results["invalid"].passed
    
    def test_batch_validation_with_validator(self, valid_ohlcv_data, invalid_ohlc_data):
        """Test batch validation using DataValidator.validate_batch."""
        validator = DataValidator(timezone="US/Eastern")
        datasets = {
            "dataset1": valid_ohlcv_data,
            "dataset2": invalid_ohlc_data,
        }
        
        results = validator.validate_batch(datasets)
        
        assert len(results) == 2
        assert results["dataset1"].passed
        assert not results["dataset2"].passed


# ============================================================================
# Tests: Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience validation functions."""
    
    def test_validate_ohlcv_data_function(self, valid_ohlcv_data):
        """Test the validate_ohlcv_data convenience function."""
        result = validate_ohlcv_data(valid_ohlcv_data)
        
        assert result.passed
        assert result.total_bars_checked == len(valid_ohlcv_data)
    
    def test_validate_ohlcv_with_custom_settings(self, valid_ohlcv_data):
        """Test validate_ohlcv_data with custom settings."""
        result = validate_ohlcv_data(
            valid_ohlcv_data,
            float_tolerance=1e-6,
            timezone="US/Eastern",
        )
        
        assert result.passed
