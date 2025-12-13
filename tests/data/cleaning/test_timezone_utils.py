"""
Comprehensive test suite for timezone conversion utilities.

Tests cover:
- UTC to NY timezone conversion
- Other timezones to NY conversion
- DST transitions (spring forward and fall back)
- Naive timestamp detection and handling
- Timezone validation
- Performance with large datasets (1M+ rows)
- Edge cases (NaT values, empty DataFrames, etc.)

Note: These tests are timezone-aware and may behave differently
depending on the system timezone. All assertions use explicit timezone
specifications to ensure consistency.
"""

import pytest
import pandas as pd
import numpy as np
import pytz
import logging
from datetime import datetime, timedelta
import time

from src.data.cleaning.timezone_utils import (
    normalize_to_ny_timezone,
    validate_timezone,
    detect_timezone,
    has_naive_timestamps,
    get_ny_timezone_offset,
    localize_to_ny,
    NY_TIMEZONE,
    UTC_TIMEZONE,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_utc_data():
    """Create sample OHLCV data in UTC timezone."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1H", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 100),
        "high": np.random.uniform(110, 120, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(100, 110, 100),
        "volume": np.random.randint(1000, 10000, 100),
    })


@pytest.fixture
def sample_naive_data():
    """Create sample OHLCV data with naive timestamps."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1H")
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 100),
        "high": np.random.uniform(110, 120, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(100, 110, 100),
        "volume": np.random.randint(1000, 10000, 100),
    })


@pytest.fixture
def sample_london_data():
    """Create sample OHLCV data in London (Europe/London) timezone."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1H", tz="Europe/London")
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 100),
        "high": np.random.uniform(110, 120, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(100, 110, 100),
        "volume": np.random.randint(1000, 10000, 100),
    })


@pytest.fixture
def sample_tokyo_data():
    """Create sample OHLCV data in Tokyo (Asia/Tokyo) timezone."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1H", tz="Asia/Tokyo")
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 100),
        "high": np.random.uniform(110, 120, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(100, 110, 100),
        "volume": np.random.randint(1000, 10000, 100),
    })


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame with timestamp column."""
    return pd.DataFrame({
        "timestamp": pd.DatetimeIndex([], tz="UTC"),
        "value": [],
    })


# ============================================================================
# Tests: UTC to NY Conversion
# ============================================================================


class TestUTCToNYConversion:
    """Tests for converting UTC timestamps to US/Eastern."""

    def test_utc_to_ny_basic(self, sample_utc_data):
        """Test basic UTC to NY conversion."""
        df_ny = normalize_to_ny_timezone(sample_utc_data)

        # Check timezone is correct
        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"

        # Check data integrity (same number of rows)
        assert len(df_ny) == len(sample_utc_data)

        # Check timestamp values are correctly converted
        # 2024-01-01 00:00:00 UTC = 2024-01-01 19:00:00 EST (UTC-5)
        first_ts_utc = sample_utc_data["timestamp"].iloc[0]
        first_ts_ny = df_ny["timestamp"].iloc[0]

        assert first_ts_utc.hour == 0  # UTC hour
        assert first_ts_ny.hour == 19  # NY hour (EST, UTC-5)
        assert first_ts_ny.day == 0 if first_ts_ny.hour == 19 else 1

    def test_utc_to_ny_preserves_data_integrity(self, sample_utc_data):
        """Test that conversion preserves all other columns."""
        df_ny = normalize_to_ny_timezone(sample_utc_data)

        # Check other columns are preserved
        assert "open" in df_ny.columns
        assert "close" in df_ny.columns
        assert "volume" in df_ny.columns

        # Check values match
        pd.testing.assert_series_equal(
            df_ny["open"], sample_utc_data["open"], check_names=True
        )
        pd.testing.assert_series_equal(
            df_ny["volume"], sample_utc_data["volume"], check_names=True
        )

    def test_utc_to_ny_with_custom_column_name(self, sample_utc_data):
        """Test conversion with custom timestamp column name."""
        df = sample_utc_data.rename(columns={"timestamp": "time"})

        df_ny = normalize_to_ny_timezone(df, timestamp_col="time")

        assert str(df_ny["time"].dt.tz) == "US/Eastern"

    def test_utc_to_ny_empty_dataframe(self, empty_dataframe):
        """Test conversion of empty DataFrame."""
        df_ny = normalize_to_ny_timezone(empty_dataframe)

        assert len(df_ny) == 0
        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"

    def test_utc_to_ny_invalid_column_raises_error(self, sample_utc_data):
        """Test that invalid column name raises error."""
        with pytest.raises(ValueError):
            normalize_to_ny_timezone(sample_utc_data, timestamp_col="nonexistent")

    def test_utc_to_ny_does_not_modify_original(self, sample_utc_data):
        """Test that original DataFrame is not modified."""
        original_tz = str(sample_utc_data["timestamp"].dt.tz)

        _ = normalize_to_ny_timezone(sample_utc_data)

        # Original should still be UTC
        assert str(sample_utc_data["timestamp"].dt.tz) == original_tz


# ============================================================================
# Tests: Other Timezones to NY Conversion
# ============================================================================


class TestOtherTimezonesToNYConversion:
    """Tests for converting other timezones to US/Eastern."""

    def test_london_to_ny(self, sample_london_data):
        """Test conversion from London (Europe/London) to NY."""
        df_ny = normalize_to_ny_timezone(sample_london_data)

        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"

        # 2024-01-01 00:00:00 London = 2024-01-01 19:00:00 EST
        # (London is UTC+0 in January, NY is UTC-5)
        first_ts_london = sample_london_data["timestamp"].iloc[0]
        first_ts_ny = df_ny["timestamp"].iloc[0]

        # UTC times should match
        london_utc = first_ts_london.tz_convert("UTC")
        ny_utc = first_ts_ny.tz_convert("UTC")
        assert london_utc == ny_utc

    def test_tokyo_to_ny(self, sample_tokyo_data):
        """Test conversion from Tokyo (Asia/Tokyo) to NY."""
        df_ny = normalize_to_ny_timezone(sample_tokyo_data)

        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"

        # UTC times should match
        first_ts_tokyo = sample_tokyo_data["timestamp"].iloc[0]
        first_ts_ny = df_ny["timestamp"].iloc[0]

        tokyo_utc = first_ts_tokyo.tz_convert("UTC")
        ny_utc = first_ts_ny.tz_convert("UTC")
        assert tokyo_utc == ny_utc

    def test_multiple_timezones_consistency(self):
        """Test that same moment in different timezones converts correctly."""
        # Create the same moment in different timezones
        moment_utc = pd.Timestamp("2024-06-15 12:00:00", tz="UTC")
        moment_london = moment_utc.tz_convert("Europe/London")
        moment_tokyo = moment_utc.tz_convert("Asia/Tokyo")

        df_utc = pd.DataFrame({"timestamp": [moment_utc]})
        df_london = pd.DataFrame({"timestamp": [moment_london]})
        df_tokyo = pd.DataFrame({"timestamp": [moment_tokyo]})

        df_ny_from_utc = normalize_to_ny_timezone(df_utc)
        df_ny_from_london = normalize_to_ny_timezone(df_london)
        df_ny_from_tokyo = normalize_to_ny_timezone(df_tokyo)

        # All should result in the same NY time
        ny_time = df_ny_from_utc["timestamp"].iloc[0]
        assert df_ny_from_london["timestamp"].iloc[0] == ny_time
        assert df_ny_from_tokyo["timestamp"].iloc[0] == ny_time


# ============================================================================
# Tests: DST Transitions
# ============================================================================


class TestDSTTransitions:
    """Tests for DST (Daylight Saving Time) handling."""

    def test_dst_spring_forward_2024(self):
        """Test spring forward DST transition (March 10, 2024 at 2:00 AM)."""
        # Create timestamps around spring forward
        # 2024-03-10 at 2:00 AM EST -> 3:00 AM EDT
        dates = pd.date_range(
            "2024-03-10 00:00:00",
            periods=6,
            freq="1H",
            tz="US/Eastern"
        )

        df = pd.DataFrame({
            "timestamp": dates,
            "value": [1, 2, 3, 4, 5, 6],
        })

        # Validate timestamps are valid (no gaps)
        assert len(df) == 6
        assert all(pd.notna(df["timestamp"]))

        # Check DST transition happened
        # 01:00 EDT exists, then 03:00 EDT (2:00 AM skipped)
        times = [ts.time() for ts in df["timestamp"]]
        assert datetime.time(hour=1) in times  # Before transition
        assert datetime.time(hour=3) in times  # After transition

    def test_dst_fall_back_2024(self):
        """Test fall back DST transition (November 3, 2024 at 2:00 AM)."""
        # Create timestamps around fall back
        # 2024-11-03 at 2:00 AM EDT -> 1:00 AM EST
        dates = pd.date_range(
            "2024-11-02 23:00:00",
            periods=6,
            freq="1H",
            tz="UTC"
        )

        df = pd.DataFrame({
            "timestamp": dates,
            "value": [1, 2, 3, 4, 5, 6],
        })

        df_ny = normalize_to_ny_timezone(df)

        # Check all timestamps are preserved
        assert len(df_ny) == 6
        assert all(pd.notna(df_ny["timestamp"]))
        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"

    def test_dst_transition_preserves_data_order(self):
        """Test that DST transitions don't affect data ordering."""
        # Create data spanning DST transition
        dates = pd.date_range(
            "2024-03-09",
            periods=48,
            freq="1H",
            tz="UTC"
        )

        df = pd.DataFrame({
            "timestamp": dates,
            "value": range(48),  # Sequential values
        })

        df_ny = normalize_to_ny_timezone(df)

        # Values should still be in order
        assert all(df_ny["value"].iloc[i] <= df_ny["value"].iloc[i + 1]
                   for i in range(len(df_ny) - 1))

    def test_naive_timestamps_across_dst(self):
        """Test handling of naive timestamps across DST transition."""
        # Create naive timestamps around DST
        dates = pd.date_range("2024-03-09", periods=48, freq="1H")

        df = pd.DataFrame({
            "timestamp": dates,
            "value": range(48),
        })

        # Should handle without error
        df_ny = normalize_to_ny_timezone(df)

        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"
        assert len(df_ny) == 48

    def test_est_vs_edt_offset(self):
        """Test that EST/EDT offsets are correct."""
        # January (EST, UTC-5)
        jan_ts = pd.Timestamp("2024-01-15 12:00:00", tz="US/Eastern")
        assert get_ny_timezone_offset(jan_ts) == -5

        # July (EDT, UTC-4)
        jul_ts = pd.Timestamp("2024-07-15 12:00:00", tz="US/Eastern")
        assert get_ny_timezone_offset(jul_ts) == -4


# ============================================================================
# Tests: Naive Timestamp Handling
# ============================================================================


class TestNaiveTimestampHandling:
    """Tests for handling naive (timezone-unaware) timestamps."""

    def test_naive_timestamps_detected(self, sample_naive_data):
        """Test that naive timestamps are detected."""
        assert has_naive_timestamps(sample_naive_data)

    def test_naive_timestamps_assumed_utc(self, sample_naive_data):
        """Test that naive timestamps are assumed to be UTC."""
        df_ny = normalize_to_ny_timezone(sample_naive_data)

        # Check conversion is valid
        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"

        # Check that first time is correctly converted
        # 2024-01-01 00:00:00 (assumed UTC) -> 2024-01-01 19:00:00 EST
        first_ny = df_ny["timestamp"].iloc[0]
        assert first_ny.hour == 19  # EST offset is -5

    def test_naive_timestamps_with_explicit_source_tz(self):
        """Test naive timestamps with explicit source timezone."""
        dates = pd.date_range("2024-01-01", periods=10, freq="1H")
        df = pd.DataFrame({
            "timestamp": dates,
            "value": range(10),
        })

        # Specify source timezone as UTC
        df_ny_utc = normalize_to_ny_timezone(df, source_timezone="UTC")

        # Specify source timezone as Europe/London
        df_ny_london = normalize_to_ny_timezone(df, source_timezone="Europe/London")

        # Results should be different
        assert not (df_ny_utc["timestamp"] == df_ny_london["timestamp"]).all()

    def test_naive_timestamp_warning_logged(self, sample_naive_data, caplog):
        """Test that warning is logged for naive timestamps."""
        with caplog.at_level(logging.WARNING):
            normalize_to_ny_timezone(sample_naive_data)

        # Check that a warning was logged
        assert any("Naive timestamps" in record.message for record in caplog.records)

    def test_localize_to_ny_function(self, sample_naive_data):
        """Test convenience function for localizing naive timestamps."""
        df_ny = localize_to_ny(sample_naive_data, naive_tz="UTC")

        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"
        assert len(df_ny) == len(sample_naive_data)


# ============================================================================
# Tests: Timezone Validation
# ============================================================================


class TestTimezoneValidation:
    """Tests for timezone validation."""

    def test_validate_ny_timezone_success(self, sample_utc_data):
        """Test validation succeeds for NY timezone data."""
        df_ny = normalize_to_ny_timezone(sample_utc_data)

        assert validate_timezone(df_ny)

    def test_validate_ny_timezone_fails_utc(self, sample_utc_data):
        """Test validation fails for UTC data."""
        assert not validate_timezone(sample_utc_data)

    def test_validate_ny_timezone_fails_naive(self, sample_naive_data):
        """Test validation fails for naive timestamps."""
        assert not validate_timezone(sample_naive_data)

    def test_validate_with_custom_timezone(self, sample_utc_data):
        """Test validation with custom expected timezone."""
        assert validate_timezone(sample_utc_data, expected_tz="UTC")

    def test_validate_empty_dataframe(self, empty_dataframe):
        """Test validation of empty DataFrame."""
        df_ny = normalize_to_ny_timezone(empty_dataframe)
        assert validate_timezone(df_ny)

    def test_validate_dataframe_with_nat(self, sample_utc_data):
        """Test validation with NaT values."""
        df = sample_utc_data.copy()
        df.loc[0, "timestamp"] = pd.NaT

        df_ny = df.copy()
        df_ny["timestamp"] = df_ny["timestamp"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")

        # Should still be valid but should warn about NaT
        with pytest.warns(None) as warning_list:
            result = validate_timezone(df_ny)

        # Result should be True (NaT doesn't invalidate, just warns)
        assert result

    def test_validate_invalid_column_raises_error(self, sample_utc_data):
        """Test validation with invalid column name."""
        with pytest.raises(ValueError):
            validate_timezone(sample_utc_data, timestamp_col="nonexistent")


# ============================================================================
# Tests: Timezone Detection
# ============================================================================


class TestTimezoneDetection:
    """Tests for timezone detection."""

    def test_detect_utc_timezone(self, sample_utc_data):
        """Test detection of UTC timezone."""
        tz = detect_timezone(sample_utc_data)
        assert tz == "UTC"

    def test_detect_naive_timezone(self, sample_naive_data):
        """Test detection of naive timestamps."""
        tz = detect_timezone(sample_naive_data)
        assert tz is None

    def test_detect_ny_timezone(self, sample_utc_data):
        """Test detection of NY timezone."""
        df_ny = normalize_to_ny_timezone(sample_utc_data)
        tz = detect_timezone(df_ny)
        assert tz == "US/Eastern"

    def test_detect_london_timezone(self, sample_london_data):
        """Test detection of London timezone."""
        tz = detect_timezone(sample_london_data)
        assert tz == "Europe/London"

    def test_detect_with_custom_column(self, sample_utc_data):
        """Test timezone detection with custom column name."""
        df = sample_utc_data.rename(columns={"timestamp": "time"})
        tz = detect_timezone(df, timestamp_col="time")
        assert tz == "UTC"

    def test_detect_invalid_column_raises_error(self, sample_utc_data):
        """Test timezone detection with invalid column."""
        with pytest.raises(ValueError):
            detect_timezone(sample_utc_data, timestamp_col="nonexistent")


# ============================================================================
# Tests: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_non_datetime_column_raises_error(self):
        """Test that non-datetime columns raise error."""
        df = pd.DataFrame({
            "timestamp": ["2024-01-01", "2024-01-02"],
            "value": [1, 2],
        })

        with pytest.raises(TypeError):
            normalize_to_ny_timezone(df)

    def test_single_row_dataframe(self, sample_utc_data):
        """Test conversion of single-row DataFrame."""
        df_single = sample_utc_data.iloc[[0]].copy()
        df_ny = normalize_to_ny_timezone(df_single)

        assert len(df_ny) == 1
        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"

    def test_duplicate_timestamps(self, sample_utc_data):
        """Test handling of duplicate timestamps."""
        df = sample_utc_data.copy()
        df.loc[1, "timestamp"] = df.loc[0, "timestamp"]

        df_ny = normalize_to_ny_timezone(df)

        # Should handle duplicates without error
        assert len(df_ny) == len(df)

    def test_unsorted_timestamps(self, sample_utc_data):
        """Test handling of unsorted timestamps."""
        df = sample_utc_data.copy()
        df = df.sample(frac=1).reset_index(drop=True)

        df_ny = normalize_to_ny_timezone(df)

        # Order may be different, but conversion should work
        assert len(df_ny) == len(df)
        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"

    def test_already_ny_timezone(self, sample_utc_data):
        """Test conversion of already-NY-timezone data."""
        df = sample_utc_data.copy()
        df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")

        df_ny = normalize_to_ny_timezone(df)

        # Should remain NY timezone
        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"
        pd.testing.assert_frame_equal(df, df_ny)

    def test_very_large_date_range(self):
        """Test handling of very large date ranges."""
        # Create 10 years of hourly data
        dates = pd.date_range(
            "2010-01-01",
            periods=87660,  # 10 years of hours
            freq="1H",
            tz="UTC"
        )

        df = pd.DataFrame({
            "timestamp": dates,
            "value": np.random.random(len(dates)),
        })

        df_ny = normalize_to_ny_timezone(df)

        assert len(df_ny) == len(df)
        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"


# ============================================================================
# Tests: Performance
# ============================================================================


class TestPerformance:
    """Tests for performance with large datasets."""

    def test_performance_1m_rows(self):
        """Test performance: process 1M rows in < 5 seconds."""
        # Create 1M rows of hourly data (about 114 years)
        dates = pd.date_range("2000-01-01", periods=1_000_000, freq="1H", tz="UTC")

        df = pd.DataFrame({
            "timestamp": dates,
            "value": np.random.random(1_000_000),
        })

        start_time = time.time()
        df_ny = normalize_to_ny_timezone(df)
        elapsed = time.time() - start_time

        assert len(df_ny) == 1_000_000
        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"
        assert elapsed < 5.0, f"Processing took {elapsed:.2f}s, expected < 5s"

    def test_performance_validation_1m_rows(self):
        """Test validation performance on 1M rows."""
        dates = pd.date_range("2000-01-01", periods=1_000_000, freq="1H", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "value": np.random.random(1_000_000),
        })

        df_ny = normalize_to_ny_timezone(df)

        start_time = time.time()
        result = validate_timezone(df_ny)
        elapsed = time.time() - start_time

        assert result
        assert elapsed < 1.0, f"Validation took {elapsed:.2f}s, expected < 1s"

    def test_performance_timezone_detection_1m_rows(self):
        """Test timezone detection performance on 1M rows."""
        dates = pd.date_range("2000-01-01", periods=1_000_000, freq="1H", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "value": np.random.random(1_000_000),
        })

        start_time = time.time()
        tz = detect_timezone(df)
        elapsed = time.time() - start_time

        assert tz == "UTC"
        assert elapsed < 0.1, f"Detection took {elapsed:.2f}s, expected < 0.1s"

    def test_performance_scaling(self):
        """Test that performance scales linearly."""
        sizes = [100_000, 500_000, 1_000_000]
        times = []

        for size in sizes:
            dates = pd.date_range("2000-01-01", periods=size, freq="1H", tz="UTC")
            df = pd.DataFrame({
                "timestamp": dates,
                "value": np.random.random(size),
            })

            start_time = time.time()
            _ = normalize_to_ny_timezone(df)
            elapsed = time.time() - start_time
            times.append(elapsed)

        # Check roughly linear scaling (time should increase proportionally to size)
        # Rough check: 500K should take ~2.5x time of 100K
        ratio_1 = times[1] / times[0]
        assert 2.0 < ratio_1 < 6.0, f"Scaling ratio {ratio_1} is not linear"

        # 1M should take ~2x time of 500K
        ratio_2 = times[2] / times[1]
        assert 1.5 < ratio_2 < 3.0, f"Scaling ratio {ratio_2} is not linear"


# ============================================================================
# Tests: Integration with OHLCV Data
# ============================================================================


class TestOHLCVIntegration:
    """Tests for integration with OHLCV trading data."""

    def test_full_ohlcv_pipeline(self, sample_utc_data):
        """Test complete pipeline: load UTC data, convert to NY, validate."""
        # Step 1: Load (simulated - already have UTC data)
        df = sample_utc_data.copy()

        # Step 2: Normalize
        df = normalize_to_ny_timezone(df)

        # Step 3: Validate
        is_valid = validate_timezone(df)

        # Step 4: Use in analysis
        assert is_valid
        assert len(df) > 0
        assert str(df["timestamp"].dt.tz) == "US/Eastern"

    def test_market_hours_conversion(self):
        """Test conversion of market hours (9:30 AM - 4:00 PM NY time)."""
        # Create some market hours in UTC
        # 9:30 AM EST = 2:30 PM UTC, 4:00 PM EST = 9:00 PM UTC
        dates = pd.date_range(
            "2024-01-02 14:30:00",  # 9:30 AM EST
            "2024-01-02 21:00:00",  # 4:00 PM EST
            freq="1H",
            tz="UTC"
        )

        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        })

        df_ny = normalize_to_ny_timezone(df)

        # Check market hours times
        ny_hours = df_ny["timestamp"].dt.hour
        assert ny_hours.iloc[0] == 9  # 9:XX AM
        assert ny_hours.iloc[-1] >= 15  # >= 3:00 PM (may be 4:00)

    def test_different_data_source_timezones(self):
        """Test that data from different sources converts correctly."""
        # Data from US broker (local NY time, but as naive)
        ny_naive = pd.date_range("2024-01-02 09:30:00", periods=5, freq="1H")

        # Data from UK broker (London time)
        london_ts = pd.date_range(
            "2024-01-02 09:30:00",
            periods=5,
            freq="1H",
            tz="Europe/London"
        )

        # Data from Asia broker (Tokyo time)
        tokyo_ts = pd.date_range(
            "2024-01-02 23:30:00",  # Next day in Tokyo
            periods=5,
            freq="1H",
            tz="Asia/Tokyo"
        )

        df_ny = pd.DataFrame({"timestamp": ny_naive})
        df_london = pd.DataFrame({"timestamp": london_ts})
        df_tokyo = pd.DataFrame({"timestamp": tokyo_ts})

        df_ny_norm = normalize_to_ny_timezone(df_ny, source_timezone="US/Eastern")
        df_london_norm = normalize_to_ny_timezone(df_london)
        df_tokyo_norm = normalize_to_ny_timezone(df_tokyo)

        # All should be in NY timezone
        assert str(df_ny_norm["timestamp"].dt.tz) == "US/Eastern"
        assert str(df_london_norm["timestamp"].dt.tz) == "US/Eastern"
        assert str(df_tokyo_norm["timestamp"].dt.tz) == "US/Eastern"


# ============================================================================
# Tests: Real-world Scenarios
# ============================================================================


class TestRealWorldScenarios:
    """Tests for real-world trading scenarios."""

    def test_24_hour_trading_cycle(self):
        """Test handling of 24-hour trading cycle."""
        # Create 24 hours of data starting from UTC midnight
        dates = pd.date_range("2024-01-01 00:00:00", periods=24, freq="1H", tz="UTC")

        df = pd.DataFrame({
            "timestamp": dates,
            "value": range(1, 25),
        })

        df_ny = normalize_to_ny_timezone(df)

        # 2024-01-01 00:00 UTC = 2023-12-31 19:00 EST (previous day, 7 PM)
        assert df_ny["timestamp"].iloc[0].day == 31  # Dec 31
        assert df_ny["timestamp"].iloc[-1].day == 1   # Jan 1

    def test_weekend_gap_handling(self):
        """Test handling of weekend data gaps."""
        # Friday close to Monday open (missing Sat, Sun)
        dates = pd.to_datetime([
            "2024-01-05 20:00:00",  # Friday 3 PM EST
            "2024-01-08 14:30:00",  # Monday 9:30 AM EST
        ], utc=True)

        df = pd.DataFrame({
            "timestamp": dates,
            "value": [100, 101],
        })

        df_ny = normalize_to_ny_timezone(df)

        assert len(df_ny) == 2
        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"

    def test_high_frequency_data(self):
        """Test handling of high-frequency (millisecond) data."""
        # Create microsecond-level timestamps
        dates = pd.date_range(
            "2024-01-01 13:30:00",
            periods=1000,
            freq="1ms",
            tz="UTC"
        )

        df = pd.DataFrame({
            "timestamp": dates,
            "value": np.random.random(1000),
        })

        df_ny = normalize_to_ny_timezone(df)

        assert len(df_ny) == 1000
        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"
        # Microsecond precision should be preserved
        assert df_ny["timestamp"].iloc[1] - df_ny["timestamp"].iloc[0] == timedelta(milliseconds=1)

    def test_mixed_frequency_data(self):
        """Test handling of mixed frequency data (e.g., different time gaps)."""
        dates = pd.to_datetime([
            "2024-01-01 13:30:00",
            "2024-01-01 13:31:00",
            "2024-01-01 14:00:00",  # Gap
            "2024-01-01 14:01:00",
            "2024-01-01 14:02:00",
        ], utc=True)

        df = pd.DataFrame({
            "timestamp": dates,
            "value": [1, 2, 3, 4, 5],
        })

        df_ny = normalize_to_ny_timezone(df)

        assert len(df_ny) == 5
        assert str(df_ny["timestamp"].dt.tz) == "US/Eastern"

