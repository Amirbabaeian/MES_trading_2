"""
Comprehensive test suite for trading calendar functionality.

Tests cover:
- Regular trading days (weekdays, Monday-Friday)
- Weekend and holiday detection
- Trading hours for different assets (ES/MES vs VIX)
- Session type identification (regular, pre-market, post-market, overnight)
- Partial trading days (early closes)
- Session transitions
- DataFrame filtering for trading hours
- Support for different assets (ES, MES, VIX)
"""

import pytest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, time
from typing import List

from src.data.cleaning.trading_calendar import (
    is_trading_day,
    get_trading_hours,
    get_session_type,
    filter_trading_hours,
    get_supported_assets,
    get_next_trading_day,
    get_previous_trading_day,
    get_trading_days,
    SessionType,
    ASSETS,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data spanning a full week."""
    # 2024-01-01 is New Year's Day (Monday)
    # 2024-01-02 is Tuesday (trading)
    dates = pd.date_range("2024-01-01", periods=100, freq="1H", tz="US/Eastern")
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 100),
        "high": np.random.uniform(110, 120, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(100, 110, 100),
        "volume": np.random.randint(1000, 10000, 100),
    })


@pytest.fixture
def sample_extended_hours_data():
    """Create OHLCV data including extended hours."""
    # 2024-01-02 is Tuesday (full trading day)
    dates = pd.date_range("2024-01-02", periods=24, freq="1H", tz="US/Eastern")
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 24),
        "high": np.random.uniform(110, 120, 24),
        "low": np.random.uniform(90, 100, 24),
        "close": np.random.uniform(100, 110, 24),
        "volume": np.random.randint(1000, 10000, 24),
    })


@pytest.fixture
def sample_week_data():
    """Create OHLCV data spanning a full week (2024-01-01 to 2024-01-07)."""
    # Monday 1/1 is holiday, Tue-Fri 1/2-1/5 are trading days, Sat-Sun 1/6-1/7 are weekend
    dates = pd.date_range("2024-01-01", periods=168, freq="1H", tz="US/Eastern")
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 168),
        "high": np.random.uniform(110, 120, 168),
        "low": np.random.uniform(90, 100, 168),
        "close": np.random.uniform(100, 110, 168),
        "volume": np.random.randint(1000, 10000, 168),
    })


@pytest.fixture
def naive_timestamp_data():
    """Create OHLCV data with naive timestamps."""
    dates = pd.date_range("2024-01-02", periods=24, freq="1H")
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 24),
        "high": np.random.uniform(110, 120, 24),
        "low": np.random.uniform(90, 100, 24),
        "close": np.random.uniform(100, 110, 24),
        "volume": np.random.randint(1000, 10000, 24),
    })


# ============================================================================
# Tests: Regular Trading Days
# ============================================================================


class TestRegularTradingDays:
    """Tests for identifying regular trading days."""

    def test_weekday_is_trading_day(self):
        """Test that weekdays are trading days."""
        # 2024-01-02 is Tuesday
        assert is_trading_day(pd.Timestamp("2024-01-02")) is True
        # 2024-01-03 is Wednesday
        assert is_trading_day(pd.Timestamp("2024-01-03")) is True
        # 2024-01-04 is Thursday
        assert is_trading_day(pd.Timestamp("2024-01-04")) is True
        # 2024-01-05 is Friday
        assert is_trading_day(pd.Timestamp("2024-01-05")) is True

    def test_weekend_is_not_trading_day(self):
        """Test that weekends are not trading days."""
        # 2024-01-06 is Saturday
        assert is_trading_day(pd.Timestamp("2024-01-06")) is False
        # 2024-01-07 is Sunday
        assert is_trading_day(pd.Timestamp("2024-01-07")) is False

    def test_morning_monday_after_weekend(self):
        """Test Monday after weekend is a trading day (if not a holiday)."""
        # 2024-01-08 is Monday (not a holiday)
        assert is_trading_day(pd.Timestamp("2024-01-08")) is True

    def test_trading_day_accepts_timezone_aware_timestamp(self):
        """Test that is_trading_day works with timezone-aware timestamps."""
        ts = pd.Timestamp("2024-01-02 09:30:00", tz="US/Eastern")
        assert is_trading_day(ts) is True

    def test_trading_day_accepts_utc_timestamp(self):
        """Test that is_trading_day works with UTC timestamps."""
        # 2024-01-02 14:30 UTC = 2024-01-02 09:30 EST
        ts = pd.Timestamp("2024-01-02 14:30:00", tz="UTC")
        assert is_trading_day(ts) is True


# ============================================================================
# Tests: Holidays and Special Dates
# ============================================================================


class TestHolidaysAndSpecialDates:
    """Tests for holiday and special date handling."""

    def test_new_years_day_not_trading(self):
        """Test that New Year's Day is not a trading day."""
        # 2024-01-01 is New Year's Day
        assert is_trading_day(pd.Timestamp("2024-01-01")) is False
        assert is_trading_day(pd.Timestamp("2025-01-01")) is False

    def test_good_friday_not_trading(self):
        """Test that Good Friday is not a trading day."""
        # 2024-03-29 is Good Friday
        assert is_trading_day(pd.Timestamp("2024-03-29")) is False
        # 2025-04-18 is Good Friday
        assert is_trading_day(pd.Timestamp("2025-04-18")) is False

    def test_independence_day_not_trading(self):
        """Test that Independence Day is not a trading day."""
        # 2024-07-04 is July 4th
        assert is_trading_day(pd.Timestamp("2024-07-04")) is False
        assert is_trading_day(pd.Timestamp("2025-07-04")) is False

    def test_thanksgiving_not_trading(self):
        """Test that Thanksgiving is not a trading day."""
        # 2024-11-28 is Thanksgiving
        assert is_trading_day(pd.Timestamp("2024-11-28")) is False
        # 2025-11-27 is Thanksgiving
        assert is_trading_day(pd.Timestamp("2025-11-27")) is False

    def test_christmas_not_trading(self):
        """Test that Christmas is not a trading day."""
        # 2024-12-25 is Christmas
        assert is_trading_day(pd.Timestamp("2024-12-25")) is False
        assert is_trading_day(pd.Timestamp("2025-12-25")) is False

    def test_day_after_thanksgiving_is_trading(self):
        """Test that the day after Thanksgiving is a trading day (with early close)."""
        # 2024-11-29 is the day after Thanksgiving (Friday)
        assert is_trading_day(pd.Timestamp("2024-11-29")) is True

    def test_supported_assets_for_holidays(self):
        """Test that holiday dates are the same across all assets."""
        holiday = pd.Timestamp("2024-01-01")
        for asset in ASSETS:
            assert is_trading_day(holiday, asset) is False


# ============================================================================
# Tests: Trading Hours
# ============================================================================


class TestTradingHours:
    """Tests for getting trading hours."""

    def test_get_trading_hours_returns_dict(self):
        """Test that get_trading_hours returns a dictionary."""
        hours = get_trading_hours(pd.Timestamp("2024-01-02"))
        assert isinstance(hours, dict)

    def test_get_trading_hours_not_trading_day_returns_none(self):
        """Test that get_trading_hours returns None for non-trading days."""
        # Weekend
        assert get_trading_hours(pd.Timestamp("2024-01-06")) is None
        # Holiday
        assert get_trading_hours(pd.Timestamp("2024-01-01")) is None

    def test_get_trading_hours_es_includes_globex(self):
        """Test that ES trading hours include Globex times."""
        hours = get_trading_hours(pd.Timestamp("2024-01-02"), asset="ES")
        assert "globex_open" in hours
        assert "globex_close" in hours
        assert hours["globex_open"] == "18:00"
        assert hours["globex_close"] == "17:00"

    def test_get_trading_hours_es_regular_hours(self):
        """Test that ES regular hours are 9:30am-4:00pm."""
        hours = get_trading_hours(pd.Timestamp("2024-01-02"), asset="ES")
        assert hours["regular_open"] == "09:30"
        assert hours["regular_close"] == "16:00"

    def test_get_trading_hours_vix_regular_hours(self):
        """Test that VIX regular hours are 9:30am-4:15pm (not 4:00pm)."""
        hours = get_trading_hours(pd.Timestamp("2024-01-02"), asset="VIX")
        assert hours["regular_open"] == "09:30"
        assert hours["regular_close"] == "16:15"

    def test_get_trading_hours_vix_no_globex(self):
        """Test that VIX hours don't include Globex times."""
        hours = get_trading_hours(pd.Timestamp("2024-01-02"), asset="VIX")
        assert "globex_open" not in hours
        assert "globex_close" not in hours

    def test_get_trading_hours_early_close_day(self):
        """Test that early close days are marked and have updated close time."""
        # 2024-11-27 is day before Thanksgiving (early close at 2:00pm)
        hours = get_trading_hours(pd.Timestamp("2024-11-27"))
        assert hours["is_early_close"] is True
        assert hours["early_close_time"] == "14:00"
        assert hours["regular_close"] == "14:00"

    def test_get_trading_hours_mes_same_as_es(self):
        """Test that MES has same hours as ES."""
        es_hours = get_trading_hours(pd.Timestamp("2024-01-02"), asset="ES")
        mes_hours = get_trading_hours(pd.Timestamp("2024-01-02"), asset="MES")
        assert es_hours == mes_hours

    def test_invalid_asset_raises_error(self):
        """Test that invalid asset name raises ValueError."""
        with pytest.raises(ValueError):
            get_trading_hours(pd.Timestamp("2024-01-02"), asset="INVALID")


# ============================================================================
# Tests: Session Type Classification
# ============================================================================


class TestSessionType:
    """Tests for session type classification."""

    def test_regular_trading_hours_es(self):
        """Test that regular trading hours are classified as REGULAR."""
        # 2024-01-02 9:30am EST
        ts = pd.Timestamp("2024-01-02 09:30:00", tz="US/Eastern")
        assert get_session_type(ts, asset="ES") == SessionType.REGULAR

        # 2024-01-02 3:00pm EST
        ts = pd.Timestamp("2024-01-02 15:00:00", tz="US/Eastern")
        assert get_session_type(ts, asset="ES") == SessionType.REGULAR

    def test_premarket_session_es(self):
        """Test that pre-market times are classified as PRE_MARKET."""
        # 2024-01-02 6:00am EST (before 9:30am open)
        ts = pd.Timestamp("2024-01-02 06:00:00", tz="US/Eastern")
        assert get_session_type(ts, asset="ES") == SessionType.PRE_MARKET

    def test_postmarket_session_es(self):
        """Test that post-market times are classified as POST_MARKET."""
        # 2024-01-02 5:00pm EST (after 4:00pm close)
        ts = pd.Timestamp("2024-01-02 17:00:00", tz="US/Eastern")
        assert get_session_type(ts, asset="ES") == SessionType.POST_MARKET

    def test_overnight_session_es(self):
        """Test that overnight times are classified as OVERNIGHT."""
        # 2024-01-02 10:00pm EST (after 8pm post-market end)
        ts = pd.Timestamp("2024-01-02 22:00:00", tz="US/Eastern")
        assert get_session_type(ts, asset="ES") == SessionType.OVERNIGHT

    def test_closed_session_weekend(self):
        """Test that weekend times are classified as CLOSED."""
        # 2024-01-06 12:00pm EST (Saturday)
        ts = pd.Timestamp("2024-01-06 12:00:00", tz="US/Eastern")
        assert get_session_type(ts, asset="ES") == SessionType.CLOSED

    def test_closed_session_holiday(self):
        """Test that holiday times are classified as CLOSED."""
        # 2024-01-01 12:00pm EST (New Year's Day)
        ts = pd.Timestamp("2024-01-01 12:00:00", tz="US/Eastern")
        assert get_session_type(ts, asset="ES") == SessionType.CLOSED

    def test_vix_no_extended_hours(self):
        """Test that VIX doesn't have pre/post-market sessions."""
        # 6:00am EST (before 9:30am)
        ts = pd.Timestamp("2024-01-02 06:00:00", tz="US/Eastern")
        assert get_session_type(ts, asset="VIX") == SessionType.CLOSED

        # 5:00pm EST (after 4:15pm)
        ts = pd.Timestamp("2024-01-02 17:00:00", tz="US/Eastern")
        assert get_session_type(ts, asset="VIX") == SessionType.CLOSED

    def test_session_type_with_utc_timestamp(self):
        """Test that session type works with UTC timestamps."""
        # 2024-01-02 14:30 UTC = 2024-01-02 09:30 EST
        ts = pd.Timestamp("2024-01-02 14:30:00", tz="UTC")
        assert get_session_type(ts, asset="ES") == SessionType.REGULAR

    def test_session_type_with_naive_timestamp(self):
        """Test that session type works with naive timestamps."""
        ts = pd.Timestamp("2024-01-02 09:30:00")  # naive
        # Should assume US/Eastern and still classify correctly
        assert get_session_type(ts, asset="ES") == SessionType.REGULAR

    def test_invalid_asset_raises_error(self):
        """Test that invalid asset raises ValueError."""
        ts = pd.Timestamp("2024-01-02 09:30:00", tz="US/Eastern")
        with pytest.raises(ValueError):
            get_session_type(ts, asset="INVALID")


# ============================================================================
# Tests: DataFrame Filtering
# ============================================================================


class TestDataFrameFiltering:
    """Tests for filtering DataFrames to trading hours."""

    def test_filter_trading_hours_removes_weekend(self, sample_week_data):
        """Test that filtering removes weekend data."""
        filtered = filter_trading_hours(sample_week_data, asset="ES", include_extended=True)

        # Original has 168 hours (7 days)
        # Should remove: Mon 1/1 (holiday) + Sat 1/6-Sun 1/7 (weekend)
        # That's 24 + 48 = 72 hours
        # Remaining: 168 - 72 = 96 hours
        assert len(filtered) < len(sample_week_data)
        assert len(filtered) > 0

    def test_filter_trading_hours_removes_holiday(self, sample_ohlcv_data):
        """Test that filtering removes holiday data."""
        # Data starts 2024-01-01 (New Year's Day, holiday)
        filtered = filter_trading_hours(sample_ohlcv_data, asset="ES", include_extended=True)

        # Should remove the first 24 hours (Jan 1)
        # Plus some weekend hours depending on the range
        assert len(filtered) < len(sample_ohlcv_data)

    def test_filter_trading_hours_keeps_regular_hours(self, sample_extended_hours_data):
        """Test that filtering keeps regular trading hours."""
        # 2024-01-02 is a Tuesday (trading day)
        # Hours 9:30am-4pm should be kept with include_extended=False
        filtered = filter_trading_hours(
            sample_extended_hours_data,
            asset="ES",
            include_extended=False
        )

        # Hour 0 = midnight (OVERNIGHT for ES)
        # Hours 4-8 = 4am-8am (PRE_MARKET)
        # Hours 9-15 = 9am-3pm (REGULAR for part of range)
        # Hours 16-23 = 4pm-11pm (POST_MARKET and OVERNIGHT)
        # Only 9:30am-4pm is REGULAR = 6.5 hours
        assert len(filtered) > 0
        assert len(filtered) < len(sample_extended_hours_data)

    def test_filter_trading_hours_with_extended_hours(self, sample_extended_hours_data):
        """Test that filtering with include_extended=True keeps extended hours."""
        filtered = filter_trading_hours(
            sample_extended_hours_data,
            asset="ES",
            include_extended=True
        )

        # Should keep most of the day
        assert len(filtered) > len(filter_trading_hours(
            sample_extended_hours_data,
            asset="ES",
            include_extended=False
        ))

    def test_filter_trading_hours_preserves_columns(self, sample_extended_hours_data):
        """Test that filtering preserves all columns."""
        filtered = filter_trading_hours(sample_extended_hours_data, asset="ES")

        # All columns should be present
        assert list(filtered.columns) == list(sample_extended_hours_data.columns)

    def test_filter_trading_hours_naive_timestamps(self, naive_timestamp_data):
        """Test that filtering works with naive timestamps."""
        # Should not raise error, should log warning
        filtered = filter_trading_hours(naive_timestamp_data, asset="ES")
        assert filtered is not None

    def test_filter_trading_hours_invalid_column_raises_error(self, sample_extended_hours_data):
        """Test that invalid column name raises ValueError."""
        with pytest.raises(ValueError):
            filter_trading_hours(sample_extended_hours_data, timestamp_col="invalid_col")

    def test_filter_trading_hours_empty_dataframe(self):
        """Test that filtering an empty DataFrame returns empty."""
        empty_df = pd.DataFrame({
            "timestamp": pd.DatetimeIndex([], tz="US/Eastern"),
            "close": [],
        })
        filtered = filter_trading_hours(empty_df, asset="ES")
        assert len(filtered) == 0

    def test_filter_trading_hours_vix_no_extended(self, sample_extended_hours_data):
        """Test that VIX filtering respects no extended hours."""
        es_filtered = filter_trading_hours(
            sample_extended_hours_data,
            asset="ES",
            include_extended=True
        )
        vix_filtered = filter_trading_hours(
            sample_extended_hours_data,
            asset="VIX",
            include_extended=True
        )

        # VIX should have fewer data points (no pre/post market)
        assert len(vix_filtered) < len(es_filtered)


# ============================================================================
# Tests: Utility Functions
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_supported_assets(self):
        """Test that get_supported_assets returns correct list."""
        assets = get_supported_assets()
        assert "ES" in assets
        assert "MES" in assets
        assert "VIX" in assets
        assert len(assets) == 3

    def test_get_next_trading_day(self):
        """Test getting the next trading day."""
        # 2024-01-01 is a holiday
        next_day = get_next_trading_day(pd.Timestamp("2024-01-01"))
        # Should be 2024-01-02 (Tuesday)
        assert next_day.date().strftime("%Y-%m-%d") == "2024-01-02"

    def test_get_next_trading_day_from_friday(self):
        """Test getting next trading day from Friday."""
        # 2024-01-05 is a Friday
        next_day = get_next_trading_day(pd.Timestamp("2024-01-05"))
        # Should be 2024-01-08 (Monday)
        assert next_day.date().strftime("%Y-%m-%d") == "2024-01-08"

    def test_get_previous_trading_day(self):
        """Test getting the previous trading day."""
        # 2024-01-02 is a Tuesday
        prev_day = get_previous_trading_day(pd.Timestamp("2024-01-02"))
        # Should be 2023-12-29 (Friday)
        assert prev_day < pd.Timestamp("2024-01-02")
        assert is_trading_day(prev_day)

    def test_get_trading_days(self):
        """Test getting all trading days in a range."""
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-08")

        trading_days = get_trading_days(start, end)

        # Should exclude: 2024-01-01 (holiday), 2024-01-06 (Sat), 2024-01-07 (Sun)
        # Included: 2024-01-02, 2024-01-03, 2024-01-04, 2024-01-05, 2024-01-08
        assert len(trading_days) == 5
        assert trading_days[0].date().strftime("%Y-%m-%d") == "2024-01-02"

    def test_get_trading_days_invalid_range_raises_error(self):
        """Test that invalid date range raises ValueError."""
        with pytest.raises(ValueError):
            get_trading_days(
                pd.Timestamp("2024-01-10"),
                pd.Timestamp("2024-01-01")
            )


# ============================================================================
# Tests: Edge Cases and Integration
# ============================================================================


class TestEdgeCasesAndIntegration:
    """Tests for edge cases and integration scenarios."""

    def test_full_week_of_trading_data(self):
        """Test filtering a full week of trading data."""
        # Create hourly data for full week starting Monday
        # 2024-01-01 is Monday (but New Year's holiday)
        # So 2024-01-02 to 2024-01-08
        dates = pd.date_range("2024-01-02", periods=168, freq="1H", tz="US/Eastern")
        df = pd.DataFrame({
            "timestamp": dates,
            "close": np.random.uniform(100, 110, 168),
        })

        # Filter to trading hours only
        filtered = filter_trading_hours(df, asset="ES", include_extended=False)

        # Should keep: 5 trading days * ~6.5 hours of regular market = ~32-33 rows minimum
        # (9:30am-4pm is 6.5 hours)
        assert len(filtered) > 0
        assert len(filtered) < len(df)

    def test_es_and_vix_schedule_differences(self):
        """Test that ES and VIX have different schedules."""
        date = pd.Timestamp("2024-01-02")

        es_hours = get_trading_hours(date, asset="ES")
        vix_hours = get_trading_hours(date, asset="VIX")

        # ES should have 16:00 close, VIX should have 16:15
        assert es_hours["regular_close"] == "16:00"
        assert vix_hours["regular_close"] == "16:15"

        # ES should have globex, VIX should not
        assert "globex_open" in es_hours
        assert "globex_open" not in vix_hours

    def test_consistency_across_assets(self):
        """Test that holidays are consistent across assets."""
        holiday = pd.Timestamp("2024-01-01")

        for asset in ASSETS:
            assert is_trading_day(holiday, asset) is False
            assert get_trading_hours(holiday, asset) is None

    def test_filtering_respects_asset_schedule(self):
        """Test that filtering respects each asset's schedule."""
        dates = pd.date_range("2024-01-02 04:00", periods=24, freq="1H", tz="US/Eastern")
        df = pd.DataFrame({
            "timestamp": dates,
            "close": np.random.uniform(100, 110, 24),
        })

        es_filtered = filter_trading_hours(df, asset="ES", include_extended=True)
        vix_filtered = filter_trading_hours(df, asset="VIX", include_extended=True)

        # ES should have more data (includes pre-market and extended hours)
        # VIX should have fewer (only regular hours)
        assert len(es_filtered) > 0
        assert len(vix_filtered) > 0
        assert len(es_filtered) > len(vix_filtered)
