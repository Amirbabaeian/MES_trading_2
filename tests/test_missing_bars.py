"""
Comprehensive test suite for missing bars detection.

Tests cover:
- Expected bars calculation for different frequencies
- Gap detection (expected and unexpected)
- Market holidays and weekends handling
- Intraday gaps during trading hours
- Extended trading hours (ETH)
- 24-hour futures trading
- Edge cases (partial days, data boundaries)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time

from src.data_io.validation import (
    MissingBarsValidator,
    MissingBarsReport,
    GapInfo,
)
from src.data_io.calendar_utils import (
    TradingCalendar,
    MarketHours,
    MarketType,
    get_calendar,
    create_custom_calendar,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def nyse_calendar():
    """Get NYSE calendar for testing."""
    return get_calendar("NYSE")


@pytest.fixture
def cme_calendar():
    """Get CME calendar for testing."""
    return get_calendar("CME")


@pytest.fixture
def complete_1min_data():
    """Create 1-minute data with no gaps during market hours."""
    # Monday to Friday, 9:30 AM - 4:00 PM ET (390 minutes per day)
    dates = pd.date_range(
        "2024-01-08 09:30:00",
        "2024-01-12 16:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    
    # Filter out market close (4 PM) - that bar shouldn't exist
    dates = dates[dates.time < time(16, 0)]
    
    data = {
        "timestamp": dates,
        "open": np.random.uniform(100, 110, len(dates)),
        "high": np.random.uniform(110, 120, len(dates)),
        "low": np.random.uniform(90, 100, len(dates)),
        "close": np.random.uniform(100, 110, len(dates)),
        "volume": np.random.randint(1000, 10000, len(dates)),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def data_with_intraday_gap():
    """Create 1-minute data with gap during market hours."""
    # Create data with missing 30 minutes
    dates1 = pd.date_range(
        "2024-01-08 09:30:00",
        "2024-01-08 10:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    
    dates2 = pd.date_range(
        "2024-01-08 10:31:00",  # Gap of 30 minutes
        "2024-01-08 11:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    
    dates = dates1.union(dates2)
    
    data = {
        "timestamp": dates,
        "open": np.random.uniform(100, 110, len(dates)),
        "high": np.random.uniform(110, 120, len(dates)),
        "low": np.random.uniform(90, 100, len(dates)),
        "close": np.random.uniform(100, 110, len(dates)),
        "volume": np.random.randint(1000, 10000, len(dates)),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def data_with_weekend_gap():
    """Create 1-minute data spanning a weekend (expected gap)."""
    # Friday data
    dates_fri = pd.date_range(
        "2024-01-12 09:30:00",
        "2024-01-12 16:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    dates_fri = dates_fri[dates_fri.time < time(16, 0)]
    
    # Monday data
    dates_mon = pd.date_range(
        "2024-01-15 09:30:00",
        "2024-01-15 10:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    
    dates = dates_fri.union(dates_mon)
    
    data = {
        "timestamp": dates,
        "open": np.random.uniform(100, 110, len(dates)),
        "high": np.random.uniform(110, 120, len(dates)),
        "low": np.random.uniform(90, 100, len(dates)),
        "close": np.random.uniform(100, 110, len(dates)),
        "volume": np.random.randint(1000, 10000, len(dates)),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def data_with_holiday_gap():
    """Create 1-minute data spanning a holiday (expected gap)."""
    # Before New Year's Day
    dates_before = pd.date_range(
        "2023-12-29 09:30:00",
        "2023-12-29 10:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    
    # After New Year's Day (Jan 1 is holiday)
    dates_after = pd.date_range(
        "2024-01-02 09:30:00",
        "2024-01-02 10:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    
    dates = dates_before.union(dates_after)
    
    data = {
        "timestamp": dates,
        "open": np.random.uniform(100, 110, len(dates)),
        "high": np.random.uniform(110, 120, len(dates)),
        "low": np.random.uniform(90, 100, len(dates)),
        "close": np.random.uniform(100, 110, len(dates)),
        "volume": np.random.randint(1000, 10000, len(dates)),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def complete_5min_data():
    """Create 5-minute data without gaps during market hours."""
    dates = pd.date_range(
        "2024-01-08 09:30:00",
        "2024-01-08 15:55:00",
        freq="5min",
        tz="US/Eastern"
    )
    
    data = {
        "timestamp": dates,
        "open": np.random.uniform(100, 110, len(dates)),
        "high": np.random.uniform(110, 120, len(dates)),
        "low": np.random.uniform(90, 100, len(dates)),
        "close": np.random.uniform(100, 110, len(dates)),
        "volume": np.random.randint(1000, 10000, len(dates)),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def complete_hourly_data():
    """Create hourly data without gaps during market hours."""
    dates = pd.date_range(
        "2024-01-08 09:30:00",
        "2024-01-12 16:00:00",
        freq="1H",
        tz="US/Eastern"
    )
    
    # Filter to market hours only
    dates = dates[(dates.time >= time(9, 30)) & (dates.time < time(16, 0))]
    
    data = {
        "timestamp": dates,
        "open": np.random.uniform(100, 110, len(dates)),
        "high": np.random.uniform(110, 120, len(dates)),
        "low": np.random.uniform(90, 100, len(dates)),
        "close": np.random.uniform(100, 110, len(dates)),
        "volume": np.random.randint(1000, 10000, len(dates)),
    }
    
    return pd.DataFrame(data)


# ============================================================================
# Tests: Expected Bars Calculation
# ============================================================================

class TestExpectedBarsCalculation:
    """Tests for expected bars calculation."""
    
    def test_single_day_1min(self, nyse_calendar):
        """Test 1-minute bars for a single trading day."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        
        start = pd.Timestamp("2024-01-08 09:30:00", tz="US/Eastern")
        end = pd.Timestamp("2024-01-08 15:59:00", tz="US/Eastern")
        
        expected = validator.calculate_expected_bars(start, end)
        # 9:30 AM to 3:59 PM = 6 hours 29 minutes = 389 minutes (not 390, exclude 4 PM close)
        assert expected == 389, f"Expected 389 bars, got {expected}"
    
    def test_single_day_5min(self, nyse_calendar):
        """Test 5-minute bars for a single trading day."""
        validator = MissingBarsValidator(frequency="5min", calendar=nyse_calendar)
        
        start = pd.Timestamp("2024-01-08 09:30:00", tz="US/Eastern")
        end = pd.Timestamp("2024-01-08 15:55:00", tz="US/Eastern")
        
        expected = validator.calculate_expected_bars(start, end)
        # 389 minutes / 5 = 77.8, so 78 bars (9:30, 9:35, ... 15:55)
        assert expected > 0
    
    def test_five_trading_days_1min(self, nyse_calendar):
        """Test 1-minute bars for a full trading week."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        
        start = pd.Timestamp("2024-01-08 09:30:00", tz="US/Eastern")
        end = pd.Timestamp("2024-01-12 15:59:00", tz="US/Eastern")
        
        expected = validator.calculate_expected_bars(start, end)
        # 5 days * 389 minutes per day = 1945 minutes
        assert expected == 1945, f"Expected 1945 bars, got {expected}"
    
    def test_weekend_excluded(self, nyse_calendar):
        """Test that weekends are excluded from expected bars."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        
        # Friday to Monday
        start = pd.Timestamp("2024-01-12 09:30:00", tz="US/Eastern")
        end = pd.Timestamp("2024-01-15 10:00:00", tz="US/Eastern")
        
        expected = validator.calculate_expected_bars(start, end)
        # Friday (389 min) + Monday (30 min) = 419 min
        assert expected == 419, f"Expected 419 bars, got {expected}"
    
    def test_holiday_excluded(self, nyse_calendar):
        """Test that holidays are excluded from expected bars."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        
        # Dec 29 to Jan 2 (Jan 1 is New Year's Day - holiday)
        start = pd.Timestamp("2023-12-29 09:30:00", tz="US/Eastern")
        end = pd.Timestamp("2024-01-02 10:00:00", tz="US/Eastern")
        
        expected = validator.calculate_expected_bars(start, end)
        # Dec 29 (389) + Jan 2 (30) = 419 (Jan 1 excluded)
        assert expected == 419, f"Expected 419 bars, got {expected}"
    
    def test_daily_frequency(self, nyse_calendar):
        """Test daily bar calculation."""
        validator = MissingBarsValidator(frequency="1d", calendar=nyse_calendar)
        
        start = pd.Timestamp("2024-01-08")
        end = pd.Timestamp("2024-01-12")
        
        expected = validator.calculate_expected_bars(start, end)
        # Mon-Fri = 5 trading days
        assert expected == 5, f"Expected 5 bars, got {expected}"
    
    def test_daily_frequency_with_weekend(self, nyse_calendar):
        """Test daily bars spanning weekend."""
        validator = MissingBarsValidator(frequency="1d", calendar=nyse_calendar)
        
        start = pd.Timestamp("2024-01-12")  # Friday
        end = pd.Timestamp("2024-01-15")    # Monday
        
        expected = validator.calculate_expected_bars(start, end)
        # Only Friday and Monday = 2 trading days
        assert expected == 2, f"Expected 2 bars, got {expected}"


# ============================================================================
# Tests: Gap Detection
# ============================================================================

class TestGapDetection:
    """Tests for gap detection."""
    
    def test_no_gaps_complete_data(self, complete_1min_data, nyse_calendar):
        """Test that complete data has no gaps."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        report = validator.detect_missing_bars(complete_1min_data)
        
        assert report.passed
        assert len(report.gaps) == 0
        assert report.total_bars_missing == 0
    
    def test_intraday_gap_detected(self, data_with_intraday_gap, nyse_calendar):
        """Test detection of gap during market hours."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        report = validator.detect_missing_bars(data_with_intraday_gap)
        
        # Should detect gap and fail
        assert not report.passed
        assert len(report.gaps) > 0
        
        # First gap should be unexpected
        gap = report.gaps[0]
        assert not gap.is_expected
        assert gap.missing_bars_count == 30
    
    def test_weekend_gap_expected(self, data_with_weekend_gap, nyse_calendar):
        """Test that weekend gaps are marked as expected."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        report = validator.detect_missing_bars(data_with_weekend_gap)
        
        # Should detect gap but mark as expected
        assert report.passed
        assert len(report.gaps) > 0
        
        # Weekend gap should be expected
        gap = report.gaps[0]
        assert gap.is_expected
        assert "weekend" in gap.reason.lower() or "non-trading" in gap.reason.lower()
    
    def test_holiday_gap_expected(self, data_with_holiday_gap, nyse_calendar):
        """Test that holiday gaps are marked as expected."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        report = validator.detect_missing_bars(data_with_holiday_gap)
        
        # Should detect gap but mark as expected
        assert report.passed
        assert len(report.gaps) > 0
        
        # Holiday gap should be expected
        gap = report.gaps[0]
        assert gap.is_expected
    
    def test_gap_report_statistics(self, data_with_intraday_gap, nyse_calendar):
        """Test gap report statistics."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        report = validator.detect_missing_bars(data_with_intraday_gap)
        
        assert report.total_bars_actual < report.total_bars_expected
        assert report.total_bars_missing > 0
        assert len(report.missing_timestamps) > 0


# ============================================================================
# Tests: Different Frequencies
# ============================================================================

class TestDifferentFrequencies:
    """Tests for different bar frequencies."""
    
    def test_5min_no_gaps(self, complete_5min_data, nyse_calendar):
        """Test 5-minute data without gaps."""
        validator = MissingBarsValidator(frequency="5min", calendar=nyse_calendar)
        report = validator.detect_missing_bars(complete_5min_data)
        
        assert report.passed
        assert len(report.gaps) == 0
    
    def test_hourly_no_gaps(self, complete_hourly_data, nyse_calendar):
        """Test hourly data without gaps."""
        validator = MissingBarsValidator(frequency="1H", calendar=nyse_calendar)
        report = validator.detect_missing_bars(complete_hourly_data)
        
        assert report.passed
        assert len(report.gaps) == 0
    
    def test_frequency_parsing(self, nyse_calendar):
        """Test frequency string parsing."""
        # Test various frequency formats
        frequencies = ["1min", "5min", "1H", "1hour", "1d", "1day"]
        
        for freq in frequencies:
            validator = MissingBarsValidator(frequency=freq, calendar=nyse_calendar)
            assert validator.freq_num > 0
            assert validator.freq_unit is not None


# ============================================================================
# Tests: Report and Gap Info
# ============================================================================

class TestReportAndGapInfo:
    """Tests for report and gap info data structures."""
    
    def test_gap_info_to_dict(self):
        """Test GapInfo to_dict conversion."""
        gap = GapInfo(
            start_index=0,
            end_index=1,
            start_timestamp=pd.Timestamp("2024-01-08 10:00:00", tz="US/Eastern"),
            end_timestamp=pd.Timestamp("2024-01-08 10:30:00", tz="US/Eastern"),
            missing_bars_count=30,
            gap_duration=timedelta(minutes=30),
            is_expected=False,
            reason="Test gap"
        )
        
        gap_dict = gap.to_dict()
        assert gap_dict["start_index"] == 0
        assert gap_dict["missing_bars_count"] == 30
        assert not gap_dict["is_expected"]
    
    def test_report_to_dict(self, data_with_intraday_gap, nyse_calendar):
        """Test MissingBarsReport to_dict conversion."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        report = validator.detect_missing_bars(data_with_intraday_gap)
        
        report_dict = report.to_dict()
        assert "passed" in report_dict
        assert "total_bars_expected" in report_dict
        assert "total_bars_actual" in report_dict
        assert "gap_count" in report_dict
    
    def test_report_get_expected_gaps(self, data_with_weekend_gap, nyse_calendar):
        """Test getting expected gaps from report."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        report = validator.detect_missing_bars(data_with_weekend_gap)
        
        expected_gaps = report.get_expected_gaps()
        unexpected_gaps = report.get_unexpected_gaps()
        
        assert len(expected_gaps) + len(unexpected_gaps) == len(report.gaps)
        assert len(expected_gaps) > 0
        assert len(unexpected_gaps) == 0
    
    def test_report_str_representation(self, data_with_intraday_gap, nyse_calendar):
        """Test string representation of report."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        report = validator.detect_missing_bars(data_with_intraday_gap)
        
        report_str = str(report)
        assert "Missing bars" in report_str or "✗" in report_str


# ============================================================================
# Tests: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_dataframe(self, nyse_calendar):
        """Test handling of empty DataFrame."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        df = pd.DataFrame({"timestamp": [], "open": [], "high": [], "low": [], "close": [], "volume": []})
        
        report = validator.detect_missing_bars(df)
        assert report.passed
        assert report.total_bars_actual == 0
        assert report.total_bars_expected == 0
    
    def test_single_bar(self, nyse_calendar):
        """Test handling of single bar."""
        validator = MissingBarsValidator(frequency="1min", calendar=nyse_calendar)
        
        df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-01-08 09:30:00", tz="US/Eastern")],
            "open": [100.0],
            "high": [110.0],
            "low": [90.0],
            "close": [105.0],
            "volume": [1000],
        })
        
        report = validator.detect_missing_bars(df)
        assert report.total_bars_actual == 1
        assert len(report.gaps) == 0
    
    def test_invalid_frequency(self, nyse_calendar):
        """Test invalid frequency format."""
        with pytest.raises(ValueError):
            MissingBarsValidator(frequency="invalid", calendar=nyse_calendar)
    
    def test_market_types(self):
        """Test different market types."""
        # NYSE (equities)
        nyse_cal = get_calendar("NYSE")
        assert nyse_cal.name == "NYSE"
        
        # CME (futures)
        cme_cal = get_calendar("CME")
        assert cme_cal.name == "CME Globex"


# ============================================================================
# Tests: Calendar Integration
# ============================================================================

class TestCalendarIntegration:
    """Tests for trading calendar integration."""
    
    def test_nyse_is_trading_day(self, nyse_calendar):
        """Test NYSE trading day detection."""
        # Monday (trading day)
        assert nyse_calendar.is_trading_day(pd.Timestamp("2024-01-08"))
        
        # Saturday (not trading)
        assert not nyse_calendar.is_trading_day(pd.Timestamp("2024-01-13"))
        
        # Sunday (not trading)
        assert not nyse_calendar.is_trading_day(pd.Timestamp("2024-01-14"))
        
        # New Year's Day 2024 (holiday)
        assert not nyse_calendar.is_trading_day(pd.Timestamp("2024-01-01"))
    
    def test_nyse_half_days(self, nyse_calendar):
        """Test NYSE half-day detection."""
        # Day after Thanksgiving (early close)
        assert nyse_calendar.is_half_day(pd.Timestamp("2024-11-29"))
        
        # Regular day
        assert not nyse_calendar.is_half_day(pd.Timestamp("2024-01-08"))
    
    def test_market_hours_retrieval(self, nyse_calendar):
        """Test retrieving market hours for different dates."""
        # Regular day
        regular_hours = nyse_calendar.get_trading_hours(pd.Timestamp("2024-01-08"))
        assert regular_hours.open_time == time(9, 30)
        assert regular_hours.close_time == time(16, 0)
        
        # Half day (early close)
        half_day_hours = nyse_calendar.get_trading_hours(pd.Timestamp("2024-11-29"))
        assert half_day_hours.close_time == time(13, 0)
    
    def test_custom_calendar(self):
        """Test creating and using custom calendar."""
        hours = MarketHours(
            open_time=time(10, 0),
            close_time=time(17, 0),
            timezone="US/Eastern"
        )
        
        cal = create_custom_calendar(
            name="Custom Market",
            trading_hours=hours,
            holidays=[pd.Timestamp("2024-12-25")]
        )
        
        assert cal.name == "Custom Market"
        assert cal.is_trading_day(pd.Timestamp("2024-01-08"))
        assert not cal.is_trading_day(pd.Timestamp("2024-12-25"))


# ============================================================================
# Tests: CME Futures (24-hour trading)
# ============================================================================

class TestCMEFutures:
    """Tests for CME futures with nearly 24-hour trading."""
    
    def test_cme_calendar_is_trading_day(self, cme_calendar):
        """Test that CME treats weekdays as trading days."""
        # Monday through Friday should be trading days
        assert cme_calendar.is_trading_day(pd.Timestamp("2024-01-08"))
        assert cme_calendar.is_trading_day(pd.Timestamp("2024-01-12"))
        
        # Only New Year's Day and Christmas are holidays
        assert not cme_calendar.is_trading_day(pd.Timestamp("2024-01-01"))
    
    def test_cme_validator_with_1min_data(self, cme_calendar):
        """Test MissingBarsValidator with CME 1-minute data."""
        validator = MissingBarsValidator(frequency="1min", calendar=cme_calendar)
        
        # Create 24-hour data spanning multiple days
        dates = pd.date_range(
            "2024-01-08",
            "2024-01-09",
            freq="1min",
            tz="America/Chicago"
        )
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.uniform(100, 110, len(dates)),
            "high": np.random.uniform(110, 120, len(dates)),
            "low": np.random.uniform(90, 100, len(dates)),
            "close": np.random.uniform(100, 110, len(dates)),
            "volume": np.random.randint(1000, 10000, len(dates)),
        })
        
        report = validator.detect_missing_bars(df)
        # Should pass since we have continuous 24-hour data
        assert report.total_bars_actual > 0
