"""
Test fixtures for missing bars detection validation.

Provides pre-built DataFrames with known gaps for testing missing bars detection.
Includes scenarios for:
- Complete data (no gaps)
- Intraday gaps during market hours
- Weekend gaps
- Holiday gaps
- Multiple consecutive gaps
- Partial day data
"""

import pandas as pd
import numpy as np
from datetime import time


def create_complete_1min_data(start_date="2024-01-08", num_days=5):
    """
    Create complete 1-minute OHLCV data without gaps.
    
    Args:
        start_date: Start date as string (YYYY-MM-DD)
        num_days: Number of trading days to include
        
    Returns:
        DataFrame with complete 1-minute data
    """
    # NYSE market hours: 9:30 AM - 4:00 PM ET
    dates = pd.date_range(
        f"{start_date} 09:30:00",
        periods=num_days * 390,  # 390 minutes per trading day
        freq="1min",
        tz="US/Eastern"
    )
    
    # Filter out non-trading hours (after 4 PM)
    dates = dates[dates.time < time(16, 0)]
    
    np.random.seed(42)
    data = {
        "timestamp": dates,
        "open": 100 + np.random.uniform(-5, 5, len(dates)),
        "high": 105 + np.random.uniform(0, 10, len(dates)),
        "low": 95 + np.random.uniform(-10, 0, len(dates)),
        "close": 100 + np.random.uniform(-5, 5, len(dates)),
        "volume": np.random.randint(1000, 10000, len(dates)),
    }
    
    df = pd.DataFrame(data)
    return df.sort_values("timestamp").reset_index(drop=True)


def create_data_with_30min_intraday_gap():
    """
    Create 1-minute data with a 30-minute gap during market hours.
    
    Gap occurs at 10:00 AM (missing 10:01-10:30).
    This represents a data delivery issue and should be flagged as unexpected.
    
    Returns:
        DataFrame with intraday gap
    """
    dates1 = pd.date_range(
        "2024-01-08 09:30:00",
        "2024-01-08 10:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    
    dates2 = pd.date_range(
        "2024-01-08 10:31:00",  # Gap starts at 10:01
        "2024-01-08 16:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    
    dates = dates1.union(dates2)
    
    np.random.seed(42)
    data = {
        "timestamp": dates,
        "open": 100 + np.random.uniform(-5, 5, len(dates)),
        "high": 105 + np.random.uniform(0, 10, len(dates)),
        "low": 95 + np.random.uniform(-10, 0, len(dates)),
        "close": 100 + np.random.uniform(-5, 5, len(dates)),
        "volume": np.random.randint(1000, 10000, len(dates)),
    }
    
    return pd.DataFrame(data)


def create_data_with_weekend_gap():
    """
    Create 1-minute data with weekend gap (expected).
    
    Friday 4:00 PM to Monday 9:30 AM gap.
    This is expected and should not be flagged as a data issue.
    
    Returns:
        DataFrame with weekend gap
    """
    dates_fri = pd.date_range(
        "2024-01-12 09:30:00",
        "2024-01-12 16:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    dates_fri = dates_fri[dates_fri.time < time(16, 0)]
    
    dates_mon = pd.date_range(
        "2024-01-15 09:30:00",
        "2024-01-15 10:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    
    dates = dates_fri.union(dates_mon)
    
    np.random.seed(42)
    data = {
        "timestamp": dates,
        "open": 100 + np.random.uniform(-5, 5, len(dates)),
        "high": 105 + np.random.uniform(0, 10, len(dates)),
        "low": 95 + np.random.uniform(-10, 0, len(dates)),
        "close": 100 + np.random.uniform(-5, 5, len(dates)),
        "volume": np.random.randint(1000, 10000, len(dates)),
    }
    
    return pd.DataFrame(data)


def create_data_with_holiday_gap():
    """
    Create 1-minute data with holiday gap (expected).
    
    Gap spans New Year's Day (2024-01-01).
    December 29 to January 2.
    
    Returns:
        DataFrame with holiday gap
    """
    dates_before = pd.date_range(
        "2023-12-29 09:30:00",
        "2023-12-29 10:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    
    dates_after = pd.date_range(
        "2024-01-02 09:30:00",
        "2024-01-02 10:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    
    dates = dates_before.union(dates_after)
    
    np.random.seed(42)
    data = {
        "timestamp": dates,
        "open": 100 + np.random.uniform(-5, 5, len(dates)),
        "high": 105 + np.random.uniform(0, 10, len(dates)),
        "low": 95 + np.random.uniform(-10, 0, len(dates)),
        "close": 100 + np.random.uniform(-5, 5, len(dates)),
        "volume": np.random.randint(1000, 10000, len(dates)),
    }
    
    return pd.DataFrame(data)


def create_data_with_multiple_gaps():
    """
    Create 1-minute data with multiple gaps:
    - One expected (weekend)
    - One unexpected (intraday)
    
    Returns:
        DataFrame with multiple gaps
    """
    # Friday morning
    dates_fri = pd.date_range(
        "2024-01-12 09:30:00",
        "2024-01-12 10:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    
    # Friday with intraday gap
    dates_fri2 = pd.date_range(
        "2024-01-12 11:00:00",
        "2024-01-12 16:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    dates_fri2 = dates_fri2[dates_fri2.time < time(16, 0)]
    
    # Monday
    dates_mon = pd.date_range(
        "2024-01-15 09:30:00",
        "2024-01-15 10:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    
    dates = dates_fri.union(dates_fri2).union(dates_mon)
    
    np.random.seed(42)
    data = {
        "timestamp": dates,
        "open": 100 + np.random.uniform(-5, 5, len(dates)),
        "high": 105 + np.random.uniform(0, 10, len(dates)),
        "low": 95 + np.random.uniform(-10, 0, len(dates)),
        "close": 100 + np.random.uniform(-5, 5, len(dates)),
        "volume": np.random.randint(1000, 10000, len(dates)),
    }
    
    return pd.DataFrame(data)


def create_5min_data_complete(num_days=1):
    """
    Create complete 5-minute OHLCV data.
    
    Args:
        num_days: Number of trading days
        
    Returns:
        DataFrame with complete 5-minute data
    """
    dates = pd.date_range(
        "2024-01-08 09:30:00",
        periods=num_days * 78,  # ~78 bars per day (390/5)
        freq="5min",
        tz="US/Eastern"
    )
    
    # Filter to market hours
    dates = dates[(dates.time >= time(9, 30)) & (dates.time < time(16, 0))]
    
    np.random.seed(42)
    data = {
        "timestamp": dates,
        "open": 100 + np.random.uniform(-5, 5, len(dates)),
        "high": 105 + np.random.uniform(0, 10, len(dates)),
        "low": 95 + np.random.uniform(-10, 0, len(dates)),
        "close": 100 + np.random.uniform(-5, 5, len(dates)),
        "volume": np.random.randint(5000, 50000, len(dates)),
    }
    
    return pd.DataFrame(data)


def create_hourly_data_complete(num_days=5):
    """
    Create complete hourly OHLCV data.
    
    Args:
        num_days: Number of trading days
        
    Returns:
        DataFrame with complete hourly data
    """
    dates = pd.date_range(
        "2024-01-08 09:30:00",
        "2024-01-12 16:00:00",
        freq="1H",
        tz="US/Eastern"
    )
    
    # Filter to market hours
    dates = dates[(dates.time >= time(9, 30)) & (dates.time < time(16, 0))]
    
    np.random.seed(42)
    data = {
        "timestamp": dates,
        "open": 100 + np.random.uniform(-5, 5, len(dates)),
        "high": 105 + np.random.uniform(0, 10, len(dates)),
        "low": 95 + np.random.uniform(-10, 0, len(dates)),
        "close": 100 + np.random.uniform(-5, 5, len(dates)),
        "volume": np.random.randint(100000, 1000000, len(dates)),
    }
    
    return pd.DataFrame(data)


def create_partial_day_data():
    """
    Create data for a partial trading day (early market close).
    
    Returns:
        DataFrame with partial day data (half-day)
    """
    # Day after Thanksgiving (early close at 1 PM ET)
    dates = pd.date_range(
        "2024-11-29 09:30:00",
        "2024-11-29 13:00:00",
        freq="1min",
        tz="US/Eastern"
    )
    dates = dates[dates.time < time(13, 1)]
    
    np.random.seed(42)
    data = {
        "timestamp": dates,
        "open": 100 + np.random.uniform(-5, 5, len(dates)),
        "high": 105 + np.random.uniform(0, 10, len(dates)),
        "low": 95 + np.random.uniform(-10, 0, len(dates)),
        "close": 100 + np.random.uniform(-5, 5, len(dates)),
        "volume": np.random.randint(1000, 10000, len(dates)),
    }
    
    return pd.DataFrame(data)


def create_cme_24hour_data():
    """
    Create 1-minute data for CME futures (nearly 24-hour trading).
    
    Spans 2 calendar days (Monday-Tuesday) with no gaps.
    
    Returns:
        DataFrame with 24-hour CME data
    """
    # Monday 5 PM CT to Tuesday 5 PM CT (24 hours)
    dates = pd.date_range(
        "2024-01-08 17:00:00",
        "2024-01-09 17:00:00",
        freq="1min",
        tz="America/Chicago"
    )
    
    np.random.seed(42)
    data = {
        "timestamp": dates,
        "open": 5000 + np.random.uniform(-50, 50, len(dates)),
        "high": 5025 + np.random.uniform(0, 100, len(dates)),
        "low": 4975 + np.random.uniform(-100, 0, len(dates)),
        "close": 5000 + np.random.uniform(-50, 50, len(dates)),
        "volume": np.random.randint(1000, 10000, len(dates)),
    }
    
    return pd.DataFrame(data)


# Dictionary of all fixture data for easy access
FIXTURE_DATA = {
    "complete_1min": create_complete_1min_data(),
    "intraday_gap_30min": create_data_with_30min_intraday_gap(),
    "weekend_gap": create_data_with_weekend_gap(),
    "holiday_gap": create_data_with_holiday_gap(),
    "multiple_gaps": create_data_with_multiple_gaps(),
    "complete_5min": create_5min_data_complete(),
    "complete_hourly": create_hourly_data_complete(),
    "partial_day": create_partial_day_data(),
    "cme_24hour": create_cme_24hour_data(),
}
