"""
Trading calendar utilities for identifying valid trading days and hours.

Provides functions to filter OHLCV data to valid trading periods for ES/MES
(S&P 500 E-mini futures) and VIX futures based on CME Globex and CBOE schedules.

Key Functions:
- is_trading_day: Check if a date is a valid trading day
- get_trading_hours: Get trading session hours for a date
- get_session_type: Identify the session type (regular/extended/closed)
- filter_trading_hours: Remove data outside trading hours

CME Globex Schedule (ES/MES):
- Sunday 6pm ET to Friday 5pm ET (continuous market)
- Regular Trading Hours (RTH): Monday-Friday 9:30am-4pm ET
- Extended Hours: Pre-market and post-market sessions
- Overnight: Sunday-Thursday evening sessions

CBOE Schedule (VIX):
- Monday-Friday 9:30am-4:15pm ET (note: 4:15pm not 4:00pm)
- Regular market hours only

Holiday Handling:
- Full trading halts: New Year's, Good Friday, July 4th, Thanksgiving, Christmas
- Early closes: Day before Thanksgiving (2pm ET), Day before Christmas (1:30pm ET)

Example:
    >>> import pandas as pd
    >>> from src.data.cleaning.trading_calendar import (
    ...     is_trading_day, get_session_type, filter_trading_hours
    ... )
    >>> 
    >>> # Check if a date is a trading day
    >>> is_trading_day(pd.Timestamp('2024-01-01'), asset='ES')
    False  # New Year's Day
    >>> 
    >>> # Identify session type for a timestamp
    >>> ts = pd.Timestamp('2024-01-02 09:30:00', tz='US/Eastern')
    >>> get_session_type(ts, asset='ES')
    'REGULAR'
    >>> 
    >>> # Filter DataFrame to trading hours only
    >>> df_filtered = filter_trading_hours(df, asset='ES', include_extended=False)
"""

import json
import logging
import os
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import pytz
import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Constants and Enumerations
# ============================================================================

NY_TZ = pytz.timezone("US/Eastern")
UTC_TZ = pytz.UTC

ASSETS = {"ES", "MES", "VIX"}  # Supported assets
DEFAULT_ASSET = "ES"


class SessionType(Enum):
    """Enum for different session types."""
    REGULAR = "REGULAR"
    PRE_MARKET = "PRE_MARKET"
    POST_MARKET = "POST_MARKET"
    OVERNIGHT = "OVERNIGHT"
    CLOSED = "CLOSED"


class TradingSchedule:
    """Definition of trading schedule for an asset."""

    def __init__(
        self,
        asset: str,
        market_open: time,
        market_close: time,
        globex_open: Optional[time] = None,
        globex_close: Optional[time] = None,
        premarket_start: Optional[time] = None,
        postmarket_end: Optional[time] = None,
        support_extended_hours: bool = False,
    ):
        """
        Initialize a trading schedule.

        Args:
            asset: Asset name (ES, MES, VIX)
            market_open: Regular market open time (ET)
            market_close: Regular market close time (ET)
            globex_open: Globex open time for 24-hour markets (ET)
            globex_close: Globex close time for 24-hour markets (ET)
            premarket_start: Pre-market start time (ET)
            postmarket_end: Post-market end time (ET)
            support_extended_hours: Whether this asset trades extended hours
        """
        self.asset = asset
        self.market_open = market_open
        self.market_close = market_close
        self.globex_open = globex_open
        self.globex_close = globex_close
        self.premarket_start = premarket_start
        self.postmarket_end = postmarket_end
        self.support_extended_hours = support_extended_hours


# ============================================================================
# Schedule Definitions
# ============================================================================

# ES/MES Schedule: CME Globex 6pm Sunday - 5pm Friday
ES_SCHEDULE = TradingSchedule(
    asset="ES",
    market_open=time(9, 30),  # 9:30am ET
    market_close=time(16, 0),  # 4:00pm ET
    globex_open=time(18, 0),  # 6:00pm ET (previous day)
    globex_close=time(17, 0),  # 5:00pm ET (next day)
    premarket_start=time(4, 0),  # 4:00am ET
    postmarket_end=time(20, 0),  # 8:00pm ET
    support_extended_hours=True,
)

MES_SCHEDULE = TradingSchedule(
    asset="MES",
    market_open=time(9, 30),
    market_close=time(16, 0),
    globex_open=time(18, 0),
    globex_close=time(17, 0),
    premarket_start=time(4, 0),
    postmarket_end=time(20, 0),
    support_extended_hours=True,
)

# VIX Schedule: CBOE hours (9:30am-4:15pm ET, no extended hours)
VIX_SCHEDULE = TradingSchedule(
    asset="VIX",
    market_open=time(9, 30),  # 9:30am ET
    market_close=time(16, 15),  # 4:15pm ET (note: VIX closes at 4:15pm, not 4:00pm)
    support_extended_hours=False,
)

SCHEDULES = {
    "ES": ES_SCHEDULE,
    "MES": MES_SCHEDULE,
    "VIX": VIX_SCHEDULE,
}


# ============================================================================
# Holiday and Special Date Handling
# ============================================================================

def _load_holidays() -> Dict:
    """
    Load CME holidays from configuration file.

    Returns:
        Dictionary with 'full_trading_halts' and 'early_closes' lists
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config",
        "cme_holidays.json",
    )

    if not os.path.exists(config_path):
        logger.warning(f"CME holidays config not found at {config_path}")
        return {"full_trading_halts": [], "early_closes": []}

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading CME holidays: {e}")
        return {"full_trading_halts": [], "early_closes": []}


def _build_holiday_set() -> Tuple[set, Dict]:
    """
    Build sets of full trading halt dates and early close dates.

    Returns:
        Tuple of (full_halt_dates_set, early_close_dict)
        - full_halt_dates_set: Set of date strings (YYYY-MM-DD) with full trading halts
        - early_close_dict: Dict mapping date strings to early close times
    """
    holidays_config = _load_holidays()
    full_halts = set()
    early_closes = {}

    # Process full trading halts
    for halt in holidays_config.get("full_trading_halts", []):
        if "date" in halt:
            # Specific date
            full_halts.add(halt["date"])
        elif "month" in halt and "day" in halt:
            # Recurring date - add for all specified years
            for year in halt.get("years", []):
                date_str = f"{year:04d}-{halt['month']:02d}-{halt['day']:02d}"
                full_halts.add(date_str)

    # Process early closes
    for early in holidays_config.get("early_closes", []):
        if "date" in early and "close_time_et" in early:
            early_closes[early["date"]] = early["close_time_et"]

    return full_halts, early_closes


# Initialize holiday/early close caches
_FULL_HALT_DATES, _EARLY_CLOSE_TIMES = _build_holiday_set()


# ============================================================================
# Core Functions
# ============================================================================

def is_trading_day(date: pd.Timestamp, asset: str = DEFAULT_ASSET) -> bool:
    """
    Check if a date is a valid trading day for the given asset.

    A trading day is any weekday that is not a full trading halt (holiday).

    Args:
        date: Date to check (can be timezone-aware or naive; date part is used)
        asset: Asset name ('ES', 'MES', or 'VIX'). Default is 'ES'

    Returns:
        True if the date is a valid trading day, False otherwise

    Raises:
        ValueError: If asset is not supported

    Example:
        >>> import pandas as pd
        >>> is_trading_day(pd.Timestamp('2024-01-01'))  # New Year's
        False
        >>> is_trading_day(pd.Timestamp('2024-01-02'))  # Regular Tuesday
        True
        >>> is_trading_day(pd.Timestamp('2024-01-06'))  # Saturday
        False
    """
    if asset not in ASSETS:
        raise ValueError(f"Unsupported asset: {asset}. Use one of {ASSETS}")

    # Convert to date if it's a Timestamp
    if isinstance(date, pd.Timestamp):
        date_obj = date.date() if hasattr(date, 'date') else date
    else:
        date_obj = pd.Timestamp(date).date()

    # Check if weekday (Monday=0, Sunday=6)
    if date_obj.weekday() >= 5:  # Saturday or Sunday
        return False

    # Check if it's a full trading halt
    date_str = date_obj.strftime("%Y-%m-%d")
    if date_str in _FULL_HALT_DATES:
        return False

    return True


def get_trading_hours(
    date: pd.Timestamp, asset: str = DEFAULT_ASSET
) -> Optional[Dict[str, str]]:
    """
    Get trading hours for a given date and asset.

    Returns the regular market hours and, for CME Globex assets, the
    overnight (Globex) hours.

    Args:
        date: Date to get hours for (can be timezone-aware or naive)
        asset: Asset name ('ES', 'MES', or 'VIX'). Default is 'ES'

    Returns:
        Dictionary with trading hours or None if not a trading day:
        {
            'date': 'YYYY-MM-DD',
            'regular_open': 'HH:MM',
            'regular_close': 'HH:MM',
            'globex_open': 'HH:MM' (optional, only for ES/MES),
            'globex_close': 'HH:MM' (optional, only for ES/MES),
            'is_early_close': bool,
            'early_close_time': 'HH:MM' (optional)
        }

    Raises:
        ValueError: If asset is not supported

    Example:
        >>> import pandas as pd
        >>> hours = get_trading_hours(pd.Timestamp('2024-01-02'), asset='ES')
        >>> print(hours)
        {
            'date': '2024-01-02',
            'regular_open': '09:30',
            'regular_close': '16:00',
            'globex_open': '18:00',
            'globex_close': '17:00',
            'is_early_close': False
        }
    """
    if asset not in ASSETS:
        raise ValueError(f"Unsupported asset: {asset}. Use one of {ASSETS}")

    # Check if it's a trading day
    if not is_trading_day(date, asset):
        return None

    # Get date string
    if isinstance(date, pd.Timestamp):
        date_obj = date.date() if hasattr(date, 'date') else date
    else:
        date_obj = pd.Timestamp(date).date()

    date_str = date_obj.strftime("%Y-%m-%d")
    schedule = SCHEDULES[asset]

    # Build base hours
    hours = {
        "date": date_str,
        "regular_open": schedule.market_open.strftime("%H:%M"),
        "regular_close": schedule.market_close.strftime("%H:%M"),
        "is_early_close": False,
    }

    # Add Globex hours if applicable
    if schedule.globex_open and schedule.globex_close:
        hours["globex_open"] = schedule.globex_open.strftime("%H:%M")
        hours["globex_close"] = schedule.globex_close.strftime("%H:%M")

    # Check for early close
    if date_str in _EARLY_CLOSE_TIMES:
        hours["is_early_close"] = True
        hours["early_close_time"] = _EARLY_CLOSE_TIMES[date_str]
        # Update regular_close to early close time
        hours["regular_close"] = _EARLY_CLOSE_TIMES[date_str]

    return hours


def get_session_type(
    timestamp: pd.Timestamp, asset: str = DEFAULT_ASSET
) -> SessionType:
    """
    Identify the session type for a given timestamp.

    Classifies timestamps into:
    - REGULAR: Regular trading hours (9:30am-4pm ET for most, 4:15pm for VIX)
    - PRE_MARKET: Before regular hours (4am-9:30am ET)
    - POST_MARKET: After regular hours (4pm-8pm ET or later)
    - OVERNIGHT: Overnight session (6pm-4am ET, CME Globex only)
    - CLOSED: Weekends, holidays, or outside all trading hours

    Args:
        timestamp: Timestamp to classify (must be tz-aware or will assume US/Eastern)
        asset: Asset name ('ES', 'MES', or 'VIX'). Default is 'ES'

    Returns:
        SessionType enum value

    Raises:
        ValueError: If asset is not supported

    Example:
        >>> import pandas as pd
        >>> ts = pd.Timestamp('2024-01-02 09:30:00', tz='US/Eastern')
        >>> get_session_type(ts, asset='ES')
        <SessionType.REGULAR: 'REGULAR'>
        
        >>> ts_after_hours = pd.Timestamp('2024-01-02 17:00:00', tz='US/Eastern')
        >>> get_session_type(ts_after_hours, asset='ES')
        <SessionType.POST_MARKET: 'POST_MARKET'>
    """
    if asset not in ASSETS:
        raise ValueError(f"Unsupported asset: {asset}. Use one of {ASSETS}")

    # Ensure timestamp is timezone-aware
    if timestamp.tz is None:
        timestamp = timestamp.replace(tzinfo=NY_TZ)
    else:
        timestamp = timestamp.tz_convert(NY_TZ)

    # Get the date and time
    date_obj = timestamp.date()
    time_obj = timestamp.time()

    # Check if it's a trading day
    if not is_trading_day(date_obj, asset):
        return SessionType.CLOSED

    schedule = SCHEDULES[asset]

    # Regular trading hours
    if schedule.market_open <= time_obj < schedule.market_close:
        return SessionType.REGULAR

    # Check early close time if applicable
    date_str = date_obj.strftime("%Y-%m-%d")
    if date_str in _EARLY_CLOSE_TIMES:
        early_close_str = _EARLY_CLOSE_TIMES[date_str]
        early_close_time = datetime.strptime(early_close_str, "%H:%M").time()
        if schedule.market_open <= time_obj < early_close_time:
            return SessionType.REGULAR

    # Extended hours for assets that support it
    if schedule.support_extended_hours:
        # Pre-market
        if schedule.premarket_start and schedule.premarket_start <= time_obj < schedule.market_open:
            return SessionType.PRE_MARKET

        # Post-market
        if schedule.postmarket_end and schedule.market_close <= time_obj < schedule.postmarket_end:
            return SessionType.POST_MARKET

        # Overnight (after post-market or before pre-market on next day)
        if time_obj >= schedule.postmarket_end or time_obj < schedule.premarket_start:
            return SessionType.OVERNIGHT

    # Default to closed
    return SessionType.CLOSED


def filter_trading_hours(
    df: pd.DataFrame,
    asset: str = DEFAULT_ASSET,
    include_extended: bool = True,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Filter DataFrame to remove data outside trading hours.

    Removes rows that occur during non-trading periods (weekends, holidays,
    or outside trading hours). For assets with extended hours, can optionally
    exclude pre-market and post-market sessions.

    Args:
        df: DataFrame with timestamp column
        asset: Asset name ('ES', 'MES', or 'VIX'). Default is 'ES'
        include_extended: Include pre-market and post-market data (default: True).
                         Only applies to assets with extended hours (ES/MES).
        timestamp_col: Name of timestamp column (default: 'timestamp')

    Returns:
        Filtered DataFrame containing only trading hours

    Raises:
        ValueError: If asset is not supported or timestamp column not found
        TypeError: If timestamp column is not datetime type

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H', tz='US/Eastern'),
        ...     'close': [100.0] * 100,
        ... })
        >>> df_trading = filter_trading_hours(df, asset='ES', include_extended=False)
        >>> len(df_trading) < len(df)
        True
    """
    if asset not in ASSETS:
        raise ValueError(f"Unsupported asset: {asset}. Use one of {ASSETS}")

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")

    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        raise TypeError(f"Column '{timestamp_col}' is not datetime type")

    if df.empty:
        return df.copy()

    # Ensure timestamps are in NY timezone
    timestamps = df[timestamp_col]
    if timestamps.dt.tz is None:
        logger.warning(
            f"Naive timestamps detected in {timestamp_col}. "
            "Assuming US/Eastern timezone for filtering."
        )
        timestamps = timestamps.dt.tz_localize(NY_TZ)
    else:
        timestamps = timestamps.dt.tz_convert(NY_TZ)

    # Vectorized session type detection
    session_types = timestamps.apply(
        lambda ts: get_session_type(ts, asset=asset)
    )

    # Filter based on include_extended flag
    if include_extended:
        # Keep REGULAR, PRE_MARKET, POST_MARKET, and OVERNIGHT for supported assets
        valid_sessions = {SessionType.REGULAR, SessionType.PRE_MARKET,
                         SessionType.POST_MARKET, SessionType.OVERNIGHT}
    else:
        # Keep only REGULAR sessions
        valid_sessions = {SessionType.REGULAR}

    # Apply filter
    mask = session_types.isin(valid_sessions)
    return df[mask].copy()


# ============================================================================
# Utility Functions
# ============================================================================

def get_supported_assets() -> List[str]:
    """
    Get list of supported assets.

    Returns:
        List of supported asset names
    """
    return sorted(list(ASSETS))


def get_next_trading_day(
    date: pd.Timestamp, asset: str = DEFAULT_ASSET
) -> pd.Timestamp:
    """
    Get the next trading day after the given date.

    Args:
        date: Starting date
        asset: Asset name ('ES', 'MES', or 'VIX'). Default is 'ES'

    Returns:
        First trading day after the given date

    Raises:
        ValueError: If asset is not supported
    """
    if asset not in ASSETS:
        raise ValueError(f"Unsupported asset: {asset}. Use one of {ASSETS}")

    next_date = pd.Timestamp(date) + timedelta(days=1)
    while not is_trading_day(next_date, asset):
        next_date += timedelta(days=1)
    return next_date


def get_previous_trading_day(
    date: pd.Timestamp, asset: str = DEFAULT_ASSET
) -> pd.Timestamp:
    """
    Get the previous trading day before the given date.

    Args:
        date: Starting date
        asset: Asset name ('ES', 'MES', or 'VIX'). Default is 'ES'

    Returns:
        Last trading day before the given date

    Raises:
        ValueError: If asset is not supported
    """
    if asset not in ASSETS:
        raise ValueError(f"Unsupported asset: {asset}. Use one of {ASSETS}")

    prev_date = pd.Timestamp(date) - timedelta(days=1)
    while not is_trading_day(prev_date, asset):
        prev_date -= timedelta(days=1)
    return prev_date


def get_trading_days(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    asset: str = DEFAULT_ASSET,
) -> List[pd.Timestamp]:
    """
    Get all trading days between two dates (inclusive).

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        asset: Asset name ('ES', 'MES', or 'VIX'). Default is 'ES'

    Returns:
        List of trading days as pandas Timestamps

    Raises:
        ValueError: If asset is not supported or if start_date > end_date
    """
    if asset not in ASSETS:
        raise ValueError(f"Unsupported asset: {asset}. Use one of {ASSETS}")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError(f"start_date ({start}) cannot be after end_date ({end})")

    trading_days = []
    current = start

    while current <= end:
        if is_trading_day(current, asset):
            trading_days.append(current)
        current += timedelta(days=1)

    return trading_days


__all__ = [
    "is_trading_day",
    "get_trading_hours",
    "get_session_type",
    "filter_trading_hours",
    "get_supported_assets",
    "get_next_trading_day",
    "get_previous_trading_day",
    "get_trading_days",
    "SessionType",
    "TradingSchedule",
    "ASSETS",
    "SCHEDULES",
]
