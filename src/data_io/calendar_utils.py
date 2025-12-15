"""
Trading Calendar Utilities

Provides calendar support for different market types, including trading hours,
holidays, and market sessions. Supports equities (NYSE), futures (CME Globex),
and custom trading calendars.

Features:
- Built-in calendars for NYSE (equities) and CME (futures)
- Holiday detection for US markets
- Trading hours validation
- Market session definitions (RTH, ETH, 24-hour)
- Custom calendar support

Example:
    >>> from src.data_io.calendar_utils import get_calendar
    >>> 
    >>> # Get NYSE calendar for equities
    >>> nyse_cal = get_calendar('NYSE')
    >>> 
    >>> # Check if a date is a trading day
    >>> import pandas as pd
    >>> date = pd.Timestamp('2024-01-15')
    >>> is_trading = nyse_cal.is_trading_day(date)
    >>> 
    >>> # Get trading hours for a date
    >>> hours = nyse_cal.get_trading_hours(date)
    >>> print(hours.open_time, hours.close_time)
"""

import pandas as pd
from datetime import time, datetime, timedelta
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketType(str, Enum):
    """Types of markets supported."""
    NYSE = "NYSE"  # US Equities - Regular Trading Hours
    CME_GLOBEX = "CME_GLOBEX"  # Futures - Nearly 24-hour trading
    CUSTOM = "CUSTOM"  # Custom trading calendar


@dataclass
class MarketHours:
    """Trading hours for a market."""
    open_time: time  # Market open time
    close_time: time  # Market close time
    open_time_eth: Optional[time] = None  # Extended hours open (if applicable)
    close_time_eth: Optional[time] = None  # Extended hours close (if applicable)
    timezone: str = "UTC"  # Timezone for times


class TradingCalendar:
    """
    Trading calendar for market data validation.
    
    Handles trading days, holidays, and market hours for different asset types.
    """
    
    def __init__(
        self,
        name: str,
        market_type: MarketType,
        trading_hours: MarketHours,
        holidays: Optional[List[pd.Timestamp]] = None,
        half_days: Optional[Dict[pd.Timestamp, MarketHours]] = None,
    ):
        """
        Initialize a trading calendar.
        
        Args:
            name: Calendar name (e.g., 'NYSE', 'CME')
            market_type: Type of market
            trading_hours: MarketHours for regular trading
            holidays: List of holiday dates (when market is closed)
            half_days: Dictionary mapping dates to special trading hours (early closes)
        """
        self.name = name
        self.market_type = market_type
        self.trading_hours = trading_hours
        self.holidays = set(pd.Timestamp(h).date() if hasattr(h, 'date') else h.date() 
                           for h in (holidays or []))
        self.half_days = {
            (pd.Timestamp(d).date() if hasattr(d, 'date') else d.date()): hours
            for d, hours in (half_days or {}).items()
        }
    
    def is_trading_day(self, date: pd.Timestamp) -> bool:
        """
        Check if a date is a trading day.
        
        Args:
            date: Date to check
            
        Returns:
            True if market is open on this date
        """
        ts = pd.Timestamp(date)
        date_only = ts.date()
        
        # Check if it's a weekend (Saturday=5, Sunday=6)
        if ts.dayofweek >= 5:
            return False
        
        # Check if it's a holiday
        if date_only in self.holidays:
            return False
        
        return True
    
    def is_half_day(self, date: pd.Timestamp) -> bool:
        """
        Check if a date is a half-day (early close).
        
        Args:
            date: Date to check
            
        Returns:
            True if market has early close on this date
        """
        ts = pd.Timestamp(date)
        date_only = ts.date()
        return date_only in self.half_days
    
    def get_trading_hours(self, date: pd.Timestamp) -> MarketHours:
        """
        Get trading hours for a specific date.
        
        Args:
            date: Date to get hours for
            
        Returns:
            MarketHours for the date (normal or half-day)
        """
        ts = pd.Timestamp(date)
        date_only = ts.date()
        
        # Return half-day hours if applicable
        if date_only in self.half_days:
            return self.half_days[date_only]
        
        return self.trading_hours
    
    def get_market_open_time(self, date: pd.Timestamp) -> Optional[datetime]:
        """
        Get market open time for a date.
        
        Args:
            date: Date to get open time for
            
        Returns:
            datetime of market open, or None if not a trading day
        """
        if not self.is_trading_day(date):
            return None
        
        hours = self.get_trading_hours(date)
        ts = pd.Timestamp(date)
        return datetime.combine(ts.date(), hours.open_time)
    
    def get_market_close_time(self, date: pd.Timestamp) -> Optional[datetime]:
        """
        Get market close time for a date.
        
        Args:
            date: Date to get close time for
            
        Returns:
            datetime of market close, or None if not a trading day
        """
        if not self.is_trading_day(date):
            return None
        
        hours = self.get_trading_hours(date)
        ts = pd.Timestamp(date)
        return datetime.combine(ts.date(), hours.close_time)


# ============================================================================
# Pre-defined Trading Calendars
# ============================================================================

# NYSE Trading Hours (US Equities - Regular Trading Hours)
NYSE_HOURS = MarketHours(
    open_time=time(9, 30),
    close_time=time(16, 0),
    open_time_eth=time(4, 0),  # Pre-market
    close_time_eth=time(20, 0),  # After-hours
    timezone="US/Eastern",
)

# NYSE 2024 Holidays (US market holidays)
NYSE_2024_HOLIDAYS = [
    pd.Timestamp("2024-01-01"),  # New Year's Day
    pd.Timestamp("2024-01-15"),  # MLK Jr. Day
    pd.Timestamp("2024-02-19"),  # Presidents' Day
    pd.Timestamp("2024-03-29"),  # Good Friday
    pd.Timestamp("2024-05-27"),  # Memorial Day
    pd.Timestamp("2024-06-19"),  # Juneteenth
    pd.Timestamp("2024-07-04"),  # Independence Day
    pd.Timestamp("2024-09-02"),  # Labor Day
    pd.Timestamp("2024-11-28"),  # Thanksgiving
    pd.Timestamp("2024-12-25"),  # Christmas
]

# NYSE 2024 Early Closes (half-days at 1 PM ET)
NYSE_2024_HALF_DAYS = {
    pd.Timestamp("2024-11-29"): MarketHours(  # Day after Thanksgiving
        open_time=time(9, 30),
        close_time=time(13, 0),
        timezone="US/Eastern",
    ),
    pd.Timestamp("2024-12-24"): MarketHours(  # Christmas Eve
        open_time=time(9, 30),
        close_time=time(13, 0),
        timezone="US/Eastern",
    ),
}

# CME Globex Trading Hours (Futures - Nearly 24-hour, Sunday 5 PM - Friday 4 PM CT)
CME_GLOBEX_HOURS = MarketHours(
    open_time=time(17, 0),  # 5 PM CT (Sunday for ES/MES)
    close_time=time(16, 0),  # 4 PM CT (Friday, next day)
    timezone="America/Chicago",
)

# CME 2024 Holidays (when Globex is closed or has reduced hours)
CME_2024_HOLIDAYS = [
    pd.Timestamp("2024-01-01"),  # New Year's Day
    pd.Timestamp("2024-07-04"),  # Independence Day
    pd.Timestamp("2024-11-28"),  # Thanksgiving (early close)
    pd.Timestamp("2024-12-25"),  # Christmas
]


def get_calendar(market: str = "NYSE") -> TradingCalendar:
    """
    Get a pre-defined trading calendar.
    
    Args:
        market: Market type ('NYSE', 'CME_GLOBEX', or custom name)
        
    Returns:
        TradingCalendar instance
        
    Raises:
        ValueError: If market type is not recognized
    """
    if market.upper() == "NYSE":
        return TradingCalendar(
            name="NYSE",
            market_type=MarketType.NYSE,
            trading_hours=NYSE_HOURS,
            holidays=NYSE_2024_HOLIDAYS,
            half_days=NYSE_2024_HALF_DAYS,
        )
    elif market.upper() in ("CME", "CME_GLOBEX", "FUTURES", "MES", "ES"):
        return TradingCalendar(
            name="CME Globex",
            market_type=MarketType.CME_GLOBEX,
            trading_hours=CME_GLOBEX_HOURS,
            holidays=CME_2024_HOLIDAYS,
        )
    else:
        raise ValueError(f"Unknown market type: {market}")


def create_custom_calendar(
    name: str,
    trading_hours: MarketHours,
    holidays: Optional[List[pd.Timestamp]] = None,
    half_days: Optional[Dict[pd.Timestamp, MarketHours]] = None,
) -> TradingCalendar:
    """
    Create a custom trading calendar.
    
    Args:
        name: Calendar name
        trading_hours: Regular trading hours
        holidays: List of holiday dates
        half_days: Dictionary of early close dates
        
    Returns:
        TradingCalendar instance
    """
    return TradingCalendar(
        name=name,
        market_type=MarketType.CUSTOM,
        trading_hours=trading_hours,
        holidays=holidays,
        half_days=half_days,
    )
