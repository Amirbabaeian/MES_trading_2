"""
Data validation module for OHLCV market data.

Provides comprehensive validation logic to ensure data schema consistency
and correctness across all ingested market data. Validation runs after
data ingestion and before promotion to the cleaned data layer.

Features:
- Schema validation (columns, data types)
- Timestamp validation (ordering, timezone, format)
- OHLC relationship validation
- Structured violation reporting
- Batch validation support
- Configurable floating-point tolerance
"""

import pandas as pd
import numpy as np
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import logging
from datetime import datetime, timedelta, time
from .calendar_utils import TradingCalendar, get_calendar

logger = logging.getLogger(__name__)


class TimezoneType(str, Enum):
    """Supported timezone types for timestamp validation."""
    UTC = "UTC"
    US_EASTERN = "US/Eastern"
    NAIVE = "naive"


@dataclass
class Violation:
    """
    Represents a single data validation violation.
    
    Attributes:
        violation_type: Type of violation (schema, timestamp, ohlc, etc.)
        bar_index: Index of the bar with the violation
        timestamp: Timestamp of the bar (if available)
        column: Column name involved in violation (if applicable)
        message: Detailed violation message
    """
    violation_type: str
    bar_index: int
    timestamp: Optional[Any] = None
    column: Optional[str] = None
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary."""
        return {
            "violation_type": self.violation_type,
            "bar_index": self.bar_index,
            "timestamp": str(self.timestamp) if self.timestamp is not None else None,
            "column": self.column,
            "message": self.message,
        }


@dataclass
class ValidationResult:
    """
    Structured validation result with pass/fail status and details.
    
    Attributes:
        passed: Whether validation passed overall
        total_bars_checked: Total number of bars validated
        violations: List of Violation objects
        summary: Summary statistics and messages
    """
    passed: bool
    total_bars_checked: int
    violations: List[Violation] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_violation(self, violation: Violation) -> None:
        """Add a violation to the result."""
        self.violations.append(violation)
        self.passed = False
    
    def get_violation_count(self) -> int:
        """Get total number of violations."""
        return len(self.violations)
    
    def get_violations_by_type(self) -> Dict[str, int]:
        """Get count of violations by type."""
        counts = {}
        for violation in self.violations:
            counts[violation.violation_type] = counts.get(violation.violation_type, 0) + 1
        return counts
    
    def get_violations_by_column(self) -> Dict[str, int]:
        """Get count of violations by column."""
        counts = {}
        for violation in self.violations:
            if violation.column:
                counts[violation.column] = counts.get(violation.column, 0) + 1
        return counts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "passed": self.passed,
            "total_bars_checked": self.total_bars_checked,
            "total_violations": self.get_violation_count(),
            "violations_by_type": self.get_violations_by_type(),
            "violations_by_column": self.get_violations_by_column(),
            "summary": self.summary,
            "violations": [v.to_dict() for v in self.violations],
        }
    
    def __str__(self) -> str:
        """String representation of validation result."""
        if self.passed:
            return f"✓ Validation passed ({self.total_bars_checked} bars)"
        else:
            msg = f"✗ Validation failed ({self.get_violation_count()} violations, "
            msg += f"{self.total_bars_checked} bars)"
            if self.get_violations_by_type():
                msg += f"\n  By type: {self.get_violations_by_type()}"
            return msg


class DataValidator:
    """
    Comprehensive data validator for OHLCV market data.
    
    Validates schema consistency, timestamp ordering, timezone,
    and OHLC relationship constraints.
    """
    
    # Required OHLCV columns
    REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
    
    # Expected data types (numpy dtype or pandas type strings)
    EXPECTED_DTYPES = {
        "timestamp": ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[ns, US/Eastern]"],
        "open": ["float64", "float32"],
        "high": ["float64", "float32"],
        "low": ["float64", "float32"],
        "close": ["float64", "float32"],
        "volume": ["int64", "int32", "float64"],  # Allow float for some data sources
    }
    
    def __init__(
        self,
        float_tolerance: float = 1e-9,
        timezone: str = "US/Eastern",
        allow_duplicate_timestamps: bool = False,
    ):
        """
        Initialize the DataValidator.
        
        Args:
            float_tolerance: Tolerance for floating-point comparisons
            timezone: Expected timezone for timestamps (or "naive" for no timezone)
            allow_duplicate_timestamps: Whether to allow duplicate timestamps
        """
        self.float_tolerance = float_tolerance
        self.timezone = timezone
        self.allow_duplicate_timestamps = allow_duplicate_timestamps
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a DataFrame against schema, timestamp, and OHLC constraints.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            ValidationResult with pass/fail status and detailed violations
        """
        result = ValidationResult(
            passed=True,
            total_bars_checked=len(df),
        )
        
        # Handle empty DataFrame
        if len(df) == 0:
            result.summary["message"] = "Empty DataFrame"
            return result
        
        # Validate schema
        self._validate_schema(df, result)
        
        # If schema validation failed, return early
        if not result.passed:
            result.summary["validation_stage"] = "schema"
            return result
        
        # Validate timestamps
        self._validate_timestamps(df, result)
        
        # Validate OHLC relationships
        self._validate_ohlc_relationships(df, result)
        
        # Add summary statistics
        if result.passed:
            result.summary["message"] = f"All validations passed for {len(df)} bars"
        else:
            result.summary["message"] = f"Validation completed with {len(result.violations)} violations"
        
        result.summary["validation_stages"] = ["schema", "timestamp", "ohlc"]
        
        return result
    
    def _validate_schema(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate DataFrame schema (columns and data types)."""
        # Check for required columns
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            for col in missing_cols:
                result.add_violation(
                    Violation(
                        violation_type="schema",
                        bar_index=0,
                        column=col,
                        message=f"Missing required column: {col}",
                    )
                )
            return
        
        # Check data types
        for col, expected_types in self.EXPECTED_DTYPES.items():
            actual_type = str(df[col].dtype)
            
            # Check if actual type matches one of expected types
            if not any(exp_type in actual_type for exp_type in expected_types):
                result.add_violation(
                    Violation(
                        violation_type="schema",
                        bar_index=0,
                        column=col,
                        message=f"Column '{col}': expected type {expected_types}, got {actual_type}",
                    )
                )
    
    def _validate_timestamps(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate timestamp ordering, timezone, and format."""
        timestamps = df["timestamp"]
        
        # Check for null values
        null_count = timestamps.isna().sum()
        if null_count > 0:
            result.add_violation(
                Violation(
                    violation_type="timestamp",
                    bar_index=0,
                    column="timestamp",
                    message=f"Found {null_count} null timestamps",
                )
            )
            return
        
        # Check timezone consistency
        self._check_timezone_consistency(df, result)
        
        # Check timestamp ordering (monotonically increasing)
        self._check_timestamp_ordering(df, result)
        
        # Check for duplicate timestamps
        self._check_duplicate_timestamps(df, result)
    
    def _check_timezone_consistency(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that all timestamps have consistent timezone."""
        timestamps = df["timestamp"]
        
        if self.timezone == "naive":
            # Check that timestamps are naive (no timezone)
            if hasattr(timestamps.dt, "tz") and timestamps.dt.tz is not None:
                result.add_violation(
                    Violation(
                        violation_type="timestamp",
                        bar_index=0,
                        column="timestamp",
                        message=f"Expected naive timestamps, but found timezone-aware",
                    )
                )
        else:
            # Check that timestamps are timezone-aware
            if not hasattr(timestamps.dt, "tz") or timestamps.dt.tz is None:
                result.add_violation(
                    Violation(
                        violation_type="timestamp",
                        bar_index=0,
                        column="timestamp",
                        message=f"Expected timezone-aware timestamps, but found naive",
                    )
                )
            else:
                # Check specific timezone
                tz_str = str(timestamps.dt.tz)
                if self.timezone not in tz_str:
                    result.add_violation(
                        Violation(
                            violation_type="timestamp",
                            bar_index=0,
                            column="timestamp",
                            message=f"Expected timezone '{self.timezone}', got '{tz_str}'",
                        )
                    )
    
    def _check_timestamp_ordering(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that timestamps are monotonically increasing."""
        timestamps = df["timestamp"].values
        
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i - 1]:
                result.add_violation(
                    Violation(
                        violation_type="timestamp",
                        bar_index=i,
                        timestamp=timestamps[i],
                        column="timestamp",
                        message=f"Timestamp not monotonically increasing: "
                                f"{timestamps[i-1]} >= {timestamps[i]}",
                    )
                )
    
    def _check_duplicate_timestamps(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for duplicate timestamps."""
        if self.allow_duplicate_timestamps:
            return
        
        timestamps = df["timestamp"]
        duplicates = timestamps[timestamps.duplicated(keep=False)]
        
        if len(duplicates) > 0:
            # Find indices of duplicates
            dup_indices = duplicates.index.tolist()
            for idx in dup_indices:
                result.add_violation(
                    Violation(
                        violation_type="timestamp",
                        bar_index=int(idx),
                        timestamp=df.loc[idx, "timestamp"],
                        column="timestamp",
                        message=f"Duplicate timestamp: {df.loc[idx, 'timestamp']}",
                    )
                )
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate OHLC relationship constraints."""
        # Check for null/zero values
        self._check_null_ohlc_values(df, result)
        
        # Check for negative prices
        self._check_positive_prices(df, result)
        
        # Check OHLC relationships: high >= max(open, close)
        self._check_high_relationship(df, result)
        
        # Check OHLC relationships: low <= min(open, close)
        self._check_low_relationship(df, result)
    
    def _check_null_ohlc_values(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check for null or zero values in OHLC columns."""
        for col in ["open", "high", "low", "close", "volume"]:
            null_mask = df[col].isna()
            zero_mask = df[col] == 0
            
            # Check nulls
            null_indices = df[null_mask].index.tolist()
            for idx in null_indices:
                result.add_violation(
                    Violation(
                        violation_type="ohlc",
                        bar_index=int(idx),
                        timestamp=df.loc[idx, "timestamp"],
                        column=col,
                        message=f"Null value in {col}",
                    )
                )
            
            # Check zeros (but volume can be zero)
            if col != "volume":
                zero_indices = df[zero_mask].index.tolist()
                for idx in zero_indices:
                    result.add_violation(
                        Violation(
                            violation_type="ohlc",
                            bar_index=int(idx),
                            timestamp=df.loc[idx, "timestamp"],
                            column=col,
                            message=f"Zero value in {col}",
                        )
                    )
    
    def _check_positive_prices(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that all OHLC prices are positive."""
        for col in ["open", "high", "low", "close"]:
            negative_mask = df[col] < 0
            negative_indices = df[negative_mask].index.tolist()
            
            for idx in negative_indices:
                result.add_violation(
                    Violation(
                        violation_type="ohlc",
                        bar_index=int(idx),
                        timestamp=df.loc[idx, "timestamp"],
                        column=col,
                        message=f"Negative price in {col}: {df.loc[idx, col]}",
                    )
                )
    
    def _check_high_relationship(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that high >= max(open, close)."""
        max_oc = np.maximum(df["open"], df["close"])
        violations = df["high"] < (max_oc - self.float_tolerance)
        
        violation_indices = df[violations].index.tolist()
        for idx in violation_indices:
            open_val = df.loc[idx, "open"]
            close_val = df.loc[idx, "close"]
            high_val = df.loc[idx, "high"]
            
            result.add_violation(
                Violation(
                    violation_type="ohlc",
                    bar_index=int(idx),
                    timestamp=df.loc[idx, "timestamp"],
                    column="high",
                    message=f"High ({high_val}) < max(open={open_val}, close={close_val})",
                )
            )
    
    def _check_low_relationship(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Check that low <= min(open, close)."""
        min_oc = np.minimum(df["open"], df["close"])
        violations = df["low"] > (min_oc + self.float_tolerance)
        
        violation_indices = df[violations].index.tolist()
        for idx in violation_indices:
            open_val = df.loc[idx, "open"]
            close_val = df.loc[idx, "close"]
            low_val = df.loc[idx, "low"]
            
            result.add_violation(
                Violation(
                    violation_type="ohlc",
                    bar_index=int(idx),
                    timestamp=df.loc[idx, "timestamp"],
                    column="low",
                    message=f"Low ({low_val}) > min(open={open_val}, close={close_val})",
                )
            )
    
    def validate_batch(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, ValidationResult]:
        """
        Validate multiple DataFrames in batch.
        
        Args:
            dfs: Dictionary mapping data identifiers to DataFrames
        
        Returns:
            Dictionary mapping identifiers to ValidationResult objects
        """
        results = {}
        for identifier, df in dfs.items():
            logger.info(f"Validating {identifier}...")
            results[identifier] = self.validate(df)
            if results[identifier].passed:
                logger.info(f"  ✓ {identifier} passed validation")
            else:
                logger.warning(
                    f"  ✗ {identifier} failed with {len(results[identifier].violations)} violations"
                )
        return results


# ============================================================================
# Missing Bars Detection
# ============================================================================

@dataclass
class GapInfo:
    """
    Information about a gap (missing bars) in the data.
    
    Attributes:
        start_index: Index of last bar before gap
        end_index: Index of first bar after gap
        start_timestamp: Timestamp of last bar before gap
        end_timestamp: Timestamp of first bar after gap
        missing_bars_count: Number of bars that should have been present
        gap_duration: Timedelta representing the gap duration
        is_expected: Whether the gap is expected (weekend, holiday, market closure)
        reason: Reason for gap (if expected)
    """
    start_index: int
    end_index: int
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    missing_bars_count: int
    gap_duration: timedelta
    is_expected: bool = False
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert gap info to dictionary."""
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_timestamp": str(self.start_timestamp),
            "end_timestamp": str(self.end_timestamp),
            "missing_bars_count": self.missing_bars_count,
            "gap_duration": str(self.gap_duration),
            "is_expected": self.is_expected,
            "reason": self.reason,
        }


@dataclass
class MissingBarsReport:
    """
    Comprehensive report of missing bars in time series data.
    
    Attributes:
        passed: Whether validation passed (no unexpected gaps)
        total_bars_expected: Expected number of bars for the date range
        total_bars_actual: Actual number of bars in the dataset
        total_bars_missing: Total bars missing
        bars_missing_count: Count of bars missing
        gaps: List of GapInfo objects for each gap
        missing_timestamps: Set of timestamps that should exist but are missing
        summary: Summary statistics and messages
    """
    passed: bool
    total_bars_expected: int
    total_bars_actual: int
    total_bars_missing: int
    gaps: List[GapInfo] = field(default_factory=list)
    missing_timestamps: Set[pd.Timestamp] = field(default_factory=set)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_gap(self, gap: GapInfo) -> None:
        """Add a gap to the report."""
        self.gaps.append(gap)
        if not gap.is_expected:
            self.passed = False
    
    def get_expected_gaps(self) -> List[GapInfo]:
        """Get list of expected gaps (weekends, holidays, etc.)."""
        return [gap for gap in self.gaps if gap.is_expected]
    
    def get_unexpected_gaps(self) -> List[GapInfo]:
        """Get list of unexpected gaps (potential data issues)."""
        return [gap for gap in self.gaps if not gap.is_expected]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "passed": self.passed,
            "total_bars_expected": self.total_bars_expected,
            "total_bars_actual": self.total_bars_actual,
            "total_bars_missing": self.total_bars_missing,
            "gap_count": len(self.gaps),
            "expected_gaps": len(self.get_expected_gaps()),
            "unexpected_gaps": len(self.get_unexpected_gaps()),
            "summary": self.summary,
            "gaps": [gap.to_dict() for gap in self.gaps],
            "missing_timestamps_count": len(self.missing_timestamps),
        }
    
    def __str__(self) -> str:
        """String representation of the report."""
        if self.passed:
            msg = f"✓ All bars present ({self.total_bars_actual}/{self.total_bars_expected})"
        else:
            msg = f"✗ Missing bars detected ({self.total_bars_actual}/{self.total_bars_expected})"
            msg += f"\n  Unexpected gaps: {len(self.get_unexpected_gaps())}"
            if self.get_unexpected_gaps():
                for gap in self.get_unexpected_gaps()[:3]:  # Show first 3
                    msg += f"\n    Gap at {gap.start_timestamp}: {gap.missing_bars_count} bars missing"
        return msg


class MissingBarsValidator:
    """
    Validator for detecting missing bars in time series data.
    
    Compares expected bar counts (based on market hours and frequency) against
    actual data received. Handles market holidays, trading hours, and different
    market sessions (RTH, ETH, 24-hour).
    """
    
    def __init__(
        self,
        frequency: str = "1min",
        calendar: Optional[TradingCalendar] = None,
        market: str = "NYSE",
    ):
        """
        Initialize the missing bars validator.
        
        Args:
            frequency: Bar frequency (e.g., '1min', '5min', '1H', '1D')
            calendar: TradingCalendar instance (uses default for market if None)
            market: Market type ('NYSE', 'CME', etc.) used if calendar is None
        """
        self.frequency = frequency
        self.calendar = calendar or get_calendar(market)
        self._parse_frequency()
    
    def _parse_frequency(self) -> None:
        """Parse frequency string into components."""
        freq_str = self.frequency.lower().strip()
        
        # Extract number and unit
        match = re.match(r'(\d+)([a-z]+)', freq_str)
        
        if not match:
            raise ValueError(f"Invalid frequency format: {self.frequency}")
        
        self.freq_num = int(match.group(1))
        self.freq_unit = match.group(2)
        
        # Validate frequency unit
        valid_units = ['min', 'h', 'hour', 'd', 'day', 'w', 'week']
        if self.freq_unit not in valid_units:
            raise ValueError(f"Unknown frequency unit: {self.freq_unit}")
    
    def _get_frequency_minutes(self) -> int:
        """Get frequency in minutes."""
        if self.freq_unit == 'min':
            return self.freq_num
        elif self.freq_unit in ('h', 'hour'):
            return self.freq_num * 60
        elif self.freq_unit in ('d', 'day'):
            return self.freq_num * 24 * 60
        elif self.freq_unit in ('w', 'week'):
            return self.freq_num * 7 * 24 * 60
        else:
            raise ValueError(f"Unknown frequency unit: {self.freq_unit}")
    
    def calculate_expected_bars(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> int:
        """
        Calculate expected number of bars for a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            Expected number of bars
        """
        freq_minutes = self._get_frequency_minutes()
        expected_bars = 0
        
        # Handle daily and weekly frequencies
        if self.freq_unit in ('d', 'day'):
            current_date = pd.Timestamp(start_date).normalize()
            end_date_norm = pd.Timestamp(end_date).normalize()
            
            while current_date <= end_date_norm:
                if self.calendar.is_trading_day(current_date):
                    expected_bars += 1
                current_date += timedelta(days=self.freq_num)
            
            return expected_bars
        
        # For intraday frequencies (1min, 5min, 1H, etc.)
        current_time = pd.Timestamp(start_date)
        end_time = pd.Timestamp(end_date)
        
        while current_time <= end_time:
            date = current_time.normalize()
            
            # Check if this date is a trading day
            if self.calendar.is_trading_day(date):
                hours = self.calendar.get_trading_hours(date)
                
                # Create market open and close times for this date
                market_open = datetime.combine(date.date(), hours.open_time)
                market_close = datetime.combine(date.date(), hours.close_time)
                
                # Handle overnight sessions (when close_time is "earlier" than open_time)
                # This happens with futures that trade across midnight
                if hours.close_time < hours.open_time:
                    # Session spans midnight
                    market_close = datetime.combine(
                        (date + timedelta(days=1)).date(),
                        hours.close_time
                    )
                
                # Make sure current_time is within market hours for this date
                if current_time > market_close:
                    # Move to next trading day
                    next_date = date + timedelta(days=1)
                    current_time = datetime.combine(next_date.date(), hours.open_time)
                    if hours.close_time < hours.open_time:
                        # This won't work for overnight sessions, skip forward
                        current_time = next_date.normalize()
                    continue
                
                # If before market open, move to open time
                if current_time < market_open:
                    current_time = pd.Timestamp(market_open, tz=current_time.tz)
                
                # Count bars within market hours
                while current_time <= market_close and current_time <= end_time:
                    expected_bars += 1
                    current_time += timedelta(minutes=freq_minutes)
            else:
                # Move to next day
                current_time = (date + timedelta(days=1)).normalize()
        
        return expected_bars
    
    def detect_missing_bars(self, df: pd.DataFrame) -> MissingBarsReport:
        """
        Detect missing bars in a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data (must have 'timestamp' column)
            
        Returns:
            MissingBarsReport with missing bars details
        """
        if len(df) == 0:
            return MissingBarsReport(
                passed=True,
                total_bars_expected=0,
                total_bars_actual=0,
                total_bars_missing=0,
                summary={"message": "Empty DataFrame"}
            )
        
        # Get timestamps
        timestamps = pd.to_datetime(df['timestamp'])
        start_ts = timestamps.iloc[0]
        end_ts = timestamps.iloc[-1]
        
        # Calculate expected bars
        expected_count = self.calculate_expected_bars(start_ts, end_ts)
        actual_count = len(df)
        missing_count = expected_count - actual_count
        
        report = MissingBarsReport(
            passed=(missing_count == 0),
            total_bars_expected=expected_count,
            total_bars_actual=actual_count,
            total_bars_missing=missing_count,
        )
        
        # Detect gaps
        freq_minutes = self._get_frequency_minutes()
        freq_delta = timedelta(minutes=freq_minutes)
        
        for i in range(len(timestamps) - 1):
            current_ts = timestamps.iloc[i]
            next_ts = timestamps.iloc[i + 1]
            time_delta = next_ts - current_ts
            
            # Check if there's a gap
            if time_delta > freq_delta:
                # Calculate missing bars
                missing_bars = int(time_delta.total_seconds() / (freq_minutes * 60)) - 1
                
                # Determine if gap is expected
                is_expected, reason = self._is_gap_expected(
                    current_ts,
                    next_ts,
                    freq_delta
                )
                
                gap = GapInfo(
                    start_index=i,
                    end_index=i + 1,
                    start_timestamp=current_ts,
                    end_timestamp=next_ts,
                    missing_bars_count=missing_bars,
                    gap_duration=time_delta,
                    is_expected=is_expected,
                    reason=reason,
                )
                
                report.add_gap(gap)
                
                # Add missing timestamps
                if not is_expected:
                    current = current_ts + freq_delta
                    while current < next_ts:
                        report.missing_timestamps.add(current)
                        current += freq_delta
        
        # Add summary statistics
        report.summary = {
            "start_date": str(start_ts),
            "end_date": str(end_ts),
            "frequency": self.frequency,
            "expected_vs_actual": f"{expected_count} expected, {actual_count} actual",
            "total_missing": missing_count,
            "expected_gap_count": len(report.get_expected_gaps()),
            "unexpected_gap_count": len(report.get_unexpected_gaps()),
        }
        
        return report
    
    def _is_gap_expected(
        self,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        freq_delta: timedelta,
    ) -> Tuple[bool, str]:
        """
        Determine if a gap between two timestamps is expected.
        
        Args:
            start_ts: Start timestamp
            end_ts: End timestamp
            freq_delta: Expected frequency interval
            
        Returns:
            Tuple of (is_expected, reason)
        """
        current = start_ts + freq_delta
        
        while current < end_ts:
            # Check if current timestamp is a trading time
            date = current.normalize()
            
            if not self.calendar.is_trading_day(date):
                # Non-trading day (weekend/holiday) - gap is expected
                current += freq_delta
                continue
            
            # Check if within market hours
            hours = self.calendar.get_trading_hours(date)
            market_open = datetime.combine(date.date(), hours.open_time)
            market_close = datetime.combine(date.date(), hours.close_time)
            
            # Handle overnight sessions
            if hours.close_time < hours.open_time:
                market_close = datetime.combine(
                    (date + timedelta(days=1)).date(),
                    hours.close_time
                )
            
            market_open = pd.Timestamp(market_open, tz=current.tz)
            market_close = pd.Timestamp(market_close, tz=current.tz)
            
            # If timestamp is outside market hours, it's expected
            if current < market_open or current > market_close:
                current += freq_delta
                continue
            
            # If we get here, there's a bar that should exist during market hours
            # This is an unexpected gap
            return False, f"Missing bar during market hours at {current}"
        
        # All missing timestamps are outside market hours or non-trading days
        return True, "Gap spans weekends/holidays/non-trading hours"


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_ohlcv_data(
    df: pd.DataFrame,
    float_tolerance: float = 1e-9,
    timezone: str = "US/Eastern",
    allow_duplicate_timestamps: bool = False,
) -> ValidationResult:
    """
    Validate OHLCV data with default settings.
    
    Args:
        df: DataFrame to validate
        float_tolerance: Tolerance for floating-point comparisons
        timezone: Expected timezone for timestamps
        allow_duplicate_timestamps: Whether to allow duplicate timestamps
    
    Returns:
        ValidationResult with validation details
    """
    validator = DataValidator(
        float_tolerance=float_tolerance,
        timezone=timezone,
        allow_duplicate_timestamps=allow_duplicate_timestamps,
    )
    return validator.validate(df)


def validate_multiple_datasets(
    datasets: Dict[str, pd.DataFrame],
    float_tolerance: float = 1e-9,
    timezone: str = "US/Eastern",
) -> Dict[str, ValidationResult]:
    """
    Validate multiple datasets in batch.
    
    Args:
        datasets: Dictionary mapping identifiers to DataFrames
        float_tolerance: Tolerance for floating-point comparisons
        timezone: Expected timezone for timestamps
    
    Returns:
        Dictionary mapping identifiers to ValidationResult objects
    """
    validator = DataValidator(
        float_tolerance=float_tolerance,
        timezone=timezone,
    )
    return validator.validate_batch(datasets)
