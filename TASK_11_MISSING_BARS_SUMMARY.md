# Task 11: Missing Bars Detection - Implementation Summary

## Overview
Implemented comprehensive validation logic to detect missing bars in time series data. The validator ensures data completeness by comparing expected bar counts against actual data received, accounting for market hours, holidays, and different trading sessions.

## Key Components Implemented

### 1. Trading Calendar Utilities (`src/data_io/calendar_utils.py`)
- **MarketType Enum**: Defines supported market types (NYSE, CME_GLOBEX, CUSTOM)
- **MarketHours Dataclass**: Defines trading hours with support for extended trading hours
- **TradingCalendar Class**: Core calendar implementation with:
  - `is_trading_day()`: Checks if a date is a trading day
  - `is_half_day()`: Detects early closes
  - `get_trading_hours()`: Retrieves trading hours for specific dates
  - `get_market_open_time()` / `get_market_close_time()`: Specific time queries
  
- **Pre-defined Calendars**:
  - **NYSE Calendar**: US equities with RTH (9:30 AM - 4:00 PM ET)
    - Holidays: New Year's Day, MLK Jr. Day, Presidents' Day, Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas
    - Half-days: Day after Thanksgiving, Christmas Eve (1 PM ET close)
  - **CME Globex Calendar**: Nearly 24-hour futures trading
    - Holidays: New Year's Day, Independence Day, Thanksgiving (reduced hours), Christmas
    - Open: Sunday 5 PM CT through Friday 4 PM CT

- **Helper Functions**:
  - `get_calendar()`: Retrieves pre-defined calendars by market name
  - `create_custom_calendar()`: Creates custom trading calendars

### 2. Missing Bars Detection (`src/data_io/validation.py` additions)

#### GapInfo Dataclass
Structured representation of a gap in the data:
- **Attributes**:
  - `start_index` / `end_index`: Bar indices before and after gap
  - `start_timestamp` / `end_timestamp`: Timestamps of surrounding bars
  - `missing_bars_count`: Number of bars that should exist
  - `gap_duration`: Timedelta of the gap
  - `is_expected`: Boolean flag (weekends, holidays, etc.)
  - `reason`: Human-readable explanation

#### MissingBarsReport Dataclass
Comprehensive validation report:
- **Attributes**:
  - `passed`: Overall validation status
  - `total_bars_expected`: Expected bar count
  - `total_bars_actual`: Actual bar count received
  - `total_bars_missing`: Difference
  - `gaps`: List of GapInfo objects
  - `missing_timestamps`: Set of specific missing timestamps (only for unexpected gaps)
  - `summary`: Dictionary of summary statistics

- **Methods**:
  - `add_gap()`: Register a gap and update pass/fail status
  - `get_expected_gaps()`: Filter gaps that are expected
  - `get_unexpected_gaps()`: Filter gaps that indicate data issues
  - `to_dict()`: Serialize for output
  - `__str__()`: Human-readable summary

#### MissingBarsValidator Class
Core validation engine:
- **Initialization**:
  - `frequency`: Bar frequency ('1min', '5min', '1H', '1d', etc.)
  - `calendar`: Optional TradingCalendar instance
  - `market`: Market type ('NYSE', 'CME') if calendar not provided

- **Key Methods**:
  - `calculate_expected_bars(start_date, end_date)`: 
    - Calculates expected bar count based on frequency and trading calendar
    - Handles daily/weekly frequencies by counting trading days
    - Handles intraday frequencies by iterating through market hours
    - Respects market holidays and early closes
    - Supports overnight sessions (futures markets)
    
  - `detect_missing_bars(df)`:
    - Analyzes DataFrame for missing bars
    - Compares actual vs expected bar counts
    - Identifies specific gaps and missing timestamps
    - Classifies gaps as expected or unexpected
    - Returns detailed MissingBarsReport
    
  - `_is_gap_expected(start_ts, end_ts, freq_delta)`:
    - Determines if gap falls outside trading hours
    - Checks for weekends and holidays
    - Returns (is_expected, reason) tuple

- **Features**:
  - Timezone-aware timestamp handling
  - Support for multiple bar frequencies (1min, 5min, 1H, 1d, 1w)
  - Market holiday awareness
  - Extended trading hours support (optional)
  - Overnight session handling (futures)
  - Efficient gap detection
  - Detailed reporting with missing timestamp collection

## Testing (`tests/test_missing_bars.py`)

### Test Coverage
- **Expected Bars Calculation**: Single day, multiple days, weekends, holidays, various frequencies
- **Gap Detection**: Intraday gaps, weekend gaps, holiday gaps
- **Different Frequencies**: 1-minute, 5-minute, hourly, daily
- **Report Generation**: Data structure serialization, gap filtering
- **Edge Cases**: Empty DataFrames, single bars, invalid frequencies
- **Calendar Integration**: Trading day detection, half-day detection, market hours retrieval
- **CME Futures**: 24-hour trading validation

### Test Fixtures (`tests/fixtures/missing_bars_data.py`)
Pre-built DataFrames with known gaps:
- `create_complete_1min_data()`: Complete 1-minute data (5 trading days)
- `create_data_with_30min_intraday_gap()`: Gap during market hours (unexpected)
- `create_data_with_weekend_gap()`: Friday to Monday gap (expected)
- `create_data_with_holiday_gap()`: Gap spanning New Year's Day (expected)
- `create_data_with_multiple_gaps()`: Mixed expected/unexpected gaps
- `create_5min_data_complete()`: Complete 5-minute data
- `create_hourly_data_complete()`: Complete hourly data
- `create_partial_day_data()`: Day after Thanksgiving (early close)
- `create_cme_24hour_data()`: CME futures spanning 24 hours
- `FIXTURE_DATA` dictionary: Quick access to all fixtures

## Integration with Existing Code

### Updated Exports (`src/data_io/__init__.py`)
- `MissingBarsValidator`
- `MissingBarsReport`
- `GapInfo`
- `TradingCalendar`
- `MarketHours`
- `MarketType`
- `get_calendar()`
- `create_custom_calendar()`

### Validation Pipeline
Missing bars detection complements existing validation:
1. **Schema Validation** (DataValidator): Ensures required columns exist
2. **Timestamp Validation** (DataValidator): Checks ordering and timezone
3. **OHLC Validation** (DataValidator): Verifies price relationships
4. **Missing Bars Detection** (MissingBarsValidator): ✨ NEW - Ensures completeness

## Usage Examples

### Basic Usage
```python
from src.data_io.validation import MissingBarsValidator
from src.data_io.calendar_utils import get_calendar

# Create validator for NYSE 1-minute data
validator = MissingBarsValidator(frequency="1min", market="NYSE")

# Analyze DataFrame
report = validator.detect_missing_bars(df)

# Check results
if not report.passed:
    print(f"Missing {report.total_bars_missing} bars")
    for gap in report.get_unexpected_gaps():
        print(f"  Gap at {gap.start_timestamp}: {gap.missing_bars_count} bars")
```

### Custom Calendar
```python
from src.data_io.calendar_utils import create_custom_calendar, MarketHours
from datetime import time

hours = MarketHours(
    open_time=time(10, 0),
    close_time=time(17, 0),
    timezone="US/Eastern"
)

custom_cal = create_custom_calendar(
    name="Custom Market",
    trading_hours=hours,
    holidays=[pd.Timestamp("2024-12-25")]
)

validator = MissingBarsValidator(frequency="5min", calendar=custom_cal)
```

### Gap Analysis
```python
# Get detailed gap information
for gap in report.gaps:
    print(f"Gap from {gap.start_timestamp} to {gap.end_timestamp}")
    print(f"  Duration: {gap.gap_duration}")
    print(f"  Missing bars: {gap.missing_bars_count}")
    print(f"  Expected: {gap.is_expected} ({gap.reason})")

# Serialize to JSON
import json
report_dict = report.to_dict()
json_str = json.dumps(report_dict, indent=2)
```

## Validation Success Criteria

✅ **Accurate Expected Bars Calculation**
- Correctly calculates expected bars for date range and frequency
- Handles various frequencies (1min, 5min, 1H, 1d, 1w)
- Supports custom date ranges

✅ **Market Calendar Integration**
- NYSE holidays properly excluded
- CME holidays and reduced hours handled
- Weekend gaps classified as expected

✅ **Missing Bars Identification**
- Specific missing timestamps collected
- Gap locations accurately reported
- Both intraday and multi-day gaps detected

✅ **Comprehensive Reporting**
- Expected vs unexpected gaps clearly distinguished
- Summary statistics provided
- Serialization support for persistence

✅ **Efficient Processing**
- Reasonable performance on large datasets (months of 1-minute data)
- Minimal memory overhead

✅ **Multi-Asset Support**
- NYSE equities (regular and extended hours)
- CME futures (24-hour trading)
- Custom calendars for other assets

✅ **Edge Case Handling**
- Empty DataFrames
- Single bar datasets
- Partial trading days
- Overnight sessions (futures)

## Implementation Notes

### Design Decisions

1. **Trading Calendar as Separate Module**
   - Decoupled from validation logic for reusability
   - Can be extended to other asset types
   - Testable independently

2. **Gap Classification**
   - Distinguishes expected (weekends, holidays) from unexpected gaps
   - Allows flagging only data issues as validation failures

3. **Missing Timestamps Collection**
   - Only collected for unexpected gaps to avoid large datasets
   - Useful for gap filling or data reconstruction

4. **Timezone Support**
   - Handles timezone-aware and naive timestamps
   - Properly converts between market timezones

5. **Frequency Parsing**
   - Flexible format support ('1min', '1H', '1day', etc.)
   - Conversion to minutes for internal calculations

### Known Limitations

1. **Fixed Holiday Calendar**
   - 2024 holidays hardcoded for NYSE and CME
   - Can be extended with dynamic calendar library (future enhancement)
   - Custom calendars support dynamic dates

2. **No Macro Factors**
   - Does not account for emergency closures
   - Assumes standard market hours apply

3. **Performance on Large Ranges**
   - Intraday frequency calculations iterate through all intervals
   - Large date ranges (years of 1-minute data) may be slow
   - Could be optimized with mathematical calculations

## Future Enhancements

1. **Dynamic Holiday Calendars**
   - Integrate `pandas_market_calendars` or `pytz` for automatic holiday detection

2. **Advanced Gap Analysis**
   - Pattern detection for systematic gaps
   - Gap prediction based on historical patterns

3. **Data Repair Suggestions**
   - Interpolation recommendations for missing bars
   - Forward-fill or backward-fill suggestions

4. **Performance Optimization**
   - Direct calculation instead of iteration for intraday frequencies
   - Batch processing for multiple assets

5. **Extended Hours Support**
   - Separate validation for pre-market and after-hours data
   - Optional inclusion/exclusion of extended hours

## Files Modified/Created

### Created
- `src/data_io/calendar_utils.py`: Trading calendar utilities
- `tests/test_missing_bars.py`: Comprehensive test suite
- `tests/fixtures/missing_bars_data.py`: Test fixtures
- `tests/fixtures/__init__.py`: Package marker

### Modified
- `src/data_io/validation.py`: Added GapInfo, MissingBarsReport, MissingBarsValidator
- `src/data_io/__init__.py`: Export new classes and functions

## Testing
All tests pass with comprehensive coverage:
- 20+ test methods
- Multiple test fixtures with known gaps
- Edge case coverage
- Multi-market validation
