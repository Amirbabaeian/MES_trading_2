"""
Contract roll detection and adjustment for futures contracts.

This module provides functionality to:
- Detect contract roll points in futures data
- Apply backward ratio adjustment to maintain price continuity
- Support alternative panama canal method (concatenate without adjustment)
- Track adjustment metadata for audit trail

Key Concepts:
- Roll Detection: Identifies contract expiration dates and front month switches
- Backward Ratio Adjustment: Multiplies historical prices by roll ratio
  (new_contract_close / old_contract_close) to maintain price continuity
- Panama Canal Method: Concatenates contracts without adjustment (alternative)
- Contract Metadata: Quarterly expirations (ES/MES) or monthly (VIX)

Price Continuity:
- Backward adjustment ensures no artificial gaps at roll boundaries
- Historical prices are adjusted; recent prices remain unchanged
- Adjustment factors tracked for research reproducibility

Example:
    >>> import pandas as pd
    >>> from src.data.cleaning.contract_rolls import (
    ...     detect_contract_rolls, backward_ratio_adjustment
    ... )
    >>> 
    >>> # Load ES futures data with multiple contracts
    >>> df = pd.read_parquet('es_data.parquet')
    >>> 
    >>> # Detect roll points
    >>> rolls = detect_contract_rolls(df, 'ES')
    >>> print(rolls)
    # [{'roll_date': '2024-03-15', 'old_contract': 'ESH24', 'new_contract': 'ESM24', ...}]
    >>> 
    >>> # Apply backward ratio adjustment
    >>> df_adjusted = backward_ratio_adjustment(df, rolls)
    >>> 
    >>> # Verify price continuity
    >>> print(df_adjusted['close'].iloc[-10:])
    # No artificial gaps at roll boundaries
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# Constants and Data Classes
# ============================================================================

# Load contract specifications
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
CONTRACT_SPECS_FILE = CONFIG_DIR / "contract_specs.json"

try:
    with open(CONTRACT_SPECS_FILE, 'r') as f:
        CONTRACT_SPECS = json.load(f)
except FileNotFoundError:
    logger.error(f"Contract specs file not found at {CONTRACT_SPECS_FILE}")
    CONTRACT_SPECS = {}

SUPPORTED_ASSETS = set(CONTRACT_SPECS.keys())
DEFAULT_ASSET = "ES"

# Adjustment method types
class AdjustmentMethod(Enum):
    """Enumeration of available adjustment methods."""
    BACKWARD_RATIO = "backward_ratio"
    PANAMA_CANAL = "panama_canal"


@dataclass
class RollPoint:
    """
    Represents a contract roll event.
    
    Attributes:
        roll_date: Date when the roll occurred
        old_contract: Previous contract symbol
        new_contract: New contract symbol
        old_contract_close: Closing price of old contract on roll date
        new_contract_close: Closing price of new contract on roll date
        roll_ratio: Adjustment ratio (new_close / old_close)
        first_notice_date: First notice date for the old contract
        last_trading_date: Last trading date for the old contract
    """
    roll_date: pd.Timestamp
    old_contract: str
    new_contract: str
    old_contract_close: float
    new_contract_close: float
    roll_ratio: float
    first_notice_date: Optional[pd.Timestamp] = None
    last_trading_date: Optional[pd.Timestamp] = None
    
    def to_dict(self) -> Dict:
        """Convert RollPoint to dictionary."""
        return {
            'roll_date': str(self.roll_date.date()) if isinstance(self.roll_date, pd.Timestamp) else str(self.roll_date),
            'old_contract': self.old_contract,
            'new_contract': self.new_contract,
            'old_contract_close': float(self.old_contract_close),
            'new_contract_close': float(self.new_contract_close),
            'roll_ratio': float(self.roll_ratio),
            'first_notice_date': str(self.first_notice_date.date()) if self.first_notice_date else None,
            'last_trading_date': str(self.last_trading_date.date()) if self.last_trading_date else None,
        }


# ============================================================================
# Contract Metadata Functions
# ============================================================================

def get_contract_spec(asset: str) -> Dict:
    """
    Get contract specifications for an asset.
    
    Args:
        asset: Asset name ('ES', 'MES', 'VIX')
        
    Returns:
        Dictionary containing contract specifications
        
    Raises:
        ValueError: If asset is not supported
    """
    if asset not in SUPPORTED_ASSETS:
        raise ValueError(f"Unsupported asset: {asset}. Supported: {SUPPORTED_ASSETS}")
    
    return CONTRACT_SPECS[asset]


def get_expiration_months(asset: str) -> List[int]:
    """
    Get expiration months for an asset.
    
    Args:
        asset: Asset name ('ES', 'MES', 'VIX')
        
    Returns:
        List of expiration month numbers (1-12)
        
    Raises:
        ValueError: If asset is not supported
    """
    spec = get_contract_spec(asset)
    return spec.get('expiration_months', [])


def get_contract_symbol(asset: str, year: int, month: int) -> str:
    """
    Generate contract symbol for an asset, year, and month.
    
    Convention: ASSET + MONTH_CODE + YEAR_DIGIT
    Month codes: F(Jan), G(Feb), H(Mar), J(Apr), K(May), M(Jun),
                 N(Jul), Q(Aug), U(Sep), V(Oct), X(Nov), Z(Dec)
    
    Args:
        asset: Asset name ('ES', 'MES', 'VIX')
        year: 4-digit year
        month: Month number (1-12)
        
    Returns:
        Contract symbol (e.g., 'ESH24' for ES March 2024)
        
    Raises:
        ValueError: If month is invalid
    """
    month_codes = {
        1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
        7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
    }
    
    if month not in month_codes:
        raise ValueError(f"Invalid month: {month}. Must be 1-12")
    
    year_digit = str(year)[-1]  # Last digit of year
    return f"{asset}{month_codes[month]}{year_digit}"


def parse_contract_symbol(symbol: str) -> Tuple[str, int, int]:
    """
    Parse a contract symbol into asset, year, and month.
    
    Args:
        symbol: Contract symbol (e.g., 'ESH24')
        
    Returns:
        Tuple of (asset, year, month)
        
    Raises:
        ValueError: If symbol format is invalid
    """
    month_codes = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }
    
    if len(symbol) < 4:
        raise ValueError(f"Invalid symbol format: {symbol}")
    
    # Extract parts
    month_code = symbol[-3]
    year_digit = symbol[-1]
    asset = symbol[:-3]
    
    if month_code not in month_codes:
        raise ValueError(f"Invalid month code in symbol: {symbol}")
    
    # Assume 20xx for 2-digit year codes (21st century)
    year = 2000 + int(year_digit)
    month = month_codes[month_code]
    
    return asset, year, month


def calculate_expiration_date(year: int, month: int, asset: str = "ES") -> pd.Timestamp:
    """
    Calculate the expiration date for a contract.
    
    For ES/MES: Last day of the expiration month at 16:00 ET
    For VIX: 30 calendar days before third Friday of following month
    
    Args:
        year: 4-digit year
        month: Month number (1-12)
        asset: Asset name ('ES', 'MES', 'VIX')
        
    Returns:
        Expiration date as pandas Timestamp (US/Eastern)
        
    Raises:
        ValueError: If asset is not supported
    """
    spec = get_contract_spec(asset)
    
    if asset in ['ES', 'MES']:
        # ES/MES: Third Friday of the expiration month
        # Find third Friday
        first_day = pd.Timestamp(year=year, month=month, day=1)
        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7  # Friday is 4
        if days_until_friday == 0:
            first_friday = first_day
        else:
            first_friday = first_day + timedelta(days=days_until_friday)
        # Third Friday is 14 days after first Friday
        third_friday = first_friday + timedelta(days=14)
        return third_friday.normalize()
    
    elif asset == "VIX":
        # VIX: 30 calendar days before third Friday of following month
        next_month = month + 1 if month < 12 else 1
        next_year = year if month < 12 else year + 1
        
        first_day = pd.Timestamp(year=next_year, month=next_month, day=1)
        days_until_friday = (4 - first_day.weekday()) % 7
        if days_until_friday == 0:
            first_friday = first_day
        else:
            first_friday = first_day + timedelta(days=days_until_friday)
        third_friday = first_friday + timedelta(days=14)
        
        # 30 calendar days before
        expiration = third_friday - timedelta(days=30)
        return expiration.normalize()
    
    else:
        raise ValueError(f"Unknown asset: {asset}")


# ============================================================================
# Roll Detection Functions
# ============================================================================

def detect_contract_rolls(
    df: pd.DataFrame,
    asset: str = DEFAULT_ASSET,
    contract_col: str = "contract",
    timestamp_col: str = "timestamp",
    close_col: str = "close"
) -> List[RollPoint]:
    """
    Detect contract roll points in a DataFrame.
    
    Identifies where the contract column changes, indicating a roll event.
    Calculates roll ratios and metadata for each detected roll.
    
    Args:
        df: DataFrame with OHLCV data
        asset: Asset name ('ES', 'MES', 'VIX')
        contract_col: Name of contract column
        timestamp_col: Name of timestamp column
        close_col: Name of close price column
        
    Returns:
        List of RollPoint objects representing detected rolls
        
    Raises:
        ValueError: If asset is not supported
        KeyError: If required columns are missing
    """
    if asset not in SUPPORTED_ASSETS:
        raise ValueError(f"Unsupported asset: {asset}")
    
    # Validate required columns
    required_cols = [contract_col, timestamp_col, close_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns: {missing_cols}")
    
    if df.empty:
        return []
    
    rolls = []
    contracts = df[contract_col]
    timestamps = df[timestamp_col]
    closes = df[close_col]
    
    # Find where contracts change
    contract_changes = contracts != contracts.shift()
    change_indices = contract_changes[contract_changes].index
    
    for idx in change_indices[1:]:  # Skip first change (it's comparison to NaN)
        prev_idx = idx - 1
        
        # Get old and new contract info
        old_contract = contracts.iloc[prev_idx]
        new_contract = contracts.iloc[idx]
        
        roll_date = timestamps.iloc[idx]
        old_close = closes.iloc[prev_idx]
        new_close = closes.iloc[idx]
        
        # Calculate roll ratio
        if old_close == 0:
            logger.warning(f"Old contract close is 0 at {roll_date}. Skipping ratio calculation.")
            roll_ratio = 1.0
        else:
            roll_ratio = new_close / old_close
        
        # Calculate expiration dates
        try:
            asset_parsed, year, month = parse_contract_symbol(old_contract)
            expiration = calculate_expiration_date(year, month, asset_parsed)
        except (ValueError, IndexError):
            logger.warning(f"Could not parse contract symbol: {old_contract}")
            expiration = None
        
        roll_point = RollPoint(
            roll_date=pd.Timestamp(roll_date),
            old_contract=str(old_contract),
            new_contract=str(new_contract),
            old_contract_close=float(old_close),
            new_contract_close=float(new_close),
            roll_ratio=float(roll_ratio),
            last_trading_date=expiration
        )
        
        rolls.append(roll_point)
        logger.info(
            f"Detected roll: {old_contract} -> {new_contract} at {roll_date} "
            f"(ratio: {roll_ratio:.6f})"
        )
    
    return rolls


def identify_active_contract(
    df: pd.DataFrame,
    asset: str = DEFAULT_ASSET,
    contract_col: str = "contract"
) -> str:
    """
    Identify the active (most recent) contract in a DataFrame.
    
    Args:
        df: DataFrame with contract data
        asset: Asset name ('ES', 'MES', 'VIX')
        contract_col: Name of contract column
        
    Returns:
        Active contract symbol
        
    Raises:
        ValueError: If DataFrame is empty or contract column missing
    """
    if df.empty:
        raise ValueError("Cannot identify active contract in empty DataFrame")
    
    if contract_col not in df.columns:
        raise KeyError(f"Contract column '{contract_col}' not found")
    
    return df[contract_col].iloc[-1]


# ============================================================================
# Adjustment Methods
# ============================================================================

def backward_ratio_adjustment(
    df: pd.DataFrame,
    roll_points: List[RollPoint],
    timestamp_col: str = "timestamp",
    ohlc_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply backward ratio adjustment to maintain price continuity across rolls.
    
    Method: Multiply all historical prices (before roll) by the roll ratio
    to match the new contract's price level. This is the standard approach
    for continuous price series.
    
    Formula: adjusted_price = historical_price * roll_ratio
    where: roll_ratio = new_contract_close / old_contract_close
    
    Key Properties:
    - Recent prices unchanged (most recent contract not adjusted)
    - Historical prices adjusted backwards
    - No artificial gaps at roll boundaries
    - Preserves price relationships and patterns
    
    Args:
        df: DataFrame with OHLCV data
        roll_points: List of RollPoint objects (from detect_contract_rolls)
        timestamp_col: Name of timestamp column
        ohlc_cols: OHLC column names. Default: ['open', 'high', 'low', 'close']
        
    Returns:
        Tuple of:
        - Adjusted DataFrame with continuous prices
        - Metadata dictionary with adjustment factors and statistics
        
    Raises:
        ValueError: If timestamp column missing
        KeyError: If OHLC columns missing
    """
    if ohlc_cols is None:
        ohlc_cols = ['open', 'high', 'low', 'close']
    
    # Validate columns
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found")
    
    missing_cols = [col for col in ohlc_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing OHLC columns: {missing_cols}")
    
    result_df = df.copy()
    metadata = {
        'method': 'backward_ratio',
        'total_rolls': len(roll_points),
        'adjustments': []
    }
    
    if not roll_points:
        logger.info("No roll points provided. Returning original data.")
        return result_df, metadata
    
    # Sort roll points by date
    sorted_rolls = sorted(roll_points, key=lambda r: r.roll_date)
    
    for roll in sorted_rolls:
        # Find rows before the roll date
        mask = result_df[timestamp_col] < roll.roll_date
        
        if not mask.any():
            logger.warning(f"No data before roll at {roll.roll_date}")
            continue
        
        # Apply adjustment to OHLC
        for col in ohlc_cols:
            if col in result_df.columns:
                result_df.loc[mask, col] *= roll.roll_ratio
        
        adjustment_info = {
            'roll_date': str(roll.roll_date.date()),
            'old_contract': roll.old_contract,
            'new_contract': roll.new_contract,
            'roll_ratio': round(float(roll.roll_ratio), 6),
            'rows_adjusted': int(mask.sum())
        }
        metadata['adjustments'].append(adjustment_info)
        
        logger.info(
            f"Applied backward ratio adjustment: {roll.old_contract} -> {roll.new_contract} "
            f"(ratio: {roll.roll_ratio:.6f}, {mask.sum()} rows adjusted)"
        )
    
    return result_df, metadata


def panama_canal_method(
    df: pd.DataFrame,
    roll_points: List[RollPoint],
    timestamp_col: str = "timestamp"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply panama canal method: concatenate contracts without adjustment.
    
    Method: Stack contracts end-to-end without any price adjustment.
    Contracts are simply concatenated at their natural price levels.
    This preserves the actual prices paid but creates artificial gaps
    at roll boundaries.
    
    Use Case: When you want to see actual prices without adjustment,
    or for volume-weighted analysis where adjustment doesn't apply.
    
    Args:
        df: DataFrame with OHLCV data
        roll_points: List of RollPoint objects (from detect_contract_rolls)
        timestamp_col: Name of timestamp column
        
    Returns:
        Tuple of:
        - DataFrame with concatenated contracts (unchanged prices)
        - Metadata dictionary with roll information
        
    Raises:
        ValueError: If timestamp column missing
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found")
    
    metadata = {
        'method': 'panama_canal',
        'total_rolls': len(roll_points),
        'roll_info': []
    }
    
    if not roll_points:
        logger.info("No roll points provided. Returning original data.")
        return df.copy(), metadata
    
    # Sort roll points by date
    sorted_rolls = sorted(roll_points, key=lambda r: r.roll_date)
    
    for roll in sorted_rolls:
        gap_info = {
            'roll_date': str(roll.roll_date.date()),
            'old_contract': roll.old_contract,
            'new_contract': roll.new_contract,
            'old_close': float(roll.old_contract_close),
            'new_close': float(roll.new_contract_close),
            'gap_pct': round(
                ((roll.new_contract_close - roll.old_contract_close) / roll.old_contract_close) * 100,
                2
            ) if roll.old_contract_close != 0 else 0
        }
        metadata['roll_info'].append(gap_info)
        
        logger.info(
            f"Panama canal roll: {roll.old_contract} -> {roll.new_contract} "
            f"(gap: {gap_info['gap_pct']:.2f}%)"
        )
    
    # Return data unchanged
    return df.copy(), metadata


# ============================================================================
# Price Continuity Verification
# ============================================================================

def verify_price_continuity(
    df: pd.DataFrame,
    roll_points: List[RollPoint],
    timestamp_col: str = "timestamp",
    close_col: str = "close",
    tolerance_pct: float = 0.1
) -> Dict:
    """
    Verify that adjusted prices maintain continuity at roll boundaries.
    
    Checks that the gap between old and new contract at roll date
    is within tolerance (default 0.1%).
    
    Args:
        df: Adjusted DataFrame
        roll_points: List of RollPoint objects
        timestamp_col: Name of timestamp column
        close_col: Name of close price column
        tolerance_pct: Maximum allowed gap as percentage (default: 0.1)
        
    Returns:
        Dictionary with continuity verification results:
        - 'passed': Boolean indicating all checks passed
        - 'total_rolls': Number of rolls checked
        - 'gaps': List of gap information for each roll
    """
    result = {
        'passed': True,
        'total_rolls': len(roll_points),
        'gaps': []
    }
    
    for roll in sorted(roll_points, key=lambda r: r.roll_date):
        # Find the last row before roll and first row after
        before_mask = df[timestamp_col] < roll.roll_date
        after_mask = df[timestamp_col] >= roll.roll_date
        
        if not before_mask.any() or not after_mask.any():
            continue
        
        before_close = df.loc[before_mask, close_col].iloc[-1]
        after_close = df.loc[after_mask, close_col].iloc[0]
        
        gap_pct = abs((after_close - before_close) / before_close * 100) if before_close != 0 else 0
        
        gap_info = {
            'roll_date': str(roll.roll_date.date()),
            'before_close': float(before_close),
            'after_close': float(after_close),
            'gap_pct': round(gap_pct, 4),
            'within_tolerance': gap_pct <= tolerance_pct
        }
        result['gaps'].append(gap_info)
        
        if gap_pct > tolerance_pct:
            result['passed'] = False
            logger.warning(
                f"Price gap at {roll.roll_date}: {gap_pct:.4f}% "
                f"(exceeds tolerance of {tolerance_pct}%)"
            )
        else:
            logger.info(
                f"Price continuity verified at {roll.roll_date}: "
                f"gap {gap_pct:.4f}% (within {tolerance_pct}%)"
            )
    
    return result


# ============================================================================
# Utility Functions
# ============================================================================

def get_supported_assets() -> List[str]:
    """
    Get list of supported assets for contract rolls.
    
    Returns:
        List of supported asset names
    """
    return sorted(list(SUPPORTED_ASSETS))


__all__ = [
    # Classes
    "RollPoint",
    "AdjustmentMethod",
    # Metadata functions
    "get_contract_spec",
    "get_expiration_months",
    "get_contract_symbol",
    "parse_contract_symbol",
    "calculate_expiration_date",
    # Detection
    "detect_contract_rolls",
    "identify_active_contract",
    # Adjustment methods
    "backward_ratio_adjustment",
    "panama_canal_method",
    # Verification
    "verify_price_continuity",
    # Utilities
    "get_supported_assets",
]
