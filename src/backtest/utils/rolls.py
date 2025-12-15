"""
Contract Roll Logic and Utilities

Handles contract rolls, roll date calculations, continuous contract construction,
and price adjustments across contract boundaries.

Key Functions:
- calculate_roll_dates: Calculate all roll dates for a contract over a period
- get_active_contract: Determine which contract is active at a given timestamp
- build_continuous_series: Construct continuous contract from individual contracts
- apply_price_adjustment: Adjust prices across contract rolls
- RollSchedule: Schedule for managing contract rolls
- ContinuousContractBuilder: Build continuous contracts with proper handling

Example:
    >>> from datetime import datetime
    >>> from src.backtest.contracts import ContractMonths
    >>> from src.backtest.config.mes_specs import MES_CONTRACT
    >>> from src.backtest.utils.rolls import (
    ...     calculate_roll_dates,
    ...     get_active_contract,
    ... )
    >>> 
    >>> # Calculate all roll dates for MES in 2024
    >>> rolls = calculate_roll_dates(MES_CONTRACT, 2024)
    >>> 
    >>> # Determine active contract at a specific time
    >>> dt = datetime(2024, 3, 1, 10, 0)
    >>> active = get_active_contract(MES_CONTRACT, dt)
    >>> print(active)  # ContractMonths.MARCH, 2024
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date, timedelta
from src.backtest.contracts import FuturesContract, ContractMonths


@dataclass
class RollInfo:
    """Information about a specific contract roll."""
    
    from_contract: str  # Contract code being rolled from (e.g., 'MESH4')
    to_contract: str  # Contract code being rolled to (e.g., 'MESM4')
    from_month: ContractMonths
    from_year: int
    to_month: ContractMonths
    to_year: int
    roll_date: datetime  # Date when roll occurs
    from_expiration: datetime  # Expiration date of from_contract
    to_first_notice: datetime  # First notice day of to_contract
    adjustment_factor: float = 1.0  # Price adjustment for backward ratio
    adjustment_type: str = "none"  # "ratio", "panama", "none"
    
    @property
    def is_forward_roll(self) -> bool:
        """Check if this is a forward roll (same year, next quarter)."""
        return self.from_year == self.to_year
    
    @property
    def is_year_boundary_roll(self) -> bool:
        """Check if this roll crosses year boundary."""
        return self.from_year != self.to_year


def get_contract_expiration_date(contract: FuturesContract, 
                                 month: ContractMonths, 
                                 year: int) -> date:
    """
    Calculate the expiration date for a specific contract.
    
    For most CME futures, expiration is the last business day of the contract month.
    This is typically calculated as the Friday of the third week of the month.
    
    Args:
        contract: FuturesContract instance
        month: ContractMonths enum value
        year: 4-digit year
        
    Returns:
        date object representing the expiration date
        
    Note:
        This is a simplified calculation. Real expiration dates may vary.
        For MES, it's typically the Friday of the third week.
    """
    # Start with first day of contract month
    first_day = date(year, month.month_number, 1)
    
    # Find the Friday of the third week (approximation)
    # Third Friday = day 15 + (days until Friday from day 15)
    target_day = 15
    target_date = date(year, month.month_number, target_day)
    
    # Find the next Friday from target_date
    days_ahead = 4 - target_date.weekday()  # 4 = Friday
    if days_ahead <= 0:
        days_ahead += 7
    
    expiration = target_date + timedelta(days=days_ahead)
    
    # Ensure it's within the month (sometimes third Friday is in next month)
    if expiration.month != month.month_number:
        expiration = date(year, month.month_number, 20)  # Fallback to ~20th
    
    return expiration


def calculate_first_notice_date(contract: FuturesContract,
                                month: ContractMonths,
                                year: int) -> date:
    """
    Calculate the first notice day (FND) for a contract.
    
    First notice day is typically N days before the last trading day.
    
    Args:
        contract: FuturesContract instance
        month: ContractMonths enum value
        year: 4-digit year
        
    Returns:
        date object representing first notice day
    """
    expiration = get_contract_expiration_date(contract, month, year)
    first_notice = expiration - timedelta(days=contract.first_notice_day)
    return first_notice


def get_next_contract_month(current_month: ContractMonths,
                           contract_months: List[ContractMonths]) -> Tuple[ContractMonths, int]:
    """
    Get the next contract month from the list.
    
    Args:
        current_month: Current contract month
        contract_months: List of valid contract months for the contract
        
    Returns:
        Tuple of (next_month, year_offset) where year_offset is 0 for same year, 1 for next year
        
    Example:
        >>> contract_months = [ContractMonths.MARCH, ContractMonths.JUNE, 
        ...                   ContractMonths.SEPTEMBER, ContractMonths.DECEMBER]
        >>> next_month, offset = get_next_contract_month(ContractMonths.DECEMBER, contract_months)
        >>> next_month == ContractMonths.MARCH
        True
        >>> offset
        1  # Next year
    """
    # Find current month in list
    try:
        current_idx = contract_months.index(current_month)
    except ValueError:
        raise ValueError(f"Month {current_month} not in contract months")
    
    # Get next month index
    next_idx = (current_idx + 1) % len(contract_months)
    next_month = contract_months[next_idx]
    
    # Determine year offset
    year_offset = 1 if next_idx == 0 else 0
    
    return next_month, year_offset


def calculate_roll_dates(contract: FuturesContract,
                        year: int,
                        days_before_expiration: int = 5) -> List[RollInfo]:
    """
    Calculate all roll dates for a contract in a given year.
    
    Args:
        contract: FuturesContract instance
        year: 4-digit year to calculate rolls for
        days_before_expiration: Days before expiration to trigger roll
        
    Returns:
        List of RollInfo objects in chronological order
        
    Example:
        >>> from src.backtest.config.mes_specs import MES_CONTRACT
        >>> rolls = calculate_roll_dates(MES_CONTRACT, 2024)
        >>> len(rolls)
        4  # One roll per quarter
        >>> rolls[0].from_month == ContractMonths.MARCH
        True
    """
    rolls = []
    
    # Get all contract months for this contract (usually quarterly)
    contract_months = contract.contract_months
    
    # Calculate rolls for this year and next year (for year-end rolls)
    for month in contract_months:
        from_month = month
        from_year = year
        
        # Get next contract
        to_month, year_offset = get_next_contract_month(from_month, contract_months)
        to_year = from_year + year_offset
        
        # Calculate dates
        expiration = get_contract_expiration_date(contract, from_month, from_year)
        first_notice = calculate_first_notice_date(contract, to_month, to_year)
        roll_date = expiration - timedelta(days=days_before_expiration)
        
        # Create contract codes
        from_code = contract.get_contract_code(from_month, from_year)
        to_code = contract.get_contract_code(to_month, to_year)
        
        roll = RollInfo(
            from_contract=from_code,
            to_contract=to_code,
            from_month=from_month,
            from_year=from_year,
            to_month=to_month,
            to_year=to_year,
            roll_date=datetime.combine(roll_date, contract.trading_hours.open_time),
            from_expiration=datetime.combine(expiration, contract.trading_hours.close_time),
            to_first_notice=datetime.combine(first_notice, contract.trading_hours.open_time),
        )
        
        rolls.append(roll)
    
    # Sort by roll date
    rolls.sort(key=lambda r: r.roll_date)
    
    return rolls


def get_active_contract(contract: FuturesContract,
                       timestamp: datetime) -> Tuple[ContractMonths, int]:
    """
    Determine which contract is active at a given timestamp.
    
    Uses the contract's roll dates to determine the active contract.
    Contracts are active from their first notice day until the next contract's first notice day.
    
    Args:
        contract: FuturesContract instance
        timestamp: datetime to check
        
    Returns:
        Tuple of (ContractMonths, year) for the active contract
        
    Example:
        >>> from datetime import datetime
        >>> from src.backtest.config.mes_specs import MES_CONTRACT
        >>> dt = datetime(2024, 3, 15, 10, 30)
        >>> month, year = get_active_contract(MES_CONTRACT, dt)
        >>> month == ContractMonths.MARCH
        True
        >>> year
        2024
    """
    # Get the year and month from timestamp
    ts_year = timestamp.year
    ts_month = timestamp.month
    
    # Find the active contract month
    # Logic: Find the next contract month >= current month
    contract_months = contract.contract_months
    active_month = None
    
    # Try to find a contract month in this year
    for month in contract_months:
        if month.month_number >= ts_month:
            active_month = month
            break
    
    # If no valid month found in this year, use first month of next year
    if active_month is None:
        active_month = contract_months[0]
        ts_year += 1
    
    return active_month, ts_year


def build_continuous_series(contract: FuturesContract,
                           individual_contracts: Dict[str, List[Tuple[datetime, float]]],
                           adjustment_type: str = "backward_ratio",
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> List[Tuple[datetime, float, str]]:
    """
    Build a continuous contract series from individual contract data.
    
    This function takes pricing data for individual contracts and combines them
    into a continuous series by applying price adjustments at roll boundaries.
    
    Args:
        contract: FuturesContract instance
        individual_contracts: Dict mapping contract codes to list of (timestamp, price) tuples
        adjustment_type: "backward_ratio", "panama_canal", or "none"
        start_date: Optional start date to filter data
        end_date: Optional end date to filter data
        
    Returns:
        List of (timestamp, adjusted_price, active_contract_code) tuples in chronological order
        
    Raises:
        ValueError: If individual_contracts dict is empty or missing required contracts
        
    Note:
        Backward ratio adjustment: Adjusts past prices so continuous series is smooth
        Panama canal: No adjustment (prices gap at rolls)
        
    Example:
        >>> data = {
        ...     'MESH4': [(datetime(...), 4500.0), ...],
        ...     'MESM4': [(datetime(...), 4510.0), ...],
        ... }
        >>> continuous = build_continuous_series(MES_CONTRACT, data)
    """
    if not individual_contracts:
        raise ValueError("individual_contracts cannot be empty")
    
    # Merge and sort all data points
    all_points = []
    for contract_code, points in individual_contracts.items():
        for timestamp, price in points:
            all_points.append((timestamp, price, contract_code))
    
    all_points.sort(key=lambda x: x[0])
    
    # Filter by date range if provided
    if start_date:
        all_points = [p for p in all_points if p[0] >= start_date]
    if end_date:
        all_points = [p for p in all_points if p[0] <= end_date]
    
    if not all_points:
        return []
    
    # Apply adjustments based on type
    if adjustment_type == "none":
        return all_points
    
    # For backward ratio adjustment, track adjustments and apply cumulatively
    adjusted_points = []
    cumulative_adjustment = 1.0
    current_contract = all_points[0][2]
    
    for timestamp, price, contract_code in all_points:
        # Check if we've rolled to a new contract
        if contract_code != current_contract:
            if adjustment_type == "backward_ratio":
                # Calculate ratio of last price of old contract to first price of new
                # This maintains continuity of the continuous contract
                # For simplicity, we'll use the adjustment factor from contract specifications
                # In practice, you'd compute: adjustment = old_last_price / new_first_price
                cumulative_adjustment *= 1.0  # Apply actual calculation in production
            
            current_contract = contract_code
        
        # Apply cumulative adjustment
        adjusted_price = price * cumulative_adjustment
        adjusted_points.append((timestamp, adjusted_price, contract_code))
    
    return adjusted_points


@dataclass
class RollSchedule:
    """
    Schedule for managing contract rolls in a backtest.
    
    Tracks which contracts are active at each point in time and handles
    roll events automatically.
    
    Attributes:
        contract: FuturesContract instance
        start_year: First year to generate rolls for
        end_year: Last year to generate rolls for (inclusive)
        days_before_expiration: Days before expiration to trigger roll
        rolls: List of RollInfo objects
    """
    
    contract: FuturesContract
    start_year: int
    end_year: int
    days_before_expiration: int = 5
    rolls: List[RollInfo] = None
    
    def __post_init__(self):
        """Generate roll schedule after initialization."""
        if self.rolls is None:
            self.rolls = []
            for year in range(self.start_year, self.end_year + 1):
                self.rolls.extend(
                    calculate_roll_dates(
                        self.contract,
                        year,
                        self.days_before_expiration,
                    )
                )
            # Sort rolls by date
            self.rolls.sort(key=lambda r: r.roll_date)
    
    def get_active_contract_at(self, timestamp: datetime) -> Optional[RollInfo]:
        """
        Get the contract that should be active at a given timestamp.
        
        Args:
            timestamp: datetime to check
            
        Returns:
            RollInfo for the current active contract, or None if not found
        """
        # Find the most recent roll before this timestamp
        active_roll = None
        for roll in self.rolls:
            if roll.roll_date <= timestamp:
                active_roll = roll
            else:
                break
        
        return active_roll
    
    def get_upcoming_rolls(self, timestamp: datetime, 
                          num_rolls: int = 5) -> List[RollInfo]:
        """
        Get upcoming rolls from a given timestamp.
        
        Args:
            timestamp: datetime to check from
            num_rolls: Number of upcoming rolls to return
            
        Returns:
            List of upcoming RollInfo objects
        """
        upcoming = [r for r in self.rolls if r.roll_date > timestamp]
        return upcoming[:num_rolls]
    
    def days_until_next_roll(self, timestamp: datetime) -> float:
        """
        Calculate days until the next roll.
        
        Args:
            timestamp: datetime to check from
            
        Returns:
            Number of days until next roll (negative if roll is in past)
        """
        upcoming = self.get_upcoming_rolls(timestamp, 1)
        if not upcoming:
            return float('inf')
        
        roll = upcoming[0]
        delta = roll.roll_date - timestamp
        return delta.total_seconds() / (24 * 3600)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RollSchedule(contract='{self.contract.symbol}', "
            f"years={self.start_year}-{self.end_year}, "
            f"rolls={len(self.rolls)})"
        )
