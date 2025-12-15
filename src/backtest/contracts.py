"""
Futures Contract Configuration and Management

Provides a flexible framework for defining and managing futures contract specifications.
Supports multiple contract types (MES, ES, NQ, etc.) with configurable parameters.

Key Classes:
- FuturesContract: Base configuration class for any futures contract
- ContractMonths: Enumeration of contract months with standard abbreviations
- ContractRegistry: Registry for managing multiple contract types
- PriceAdjustment: Handles price adjustments across contract rolls

Example:
    >>> from src.backtest.contracts import FuturesContract, ContractRegistry
    >>> from src.backtest.config.mes_specs import MES_CONTRACT
    >>> 
    >>> # Use predefined MES contract
    >>> registry = ContractRegistry()
    >>> registry.register(MES_CONTRACT)
    >>> mes_spec = registry.get('MES')
    >>> 
    >>> # Or create custom contract
    >>> custom = FuturesContract(
    ...     symbol='ES',
    ...     name='E-mini S&P 500',
    ...     multiplier=50.0,
    ...     tick_size=0.25,
    ...     ...
    ... )
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
import re


class ContractMonths(Enum):
    """Standard futures contract months with their abbreviations."""
    
    JANUARY = ('F', 1)
    FEBRUARY = ('G', 2)
    MARCH = ('H', 3)
    APRIL = ('J', 4)
    MAY = ('K', 5)
    JUNE = ('M', 6)
    JULY = ('N', 7)
    AUGUST = ('Q', 8)
    SEPTEMBER = ('U', 9)
    OCTOBER = ('V', 10)
    NOVEMBER = ('X', 11)
    DECEMBER = ('Z', 12)
    
    @classmethod
    def from_abbreviation(cls, abbrev: str) -> Optional['ContractMonths']:
        """
        Get ContractMonths from single-letter abbreviation.
        
        Args:
            abbrev: Single character abbreviation (e.g., 'H' for March)
            
        Returns:
            ContractMonths enum value or None if not found
        """
        abbrev = abbrev.upper()
        for month in cls:
            if month.value[0] == abbrev:
                return month
        return None
    
    @property
    def abbreviation(self) -> str:
        """Get the single-letter abbreviation for this month."""
        return self.value[0]
    
    @property
    def month_number(self) -> int:
        """Get the month number (1-12)."""
        return self.value[1]


@dataclass
class TradingHours:
    """Definition of trading hours for a contract."""
    
    open_time: time  # Market open time (e.g., 17:00 for Sunday Globex open)
    close_time: time  # Market close time (e.g., 16:00 for Friday close)
    name: str = "Regular Trading Hours"  # Description
    timezone: str = "America/Chicago"  # CME timezone
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'open_time': self.open_time.isoformat(),
            'close_time': self.close_time.isoformat(),
            'name': self.name,
            'timezone': self.timezone,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradingHours':
        """Create from dictionary."""
        if isinstance(data.get('open_time'), str):
            open_time = datetime.fromisoformat(data['open_time']).time()
        else:
            open_time = data['open_time']
        
        if isinstance(data.get('close_time'), str):
            close_time = datetime.fromisoformat(data['close_time']).time()
        else:
            close_time = data['close_time']
        
        return cls(
            open_time=open_time,
            close_time=close_time,
            name=data.get('name', 'Regular Trading Hours'),
            timezone=data.get('timezone', 'America/Chicago'),
        )


@dataclass
class FuturesContract:
    """
    Base configuration class for futures contracts.
    
    Defines all specifications needed to trade a futures contract, including
    pricing mechanics, margin requirements, contract schedule, and trading hours.
    
    Attributes:
        symbol: Contract symbol (e.g., 'MES', 'ES', 'NQ')
        name: Full name (e.g., 'Micro E-mini S&P 500')
        exchange: Exchange name (e.g., 'CME')
        multiplier: Dollar value per index/price point (e.g., 5.0 for MES)
        tick_size: Minimum price movement (e.g., 0.25 for MES)
        tick_value: Dollar value of minimum tick = tick_size * multiplier
        initial_margin: Initial margin requirement per contract ($)
        maintenance_margin: Maintenance margin requirement per contract ($)
        contract_months: List of months when contracts are traded
        trading_hours: TradingHours object defining market hours
        first_notice_day: Days before contract expiration when first notice occurs
        last_trading_day_offset: Days before first day of next month for last trading day
        price_precision: Decimal places for price (inferred from tick_size)
        notes: Additional notes about the contract
    """
    
    symbol: str
    name: str
    exchange: str
    multiplier: float
    tick_size: float
    initial_margin: float
    maintenance_margin: Optional[float] = None
    contract_months: List[ContractMonths] = field(default_factory=list)
    trading_hours: TradingHours = field(default_factory=lambda: TradingHours(
        open_time=time(17, 0),  # 5 PM Sunday for Globex
        close_time=time(16, 0),  # 4 PM Friday
        name="Globex Hours",
    ))
    first_notice_day: int = 5  # Days before contract expiration
    last_trading_day_offset: int = 5  # Days before next month's first day
    notes: str = ""
    
    def __post_init__(self):
        """Validate and finalize contract specification."""
        if not self.symbol:
            raise ValueError("Contract symbol cannot be empty")
        if self.multiplier <= 0:
            raise ValueError(f"Multiplier must be positive, got {self.multiplier}")
        if self.tick_size <= 0:
            raise ValueError(f"Tick size must be positive, got {self.tick_size}")
        if self.initial_margin <= 0:
            raise ValueError(f"Initial margin must be positive, got {self.initial_margin}")
        
        # Set maintenance margin to 75% of initial if not specified
        if self.maintenance_margin is None:
            self.maintenance_margin = self.initial_margin * 0.75
        
        # Ensure contract months is a list
        if not self.contract_months:
            # Default to all 12 months if not specified
            self.contract_months = list(ContractMonths)
    
    @property
    def tick_value(self) -> float:
        """Dollar value of one tick movement."""
        return self.tick_size * self.multiplier
    
    @property
    def price_precision(self) -> int:
        """Number of decimal places for prices based on tick size."""
        # Count decimal places in tick_size
        tick_str = str(self.tick_size)
        if '.' in tick_str:
            return len(tick_str.split('.')[1])
        return 0
    
    def get_month_code(self, month: ContractMonths) -> str:
        """
        Get the standard futures month code for a contract month.
        
        Args:
            month: ContractMonths enum value
            
        Returns:
            Single-letter month code (e.g., 'H' for March)
        """
        return month.abbreviation
    
    def get_contract_code(self, month: ContractMonths, year: int) -> str:
        """
        Generate contract code from month and year.
        
        Args:
            month: ContractMonths enum value
            year: 4-digit year or 2-digit year
            
        Returns:
            Contract code (e.g., 'MESH4' for MES March 2024)
        """
        if year >= 100:
            year = year % 100  # Convert 2024 to 24
        year_str = f"{year:02d}"
        return f"{self.symbol}{month.abbreviation}{year_str}"
    
    def parse_contract_code(self, code: str) -> Optional[Tuple[ContractMonths, int]]:
        """
        Parse contract code to extract month and year.
        
        Args:
            code: Contract code (e.g., 'MESH4')
            
        Returns:
            Tuple of (ContractMonths, year) or None if invalid
            
        Example:
            >>> mes = FuturesContract(...)
            >>> month, year = mes.parse_contract_code('MESH4')
            >>> month == ContractMonths.MARCH
            True
            >>> year
            2024
        """
        # Pattern: SYMBOL + MONTH_CODE + YEAR(2)
        pattern = rf'^{re.escape(self.symbol)}([A-Z])(\d{{2}})$'
        match = re.match(pattern, code.upper())
        
        if not match:
            return None
        
        month_code = match.group(1)
        year_code = match.group(2)
        
        month = ContractMonths.from_abbreviation(month_code)
        if not month:
            return None
        
        # Convert 2-digit year to 4-digit (assume 2000s for 00-99)
        year = 2000 + int(year_code)
        
        return (month, year)
    
    def to_dict(self) -> Dict:
        """Convert contract specification to dictionary."""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'exchange': self.exchange,
            'multiplier': self.multiplier,
            'tick_size': self.tick_size,
            'tick_value': self.tick_value,
            'initial_margin': self.initial_margin,
            'maintenance_margin': self.maintenance_margin,
            'contract_months': [m.name for m in self.contract_months],
            'trading_hours': self.trading_hours.to_dict(),
            'first_notice_day': self.first_notice_day,
            'last_trading_day_offset': self.last_trading_day_offset,
            'price_precision': self.price_precision,
            'notes': self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FuturesContract':
        """Create contract specification from dictionary."""
        # Convert month names back to enum values
        contract_months = []
        if 'contract_months' in data:
            for month_name in data['contract_months']:
                try:
                    contract_months.append(ContractMonths[month_name])
                except KeyError:
                    pass
        
        # Reconstruct trading hours
        trading_hours = TradingHours.from_dict(data['trading_hours']) \
            if 'trading_hours' in data else TradingHours(
                open_time=time(17, 0),
                close_time=time(16, 0),
            )
        
        return cls(
            symbol=data['symbol'],
            name=data['name'],
            exchange=data['exchange'],
            multiplier=data['multiplier'],
            tick_size=data['tick_size'],
            initial_margin=data['initial_margin'],
            maintenance_margin=data.get('maintenance_margin'),
            contract_months=contract_months or list(ContractMonths),
            trading_hours=trading_hours,
            first_notice_day=data.get('first_notice_day', 5),
            last_trading_day_offset=data.get('last_trading_day_offset', 5),
            notes=data.get('notes', ''),
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FuturesContract(symbol='{self.symbol}', name='{self.name}', "
            f"multiplier={self.multiplier}, tick_size={self.tick_size}, "
            f"tick_value=${self.tick_value:.2f}, margin=${self.initial_margin:.0f})"
        )


class ContractRegistry:
    """
    Registry for managing multiple futures contracts.
    
    Allows registration, retrieval, and validation of contract specifications.
    Supports easy addition of new contracts.
    
    Example:
        >>> registry = ContractRegistry()
        >>> registry.register(MES_CONTRACT)
        >>> registry.register(ES_CONTRACT)
        >>> mes_spec = registry.get('MES')
        >>> all_symbols = registry.list_symbols()
    """
    
    def __init__(self):
        """Initialize empty contract registry."""
        self._contracts: Dict[str, FuturesContract] = {}
    
    def register(self, contract: FuturesContract) -> None:
        """
        Register a contract specification.
        
        Args:
            contract: FuturesContract instance
            
        Raises:
            ValueError: If contract symbol already registered
        """
        if contract.symbol in self._contracts:
            raise ValueError(f"Contract '{contract.symbol}' already registered")
        self._contracts[contract.symbol] = contract
    
    def register_or_update(self, contract: FuturesContract) -> None:
        """
        Register or update a contract specification.
        
        Args:
            contract: FuturesContract instance
        """
        self._contracts[contract.symbol] = contract
    
    def get(self, symbol: str) -> Optional[FuturesContract]:
        """
        Get contract specification by symbol.
        
        Args:
            symbol: Contract symbol (e.g., 'MES')
            
        Returns:
            FuturesContract or None if not found
        """
        return self._contracts.get(symbol)
    
    def list_symbols(self) -> List[str]:
        """
        Get list of all registered contract symbols.
        
        Returns:
            List of symbols sorted alphabetically
        """
        return sorted(self._contracts.keys())
    
    def list_contracts(self) -> List[FuturesContract]:
        """
        Get list of all registered contracts.
        
        Returns:
            List of FuturesContract instances
        """
        return [self._contracts[symbol] for symbol in self.list_symbols()]
    
    def unregister(self, symbol: str) -> bool:
        """
        Unregister a contract.
        
        Args:
            symbol: Contract symbol
            
        Returns:
            True if unregistered, False if not found
        """
        if symbol in self._contracts:
            del self._contracts[symbol]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all registered contracts."""
        self._contracts.clear()
    
    def __len__(self) -> int:
        """Get number of registered contracts."""
        return len(self._contracts)
    
    def __contains__(self, symbol: str) -> bool:
        """Check if symbol is registered."""
        return symbol in self._contracts
    
    def __iter__(self):
        """Iterate over contract symbols."""
        return iter(self.list_symbols())
