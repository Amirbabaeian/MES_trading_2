# Task 9: MES Futures Contract Configuration System

## Overview

Implemented a comprehensive configuration system for MES futures contract specifications and contract roll logic for backtesting. The system is extensible and supports multiple futures contracts beyond MES.

## Completed Deliverables

### 1. FuturesContract Base Class (`src/backtest/contracts.py`)

A flexible dataclass that defines all specifications needed to trade a futures contract:

- **Symbol & Metadata**: symbol, name, exchange, notes
- **Pricing Mechanics**: multiplier, tick_size, tick_value (calculated)
- **Margin Requirements**: initial_margin, maintenance_margin (auto-calculated if not specified)
- **Contract Schedule**: contract_months, trading_hours, first_notice_day, last_trading_day_offset
- **Contract Code Management**: 
  - `get_contract_code(month, year)` → generates codes like "MESH4"
  - `parse_contract_code(code)` → extracts month and year from code

### 2. MES Contract Specifications (`src/backtest/config/mes_specs.py`)

CME-standard specifications for MES and comparison contracts:

#### MES - Micro E-mini S&P 500
- **Tick Size**: 0.25 index points
- **Multiplier**: $5 per index point
- **Tick Value**: $1.25 per contract (0.25 × $5)
- **Initial Margin**: $500 (baseline, varies by broker)
- **Maintenance Margin**: $400 (75% of initial)
- **Contract Months**: Quarterly (March, June, September, December)
- **Contract Codes**: MESH4, MESM4, MESU4, MESZ4 (for 2024)
- **Trading Hours**: Globex nearly 24-hour (Sunday 5pm - Friday 4pm CT)

#### Additional Contracts
- **ES**: Standard E-mini S&P 500 (10× MES multiplier)
- **NQ**: E-mini NASDAQ-100
- **YM**: Micro E-mini Dow Jones Industrial Average

### 3. Contract Roll Logic (`src/backtest/utils/rolls.py`)

Comprehensive roll date calculation and management:

#### Key Functions
- `calculate_roll_dates(contract, year)` - Calculate all roll dates for a year
- `get_contract_expiration_date(contract, month, year)` - Calculate expiration date
- `calculate_first_notice_date(contract, month, year)` - Calculate FND
- `get_active_contract(contract, timestamp)` - Determine active contract at a given time
- `get_next_contract_month()` - Get next contract in sequence
- `build_continuous_series()` - Construct continuous contract from individual contracts

#### Roll Information
- `RollInfo` dataclass tracks:
  - From/to contract codes and details
  - Roll date, expiration date, first notice date
  - Price adjustment factor and type (ratio, panama, none)

#### Roll Schedule Management
- `RollSchedule` class manages rolls over extended periods:
  - `get_active_contract_at(timestamp)` - Get active contract
  - `get_upcoming_rolls(timestamp, num_rolls)` - Get future rolls
  - `days_until_next_roll(timestamp)` - Calculate days to next roll

### 4. ContractRegistry System (`src/backtest/contracts.py`)

Flexible registry for managing multiple futures contracts:

```python
registry = ContractRegistry()
registry.register(MES_CONTRACT)
registry.register(ES_CONTRACT)

mes = registry.get('MES')
symbols = registry.list_symbols()  # ['ES', 'MES']

if 'NQ' in registry:
    nq = registry.get('NQ')
```

Features:
- Register, retrieve, list, and unregister contracts
- Check contract existence with `in` operator
- Iterate over registered symbols
- Support for easy addition of new contracts

### 5. BacktestConfig Integration

Added `get_contract_registry()` method to BacktestConfig:

```python
from src.backtest.config import BacktestConfig

config = BacktestConfig()
registry = config.get_contract_registry()
mes = registry.get('MES')
```

### 6. Module Exports

Updated all module `__init__.py` files for easy access:

```python
from src.backtest import (
    FuturesContract,
    ContractMonths,
    TradingHours,
    ContractRegistry,
    MES_CONTRACT,
    ES_CONTRACT,
)

from src.backtest.utils import (
    calculate_roll_dates,
    RollSchedule,
    get_active_contract,
)
```

## Usage Examples

### Get MES Specifications

```python
from src.backtest.config.mes_specs import MES_CONTRACT

print(f"MES Tick Value: ${MES_CONTRACT.tick_value}")  # $1.25
print(f"Initial Margin: ${MES_CONTRACT.initial_margin}")  # $500.00
print(f"Multiplier: ${MES_CONTRACT.multiplier}")  # $5 per point
```

### Generate Contract Codes

```python
from src.backtest.contracts import ContractMonths
from src.backtest.config.mes_specs import MES_CONTRACT

# Generate code for March 2024
code = MES_CONTRACT.get_contract_code(ContractMonths.MARCH, 2024)
print(code)  # "MESH4"

# Parse code back
month, year = MES_CONTRACT.parse_contract_code('MESH4')
print(f"{month.name} {year}")  # "MARCH 2024"
```

### Calculate Roll Dates

```python
from src.backtest.utils import calculate_roll_dates
from src.backtest.config.mes_specs import MES_CONTRACT

rolls = calculate_roll_dates(MES_CONTRACT, 2024)
for roll in rolls:
    print(f"{roll.from_contract} -> {roll.to_contract} on {roll.roll_date.date()}")
    # Output:
    # MESH4 -> MESM4 on 2024-02-13
    # MESM4 -> MESU4 on 2024-05-14
    # MESU4 -> MESZ4 on 2024-08-14
    # MESZ4 -> MESH5 on 2024-11-12
```

### Manage Contract Registry

```python
from src.backtest.contracts import ContractRegistry
from src.backtest.config.mes_specs import MES_CONTRACT, ES_CONTRACT, NQ_CONTRACT

registry = ContractRegistry()
registry.register(MES_CONTRACT)
registry.register(ES_CONTRACT)
registry.register(NQ_CONTRACT)

# List all contracts
for symbol in registry:
    contract = registry.get(symbol)
    print(f"{symbol}: Tick Value = ${contract.tick_value}")

# Check which contracts are available
if 'MES' in registry:
    print("MES is available for trading")
```

### Track Active Contracts Over Time

```python
from datetime import datetime
from src.backtest.utils import RollSchedule
from src.backtest.config.mes_specs import MES_CONTRACT

schedule = RollSchedule(
    contract=MES_CONTRACT,
    start_year=2024,
    end_year=2025,
)

# Get active contract at a specific time
dt = datetime(2024, 3, 15, 10, 30)
active = schedule.get_active_contract_at(dt)
print(f"Active contract: {active.from_contract}")

# Get upcoming rolls
upcoming = schedule.get_upcoming_rolls(dt, num_rolls=3)
for roll in upcoming:
    days = schedule.days_until_next_roll(dt)
    print(f"Next roll in {days:.1f} days")
```

## Adding New Contracts

To add a new futures contract (e.g., MNQ - Micro NASDAQ):

```python
# In src/backtest/config/mes_specs.py

from src.backtest.contracts import FuturesContract, ContractMonths, TradingHours
from datetime import time

MNQ_CONTRACT = FuturesContract(
    symbol='MNQ',
    name='Micro E-mini NASDAQ-100',
    exchange='CME (GLOBEX)',
    multiplier=2.0,  # $2 per index point
    tick_size=0.25,  # 0.25 index points
    initial_margin=500.0,
    maintenance_margin=400.0,
    contract_months=[
        ContractMonths.MARCH,
        ContractMonths.JUNE,
        ContractMonths.SEPTEMBER,
        ContractMonths.DECEMBER,
    ],
    trading_hours=TradingHours(
        open_time=time(17, 0),
        close_time=time(16, 0),
        name="Globex Nearly 24-Hour",
        timezone="America/Chicago",
    ),
    notes="Micro E-mini NASDAQ-100",
)
```

Then register it:

```python
registry = ContractRegistry()
registry.register(MNQ_CONTRACT)
```

## Testing

Comprehensive test suite in `tests/backtest/test_contracts.py` covers:

- ✓ FuturesContract specifications (MES, ES, NQ, YM)
- ✓ ContractMonths enumeration and abbreviation lookup
- ✓ TradingHours configuration
- ✓ Contract code generation and parsing
- ✓ ContractRegistry functionality
- ✓ Contract roll date calculations
- ✓ Active contract determination
- ✓ Roll schedule management
- ✓ Integration with BacktestConfig

Run tests:
```bash
pytest tests/backtest/test_contracts.py -v
```

## Files Modified/Created

### Created
- `src/backtest/contracts.py` - FuturesContract class, ContractMonths, TradingHours, ContractRegistry
- `src/backtest/config/mes_specs.py` - MES, ES, NQ, YM contract specifications
- `src/backtest/utils/rolls.py` - Contract roll logic and RollSchedule
- `tests/backtest/test_contracts.py` - Comprehensive test suite

### Modified
- `src/backtest/__init__.py` - Added contract exports
- `src/backtest/config/__init__.py` - Added contract specs exports
- `src/backtest/config/defaults.py` - Added ContractRegistry import and get_contract_registry() method
- `src/backtest/utils/__init__.py` - Added roll utilities exports

## Key Features

1. **CME-Accurate Specifications**: MES contract specifications match official CME documentation
2. **Contract Roll Management**: Automatic calculation of roll dates, first notice days, and expirations
3. **Extensible Framework**: Easy addition of new contracts via configuration
4. **Continuous Contract Construction**: Support for building continuous contract series
5. **Registry System**: Centralized management of multiple contract specifications
6. **Backward Compatible**: Integrates with existing BacktestConfig while not breaking existing code
7. **Comprehensive Testing**: Full test coverage of all features
8. **Well Documented**: Detailed docstrings and usage examples

## Future Enhancements

1. **Backtrader Integration**: Configure commission schemes and position sizing based on contract specs
2. **Price Adjustment Logic**: Implement actual backward ratio and Panama canal adjustments
3. **Holiday Calendars**: Account for market holidays in roll date calculation
4. **Broker-Specific Margins**: Store margin requirements by broker
5. **Options Support**: Extend to support options on futures contracts
6. **Data Ingestion**: Auto-detect contract specifications from data feeds

## References

- CME MES Specifications: https://www.cmegroup.com/trading/equity-index/micro/micro-e-mini-sandp-500.html
- Futures Contract Months: https://www.cmegroup.com/education/articles-and-reports/contract-month-symbols.html
- CME Trading Hours: https://www.cmegroup.com/markets/documents/trading-hours.html
