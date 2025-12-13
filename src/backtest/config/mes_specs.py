"""
MES Futures Contract Specifications

Provides CME-standard specifications for the Micro E-mini S&P 500 (MES) contract.
Based on official CME documentation and exchange standards.

References:
- CME Group: https://www.cmegroup.com/trading/equity-index/micro/micro-e-mini-sandp-500.html
- Contract Code: MESH, MESM, MESU, MESZ for Mar, Jun, Sep, Dec
- Trading Symbol: MES

Key Specifications:
- Tick Size: 0.25 index points
- Multiplier: $5 per index point (0.25 × $20 per point = $1.25 per tick)
- Tick Value: $1.25 per contract
- Initial Margin: ~$500-700 (varies by broker)
- Maintenance Margin: ~$400-560
- Contract Months: Quarterly (H, M, U, Z)
- Trading Hours: Nearly 24-hour (Sunday 5pm - Friday 4pm CT)

Example:
    >>> from src.backtest.config.mes_specs import MES_CONTRACT, ES_CONTRACT
    >>> print(MES_CONTRACT.tick_value)
    1.25
    >>> print(MES_CONTRACT.get_contract_code(ContractMonths.MARCH, 2024))
    MESH4
"""

from datetime import time
from src.backtest.contracts import FuturesContract, ContractMonths, TradingHours


# ============================================================================
# MES - Micro E-mini S&P 500 (Primary contract for this system)
# ============================================================================

MES_CONTRACT = FuturesContract(
    symbol='MES',
    name='Micro E-mini S&P 500',
    exchange='CME (GLOBEX)',
    multiplier=5.0,  # $5 per index point
    tick_size=0.25,  # 0.25 index points
    initial_margin=500.0,  # Standard initial margin (varies by broker, use as baseline)
    maintenance_margin=400.0,  # Standard maintenance margin
    contract_months=[
        ContractMonths.MARCH,
        ContractMonths.JUNE,
        ContractMonths.SEPTEMBER,
        ContractMonths.DECEMBER,
    ],
    trading_hours=TradingHours(
        open_time=time(17, 0),  # 5 PM CT Sunday (Globex open)
        close_time=time(16, 0),  # 4 PM CT Friday
        name="Globex Nearly 24-Hour",
        timezone="America/Chicago",
    ),
    first_notice_day=5,  # 5 days before contract expiration
    last_trading_day_offset=5,  # Last trading day is 5 days before first of next month
    notes="Micro E-mini S&P 500. Each contract represents 1/10 of standard E-mini (ES).",
)

# Verify tick value: 0.25 points × $5 = $1.25 per tick
assert MES_CONTRACT.tick_value == 1.25, "MES tick value should be $1.25"


# ============================================================================
# ES - Standard E-mini S&P 500 (For comparison and testing)
# ============================================================================

ES_CONTRACT = FuturesContract(
    symbol='ES',
    name='E-mini S&P 500',
    exchange='CME (GLOBEX)',
    multiplier=50.0,  # $50 per index point
    tick_size=0.25,  # 0.25 index points
    initial_margin=12500.0,  # Standard initial margin
    maintenance_margin=10000.0,  # Standard maintenance margin
    contract_months=[
        ContractMonths.MARCH,
        ContractMonths.JUNE,
        ContractMonths.SEPTEMBER,
        ContractMonths.DECEMBER,
    ],
    trading_hours=TradingHours(
        open_time=time(17, 0),  # 5 PM CT Sunday (Globex open)
        close_time=time(16, 0),  # 4 PM CT Friday
        name="Globex Nearly 24-Hour",
        timezone="America/Chicago",
    ),
    first_notice_day=5,
    last_trading_day_offset=5,
    notes="Standard E-mini S&P 500. Each contract represents 10× the micro E-mini (MES).",
)

# Verify tick value: 0.25 points × $50 = $12.50 per tick
assert ES_CONTRACT.tick_value == 12.50, "ES tick value should be $12.50"


# ============================================================================
# NQ - Micro E-mini NASDAQ-100 (For extensibility testing)
# ============================================================================

NQ_CONTRACT = FuturesContract(
    symbol='NQ',
    name='E-mini NASDAQ-100',
    exchange='CME (GLOBEX)',
    multiplier=20.0,  # $20 per index point
    tick_size=0.25,  # 0.25 index points
    initial_margin=5000.0,
    maintenance_margin=4000.0,
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
    first_notice_day=5,
    last_trading_day_offset=5,
    notes="E-mini NASDAQ-100. Quarterly contracts.",
)


# ============================================================================
# YM - Mini Dow Jones Industrial Average
# ============================================================================

YM_CONTRACT = FuturesContract(
    symbol='YM',
    name='Micro E-mini Dow',
    exchange='CME (GLOBEX)',
    multiplier=5.0,  # $5 per index point
    tick_size=1.0,  # 1 index point
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
    first_notice_day=5,
    last_trading_day_offset=5,
    notes="Micro E-mini Dow Jones. Each contract is 1/10 of standard Dow (YM).",
)
