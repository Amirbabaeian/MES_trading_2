"""
Backtest Module

A modular backtesting framework built on top of backtrader for testing trading strategies
against historical market data. Provides strategy implementations, data feeds, analyzers,
configuration management, and futures contract specifications.

Key Components:
|- strategies: Strategy implementations (base class and custom strategies)
|- feeds: Custom data feed implementations (Parquet, etc.)
|- analyzers: Custom analyzers for performance metrics
|- config: Configuration management (defaults, environment-based)
|- contracts: Futures contract specifications and management
|- utils: Utilities including logging and contract roll logic

Example:
    >>> from src.backtest.config import BacktestConfig
    >>> from src.backtest.contracts import ContractRegistry
    >>> from src.backtest.strategies.base import BaseStrategy
    >>> from src.backtest.feeds import ParquetDataFeed, create_parquet_feed
    >>> config = BacktestConfig.from_yaml('config/backtest_config.yaml')
    >>> registry = config.get_contract_registry()
    >>> mes = registry.get('MES')
    >>> feed = create_parquet_feed('MES', start_date=..., end_date=...)
    >>> # Use in backtrader cerebro
"""

from src.backtest.config.defaults import BacktestConfig
from src.backtest.strategies.base import BaseStrategy
from src.backtest.feeds import (
    ParquetDataFeed,
    create_parquet_feed,
    create_multi_feeds,
    validate_feed_exists,
    list_available_feeds,
    get_feed_date_range,
)
from src.backtest.contracts import (
    FuturesContract,
    ContractMonths,
    TradingHours,
    ContractRegistry,
)
from src.backtest.config.mes_specs import (
    MES_CONTRACT,
    ES_CONTRACT,
    NQ_CONTRACT,
    YM_CONTRACT,
)
from src.backtest.engine import (
    BacktestEngine,
    EngineConfig,
)
from src.backtest.orchestrator import (
    SimulationOrchestrator,
    SimulationConfig,
    SimulationState,
    LookAheadBiasDetector,
    run_single_asset_backtest,
    run_multi_asset_backtest,
)

__all__ = [
    'BacktestConfig',
    'BaseStrategy',
    'ParquetDataFeed',
    'create_parquet_feed',
    'create_multi_feeds',
    'validate_feed_exists',
    'list_available_feeds',
    'get_feed_date_range',
    'FuturesContract',
    'ContractMonths',
    'TradingHours',
    'ContractRegistry',
    'MES_CONTRACT',
    'ES_CONTRACT',
    'NQ_CONTRACT',
    'YM_CONTRACT',
    'BacktestEngine',
    'EngineConfig',
    'SimulationOrchestrator',
    'SimulationConfig',
    'SimulationState',
    'LookAheadBiasDetector',
    'run_single_asset_backtest',
    'run_multi_asset_backtest',
]

__version__ = '0.1.0'
