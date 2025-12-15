"""
Configuration Module

Handles backtest configuration management including defaults, environment
variables, and format-specific loaders (YAML, JSON, dict).

Also provides futures contract specifications via mes_specs module.
"""

from src.backtest.config.defaults import BacktestConfig
from src.backtest.config.mes_specs import (
    MES_CONTRACT,
    ES_CONTRACT,
    NQ_CONTRACT,
    YM_CONTRACT,
)

__all__ = [
    'BacktestConfig',
    'MES_CONTRACT',
    'ES_CONTRACT',
    'NQ_CONTRACT',
    'YM_CONTRACT',
]
