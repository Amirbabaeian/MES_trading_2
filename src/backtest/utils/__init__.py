"""
Utilities Module

Provides helper functions and utilities for backtesting including logging,
debugging, contract roll logic, and common operations.
"""

from src.backtest.utils.logging import get_logger, setup_logging
from src.backtest.utils.rolls import (
    calculate_roll_dates,
    get_active_contract,
    get_contract_expiration_date,
    calculate_first_notice_date,
    get_next_contract_month,
    build_continuous_series,
    RollSchedule,
    RollInfo,
)
from src.backtest.utils.bias_detection import (
    assert_current_bar_timestamp,
    assert_bar_not_in_future,
    validate_price_within_bar_range,
    detect_future_data_usage,
    validate_data_isolation,
    create_bias_check_decorator,
    setup_bias_monitoring,
    LookAheadBiasError,
    DataIsolationViolation,
)

__all__ = [
    'get_logger',
    'setup_logging',
    'calculate_roll_dates',
    'get_active_contract',
    'get_contract_expiration_date',
    'calculate_first_notice_date',
    'get_next_contract_month',
    'build_continuous_series',
    'RollSchedule',
    'RollInfo',
    'assert_current_bar_timestamp',
    'assert_bar_not_in_future',
    'validate_price_within_bar_range',
    'detect_future_data_usage',
    'validate_data_isolation',
    'create_bias_check_decorator',
    'setup_bias_monitoring',
    'LookAheadBiasError',
    'DataIsolationViolation',
]
