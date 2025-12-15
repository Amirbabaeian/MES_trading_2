"""
Logging Utilities

Provides structured logging for backtesting with support for:
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Strategy signal logging
- Order and trade logging
- Performance and execution time logging
- Debug mode with verbose output

Log Format:
    %(asctime)s - %(name)s - %(levelname)s - %(message)s

Loggers:
    - backtest.*: Main backtest logger hierarchy
    - backtest.strategies.*: Strategy-specific loggers
    - backtest.orders: Order execution logging
    - backtest.trades: Trade (completed position) logging
    - backtest.performance: Performance metrics logging
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional


# Global logging configuration
_LOG_LEVEL = os.getenv('BACKTEST_LOG_LEVEL', 'INFO').upper()
_LOG_FILE = os.getenv('BACKTEST_LOG_FILE', None)
_DEBUG_MODE = os.getenv('BACKTEST_DEBUG', '').lower() in ('true', '1', 'yes')

# Configure root backtest logger once
_root_configured = False
_root_logger = None


def _configure_root_logger():
    """
    Configure the root backtest logger.
    
    Sets up handlers for console and optionally file output.
    Only called once during first logger creation.
    """
    global _root_configured, _root_logger
    
    if _root_configured:
        return
    
    # Create root logger
    _root_logger = logging.getLogger('backtest')
    
    # Set level based on debug mode
    if _DEBUG_MODE:
        _root_logger.setLevel(logging.DEBUG)
    else:
        log_level = getattr(logging, _LOG_LEVEL, logging.INFO)
        _root_logger.setLevel(log_level)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    if _DEBUG_MODE:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    
    _root_logger.addHandler(console_handler)
    
    # Add file handler if configured
    if _LOG_FILE:
        try:
            os.makedirs(os.path.dirname(_LOG_FILE) or '.', exist_ok=True)
            file_handler = logging.FileHandler(_LOG_FILE)
            file_handler.setFormatter(formatter)
            if _DEBUG_MODE:
                file_handler.setLevel(logging.DEBUG)
            else:
                file_handler.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
            _root_logger.addHandler(file_handler)
        except (OSError, IOError) as e:
            console_handler.emit(
                logging.LogRecord(
                    'backtest',
                    logging.WARNING,
                    __file__,
                    0,
                    f'Failed to create log file {_LOG_FILE}: {e}',
                    (),
                    None,
                )
            )
    
    _root_configured = True


def get_logger(name: str = 'backtest', level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for backtesting.
    
    Loggers are named hierarchically using dots for organization:
    - 'backtest': Root logger
    - 'backtest.strategies': Strategy base logger
    - 'backtest.strategies.MyStrategy': Specific strategy logger
    - 'backtest.orders': Order logging
    - 'backtest.trades': Trade logging
    - 'backtest.performance': Performance metrics
    
    Args:
        name: Logger name (e.g., 'backtest.strategies.MyStrategy')
        level: Optional log level override (DEBUG, INFO, WARNING, ERROR)
               If not specified, uses environment configuration
    
    Returns:
        Configured logging.Logger instance
        
    Example:
        >>> logger = get_logger('backtest.strategies.MovingAverageCross')
        >>> logger.info('Strategy initialized')
        >>> logger.debug('Signal details: price=100.5, ma=102.3')
    """
    # Ensure root logger is configured
    _configure_root_logger()
    
    # Get or create child logger
    logger = logging.getLogger(name)
    
    # Set level if explicitly specified
    if level is not None:
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(log_level)
    elif _DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    
    return logger


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    debug: bool = False,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration for backtesting.
    
    Should be called once at application startup to configure
    the logging system before creating loggers.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        debug: Enable debug mode (verbose output)
        format_string: Custom format string for log messages
        
    Returns:
        Configured root logger
        
    Example:
        >>> setup_logging(level='DEBUG', log_file='backtest.log')
        >>> logger = get_logger('backtest.strategies.MyStrategy')
        >>> logger.debug('This will be logged to console and file')
    """
    global _root_configured, _LOG_LEVEL, _LOG_FILE, _DEBUG_MODE
    
    # Reset previous configuration
    _root_configured = False
    
    # Update globals
    _LOG_LEVEL = level.upper()
    _LOG_FILE = log_file
    _DEBUG_MODE = debug
    
    # Reconfigure root logger
    _configure_root_logger()
    
    return _root_logger


def get_trade_logger() -> logging.Logger:
    """
    Get a logger for trade (closed position) events.
    
    Returns:
        Logger configured for trade logging
        
    Example:
        >>> trade_logger = get_trade_logger()
        >>> trade_logger.info('Trade closed: entry=100.5, exit=102.3, pnl=1800')
    """
    return get_logger('backtest.trades')


def get_order_logger() -> logging.Logger:
    """
    Get a logger for order execution events.
    
    Returns:
        Logger configured for order logging
        
    Example:
        >>> order_logger = get_order_logger()
        >>> order_logger.info('Order submitted: BUY 10 contracts @ 100.5')
    """
    return get_logger('backtest.orders')


def get_signal_logger(strategy_name: str = 'strategy') -> logging.Logger:
    """
    Get a logger for trading signals.
    
    Args:
        strategy_name: Name of the strategy for logging identification
        
    Returns:
        Logger configured for signal logging
        
    Example:
        >>> signal_logger = get_signal_logger('MovingAverageCross')
        >>> signal_logger.info('BUY signal: price crossed above MA20')
    """
    return get_logger(f'backtest.signals.{strategy_name}')


def get_performance_logger() -> logging.Logger:
    """
    Get a logger for performance metrics.
    
    Logs execution time, memory usage, and other performance data.
    
    Returns:
        Logger configured for performance logging
        
    Example:
        >>> perf_logger = get_performance_logger()
        >>> perf_logger.info('Backtest execution: 1000 bars in 2.5s (400 bars/s)')
    """
    return get_logger('backtest.performance')
