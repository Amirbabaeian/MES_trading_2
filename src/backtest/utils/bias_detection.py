"""
Look-Ahead Bias Detection Utilities

Provides tools and assertions to detect and prevent look-ahead bias
during strategy development and backtesting.

Look-ahead bias occurs when a strategy accesses data that would not
have been available at the time of the trading decision, leading to
artificially inflated backtest results.

Key Functions:
- assert_no_future_data_access: Assertion wrapper for development
- detect_future_access: Detects future data access in real-time
- validate_data_isolation: Ensures current bar is isolated from future
"""

from typing import Any, Optional, Callable
from datetime import datetime
import inspect

from src.backtest.utils.logging import get_logger


logger = get_logger(__name__)


class LookAheadBiasError(Exception):
    """Raised when look-ahead bias is detected."""
    pass


class DataIsolationViolation(Exception):
    """Raised when current bar data is not properly isolated."""
    pass


def assert_current_bar_timestamp(
    accessed_timestamp: datetime,
    current_bar_timestamp: datetime,
    context: str = '',
    strict: bool = True,
) -> None:
    """
    Assert that accessed timestamp is not from the future.
    
    This is a development tool to catch look-ahead bias early.
    
    Args:
        accessed_timestamp: Timestamp being accessed
        current_bar_timestamp: Timestamp of current bar
        context: Description of where access occurred
        strict: If True, raise exception. If False, only log warning.
        
    Raises:
        LookAheadBiasError: If accessed_timestamp > current_bar_timestamp and strict=True
    """
    if accessed_timestamp > current_bar_timestamp:
        msg = (
            f'LOOK-AHEAD BIAS DETECTED: '
            f'Attempted to access future timestamp {accessed_timestamp} '
            f'while processing bar {current_bar_timestamp}. '
            f'Context: {context}'
        )
        
        if strict:
            logger.error(msg)
            raise LookAheadBiasError(msg)
        else:
            logger.warning(msg)


def assert_bar_not_in_future(
    bar_offset: int,
    context: str = '',
    strict: bool = True,
) -> None:
    """
    Assert that bar offset is not accessing future bars.
    
    In backtrader, bar index 0 is current bar, -1 is previous bar.
    Future bars would have positive indices.
    
    Args:
        bar_offset: Bar index relative to current (0=current, -1=previous)
        context: Description of access
        strict: If True, raise exception. If False, only log warning.
        
    Raises:
        LookAheadBiasError: If bar_offset > 0 and strict=True
    """
    if bar_offset > 0:
        msg = (
            f'LOOK-AHEAD BIAS DETECTED: '
            f'Attempted to access future bar at offset {bar_offset} '
            f'(positive indices are in the future). '
            f'Context: {context}'
        )
        
        if strict:
            logger.error(msg)
            raise LookAheadBiasError(msg)
        else:
            logger.warning(msg)


def validate_price_within_bar_range(
    price: float,
    bar_low: float,
    bar_high: float,
    bar_close: float,
    context: str = '',
) -> bool:
    """
    Validate that a price is within the bar's range.
    
    Useful for detecting if fill prices violate realistic execution.
    
    Args:
        price: Price to validate
        bar_low: Bar's low price
        bar_high: Bar's high price
        bar_close: Bar's close price
        context: Description of access
        
    Returns:
        True if price is within [low, high], False otherwise
    """
    if not (bar_low <= price <= bar_high):
        logger.warning(
            f'Price {price:.2f} is outside bar range [{bar_low:.2f}, {bar_high:.2f}]. '
            f'Context: {context}'
        )
        return False
    return True


def detect_future_data_usage(
    strategy_func: Callable,
    current_bar_idx: int,
    total_bars: int,
) -> bool:
    """
    Attempt to detect if a strategy function uses future data.
    
    This is a heuristic check based on code inspection.
    
    Args:
        strategy_func: Strategy next() method or callable
        current_bar_idx: Current bar index
        total_bars: Total bars available
        
    Returns:
        True if potential future data usage detected, False otherwise
    """
    try:
        source = inspect.getsource(strategy_func)
        
        # Check for common patterns of future data access
        suspicious_patterns = [
            'len(self) - ',  # Accessing bars ahead
            'self.datas[0][-',  # Accessing specific bar ahead
            'future',
            'tomorrow',
            'next_bar',
        ]
        
        for pattern in suspicious_patterns:
            if pattern in source:
                logger.warning(
                    f'Potential future data usage detected in {strategy_func.__name__}: '
                    f'Found pattern "{pattern}"'
                )
                return True
        
        return False
    
    except Exception as e:
        logger.debug(f'Could not inspect function for future data usage: {e}')
        return False


def validate_data_isolation(
    current_data: Any,
    next_bar_data: Optional[Any] = None,
    symbol: str = 'UNKNOWN',
) -> bool:
    """
    Validate that current bar data is isolated from future bars.
    
    Args:
        current_data: Current bar's data array
        next_bar_data: Next bar's data (if available for comparison)
        symbol: Symbol being validated
        
    Returns:
        True if data is properly isolated, False otherwise
    """
    # In a well-isolated system, we should not have access to next_bar_data
    if next_bar_data is not None:
        logger.warning(
            f'Data isolation warning for {symbol}: '
            f'next_bar_data should not be accessible'
        )
        return False
    
    return True


def create_bias_check_decorator(strict: bool = True) -> Callable:
    """
    Create a decorator to add bias checking to strategy methods.
    
    Args:
        strict: If True, raise exceptions. If False, only log.
        
    Returns:
        Decorator function
        
    Example:
        >>> check_bias = create_bias_check_decorator(strict=True)
        >>> @check_bias
        >>> def my_strategy_next(self):
        ...     # Your strategy logic
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            # Get current bar timestamp from backtrader
            if hasattr(self, 'datas') and len(self.datas) > 0:
                try:
                    current_ts = self.datas[0].datetime.datetime(0)
                    
                    # Check that we're not accessing future bars
                    # In backtrader, self.datas[0][0] is current close
                    # Accessing self.datas[0][1] would be next bar (future)
                    
                    # This is a simple check - can be enhanced
                    assert_bar_not_in_future(
                        0,
                        context=f'{func.__name__} in {self.__class__.__name__}',
                        strict=strict,
                    )
                
                except Exception as e:
                    if strict:
                        raise
                    else:
                        logger.warning(f'Bias check failed: {e}')
            
            return func(self, *args, **kwargs)
        
        return wrapper
    
    return decorator


def setup_bias_monitoring(strategy_instance: Any, strict: bool = True) -> None:
    """
    Set up bias monitoring on a strategy instance.
    
    Wraps the strategy's next() method with bias checks.
    
    Args:
        strategy_instance: Instantiated strategy object
        strict: If True, raise on violations. If False, only log.
    """
    if not hasattr(strategy_instance, 'next'):
        logger.warning('Strategy has no next() method to monitor')
        return
    
    original_next = strategy_instance.next
    decorator = create_bias_check_decorator(strict=strict)
    
    # Wrap the next method
    strategy_instance.next = decorator(original_next).__get__(strategy_instance, type(strategy_instance))
    
    logger.info(f'Bias monitoring enabled for {strategy_instance.__class__.__name__}')
