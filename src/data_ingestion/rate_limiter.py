"""
Rate limiting and retry logic for data provider adapters.

This module provides utilities for implementing rate limiting, exponential
backoff, and request tracking for API-based data providers.
"""

import time
import logging
from typing import Callable, Any, TypeVar, Optional
from datetime import datetime, timedelta
from functools import wraps

T = TypeVar('T')

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Simple rate limiter that tracks requests and enforces delays.
    
    Parameters
    ----------
    requests_per_second : float
        Maximum requests per second (default: 10)
    burst_size : int
        Maximum burst size before enforcing delays (default: 5)
    """
    
    def __init__(self, requests_per_second: float = 10.0, burst_size: int = 5):
        """Initialize the rate limiter."""
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.min_interval = 1.0 / requests_per_second
        self.request_times = []
        
    def wait_if_needed(self) -> None:
        """
        Wait if necessary to maintain rate limit.
        
        Tracks request times and enforces delays based on configured limits.
        """
        now = time.time()
        
        # Remove old requests outside the window
        cutoff_time = now - 1.0
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # If at burst limit, wait
        if len(self.request_times) >= self.burst_size:
            sleep_time = self.min_interval
            logger.debug(f"Rate limit: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)
            now = time.time()
            self.request_times = [t for t in self.request_times if t > now - 1.0]
        
        self.request_times.append(now)
    
    def reset(self) -> None:
        """Reset the request tracking."""
        self.request_times = []


class ExponentialBackoff:
    """
    Exponential backoff strategy for retrying failed requests.
    
    Parameters
    ----------
    initial_delay : float
        Initial delay in seconds (default: 1.0)
    max_delay : float
        Maximum delay in seconds (default: 300.0, i.e., 5 minutes)
    exponential_base : float
        Base for exponential calculation (default: 2.0)
    max_retries : int
        Maximum number of retries (default: 5)
    """
    
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 300.0,
        exponential_base: float = 2.0,
        max_retries: int = 5,
    ):
        """Initialize the backoff strategy."""
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.max_retries = max_retries
    
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for the given attempt number.
        
        Parameters
        ----------
        attempt : int
            Attempt number (0-indexed)
        
        Returns
        -------
        float
            Delay in seconds
        """
        if attempt < 0:
            return 0.0
        if attempt >= self.max_retries:
            return self.max_delay
        
        delay = self.initial_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)
    
    def sleep(self, attempt: int) -> None:
        """
        Sleep for the calculated delay duration.
        
        Parameters
        ----------
        attempt : int
            Attempt number (0-indexed)
        """
        delay = self.get_delay(attempt)
        if delay > 0:
            logger.debug(f"Backoff: sleeping {delay:.3f}s after attempt {attempt}")
            time.sleep(delay)


def retry_with_backoff(
    max_retries: int = 3,
    backoff: Optional[ExponentialBackoff] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Parameters
    ----------
    max_retries : int
        Maximum number of retries (default: 3)
    backoff : ExponentialBackoff, optional
        Backoff strategy. If None, uses default.
    on_retry : Callable, optional
        Callback function called on retry with (attempt, exception)
    
    Returns
    -------
    Callable
        Decorated function that retries on exception
    
    Examples
    --------
    >>> @retry_with_backoff(max_retries=3)
    >>> def fetch_data():
    ...     # This function will be retried up to 3 times on exception
    ...     return api_call()
    """
    if backoff is None:
        backoff = ExponentialBackoff(max_retries=max_retries)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying..."
                    )
                    
                    if on_retry:
                        on_retry(attempt, e)
                    
                    backoff.sleep(attempt)
            
            # All retries failed
            logger.error(
                f"All {max_retries} attempts failed for {func.__name__}: "
                f"{last_exception}"
            )
            raise last_exception
        
        return wrapper
    
    return decorator
