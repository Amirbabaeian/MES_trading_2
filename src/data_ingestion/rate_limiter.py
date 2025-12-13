"""
Rate limiting and retry logic for data provider adapters.

This module provides utilities for managing API rate limits and implementing
exponential backoff retry strategies. It enables adapters to gracefully handle
rate limiting without exceeding vendor-specific limits.
"""

import time
import logging
from typing import Callable, TypeVar, Optional, Any
from datetime import datetime, timedelta
from functools import wraps
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RateLimiter:
    """
    Token bucket rate limiter for controlling API request frequency.
    
    Implements a simple token bucket algorithm where requests are allowed
    up to a maximum rate. Once the limit is reached, the caller must wait
    before making new requests.
    
    Attributes:
        max_requests: Maximum number of requests allowed per period.
        period_seconds: Time period in seconds for rate limit (default: 60).
        request_times: Deque tracking timestamps of recent requests.
    
    Example:
        >>> limiter = RateLimiter(max_requests=10, period_seconds=60)
        >>> limiter.wait_if_needed()  # Check rate limit and wait if necessary
        >>> # Make API request here
    """
    
    def __init__(self, max_requests: int = 100, period_seconds: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per period.
            period_seconds: Time period in seconds.
        """
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self.request_times = deque(maxlen=max_requests)
        self._lock_until = 0.0
    
    def wait_if_needed(self) -> None:
        """
        Check rate limit and wait if necessary.
        
        This method should be called before making a request. If the rate limit
        has been exceeded, it will sleep until a request can be made.
        """
        current_time = time.time()
        
        # If we're in a locked state, wait
        if current_time < self._lock_until:
            wait_time = self._lock_until - current_time
            logger.debug(f"Rate limit: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)
            current_time = time.time()
        
        # Remove old request times outside the period
        cutoff_time = current_time - self.period_seconds
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
        
        # If we're at the limit, calculate when we can proceed
        if len(self.request_times) >= self.max_requests:
            oldest_request = self.request_times[0]
            time_until_oldest_expires = oldest_request + self.period_seconds - current_time
            self._lock_until = current_time + time_until_oldest_expires
            logger.debug(f"Rate limit exceeded: will retry in {time_until_oldest_expires:.2f}s")
    
    def record_request(self) -> None:
        """Record that a request was made (call after successful request)."""
        self.request_times.append(time.time())
    
    def get_available_capacity(self) -> int:
        """
        Get the number of requests that can be made before rate limiting.
        
        Returns:
            Number of requests available.
        """
        current_time = time.time()
        cutoff_time = current_time - self.period_seconds
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
        return max(0, self.max_requests - len(self.request_times))


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Automatically retries failed function calls with exponential backoff and
    optional jitter to prevent thundering herd issues.
    
    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Base for exponential calculation (default: 2.0).
        jitter: Whether to add random jitter to delays.
    
    Returns:
        Decorated function that implements retry logic.
    
    Example:
        >>> @retry_with_backoff(max_retries=3)
        ... def fetch_data():
        ...     # This function will be retried up to 3 times if it raises an exception
        ...     return api_call()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"{func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        # Calculate backoff delay
                        delay = base_delay * (exponential_base ** attempt)
                        delay = min(delay, max_delay)
                        
                        # Add jitter if enabled
                        if jitter:
                            import random
                            delay *= (0.5 + random.random())
                        
                        logger.warning(
                            f"{func.__name__} failed on attempt {attempt + 1}: {str(e)}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts"
                        )
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator


class RateLimiterMixin:
    """
    Mixin class that provides rate limiting capability to data providers.
    
    Providers that inherit from this mixin get automatic rate limiting
    through the rate_limit() context manager or check_rate_limit() method.
    
    Attributes:
        rate_limiter: RateLimiter instance.
        request_count: Total number of requests made.
        request_count_since_reset: Requests since last reset.
    """
    
    def __init__(self, *args, max_requests: int = 100, period_seconds: float = 60.0, **kwargs):
        """
        Initialize rate limiter mixin.
        
        Args:
            max_requests: Max requests per period.
            period_seconds: Period in seconds.
        """
        super().__init__(*args, **kwargs)
        self.rate_limiter = RateLimiter(max_requests, period_seconds)
        self.request_count = 0
        self.request_count_since_reset = 0
    
    def check_rate_limit(self) -> None:
        """
        Check and enforce rate limit before making a request.
        
        Call this before making an API request. It will wait if necessary
        to comply with rate limits.
        """
        self.rate_limiter.wait_if_needed()
    
    def record_api_request(self) -> None:
        """Record that an API request was successfully made."""
        self.rate_limiter.record_request()
        self.request_count += 1
        self.request_count_since_reset += 1
        logger.debug(f"Recorded API request (total: {self.request_count})")
    
    def reset_request_count(self) -> None:
        """Reset the request counter since last reset."""
        count = self.request_count_since_reset
        self.request_count_since_reset = 0
        logger.info(f"Reset request count (had {count} requests)")
    
    def get_available_requests(self) -> int:
        """Get number of requests available before hitting rate limit."""
        return self.rate_limiter.get_available_capacity()
