"""
Advanced retry logic with exponential backoff and request rate limiting.

This module provides utilities for retrying failed operations with exponential
backoff, jitter, and rate limiting to handle transient failures gracefully.
"""

import time
import logging
from typing import Callable, TypeVar, Optional, Any, Type, Tuple
from functools import wraps
import random

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_range: Tuple[float, float] = (0.5, 1.5),
        retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts (not including initial attempt).
            base_delay: Initial delay in seconds between retries.
            max_delay: Maximum delay cap in seconds.
            exponential_base: Base for exponential backoff calculation.
            jitter: Whether to add randomness to delays.
            jitter_range: Tuple of (min_multiplier, max_multiplier) for jitter.
            retryable_exceptions: Tuple of exception types that trigger retries.
                If None, all exceptions trigger retries.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_range = jitter_range
        self.retryable_exceptions = retryable_exceptions
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.
        
        Args:
            attempt: Zero-indexed attempt number.
            
        Returns:
            Delay in seconds.
        """
        # Exponential backoff
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            min_mult, max_mult = self.jitter_range
            jitter_mult = random.uniform(min_mult, max_mult)
            delay *= jitter_mult
        
        return delay
    
    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if exception should trigger a retry.
        
        Args:
            exception: The exception that was raised.
            
        Returns:
            True if should retry, False otherwise.
        """
        if self.retryable_exceptions is None:
            return True
        return isinstance(exception, self.retryable_exceptions)


class RetryableError(Exception):
    """Raised after all retry attempts have been exhausted."""
    
    def __init__(self, message: str, last_exception: Exception, attempts: int):
        """
        Initialize RetryableError.
        
        Args:
            message: Error message.
            last_exception: The last exception that was raised.
            attempts: Total number of attempts made.
        """
        self.message = message
        self.last_exception = last_exception
        self.attempts = attempts
        super().__init__(
            f"{message} (after {attempts} attempts). "
            f"Last error: {type(last_exception).__name__}: {str(last_exception)}"
        )


def retry_with_config(config: RetryConfig) -> Callable:
    """
    Decorator for retrying functions with custom retry configuration.
    
    Args:
        config: RetryConfig instance defining retry behavior.
        
    Returns:
        Decorator function.
        
    Example:
        >>> config = RetryConfig(max_retries=3, base_delay=0.5)
        >>> @retry_with_config(config)
        ... def fetch_data():
        ...     return api_call()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            total_wait = 0.0
            
            for attempt in range(config.max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(
                            f"{func.__name__} succeeded on attempt {attempt + 1} "
                            f"(waited {total_wait:.2f}s total)"
                        )
                    return result
                except Exception as e:
                    last_exception = e
                    
                    if not config.should_retry(e):
                        logger.error(f"{func.__name__} failed with non-retryable error: {str(e)}")
                        raise
                    
                    if attempt < config.max_retries:
                        delay = config.calculate_delay(attempt)
                        total_wait += delay
                        logger.warning(
                            f"{func.__name__} failed on attempt {attempt + 1}: {type(e).__name__}: {str(e)}. "
                            f"Retrying in {delay:.2f} seconds..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"{func.__name__} failed after {config.max_retries + 1} attempts"
                        )
            
            # All retries exhausted
            raise RetryableError(
                f"Failed to execute {func.__name__}",
                last_exception,
                config.max_retries + 1
            )
        
        return wrapper
    return decorator


class RequestRateLimiter:
    """
    Rate limiter for controlling request frequency with token bucket algorithm.
    
    Ensures that requests respect rate limits without bursty behavior.
    """
    
    def __init__(self, max_requests: int, period_seconds: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per period.
            period_seconds: Time period in seconds.
        """
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self.requests = []
        self._lock_until = 0.0
    
    def wait_if_needed(self) -> float:
        """
        Wait if rate limit would be exceeded.
        
        Returns:
            Seconds waited (0.0 if no wait was needed).
        """
        current_time = time.time()
        
        # If locked, wait
        if current_time < self._lock_until:
            wait_time = self._lock_until - current_time
            logger.debug(f"Rate limit: waiting {wait_time:.2f}s before next request")
            time.sleep(wait_time)
            current_time = time.time()
        
        # Remove old requests outside the period
        cutoff = current_time - self.period_seconds
        self.requests = [t for t in self.requests if t >= cutoff]
        
        # Check if at limit
        if len(self.requests) >= self.max_requests:
            oldest = self.requests[0]
            wait_time = oldest + self.period_seconds - current_time
            if wait_time > 0:
                self._lock_until = current_time + wait_time
                logger.debug(
                    f"Rate limit: hit limit ({self.max_requests} requests). "
                    f"Waiting {wait_time:.2f}s"
                )
                time.sleep(wait_time)
                return wait_time
        
        return 0.0
    
    def record_request(self) -> None:
        """Record that a request was made."""
        self.requests.append(time.time())
    
    def get_available_capacity(self) -> int:
        """Get number of requests available before rate limiting."""
        current_time = time.time()
        cutoff = current_time - self.period_seconds
        self.requests = [t for t in self.requests if t >= cutoff]
        return max(0, self.max_requests - len(self.requests))
