"""
Orchestration system for automated historical data fetching and ingestion.

This module provides the Orchestrator class that coordinates data provider adapters,
manages date ranges, handles retries, tracks progress, and provides comprehensive
logging for long-running ingestion jobs.

Key features:
- Configurable date ranges and asset lists
- Parallel fetching with ThreadPoolExecutor
- Automatic retry with exponential backoff
- Incremental update mode (fetch only new data)
- Progress persistence for resumable jobs
- Dry-run mode for testing without API calls
- Comprehensive logging and reporting
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import pytz

from .base_provider import DataProvider
from .exceptions import (
    DataProviderError,
    AuthenticationError,
    DataNotAvailableError,
    RateLimitError,
    ValidationError,
    TimeoutError,
)
from .retry import RetryConfig, retry_with_config, RequestRateLimiter
from .progress import ProgressTracker, ProgressState
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when fetched data fails validation."""
    pass


class IngestionTask:
    """Represents a single data ingestion task."""
    
    def __init__(
        self,
        asset: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        provider_name: str = "unknown"
    ):
        """
        Initialize ingestion task.
        
        Args:
            asset: Asset symbol (e.g., 'ES', 'MES', 'VIX').
            timeframe: Timeframe specification (e.g., '1min', '5min', '1D').
            start_date: Start of date range.
            end_date: End of date range.
            provider_name: Name of data provider.
        """
        self.asset = asset
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.provider_name = provider_name
        self.created_at = datetime.now(tz=pytz.UTC)
    
    def __repr__(self) -> str:
        return (
            f"IngestionTask(asset={self.asset}, timeframe={self.timeframe}, "
            f"start={self.start_date.date()}, end={self.end_date.date()})"
        )


class IngestionResult:
    """Result of a completed ingestion task."""
    
    def __init__(
        self,
        task: IngestionTask,
        success: bool,
        bars_fetched: int = 0,
        data: Optional[pd.DataFrame] = None,
        error: Optional[str] = None,
        duration_seconds: float = 0.0
    ):
        """
        Initialize ingestion result.
        
        Args:
            task: The IngestionTask that was executed.
            success: Whether the ingestion succeeded.
            bars_fetched: Number of bars successfully fetched.
            data: The fetched data (if successful).
            error: Error message (if failed).
            duration_seconds: How long the task took.
        """
        self.task = task
        self.success = success
        self.bars_fetched = bars_fetched
        self.data = data
        self.error = error
        self.duration_seconds = duration_seconds
        self.completed_at = datetime.now(tz=pytz.UTC)
    
    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"IngestionResult({status}, {self.task.asset}/{self.task.timeframe}, "
            f"bars={self.bars_fetched}, time={self.duration_seconds:.1f}s)"
        )


class Orchestrator:
    """
    Main orchestration system for coordinated data ingestion.
    
    Manages:
    - Fetching data from multiple data providers
    - Coordinating multiple assets and timeframes
    - Retrying failed requests with exponential backoff
    - Tracking progress with state persistence
    - Parallel fetching of multiple assets
    - Incremental updates (fetch only new data)
    - Dry-run mode for testing
    - Comprehensive logging
    """
    
    def __init__(
        self,
        providers: Dict[str, DataProvider],
        progress_file: str = ".ingestion_state.json",
        max_workers: int = 3,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
        request_rate_limit: int = 10,
        rate_limit_period: float = 60.0,
        validate_data: bool = True,
        dry_run: bool = False,
    ):
        """
        Initialize orchestrator.
        
        Args:
            providers: Dictionary mapping provider names to DataProvider instances.
            progress_file: Path to JSON file for persisting progress state.
            max_workers: Maximum number of parallel fetch tasks.
            max_retries: Maximum retry attempts per task.
            retry_delay: Initial retry delay in seconds.
            max_retry_delay: Maximum retry delay cap.
            request_rate_limit: Maximum requests per rate limit period.
            rate_limit_period: Rate limit period in seconds.
            validate_data: Whether to validate fetched data.
            dry_run: If True, log actions without making API calls.
        """
        self.providers = providers
        self.progress = ProgressTracker(progress_file)
        self.max_workers = max_workers
        
        # Retry configuration
        self.retry_config = RetryConfig(
            max_retries=max_retries,
            base_delay=retry_delay,
            max_delay=max_retry_delay,
        )
        
        # Rate limiting
        self.request_limiter = RequestRateLimiter(
            max_requests=request_rate_limit,
            period_seconds=rate_limit_period,
        )
        
        self.validate_data = validate_data
        self.dry_run = dry_run
        
        # Statistics
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_bars_fetched = 0
        self.start_time = None
    
    def authenticate_providers(self) -> None:
        """
        Authenticate all configured providers.
        
        Raises:
            AuthenticationError: If any provider fails to authenticate.
        """
        logger.info(f"Authenticating {len(self.providers)} data provider(s)...")
        
        for name, provider in self.providers.items():
            try:
                logger.debug(f"Authenticating {name}...")
                provider.authenticate()
                logger.info(f"✓ Successfully authenticated {name}")
            except AuthenticationError as e:
                logger.error(f"✗ Failed to authenticate {name}: {e}")
                raise
            except Exception as e:
                logger.error(f"✗ Unexpected error authenticating {name}: {e}")
                raise AuthenticationError(f"Failed to authenticate {name}: {e}", provider=name)
    
    def validate_fetched_data(
        self,
        data: pd.DataFrame,
        asset: str,
        timeframe: str,
        expected_min_bars: int = 1
    ) -> None:
        """
        Validate fetched OHLCV data.
        
        Args:
            data: DataFrame to validate.
            asset: Asset symbol (for error messages).
            timeframe: Timeframe (for error messages).
            expected_min_bars: Minimum bars expected.
            
        Raises:
            DataValidationError: If validation fails.
        """
        if not self.validate_data:
            return
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(data.columns)
        if missing:
            raise DataValidationError(
                f"{asset}/{timeframe}: Missing columns: {missing}"
            )
        
        # Check minimum bars
        if len(data) < expected_min_bars:
            raise DataValidationError(
                f"{asset}/{timeframe}: Expected at least {expected_min_bars} bars, "
                f"got {len(data)}"
            )
        
        # Check index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            raise DataValidationError(
                f"{asset}/{timeframe}: Index must be DatetimeIndex"
            )
        
        # Check timestamps are sorted
        if not data.index.is_monotonic_increasing:
            raise DataValidationError(
                f"{asset}/{timeframe}: Timestamps are not sorted in ascending order"
            )
        
        # Check for duplicates
        if data.index.duplicated().any():
            raise DataValidationError(
                f"{asset}/{timeframe}: Duplicate timestamps found"
            )
        
        # Check OHLC relationships
        invalid_highs = (data['high'] < data['low']).any()
        if invalid_highs:
            raise DataValidationError(
                f"{asset}/{timeframe}: Found high < low values"
            )
        
        invalid_opens = (
            (data['open'] > data['high']) | (data['open'] < data['low'])
        ).any()
        if invalid_opens:
            raise DataValidationError(
                f"{asset}/{timeframe}: Open price outside high/low range"
            )
        
        invalid_closes = (
            (data['close'] > data['high']) | (data['close'] < data['low'])
        ).any()
        if invalid_closes:
            raise DataValidationError(
                f"{asset}/{timeframe}: Close price outside high/low range"
            )
        
        logger.debug(
            f"✓ Data validation passed for {asset}/{timeframe}: "
            f"{len(data)} bars, "
            f"timestamps {data.index[0]} to {data.index[-1]}"
        )
    
    def _fetch_from_provider(
        self,
        provider: DataProvider,
        asset: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Fetch data from provider with rate limiting.
        
        Args:
            provider: DataProvider instance.
            asset: Asset symbol.
            start_date: Start of date range.
            end_date: End of date range.
            timeframe: Timeframe specification.
            
        Returns:
            OHLCV DataFrame.
            
        Raises:
            DataProviderError: If fetch fails.
        """
        self.request_limiter.wait_if_needed()
        
        logger.debug(
            f"Fetching {asset}/{timeframe} from {start_date.date()} to {end_date.date()}"
        )
        
        data = provider.fetch_ohlcv(asset, start_date, end_date, timeframe)
        
        self.request_limiter.record_request()
        
        return data
    
    def ingest_asset(
        self,
        provider: DataProvider,
        asset: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        dry_run: Optional[bool] = None,
    ) -> IngestionResult:
        """
        Ingest data for a single asset.
        
        Args:
            provider: DataProvider instance.
            asset: Asset symbol.
            start_date: Start of date range.
            end_date: End of date range.
            timeframe: Timeframe specification.
            dry_run: Override default dry_run setting.
            
        Returns:
            IngestionResult with outcome.
        """
        task = IngestionTask(asset, timeframe, start_date, end_date, provider.name)
        task_start = time.time()
        
        dry_run = dry_run if dry_run is not None else self.dry_run
        
        try:
            # Log task start
            if dry_run:
                logger.info(
                    f"[DRY-RUN] Would fetch {asset}/{timeframe} "
                    f"from {start_date.date()} to {end_date.date()}"
                )
                return IngestionResult(
                    task=task,
                    success=True,
                    bars_fetched=0,
                    duration_seconds=time.time() - task_start
                )
            
            logger.info(
                f"Fetching {asset}/{timeframe} "
                f"from {start_date.date()} to {end_date.date()}"
            )
            
            # Fetch with retry logic
            @retry_with_config(self.retry_config)
            def fetch():
                return self._fetch_from_provider(
                    provider, asset, start_date, end_date, timeframe
                )
            
            data = fetch()
            
            # Validate data
            if len(data) > 0:
                self.validate_fetched_data(data, asset, timeframe)
            
            # Update progress
            last_timestamp = data.index[-1] if len(data) > 0 else None
            self.progress.update_state(
                asset=asset,
                timeframe=timeframe,
                last_fetched_timestamp=last_timestamp,
                bars_fetched=len(data),
                status="completed"
            )
            
            duration = time.time() - task_start
            self.tasks_completed += 1
            self.total_bars_fetched += len(data)
            
            logger.info(
                f"✓ Successfully fetched {asset}/{timeframe}: "
                f"{len(data)} bars in {duration:.1f}s"
            )
            
            return IngestionResult(
                task=task,
                success=True,
                bars_fetched=len(data),
                data=data,
                duration_seconds=duration
            )
        
        except Exception as e:
            duration = time.time() - task_start
            self.tasks_failed += 1
            
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(
                f"✗ Failed to fetch {asset}/{timeframe} after {duration:.1f}s: {error_msg}"
            )
            
            # Update progress with error
            self.progress.update_state(
                asset=asset,
                timeframe=timeframe,
                error_occurred=True,
                status="failed"
            )
            
            return IngestionResult(
                task=task,
                success=False,
                error=error_msg,
                duration_seconds=duration
            )
    
    def ingest_batch(
        self,
        provider: DataProvider,
        assets: List[str],
        timeframes: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> List[IngestionResult]:
        """
        Ingest data for multiple assets and timeframes in parallel.
        
        Args:
            provider: DataProvider instance.
            assets: List of asset symbols.
            timeframes: List of timeframe specifications.
            start_date: Start of date range.
            end_date: End of date range.
            
        Returns:
            List of IngestionResult objects.
        """
        results = []
        
        tasks = [
            (provider, asset, timeframe, start_date, end_date)
            for asset in assets
            for timeframe in timeframes
        ]
        
        logger.info(
            f"Starting batch ingestion: {len(assets)} assets × {len(timeframes)} "
            f"timeframes = {len(tasks)} tasks"
        )
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.ingest_asset, prov, asset, start, end, tf
                ): (asset, tf)
                for prov, asset, tf, start, end in tasks
            }
            
            for future in as_completed(futures):
                asset, tf = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task {asset}/{tf} raised exception: {e}")
                    task = IngestionTask(asset, tf, start_date, end_date, provider.name)
                    results.append(IngestionResult(
                        task=task,
                        success=False,
                        error=f"Task execution failed: {str(e)}",
                        duration_seconds=0.0
                    ))
        
        return results
    
    def ingest_incremental(
        self,
        provider: DataProvider,
        assets: List[str],
        timeframes: List[str],
        end_date: datetime,
        lookback_days: int = 30,
    ) -> List[IngestionResult]:
        """
        Fetch only new data since last successful fetch (incremental update).
        
        Args:
            provider: DataProvider instance.
            assets: List of asset symbols.
            timeframes: List of timeframe specifications.
            end_date: End date for fetch range.
            lookback_days: Fallback lookback if no prior data exists.
            
        Returns:
            List of IngestionResult objects.
        """
        results = []
        
        logger.info(
            f"Starting incremental ingestion for {len(assets)} assets, "
            f"up to {end_date.date()}"
        )
        
        for asset in assets:
            for timeframe in timeframes:
                # Get last successful fetch
                state = self.progress.get_state(asset, timeframe)
                
                if state.last_fetched_timestamp:
                    # Fetch from last known timestamp + 1 unit
                    start_date = state.last_fetched_timestamp + timedelta(minutes=1)
                    logger.info(
                        f"{asset}/{timeframe}: Resuming from {start_date.date()} "
                        f"(last fetched: {state.last_fetched_timestamp.isoformat()})"
                    )
                else:
                    # No prior data, use lookback
                    start_date = end_date - timedelta(days=lookback_days)
                    logger.info(
                        f"{asset}/{timeframe}: No prior data, using {lookback_days}-day lookback"
                    )
                
                # Fetch
                result = self.ingest_asset(
                    provider, asset, start_date, end_date, timeframe
                )
                results.append(result)
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of ingestion results."""
        duration = None
        if self.start_time:
            duration = time.time() - self.start_time
        
        return {
            'duration_seconds': duration,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'total_bars_fetched': self.total_bars_fetched,
            'success_rate': (
                self.tasks_completed / (self.tasks_completed + self.tasks_failed)
                if (self.tasks_completed + self.tasks_failed) > 0
                else 0.0
            ),
            'progress': self.progress.get_summary(),
        }
    
    def print_summary(self) -> None:
        """Print formatted summary of ingestion results."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("INGESTION SUMMARY")
        print("="*70)
        
        if summary['duration_seconds']:
            hours = int(summary['duration_seconds'] // 3600)
            minutes = int((summary['duration_seconds'] % 3600) // 60)
            seconds = int(summary['duration_seconds'] % 60)
            print(f"Duration: {hours}h {minutes}m {seconds}s")
        
        print(f"Tasks completed: {summary['tasks_completed']}")
        print(f"Tasks failed: {summary['tasks_failed']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Total bars fetched: {summary['total_bars_fetched']:,}")
        
        print("\nPer-asset summary:")
        print("-" * 70)
        
        for asset_info in summary['progress']['assets']:
            status_symbol = "✓" if asset_info['status'] == 'completed' else "✗"
            print(
                f"{status_symbol} {asset_info['asset']}/{asset_info['timeframe']:<6} | "
                f"bars={asset_info['bars_fetched']:>6} | "
                f"errors={asset_info['errors']} | "
                f"status={asset_info['status']}"
            )
        
        print("="*70 + "\n")
