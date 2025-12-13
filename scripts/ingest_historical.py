#!/usr/bin/env python3
"""
Historical data ingestion script.

Fetches historical OHLCV data for configured assets and timeframes
from specified date ranges using the data ingestion orchestrator.

Usage:
    # Fetch ES/MES/VIX daily data for 2024
    python scripts/ingest_historical.py \\
        --assets ES MES VIX \\
        --timeframes 1D \\
        --start-date 2024-01-01 \\
        --end-date 2024-12-31

    # Dry-run (log what would be fetched without making API calls)
    python scripts/ingest_historical.py \\
        --assets ES \\
        --timeframes 1D 5M \\
        --dry-run

    # Fetch with custom retry settings
    python scripts/ingest_historical.py \\
        --assets MES \\
        --max-retries 5 \\
        --retry-delay 2.0 \\
        --verbose
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pytz

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_ingestion import (
    MockProvider,
    IBDataProvider,
    PolygonDataProvider,
    DatabentoDataProvider,
    CredentialLoader,
)
from data_ingestion.orchestrator import Orchestrator


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def parse_date(date_str: str) -> datetime:
    """Parse date string to UTC datetime."""
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return dt.replace(tzinfo=pytz.UTC)


def get_default_dates() -> tuple:
    """Get default date range (past 30 days)."""
    end_date = datetime.now(tz=pytz.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=30)
    return start_date, end_date


def initialize_providers(use_mock: bool = False) -> dict:
    """
    Initialize data providers.
    
    Args:
        use_mock: If True, use MockProvider instead of real providers.
        
    Returns:
        Dictionary mapping provider names to provider instances.
    """
    providers = {}
    
    if use_mock:
        logging.info("Using MockProvider for all data")
        providers['mock'] = MockProvider()
    else:
        # Try to initialize real providers with credentials
        loader = CredentialLoader()
        
        # Interactive Brokers
        try:
            ib_creds = loader.load_credentials('interactive_brokers')
            if ib_creds:
                providers['ib'] = IBDataProvider(**ib_creds)
                logging.info("Initialized Interactive Brokers provider")
        except Exception as e:
            logging.warning(f"Could not initialize Interactive Brokers: {e}")
        
        # Polygon.io
        try:
            polygon_creds = loader.load_credentials('polygon')
            if polygon_creds:
                providers['polygon'] = PolygonDataProvider(**polygon_creds)
                logging.info("Initialized Polygon provider")
        except Exception as e:
            logging.warning(f"Could not initialize Polygon: {e}")
        
        # Databento
        try:
            databento_creds = loader.load_credentials('databento')
            if databento_creds:
                providers['databento'] = DatabentoDataProvider(**databento_creds)
                logging.info("Initialized Databento provider")
        except Exception as e:
            logging.warning(f"Could not initialize Databento: {e}")
        
        # Fallback to MockProvider if no real providers available
        if not providers:
            logging.warning(
                "No credentials found for real providers. Using MockProvider instead. "
                "Set up credentials in .env or config/credentials.json to use real providers."
            )
            providers['mock'] = MockProvider()
    
    return providers


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch historical OHLCV data for configured assets and timeframes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        '--assets',
        nargs='+',
        default=['ES', 'MES', 'VIX'],
        help='Assets to fetch (default: ES MES VIX)'
    )
    
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=['1D'],
        help='Timeframes to fetch (default: 1D)'
    )
    
    # Date range arguments
    start_default, end_default = get_default_dates()
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=start_default.strftime('%Y-%m-%d'),
        help=f'Start date (YYYY-MM-DD, default: {start_default.date()})'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=end_default.strftime('%Y-%m-%d'),
        help=f'End date (YYYY-MM-DD, default: {end_default.date()})'
    )
    
    # Orchestrator options
    parser.add_argument(
        '--max-workers',
        type=int,
        default=3,
        help='Maximum parallel fetch tasks (default: 3)'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help='Maximum retry attempts per task (default: 3)'
    )
    
    parser.add_argument(
        '--retry-delay',
        type=float,
        default=1.0,
        help='Initial retry delay in seconds (default: 1.0)'
    )
    
    parser.add_argument(
        '--max-retry-delay',
        type=float,
        default=60.0,
        help='Maximum retry delay in seconds (default: 60.0)'
    )
    
    parser.add_argument(
        '--rate-limit',
        type=int,
        default=10,
        help='Max requests per rate limit period (default: 10)'
    )
    
    # Execution options
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Log what would be fetched without making API calls'
    )
    
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use MockProvider instead of real data providers'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip data validation checks'
    )
    
    parser.add_argument(
        '--progress-file',
        type=str,
        default='.ingestion_state.json',
        help='Path to progress state file (default: .ingestion_state.json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting historical data ingestion")
    logger.info(f"Assets: {args.assets}")
    logger.info(f"Timeframes: {args.timeframes}")
    
    try:
        # Parse dates
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        
        if start_date > end_date:
            logger.error("start-date must be before end-date")
            return 1
        
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Initialize providers
        providers = initialize_providers(use_mock=args.mock)
        
        if not providers:
            logger.error("No data providers available")
            return 1
        
        logger.info(f"Available providers: {', '.join(providers.keys())}")
        
        # Create orchestrator
        orchestrator = Orchestrator(
            providers=providers,
            progress_file=args.progress_file,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            max_retry_delay=args.max_retry_delay,
            request_rate_limit=args.rate_limit,
            validate_data=not args.no_validate,
            dry_run=args.dry_run,
        )
        
        # Authenticate providers (skip for dry-run)
        if not args.dry_run:
            try:
                orchestrator.authenticate_providers()
            except Exception as e:
                logger.error(f"Provider authentication failed: {e}")
                if not args.mock:
                    logger.info("Try --mock to use MockProvider instead")
                return 1
        
        # Perform ingestion
        orchestrator.start_time = datetime.now(tz=pytz.UTC).timestamp() if hasattr(datetime.now(), 'timestamp') else None
        
        import time
        orchestrator.start_time = time.time()
        
        # Use primary provider (first in list)
        provider = next(iter(providers.values()))
        
        results = orchestrator.ingest_batch(
            provider=provider,
            assets=args.assets,
            timeframes=args.timeframes,
            start_date=start_date,
            end_date=end_date,
        )
        
        # Print summary
        orchestrator.print_summary()
        
        # Return exit code based on success rate
        summary = orchestrator.get_summary()
        if summary['success_rate'] == 1.0:
            logger.info("All tasks completed successfully")
            return 0
        elif summary['success_rate'] > 0.0:
            logger.warning(f"Partial success: {summary['success_rate']:.1%} of tasks completed")
            return 1
        else:
            logger.error("All tasks failed")
            return 1
    
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
