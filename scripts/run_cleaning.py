#!/usr/bin/env python
"""
CLI entry point for running the data cleaning pipeline.

Usage:
    python scripts/run_cleaning.py --asset ES --start-date 2024-01-01 --end-date 2024-12-31
    python scripts/run_cleaning.py --asset ES --start-date 2024-01-01 --end-date 2024-12-31 --dry-run
    python scripts/run_cleaning.py --asset ES --version v1.0.0 --approve

Examples:
    # Run cleaning pipeline with auto-version increment (minor)
    python scripts/run_cleaning.py \\
        --asset ES \\
        --start-date 2024-01-01 \\
        --end-date 2024-12-31

    # Run with specific version and approval
    python scripts/run_cleaning.py \\
        --asset ES \\
        --start-date 2024-01-01 \\
        --end-date 2024-12-31 \\
        --version v1.1.0 \\
        --approve

    # Dry-run to preview changes
    python scripts/run_cleaning.py \\
        --asset ES \\
        --start-date 2024-01-01 \\
        --end-date 2024-12-31 \\
        --dry-run

    # Use alternative panama canal adjustment method
    python scripts/run_cleaning.py \\
        --asset ES \\
        --start-date 2024-01-01 \\
        --end-date 2024-12-31 \\
        --adjustment-method panama_canal \\
        --change-type major
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cleaning.cleaning_pipeline import run_cleaning_pipeline

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run data cleaning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument(
        "--asset",
        required=True,
        choices=["ES", "MES", "VIX"],
        help="Asset symbol",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--version",
        help="Target version (e.g., v1.0.0). If not specified, auto-increment",
    )
    parser.add_argument(
        "--change-type",
        choices=["major", "minor", "patch"],
        default="minor",
        help="Type of version change for auto-increment (default: minor)",
    )
    parser.add_argument(
        "--adjustment-method",
        choices=["backward_ratio", "panama_canal"],
        default="backward_ratio",
        help="Contract roll adjustment method (default: backward_ratio)",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve version for promotion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing output",
    )
    parser.add_argument(
        "--raw-data-path",
        type=Path,
        help="Path to raw data directory (default: data/raw)",
    )
    parser.add_argument(
        "--cleaned-data-path",
        type=Path,
        help="Path to cleaned data directory (default: data/cleaned)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging output",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    logger.info(f"Starting cleaning pipeline for {args.asset}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    try:
        # Run pipeline
        result = run_cleaning_pipeline(
            asset=args.asset,
            start_date=args.start_date,
            end_date=args.end_date,
            version=args.version,
            change_type=args.change_type,
            dry_run=args.dry_run,
            approval_status=args.approve,
            adjustment_method=args.adjustment_method,
            raw_data_path=args.raw_data_path,
            cleaned_data_path=args.cleaned_data_path,
        )
        
        # Print results
        print("\n" + "="*80)
        print("CLEANING PIPELINE RESULT")
        print("="*80)
        print(f"Asset:     {result.asset}")
        print(f"Version:   {result.version}")
        print(f"Success:   {result.success}")
        print(f"Status:    {result.report.status}")
        print(f"Dry Run:   {result.report.dry_run}")
        
        if result.success:
            print(f"\nMetrics:")
            print(f"  Rows In:                 {result.report.metrics.rows_in:,}")
            print(f"  Rows Out:                {result.report.metrics.rows_out:,}")
            print(f"  Rows Filtered:           {result.report.metrics.rows_filtered:,}")
            print(f"  Timezone Normalized:     {result.report.metrics.timezone_normalized}")
            print(f"  Calendar Filtered:       {result.report.metrics.calendar_filtered}")
            print(f"  Contract Rolls Detected: {result.report.metrics.contract_rolls_detected}")
            print(f"  Validation Passed:       {result.report.validation_passed}")
            
            if result.output_dir:
                print(f"\nOutput Directory: {result.output_dir}")
        else:
            print(f"Error: {result.error_message}")
            if result.report.blocking_issues:
                print("Blocking Issues:")
                for issue in result.report.blocking_issues:
                    print(f"  - {issue}")
        
        print("="*80 + "\n")
        
        # Exit with appropriate code
        return 0 if result.success else 1
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=args.verbose)
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
