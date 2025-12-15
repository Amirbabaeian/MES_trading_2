#!/usr/bin/env python3
"""
CLI interface for validation pipeline execution.

Provides command-line interface to:
- Run validation on OHLCV data
- Generate reports in multiple formats
- Manage validation overrides
- Query validation metadata
- List recent validation results

Usage:
    python scripts/run_validation.py validate --asset MES --file data.parquet
    python scripts/run_validation.py report --asset MES --format html
    python scripts/run_validation.py override --asset MES --validator schema --user john --reason "Known issue"
    python scripts/run_validation.py metadata --asset MES
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import logging
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_io.pipeline import ValidationOrchestrator, PipelineConfig
from src.data_io.override import OverrideManager, ValidationMetadata
from src.data_io.reports import (
    JSONReportGenerator,
    TextReportGenerator,
    HTMLReportGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_command(args) -> int:
    """Execute validation command."""
    # Load data
    if not Path(args.file).exists():
        logger.error(f"File not found: {args.file}")
        return 1
    
    logger.info(f"Loading data from {args.file}")
    try:
        if args.file.endswith('.parquet'):
            df = pd.read_parquet(args.file)
        elif args.file.endswith('.csv'):
            df = pd.read_csv(args.file, parse_dates=['timestamp'])
        else:
            logger.error("Unsupported file format. Use .parquet or .csv")
            return 1
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1
    
    # Create config
    config = PipelineConfig(
        enable_schema_validation=not args.skip_schema,
        enable_timestamp_validation=not args.skip_timestamp,
        enable_missing_bars_validation=not args.skip_missing_bars,
        enable_gap_detection=not args.skip_gaps,
        enable_spike_detection=not args.skip_spikes,
        enable_volume_detection=not args.skip_volume,
        save_reports=True,
        report_dir=args.report_dir,
    )
    
    # Run validation
    logger.info(f"Starting validation for {args.asset}")
    orchestrator = ValidationOrchestrator(config=config)
    result = orchestrator.validate(df, asset=args.asset)
    
    # Print summary
    print("\n" + "=" * 80)
    if result.passed:
        print(f"✓ VALIDATION PASSED: {args.asset}")
    else:
        print(f"✗ VALIDATION FAILED: {args.asset}")
    
    if result.report and result.report.summary:
        summary = result.report.summary
        print(f"  Total bars: {summary.total_bars_validated}")
        print(f"  Validators: {summary.validators_passed}/{len(result.report.results)} passed")
        print(f"  Total issues: {summary.total_issues}")
        print(f"  Critical: {summary.issues_by_severity.get('critical', 0)}")
        print(f"  Warnings: {summary.issues_by_severity.get('warning', 0)}")
        print(f"  Info: {summary.issues_by_severity.get('info', 0)}")
    
    print(f"  Can promote: {result.can_promote}")
    print("=" * 80 + "\n")
    
    return 0 if result.passed else 1


def report_command(args) -> int:
    """Generate validation report command."""
    # Load metadata
    override_manager = OverrideManager(metadata_dir=args.metadata_dir)
    metadata = override_manager.load_metadata(args.asset)
    
    if not metadata:
        logger.error(f"No validation metadata found for {args.asset}")
        return 1
    
    # For now, we'll create a simple report message
    # In production, would load the actual report
    print(f"\nValidation Report - {args.asset}")
    print("=" * 80)
    print(f"Validation Timestamp: {metadata.validation_timestamp.isoformat()}")
    print(f"State: {metadata.validation_state}")
    print(f"Passed: {metadata.passed}")
    print(f"Total Issues: {metadata.total_issues}")
    print(f"Critical Issues: {metadata.critical_issues}")
    
    if metadata.overrides:
        print(f"Overrides: {metadata.override_count}")
        for override in metadata.overrides:
            print(f"  - {override.validator_type}: {override.justification}")
    
    print("=" * 80 + "\n")
    return 0


def override_command(args) -> int:
    """Create or manage validation override."""
    override_manager = OverrideManager(metadata_dir=args.metadata_dir)
    
    if args.action == "create":
        # Create new override
        override = override_manager.create_override(
            asset=args.asset,
            validator_type=args.validator,
            overridden_by=args.user,
            justification=args.reason,
            data_version=args.data_version,
        )
        
        # Register and save
        override_manager.register_override(args.asset, override)
        
        if args.approve:
            override_manager.approve_override(override, approved_by=args.approve)
        
        print(f"\n✓ Override created:")
        print(f"  Asset: {override.asset}")
        print(f"  Validator: {override.validator_type}")
        print(f"  Created by: {override.overridden_by}")
        print(f"  Reason: {override.justification}")
        print(f"  Status: {override.status.value}")
        print()
        
        return 0
    
    elif args.action == "approve":
        # Approve existing override
        metadata = override_manager.load_metadata(args.asset)
        if not metadata:
            logger.error(f"No metadata found for {args.asset}")
            return 1
        
        for override in metadata.overrides:
            if override.validator_type == args.validator:
                override_manager.approve_override(override, approved_by=args.user)
                override_manager.register_override(args.asset, override)
                print(f"\n✓ Override approved:")
                print(f"  Validator: {override.validator_type}")
                print(f"  Approved by: {args.user}")
                print()
                return 0
        
        logger.error(f"No override found for {args.validator}")
        return 1
    
    elif args.action == "revoke":
        # Revoke override
        metadata = override_manager.load_metadata(args.asset)
        if not metadata:
            logger.error(f"No metadata found for {args.asset}")
            return 1
        
        for override in metadata.overrides:
            if override.validator_type == args.validator:
                override.revoke()
                override_manager.register_override(args.asset, override)
                print(f"\n✓ Override revoked:")
                print(f"  Validator: {override.validator_type}")
                print()
                return 0
        
        logger.error(f"No override found for {args.validator}")
        return 1
    
    return 1


def metadata_command(args) -> int:
    """Query validation metadata."""
    override_manager = OverrideManager(metadata_dir=args.metadata_dir)
    metadata = override_manager.load_metadata(args.asset)
    
    if not metadata:
        logger.error(f"No validation metadata found for {args.asset}")
        return 1
    
    # Print metadata
    print(f"\nValidation Metadata - {args.asset}")
    print("=" * 80)
    print(json.dumps(metadata.to_dict(), indent=2, default=str))
    print("=" * 80 + "\n")
    
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validation pipeline CLI interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run validation
  python scripts/run_validation.py validate --asset MES --file data.parquet
  
  # Create override
  python scripts/run_validation.py override create --asset MES --validator schema \\
      --user john --reason "Known data quality issue"
  
  # Approve override
  python scripts/run_validation.py override approve --asset MES --validator schema --user manager
  
  # Query metadata
  python scripts/run_validation.py metadata --asset MES
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Run validation on data")
    validate_parser.add_argument("--asset", required=True, help="Asset identifier")
    validate_parser.add_argument("--file", required=True, help="Path to data file (.parquet or .csv)")
    validate_parser.add_argument("--report-dir", default="./validation_reports",
                                help="Directory to save reports")
    validate_parser.add_argument("--skip-schema", action="store_true", help="Skip schema validation")
    validate_parser.add_argument("--skip-timestamp", action="store_true", help="Skip timestamp validation")
    validate_parser.add_argument("--skip-missing-bars", action="store_true", help="Skip missing bars validation")
    validate_parser.add_argument("--skip-gaps", action="store_true", help="Skip gap detection")
    validate_parser.add_argument("--skip-spikes", action="store_true", help="Skip spike detection")
    validate_parser.add_argument("--skip-volume", action="store_true", help="Skip volume detection")
    validate_parser.set_defaults(func=validate_command)
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate validation report")
    report_parser.add_argument("--asset", required=True, help="Asset identifier")
    report_parser.add_argument("--format", choices=["json", "text", "html"],
                              default="text", help="Report format")
    report_parser.add_argument("--metadata-dir", default="./validation_metadata",
                              help="Directory with validation metadata")
    report_parser.set_defaults(func=report_command)
    
    # Override command
    override_parser = subparsers.add_parser("override", help="Manage validation overrides")
    override_subparsers = override_parser.add_subparsers(dest="action",
                                                        help="Override action")
    
    # Override create
    create_parser = override_subparsers.add_parser("create", help="Create override")
    create_parser.add_argument("--asset", required=True, help="Asset identifier")
    create_parser.add_argument("--validator", required=True, help="Validator to override")
    create_parser.add_argument("--user", required=True, help="User creating override")
    create_parser.add_argument("--reason", required=True, help="Justification for override")
    create_parser.add_argument("--data-version", help="Data version being overridden")
    create_parser.add_argument("--approve", help="User approving the override")
    create_parser.add_argument("--metadata-dir", default="./validation_metadata",
                              help="Directory with validation metadata")
    create_parser.set_defaults(func=override_command)
    
    # Override approve
    approve_parser = override_subparsers.add_parser("approve", help="Approve override")
    approve_parser.add_argument("--asset", required=True, help="Asset identifier")
    approve_parser.add_argument("--validator", required=True, help="Validator to approve")
    approve_parser.add_argument("--user", required=True, help="User approving")
    approve_parser.add_argument("--metadata-dir", default="./validation_metadata",
                               help="Directory with validation metadata")
    approve_parser.set_defaults(func=override_command)
    
    # Override revoke
    revoke_parser = override_subparsers.add_parser("revoke", help="Revoke override")
    revoke_parser.add_argument("--asset", required=True, help="Asset identifier")
    revoke_parser.add_argument("--validator", required=True, help="Validator to revoke")
    revoke_parser.add_argument("--metadata-dir", default="./validation_metadata",
                              help="Directory with validation metadata")
    revoke_parser.set_defaults(func=override_command)
    
    # Metadata command
    metadata_parser = subparsers.add_parser("metadata", help="Query validation metadata")
    metadata_parser.add_argument("--asset", required=True, help="Asset identifier")
    metadata_parser.add_argument("--metadata-dir", default="./validation_metadata",
                               help="Directory with validation metadata")
    metadata_parser.set_defaults(func=metadata_command)
    
    # Parse args
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
