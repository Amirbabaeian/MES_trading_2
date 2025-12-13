#!/usr/bin/env python3
"""
CLI utility for cloud storage synchronization.

Usage:
    python scripts/sync_data.py upload --config config/storage_config.yaml --local-dir data/raw --cloud-prefix raw/
    python scripts/sync_data.py download --config config/storage_config.yaml --cloud-prefix cleaned/ --local-dir data/cleaned
    python scripts/sync_data.py sync --config config/storage_config.yaml --local-dir data/ --cloud-prefix data/
    python scripts/sync_data.py status --config config/storage_config.yaml
    python scripts/sync_data.py retry --config config/storage_config.yaml
"""

import logging
import sys
import argparse
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from src.storage.config import SyncEngineBuilder, ConfigManager
from src.storage.cloud_sync import SyncDirection, SyncStatus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_progress(progress) -> None:
    """Print sync progress."""
    print(f"\nSync Progress:")
    print(f"  Files: {progress.completed_files}/{progress.total_files} "
          f"({progress.progress_percent:.1f}%)")
    print(f"  Bytes: {progress.transferred_bytes:,}/{progress.total_bytes:,} "
          f"({progress.bytes_percent:.1f}%)")
    print(f"  Speed: {progress.transfer_speed_mbps:.2f} MB/s")
    print(f"  Elapsed: {progress.elapsed_seconds:.1f}s")


def print_conflicts(conflicts) -> None:
    """Print conflict information."""
    if not conflicts:
        print("No conflicts detected.")
        return
    
    print(f"\nConflicts Detected: {len(conflicts)}")
    for conflict in conflicts:
        print(f"  File: {conflict.file_path}")
        print(f"    Reason: {conflict.conflict_reason}")
        print(f"    Resolution: {conflict.resolution.value if conflict.resolution else 'pending'}")
        if conflict.local_metadata:
            print(f"    Local: {conflict.local_metadata.modified_time}")
        if conflict.cloud_metadata:
            print(f"    Cloud: {conflict.cloud_metadata.modified_time}")


def upload_command(args) -> int:
    """Execute upload command."""
    logger.info(f"Uploading from {args.local_dir} to {args.cloud_prefix}")
    
    try:
        engine = SyncEngineBuilder.from_config_file(args.config)
        
        progress = engine.upload_directory(
            local_dir=args.local_dir,
            cloud_prefix=args.cloud_prefix,
            pattern=args.pattern or "**/*",
            exclude_patterns=args.exclude,
        )
        
        print_progress(progress)
        print(f"\nUpload completed!")
        print(f"  Completed: {progress.completed_files}")
        print(f"  Failed: {progress.failed_files}")
        
        engine.cloud_storage.disconnect()
        return 0
    
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return 1


def download_command(args) -> int:
    """Execute download command."""
    logger.info(f"Downloading from {args.cloud_prefix} to {args.local_dir}")
    
    try:
        engine = SyncEngineBuilder.from_config_file(args.config)
        
        progress = engine.download_directory(
            cloud_prefix=args.cloud_prefix,
            local_dir=args.local_dir,
        )
        
        print_progress(progress)
        print(f"\nDownload completed!")
        print(f"  Completed: {progress.completed_files}")
        print(f"  Failed: {progress.failed_files}")
        
        engine.cloud_storage.disconnect()
        return 0
    
    except Exception as e:
        logger.error(f"Download failed: {e}", exc_info=True)
        return 1


def sync_command(args) -> int:
    """Execute bidirectional sync command."""
    logger.info(f"Syncing {args.local_dir} <-> {args.cloud_prefix}")
    
    try:
        engine = SyncEngineBuilder.from_config_file(args.config)
        
        progress, conflicts = engine.sync_bidirectional(
            local_dir=args.local_dir,
            cloud_prefix=args.cloud_prefix,
            pattern=args.pattern or "**/*",
            exclude_patterns=args.exclude,
        )
        
        print_progress(progress)
        print_conflicts(conflicts)
        
        print(f"\nSync completed!")
        print(f"  Completed: {progress.completed_files}")
        print(f"  Failed: {progress.failed_files}")
        print(f"  Conflicts: {len(conflicts)}")
        
        engine.cloud_storage.disconnect()
        return 0
    
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        return 1


def status_command(args) -> int:
    """Execute status command."""
    try:
        engine = SyncEngineBuilder.from_config_file(args.config)
        
        # Get statistics
        report = engine.get_sync_report()
        
        print("\nSync Engine Status:")
        print(f"  Total Operations: {report['total_operations']}")
        print(f"  Completed: {report['completed']}")
        print(f"  Failed: {report['failed']}")
        print(f"  Total Bytes Transferred: {report['total_bytes_transferred']:,}")
        print(f"  Conflicts Detected: {report['conflicts_detected']}")
        print(f"  Transfer Speed: {report['transfer_speed_mbps']:.2f} MB/s")
        print(f"  Elapsed Time: {report['elapsed_seconds']:.1f}s")
        
        # Show last operations
        if args.verbose:
            print("\nLast 10 Operations:")
            for op in engine.operations[-10:]:
                print(f"  {op.operation_id}")
                print(f"    Status: {op.status.value}")
                print(f"    Size: {op.file_size:,} bytes")
                if op.error_message:
                    print(f"    Error: {op.error_message}")
        
        engine.cloud_storage.disconnect()
        return 0
    
    except Exception as e:
        logger.error(f"Failed to get status: {e}", exc_info=True)
        return 1


def retry_command(args) -> int:
    """Execute retry command."""
    logger.info("Retrying failed operations...")
    
    try:
        engine = SyncEngineBuilder.from_config_file(args.config)
        
        retried_count = engine.retry_failed_operations()
        
        print(f"\nRetry completed!")
        print(f"  Retried: {retried_count} operations")
        
        engine.cloud_storage.disconnect()
        return 0
    
    except Exception as e:
        logger.error(f"Retry failed: {e}", exc_info=True)
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cloud storage synchronization utility"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/storage_config.yaml"),
        help="Path to storage configuration file (default: config/storage_config.yaml)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Sync command")
    
    # Upload command
    upload = subparsers.add_parser("upload", help="Upload local directory to cloud")
    upload.add_argument("--local-dir", type=Path, required=True, help="Local directory to upload")
    upload.add_argument("--cloud-prefix", type=str, required=True, help="Cloud destination prefix")
    upload.add_argument("--pattern", type=str, default="**/*", help="File pattern to match")
    upload.add_argument("--exclude", nargs="+", help="Patterns to exclude")
    upload.set_defaults(func=upload_command)
    
    # Download command
    download = subparsers.add_parser("download", help="Download cloud directory to local")
    download.add_argument("--cloud-prefix", type=str, required=True, help="Cloud source prefix")
    download.add_argument("--local-dir", type=Path, required=True, help="Local destination directory")
    download.set_defaults(func=download_command)
    
    # Sync command
    sync = subparsers.add_parser("sync", help="Bidirectional sync between local and cloud")
    sync.add_argument("--local-dir", type=Path, required=True, help="Local directory")
    sync.add_argument("--cloud-prefix", type=str, required=True, help="Cloud prefix")
    sync.add_argument("--pattern", type=str, default="**/*", help="File pattern to match")
    sync.add_argument("--exclude", nargs="+", help="Patterns to exclude")
    sync.set_defaults(func=sync_command)
    
    # Status command
    status = subparsers.add_parser("status", help="Show sync status")
    status.set_defaults(func=status_command)
    
    # Retry command
    retry = subparsers.add_parser("retry", help="Retry failed operations")
    retry.set_defaults(func=retry_command)
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
