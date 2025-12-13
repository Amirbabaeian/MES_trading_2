"""
Metadata management utilities for Parquet files.

Provides utilities for:
- Reading/writing metadata JSON files
- Embedding custom metadata in Parquet files
- Extracting metadata from existing files
- Version tracking and lineage information
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib

logger = logging.getLogger(__name__)


class MetadataManager:
    """Manages metadata for data artifacts."""
    
    METADATA_VERSION = "1.0.0"
    
    def __init__(self):
        """Initialize metadata manager."""
        pass
    
    @staticmethod
    def create_metadata(
        layer: str,
        asset: str,
        record_count: int,
        file_size_bytes: int,
        timestamp_range: Optional[tuple] = None,
        custom_metadata: Optional[Dict[str, str]] = None,
        version_info: Optional[Dict[str, int]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a metadata dictionary for a data artifact.
        
        Args:
            layer: Data layer ('raw', 'cleaned', 'features')
            asset: Asset identifier (e.g., 'MES', 'ES', 'VIX')
            record_count: Number of records in the artifact
            file_size_bytes: Total size in bytes
            timestamp_range: Tuple of (start_timestamp, end_timestamp) as ISO strings
            custom_metadata: Custom key-value pairs
            version_info: Version information {'major_version': int, 'previous_version': int}
            dependencies: Dependencies information
        
        Returns:
            Dictionary with metadata content
        """
        metadata = {
            "metadata_version": MetadataManager.METADATA_VERSION,
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "layer": layer,
            "asset": asset,
        }
        
        if version_info:
            metadata["version_info"] = version_info
        
        if timestamp_range:
            metadata["data_range"] = {
                "start_timestamp": timestamp_range[0],
                "end_timestamp": timestamp_range[1],
            }
        
        metadata["record_count"] = record_count
        
        metadata["file_info"] = {
            "total_size_bytes": file_size_bytes,
            "total_size_human": MetadataManager._bytes_to_human(file_size_bytes),
        }
        
        if custom_metadata:
            metadata["custom"] = custom_metadata
        
        if dependencies:
            metadata["dependencies"] = dependencies
        
        return metadata
    
    @staticmethod
    def _bytes_to_human(num_bytes: int) -> str:
        """Convert bytes to human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if num_bytes < 1024:
                return f"{num_bytes:.1f} {unit}"
            num_bytes /= 1024
        return f"{num_bytes:.1f} PB"
    
    @staticmethod
    def write_metadata_file(
        metadata: Dict[str, Any],
        output_path: Path,
    ) -> None:
        """
        Write metadata to a JSON file.
        
        Args:
            metadata: Metadata dictionary
            output_path: Path to write metadata.json file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Wrote metadata to {output_path}")
    
    @staticmethod
    def read_metadata_file(metadata_path: Path) -> Dict[str, Any]:
        """
        Read metadata from a JSON file.
        
        Args:
            metadata_path: Path to metadata.json file
        
        Returns:
            Metadata dictionary
        
        Raises:
            FileNotFoundError: If metadata file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        metadata_path = Path(metadata_path)
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        logger.info(f"Read metadata from {metadata_path}")
        return metadata
    
    @staticmethod
    def update_metadata_file(
        metadata_path: Path,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update an existing metadata file with new values.
        
        Args:
            metadata_path: Path to metadata.json file
            updates: Dictionary of fields to update or add
        
        Returns:
            Updated metadata dictionary
        """
        metadata = MetadataManager.read_metadata_file(metadata_path)
        metadata.update(updates)
        MetadataManager.write_metadata_file(metadata, metadata_path)
        return metadata
    
    @staticmethod
    def add_file_info(
        metadata: Dict[str, Any],
        filename: str,
        size_bytes: int,
        file_format: str = "parquet",
        md5_checksum: Optional[str] = None,
        row_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Add file information to metadata.
        
        Args:
            metadata: Metadata dictionary
            filename: Name of the file
            size_bytes: File size in bytes
            file_format: File format (parquet, json, csv, arrow)
            md5_checksum: MD5 checksum of the file
            row_count: Number of rows (for tabular files)
        
        Returns:
            Updated metadata dictionary
        """
        if "file_info" not in metadata:
            metadata["file_info"] = {}
        
        if "file_list" not in metadata["file_info"]:
            metadata["file_info"]["file_list"] = []
        
        file_entry = {
            "filename": filename,
            "size_bytes": size_bytes,
            "format": file_format,
        }
        
        if md5_checksum:
            file_entry["md5_checksum"] = md5_checksum
        
        if row_count is not None:
            file_entry["row_count"] = row_count
        
        metadata["file_info"]["file_list"].append(file_entry)
        return metadata
    
    @staticmethod
    def compute_file_hash(file_path: Path, algorithm: str = "md5") -> str:
        """
        Compute hash of a file.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
        Returns:
            Hex digest of the hash
        """
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def add_schema_info(
        metadata: Dict[str, Any],
        schema_version: str,
        columns: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Add schema information to metadata.
        
        Args:
            metadata: Metadata dictionary
            schema_version: Version identifier for the schema
            columns: List of column definitions
        
        Returns:
            Updated metadata dictionary
        """
        metadata["schema_info"] = {
            "schema_version": schema_version,
            "columns": columns,
        }
        return metadata
    
    @staticmethod
    def add_quality_metrics(
        metadata: Dict[str, Any],
        null_counts: Optional[Dict[str, int]] = None,
        quality_flags: Optional[List[str]] = None,
        completeness_percentage: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Add data quality metrics to metadata.
        
        Args:
            metadata: Metadata dictionary
            null_counts: Count of null values per column
            quality_flags: List of quality issues or warnings
            completeness_percentage: Percentage of non-null values
        
        Returns:
            Updated metadata dictionary
        """
        data_quality = {}
        
        if null_counts is not None:
            data_quality["null_counts"] = null_counts
        
        if quality_flags is not None:
            data_quality["quality_flags"] = quality_flags
        
        if completeness_percentage is not None:
            data_quality["completeness_percentage"] = completeness_percentage
        
        if data_quality:
            metadata["data_quality"] = data_quality
        
        return metadata
    
    @staticmethod
    def add_dependencies(
        metadata: Dict[str, Any],
        raw_sources: Optional[List[Dict[str, Any]]] = None,
        cleaned_version: Optional[int] = None,
        processing_script: Optional[str] = None,
        processing_script_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add dependency information to metadata.
        
        Args:
            metadata: Metadata dictionary
            raw_sources: List of raw data sources used
            cleaned_version: Cleaned data version used
            processing_script: Script or code version used
            processing_script_hash: SHA-1 hash of processing script
        
        Returns:
            Updated metadata dictionary
        """
        dependencies = {}
        
        if raw_sources is not None:
            dependencies["raw_sources"] = raw_sources
        
        if cleaned_version is not None:
            dependencies["cleaned_version"] = cleaned_version
        
        if processing_script is not None:
            dependencies["processing_script"] = processing_script
        
        if processing_script_hash is not None:
            dependencies["processing_script_hash"] = processing_script_hash
        
        if dependencies:
            metadata["dependencies"] = dependencies
        
        return metadata
