"""
Metadata management utilities for Parquet files.

Handles embedding, extraction, and validation of custom metadata
in Parquet file headers.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class MetadataManager:
    """
    Manages metadata embedding and extraction for Parquet files.
    
    Parquet files support custom key-value metadata in the file footer.
    This manager provides utilities to work with this metadata in a
    structured way.
    """
    
    # Standard metadata keys
    CREATION_TIMESTAMP = "creation_timestamp"
    DATA_RANGE_START = "data_range_start"
    DATA_RANGE_END = "data_range_end"
    ROW_COUNT = "row_count"
    SCHEMA_VERSION = "schema_version"
    ASSET = "asset"
    VERSION = "version"
    CHANGELOG = "changelog"
    DEPENDENCIES = "dependencies"
    COMPRESSION = "compression"
    PARQUET_VERSION = "parquet_version"
    
    STANDARD_KEYS = {
        CREATION_TIMESTAMP,
        DATA_RANGE_START,
        DATA_RANGE_END,
        ROW_COUNT,
        SCHEMA_VERSION,
        ASSET,
        VERSION,
        CHANGELOG,
        DEPENDENCIES,
        COMPRESSION,
        PARQUET_VERSION,
    }
    
    def __init__(self):
        """Initialize metadata manager."""
        pass
    
    @staticmethod
    def create_metadata(
        creation_timestamp: Optional[str] = None,
        data_range_start: Optional[str] = None,
        data_range_end: Optional[str] = None,
        row_count: Optional[int] = None,
        schema_version: str = "1.0",
        asset: Optional[str] = None,
        version: Optional[str] = None,
        changelog: Optional[str] = None,
        dependencies: Optional[list] = None,
        compression: str = "snappy",
        **custom_metadata
    ) -> Dict[str, str]:
        """
        Create a metadata dictionary with standard and custom fields.
        
        Args:
            creation_timestamp: ISO format timestamp (auto-generated if not provided)
            data_range_start: Start of data time range (ISO format)
            data_range_end: End of data time range (ISO format)
            row_count: Number of rows in dataset
            schema_version: Version of the schema used
            asset: Asset identifier (e.g., "MES", "ES", "VIX")
            version: Data version (e.g., "v1", "v2.1")
            changelog: Summary of changes from previous version
            dependencies: List of data dependencies
            compression: Compression codec used
            **custom_metadata: Additional custom key-value pairs
            
        Returns:
            Dictionary of metadata suitable for Parquet metadata
        """
        metadata = {}
        
        # Set creation timestamp if not provided
        if creation_timestamp is None:
            creation_timestamp = datetime.utcnow().isoformat()
        
        # Add standard metadata
        if creation_timestamp:
            metadata[MetadataManager.CREATION_TIMESTAMP] = creation_timestamp
        if data_range_start:
            metadata[MetadataManager.DATA_RANGE_START] = data_range_start
        if data_range_end:
            metadata[MetadataManager.DATA_RANGE_END] = data_range_end
        if row_count is not None:
            metadata[MetadataManager.ROW_COUNT] = str(row_count)
        if schema_version:
            metadata[MetadataManager.SCHEMA_VERSION] = schema_version
        if asset:
            metadata[MetadataManager.ASSET] = asset
        if version:
            metadata[MetadataManager.VERSION] = version
        if changelog:
            metadata[MetadataManager.CHANGELOG] = changelog
        if dependencies:
            metadata[MetadataManager.DEPENDENCIES] = json.dumps(dependencies)
        if compression:
            metadata[MetadataManager.COMPRESSION] = compression
        
        # Add custom metadata (convert non-string values to JSON)
        for key, value in custom_metadata.items():
            if isinstance(value, str):
                metadata[key] = value
            else:
                # Convert complex types to JSON strings
                try:
                    metadata[key] = json.dumps(value)
                except (TypeError, ValueError):
                    metadata[key] = str(value)
                    logger.warning(
                        f"Custom metadata '{key}' could not be JSON serialized, "
                        f"using string representation"
                    )
        
        return metadata
    
    @staticmethod
    def extract_metadata(parquet_metadata: Dict[bytes, bytes]) -> Dict[str, Any]:
        """
        Extract and parse metadata from Parquet file metadata.
        
        Args:
            parquet_metadata: Raw metadata dict from Parquet file
            (as returned by parquet_file.schema_arrow.metadata)
            
        Returns:
            Dictionary with decoded metadata
        """
        if not parquet_metadata:
            return {}
        
        decoded = {}
        for key, value in parquet_metadata.items():
            # Keys and values are bytes, decode them
            try:
                k = key.decode("utf-8") if isinstance(key, bytes) else key
                v = value.decode("utf-8") if isinstance(value, bytes) else value
                
                # Try to parse JSON values
                if k in {MetadataManager.DEPENDENCIES, MetadataManager.CHANGELOG}:
                    try:
                        decoded[k] = json.loads(v)
                    except json.JSONDecodeError:
                        decoded[k] = v
                else:
                    decoded[k] = v
            except (UnicodeDecodeError, AttributeError) as e:
                logger.warning(f"Failed to decode metadata key/value: {e}")
                decoded[str(key)] = str(value)
        
        return decoded
    
    @staticmethod
    def validate_metadata(
        metadata: Dict[str, Any],
        required_keys: Optional[list] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate metadata dictionary.
        
        Args:
            metadata: Metadata dictionary to validate
            required_keys: List of required metadata keys
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(metadata, dict):
            return False, "Metadata must be a dictionary"
        
        if required_keys:
            missing = set(required_keys) - set(metadata.keys())
            if missing:
                return False, f"Missing required metadata keys: {sorted(missing)}"
        
        # Validate specific fields if present
        if MetadataManager.CREATION_TIMESTAMP in metadata:
            try:
                # Try to parse as ISO format
                datetime.fromisoformat(
                    metadata[MetadataManager.CREATION_TIMESTAMP].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                return False, (
                    f"Invalid {MetadataManager.CREATION_TIMESTAMP} format. "
                    f"Expected ISO format string"
                )
        
        if MetadataManager.ROW_COUNT in metadata:
            try:
                int(metadata[MetadataManager.ROW_COUNT])
            except (ValueError, TypeError):
                return False, (
                    f"Invalid {MetadataManager.ROW_COUNT}. Expected integer value"
                )
        
        return True, None
    
    @staticmethod
    def merge_metadata(
        base_metadata: Dict[str, str],
        updates: Dict[str, str],
        preserve_creation_timestamp: bool = True
    ) -> Dict[str, str]:
        """
        Merge updated metadata with base metadata.
        
        Args:
            base_metadata: Original metadata dictionary
            updates: Updated/new metadata key-value pairs
            preserve_creation_timestamp: Keep original creation timestamp
            
        Returns:
            Merged metadata dictionary
        """
        merged = base_metadata.copy()
        
        # Preserve creation timestamp if requested and it exists
        if preserve_creation_timestamp:
            original_timestamp = merged.get(MetadataManager.CREATION_TIMESTAMP)
        
        merged.update(updates)
        
        if preserve_creation_timestamp and original_timestamp:
            merged[MetadataManager.CREATION_TIMESTAMP] = original_timestamp
        
        return merged
    
    @staticmethod
    def format_metadata_for_display(metadata: Dict[str, Any]) -> str:
        """
        Format metadata dictionary as human-readable string.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Formatted string representation
        """
        if not metadata:
            return "No metadata"
        
        lines = []
        for key, value in sorted(metadata.items()):
            if isinstance(value, (list, dict)):
                # Format complex types nicely
                value_str = json.dumps(value, indent=2)
                lines.append(f"{key}:\n  {value_str}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
