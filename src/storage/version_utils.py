"""
Version utilities for comparison, validation, and compatibility checks.

Provides:
- Version comparison utilities
- Version validation checks (completeness, integrity)
- Compatibility checking for backtests
- Version migration utilities
"""

import logging
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from datetime import datetime

from .version_metadata import (
    VersionMetadata,
    SemanticVersion,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Version Comparison
# ============================================================================

class VersionComparison:
    """Utilities for comparing versions."""
    
    @staticmethod
    def compare_schemas(
        version1: VersionMetadata,
        version2: VersionMetadata,
    ) -> Dict[str, any]:
        """
        Compare schemas of two versions.
        
        Args:
            version1: First version
            version2: Second version
        
        Returns:
            Dictionary with comparison results:
            - added_columns: Columns in v2 not in v1
            - removed_columns: Columns in v1 not in v2
            - modified_columns: Columns with different types
            - schema_compatible: Whether v2 is backward compatible with v1
        """
        schema1 = {col.name: col.data_type for col in version1.schema_info.columns}
        schema2 = {col.name: col.data_type for col in version2.schema_info.columns}
        
        added = set(schema2.keys()) - set(schema1.keys())
        removed = set(schema1.keys()) - set(schema2.keys())
        
        modified = []
        for col_name in set(schema1.keys()) & set(schema2.keys()):
            if schema1[col_name] != schema2[col_name]:
                modified.append({
                    "column": col_name,
                    "old_type": schema1[col_name],
                    "new_type": schema2[col_name],
                })
        
        # Backward compatible if no columns removed and no type changes
        schema_compatible = len(removed) == 0 and len(modified) == 0
        
        return {
            "added_columns": sorted(list(added)),
            "removed_columns": sorted(list(removed)),
            "modified_columns": modified,
            "schema_compatible": schema_compatible,
        }
    
    @staticmethod
    def compare_data_ranges(
        version1: VersionMetadata,
        version2: VersionMetadata,
    ) -> Dict[str, any]:
        """
        Compare data ranges of two versions.
        
        Args:
            version1: First version
            version2: Second version
        
        Returns:
            Dictionary with comparison results:
            - range_overlap: Whether ranges overlap
            - v2_extends_range: Whether v2 covers wider range
            - start_date_same: Whether start dates match
            - end_date_same: Whether end dates match
        """
        start1 = version1.data_range.start_timestamp
        end1 = version1.data_range.end_timestamp
        start2 = version2.data_range.start_timestamp
        end2 = version2.data_range.end_timestamp
        
        # Check overlap
        range_overlap = not (end1 < start2 or end2 < start1)
        
        # Check if v2 extends range
        v2_extends_range = start2 <= start1 and end2 >= end1
        
        return {
            "range_overlap": range_overlap,
            "v2_extends_range": v2_extends_range,
            "start_date_same": start1 == start2,
            "end_date_same": end1 == end2,
            "v1_range": {
                "start": start1.isoformat(),
                "end": end1.isoformat(),
            },
            "v2_range": {
                "start": start2.isoformat(),
                "end": end2.isoformat(),
            },
        }
    
    @staticmethod
    def compare_metadata(
        version1: VersionMetadata,
        version2: VersionMetadata,
    ) -> Dict[str, any]:
        """
        Compare all metadata between two versions.
        
        Args:
            version1: First version
            version2: Second version
        
        Returns:
            Comprehensive comparison dictionary
        """
        return {
            "version1": str(version1.version),
            "version2": str(version2.version),
            "schema_comparison": VersionComparison.compare_schemas(version1, version2),
            "data_range_comparison": VersionComparison.compare_data_ranges(version1, version2),
            "record_count_change": version2.record_count - version1.record_count,
            "size_change_bytes": version2.compute_total_size_bytes() - version1.compute_total_size_bytes(),
            "creation_timestamp_v1": version1.creation_timestamp.isoformat(),
            "creation_timestamp_v2": version2.creation_timestamp.isoformat(),
        }


# ============================================================================
# Version Validation
# ============================================================================

class VersionValidator:
    """Validates version integrity and completeness."""
    
    @staticmethod
    def is_complete(metadata: VersionMetadata) -> bool:
        """
        Check if version is complete (has required fields and data).
        
        Args:
            metadata: Version metadata to validate
        
        Returns:
            True if version is complete
        """
        checks = [
            bool(metadata.version),
            bool(metadata.asset_or_feature_set),
            bool(metadata.creation_timestamp),
            metadata.record_count >= 0,
            bool(metadata.schema_info),
            bool(metadata.schema_info.columns),
            bool(metadata.data_range),
        ]
        
        return all(checks)
    
    @staticmethod
    def validate_integrity(metadata: VersionMetadata) -> Dict[str, any]:
        """
        Perform comprehensive integrity checks.
        
        Args:
            metadata: Version metadata to validate
        
        Returns:
            Dictionary with validation results:
            - is_valid: Overall validity
            - checks_passed: List of passed checks
            - checks_failed: List of failed checks
            - warnings: List of warnings
        """
        checks_passed = []
        checks_failed = []
        warnings = []
        
        # Check 1: Required fields present
        if VersionValidator.is_complete(metadata):
            checks_passed.append("required_fields_present")
        else:
            checks_failed.append("required_fields_present")
        
        # Check 2: Version format valid
        try:
            SemanticVersion.parse(str(metadata.version))
            checks_passed.append("semantic_version_valid")
        except ValueError as e:
            checks_failed.append(f"semantic_version_valid: {e}")
        
        # Check 3: Data range valid
        if metadata.data_range.start_timestamp <= metadata.data_range.end_timestamp:
            checks_passed.append("data_range_valid")
        else:
            checks_failed.append("data_range_invalid")
        
        # Check 4: Schema consistency
        if metadata.schema_info.columns:
            col_names = {col.name for col in metadata.schema_info.columns}
            if len(col_names) == len(metadata.schema_info.columns):
                checks_passed.append("schema_column_names_unique")
            else:
                checks_failed.append("schema_column_names_not_unique")
        
        # Check 5: File info consistency
        if metadata.files:
            total_file_size = sum(f.size_bytes for f in metadata.files)
            if total_file_size > 0:
                checks_passed.append("files_have_size")
            else:
                warnings.append("files_have_zero_size")
        else:
            warnings.append("no_files_listed")
        
        # Check 6: Data quality
        if metadata.data_quality:
            if metadata.data_quality.validation_status == "passed":
                checks_passed.append("data_quality_passed")
            elif metadata.data_quality.validation_status == "warnings":
                checks_passed.append("data_quality_warnings")
                warnings.extend(metadata.data_quality.quality_flags)
            else:
                checks_failed.append("data_quality_failed")
        else:
            warnings.append("data_quality_not_assessed")
        
        is_valid = len(checks_failed) == 0
        
        return {
            "is_valid": is_valid,
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
            "warnings": warnings,
        }


# ============================================================================
# Compatibility Checking
# ============================================================================

class CompatibilityChecker:
    """Checks version compatibility for backtests and analyses."""
    
    @staticmethod
    def check_backward_compatibility(
        current_version: VersionMetadata,
        expected_version: VersionMetadata,
    ) -> Dict[str, any]:
        """
        Check if current version is backward compatible with expected.
        
        Args:
            current_version: Actual version being used
            expected_version: Expected version by consumer
        
        Returns:
            Compatibility check results
        """
        schema_comp = VersionComparison.compare_schemas(expected_version, current_version)
        
        compatible = schema_comp["schema_compatible"]
        
        return {
            "is_compatible": compatible,
            "reason": _get_compatibility_reason(schema_comp) if not compatible else "compatible",
            "schema_comparison": schema_comp,
        }
    
    @staticmethod
    def check_version_match(
        available_version: VersionMetadata,
        required_version: str,  # Can be "v1.0.0", "v1.x", "latest"
    ) -> bool:
        """
        Check if available version matches required version specification.
        
        Args:
            available_version: Version to check
            required_version: Required version specification
        
        Returns:
            True if versions match
        """
        if required_version == "latest":
            return True
        
        try:
            # Parse required version
            if "x" in required_version:
                # Wildcard matching: v1.x means v1.*
                parts = required_version.lstrip("v").split(".")
                avail_parts = str(available_version).lstrip("v").split(".")
                
                for i, part in enumerate(parts):
                    if part != "x" and part != avail_parts[i]:
                        return False
                return True
            else:
                # Exact match
                req = SemanticVersion.parse(required_version)
                return available_version.version == req
        except ValueError:
            return False
    
    @staticmethod
    def find_compatible_versions(
        versions: List[VersionMetadata],
        required_version: str,
    ) -> List[VersionMetadata]:
        """
        Find all versions compatible with required specification.
        
        Args:
            versions: List of available versions
            required_version: Required version specification
        
        Returns:
            List of compatible versions
        """
        return [
            v for v in versions
            if CompatibilityChecker.check_version_match(v, required_version)
        ]


# ============================================================================
# Version Migration
# ============================================================================

class VersionMigration:
    """Utilities for version migration and upgrade paths."""
    
    @staticmethod
    def get_migration_path(
        from_version: SemanticVersion,
        to_version: SemanticVersion,
        available_versions: List[VersionMetadata],
    ) -> Optional[List[SemanticVersion]]:
        """
        Find migration path from one version to another.
        
        Args:
            from_version: Starting version
            to_version: Target version
            available_versions: List of all available versions
        
        Returns:
            List of versions to migrate through, or None if no path exists
        """
        if from_version == to_version:
            return [from_version]
        
        # Simple case: only go forward
        if from_version > to_version:
            logger.warning(f"Cannot migrate backward from {from_version} to {to_version}")
            return None
        
        # Collect versions between from and to
        versions_in_range = [
            v.version for v in available_versions
            if from_version <= v.version <= to_version
        ]
        
        if not versions_in_range:
            return None
        
        versions_in_range.sort()
        
        # Check for gaps
        for i in range(len(versions_in_range) - 1):
            if versions_in_range[i] == from_version and versions_in_range[i+1] == to_version:
                return [from_version, to_version]
        
        return versions_in_range if versions_in_range[0] == from_version else None
    
    @staticmethod
    def list_breaking_changes(
        from_version: VersionMetadata,
        to_version: VersionMetadata,
    ) -> List[str]:
        """
        List breaking changes between versions.
        
        Args:
            from_version: Starting version
            to_version: Target version
        
        Returns:
            List of breaking changes
        """
        changes = []
        
        schema_comp = VersionComparison.compare_schemas(from_version, to_version)
        
        if schema_comp["removed_columns"]:
            changes.append(f"Removed columns: {schema_comp['removed_columns']}")
        
        if schema_comp["modified_columns"]:
            for mod in schema_comp["modified_columns"]:
                changes.append(
                    f"Changed {mod['column']}: {mod['old_type']} → {mod['new_type']}"
                )
        
        return changes


# ============================================================================
# Helper Functions
# ============================================================================

def _get_compatibility_reason(schema_comp: Dict[str, any]) -> str:
    """Generate human-readable compatibility reason."""
    reasons = []
    
    if schema_comp["removed_columns"]:
        reasons.append(f"Removed columns: {schema_comp['removed_columns']}")
    
    if schema_comp["modified_columns"]:
        for mod in schema_comp["modified_columns"]:
            reasons.append(f"Changed {mod['column']} type")
    
    return " | ".join(reasons) if reasons else "Unknown incompatibility"
