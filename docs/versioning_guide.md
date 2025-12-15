# Version Management System Guide

Complete guide to the version management system for cleaned and features data.

## Overview

The version management system enables:
- **Semantic versioning** of cleaned and features datasets
- **Reproducibility** of backtest results by referencing specific data versions
- **Data lineage tracking** to understand how data was processed
- **Version compatibility checking** for backtests and analyses
- **Immutable versions** for production use through freezing
- **Version lifecycle management** through tagging and deprecation

## Core Concepts

### Semantic Versioning

Versions follow semantic versioning format: `v{major}.{minor}.{patch}`

- **Major**: Breaking changes (schema changes, incompatible transformations)
- **Minor**: Non-breaking additions (new columns, extended date range)
- **Patch**: Bug fixes (corrections to existing data)

Example progression:
- v1.0.0 - Initial release
- v1.1.0 - Added volatility column (non-breaking)
- v1.1.1 - Fixed duplicate records (non-breaking patch)
- v2.0.0 - Changed data normalization (breaking change)

### Metadata

Each version includes comprehensive metadata stored in `metadata.json`:

```json
{
  "layer": "cleaned",
  "version": "v1.0.0",
  "asset_or_feature_set": "MES",
  "creation_timestamp": "2024-01-15T14:30:00Z",
  "data_range": {
    "start_timestamp": "2024-01-01T00:00:00Z",
    "end_timestamp": "2024-01-31T23:59:59Z"
  },
  "record_count": 10000,
  "schema_info": {
    "schema_version": "1.0.0",
    "columns": [...]
  },
  "files": [...],
  "data_quality": {...},
  "lineage": {...},
  "changelog": [...],
  "tags": ["stable", "production"],
  "frozen": false,
  "creator": "data_team",
  "environment": "production"
}
```

### Data Lineage

Lineage tracks dependencies and processing:

```python
lineage = Lineage(
    raw_sources=[
        {"asset": "MES", "dates": ["2024-01-01", "2024-01-02"]},
        {"asset": "ES", "dates": ["2024-01-01", "2024-01-02"]},
    ],
    cleaning_parameters={
        "outlier_threshold": 2.5,
        "null_handling": "interpolate",
    },
    processing_script="cleaning_pipeline.py",
    processing_script_hash="abc123...",
)
```

## Quick Start

### Basic Usage

```python
from pathlib import Path
from datetime import datetime, timezone, timedelta
from src.storage.version_manager import VersionManager
from src.storage.version_metadata import (
    SemanticVersion, DataRange, SchemaColumn, SchemaInfo, FileInfo
)

# Initialize manager
manager = VersionManager(Path("/data/cleaned"), layer="cleaned")

# Create a schema
schema = SchemaInfo(
    schema_version="1.0.0",
    columns=[
        SchemaColumn("timestamp", "timestamp[ns]", False),
        SchemaColumn("close", "double", False),
        SchemaColumn("volume", "int64", False),
    ]
)

# Define data range
data_range = DataRange(
    start_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
    end_timestamp=datetime(2024, 1, 31, tzinfo=timezone.utc),
)

# Create version
version = manager.create_version(
    asset_or_feature_set="MES",
    version=SemanticVersion(1, 0, 0),
    data_range=data_range,
    record_count=10000,
    schema_info=schema,
    creator="data_team",
    environment="production",
)

# Retrieve version
metadata = manager.get_version("MES", SemanticVersion(1, 0, 0))

# Get latest version
latest = manager.get_latest_version("MES")

# List all versions
all_versions = manager.list_versions()

# List versions for specific asset
mes_versions = manager.list_versions("MES")
```

## Tagging System

Tags enable grouping and discovery of versions:

```python
# Add tags
manager.add_tag("MES", SemanticVersion(1, 0, 0), "stable")
manager.add_tag("MES", SemanticVersion(1, 0, 0), "production")

# Find versions by tag
stable_versions = manager.get_versions_by_tag("stable")

# Remove tags
manager.remove_tag("MES", SemanticVersion(1, 0, 0), "experimental")
```

Common tags:
- `stable` - Tested and reliable
- `production` - Used in live systems
- `latest` - Most recent version
- `research-YYYYMM` - Used in specific analysis
- `experimental` - Testing/development

## Freezing and Immutability

Freeze versions to prevent modifications (important for backtests):

```python
# Freeze version
manager.freeze_version("MES", SemanticVersion(1, 0, 0), author="data_team")

# Check if frozen
is_frozen = manager.is_frozen("MES", SemanticVersion(1, 0, 0))

# Unfreeze (if needed)
manager.unfreeze_version("MES", SemanticVersion(1, 0, 0), author="data_team")

# Frozen versions raise error when modified
# Deleting frozen version requires force=True
manager.delete_version("MES", SemanticVersion(1, 0, 0), force=True)
```

## Version Comparison

Compare versions to understand changes:

```python
from src.storage.version_utils import VersionComparison

v1 = manager.get_version("MES", SemanticVersion(1, 0, 0))
v2 = manager.get_version("MES", SemanticVersion(1, 1, 0))

# Compare schemas
schema_comp = VersionComparison.compare_schemas(v1, v2)
print(f"Added columns: {schema_comp['added_columns']}")
print(f"Removed columns: {schema_comp['removed_columns']}")
print(f"Compatible: {schema_comp['schema_compatible']}")

# Compare data ranges
range_comp = VersionComparison.compare_data_ranges(v1, v2)
print(f"Range overlap: {range_comp['range_overlap']}")
print(f"V2 extends range: {range_comp['v2_extends_range']}")

# Full comparison
full_comp = VersionComparison.compare_metadata(v1, v2)
```

## Validation

Validate version integrity:

```python
from src.storage.version_utils import VersionValidator

# Check completeness
is_complete = VersionValidator.is_complete(metadata)

# Full validation
validation = VersionValidator.validate_integrity(metadata)
print(f"Valid: {validation['is_valid']}")
print(f"Passed checks: {validation['checks_passed']}")
print(f"Failed checks: {validation['checks_failed']}")
print(f"Warnings: {validation['warnings']}")
```

## Compatibility Checking

Ensure versions are compatible with backtests:

```python
from src.storage.version_utils import CompatibilityChecker

# Check exact version match
matches = CompatibilityChecker.check_version_match(metadata, "v1.0.0")

# Check wildcard match
matches = CompatibilityChecker.check_version_match(metadata, "v1.x")  # Any v1.x

# Check 'latest'
matches = CompatibilityChecker.check_version_match(metadata, "latest")

# Find compatible versions
versions = [v1, v2, v3]
compatible = CompatibilityChecker.find_compatible_versions(versions, "v1.x")

# Check backward compatibility
current = manager.get_version("MES", SemanticVersion(2, 0, 0))
expected = manager.get_version("MES", SemanticVersion(1, 0, 0))
compat = CompatibilityChecker.check_backward_compatibility(current, expected)
print(f"Compatible: {compat['is_compatible']}")
if not compat['is_compatible']:
    print(f"Reason: {compat['reason']}")
```

## Deprecation

Mark versions as deprecated:

```python
# Deprecate old version
manager.deprecate_version(
    "MES",
    SemanticVersion(1, 0, 0),
    replacement_version=SemanticVersion(2, 0, 0),
    author="data_team",
)

# List only active versions
active = manager.get_active_versions("MES")

# Still accessible, just marked as deprecated
metadata = manager.get_version("MES", SemanticVersion(1, 0, 0))
assert "deprecated" in metadata.tags
```

## Directory Structure

Versions are stored with this structure:

```
/data/
├── cleaned/
│   ├── MES/
│   │   ├── v1.0.0/
│   │   │   ├── metadata.json
│   │   │   ├── ohlcv.parquet
│   │   │   └── quality_metrics.json
│   │   ├── v1.1.0/
│   │   │   ├── metadata.json
│   │   │   └── ohlcv.parquet
│   │   └── v2.0.0/
│   │       ├── metadata.json
│   │       └── ohlcv.parquet
│   └── ES/
│       ├── v1.0.0/
│       └── ...
└── features/
    ├── price_volatility/
    │   ├── v1.0.0/
    │   │   ├── metadata.json
    │   │   ├── MES.parquet
    │   │   └── ES.parquet
    │   └── v1.1.0/
    │       └── ...
    └── momentum_indicators/
        └── ...
```

## Changelog

Each version maintains a changelog:

```python
metadata = manager.get_version("MES", SemanticVersion(1, 0, 0))

for entry in metadata.changelog:
    print(f"{entry.version}: {entry.change_type}")
    print(f"  Author: {entry.author}")
    print(f"  Time: {entry.timestamp}")
    print(f"  Description: {entry.description}")
```

## Data Quality Tracking

Include quality metrics in version metadata:

```python
from src.storage.version_metadata import DataQuality

quality = DataQuality(
    validation_status="passed",
    quality_flags=["passed_all_checks"],
    completeness_percentage=99.8,
    null_counts={"timestamp": 0, "close": 5},
    validation_checks=[
        "schema_validation",
        "null_check",
        "range_validation",
        "duplicate_check",
    ],
)

version = manager.create_version(
    asset_or_feature_set="MES",
    version=SemanticVersion(1, 0, 0),
    data_range=data_range,
    record_count=10000,
    schema_info=schema,
    data_quality=quality,
)
```

## Backtest Integration

Reference specific versions in backtest configurations:

```yaml
# backtest_config.yaml
backtest:
  name: "momentum_strategy_v1"
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  
  data:
    cleaned_version: "v1.0.0"  # Pin to specific version
    # or
    cleaned_version: "v1.x"     # Use any v1.x
    # or
    cleaned_version: "latest"   # Use latest (not recommended for reproducibility)
    
    features:
      price_volatility: "v1.0.0"
      momentum_indicators: "v1.1.0"
```

Loading versions in backtest code:

```python
from src.storage.version_manager import VersionManager
from src.storage.version_metadata import SemanticVersion

manager = VersionManager(Path("/data/cleaned"), layer="cleaned")

# Load specific version
config = load_backtest_config("backtest_config.yaml")
version_spec = config["data"]["cleaned_version"]

# Resolve to actual version
if version_spec == "latest":
    metadata = manager.get_latest_version("MES")
else:
    version = SemanticVersion.parse(version_spec.split(".")[0] + "." + version_spec.split(".")[1] + ".0")
    metadata = manager.get_version("MES", version)

print(f"Using {metadata.version} of {metadata.asset_or_feature_set}")
```

## Best Practices

### When Creating Versions

1. **Always include metadata**: Never omit lineage, schema, or quality info
2. **Set appropriate tags**: Help others find relevant versions
3. **Document changes**: Write clear changelog entries
4. **Freeze production versions**: Prevent accidental modifications
5. **Test before promoting**: Validate version completeness

### Version Numbering

- **Major changes**: Update training/backtest code in advance
- **Minor changes**: Backward compatible, no code changes needed
- **Patch versions**: Bug fixes, safe to apply retroactively

### Data Lineage

Always record:
- Raw data sources used
- Cleaning/feature parameters
- Processing script versions
- Validation checks applied

Example:

```python
lineage = Lineage(
    raw_sources=[
        {
            "asset": "MES",
            "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    ],
    cleaning_parameters={
        "method": "interpolation",
        "outlier_threshold": 2.5,
        "timezone": "UTC",
    },
    processing_script="scripts/clean_ohlcv.py",
    processing_script_hash="abc123def456...",
)
```

### Tagging Strategy

Use consistent tagging:

```
Stability: stable, experimental, deprecated
Environment: production, staging, development
Type: backtest-2024-01, research, official
Owner: team-a, team-b
```

## Migration Guide

Migrating from v1 to v2:

1. **Create v2** alongside existing v1
2. **Tag v1 as legacy**: Mark with "legacy" or "deprecated"
3. **Test v2 thoroughly**: Run backtests with both versions
4. **Compare results**: Verify expected differences
5. **Update consumers**: Point to v2 gradually
6. **Archive v1**: Move to long-term storage after migration

## Troubleshooting

### Version Not Found

```python
# Check if version exists
if not manager.version_exists("MES", SemanticVersion(1, 0, 0)):
    print("Version not found")
    
# List available versions
all_versions = manager.list_versions("MES")
for v in all_versions:
    print(f"  {v.version}")
```

### Frozen Version Error

```python
# Check if frozen
if manager.is_frozen("MES", SemanticVersion(1, 0, 0)):
    print("Version is frozen")
    
# Unfreeze if authorized
manager.unfreeze_version("MES", SemanticVersion(1, 0, 0), author="data_admin")
```

### Compatibility Issues

```python
from src.storage.version_utils import VersionValidator

# Validate version integrity
result = VersionValidator.validate_integrity(metadata)
for issue in result['checks_failed']:
    print(f"Validation failed: {issue}")
```

## API Reference

### VersionManager

- `create_version(...)` - Create new version
- `get_version(asset, version)` - Retrieve specific version
- `get_latest_version(asset)` - Get most recent version
- `list_versions([asset])` - List all or filtered versions
- `version_exists(asset, version)` - Check existence
- `delete_version(asset, version, force=False)` - Delete version
- `add_tag(asset, version, tag)` - Add tag
- `remove_tag(asset, version, tag)` - Remove tag
- `get_versions_by_tag(tag)` - Find by tag
- `freeze_version(asset, version, author)` - Freeze for safety
- `unfreeze_version(asset, version, author)` - Unfreeze
- `is_frozen(asset, version)` - Check frozen status
- `deprecate_version(asset, version, replacement, author)` - Mark deprecated
- `get_active_versions([asset])` - Get non-deprecated versions

### VersionComparison

- `compare_schemas(v1, v2)` - Schema differences
- `compare_data_ranges(v1, v2)` - Temporal range comparison
- `compare_metadata(v1, v2)` - Full comparison

### VersionValidator

- `is_complete(metadata)` - Check required fields
- `validate_integrity(metadata)` - Full validation

### CompatibilityChecker

- `check_version_match(version, spec)` - Match against spec
- `check_backward_compatibility(current, expected)` - Compatibility check
- `find_compatible_versions(versions, spec)` - Find matching versions

### VersionMigration

- `get_migration_path(from, to, versions)` - Find upgrade path
- `list_breaking_changes(v1, v2)` - Breaking changes list

## See Also

- [Data Storage Structure](./data_storage_structure.md) - Directory layout
- [Versioning Semantics](./versioning_semantics.md) - Detailed versioning rules
