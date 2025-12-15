# Feature Storage System

A versioned storage system for computed features that ensures reproducibility and supports efficient retrieval. Features are stored in Parquet format with comprehensive metadata tracking.

## Architecture Overview

### Directory Structure

Features are organized with semantic versioning and asset-based partitioning:

```
features/
├── v1.0.0/
│   ├── ES/
│   │   ├── data.parquet          # Feature data
│   │   ├── metadata.json         # Version metadata
│   │   └── data/                 # (optional) Partitioned data
│   │       ├── 2024/01/
│   │       ├── 2024/02/
│   │       └── ...
│   ├── MES/
│   ├── VIX/
│   └── ...
├── v1.1.0/
│   ├── ES/
│   └── ...
└── v2.0.0/
    └── ...
```

### Storage Format

- **Data Format**: Apache Parquet with columnar compression
- **Compression**: Snappy (configurable to gzip, zstd)
- **Schema**: Automatically tracked and validated
- **Metadata**: JSON format with complete computation parameters

## Core Components

### 1. Metadata Management (`metadata.py`)

Tracks all information needed to reproduce features:

```python
from src.features.storage import FeatureStorageMetadata, FeatureMetadataBuilder

# Build metadata
builder = FeatureMetadataBuilder(
    version="1.0.0",
    asset="ES",
    feature_set_name="technical_indicators",
    source_data_version="2.1.0"
)

metadata = (
    builder
    .set_date_range(start_date, end_date)
    .set_record_count(10000)
    .set_compression("snappy")
    .set_creator("data_team")
    .build()
)

# Add feature details
metadata.add_feature(FeatureComputationMetadata(
    feature_name="ma_20",
    description="20-day moving average",
    computation_parameters=[
        ComputationParameter("window", 20, "int", "Window size")
    ],
    dependencies=[
        FeatureDependency("close", is_computed=False, description="Raw close price")
    ],
    schema=[
        FeatureSchema("ma_20", "float64", nullable=False)
    ]
))

# Freeze version (immutable)
metadata.freeze(author="data_team")
```

**Key Metadata Fields**:
- `version`: Semantic version (e.g., "1.0.0")
- `asset`: Asset identifier (ES, MES, VIX)
- `feature_set_name`: Logical grouping of features
- `source_data_version`: Version of cleaned data used as source
- `creation_timestamp`: When version was created
- `record_count`: Number of records
- `features`: Detailed metadata for each computed feature
- `frozen`: Whether version is immutable
- `tags`: Version tags (e.g., "production", "deprecated")
- `changelog`: History of changes

### 2. Feature Writing (`writer.py`)

Writes computed features with automatic versioning:

```python
from src.features.storage import FeatureWriter, FeatureComputationMetadata

writer = FeatureWriter(
    base_path=Path("features"),
    compression="snappy",
    validation_enabled=True,
    auto_freeze_on_write=False
)

# Write computed features
version_dir = writer.write_features(
    features_df=computed_features,
    asset="ES",
    version="1.0.0",
    feature_set_name="technical_indicators",
    source_data_version="2.1.0",
    feature_metadata_list=[feature_meta_1, feature_meta_2],
    creator="data_pipeline",
    environment="production",
    tags=["stable"],
    partition_by_date=False  # Set to True for large datasets
)

# Freeze version after validation
writer.freeze_version("ES", "1.0.0", creator="data_team")

# Incrementally add data to existing version
writer.write_incremental(
    new_features_df=new_data,
    asset="ES",
    version="1.0.0",
    feature_metadata_list=feature_metadata,
    creator="data_pipeline"
)

# Promote version to production
writer.promote_version("ES", "1.0.0", creator="data_team")
```

**Features**:
- Automatic semantic versioning
- Data validation before writing
- Metadata generation and storage
- Version freezing for immutability
- Incremental updates support
- Partitioned writing for large datasets
- Version promotion workflow

### 3. Feature Reading (`reader.py`)

Efficient loading with flexible filtering:

```python
from src.features.storage import FeatureReader, FeatureCatalog
from datetime import datetime

reader = FeatureReader(
    base_path=Path("features"),
    cache_enabled=True,
    max_cache_size_mb=500
)

# Load all features for a version
df = reader.load_features(
    asset="ES",
    version="1.0.0"
)

# Load with date filtering
df = reader.load_features(
    asset="ES",
    version="1.0.0",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31)
)

# Load specific features only
df = reader.load_features(
    asset="ES",
    version="1.0.0",
    feature_names=["ma_20", "volatility"]
)

# Load from latest version
df, version = reader.load_latest_version(
    asset="ES",
    start_date=datetime(2024, 1, 1)
)

# Load multiple assets at once
features_dict = reader.load_multiple_assets(
    assets=["ES", "MES"],
    version="1.0.0"
)

# List available versions
versions = reader.list_versions(asset="ES", include_deprecated=False)
# Returns: {"ES": ["2.0.0", "1.1.0", "1.0.0"]}

# Get metadata
metadata = reader.get_metadata("ES", "1.0.0")

# Browse the feature catalog
catalog = FeatureCatalog(reader)
catalog_dict = catalog.get_catalog()
catalog.print_catalog()
```

**Key Features**:
- Date range filtering (uses Parquet column pruning)
- Feature subset selection
- Version discovery and querying
- Metadata inspection
- Multi-asset loading
- Built-in caching
- Catalog browsing

### 4. Version Management (`versioning.py`)

Freeze, compare, and analyze versions:

```python
from src.features.storage import (
    FeatureVersionManager,
    FeatureVersionComparator,
    FeatureVersionAnalyzer
)

# Version management
manager = FeatureVersionManager(base_path=Path("features"))

# Freeze/unfreeze versions
manager.freeze_version("ES", "1.0.0", author="data_team")
is_frozen = manager.is_frozen("ES", "1.0.0")  # True

# Add/remove tags
manager.add_tag("ES", "1.0.0", "production")
manager.remove_tag("ES", "1.0.0", "staging")

# Deprecate versions
manager.deprecate_version(
    "ES", 
    "1.0.0",
    replacement_version="2.0.0",
    author="data_team"
)

# Get changelog
changelog = manager.get_changelog("ES", "1.0.0")

# Compare versions
comparator = FeatureVersionComparator(base_path=Path("features"))

# Schema comparison
schema_diff = comparator.compare_schemas("ES", "1.0.0", "1.1.0")
# Returns: added_features, removed_features, modified_features, schema_compatible

# Parameter comparison
param_diff = comparator.compare_computation_parameters("ES", "1.0.0", "1.1.0")
# Returns: parameter_changes, parameters_compatible

# Dependency comparison
dep_diff = comparator.compare_dependencies("ES", "1.0.0", "1.1.0")
# Returns: dependency_changes, dependencies_compatible

# Complete comparison
full_comparison = comparator.compare_complete("ES", "1.0.0", "1.1.0")

# Analyze version history
analyzer = FeatureVersionAnalyzer(base_path=Path("features"))

# Get version history
history = analyzer.get_version_history("ES")
# Returns sorted list of versions with metadata

# Identify breaking changes
breaking = analyzer.get_breaking_changes("ES")
```

## Workflow Examples

### Example 1: Compute and Store Features

```python
from src.features.core import FeatureComputer
from src.features.storage import FeatureWriter, FeatureComputationMetadata
import pandas as pd

# Load cleaned data
cleaned_data = pd.read_parquet("cleaned/v1.2.0/ES/data.parquet")

# Compute features
computer = FeatureComputer(features=[...])
features_df = computer.compute(cleaned_data, asset="ES")

# Prepare metadata
feature_metadata = [
    FeatureComputationMetadata(
        feature_name="feature_name",
        description="...",
        computation_parameters=[...],
        dependencies=[...],
        schema=[...]
    )
    for feature in computed_features
]

# Write to storage
writer = FeatureWriter()
version_dir = writer.write_features(
    features_df=features_df,
    asset="ES",
    version="1.0.0",
    feature_set_name="technical_indicators",
    source_data_version="1.2.0",
    feature_metadata_list=feature_metadata,
    creator="feature_pipeline",
    tags=["v1.2.0-compat"]
)

# Freeze after validation
writer.freeze_version("ES", "1.0.0", creator="data_team")
writer.promote_version("ES", "1.0.0", creator="data_team")
```

### Example 2: Load Features for Backtesting

```python
from src.features.storage import FeatureReader

reader = FeatureReader()

# Load features for specific period
features = reader.load_features(
    asset="ES",
    version="2.0.0",  # Use specific version for reproducibility
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    feature_names=["ma_20", "volatility", "rsi"]
)

# features is a DataFrame ready for backtesting
```

### Example 3: Version Migration

```python
from src.features.storage import FeatureVersionComparator, FeatureVersionManager

comparator = FeatureVersionComparator()
manager = FeatureVersionManager()

# Check compatibility
comparison = comparator.compare_schemas("ES", "1.0.0", "2.0.0")

if comparison["schema_compatible"]:
    print("✓ Can migrate directly")
else:
    print("⚠ Breaking changes detected:")
    for removed in comparison["removed_features"]:
        print(f"  - Removed: {removed}")

# Deprecate old version
manager.deprecate_version(
    "ES", "1.0.0",
    replacement_version="2.0.0",
    author="data_team"
)
```

## Validation and Data Consistency

The storage system includes comprehensive validation:

### Automatic Checks
- **Timestamp validation**: No nulls, proper ordering
- **Feature validation**: NaN detection and reporting
- **Schema validation**: Column types and names
- **Duplicate detection**: Identifies duplicate timestamps
- **Size validation**: Tracks data volumes

### Manual Validation
```python
writer = FeatureWriter(validation_enabled=True)

# Validation runs automatically during write
# Warnings are logged for data quality issues
# Errors block writes if critical issues detected
```

## Performance Considerations

### Caching
```python
reader = FeatureReader(
    cache_enabled=True,
    max_cache_size_mb=500
)

# First load caches the data
df1 = reader.load_features("ES", "1.0.0")

# Subsequent loads use cache
df2 = reader.load_features("ES", "1.0.0", feature_names=["ma_20"])

# Clear cache when done
reader.clear_cache()
```

### Partitioned Storage
For large feature sets, use date-based partitioning:

```python
writer.write_features(
    features_df=large_df,
    asset="ES",
    version="1.0.0",
    ...,
    partition_by_date=True  # Creates features/v1.0.0/ES/data/2024/01/data.parquet, etc.
)
```

This enables:
- Efficient date range queries
- Reduced memory usage
- Parallel loading of partitions
- Better compression

## Best Practices

### 1. Version Management
- Use semantic versioning: MAJOR.MINOR.PATCH
- Increment MAJOR for breaking changes
- Increment MINOR for new features
- Increment PATCH for bug fixes

### 2. Metadata Completeness
- Always document computation parameters
- Track all dependencies
- Include schema information
- Add meaningful descriptions

### 3. Reproducibility
- Freeze versions after validation
- Store source data version
- Document creator and timestamp
- Maintain complete changelog

### 4. Organization
- Use consistent asset names
- Name feature sets logically
- Tag versions for easy discovery
- Deprecate old versions properly

### 5. Performance
- Enable caching for frequently accessed versions
- Use date partitioning for large datasets
- Load only needed features
- Clear cache periodically

## API Reference

### FeatureWriter
- `write_features()`: Write computed features
- `write_incremental()`: Add data to existing version
- `freeze_version()`: Make version immutable
- `promote_version()`: Tag version as production
- `promote_version()`: Tag as production

### FeatureReader
- `load_features()`: Load features with filtering
- `load_latest_version()`: Load from newest version
- `load_multiple_assets()`: Load multiple assets
- `get_metadata()`: Retrieve version metadata
- `list_versions()`: Discover available versions
- `list_features()`: Get feature names in version
- `get_feature_dependencies()`: Get feature dependencies
- `clear_cache()`: Clear in-memory cache

### FeatureVersionManager
- `freeze_version()`: Make immutable
- `unfreeze_version()`: Allow modifications
- `is_frozen()`: Check frozen status
- `add_tag()`, `remove_tag()`: Tag management
- `deprecate_version()`: Mark as deprecated
- `get_changelog()`: Retrieve version history

### FeatureVersionComparator
- `compare_schemas()`: Compare schemas
- `compare_computation_parameters()`: Compare parameters
- `compare_dependencies()`: Compare dependencies
- `compare_complete()`: Full version comparison

### FeatureVersionAnalyzer
- `get_version_history()`: Get version timeline
- `get_breaking_changes()`: Identify breaking changes
