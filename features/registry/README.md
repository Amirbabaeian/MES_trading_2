# Feature Registry and Documentation System

A comprehensive registry and documentation system for managing features across the research platform. This system provides feature discovery, automatic documentation generation, usage tracking, and deprecation management.

## Overview

The Feature Registry system consists of five main components:

1. **Catalog (`catalog.py`)** - Core registry for feature metadata and versioning
2. **Search (`search.py`)** - Search and discovery interface with advanced filtering
3. **Documentation (`documentation.py`)** - Auto-generation of feature documentation
4. **Tracking (`tracking.py`)** - Usage tracking and validation statistics
5. **Deprecation (`deprecation.py`)** - Lifecycle management and deprecation workflow

## Quick Start

### Basic Registration

```python
from features.registry import register_feature

@register_feature(
    name="momentum_10d",
    description="10-day momentum indicator",
    computation_logic="Calculates returns over trailing 10 days",
    category="momentum",
    assets=["stocks", "futures"],
    tags={"technical", "momentum", "short-term"},
    author="Research Team",
    computation_cost="low",
    example_usage="""
    feature = Momentum10D()
    result = feature.compute(price_data)
    """
)
class Momentum10D:
    """10-day momentum technical indicator."""
    
    def compute(self, data):
        """Compute 10-day momentum."""
        return data.pct_change(10)
```

### Searching for Features

```python
from features.registry import search, find_by_category, find_by_asset

# Search by keyword
results = search("momentum")

# Search with filters
results = search("volatility", category="risk", assets=["stocks"])

# Find by category
momentum_features = find_by_category("momentum")

# Find by asset
stock_features = find_by_asset("stocks")

# Advanced search
from features.registry import FeatureSearcher, SearchQuery, SearchField

searcher = FeatureSearcher()
query = SearchQuery(
    keyword="vol",
    fields=[SearchField.NAME, SearchField.DESCRIPTION],
    category="risk",
    exclude_deprecated=True
)
results = searcher.execute_query(query)
```

### Generating Documentation

```python
from features.registry import generate_documentation, DocumentationGenerator

# Generate markdown documentation
generate_documentation("docs/features.md", format="markdown")

# Generate HTML documentation
generate_documentation("docs/features.html", format="html")

# Generate complete catalog in all formats
from features.registry import generate_feature_catalog
generate_feature_catalog("docs/", formats=["markdown", "html", "json"])

# Generate category-specific docs
generator = DocumentationGenerator()
generator.generate_category_docs("docs/categories/", format="markdown")
```

### Usage Tracking

```python
from features.registry import track_feature_usage, track_validation

# Track feature usage
track_feature_usage("momentum_10d", context="backtest", user="analyst1")

# Track validation results
track_validation(
    "momentum_10d",
    passed=True,
    coverage=0.98,
    quality=0.95,
    details={"null_values": 2, "outliers": 5}
)

# Get usage statistics
from features.registry import get_usage_tracker
tracker = get_usage_tracker()
usage_count = tracker.get_usage_count("momentum_10d")
stats = tracker.get_usage_report(period_days=30)

# Find unused features
unused = tracker.get_unused_features()
```

### Deprecation Management

```python
from features.registry import deprecated, DeprecationLevel

# Mark feature as deprecated
@deprecated(
    reason="Replaced by more efficient implementation",
    deprecated_in="2.0.0",
    remove_in="3.0.0",
    replacement="FastMomentum10D",
    migration_guide="Replace Momentum10D() with FastMomentum10D()",
    level=DeprecationLevel.WARNING
)
class Momentum10D:
    def compute(self, data):
        return data.pct_change(10)

# Manual deprecation
from features.registry import FeatureDeprecationManager
manager = FeatureDeprecationManager()
manager.deprecate_feature(
    feature_name="old_feature",
    reason="Outdated algorithm",
    deprecated_in="2.0.0",
    remove_in="3.0.0",
    replacement="new_feature"
)

# Generate deprecation report
from features.registry import generate_deprecation_report
report = generate_deprecation_report()
print(report)
```

## Detailed Usage

### Feature Metadata

Each feature registered in the system includes comprehensive metadata:

```python
from features.registry import FeatureMetadata, VersionInfo

metadata = FeatureMetadata(
    name="my_feature",
    description="Description of what the feature computes",
    computation_logic="High-level summary of computation approach",
    parameters={
        "window": {"type": "int", "default": 10, "required": False},
        "method": {"type": "str", "default": "mean", "required": True}
    },
    data_dependencies=["price_data", "volume_data"],
    current_version="1.0.0",
    category="technical",
    assets=["stocks", "futures"],
    tags={"momentum", "technical"},
    author="Research Team",
    computation_cost="medium",  # low/medium/high
    example_usage="feature = MyFeature(window=20)"
)
```

### Registry Operations

```python
from features.registry import get_registry

registry = get_registry()

# Manual registration
registry.register(
    name="custom_feature",
    description="A custom feature",
    computation_logic="Does something useful",
    category="custom",
    current_version="1.0.0"
)

# Update feature metadata
registry.update("custom_feature", category="updated_category")

# Get feature metadata
metadata = registry.get("custom_feature")

# List all features
all_features = registry.list_all(include_deprecated=False)

# Check if feature exists
exists = registry.exists("custom_feature")

# Remove feature
registry.remove("custom_feature")

# Get statistics
stats = registry.get_statistics()
```

### Versioning

```python
from features.registry import get_registry

registry = get_registry()

# Add a new version
metadata = registry.get("my_feature")
metadata.add_version(
    version="2.0.0",
    changes="Improved performance by 50%",
    author="Research Team"
)

# Get specific version
v1_metadata = registry.get("my_feature", version="1.0.0")
v2_metadata = registry.get("my_feature", version="2.0.0")
```

### Persistence

```python
from features.registry import initialize_registry, save_registry, load_registry

# Initialize with persistence
registry, tracker = initialize_registry(
    persistence_path="data/registry/",
    auto_save=True
)

# Manual save
save_registry(
    registry_path="data/registry/registry.json",
    usage_path="data/registry/usage.json"
)

# Load from files
load_registry(
    registry_path="data/registry/registry.json",
    usage_path="data/registry/usage.json"
)
```

### Advanced Search

```python
from features.registry import FeatureSearcher

searcher = FeatureSearcher()

# Find similar features
similar = searcher.find_similar("momentum_10d", similarity_threshold=0.5)

# Custom filtering
def is_low_cost(feature):
    return feature.computation_cost == "low"

low_cost_features = searcher.filter_by_criteria(predicate=is_low_cost)

# Sort results
from features.registry import SortBy
sorted_features = searcher.sort_results(
    features,
    sort_by=SortBy.UPDATED_AT,
    reverse=True
)

# Get recommendations
recommendations = searcher.get_recommendations(
    category="momentum",
    asset="stocks",
    limit=5
)
```

### Usage Tracking with Hooks

```python
from features.registry import get_usage_tracker

tracker = get_usage_tracker()

# Add custom hook
def log_usage(event):
    print(f"Feature {event.feature_name} used in {event.context}")

tracker.add_hook(log_usage)

# Automatic tracking decorator
from features.registry import TrackedFeature

@TrackedFeature(context="backtest", track_calls=True)
class MyFeature:
    def compute(self, data):
        return data.mean()
```

### Validation Statistics

```python
from features.registry import get_usage_tracker

tracker = get_usage_tracker()

# Record validation
tracker.track_validation(
    feature_name="momentum_10d",
    passed=True,
    coverage=0.99,
    quality=0.95,
    details={
        "null_count": 10,
        "outlier_count": 5,
        "validation_time": 2.5
    }
)

# Get validation stats
stats = tracker.get_validation_stats("momentum_10d")
print(f"Success rate: {stats.success_rate:.2%}")
print(f"Average quality: {stats.quality_score:.2f}")
print(f"Last validated: {stats.last_validated}")
```

## API Reference

### Catalog Module

- `FeatureRegistry` - Central registry singleton
- `FeatureMetadata` - Feature metadata container
- `FeatureStatus` - Enum for feature status (ACTIVE, DEPRECATED, EXPERIMENTAL, ARCHIVED)
- `VersionInfo` - Version information container
- `register_feature()` - Decorator for automatic registration
- `get_registry()` - Get global registry instance

### Search Module

- `FeatureSearcher` - Search and discovery interface
- `SearchQuery` - Structured search query
- `SearchField` - Enum for searchable fields
- `SortBy` - Enum for sorting options
- `search()` - Quick search function
- `find_by_category()` - Find by category
- `find_by_asset()` - Find by asset
- `find_by_tag()` - Find by tag

### Documentation Module

- `DocumentationGenerator` - Main documentation generator
- `MarkdownFormatter` - Markdown format generator
- `HTMLFormatter` - HTML format generator
- `JSONFormatter` - JSON format generator
- `generate_documentation()` - Generate docs in specified format

### Tracking Module

- `UsageTracker` - Usage tracking singleton
- `UsageEvent` - Usage event record
- `ValidationStats` - Validation statistics
- `TrackedFeature` - Decorator for automatic tracking
- `track_feature_usage()` - Track feature usage
- `track_validation()` - Track validation results
- `get_usage_tracker()` - Get global tracker instance

### Deprecation Module

- `FeatureDeprecationManager` - Deprecation manager
- `DeprecationLevel` - Enum for deprecation severity
- `DeprecationPolicy` - Enum for deprecation policies
- `deprecated()` - Deprecation decorator
- `check_feature_deprecation()` - Check deprecation status
- `warn_if_deprecated()` - Issue deprecation warning
- `get_deprecated_features()` - Get all deprecated features
- `generate_deprecation_report()` - Generate deprecation report

## Convenience Functions

The `__init__.py` module provides high-level convenience functions:

```python
from features.registry import (
    initialize_registry,
    load_registry,
    save_registry,
    get_feature_info,
    list_features,
    search_features,
    generate_feature_catalog,
    get_usage_report,
    find_unused_features,
    get_registry_stats
)
```

## Examples

### Complete Workflow Example

```python
from features.registry import (
    register_feature,
    search,
    generate_feature_catalog,
    track_feature_usage,
    get_usage_report,
    initialize_registry
)

# 1. Initialize registry with persistence
registry, tracker = initialize_registry(
    persistence_path="data/registry/",
    auto_save=True
)

# 2. Register features
@register_feature(
    name="rsi_14",
    description="14-period Relative Strength Index",
    computation_logic="Momentum oscillator measuring speed and magnitude of price changes",
    category="momentum",
    assets=["stocks", "futures", "crypto"],
    tags={"technical", "oscillator", "momentum"},
    author="Technical Analysis Team",
    computation_cost="low",
    parameters={
        "period": {"type": "int", "default": 14, "required": False}
    }
)
class RSI14:
    def __init__(self, period=14):
        self.period = period
    
    def compute(self, data):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# 3. Use the feature
rsi = RSI14()
result = rsi.compute(price_data)

# Track usage
track_feature_usage("rsi_14", context="backtest", user="analyst1")

# 4. Search for related features
momentum_features = search("momentum", category="momentum")
print(f"Found {len(momentum_features)} momentum features")

# 5. Generate documentation
generate_feature_catalog("docs/", formats=["markdown", "html"])

# 6. Get usage report
report = get_usage_report(days=30)
print(f"Total features used: {report['used_features']}")
print(f"Total events: {report['total_events']}")
```

### Deprecation Workflow Example

```python
from features.registry import (
    register_feature,
    deprecated,
    FeatureDeprecationManager,
    generate_deprecation_report
)

# Old feature (deprecated)
@deprecated(
    reason="Inefficient implementation, replaced with vectorized version",
    deprecated_in="2.0.0",
    remove_in="3.0.0",
    replacement="FastRSI",
    migration_guide="Replace RSI(period=14) with FastRSI(period=14)"
)
@register_feature(
    name="slow_rsi",
    description="Original RSI implementation (DEPRECATED)",
    computation_logic="Legacy RSI calculation",
    category="momentum"
)
class SlowRSI:
    def compute(self, data):
        return data  # Old implementation

# New feature (replacement)
@register_feature(
    name="fast_rsi",
    description="Optimized RSI implementation",
    computation_logic="Vectorized RSI calculation with 10x performance improvement",
    category="momentum",
    current_version="2.0.0"
)
class FastRSI:
    def compute(self, data):
        return data  # New implementation

# Generate deprecation report
report = generate_deprecation_report()
with open("docs/deprecation_report.md", "w") as f:
    f.write(report)
```

## Best Practices

1. **Always provide comprehensive metadata** when registering features
2. **Use meaningful categories and tags** for better discoverability
3. **Include example usage** in feature registration
4. **Track feature usage** in production workflows
5. **Document breaking changes** when adding new versions
6. **Follow semantic versioning** for feature versions
7. **Provide migration guides** when deprecating features
8. **Generate documentation regularly** to keep it up-to-date
9. **Monitor usage statistics** to identify underutilized features
10. **Clean old tracking data** periodically to manage storage

## Architecture

### Registry Storage

The registry uses an in-memory storage with optional JSON persistence:

- Features are stored in a dictionary keyed by name
- Version map maintains all versions of each feature
- Indices are maintained for efficient lookup by category, asset, and tag
- Thread-safe singleton pattern ensures consistent state

### Search Implementation

Search uses multiple strategies:
- Keyword matching (case-sensitive or insensitive)
- Regular expression support
- Multi-field search
- Filtering by metadata attributes
- Similarity scoring based on shared attributes

### Documentation Generation

Documentation is generated by:
1. Extracting metadata from registry
2. Formatting according to chosen output format
3. Grouping by category or other criteria
4. Including table of contents and navigation
5. Adding status badges and deprecation warnings

### Usage Tracking

Usage tracking records:
- Feature access events with timestamps
- Context information (backtest, notebook, production)
- Validation results with coverage and quality metrics
- Configurable hooks for custom actions

### Deprecation Management

Deprecation system provides:
- Multiple severity levels (WARNING, STRICT, PENDING_REMOVAL, REMOVED)
- Configurable policies (WARN, PREVENT_NEW, BLOCK)
- Automatic warning generation
- Migration guidance
- Timeline management

## Thread Safety

All major components use thread-safe singleton patterns:
- `FeatureRegistry` uses locks for concurrent access
- `UsageTracker` uses locks for event recording
- All operations are atomic at the feature level

## Performance Considerations

- Registry operations are O(1) for lookup by name
- Search operations scan all features but are optimized with indices
- Usage tracking is lightweight with optional async hooks
- Documentation generation is done on-demand, not on every change

## License

Copyright (c) 2024 Research Team. All rights reserved.
