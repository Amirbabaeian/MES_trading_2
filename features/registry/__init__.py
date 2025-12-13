"""
Feature Registry and Documentation System.

This package provides a comprehensive system for registering, discovering,
documenting, and managing features across the research platform.

Main Components:
- catalog: Core registry for feature metadata and versioning
- search: Search and discovery interface
- documentation: Auto-documentation generation
- tracking: Usage tracking and statistics
- deprecation: Deprecation workflow management

Quick Start:
    from features.registry import register_feature, get_registry, search
    
    # Register a feature
    @register_feature(
        name="my_feature",
        description="My awesome feature",
        computation_logic="Computes something useful",
        category="indicators",
        assets=["stocks"]
    )
    class MyFeature:
        def compute(self, data):
            return data.mean()
    
    # Search for features
    results = search("momentum", category="indicators")
    
    # Generate documentation
    from features.registry import generate_documentation
    generate_documentation("docs/features.md", format="markdown")
"""

__version__ = "1.0.0"
__author__ = "Research Team"

# Core catalog components
from .catalog import (
    FeatureMetadata,
    FeatureRegistry,
    FeatureStatus,
    VersionInfo,
    register_feature,
    extract_metadata_from_class,
    get_registry
)

# Search and discovery
from .search import (
    FeatureSearcher,
    SearchQuery,
    SearchField,
    SortBy,
    search,
    find_by_category,
    find_by_asset,
    find_by_tag
)

# Documentation generation
from .documentation import (
    DocumentationGenerator,
    MarkdownFormatter,
    HTMLFormatter,
    JSONFormatter,
    generate_documentation
)

# Usage tracking
from .tracking import (
    UsageTracker,
    UsageEvent,
    ValidationStats,
    TrackedFeature,
    track_feature_usage,
    track_validation,
    get_usage_tracker
)

# Deprecation management
from .deprecation import (
    FeatureDeprecationManager,
    DeprecationLevel,
    DeprecationPolicy,
    DeprecationWarning,
    FeatureDeprecationError,
    deprecated,
    check_feature_deprecation,
    warn_if_deprecated,
    get_deprecated_features,
    generate_deprecation_report
)

# Public API
__all__ = [
    # Catalog
    'FeatureMetadata',
    'FeatureRegistry',
    'FeatureStatus',
    'VersionInfo',
    'register_feature',
    'extract_metadata_from_class',
    'get_registry',
    
    # Search
    'FeatureSearcher',
    'SearchQuery',
    'SearchField',
    'SortBy',
    'search',
    'find_by_category',
    'find_by_asset',
    'find_by_tag',
    
    # Documentation
    'DocumentationGenerator',
    'MarkdownFormatter',
    'HTMLFormatter',
    'JSONFormatter',
    'generate_documentation',
    
    # Tracking
    'UsageTracker',
    'UsageEvent',
    'ValidationStats',
    'TrackedFeature',
    'track_feature_usage',
    'track_validation',
    'get_usage_tracker',
    
    # Deprecation
    'FeatureDeprecationManager',
    'DeprecationLevel',
    'DeprecationPolicy',
    'DeprecationWarning',
    'FeatureDeprecationError',
    'deprecated',
    'check_feature_deprecation',
    'warn_if_deprecated',
    'get_deprecated_features',
    'generate_deprecation_report',
]


# Convenience functions for common workflows

def initialize_registry(persistence_path: str = None, auto_save: bool = False):
    """
    Initialize and configure the global registry and tracker.
    
    Args:
        persistence_path: Path for persisting registry data
        auto_save: Whether to auto-save on changes
    """
    registry = get_registry()
    tracker = get_usage_tracker()
    
    if persistence_path:
        import os
        base_path = persistence_path
        registry.configure(
            persistence_path=os.path.join(base_path, "registry.json"),
            auto_save=auto_save
        )
        tracker.configure(
            persistence_path=os.path.join(base_path, "usage.json"),
            auto_save=auto_save
        )
    
    return registry, tracker


def load_registry(registry_path: str = None, usage_path: str = None):
    """
    Load registry and usage data from files.
    
    Args:
        registry_path: Path to registry JSON file
        usage_path: Path to usage tracking JSON file
    """
    registry = get_registry()
    tracker = get_usage_tracker()
    
    if registry_path:
        registry.load(registry_path)
    
    if usage_path:
        tracker.load(usage_path)
    
    return registry, tracker


def save_registry(registry_path: str = None, usage_path: str = None):
    """
    Save registry and usage data to files.
    
    Args:
        registry_path: Path to save registry JSON
        usage_path: Path to save usage tracking JSON
    """
    registry = get_registry()
    tracker = get_usage_tracker()
    
    if registry_path:
        registry.save(registry_path)
    
    if usage_path:
        tracker.save(usage_path)


def get_feature_info(feature_name: str) -> dict:
    """
    Get comprehensive information about a feature.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Dictionary with feature metadata, usage stats, and deprecation info
    """
    registry = get_registry()
    tracker = get_usage_tracker()
    
    metadata = registry.get(feature_name)
    if not metadata:
        return {'error': f"Feature '{feature_name}' not found"}
    
    usage_count = tracker.get_usage_count(feature_name)
    usage_by_context = tracker.get_usage_by_context(feature_name)
    validation_stats = tracker.get_validation_stats(feature_name)
    deprecation_info = check_feature_deprecation(feature_name)
    
    return {
        'name': metadata.name,
        'description': metadata.description,
        'version': metadata.current_version,
        'category': metadata.category,
        'status': metadata.status.value,
        'assets': metadata.assets,
        'tags': list(metadata.tags),
        'usage_count': usage_count,
        'usage_by_context': usage_by_context,
        'validation_stats': validation_stats.to_dict() if validation_stats else None,
        'deprecation_info': deprecation_info,
        'created_at': metadata.created_at.isoformat(),
        'updated_at': metadata.updated_at.isoformat()
    }


def list_features(
    category: str = None,
    asset: str = None,
    status: FeatureStatus = None,
    include_deprecated: bool = False
) -> list:
    """
    List features with optional filtering.
    
    Args:
        category: Filter by category
        asset: Filter by asset
        status: Filter by status
        include_deprecated: Include deprecated features
        
    Returns:
        List of feature names
    """
    registry = get_registry()
    
    if category:
        features = registry.list_by_category(category)
    elif asset:
        features = registry.list_by_asset(asset)
    else:
        features = registry.list_all(include_deprecated=include_deprecated)
    
    if status:
        features = [f for f in features if f.status == status]
    
    return [f.name for f in features]


def search_features(keyword: str, **filters) -> list:
    """
    Search for features by keyword.
    
    Args:
        keyword: Search term
        **filters: Additional filters (category, assets, tags, etc.)
        
    Returns:
        List of matching feature names
    """
    results = search(keyword, **filters)
    return [f.name for f in results]


def generate_feature_catalog(
    output_dir: str,
    formats: list = None,
    include_deprecated: bool = False
):
    """
    Generate complete feature catalog documentation.
    
    Args:
        output_dir: Directory to save documentation
        formats: List of formats ('markdown', 'html', 'json')
        include_deprecated: Include deprecated features
    """
    generator = DocumentationGenerator()
    generator.generate_catalog(
        output_dir=output_dir,
        formats=formats or ['markdown', 'html', 'json'],
        include_deprecated=include_deprecated
    )


def get_usage_report(days: int = 30) -> dict:
    """
    Get usage statistics report.
    
    Args:
        days: Number of days to include in report
        
    Returns:
        Dictionary with usage statistics
    """
    tracker = get_usage_tracker()
    return tracker.get_usage_report(period_days=days)


def find_unused_features() -> list:
    """
    Find features that have never been used.
    
    Returns:
        List of unused feature names
    """
    tracker = get_usage_tracker()
    return tracker.get_unused_features()


def get_registry_stats() -> dict:
    """
    Get overall registry statistics.
    
    Returns:
        Dictionary with registry statistics
    """
    registry = get_registry()
    return registry.get_statistics()
