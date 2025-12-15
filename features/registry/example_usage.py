"""
Example Usage of Feature Registry System.

This file demonstrates how to use the feature registry system with practical examples.
"""

from features.registry import (
    register_feature,
    get_registry,
    search,
    find_by_category,
    generate_documentation,
    track_feature_usage,
    track_validation,
    deprecated,
    DeprecationLevel,
    get_usage_tracker,
    initialize_registry,
    get_feature_info,
    generate_feature_catalog
)


# Example 1: Register a simple feature
@register_feature(
    name="simple_moving_average",
    description="Simple Moving Average indicator",
    computation_logic="Calculates the average of prices over a specified window",
    category="technical",
    assets=["stocks", "futures", "crypto"],
    tags={"technical", "trend", "moving_average"},
    author="Research Team",
    computation_cost="low",
    parameters={
        "window": {"type": "int", "default": 20, "required": False}
    },
    example_usage="""
    sma = SimpleMovingAverage(window=20)
    result = sma.compute(price_data)
    """
)
class SimpleMovingAverage:
    """Simple Moving Average technical indicator."""
    
    def __init__(self, window=20):
        self.window = window
    
    def compute(self, data):
        """Compute simple moving average."""
        return data.rolling(window=self.window).mean()


# Example 2: Register a complex feature with dependencies
@register_feature(
    name="bollinger_bands",
    description="Bollinger Bands volatility indicator",
    computation_logic="Calculates moving average with upper/lower bands based on standard deviation",
    category="volatility",
    assets=["stocks", "futures"],
    tags={"technical", "volatility", "bands"},
    author="Research Team",
    computation_cost="medium",
    data_dependencies=["price_data", "simple_moving_average"],
    parameters={
        "window": {"type": "int", "default": 20, "required": False},
        "num_std": {"type": "float", "default": 2.0, "required": False}
    },
    example_usage="""
    bb = BollingerBands(window=20, num_std=2.0)
    upper, middle, lower = bb.compute(price_data)
    """
)
class BollingerBands:
    """Bollinger Bands indicator."""
    
    def __init__(self, window=20, num_std=2.0):
        self.window = window
        self.num_std = num_std
    
    def compute(self, data):
        """Compute Bollinger Bands."""
        middle = data.rolling(window=self.window).mean()
        std = data.rolling(window=self.window).std()
        upper = middle + (std * self.num_std)
        lower = middle - (std * self.num_std)
        return upper, middle, lower


# Example 3: Deprecated feature with replacement
@deprecated(
    reason="Replaced with optimized implementation using vectorized operations",
    deprecated_in="1.5.0",
    remove_in="2.0.0",
    replacement="FastRSI",
    migration_guide="Replace SlowRSI() with FastRSI() - API is identical",
    level=DeprecationLevel.WARNING
)
@register_feature(
    name="slow_rsi",
    description="Legacy RSI implementation (DEPRECATED)",
    computation_logic="Original iterative RSI calculation",
    category="momentum",
    assets=["stocks"],
    tags={"technical", "momentum", "deprecated"}
)
class SlowRSI:
    """Deprecated RSI implementation."""
    
    def compute(self, data):
        """Compute RSI (legacy implementation)."""
        # Legacy implementation
        return data * 0.5  # Placeholder


# Example 4: Replacement feature
@register_feature(
    name="fast_rsi",
    description="Optimized RSI implementation",
    computation_logic="Vectorized RSI calculation with 10x performance improvement",
    category="momentum",
    assets=["stocks", "futures", "crypto"],
    tags={"technical", "momentum", "optimized"},
    author="Performance Team",
    computation_cost="low",
    current_version="1.5.0",
    parameters={
        "period": {"type": "int", "default": 14, "required": False}
    }
)
class FastRSI:
    """Optimized RSI implementation."""
    
    def __init__(self, period=14):
        self.period = period
    
    def compute(self, data):
        """Compute RSI (optimized implementation)."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def demonstrate_basic_usage():
    """Demonstrate basic registry operations."""
    print("=" * 60)
    print("BASIC REGISTRY USAGE")
    print("=" * 60)
    
    # Get registry instance
    registry = get_registry()
    
    # List all features
    print("\nAll registered features:")
    for feature in registry.list_all():
        print(f"  - {feature.name} (v{feature.current_version}) - {feature.category}")
    
    # Get specific feature
    feature = registry.get("simple_moving_average")
    print(f"\nFeature details: {feature.name}")
    print(f"  Description: {feature.description}")
    print(f"  Category: {feature.category}")
    print(f"  Assets: {', '.join(feature.assets)}")
    print(f"  Tags: {', '.join(feature.tags)}")
    
    # Registry statistics
    stats = registry.get_statistics()
    print(f"\nRegistry Statistics:")
    print(f"  Total features: {stats['total_features']}")
    print(f"  Active features: {stats['active_features']}")
    print(f"  Categories: {stats['categories']}")


def demonstrate_search():
    """Demonstrate search capabilities."""
    print("\n" + "=" * 60)
    print("SEARCH AND DISCOVERY")
    print("=" * 60)
    
    # Search by keyword
    print("\nSearch for 'moving':")
    results = search("moving")
    for feature in results:
        print(f"  - {feature.name}: {feature.description}")
    
    # Search with filters
    print("\nSearch for momentum features:")
    results = search("", category="momentum")
    for feature in results:
        print(f"  - {feature.name} (v{feature.current_version})")
    
    # Find by category
    print("\nTechnical indicators:")
    technical = find_by_category("technical")
    for feature in technical:
        print(f"  - {feature.name}")
    
    # Advanced search
    print("\nFeatures for stocks with low computation cost:")
    from features.registry import FeatureSearcher
    searcher = FeatureSearcher()
    
    def is_low_cost_stock_feature(f):
        return f.computation_cost == "low" and "stocks" in f.assets
    
    results = searcher.filter_by_criteria(predicate=is_low_cost_stock_feature)
    for feature in results:
        print(f"  - {feature.name}")


def demonstrate_documentation():
    """Demonstrate documentation generation."""
    print("\n" + "=" * 60)
    print("DOCUMENTATION GENERATION")
    print("=" * 60)
    
    # Generate markdown documentation
    print("\nGenerating markdown documentation...")
    doc = generate_documentation(
        output_path=None,  # Don't save, just return
        format="markdown",
        title="Feature Catalog Example",
        include_toc=True,
        group_by_category=True
    )
    print(f"Generated {len(doc)} characters of documentation")
    
    # Show a preview
    print("\nDocumentation preview (first 500 chars):")
    print(doc[:500])
    print("...")


def demonstrate_tracking():
    """Demonstrate usage tracking."""
    print("\n" + "=" * 60)
    print("USAGE TRACKING")
    print("=" * 60)
    
    # Track feature usage
    print("\nTracking feature usage...")
    track_feature_usage("simple_moving_average", context="backtest", user="analyst1")
    track_feature_usage("simple_moving_average", context="notebook", user="analyst2")
    track_feature_usage("bollinger_bands", context="backtest", user="analyst1")
    
    # Track validation
    print("Tracking validation results...")
    track_validation(
        "simple_moving_average",
        passed=True,
        coverage=0.99,
        quality=0.95,
        details={"null_values": 10}
    )
    track_validation(
        "bollinger_bands",
        passed=True,
        coverage=0.98,
        quality=0.92
    )
    
    # Get usage statistics
    tracker = get_usage_tracker()
    print("\nUsage statistics:")
    
    for feature_name in ["simple_moving_average", "bollinger_bands"]:
        count = tracker.get_usage_count(feature_name)
        by_context = tracker.get_usage_by_context(feature_name)
        print(f"\n  {feature_name}:")
        print(f"    Total usage: {count}")
        print(f"    By context: {by_context}")
        
        val_stats = tracker.get_validation_stats(feature_name)
        if val_stats:
            print(f"    Validations: {val_stats.total_validations}")
            print(f"    Success rate: {val_stats.success_rate:.1%}")
            print(f"    Quality score: {val_stats.quality_score:.2f}")
    
    # Most used features
    print("\nMost used features:")
    most_used = tracker.get_most_used(limit=5)
    for name, count in most_used:
        print(f"  - {name}: {count} uses")


def demonstrate_deprecation():
    """Demonstrate deprecation management."""
    print("\n" + "=" * 60)
    print("DEPRECATION MANAGEMENT")
    print("=" * 60)
    
    from features.registry import (
        check_feature_deprecation,
        get_deprecated_features,
        FeatureDeprecationManager
    )
    
    # Check deprecation
    print("\nChecking deprecation status:")
    deprecated_info = check_feature_deprecation("slow_rsi")
    if deprecated_info:
        print(f"  slow_rsi is deprecated:")
        print(f"    Reason: {deprecated_info.get('reason')}")
        print(f"    Replacement: {deprecated_info.get('replacement')}")
        print(f"    Remove in: {deprecated_info.get('remove_in')}")
    
    # List all deprecated features
    print("\nAll deprecated features:")
    deprecated = get_deprecated_features()
    for name, metadata in deprecated.items():
        info = metadata.deprecation_info
        print(f"  - {name} -> {info.get('replacement', 'N/A')}")
    
    # Generate deprecation report
    print("\nGenerating deprecation report...")
    from features.registry import generate_deprecation_report
    report = generate_deprecation_report()
    print("Report preview (first 300 chars):")
    print(report[:300])
    print("...")


def demonstrate_complete_workflow():
    """Demonstrate a complete workflow."""
    print("\n" + "=" * 60)
    print("COMPLETE WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # 1. Initialize with persistence
    print("\n1. Initializing registry...")
    # registry, tracker = initialize_registry(
    #     persistence_path="data/registry/",
    #     auto_save=True
    # )
    # Note: Commented out to avoid file creation in example
    
    # 2. Get feature information
    print("\n2. Getting feature information:")
    info = get_feature_info("simple_moving_average")
    print(f"  Name: {info['name']}")
    print(f"  Version: {info['version']}")
    print(f"  Category: {info['category']}")
    print(f"  Status: {info['status']}")
    print(f"  Usage count: {info['usage_count']}")
    
    # 3. Search for related features
    print("\n3. Finding related features:")
    from features.registry import FeatureSearcher
    searcher = FeatureSearcher()
    similar = searcher.find_similar("simple_moving_average", similarity_threshold=0.3)
    print(f"  Found {len(similar)} similar features")
    for feature in similar[:3]:
        print(f"    - {feature.name}")
    
    # 4. Generate documentation (would save to file normally)
    print("\n4. Documentation generation:")
    print("  Would generate: docs/feature_catalog.md")
    print("  Would generate: docs/feature_catalog.html")
    print("  Would generate: docs/feature_catalog.json")
    
    # 5. Get usage report
    print("\n5. Usage report:")
    report = get_usage_tracker().get_usage_report(period_days=30)
    print(f"  Total features: {report['total_features']}")
    print(f"  Used features: {report['used_features']}")
    print(f"  Total events: {report['total_events']}")


if __name__ == "__main__":
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("FEATURE REGISTRY SYSTEM - EXAMPLE USAGE")
    print("=" * 60)
    
    # Run demonstrations
    demonstrate_basic_usage()
    demonstrate_search()
    demonstrate_documentation()
    demonstrate_tracking()
    demonstrate_deprecation()
    demonstrate_complete_workflow()
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)
    print("\nFor more information, see the README.md file.")
