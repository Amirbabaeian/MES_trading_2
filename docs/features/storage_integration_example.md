# Feature Storage Integration Example

Complete end-to-end example of computing and storing features with reproducibility.

## Scenario: Computing Technical Indicators for ES

We'll compute technical indicators from cleaned market data, store them with metadata, and later reproduce the exact same results.

## Step 1: Prepare Data

```python
import pandas as pd
from pathlib import Path

# Load cleaned data
cleaned_data = pd.read_parquet("cleaned/v1.2.0/ES/data.parquet")
print(f"Loaded {len(cleaned_data)} records from 2024-01-01 to 2024-03-31")
```

## Step 2: Define Features

```python
from src.features.core import Feature
import pandas as pd
import numpy as np

class SimpleMovingAverage(Feature):
    """Compute simple moving average."""
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"sma_{window}",
            description=f"{window}-day simple moving average",
            parameters={"window": window},
            dependencies=["close"]
        )
        self.window = window
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Compute SMA."""
        return data['close'].rolling(window=self.window).mean()


class Volatility(Feature):
    """Compute rolling volatility."""
    
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"volatility_{window}",
            description=f"{window}-day rolling volatility",
            parameters={"window": window},
            dependencies=["returns"]
        )
        self.window = window
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Compute rolling volatility."""
        returns = data['close'].pct_change()
        return returns.rolling(window=self.window).std() * np.sqrt(252)


class RSI(Feature):
    """Compute Relative Strength Index."""
    
    def __init__(self, window: int = 14):
        super().__init__(
            name=f"rsi_{window}",
            description=f"{window}-period RSI",
            parameters={"window": window},
            dependencies=["close"]
        )
        self.window = window
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Compute RSI."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
```

## Step 3: Compute Features

```python
from src.features.core import FeatureComputer

# Create feature computer
computer = FeatureComputer()

# Add features
computer.add_features([
    SimpleMovingAverage(window=20),
    SimpleMovingAverage(window=50),
    Volatility(window=20),
    RSI(window=14),
])

# Compute all features
features_df = computer.compute(
    cleaned_data,
    asset="ES",
    start_date=None,  # Use all data
    end_date=None
)

print(f"Computed {len(features_df.columns)-1} features for {len(features_df)} records")
print(features_df.head())
```

## Step 4: Prepare Metadata

```python
from src.features.storage import (
    FeatureComputationMetadata,
    ComputationParameter,
    FeatureDependency,
    FeatureSchema,
)

# Metadata for SMA 20
sma_20_meta = FeatureComputationMetadata(
    feature_name="sma_20",
    description="20-day simple moving average of close price",
    computation_parameters=[
        ComputationParameter(
            name="window",
            value=20,
            data_type="int",
            description="Number of periods for the moving average"
        )
    ],
    dependencies=[
        FeatureDependency(
            feature_name="close",
            is_computed=False,
            description="Raw close price from market data"
        )
    ],
    schema=[
        FeatureSchema(
            column_name="sma_20",
            data_type="float64",
            nullable=False,
            description="20-period simple moving average value"
        )
    ]
)

# Metadata for Volatility
volatility_meta = FeatureComputationMetadata(
    feature_name="volatility_20",
    description="20-day rolling volatility (annualized)",
    computation_parameters=[
        ComputationParameter(
            name="window",
            value=20,
            data_type="int",
            description="Rolling window in days"
        ),
        ComputationParameter(
            name="annualization_factor",
            value=252,
            data_type="int",
            description="Trading days per year"
        )
    ],
    dependencies=[
        FeatureDependency(
            feature_name="returns",
            is_computed=True,
            description="Daily returns computed from close prices"
        )
    ],
    schema=[
        FeatureSchema(
            column_name="volatility_20",
            data_type="float64",
            nullable=False,
            description="Annualized rolling volatility"
        )
    ]
)

# Metadata for RSI
rsi_meta = FeatureComputationMetadata(
    feature_name="rsi_14",
    description="14-period Relative Strength Index",
    computation_parameters=[
        ComputationParameter(
            name="window",
            value=14,
            data_type="int",
            description="RSI period"
        )
    ],
    dependencies=[
        FeatureDependency(
            feature_name="close",
            is_computed=False,
            description="Raw close price"
        )
    ],
    schema=[
        FeatureSchema(
            column_name="rsi_14",
            data_type="float64",
            nullable=False,
            description="RSI value (0-100)"
        )
    ]
)

feature_metadata = [sma_20_meta, volatility_meta, rsi_meta]
```

## Step 5: Write Features to Storage

```python
from src.features.storage import FeatureWriter
from pathlib import Path

# Create writer
writer = FeatureWriter(
    base_path=Path("features"),
    compression="snappy",
    validation_enabled=True
)

# Write features
version_dir = writer.write_features(
    features_df=features_df,
    asset="ES",
    version="1.0.0",
    feature_set_name="technical_indicators",
    source_data_version="1.2.0",  # Must match cleaned data version
    feature_metadata_list=feature_metadata,
    creator="feature_engineering_team",
    environment="production",
    tags=["v1.2.0-compatible", "stable"],
    partition_by_date=False  # Single file for this dataset size
)

print(f"✓ Wrote features to {version_dir}")
```

## Step 6: Freeze Version

```python
# Freeze to prevent accidental modifications
writer.freeze_version("ES", "1.0.0", creator="data_team")
writer.promote_version("ES", "1.0.0", creator="data_team")

print("✓ Frozen and promoted version 1.0.0")
```

## Step 7: Load and Verify

```python
from src.features.storage import FeatureReader

# Create reader
reader = FeatureReader(cache_enabled=True)

# Load all features
loaded_features = reader.load_features("ES", "1.0.0")

print(f"Loaded {len(loaded_features)} records with columns:")
print(loaded_features.columns.tolist())

# Verify data matches
assert len(loaded_features) == len(features_df)
assert set(loaded_features.columns) == set(features_df.columns)
print("✓ Data integrity verified")
```

## Step 8: Load with Filters

```python
from datetime import datetime

# Load specific date range
march_data = reader.load_features(
    asset="ES",
    version="1.0.0",
    start_date=datetime(2024, 3, 1),
    end_date=datetime(2024, 3, 31)
)

print(f"Loaded {len(march_data)} records for March 2024")

# Load specific features only
sma_data = reader.load_features(
    asset="ES",
    version="1.0.0",
    feature_names=["sma_20", "sma_50"]
)

print(f"Loaded SMA data with columns: {sma_data.columns.tolist()}")
```

## Step 9: Inspect Metadata

```python
# Get metadata
metadata = reader.get_metadata("ES", "1.0.0")

print(f"Version: {metadata.version}")
print(f"Asset: {metadata.asset}")
print(f"Created: {metadata.creation_timestamp}")
print(f"Records: {metadata.record_count}")
print(f"Source data: v{metadata.source_data_version}")
print(f"Frozen: {metadata.frozen}")
print(f"Features: {[f.feature_name for f in metadata.features]}")

# Get feature details
for feature in metadata.features:
    print(f"\n{feature.feature_name}:")
    print(f"  Description: {feature.description}")
    print(f"  Parameters:")
    for param in feature.computation_parameters:
        print(f"    - {param.name} = {param.value} ({param.data_type})")
    print(f"  Dependencies:")
    for dep in feature.dependencies:
        print(f"    - {dep.feature_name} (computed={dep.is_computed})")
```

## Step 10: Version Management

```python
from src.features.storage import (
    FeatureVersionManager,
    FeatureVersionComparator,
    FeatureCatalog
)

# Check version status
manager = FeatureVersionManager()
is_frozen = manager.is_frozen("ES", "1.0.0")
print(f"Version frozen: {is_frozen}")

# Browse catalog
catalog = FeatureCatalog(reader)
catalog_dict = catalog.get_catalog()

print("\nFeature Catalog:")
for asset, versions in catalog_dict.items():
    print(f"\n{asset}:")
    for version, info in versions.items():
        print(f"  v{version}: {info['features']} features ({info['record_count']} records)")
        if info['tags']:
            print(f"    Tags: {', '.join(info['tags'])}")
```

## Step 11: Version Comparison

```python
# If we had a second version, we could compare
comparator = FeatureVersionComparator()

# This would show schema differences
# comparison = comparator.compare_schemas("ES", "1.0.0", "1.1.0")

# For now, let's create a second version with a small change
# and then compare

# Simulate creating v1.1.0 with additional feature
features_df_v2 = features_df.copy()
features_df_v2['macd'] = (
    features_df_v2['sma_20'] - features_df_v2['sma_50']
)

macd_meta = FeatureComputationMetadata(
    feature_name="macd",
    description="MACD indicator (SMA20 - SMA50)",
    computation_parameters=[],
    dependencies=[
        FeatureDependency("sma_20", True),
        FeatureDependency("sma_50", True),
    ],
    schema=[
        FeatureSchema("macd", "float64", False, "MACD value")
    ]
)

writer.write_features(
    features_df=features_df_v2,
    asset="ES",
    version="1.1.0",
    feature_set_name="technical_indicators",
    source_data_version="1.2.0",
    feature_metadata_list=feature_metadata + [macd_meta],
    creator="feature_engineering_team",
    tags=["v1.2.0-compatible"],
)

# Now compare versions
comparison = comparator.compare_schemas("ES", "1.0.0", "1.1.0")

print("\nSchema Comparison (v1.0.0 → v1.1.0):")
print(f"Added features: {comparison['added_features']}")
print(f"Removed features: {comparison['removed_features']}")
print(f"Schema compatible: {comparison['schema_compatible']}")
```

## Step 12: For Backtesting

```python
# Load latest version for backtesting
backtest_features, version_used = reader.load_latest_version(
    asset="ES",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31)
)

print(f"Backtesting with features version: {version_used}")
print(f"Data shape: {backtest_features.shape}")

# Your backtesting logic here...
# backtest_engine.run(backtest_features)
```

## Key Takeaways

1. **Versioning**: Each computation gets a unique version (1.0.0, 1.1.0, etc.)
2. **Reproducibility**: All parameters are stored; future computations with same params produce same results
3. **Immutability**: Frozen versions cannot be modified, ensuring data integrity
4. **Metadata**: Complete tracking of dependencies and computation logic
5. **Efficiency**: Parquet format with optional date partitioning
6. **Discovery**: Easy browsing of available features across versions and assets

## Directory Structure After Example

```
features/
└── v1.0.0/
    ├── ES/
    │   ├── data.parquet          (100+ KB with compressed features)
    │   └── metadata.json         (5-10 KB with complete metadata)
    └── v1.1.0/
        └── ES/
            ├── data.parquet
            └── metadata.json
```

## Notes

- All versions are stored independently, enabling comparisons
- Metadata includes exact computation parameters for reproducibility
- The source data version (1.2.0) links to the specific cleaned data version
- Tags help organize versions (stable, production, deprecated, etc.)
- The frozen flag prevents accidental modifications
- Both single-file and partitioned storage are supported

This workflow ensures complete reproducibility and traceability of computed features!
