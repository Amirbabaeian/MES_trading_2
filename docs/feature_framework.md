# Feature Computation Framework

## Overview

The Feature Computation Framework provides an abstract, extensible, and deterministic system for computing trading features from market data. It enables researchers and traders to define features once and apply them consistently across different assets and time periods.

**Key Design Principles:**
- **Deterministic**: All computations are reproducible with no randomness or time-dependent logic
- **Extensible**: Add new features by subclassing without modifying the framework
- **Parameterizable**: Features support configurable parameters (window sizes, thresholds, etc.)
- **Dependency-Aware**: Automatic resolution of feature dependencies with topological sorting
- **Batch-Efficient**: Process multiple date ranges and large datasets efficiently
- **Incremental-Update Capable**: Compute only new/missing feature values for efficiency
- **Parquet-Integrated**: Seamless read/write with Parquet-based data storage

## Architecture

### Core Components

#### 1. **Feature** (Abstract Base Class)
Represents a single computed quantity derived from market data.

```python
class Feature(ABC):
    """Base class for all features."""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
    ):
        # Feature metadata
        self.name = name  # Unique identifier
        self.description = description  # Human-readable description
        self.parameters = parameters or {}  # Configuration parameters
        self.dependencies = dependencies or []  # Feature dependencies
    
    def validate_parameters(self) -> None:
        """Override to validate parameters. Raises ParameterValidationError if invalid."""
        pass
    
    @abstractmethod
    def compute(self, data: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Compute the feature from input data."""
        pass
```

**Key Properties:**
- `name`: Unique identifier used to reference the feature
- `description`: What the feature represents
- `parameters`: Configuration dict (e.g., `{"window": 20}`)
- `dependencies`: List of other feature names this feature requires
- `compute()`: Returns Series or DataFrame with computed values

**Example:**
```python
class SimpleReturn(Feature):
    def __init__(self):
        super().__init__(
            name="simple_return",
            description="Daily simple return (log-based)",
            parameters={},
            dependencies=[],
        )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        returns = np.log(data['close'] / data['close'].shift(1))
        return returns
```

#### 2. **FeatureSet** (Container)
Groups related features together for bulk management.

```python
class FeatureSet:
    """Container for a collection of related features."""
    
    def __init__(
        self,
        name: str,
        description: str,
        features: Optional[List[Feature]] = None,
        version: str = "1.0.0",
        asset: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.features: Dict[str, Feature] = {}
        self.version = version
        self.asset = asset
    
    def add_feature(self, feature: Feature) -> None:
        """Add a feature to the set."""
        pass
    
    def get_feature(self, feature_name: str) -> Feature:
        """Retrieve a feature by name."""
        pass
    
    def list_features(self) -> List[str]:
        """Get all feature names in the set."""
        pass
```

**Use Cases:**
- Group volatility features together
- Create asset-specific feature sets (ES features vs VIX features)
- Version feature collections for reproducibility

#### 3. **FeatureComputer** (Orchestration Engine)
Orchestrates feature computation with dependency resolution.

```python
class FeatureComputer:
    """Engine for computing features with dependency resolution."""
    
    def __init__(
        self,
        features: Optional[List[Feature]] = None,
        cache_intermediate: bool = True,
        strict_dependencies: bool = True,
    ):
        self.features: Dict[str, Feature] = {}
        self.cache_intermediate = cache_intermediate
        self.strict_dependencies = strict_dependencies
    
    def add_feature(self, feature: Feature) -> None:
        """Register a feature."""
        pass
    
    def get_computation_order(self) -> List[str]:
        """Get topological order for feature computation."""
        pass
    
    def compute(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None,
        asset: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Compute features for the given data."""
        pass
    
    def compute_batch(
        self,
        data: pd.DataFrame,
        date_ranges: List[Tuple[datetime, datetime]],
        features: Optional[List[str]] = None,
        asset: Optional[str] = None,
    ) -> BatchComputationResult:
        """Compute features across multiple date ranges."""
        pass
    
    def compute_incremental(
        self,
        data: pd.DataFrame,
        existing_features: pd.DataFrame,
        features: Optional[List[str]] = None,
        asset: Optional[str] = None,
    ) -> pd.DataFrame:
        """Incrementally update features with new data."""
        pass
```

**Key Responsibilities:**
- Register and manage features
- Resolve dependencies via topological sorting
- Orchestrate feature computation in correct order
- Provide caching for intermediate results
- Support batch and incremental computation modes
- Handle date filtering and data management

#### 4. **Dependency Graph** (Resolution Logic)
Manages feature dependencies and provides topological sorting.

```python
class DependencyGraph:
    """Manages dependencies and provides topological sort."""
    
    def add_feature(self, feature_name: str, dependencies: List[str]) -> None:
        """Add a feature and its dependencies."""
        pass
    
    def detect_cycles(self) -> List[List[str]]:
        """Find circular dependencies."""
        pass
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate the dependency graph."""
        pass
    
    def topological_sort(self) -> List[str]:
        """Compute computation order (dependencies first)."""
        pass
```

## Usage Guide

### Basic Usage

#### 1. Define Custom Features

```python
from src.features.core.base import Feature
from src.features.core.errors import ComputationError, ParameterValidationError
import pandas as pd
import numpy as np

class CustomVolatility(Feature):
    """Custom volatility with configurable window and annualization."""
    
    def __init__(self, window: int = 20, annualize: bool = True):
        super().__init__(
            name=f"custom_vol_{window}",
            description=f"Custom volatility (window={window})",
            parameters={"window": window, "annualize": annualize},
            dependencies=[],
        )
    
    def validate_parameters(self) -> None:
        """Validate parameters."""
        window = self.parameters.get("window")
        if not isinstance(window, int) or window < 1:
            raise ParameterValidationError(
                self.name,
                f"window must be positive integer"
            )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Compute volatility."""
        try:
            window = self.parameters["window"]
            annualize = self.parameters["annualize"]
            
            returns = np.log(data['close'] / data['close'].shift(1))
            volatility = returns.rolling(window=window, min_periods=1).std()
            
            if annualize:
                volatility = volatility * np.sqrt(252)
            
            return volatility
        except Exception as e:
            raise ComputationError(self.name, str(e))
```

#### 2. Create Feature Sets

```python
from src.features.core.base import FeatureSet
from src.features.examples.basic import (
    SimpleReturn,
    RollingMean,
    RollingVolatility,
    PriceRange,
)

# Create a volatility feature set
volatility_set = FeatureSet(
    name="volatility_features",
    description="Volatility and related indicators",
    features=[
        RollingVolatility(window=20),
        RollingVolatility(window=60),
        RollingMean(window=20),
    ],
    version="1.0.0",
    asset="ES",
)
```

#### 3. Compute Features

```python
from src.features.core.base import FeatureComputer

# Create computer and add features
computer = FeatureComputer(cache_intermediate=True)
computer.add_features([
    SimpleReturn(),
    RollingMean(window=20),
    RollingVolatility(window=20),
])

# Or add a feature set
computer.add_feature_set(volatility_set)

# Compute features
result = computer.compute(
    data,  # DataFrame with OHLCV columns
    asset='ES',
)

# result is DataFrame with columns: timestamp, simple_return, rolling_mean_20, rolling_volatility_20, ...
```

#### 4. Handle Dependencies

```python
from src.features.core.base import Feature
from src.features.examples.basic import SimpleReturn, VolatilityOfReturns

# VolatilityOfReturns depends on SimpleReturn
computer = FeatureComputer()
computer.add_feature(SimpleReturn())
computer.add_feature(VolatilityOfReturns(window=20))

# Computer automatically computes SimpleReturn first, then VolatilityOfReturns
result = computer.compute(data)

# Get computation order
order = computer.get_computation_order()
# ['simple_return', 'volatility_of_returns_20']
```

### Advanced Usage

#### Batch Computation

```python
from datetime import datetime

# Define date ranges to process
date_ranges = [
    (datetime(2024, 1, 1), datetime(2024, 1, 31)),
    (datetime(2024, 2, 1), datetime(2024, 2, 29)),
    (datetime(2024, 3, 1), datetime(2024, 3, 31)),
]

# Compute features across all ranges
result = computer.compute_batch(
    data,
    date_ranges=date_ranges,
    features=['simple_return', 'rolling_volatility_20'],
    asset='ES',
)

# Check results
print(f"Computed: {result.features_computed}")
print(f"Failed: {result.features_failed}")
print(f"Total time: {result.total_time:.2f}s")
```

#### Incremental Updates

```python
# Initial computation
initial_data = data.iloc[:100]
features_v1 = computer.compute(initial_data)

# New data arrives
new_data = data.iloc[100:]

# Incremental update (much faster)
features_v2 = computer.compute_incremental(
    new_data,
    features_v1,
)

# features_v2 contains original + new features, no duplicates
print(len(features_v2))  # Should be 200 (or fewer if overlapping)
```

#### Date Filtering

```python
# Compute for a specific date range
result = computer.compute(
    data,
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    asset='ES',
)
```

#### Caching Control

```python
# Enable caching for reuse (default)
computer = FeatureComputer(cache_intermediate=True)

# With caching, intermediate features available:
computer._computation_cache['simple_return']

# Disable caching to save memory
computer = FeatureComputer(cache_intermediate=False)
# Cache is cleared after each compute() call
```

## Example Features

The framework includes ready-to-use example features in `src/features/examples/basic.py`:

### Return Features
- **SimpleReturn**: Daily log return
- **LogReturn**: Daily log return (identical to SimpleReturn)
- **CumulativeReturn**: Cumulative return from period start

### Rolling Statistics
- **RollingMean**: Configurable rolling mean of prices
- **RollingVolatility**: Configurable rolling volatility with optional annualization

### Price-Based Features
- **PriceRange**: Daily range as percentage of close
- **HighLowRatio**: Ratio of high to low price
- **CloseToOpen**: Ratio of close to open price

### Volume Features
- **RelativeVolume**: Volume relative to rolling mean

### Dependent Features
- **VolatilityOfReturns**: Volatility of returns (depends on SimpleReturn)

### Usage Example
```python
from src.features.examples.basic import (
    SimpleReturn,
    RollingVolatility,
    PriceRange,
)

features = [
    SimpleReturn(),
    RollingVolatility(window=20, annualize=True),
    RollingVolatility(window=60, annualize=True),
    PriceRange(),
]

computer = FeatureComputer(features)
result = computer.compute(data)
```

## Design Principles

### 1. Determinism

All computations are strictly deterministic:
- No random number generation
- No time-dependent logic (except timestamps in data)
- Fixed numerical precision (double-precision floating point)
- Same input always produces identical output

**Verification:**
```python
result1 = feature.compute(data)
result2 = feature.compute(data)
assert (result1 == result2).all()  # Always True (handling NaN)
```

### 2. Extensibility

Adding new features requires only subclassing:

```python
class MyCustomFeature(Feature):
    def __init__(self):
        super().__init__(
            name="my_feature",
            description="My custom feature",
            parameters={},
            dependencies=[],
        )
    
    def compute(self, data, **kwargs):
        # Your computation logic
        return result
```

No framework modifications needed.

### 3. Dependency Management

Features can depend on other features:

```python
class DependentFeature(Feature):
    def __init__(self):
        super().__init__(
            name="dependent",
            description="Depends on returns",
            parameters={},
            dependencies=["simple_return"],  # Will be computed first
        )
    
    def compute(self, data, **kwargs):
        # data will have 'simple_return' column available
        return data['simple_return'].rolling(20).mean()
```

Circular dependencies are automatically detected and reported.

### 4. Parameter Validation

Parameters are validated at construction time:

```python
class RollingMean(Feature):
    def validate_parameters(self) -> None:
        window = self.parameters.get("window")
        if not isinstance(window, int) or window < 1:
            raise ParameterValidationError(
                self.name,
                f"window must be positive integer"
            )
```

This catches configuration errors early.

## Error Handling

The framework provides specific exception types:

```python
from src.features.core.errors import (
    FeatureError,  # Base exception
    FeatureNotFoundError,  # Feature not registered
    CircularDependencyError,  # Circular dependency detected
    MissingDependencyError,  # Required feature missing
    ParameterValidationError,  # Invalid parameter
    ComputationError,  # Feature computation failed
    IncrementalUpdateError,  # Incremental update failed
)
```

**Example:**
```python
try:
    computer.compute(data)
except CircularDependencyError as e:
    print(f"Circular dependency: {e.cycle}")
except ParameterValidationError as e:
    print(f"Invalid parameter in {e.feature_name}: {e}")
except ComputationError as e:
    print(f"Failed to compute {e.feature_name}: {e}")
```

## Integration with Data Layer

The framework integrates seamlessly with Parquet-based storage:

### Reading Raw Data

```python
from src.data_io.parquet_utils import read_raw

raw_data = read_raw(
    file_path="data/raw/ES/data.parquet",
    schema=OHLCV_SCHEMA,
)

result = computer.compute(raw_data)
```

### Writing Features

```python
from src.data_io.parquet_utils import write_features
from src.data_io.schemas import get_schema

write_features(
    result,
    file_path="data/features/ES/volatility_features_v1.parquet",
    schema=get_schema("price_volatility"),
)
```

## Performance Considerations

### Batch Computation
For processing multiple date ranges, use `compute_batch()`:
- Processes ranges sequentially
- Enables date-based parallelization
- Handles computation errors per batch

### Incremental Updates
For new data periods, use `compute_incremental()`:
- Avoids recomputing existing data
- Merges old and new results
- Removes duplicates automatically

### Caching
- Enable caching when features depend on common intermediate results
- Disable caching when memory is constrained
- Clear cache manually with `clear_cache()`

## Best Practices

1. **Parameter Validation**: Always validate parameters in `validate_parameters()`
2. **Error Handling**: Wrap computations in try/except, raise `ComputationError`
3. **Documentation**: Include formulas and assumptions in docstrings
4. **Testing**: Test features with determinism checks
5. **Dependencies**: Keep dependency chains shallow (< 5 levels)
6. **Naming**: Use consistent, descriptive feature names

## Testing

The framework includes comprehensive tests:

```bash
# Run all feature tests
pytest tests/features/

# Run specific test class
pytest tests/features/test_base.py::TestFeatureBase

# Run with coverage
pytest tests/features/ --cov=src/features --cov-report=html
```

Test files:
- `tests/features/test_base.py`: Feature and FeatureSet tests
- `tests/features/test_computer.py`: FeatureComputer tests
- `tests/features/test_examples.py`: Example feature tests

## Conclusion

The Feature Computation Framework provides a robust, extensible system for feature engineering in trading applications. It emphasizes determinism, extensibility, and ease of use while maintaining clean separation between feature definition and computation logic.
