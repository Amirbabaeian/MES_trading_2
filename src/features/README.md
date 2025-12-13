# Feature Computation Framework

A deterministic, extensible framework for computing trading features from OHLCV market data.

## Quick Start

```python
from src.features.core import FeatureComputer
from src.features.examples import SimpleReturn, RollingVolatility

# Create computer and add features
computer = FeatureComputer()
computer.add_features([
    SimpleReturn(),
    RollingVolatility(window=20),
])

# Compute features for your data
result = computer.compute(ohlcv_data)
```

## Features

**Core Components:**
- `Feature`: Abstract base class for defining features
- `FeatureSet`: Container for grouping related features
- `FeatureComputer`: Orchestration engine for feature computation
- `DependencyGraph`: Dependency resolution with topological sorting

**Example Features:**
- Simple returns and log returns
- Rolling mean and rolling volatility
- Price range and high/low ratio
- Volume profile features
- And more...

## Key Capabilities

✓ **Deterministic**: Reproducible results, no randomness  
✓ **Extensible**: Define custom features by subclassing  
✓ **Parameterizable**: Configurable window sizes, thresholds, etc.  
✓ **Dependency-Aware**: Automatic dependency resolution  
✓ **Batch Processing**: Efficient multi-period computation  
✓ **Incremental Updates**: Compute only new data periods  
✓ **Type Safe**: Parameter validation and error handling  

## Directory Structure

```
src/features/
├── core/
│   ├── base.py         # Feature, FeatureSet, FeatureComputer
│   ├── dependency.py   # Dependency graph and topological sort
│   └── errors.py       # Custom exceptions
├── examples/
│   └── basic.py        # Example feature implementations
└── README.md           # This file

tests/features/
├── test_base.py        # Feature and FeatureSet tests
├── test_computer.py    # FeatureComputer tests
├── test_dependency.py  # Dependency graph tests
├── test_examples.py    # Example feature tests
└── test_integration.py # End-to-end integration tests
```

## Documentation

See `docs/feature_framework.md` for comprehensive documentation including:
- Architecture and design patterns
- Usage guide and examples
- API reference
- Best practices
- Performance considerations

## Testing

Run the test suite:

```bash
# All feature tests
pytest tests/features/

# Specific test file
pytest tests/features/test_base.py

# With coverage
pytest tests/features/ --cov=src/features
```

## Example Usage

### Define Custom Features

```python
from src.features.core import Feature
from src.features.core.errors import ParameterValidationError
import pandas as pd
import numpy as np

class MyFeature(Feature):
    def __init__(self, window: int = 20):
        super().__init__(
            name=f"my_feature_{window}",
            description=f"Custom feature with window={window}",
            parameters={"window": window},
            dependencies=[],
        )
    
    def validate_parameters(self) -> None:
        window = self.parameters.get("window")
        if not isinstance(window, int) or window < 1:
            raise ParameterValidationError(
                self.name,
                f"window must be positive integer"
            )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        return data['close'].rolling(self.parameters['window']).mean()
```

### Compute Features

```python
from src.features.core import FeatureComputer

computer = FeatureComputer()
computer.add_feature(MyFeature(window=20))

result = computer.compute(ohlcv_data)
# result has columns: timestamp, my_feature_20
```

### Handle Dependencies

```python
from src.features.examples import SimpleReturn, VolatilityOfReturns

computer = FeatureComputer()
computer.add_features([
    SimpleReturn(),
    VolatilityOfReturns(window=20),  # Depends on SimpleReturn
])

# Dependencies are automatically resolved
result = computer.compute(data)
```

## Design Principles

### Determinism
All computations are reproducible. Same input always produces identical output.

### Extensibility
Add new features without modifying the framework - just subclass `Feature`.

### Dependency Management
Features can declare dependencies on other features. Circular dependencies are detected automatically.

### Parameter Validation
Parameters are validated at initialization time, catching configuration errors early.

## Performance Notes

- **Batch Computation**: Use `compute_batch()` for multiple date ranges
- **Incremental Updates**: Use `compute_incremental()` for new data periods
- **Caching**: Enable `cache_intermediate=True` for features with common dependencies
- **Memory**: Disable caching for large datasets to reduce memory usage

## Contributing

To add a new feature:

1. Subclass `Feature` in `src/features/examples/basic.py` or create a new file
2. Implement `compute()` method and optional `validate_parameters()`
3. Add tests in `tests/features/test_examples.py`
4. Update documentation if needed

Example:

```python
class MyNewFeature(Feature):
    def __init__(self):
        super().__init__(
            name="my_new_feature",
            description="Description of feature",
            parameters={},
            dependencies=[],
        )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Your computation logic
        return result
```

## License

See LICENSE file in repository root.
