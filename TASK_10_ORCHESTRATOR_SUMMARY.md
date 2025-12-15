# Task 10: Event-Driven Simulation Orchestrator - Implementation Summary

**Status:** ✅ COMPLETE

Successfully implemented a comprehensive event-driven orchestrator for bar-by-bar backtesting in backtrader. The system ensures proper event sequencing, prevents look-ahead bias, and forms the foundation for deterministic backtesting.

## Overview

The orchestrator manages the complete simulation lifecycle:
1. **Data Delivery**: Bars arrive one at a time in chronological order
2. **Indicator Computation**: Indicators use only current and past data
3. **Strategy Execution**: Strategy signals generated for each bar
4. **Order Processing**: Orders submitted and processed sequentially
5. **Fill Simulation**: Fills computed based on current bar OHLC
6. **State Tracking**: Complete simulation state recorded at each bar

## Architecture

### Core Components

#### 1. **BacktestEngine** (`src/backtest/engine.py`)
High-level wrapper around backtrader's Cerebro engine.

**Key Responsibilities:**
- Initialize Cerebro with proper commission and slippage configuration
- Manage data feed registration (single or multiple assets)
- Add strategies and analyzers
- Execute backtests and return results
- Track engine state (current bar, portfolio value, cash)

**Key Classes:**
- `EngineConfig`: Dataclass for engine parameters
  - `initial_capital`: Starting cash (default: $100,000)
  - `commission`: Trading commission as fraction (default: 0.001 = 0.1%)
  - `slippage`: Price slippage as fraction (default: 0.0001 = 0.01%)
  - `cash_required_pct`: Percent of portfolio to hold as cash
  - `max_bars_back`: Max bars for indicator lookback
  - `verbose`: Enable verbose logging

- `BacktestEngine`: Main engine wrapper
  - `add_data(feed, name)`: Add data feed
  - `add_strategy(strategy_class, **kwargs)`: Add strategy
  - `run()`: Execute backtest
  - `get_portfolio_value()`: Get current portfolio value
  - `get_cash()`: Get available cash
  - `get_state()`: Get engine state snapshot

**Example Usage:**
```python
from src.backtest.engine import BacktestEngine, EngineConfig
from src.backtest.feeds import create_parquet_feed
from src.backtest.strategies.base import BaseStrategy

# Create engine with configuration
config = EngineConfig(
    initial_capital=100000.0,
    commission=0.001,
    slippage=0.0001,
)
engine = BacktestEngine(config)

# Add data feeds
mes_feed = create_parquet_feed('MES', start_date=..., end_date=...)
engine.add_data(mes_feed, name='MES')

# Add strategy
engine.add_strategy(MyStrategy, param1=value1)

# Run backtest
results = engine.run()
```

#### 2. **SimulationOrchestrator** (`src/backtest/orchestrator.py`)
Event-driven orchestrator managing bar-by-bar simulation flow.

**Key Responsibilities:**
- Coordinate simulation flow through BacktestEngine
- Create state snapshots at each bar
- Track orders (submitted, pending, filled)
- Detect look-ahead bias violations
- Compute final metrics (returns, drawdown, Sharpe, etc.)

**Key Classes:**
- `SimulationConfig`: Configuration for orchestrator
  - `start_date`, `end_date`: Date range for simulation
  - `max_bars`: Max bars to process (for testing)
  - `log_every_n_bars`: Progress logging frequency
  - `detect_lookahead_bias`: Enable bias detection
  - `strict_bias_checks`: Raise exceptions vs log warnings
  - `track_equity_curve`: Record portfolio value per bar
  - `track_drawdown`: Compute max drawdown
  - `verbose`, `debug_mode`: Logging control

- `SimulationState`: Dataclass for state at each bar
  - `bar_number`: Bar index
  - `timestamp`: Bar's datetime
  - `prices`: OHLCV data per symbol
  - `positions`: Current position sizes per symbol
  - `portfolio_value`: Total portfolio value
  - `cash`: Available cash
  - `pending_orders`: Orders awaiting execution
  - `filled_orders`: Completed orders

- `LookAheadBiasDetector`: Detects illegal future data access
  - `check_timestamp_access()`: Validate timestamp is not from future
  - `check_bar_index_access()`: Validate bar index is not in future
  - `get_violations()`: Get list of detected violations
  - `has_violations()`: Check if any violations detected

- `SimulationOrchestrator`: Main orchestrator
  - `run()`: Execute complete simulation
  - `_create_state_snapshot()`: Create state at bar
  - `_compute_final_metrics()`: Calculate final statistics
  - `record_order_submitted()`: Record order submission
  - `record_order_filled()`: Record order fill
  - `get_state_history()`: Get all state snapshots
  - `get_metrics()`: Get computed metrics

**Example Usage:**
```python
from src.backtest.orchestrator import SimulationOrchestrator, SimulationConfig

# Configure orchestrator
sim_config = SimulationConfig(
    detect_lookahead_bias=True,
    strict_bias_checks=True,
    track_equity_curve=True,
    track_drawdown=True,
)

# Create orchestrator with engine
orchestrator = SimulationOrchestrator(engine, sim_config)

# Run simulation
results = orchestrator.run()

# Access results
print(f"Final Value: ${results['metrics']['final_value']:.2f}")
print(f"Total Return: {results['metrics']['total_return_pct']:.2f}%")
print(f"Max Drawdown: {results['metrics']['max_drawdown_pct']:.2f}%")
print(f"Bias Violations: {len(results['bias_violations'])}")
```

#### 3. **Look-Ahead Bias Detection** (`src/backtest/utils/bias_detection.py`)
Utilities for detecting and preventing look-ahead bias.

**Key Functions:**
- `assert_current_bar_timestamp(accessed_ts, current_ts, context, strict)`: Assert timestamp not in future
- `assert_bar_not_in_future(bar_offset, context, strict)`: Assert bar index valid
- `validate_price_within_bar_range(price, low, high, close)`: Check fill price realistic
- `detect_future_data_usage(func)`: Detect suspicious code patterns
- `validate_data_isolation(current_data, next_data)`: Check data properly isolated
- `create_bias_check_decorator(strict)`: Create decorator for strategy methods
- `setup_bias_monitoring(strategy_instance, strict)`: Enable bias checking on strategy

**Key Exceptions:**
- `LookAheadBiasError`: Raised when bias detected and strict=True
- `DataIsolationViolation`: Raised when data isolation violated

**Example Usage:**
```python
from src.backtest.utils.bias_detection import (
    assert_current_bar_timestamp,
    assert_bar_not_in_future,
    LookAheadBiasError,
)

# In strategy next() method:
try:
    current_ts = self.datas[0].datetime.datetime(0)
    
    # Check timestamp access
    assert_current_bar_timestamp(
        accessed_ts,
        current_ts,
        context="RSI calculation",
        strict=True,
    )
    
    # Check bar index access
    assert_bar_not_in_future(0, context="current bar close", strict=True)
    
except LookAheadBiasError as e:
    print(f"Bias detected: {e}")
```

## Event Flow

### Strict Event Ordering (Per Bar)

```
Current Bar Arrives
    ↓
1. Feed bar to all data feeds
    ↓
2. Update all indicators (frozen to current bar)
    ↓
3. Call strategy.next() method
    - Strategy accesses current and past bars only
    - Orders generated
    ↓
4. Process orders submitted
    - Check size, price, margin
    - Route to broker
    ↓
5. Simulate fills
    - Fill prices within bar OHLC
    - Commission and slippage applied
    ↓
6. Update positions and PnL
    - Realized PnL for closed positions
    - Unrealized PnL for open positions
    ↓
7. Create state snapshot
    - Record bar timestamp, prices
    - Record positions, orders
    - Record portfolio value
    ↓
Move to Next Bar
```

### Look-Ahead Bias Prevention

The orchestrator prevents bias through:

1. **Data Isolation**
   - Each bar processed in isolation
   - Future bars not accessible
   - Indicators use only current and past data

2. **Timestamp Validation**
   - All data access checked against current bar time
   - Future timestamps raise AssertionError in strict mode
   - Develops alerts in development before production

3. **Bar Index Validation**
   - Backtrader convention: 0 = current, -1 = previous
   - Positive indices (future bars) detected
   - Negative indices allowed (past bars)

4. **Price Validation**
   - Order fills must be within bar OHLC range
   - Unrealistic fills detected
   - Prevents impossible execution scenarios

5. **Code Pattern Detection**
   - Scans strategy code for suspicious patterns
   - Detects "future", "tomorrow", etc.
   - Heuristic-based development tool

## Multi-Asset Support

Handles multiple data feeds with synchronized timestamps:

```python
from src.backtest.orchestrator import run_multi_asset_backtest

# Define data feeds
feeds = {
    'MES': mes_feed,      # Trading asset
    'ES': es_feed,        # Context data (10× leverage)
    'VIX': vix_feed,      # Context data (volatility)
}

# Run multi-asset backtest
results = run_multi_asset_backtest(
    strategy_class=MyMultiAssetStrategy,
    data_feeds=feeds,
    engine_config=config,
    sim_config=sim_config,
)
```

Features:
- Synchronizes bar timestamps across assets
- Handles data gaps (one asset missing bars)
- Supports conditional trading (e.g., trade MES only if VIX < 20)
- Maintains correct position sizes across assets

## Convenience Functions

For simplified usage:

```python
# Single-asset backtest
from src.backtest.orchestrator import run_single_asset_backtest

results = run_single_asset_backtest(
    strategy_class=MyStrategy,
    data_feed=mes_feed,
    symbol='MES',
)

# Multi-asset backtest
from src.backtest.orchestrator import run_multi_asset_backtest

results = run_multi_asset_backtest(
    strategy_class=MyMultiAssetStrategy,
    data_feeds={'MES': mes_feed, 'ES': es_feed},
)
```

## State Tracking and Metrics

### State History
Complete snapshot at each bar:
- Bar number and timestamp
- OHLCV prices per symbol
- Current positions and entry prices
- Portfolio value and cash
- Pending and filled orders
- Metrics (orders submitted/filled)

### Computed Metrics
Final simulation statistics:
- `initial_capital`: Starting cash
- `final_value`: Ending portfolio value
- `total_return_pct`: Percentage return
- `total_bars`: Bars processed
- `max_drawdown_pct`: Maximum drawdown
- `total_orders`: Orders submitted
- `filled_orders`: Orders filled

### Example Metric Access
```python
results = orchestrator.run()

# Final metrics
metrics = results['metrics']
print(f"Final Value: ${metrics['final_value']:.2f}")
print(f"Return: {metrics['total_return_pct']:.2f}%")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")

# State history (all bars)
states = results['state_history']
for state in states:
    print(f"Bar {state.bar_number} ({state.timestamp}): "
          f"Value=${state.portfolio_value:.2f}")

# Bias violations
if results['bias_violations']:
    for violation in results['bias_violations']:
        print(f"Violation: {violation}")
```

## Testing

### Test Coverage

#### 1. Engine Tests (`tests/backtest/test_orchestrator.py`)
- Engine initialization and configuration
- Data feed management
- Portfolio value and cash tracking
- State management and reset

#### 2. Orchestrator Tests (`tests/backtest/test_orchestrator.py`)
- Orchestrator initialization
- State snapshot creation
- Metric computation
- Order tracking
- Multi-asset scenarios

#### 3. Bias Detection Tests (`tests/backtest/test_bias_prevention.py`)
- Timestamp validation
- Bar index validation
- Price range validation
- Data isolation checks
- Future data pattern detection
- Edge cases (microsecond differences, zero ranges)
- Strict vs non-strict mode behavior

#### 4. Integration Tests
- Engine + Orchestrator together
- Multi-asset backtests
- Complete simulation flow

### Running Tests

```bash
# Run orchestrator tests
pytest tests/backtest/test_orchestrator.py -v

# Run bias prevention tests
pytest tests/backtest/test_bias_prevention.py -v

# Run all backtest tests
pytest tests/backtest/ -v

# Run with coverage
pytest tests/backtest/ --cov=src.backtest --cov-report=html
```

## Performance Characteristics

### Optimization Considerations

1. **Memory**
   - State history stored in memory (one per bar)
   - For 1-minute data, ~252 * 250 bars per year = ~63k snapshots
   - Each state ~1KB → ~63MB per year of data

2. **Speed**
   - Event processing per bar: ~1-10ms
   - 1-minute bars over 1 year: 63k bars * 10ms = ~630 seconds (~10 minutes)
   - Acceptable for development, daily strategy testing

3. **Scaling**
   - Single-threaded (backtrader limitation)
   - Parallel simulations possible (multiple processes)
   - GPU acceleration not applicable

## Dependencies

- **backtrader**: Core simulation engine
- **pandas**: Data handling
- **numpy**: Numerical computations
- Custom modules:
  - `src.backtest.feeds.ParquetDataFeed`: Data feed implementation
  - `src.backtest.strategies.BaseStrategy`: Base strategy class
  - `src.backtest.contracts`: Futures specifications

## Files Modified/Created

### Created
- ✅ `src/backtest/engine.py`: BacktestEngine and EngineConfig
- ✅ `src/backtest/orchestrator.py`: SimulationOrchestrator and utilities
- ✅ `src/backtest/utils/bias_detection.py`: Look-ahead bias utilities
- ✅ `tests/backtest/test_orchestrator.py`: Engine and orchestrator tests
- ✅ `tests/backtest/test_bias_prevention.py`: Bias detection tests

### Modified
- ✅ `src/backtest/__init__.py`: Added engine and orchestrator exports
- ✅ `src/backtest/utils/__init__.py`: Added bias detection exports

## Usage Examples

### Basic Single-Asset Backtest

```python
from datetime import datetime
from src.backtest.engine import BacktestEngine, EngineConfig
from src.backtest.orchestrator import SimulationOrchestrator, SimulationConfig
from src.backtest.feeds import create_parquet_feed
from src.backtest.strategies.base import BaseStrategy

# Define strategy
class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(strategy_name='MyStrategy')
    
    def next(self):
        if len(self) > 20:
            if self.close[0] > self.close[-20]:  # Close above 20-bar high
                if not self.position:
                    self.buy()
            elif self.position:
                self.close()

# Setup
config = EngineConfig(initial_capital=100000.0)
engine = BacktestEngine(config)

mes_feed = create_parquet_feed(
    'MES',
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
)
engine.add_data(mes_feed, name='MES')
engine.add_strategy(MyStrategy)

# Run simulation
sim_config = SimulationConfig(detect_lookahead_bias=True)
orchestrator = SimulationOrchestrator(engine, sim_config)
results = orchestrator.run()

# Analyze
print(f"Final Value: ${results['metrics']['final_value']:.2f}")
print(f"Return: {results['metrics']['total_return_pct']:.2f}%")
print(f"Bias Violations: {len(results['bias_violations'])}")
```

### Multi-Asset Backtest with Context Data

```python
from src.backtest.orchestrator import run_multi_asset_backtest

class MultiAssetStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(strategy_name='MultiAssetStrategy')
    
    def next(self):
        mes = self.datas[0]  # Main trading instrument
        es = self.datas[1]   # Context data
        vix = self.datas[2]  # Volatility context
        
        # Trade MES only if VIX < 20
        if vix.close[0] < 20:
            if not self.position and mes.close[0] > mes.close[-1]:
                self.buy(data=mes)
            elif self.position and mes.close[0] < mes.close[-1]:
                self.close(data=mes)

results = run_multi_asset_backtest(
    strategy_class=MultiAssetStrategy,
    data_feeds={
        'MES': mes_feed,
        'ES': es_feed,
        'VIX': vix_feed,
    },
)
```

## Future Enhancements

1. **Adaptive State History**
   - Option to store only summary stats instead of full state
   - Reduces memory for long backtests

2. **Event Callbacks**
   - Hooks for custom processing at each bar
   - Extended metrics computation

3. **Parallel Optimization**
   - Walk-forward optimization with multiple engines
   - Parameter grid search acceleration

4. **Advanced Bias Detection**
   - Machine learning-based pattern detection
   - Execution consistency validation
   - Monte Carlo permutation testing

5. **Live Trading Integration**
   - Replay orchestrator for live data
   - Consistent state management across backtest/live

## Conclusion

The event-driven simulation orchestrator provides a robust foundation for deterministic backtesting with built-in protections against look-ahead bias. The modular design allows easy integration with custom strategies, data sources, and analyzers while maintaining strict event sequencing and proper isolation of future data.
