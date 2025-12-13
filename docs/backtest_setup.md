# Backtest Framework Setup and Usage Guide

## Overview

The backtest module provides a clean, modular framework for backtesting trading strategies using backtrader. It includes configuration management, logging utilities, and a base strategy class for consistent implementation.

## Installation

### Prerequisites
- Python 3.8+
- pandas
- numpy

### Install Dependencies

```bash
pip install backtrader pandas numpy pyyaml
```

## Project Structure

```
src/backtest/
├── __init__.py                    # Module initialization
├── strategies/                    # Strategy implementations
│   ├── __init__.py
│   └── base.py                   # Base strategy class with utilities
├── feeds/                         # Custom data feeds
│   └── __init__.py               # (parquet_feed.py in subtask #40)
├── analyzers/                     # Custom analyzers
│   └── __init__.py
├── config/                        # Configuration management
│   ├── __init__.py
│   └── defaults.py               # Configuration system
└── utils/                         # Utilities
    ├── __init__.py
    └── logging.py                # Logging infrastructure

config/
└── backtest_config.yaml          # Example configuration file
```

## Quick Start

### 1. Load Configuration

```python
from src.backtest.config import BacktestConfig

# Load from YAML
config = BacktestConfig.from_yaml('config/backtest_config.yaml')

# Or create programmatically
config = BacktestConfig(
    initial_capital=100000.0,
    commission=0.001,
    slippage=0.0001,
)

# Access MES futures specification
mes_spec = config.get_futures_spec('MES')
print(f"MES Multiplier: {mes_spec.multiplier}")  # $5 per point
```

### 2. Create a Strategy

```python
from src.backtest.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    """Simple moving average crossover strategy."""
    
    params = (
        ('strategy_name', 'MovingAverageCross'),
        ('ma_period', 20),
    )
    
    def __init__(self):
        super().__init__()
        
        # Add technical indicator
        self.ma = bt.indicators.MovingAverageSimple(
            self.datas[0].close,
            period=self.params.ma_period
        )
    
    def next(self):
        """Execute strategy logic on each bar."""
        
        if not self.position:
            # Entry signal
            if self.datas[0].close[0] > self.ma[0]:
                self.log_signal('BUY', {'reason': 'Price above MA'})
                self.buy(size=1)
        else:
            # Exit signal
            if self.datas[0].close[0] < self.ma[0]:
                self.log_signal('SELL', {'reason': 'Price below MA'})
                self.sell(size=self.position.size)
```

### 3. Set Up Backtest

```python
import backtrader as bt
from src.backtest.config import BacktestConfig
from src.backtest.strategies.base import BaseStrategy
from src.backtest.utils.logging import setup_logging, get_performance_logger

# Configure logging
setup_logging(level='INFO', debug=False)
perf_logger = get_performance_logger()

# Create cerebro engine
cerebro = bt.Cerebro()

# Configure from config
config = BacktestConfig.from_yaml('config/backtest_config.yaml')
cerebro.broker.setcash(config.initial_capital)
cerebro.broker.setcommission(commission=config.commission)

# Add strategy
cerebro.addstrategy(MyStrategy)

# Add data (to be implemented with ParquetFeed)
# data = ParquetFeed(dataname='data/ohlcv.parquet')
# cerebro.adddata(data)

# Run backtest
results = cerebro.run()
strategy = results[0]

# Get statistics
stats = strategy.get_trade_stats()
print(f"Total Trades: {stats['total_trades']}")
print(f"Win Rate: {stats['win_rate']:.1f}%")
print(f"Total PnL: ${stats['total_pnl']:.2f}")
```

## Configuration

### Configuration Methods

#### 1. From YAML File

```python
config = BacktestConfig.from_yaml('config/backtest_config.yaml')
```

See `config/backtest_config.yaml` for full example with all parameters.

#### 2. From JSON File

```python
config = BacktestConfig.from_json('config/backtest.json')
```

JSON format:
```json
{
    "initial_capital": 100000.0,
    "commission": 0.001,
    "slippage": 0.0001,
    "futures_specs": {
        "MES": {
            "symbol": "MES",
            "multiplier": 5.0,
            "tick_size": 0.25,
            "margin_requirement": 300.0,
            "contract_size": 1.0
        }
    }
}
```

#### 3. From Dictionary

```python
config_dict = {
    'initial_capital': 100000.0,
    'commission': 0.001,
    'slippage': 0.0001,
}
config = BacktestConfig.from_dict(config_dict)
```

#### 4. From Environment Variables

```python
# Set environment variables
export BACKTEST_INITIAL_CAPITAL=100000
export BACKTEST_COMMISSION=0.001
export BACKTEST_ENVIRONMENT=dev
export BACKTEST_VERBOSE=true

config = BacktestConfig.from_env()
```

Supported environment variables:
- `BACKTEST_INITIAL_CAPITAL`: Starting cash
- `BACKTEST_COMMISSION`: Commission rate
- `BACKTEST_SLIPPAGE`: Slippage amount
- `BACKTEST_ENVIRONMENT`: dev or production
- `BACKTEST_VERBOSE`: true or false

#### 5. Programmatic Configuration

```python
config = BacktestConfig(
    initial_capital=100000.0,
    commission=0.001,
    slippage=0.0001,
    cash_required_pct=0.10,
    environment='dev',
    verbose=False,
)

# Add custom futures spec
from src.backtest.config.defaults import FuturesSpec

nq_spec = FuturesSpec(
    symbol='NQ',
    multiplier=20.0,
    tick_size=0.25,
    margin_requirement=12500.0,
    contract_size=1.0,
)
config.add_futures_spec('NQ', nq_spec)

# Set strategy parameters
config.set_strategy_param('ma_period', 20)
config.set_strategy_param('position_size', 1)
```

### Futures Specifications

Pre-configured specs:

**MES (Micro E-mini S&P 500)**
- Multiplier: $5 per index point
- Tick size: 0.25 index point
- Margin requirement: ~$300 per contract

**ES (E-mini S&P 500)**
- Multiplier: $50 per index point
- Tick size: 0.25 index point
- Margin requirement: ~$12,500 per contract

Add custom specs:
```python
config.add_futures_spec('NQ', FuturesSpec(
    symbol='NQ',
    multiplier=20.0,
    tick_size=0.25,
    margin_requirement=12500.0,
    contract_size=1.0,
))
```

## Logging

### Setup Logging

```python
from src.backtest.utils.logging import setup_logging, get_logger

# Basic setup
setup_logging(level='INFO')

# With file output
setup_logging(level='DEBUG', log_file='backtest.log')

# Debug mode
setup_logging(level='INFO', debug=True, log_file='backtest_debug.log')
```

### Get Logger Instances

```python
from src.backtest.utils.logging import (
    get_logger,
    get_trade_logger,
    get_order_logger,
    get_signal_logger,
    get_performance_logger,
)

# Strategy logger
logger = get_logger('backtest.strategies.MyStrategy')

# Specific loggers
trade_logger = get_trade_logger()
order_logger = get_order_logger()
signal_logger = get_signal_logger('MyStrategy')
perf_logger = get_performance_logger()

# Log messages
logger.info('Backtest started')
logger.debug('Current price: 100.5')
logger.warning('Drawdown exceeded 10%')
logger.error('Data loading failed')
```

### Log Levels

- **DEBUG**: Detailed information for debugging (default if debug=True)
- **INFO**: General informational messages (default)
- **WARNING**: Warning messages for important events
- **ERROR**: Error messages

### Environment Variables for Logging

```bash
# Set log level
export BACKTEST_LOG_LEVEL=DEBUG

# Set log file
export BACKTEST_LOG_FILE=logs/backtest.log

# Enable debug mode
export BACKTEST_DEBUG=true
```

## Base Strategy Class

The `BaseStrategy` class provides common utilities for strategy implementation:

### Features

1. **Structured Logging**
   - Automatic timestamp logging
   - Message leveling (debug, info, warning, error)
   - Strategy name included in logs

2. **Trade Tracking**
   - Automatic logging of entry and exit prices
   - PnL calculation (gross and net of commission)
   - Trade-level statistics

3. **Order Management**
   - Order submission logging
   - Order execution tracking
   - Order rejection handling
   - Active order tracking

4. **Signal Logging**
   - Trading signal documentation
   - Signal details and reasons
   - Historical signal tracking

5. **Performance Metrics**
   - Trade statistics (win rate, average PnL)
   - Position tracking
   - Entry/exit price monitoring

### Key Methods

#### `log(msg, level='info')`
Log a message with timestamp and current close price.

```python
self.log('Price above 20-MA', level='info')
self.log('Entry signal triggered', level='debug')
```

#### `log_signal(signal_type, details)`
Log a trading signal with detailed information.

```python
self.log_signal('BUY', {
    'reason': 'Price crossed above MA20',
    'ma20': self.ma[0],
    'price': self.datas[0].close[0],
})
```

#### `get_trade_stats()`
Get statistics for all completed trades.

```python
stats = self.get_trade_stats()
print(f"Win Rate: {stats['win_rate']:.1f}%")
print(f"Total PnL: ${stats['total_pnl']:.2f}")
print(f"Average Trade: ${stats['avg_pnl']:.2f}")
```

#### `notify_order(order)`
Called automatically when order status changes. Override to customize.

```python
def notify_order(self, order):
    super().notify_order(order)
    # Custom order handling here
```

#### `notify_trade(trade)`
Called automatically when a trade closes. Override to customize.

```python
def notify_trade(self, trade):
    super().notify_trade(trade)
    # Custom trade handling here
```

## Example: Complete Strategy

```python
import backtrader as bt
from src.backtest.strategies.base import BaseStrategy
from src.backtest.config import BacktestConfig
from src.backtest.utils.logging import setup_logging

class EnhancedMA(BaseStrategy):
    """Moving average crossover with risk management."""
    
    params = (
        ('strategy_name', 'EnhancedMA'),
        ('fast_ma', 10),
        ('slow_ma', 20),
        ('risk_per_trade', 0.02),  # Risk 2% per trade
        ('atr_period', 14),
    )
    
    def __init__(self):
        super().__init__()
        self.fast_ma = bt.indicators.EMA(
            self.datas[0].close,
            period=self.params.fast_ma
        )
        self.slow_ma = bt.indicators.EMA(
            self.datas[0].close,
            period=self.params.slow_ma
        )
        self.atr = bt.indicators.ATR(self.datas[0])
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
    
    def next(self):
        if not self.position:
            if self.crossover > 0:
                # Calculate position size based on risk
                stop_distance = self.atr[0] * 2
                position_size = int(
                    (self.broker.getvalue() * self.params.risk_per_trade) /
                    (stop_distance * 5.0)  # MES multiplier
                )
                self.log_signal('BUY', {
                    'fast_ma': self.fast_ma[0],
                    'slow_ma': self.slow_ma[0],
                    'size': position_size,
                })
                self.buy(size=position_size)
        else:
            if self.crossover < 0:
                self.log_signal('SELL', {'reason': 'Crossover'})
                self.sell(size=self.position.size)

# Run backtest
if __name__ == '__main__':
    setup_logging(level='INFO')
    
    cerebro = bt.Cerebro()
    config = BacktestConfig.from_yaml('config/backtest_config.yaml')
    
    cerebro.broker.setcash(config.initial_capital)
    cerebro.broker.setcommission(commission=config.commission)
    
    cerebro.addstrategy(EnhancedMA)
    # Add data here
    
    results = cerebro.run()
    strategy = results[0]
    
    stats = strategy.get_trade_stats()
    print(f"\n{'='*50}")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']:.1f}%")
    print(f"Total PnL: ${stats['total_pnl']:.2f}")
    print(f"Avg Win: ${stats['avg_win']:.2f}")
    print(f"Avg Loss: ${stats['avg_loss']:.2f}")
    print(f"{'='*50}")
```

## Adding New Strategies

1. Create a new file in `src/backtest/strategies/`
2. Inherit from `BaseStrategy`
3. Implement the `next()` method for strategy logic
4. Use `log_signal()` for important events
5. Override `notify_trade()` for trade-specific logic if needed

Example:

```python
# src/backtest/strategies/my_strategy.py
from src.backtest.strategies.base import BaseStrategy
import backtrader as bt

class MyStrategy(BaseStrategy):
    params = (
        ('strategy_name', 'MyStrategy'),
        # Add your parameters here
    )
    
    def __init__(self):
        super().__init__()
        # Initialize indicators
    
    def next(self):
        # Implement trading logic
        pass
```

## Adding Custom Data Feeds

The `feeds/` directory is for custom data feed implementations. The `ParquetFeed` will be implemented in subtask #40 to load OHLCV data from Parquet files.

Future example:
```python
from src.backtest.feeds.parquet_feed import ParquetFeed

data = ParquetFeed(
    dataname='data/ohlcv.parquet',
    fromdate=dt.datetime(2024, 1, 1),
    todate=dt.datetime(2024, 12, 31),
)
cerebro.adddata(data)
```

## Adding Custom Analyzers

The `analyzers/` directory is for custom performance analyzers. These extend backtrader's built-in analyzers with domain-specific metrics.

Example structure:
```python
# src/backtest/analyzers/sharpe_ratio.py
import backtrader as bt

class SharpeRatioAnalyzer(bt.Analyzer):
    def __init__(self):
        super().__init__()
        # Initialize analyzer
    
    def next(self):
        # Analyze on each bar
        pass
```

## Integration with Data Ingestion

The backtest module is designed to work with the data ingestion orchestration system. After ingesting data with `scripts/ingest_historical.py`, load it using the ParquetFeed (subtask #40):

```python
from src.data_ingestion.orchestrator import Orchestrator
from src.backtest.feeds.parquet_feed import ParquetFeed

# Ingest data
orchestrator = Orchestrator(...)
orchestrator.ingest_asset('MES', '1D', ...)

# Load in backtest
data = ParquetFeed(dataname='cleaned/v1/MES/ohlcv.parquet')
cerebro.adddata(data)
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'backtrader'`

**Solution**: Install backtrader
```bash
pip install backtrader
```

### Configuration Not Loading

**Problem**: `FileNotFoundError: Configuration file not found`

**Solution**: Check file path is correct and relative to current directory
```python
import os
print(os.path.abspath('config/backtest_config.yaml'))
config = BacktestConfig.from_yaml('config/backtest_config.yaml')
```

### Logging Not Working

**Problem**: No log output appearing

**Solution**: Call `setup_logging()` before creating strategies
```python
from src.backtest.utils.logging import setup_logging
setup_logging(level='INFO')
```

### No Trades Generated

**Problem**: Strategy runs but no trades

**Solution**: 
1. Check data is loaded correctly
2. Verify strategy logic with logging
3. Check entry conditions are triggered

```python
def next(self):
    self.log(f'Close: {self.datas[0].close[0]:.2f}', level='debug')
    self.log_signal('CHECK', {'condition_value': self.some_indicator[0]})
```

## See Also

- `config/backtest_config.yaml`: Example configuration with all parameters
- `src/backtest/strategies/base.py`: Base strategy class implementation
- `src/backtest/config/defaults.py`: Configuration system
- `src/backtest/utils/logging.py`: Logging utilities
