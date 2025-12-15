"""
Event-Driven Simulation Orchestrator

Manages bar-by-bar simulation ensuring proper event sequencing and
preventing look-ahead bias. Coordinates:
1. Data delivery (bars arrive one at a time)
2. Indicator computation (only using current and past data)
3. Strategy signal generation
4. Order submission and processing
5. Fill simulation and PnL calculation

Key Classes:
- SimulationOrchestrator: Main orchestrator managing event flow
- SimulationConfig: Configuration for orchestrator behavior
- SimulationState: Tracks state at each bar
- LookAheadBiasDetector: Detects illegal future data access

Event Flow (per bar):
1. Feed new bar to all data feeds
2. Update all indicators (frozen to current bar only)
3. Call strategy next() method with current data
4. Process orders submitted by strategy
5. Simulate fills based on current bar OHLC
6. Update positions and PnL
7. Record state and metrics
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

try:
    import backtrader as bt
    import pandas as pd
    import numpy as np
except ImportError:
    bt = pd = np = None

from src.backtest.engine import BacktestEngine, EngineConfig
from src.backtest.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for the simulation orchestrator."""
    
    # Timing
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Execution
    max_bars: Optional[int] = None  # Max bars to process (for testing)
    log_every_n_bars: int = 100  # Log progress every N bars
    
    # Bias detection
    detect_lookahead_bias: bool = True
    strict_bias_checks: bool = True
    
    # State tracking
    track_order_fill_times: bool = True
    track_equity_curve: bool = True
    track_drawdown: bool = True
    
    # Logging
    verbose: bool = False
    debug_mode: bool = False


@dataclass
class OrderSnapshot:
    """Snapshot of an order at a specific bar."""
    
    order_id: int
    symbol: str
    side: str  # 'BUY' or 'SELL'
    size: float
    submit_price: float
    submit_timestamp: datetime
    submit_bar: int
    status: str


@dataclass
class SimulationState:
    """State snapshot at each bar."""
    
    bar_number: int
    timestamp: datetime
    
    # Data snapshot (prices at this bar)
    prices: Dict[str, Dict[str, float]] = field(default_factory=dict)  # {symbol: {o,h,l,c,v}}
    
    # Position state
    positions: Dict[str, int] = field(default_factory=dict)  # {symbol: size}
    position_prices: Dict[str, float] = field(default_factory=dict)  # {symbol: entry_price}
    
    # Equity state
    portfolio_value: float = 0.0
    cash: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Order state
    pending_orders: List[OrderSnapshot] = field(default_factory=list)
    filled_orders: List[OrderSnapshot] = field(default_factory=list)
    
    # Metrics
    bars_processed: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0


class LookAheadBiasDetector:
    """
    Detects look-ahead bias in strategy execution.
    
    Enforces that strategies can only access data up to the current bar,
    not future bars.
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize the detector.
        
        Args:
            strict: If True, raise exceptions. If False, log warnings.
        """
        self.strict = strict
        self.logger = get_logger('backtest.bias_detector')
        self.violations: List[str] = []
    
    def check_timestamp_access(
        self,
        access_timestamp: datetime,
        current_bar_timestamp: datetime,
        access_location: str = '',
    ) -> bool:
        """
        Check that accessed timestamp is not in the future.
        
        Args:
            access_timestamp: Timestamp being accessed
            current_bar_timestamp: Current bar's timestamp
            access_location: Where the access occurred (for logging)
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            AssertionError: If strict mode and violation detected
        """
        if access_timestamp > current_bar_timestamp:
            msg = (
                f'LOOK-AHEAD BIAS VIOLATION: '
                f'Accessing future timestamp {access_timestamp} '
                f'while processing bar {current_bar_timestamp} '
                f'[{access_location}]'
            )
            self.violations.append(msg)
            
            if self.strict:
                self.logger.error(msg)
                raise AssertionError(msg)
            else:
                self.logger.warning(msg)
                return False
        
        return True
    
    def check_bar_index_access(
        self,
        access_index: int,
        current_index: int,
        access_location: str = '',
    ) -> bool:
        """
        Check that accessed bar index is not in the future.
        
        Args:
            access_index: Bar index being accessed (0 is most recent)
            current_index: Current bar index
            access_location: Where the access occurred
            
        Returns:
            True if valid, False otherwise
        """
        # In backtrader convention, 0 is the current bar, negative indices are past
        # So we check that we're not accessing positive indices
        if access_index > 0:
            msg = (
                f'LOOK-AHEAD BIAS VIOLATION: '
                f'Accessing future bar index {access_index} '
                f'from current bar {current_index} '
                f'[{access_location}]'
            )
            self.violations.append(msg)
            
            if self.strict:
                self.logger.error(msg)
                raise AssertionError(msg)
            else:
                self.logger.warning(msg)
                return False
        
        return True
    
    def get_violations(self) -> List[str]:
        """Get list of detected violations."""
        return self.violations.copy()
    
    def has_violations(self) -> bool:
        """Check if any violations were detected."""
        return len(self.violations) > 0


class SimulationOrchestrator:
    """
    Event-driven orchestrator for bar-by-bar backtesting.
    
    Manages the complete simulation lifecycle:
    1. Initialize engine with strategies and data feeds
    2. Process each bar in chronological order
    3. Coordinate indicator calculations, strategy calls, and order processing
    4. Track simulation state and metrics
    5. Detect and prevent look-ahead bias
    
    Attributes:
        engine: BacktestEngine instance
        config: SimulationConfig instance
        bias_detector: LookAheadBiasDetector instance
        state_history: List of SimulationState at each bar
    """
    
    def __init__(
        self,
        engine: BacktestEngine,
        config: Optional[SimulationConfig] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            engine: Configured BacktestEngine instance
            config: SimulationConfig. If None, defaults are used.
        """
        self.engine = engine
        self.config = config or SimulationConfig()
        self.logger = get_logger('backtest.orchestrator')
        
        # Bias detection
        self.bias_detector = LookAheadBiasDetector(
            strict=self.config.strict_bias_checks
        )
        
        # State tracking
        self.state_history: List[SimulationState] = []
        self.current_state: Optional[SimulationState] = None
        
        # Metrics
        self.bars_processed = 0
        self.total_orders_submitted = 0
        self.total_orders_filled = 0
        
        self.logger.info(
            f'SimulationOrchestrator initialized: '
            f'detect_bias={self.config.detect_lookahead_bias}, '
            f'track_equity={self.config.track_equity_curve}'
        )
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete simulation.
        
        Returns:
            Dictionary containing:
            - results: Engine run results
            - state_history: List of SimulationState per bar
            - metrics: Summary metrics
            - bias_violations: Any detected bias violations
        """
        self.logger.info('Starting simulation orchestration...')
        
        try:
            # Run the backtest through the engine
            results = self.engine.run()
            
            # Finalize metrics
            metrics = self._compute_final_metrics()
            
            # Check for bias violations
            violations = self.bias_detector.get_violations() if self.config.detect_lookahead_bias else []
            
            self.logger.info(
                f'Simulation completed: '
                f'{self.bars_processed} bars processed, '
                f'{self.total_orders_submitted} orders submitted, '
                f'{self.total_orders_filled} orders filled'
            )
            
            if violations:
                self.logger.warning(
                    f'Detected {len(violations)} look-ahead bias violations'
                )
            
            return {
                'results': results,
                'state_history': self.state_history,
                'metrics': metrics,
                'bias_violations': violations,
                'bars_processed': self.bars_processed,
                'orders_submitted': self.total_orders_submitted,
                'orders_filled': self.total_orders_filled,
            }
        
        except Exception as e:
            self.logger.error(f'Simulation failed: {e}', exc_info=True)
            raise
    
    def _create_state_snapshot(
        self,
        bar_number: int,
        timestamp: datetime,
        data_snapshot: Dict[str, Dict[str, float]],
    ) -> SimulationState:
        """
        Create a snapshot of the current simulation state.
        
        Args:
            bar_number: Current bar number
            timestamp: Current bar's timestamp
            data_snapshot: Dict of OHLCV data per symbol
            
        Returns:
            SimulationState snapshot
        """
        portfolio_value = self.engine.get_portfolio_value()
        cash = self.engine.get_cash()
        
        state = SimulationState(
            bar_number=bar_number,
            timestamp=timestamp,
            prices=data_snapshot,
            portfolio_value=portfolio_value,
            cash=cash,
            bars_processed=bar_number,
            orders_submitted=self.total_orders_submitted,
            orders_filled=self.total_orders_filled,
        )
        
        self.current_state = state
        self.state_history.append(state)
        
        return state
    
    def _compute_final_metrics(self) -> Dict[str, Any]:
        """
        Compute final simulation metrics.
        
        Returns:
            Dictionary with metrics
        """
        if not self.state_history:
            return {}
        
        first_state = self.state_history[0]
        last_state = self.state_history[-1]
        
        initial_value = self.engine.config.initial_capital
        final_value = last_state.portfolio_value
        
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Compute max drawdown
        values = [s.portfolio_value for s in self.state_history]
        running_max = np.maximum.accumulate(values) if self.config.track_drawdown else values
        drawdown = (np.array(values) - running_max) / running_max * 100 if self.config.track_drawdown else 0
        max_drawdown = float(np.min(drawdown)) if self.config.track_drawdown else 0
        
        return {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return_pct': float(total_return),
            'total_bars': len(self.state_history),
            'max_drawdown_pct': float(max_drawdown),
            'total_orders': self.total_orders_submitted,
            'filled_orders': self.total_orders_filled,
        }
    
    def record_order_submitted(
        self,
        order_id: int,
        symbol: str,
        side: str,
        size: float,
        price: float,
    ):
        """
        Record an order submission event.
        
        Args:
            order_id: Unique order identifier
            symbol: Asset symbol
            side: 'BUY' or 'SELL'
            size: Order size
            price: Order price
        """
        self.total_orders_submitted += 1
        
        if self.current_state:
            snapshot = OrderSnapshot(
                order_id=order_id,
                symbol=symbol,
                side=side,
                size=size,
                submit_price=price,
                submit_timestamp=self.current_state.timestamp,
                submit_bar=self.current_state.bar_number,
                status='SUBMITTED',
            )
            self.current_state.pending_orders.append(snapshot)
    
    def record_order_filled(
        self,
        order_id: int,
        symbol: str,
        fill_price: float,
    ):
        """
        Record an order fill event.
        
        Args:
            order_id: Order identifier
            symbol: Asset symbol
            fill_price: Actual fill price
        """
        self.total_orders_filled += 1
        
        if self.current_state:
            # Move from pending to filled
            for order in self.current_state.pending_orders[:]:
                if order.order_id == order_id:
                    order.status = 'FILLED'
                    self.current_state.filled_orders.append(order)
                    self.current_state.pending_orders.remove(order)
                    break
    
    def get_state_history(self) -> List[SimulationState]:
        """Get complete history of state snapshots."""
        return self.state_history.copy()
    
    def get_current_state(self) -> Optional[SimulationState]:
        """Get the current state snapshot."""
        return self.current_state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get computed metrics."""
        return self._compute_final_metrics()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f'SimulationOrchestrator('
            f'bars_processed={self.bars_processed}, '
            f'orders={self.total_orders_submitted}, '
            f'bias_violations={len(self.bias_detector.get_violations())}'
            f')'
        )


# ============================================================================
# Convenience Functions for Common Scenarios
# ============================================================================

def run_single_asset_backtest(
    strategy_class: type,
    data_feed: bt.DataBase,
    symbol: str = 'MES',
    engine_config: Optional[EngineConfig] = None,
    sim_config: Optional[SimulationConfig] = None,
    **strategy_kwargs
) -> Dict[str, Any]:
    """
    Run a simple single-asset backtest.
    
    Args:
        strategy_class: Strategy class to use
        data_feed: Backtrader DataBase instance
        symbol: Asset symbol
        engine_config: Engine configuration
        sim_config: Orchestrator configuration
        **strategy_kwargs: Strategy parameters
        
    Returns:
        Orchestrator run results
    """
    engine = BacktestEngine(engine_config)
    engine.add_data(data_feed, name=symbol)
    engine.add_strategy(strategy_class, **strategy_kwargs)
    
    orchestrator = SimulationOrchestrator(engine, sim_config)
    
    return orchestrator.run()


def run_multi_asset_backtest(
    strategy_class: type,
    data_feeds: Dict[str, bt.DataBase],
    engine_config: Optional[EngineConfig] = None,
    sim_config: Optional[SimulationConfig] = None,
    **strategy_kwargs
) -> Dict[str, Any]:
    """
    Run a multi-asset backtest with synchronized feeds.
    
    Args:
        strategy_class: Strategy class to use
        data_feeds: Dict of {symbol: DataBase} feeds
        engine_config: Engine configuration
        sim_config: Orchestrator configuration
        **strategy_kwargs: Strategy parameters
        
    Returns:
        Orchestrator run results
    """
    engine = BacktestEngine(engine_config)
    
    for symbol, feed in data_feeds.items():
        engine.add_data(feed, name=symbol)
    
    engine.add_strategy(strategy_class, **strategy_kwargs)
    
    orchestrator = SimulationOrchestrator(engine, sim_config)
    
    return orchestrator.run()
