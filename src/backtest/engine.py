"""
Backtrader Engine Wrapper and Configuration

Provides a high-level wrapper around backtrader's Cerebro engine with
proper configuration, lifecycle management, and simulation state tracking.

Key Classes:
- BacktestEngine: Main engine wrapper with initialization and run logic
- EngineConfig: Configuration dataclass for engine parameters

Features:
- Automatic commission and slippage configuration
- Support for single and multi-asset backtests
- Proper cash and margin handling
- State tracking (current bar timestamp, positions, orders)
- Logging of all engine events
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

try:
    import backtrader as bt
except ImportError:
    bt = None

from src.backtest.utils.logging import get_logger
from src.backtest.config.defaults import BacktestConfig


logger = get_logger(__name__)


@dataclass
class EngineConfig:
    """Configuration parameters for BacktestEngine."""
    
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0001  # 0.01%
    cash_required_pct: float = 0.1
    max_bars_back: int = 1000
    verbose: bool = False
    log_debug: bool = False
    
    @classmethod
    def from_backtest_config(cls, config: BacktestConfig) -> 'EngineConfig':
        """Create EngineConfig from BacktestConfig."""
        return cls(
            initial_capital=config.initial_capital,
            commission=config.commission,
            slippage=config.slippage,
            cash_required_pct=config.cash_required_pct,
            verbose=config.verbose,
        )


class BacktestEngine:
    """
    High-level wrapper around backtrader's Cerebro engine.
    
    Responsibilities:
    - Initialize Cerebro with proper configuration
    - Manage data feeds
    - Execute backtests with proper event sequencing
    - Track simulation state (current bar, positions, orders)
    - Provide logging at each simulation step
    
    Attributes:
        cerebro: The underlying Cerebro instance
        config: Engine configuration
        data_feeds: Dictionary of added data feeds
        strategies: List of added strategies
        state: Current simulation state
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        """
        Initialize the BacktestEngine.
        
        Args:
            config: EngineConfig with parameters. If None, defaults are used.
        """
        if bt is None:
            raise ImportError(
                "Backtrader is required to use BacktestEngine. "
                "Install with: pip install backtrader"
            )
        
        self.config = config or EngineConfig()
        self.logger = get_logger(f'backtest.engine')
        
        # Initialize Cerebro
        self.cerebro = bt.Cerebro()
        self._configure_cerebro()
        
        # State tracking
        self.data_feeds: Dict[str, bt.DataBase] = {}
        self.strategies: List[bt.Strategy] = []
        self.state: Dict[str, Any] = {
            'current_bar': 0,
            'current_timestamp': None,
            'running': False,
            'total_bars_processed': 0,
        }
        
        self.logger.info(
            f'BacktestEngine initialized with config: '
            f'capital=${self.config.initial_capital:.2f}, '
            f'commission={self.config.commission*100:.3f}%, '
            f'slippage={self.config.slippage*100:.3f}%'
        )
    
    def _configure_cerebro(self):
        """Configure Cerebro with engine parameters."""
        # Set initial capital
        self.cerebro.broker.setcash(self.config.initial_capital)
        
        # Set commission
        # Commission is per trade, use comminfo for proper broker commission
        self.cerebro.broker.setcommission(commission=self.config.commission)
        
        # Note: Slippage in backtrader is handled via order execution policies
        # We'll implement this during order processing in the orchestrator
        
        # Set max bars back for indicators
        self.cerebro.broker.set_checksubmit(checksubmit=True)
        
        self.logger.debug(
            f'Cerebro configured: '
            f'cash=${self.config.initial_capital:.2f}, '
            f'commission={self.config.commission}'
        )
    
    def add_data(self, data: bt.DataBase, name: Optional[str] = None) -> str:
        """
        Add a data feed to the engine.
        
        Args:
            data: Backtrader DataBase instance
            name: Optional name for the feed (defaults to symbol)
            
        Returns:
            Name of the added feed
        """
        # Get name from data's symbol or params
        if not name:
            name = getattr(data.params, 'symbol', 'data')
        
        self.cerebro.adddata(data)
        self.data_feeds[name] = data
        
        self.logger.info(f'Added data feed: {name}')
        
        return name
    
    def add_strategy(
        self,
        strategy_class: type,
        *args,
        **kwargs
    ) -> None:
        """
        Add a strategy to the engine.
        
        Args:
            strategy_class: Strategy class (must inherit from bt.Strategy)
            *args: Positional arguments for strategy initialization
            **kwargs: Keyword arguments for strategy parameters
        """
        strat = self.cerebro.addstrategy(strategy_class, *args, **kwargs)
        self.strategies.append(strat)
        
        strategy_name = getattr(strat, 'name', strategy_class.__name__)
        self.logger.info(f'Added strategy: {strategy_name}')
    
    def add_analyzer(self, analyzer_class: type, *args, **kwargs) -> None:
        """
        Add an analyzer to collect statistics.
        
        Args:
            analyzer_class: Analyzer class (must inherit from bt.Analyzer)
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        self.cerebro.addsizer(analyzer_class, *args, **kwargs)
        self.logger.debug(f'Added analyzer: {analyzer_class.__name__}')
    
    def run(self) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Returns:
            Dictionary containing:
            - strategies: List of executed strategy instances
            - final_value: Final portfolio value
            - total_bars: Total bars processed
            - state: Final engine state
        """
        if not self.data_feeds:
            raise ValueError("No data feeds added. Call add_data() first.")
        
        if not self.strategies:
            raise ValueError("No strategies added. Call add_strategy() first.")
        
        self.logger.info('Starting backtest run...')
        self.state['running'] = True
        
        try:
            # Run the backtest
            results = self.cerebro.run()
            
            # Collect results
            final_value = self.cerebro.broker.getvalue()
            
            self.state['running'] = False
            self.state['total_bars_processed'] = self.state['current_bar']
            
            self.logger.info(
                f'Backtest completed: '
                f'Final value: ${final_value:.2f}, '
                f'Bars processed: {self.state["total_bars_processed"]}'
            )
            
            return {
                'strategies': results,
                'final_value': final_value,
                'total_bars': self.state['total_bars_processed'],
                'state': self.state.copy(),
                'cerebro': self.cerebro,
            }
        
        except Exception as e:
            self.state['running'] = False
            self.logger.error(f'Backtest failed: {e}', exc_info=True)
            raise
    
    def get_current_timestamp(self) -> Optional[datetime]:
        """Get the current bar's timestamp."""
        return self.state.get('current_timestamp')
    
    def get_current_bar(self) -> int:
        """Get the current bar index."""
        return self.state.get('current_bar', 0)
    
    def get_portfolio_value(self) -> float:
        """Get the current portfolio value."""
        return self.cerebro.broker.getvalue()
    
    def get_cash(self) -> float:
        """Get the current cash balance."""
        return self.cerebro.broker.getcash()
    
    def get_state(self) -> Dict[str, Any]:
        """Get a copy of the current engine state."""
        return self.state.copy()
    
    def reset_state(self):
        """Reset engine state (for rerunning backtests)."""
        self.state = {
            'current_bar': 0,
            'current_timestamp': None,
            'running': False,
            'total_bars_processed': 0,
        }
        self.logger.debug('Engine state reset')
    
    @property
    def is_running(self) -> bool:
        """Check if backtest is currently running."""
        return self.state.get('running', False)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f'BacktestEngine('
            f'feeds={len(self.data_feeds)}, '
            f'strategies={len(self.strategies)}, '
            f'capital=${self.config.initial_capital:.2f}'
            f')'
        )
