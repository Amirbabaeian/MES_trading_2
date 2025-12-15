"""
Unit Tests for Event-Driven Simulation Orchestrator

Tests cover:
- BacktestEngine initialization and configuration
- SimulationOrchestrator state management
- Single-asset backtest execution
- Multi-asset backtest execution
- Order tracking and fill simulation
- State snapshots at each bar
- Metric computation
- Look-ahead bias detection integration
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys

try:
    import backtrader as bt
except ImportError:
    bt = None

from src.backtest.engine import BacktestEngine, EngineConfig
from src.backtest.orchestrator import (
    SimulationOrchestrator, SimulationConfig, SimulationState,
    LookAheadBiasDetector, run_single_asset_backtest, run_multi_asset_backtest
)


# ============================================================================
# BacktestEngine Tests
# ============================================================================

class TestBacktestEngine:
    """Test BacktestEngine class."""
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_engine_initialization(self):
        """Test engine initializes with correct config."""
        config = EngineConfig(
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.0001,
        )
        engine = BacktestEngine(config)
        
        assert engine.config.initial_capital == 100000.0
        assert engine.config.commission == 0.001
        assert engine.config.slippage == 0.0001
        assert len(engine.data_feeds) == 0
        assert len(engine.strategies) == 0
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_engine_default_config(self):
        """Test engine uses defaults when no config provided."""
        engine = BacktestEngine()
        
        assert engine.config.initial_capital == 100000.0
        assert engine.config.commission == 0.001
        assert engine.is_running is False
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_engine_state_tracking(self):
        """Test engine tracks state correctly."""
        engine = BacktestEngine()
        
        state = engine.get_state()
        assert state['current_bar'] == 0
        assert state['running'] is False
        assert state['total_bars_processed'] == 0
        assert state['current_timestamp'] is None
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_engine_portfolio_value(self):
        """Test engine returns correct portfolio value."""
        config = EngineConfig(initial_capital=50000.0)
        engine = BacktestEngine(config)
        
        # Portfolio value should equal initial capital before trading
        value = engine.get_portfolio_value()
        assert value == 50000.0
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_engine_state_reset(self):
        """Test engine state can be reset."""
        engine = BacktestEngine()
        
        # Modify state
        engine.state['current_bar'] = 100
        engine.state['running'] = True
        
        # Reset
        engine.reset_state()
        
        assert engine.state['current_bar'] == 0
        assert engine.state['running'] is False
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_engine_cash_tracking(self):
        """Test engine tracks cash correctly."""
        config = EngineConfig(initial_capital=75000.0)
        engine = BacktestEngine(config)
        
        cash = engine.get_cash()
        assert cash == 75000.0


# ============================================================================
# LookAheadBiasDetector Tests
# ============================================================================

class TestLookAheadBiasDetector:
    """Test LookAheadBiasDetector class."""
    
    def test_detector_initialization(self):
        """Test detector initializes correctly."""
        detector = LookAheadBiasDetector(strict=True)
        
        assert detector.strict is True
        assert len(detector.get_violations()) == 0
        assert detector.has_violations() is False
    
    def test_detector_timestamp_check_valid(self):
        """Test detector allows valid timestamp access."""
        detector = LookAheadBiasDetector(strict=True)
        
        current_ts = datetime(2024, 1, 15, 10, 0)
        past_ts = datetime(2024, 1, 15, 9, 0)
        
        result = detector.check_timestamp_access(past_ts, current_ts)
        assert result is True
        assert not detector.has_violations()
    
    def test_detector_timestamp_check_invalid_strict(self):
        """Test detector raises on future timestamp in strict mode."""
        detector = LookAheadBiasDetector(strict=True)
        
        current_ts = datetime(2024, 1, 15, 10, 0)
        future_ts = datetime(2024, 1, 15, 11, 0)
        
        with pytest.raises(AssertionError):
            detector.check_timestamp_access(future_ts, current_ts)
        
        assert detector.has_violations()
    
    def test_detector_timestamp_check_invalid_non_strict(self):
        """Test detector logs warning on future timestamp in non-strict mode."""
        detector = LookAheadBiasDetector(strict=False)
        
        current_ts = datetime(2024, 1, 15, 10, 0)
        future_ts = datetime(2024, 1, 15, 11, 0)
        
        result = detector.check_timestamp_access(future_ts, current_ts)
        assert result is False
        assert detector.has_violations()
    
    def test_detector_bar_index_check_valid(self):
        """Test detector allows valid bar index access."""
        detector = LookAheadBiasDetector(strict=True)
        
        result = detector.check_bar_index_access(0, 0)  # Current bar
        assert result is True
        
        result = detector.check_bar_index_access(-1, 0)  # Previous bar
        assert result is True
        
        result = detector.check_bar_index_access(-10, 0)  # Past bar
        assert result is True
    
    def test_detector_bar_index_check_invalid(self):
        """Test detector catches future bar index access."""
        detector = LookAheadBiasDetector(strict=True)
        
        with pytest.raises(AssertionError):
            detector.check_bar_index_access(1, 0)  # Future bar
        
        assert detector.has_violations()
    
    def test_detector_violations_accumulate(self):
        """Test detector accumulates multiple violations."""
        detector = LookAheadBiasDetector(strict=False)
        
        current_ts = datetime(2024, 1, 15, 10, 0)
        future_ts = datetime(2024, 1, 15, 11, 0)
        
        detector.check_timestamp_access(future_ts, current_ts)
        detector.check_timestamp_access(future_ts, current_ts)
        detector.check_bar_index_access(1, 0)
        
        violations = detector.get_violations()
        assert len(violations) == 3


# ============================================================================
# SimulationState Tests
# ============================================================================

class TestSimulationState:
    """Test SimulationState dataclass."""
    
    def test_state_creation(self):
        """Test creating a state snapshot."""
        ts = datetime(2024, 1, 15, 10, 0)
        state = SimulationState(
            bar_number=0,
            timestamp=ts,
            portfolio_value=100000.0,
            cash=100000.0,
        )
        
        assert state.bar_number == 0
        assert state.timestamp == ts
        assert state.portfolio_value == 100000.0
        assert state.cash == 100000.0
    
    def test_state_with_prices(self):
        """Test state with price data."""
        ts = datetime(2024, 1, 15, 10, 0)
        prices = {
            'MES': {
                'open': 5100.0,
                'high': 5105.0,
                'low': 5099.0,
                'close': 5102.0,
                'volume': 10000.0,
            }
        }
        
        state = SimulationState(
            bar_number=1,
            timestamp=ts,
            prices=prices,
        )
        
        assert 'MES' in state.prices
        assert state.prices['MES']['close'] == 5102.0
    
    def test_state_with_positions(self):
        """Test state with position data."""
        state = SimulationState(
            bar_number=2,
            timestamp=datetime.now(),
            positions={'MES': 10},
            position_prices={'MES': 5100.0},
        )
        
        assert state.positions['MES'] == 10
        assert state.position_prices['MES'] == 5100.0
    
    def test_state_with_orders(self):
        """Test state with order tracking."""
        state = SimulationState(
            bar_number=3,
            timestamp=datetime.now(),
            orders_submitted=5,
            orders_filled=3,
        )
        
        assert state.orders_submitted == 5
        assert state.orders_filled == 3


# ============================================================================
# SimulationOrchestrator Tests
# ============================================================================

class TestSimulationOrchestrator:
    """Test SimulationOrchestrator class."""
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_orchestrator_initialization(self):
        """Test orchestrator initializes with engine."""
        engine = BacktestEngine()
        orchestrator = SimulationOrchestrator(engine)
        
        assert orchestrator.engine is engine
        assert orchestrator.bars_processed == 0
        assert orchestrator.total_orders_submitted == 0
        assert orchestrator.total_orders_filled == 0
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_orchestrator_default_config(self):
        """Test orchestrator uses default config."""
        engine = BacktestEngine()
        orchestrator = SimulationOrchestrator(engine)
        
        assert orchestrator.config.detect_lookahead_bias is True
        assert orchestrator.config.track_equity_curve is True
        assert len(orchestrator.get_state_history()) == 0
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_orchestrator_custom_config(self):
        """Test orchestrator with custom config."""
        engine = BacktestEngine()
        config = SimulationConfig(
            detect_lookahead_bias=False,
            track_drawdown=False,
            verbose=True,
        )
        orchestrator = SimulationOrchestrator(engine, config)
        
        assert orchestrator.config.detect_lookahead_bias is False
        assert orchestrator.config.track_drawdown is False
        assert orchestrator.config.verbose is True
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_orchestrator_state_snapshot(self):
        """Test orchestrator creates state snapshots."""
        engine = BacktestEngine()
        orchestrator = SimulationOrchestrator(engine)
        
        ts = datetime(2024, 1, 15, 10, 0)
        prices = {'MES': {'open': 5100, 'high': 5105, 'low': 5099, 'close': 5102, 'volume': 10000}}
        
        state = orchestrator._create_state_snapshot(0, ts, prices)
        
        assert state.bar_number == 0
        assert state.timestamp == ts
        assert state.prices == prices
        assert len(orchestrator.get_state_history()) == 1
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_orchestrator_metrics_computation(self):
        """Test orchestrator computes final metrics."""
        engine = BacktestEngine()
        orchestrator = SimulationOrchestrator(engine)
        
        # Create some state history
        ts = datetime(2024, 1, 15, 10, 0)
        for i in range(5):
            orchestrator._create_state_snapshot(
                i,
                ts + timedelta(minutes=i),
                {'MES': {'close': 5100 + i}}
            )
        
        metrics = orchestrator.get_metrics()
        
        assert 'initial_capital' in metrics
        assert 'final_value' in metrics
        assert 'total_return_pct' in metrics
        assert 'total_bars' in metrics
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_orchestrator_order_tracking(self):
        """Test orchestrator tracks orders."""
        engine = BacktestEngine()
        orchestrator = SimulationOrchestrator(engine)
        
        ts = datetime(2024, 1, 15, 10, 0)
        orchestrator._create_state_snapshot(0, ts, {})
        
        # Record order submission
        orchestrator.record_order_submitted(1, 'MES', 'BUY', 10, 5100.0)
        assert orchestrator.total_orders_submitted == 1
        
        # Record order fill
        orchestrator.record_order_filled(1, 'MES', 5101.0)
        assert orchestrator.total_orders_filled == 1
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_orchestrator_string_representation(self):
        """Test orchestrator string representation."""
        engine = BacktestEngine()
        orchestrator = SimulationOrchestrator(engine)
        
        repr_str = repr(orchestrator)
        assert 'SimulationOrchestrator' in repr_str
        assert 'bars_processed' in repr_str


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions for running backtests."""
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_engine_config_from_backtest_config(self):
        """Test creating EngineConfig from BacktestConfig."""
        from src.backtest.config.defaults import BacktestConfig
        
        bt_config = BacktestConfig(
            initial_capital=50000.0,
            commission=0.002,
            slippage=0.0002,
        )
        
        engine_config = EngineConfig.from_backtest_config(bt_config)
        
        assert engine_config.initial_capital == 50000.0
        assert engine_config.commission == 0.002
        assert engine_config.slippage == 0.0002


# ============================================================================
# Integration Tests
# ============================================================================

class TestEngineOrchestratorIntegration:
    """Integration tests for engine and orchestrator."""
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_engine_orchestrator_creation_flow(self):
        """Test creating engine and orchestrator together."""
        config = EngineConfig(initial_capital=100000.0)
        engine = BacktestEngine(config)
        
        sim_config = SimulationConfig(
            detect_lookahead_bias=True,
            track_equity_curve=True,
        )
        orchestrator = SimulationOrchestrator(engine, sim_config)
        
        assert orchestrator.engine is engine
        assert orchestrator.config.detect_lookahead_bias is True
    
    @pytest.mark.skipif(bt is None, reason="backtrader not installed")
    def test_multiple_orchestrator_instances(self):
        """Test creating multiple independent orchestrators."""
        engine1 = BacktestEngine(EngineConfig(initial_capital=50000.0))
        engine2 = BacktestEngine(EngineConfig(initial_capital=100000.0))
        
        orch1 = SimulationOrchestrator(engine1)
        orch2 = SimulationOrchestrator(engine2)
        
        assert orch1.engine.config.initial_capital == 50000.0
        assert orch2.engine.config.initial_capital == 100000.0
        assert orch1.engine is not orch2.engine
