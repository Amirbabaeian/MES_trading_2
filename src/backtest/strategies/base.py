"""
Base Strategy Class

Provides a foundation for strategy implementations with common utilities
for order management, logging, and metrics tracking.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Any

try:
    import backtrader as bt
except ImportError:
    bt = None

from src.backtest.utils.logging import get_logger


class BaseStrategy(bt.Strategy if bt else object):
    """
    Base strategy class that wraps backtrader.Strategy with common utilities.
    
    Provides:
    - Structured logging for trade entry/exit
    - Trade-level PnL tracking
    - Performance metrics collection
    - Order management utilities
    
    Attributes:
        name (str): Strategy identifier
        logger: Strategy-specific logger
        trades_log (list): Log of all completed trades
        signals_log (list): Log of all trading signals
        active_orders (dict): Tracking of active orders
    """
    
    params = (
        ('strategy_name', 'BaseStrategy'),
        ('log_trades', True),
        ('log_signals', True),
        ('track_metrics', True),
    )
    
    def __init__(self, *args, **kwargs):
        """Initialize the base strategy with logging and tracking."""
        super().__init__(*args, **kwargs)
        
        self.name = self.params.strategy_name
        self.logger = get_logger(f'backtest.{self.name}')
        
        # Trade tracking
        self.trades_log = []
        self.signals_log = []
        self.active_orders = {}
        
        # Metrics
        self.entry_price = None
        self.entry_size = None
        self.entry_datetime = None
        
        self.logger.info(f'Strategy initialized: {self.name}')
    
    def log(self, msg: str, level: str = 'info', *args, **kwargs):
        """
        Log a message with timestamp and close price.
        
        Args:
            msg: Message to log
            level: Log level (debug, info, warning, error)
            *args, **kwargs: Arguments for logger
        """
        dt = self.datas[0].datetime.date(0)
        log_msg = f'{dt.isoformat()} - {msg}'
        
        log_func = getattr(self.logger, level, self.logger.info)
        log_func(log_msg)
    
    def notify_order(self, order):
        """
        Process order notifications.
        
        Called by backtrader when order status changes.
        Logs submission, execution, and rejection of orders.
        
        Args:
            order: Order object from backtrader
        """
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted or accepted
            if order not in self.active_orders:
                self.active_orders[order] = {
                    'created': datetime.now(),
                    'size': order.created.size,
                    'price': order.created.price,
                    'side': 'BUY' if order.created.size > 0 else 'SELL',
                }
                if self.params.log_signals:
                    self.log(
                        f'ORDER SUBMITTED - {self.active_orders[order]["side"]} '
                        f'{abs(order.created.size)} @ ${order.created.price:.2f}',
                        level='info'
                    )
        elif order.status in [order.Completed]:
            # Order executed
            if order in self.active_orders:
                order_info = self.active_orders.pop(order)
                if self.params.log_signals:
                    self.log(
                        f'ORDER EXECUTED - {order_info["side"]} '
                        f'{abs(order_info["size"])} @ ${order.executed.price:.2f}',
                        level='info'
                    )
            
            # Update entry price for long entries
            if order.isbuy() and self.position.size > 0:
                self.entry_price = order.executed.price
                self.entry_size = order.executed.size
                self.entry_datetime = self.datas[0].datetime.datetime(0)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # Order canceled, margin, or rejected
            if order in self.active_orders:
                order_info = self.active_orders.pop(order)
                if self.params.log_signals:
                    self.log(
                        f'ORDER {order.status.name} - {order_info["side"]} '
                        f'{abs(order_info["size"])}',
                        level='warning'
                    )
    
    def notify_trade(self, trade):
        """
        Process trade notifications.
        
        Called by backtrader when a trade closes (position fully exited).
        Logs trade details including entry, exit, size, and PnL.
        
        Args:
            trade: Trade object from backtrader
        """
        if trade.isclosed:
            trade_info = {
                'entry_datetime': trade.dtopen,
                'entry_price': trade.price,
                'exit_datetime': trade.dtclose,
                'exit_price': trade.barclose,
                'size': trade.size,
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,  # PnL after commission
                'return_pct': (trade.pnlcomm / abs(trade.value)) * 100 if trade.value else 0,
            }
            
            self.trades_log.append(trade_info)
            
            if self.params.log_trades:
                self.log(
                    f'TRADE CLOSED - Entry: {trade.price:.2f}, Exit: {trade.barclose:.2f}, '
                    f'Size: {trade.size}, PnL: ${trade.pnl:.2f}, PnL (comm): ${trade.pnlcomm:.2f}',
                    level='info'
                )
    
    def log_signal(self, signal_type: str, details: Dict[str, Any]):
        """
        Log a trading signal.
        
        Args:
            signal_type: Type of signal (BUY, SELL, etc.)
            details: Dictionary with signal details
        """
        if self.params.log_signals:
            signal_info = {
                'datetime': self.datas[0].datetime.datetime(0),
                'type': signal_type,
                'details': details,
                'close': self.datas[0].close[0],
                'bar': len(self),
            }
            self.signals_log.append(signal_info)
            self.log(f'SIGNAL: {signal_type} - {details}', level='debug')
    
    def get_trade_stats(self) -> Dict[str, Any]:
        """
        Get statistics for completed trades.
        
        Returns:
            Dictionary with trade statistics (win rate, average PnL, etc.)
        """
        if not self.trades_log:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
            }
        
        winning = [t for t in self.trades_log if t['pnlcomm'] > 0]
        losing = [t for t in self.trades_log if t['pnlcomm'] < 0]
        
        return {
            'total_trades': len(self.trades_log),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(self.trades_log) * 100 if self.trades_log else 0.0,
            'avg_win': sum(t['pnlcomm'] for t in winning) / len(winning) if winning else 0.0,
            'avg_loss': sum(t['pnlcomm'] for t in losing) / len(losing) if losing else 0.0,
            'total_pnl': sum(t['pnlcomm'] for t in self.trades_log),
            'avg_pnl': sum(t['pnlcomm'] for t in self.trades_log) / len(self.trades_log) if self.trades_log else 0.0,
        }
    
    def next(self):
        """
        Main strategy logic - called for each new bar.
        
        Override this method in subclasses to implement specific strategy logic.
        """
        # Base implementation - override in subclasses
        pass
    
    def stop(self):
        """
        Called when backtest is finished.
        
        Logs final statistics and cleanup.
        """
        if self.params.log_trades:
            stats = self.get_trade_stats()
            self.log(
                f'BACKTEST FINISHED - Total Trades: {stats["total_trades"]}, '
                f'Win Rate: {stats["win_rate"]:.1f}%, Total PnL: ${stats["total_pnl"]:.2f}',
                level='info'
            )
