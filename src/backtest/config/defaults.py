"""
Default Configuration Management

Provides configuration management for backtesting with defaults for:
- Backtest parameters (initial capital, commission, slippage)
- MES futures specifications (contract size, tick size, margin)
- Strategy parameters
- Environment-based configurations (dev vs production)

Supports loading from YAML, JSON, or Python dictionaries.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from src.backtest.contracts import ContractRegistry


@dataclass
class FuturesSpec:
    """Specifications for a futures contract (e.g., MES - Micro E-mini S&P 500)."""
    
    symbol: str
    multiplier: float  # Points to dollars multiplier
    tick_size: float  # Minimum price movement
    margin_requirement: float  # Initial margin per contract
    contract_size: float  # Units per contract (usually 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FuturesSpec':
        """Create from dictionary."""
        return cls(**data)


class BacktestConfig:
    """
    Configuration manager for backtesting.
    
    Handles:
    - Default values for backtest parameters
    - Futures contract specifications
    - Strategy-specific parameters
    - Environment-based overrides
    
    Attributes:
        initial_capital (float): Starting cash for backtest
        commission (float): Trading commission (e.g., 0.001 = 0.1%)
        slippage (float): Price slippage per trade (e.g., 0.0001 = 0.01%)
        cash_required_pct (float): Percent of cash for margin
        futures_specs (dict): Contract specifications by symbol
        strategy_params (dict): Default strategy parameters
        environment (str): 'dev' or 'production'
        verbose (bool): Verbose logging
    """
    
    # Default MES futures specifications
    DEFAULT_MES_SPEC = FuturesSpec(
        symbol='MES',
        multiplier=5.0,  # $5 per index point
        tick_size=0.25,  # 0.25 index point
        margin_requirement=300.0,  # Approximate, varies by broker
        contract_size=1.0,
    )
    
    # Default ES futures specifications
    DEFAULT_ES_SPEC = FuturesSpec(
        symbol='ES',
        multiplier=50.0,  # $50 per index point
        tick_size=0.25,  # 0.25 index point
        margin_requirement=12500.0,  # Approximate, varies by broker
        contract_size=1.0,
    )
    
    # Default configuration values
    DEFAULTS = {
        'initial_capital': 100000.0,
        'commission': 0.001,  # 0.1%
        'slippage': 0.0001,  # 0.01%
        'cash_required_pct': 0.1,  # 10% cash requirement
        'environment': 'dev',
        'verbose': False,
        'futures_specs': {
            'MES': DEFAULT_MES_SPEC,
            'ES': DEFAULT_ES_SPEC,
        },
        'strategy_params': {},
    }
    
    def __init__(
        self,
        initial_capital: float = DEFAULTS['initial_capital'],
        commission: float = DEFAULTS['commission'],
        slippage: float = DEFAULTS['slippage'],
        cash_required_pct: float = DEFAULTS['cash_required_pct'],
        environment: str = DEFAULTS['environment'],
        verbose: bool = DEFAULTS['verbose'],
        futures_specs: Optional[Dict[str, FuturesSpec]] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize configuration.
        
        Args:
            initial_capital: Starting capital for backtest
            commission: Trading commission (fraction)
            slippage: Price slippage (fraction)
            cash_required_pct: Percent of portfolio to hold as cash
            environment: 'dev' or 'production'
            verbose: Enable verbose logging
            futures_specs: Dictionary of FuturesSpec by symbol
            strategy_params: Strategy-specific parameters
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.cash_required_pct = cash_required_pct
        self.environment = environment
        self.verbose = verbose
        self.futures_specs = futures_specs or self.DEFAULTS['futures_specs'].copy()
        self.strategy_params = strategy_params or {}
    
    def get_futures_spec(self, symbol: str) -> Optional[FuturesSpec]:
        """
        Get futures specification for a symbol.
        
        Args:
            symbol: Futures symbol (e.g., 'MES', 'ES')
            
        Returns:
            FuturesSpec if found, None otherwise
        """
        return self.futures_specs.get(symbol)
    
    def add_futures_spec(self, symbol: str, spec: FuturesSpec):
        """
        Add or update a futures specification.
        
        Args:
            symbol: Futures symbol
            spec: FuturesSpec object
        """
        self.futures_specs[symbol] = spec
    
    def set_strategy_param(self, param_name: str, value: Any):
        """
        Set a strategy parameter.
        
        Args:
            param_name: Parameter name
            value: Parameter value
        """
        self.strategy_params[param_name] = value
    
    def get_strategy_param(self, param_name: str, default: Any = None) -> Any:
        """
        Get a strategy parameter.
        
        Args:
            param_name: Parameter name
            default: Default value if not found
            
        Returns:
            Parameter value or default
        """
        return self.strategy_params.get(param_name, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'initial_capital': self.initial_capital,
            'commission': self.commission,
            'slippage': self.slippage,
            'cash_required_pct': self.cash_required_pct,
            'environment': self.environment,
            'verbose': self.verbose,
            'futures_specs': {
                symbol: spec.to_dict()
                for symbol, spec in self.futures_specs.items()
            },
            'strategy_params': self.strategy_params.copy(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestConfig':
        """
        Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            BacktestConfig instance
        """
        # Convert futures specs
        futures_specs = {}
        if 'futures_specs' in data:
            for symbol, spec_data in data['futures_specs'].items():
                if isinstance(spec_data, dict):
                    futures_specs[symbol] = FuturesSpec.from_dict(spec_data)
                else:
                    futures_specs[symbol] = spec_data
        
        return cls(
            initial_capital=data.get('initial_capital', cls.DEFAULTS['initial_capital']),
            commission=data.get('commission', cls.DEFAULTS['commission']),
            slippage=data.get('slippage', cls.DEFAULTS['slippage']),
            cash_required_pct=data.get('cash_required_pct', cls.DEFAULTS['cash_required_pct']),
            environment=data.get('environment', cls.DEFAULTS['environment']),
            verbose=data.get('verbose', cls.DEFAULTS['verbose']),
            futures_specs=futures_specs if futures_specs else None,
            strategy_params=data.get('strategy_params', {}),
        )
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'BacktestConfig':
        """
        Load configuration from YAML file.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            BacktestConfig instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ImportError: If PyYAML is not installed
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML configuration. "
                "Install with: pip install pyyaml"
            )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, file_path: str) -> 'BacktestConfig':
        """
        Load configuration from JSON file.
        
        Args:
            file_path: Path to JSON configuration file
            
        Returns:
            BacktestConfig instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_env(cls) -> 'BacktestConfig':
        """
        Create configuration from environment variables.
        
        Supported environment variables:
        - BACKTEST_INITIAL_CAPITAL
        - BACKTEST_COMMISSION
        - BACKTEST_SLIPPAGE
        - BACKTEST_ENVIRONMENT (dev or production)
        - BACKTEST_VERBOSE
        
        Returns:
            BacktestConfig instance with environment-based values
        """
        return cls(
            initial_capital=float(os.getenv('BACKTEST_INITIAL_CAPITAL', cls.DEFAULTS['initial_capital'])),
            commission=float(os.getenv('BACKTEST_COMMISSION', cls.DEFAULTS['commission'])),
            slippage=float(os.getenv('BACKTEST_SLIPPAGE', cls.DEFAULTS['slippage'])),
            environment=os.getenv('BACKTEST_ENVIRONMENT', cls.DEFAULTS['environment']),
            verbose=os.getenv('BACKTEST_VERBOSE', '').lower() in ('true', '1', 'yes'),
        )
    
    def get_contract_registry(self) -> ContractRegistry:
        """
        Get a ContractRegistry populated with standard contracts.
        
        Returns:
            ContractRegistry with MES, ES, NQ, and YM contracts
            
        Example:
            >>> config = BacktestConfig()
            >>> registry = config.get_contract_registry()
            >>> mes = registry.get('MES')
        """
        from src.backtest.config.mes_specs import (
            MES_CONTRACT, ES_CONTRACT, NQ_CONTRACT, YM_CONTRACT
        )
        
        registry = ContractRegistry()
        registry.register(MES_CONTRACT)
        registry.register(ES_CONTRACT)
        registry.register(NQ_CONTRACT)
        registry.register(YM_CONTRACT)
        
        return registry
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BacktestConfig(initial_capital={self.initial_capital}, "
            f"commission={self.commission}, slippage={self.slippage}, "
            f"environment='{self.environment}')"
        )
