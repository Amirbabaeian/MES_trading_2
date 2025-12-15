"""
Example feature implementations demonstrating the framework.

Includes:
- SimpleReturn: Daily simple percentage return
- LogReturn: Daily log return
- RollingMean: Configurable rolling mean of prices
- RollingVolatility: Configurable rolling volatility
- PriceRange: Daily price range as percentage
- CumulativeReturn: Cumulative return over period

All implementations are deterministic and support parameterization.
"""

import numpy as np
import pandas as pd

from src.features.core.base import Feature
from src.features.core.errors import ComputationError, ParameterValidationError


# ============================================================================
# Basic Return Features
# ============================================================================

class SimpleReturn(Feature):
    """
    Daily simple percentage return.
    
    Formula: (close - previous_close) / previous_close
    
    Parameters: None
    Dependencies: None
    """
    
    def __init__(self):
        """Initialize simple return feature."""
        super().__init__(
            name="simple_return",
            description="Daily simple percentage return (log-based)",
            parameters={},
            dependencies=[],
        )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute simple returns.
        
        Args:
            data: DataFrame with 'close' column
            **kwargs: Unused
        
        Returns:
            Series of simple returns
        """
        try:
            # Use log returns which are more numerically stable
            returns = np.log(data['close'] / data['close'].shift(1))
            return returns
        except Exception as e:
            raise ComputationError(self.name, f"Failed to compute returns: {e}")


class LogReturn(Feature):
    """
    Daily log return (natural logarithm).
    
    Formula: log(close / previous_close)
    
    Parameters: None
    Dependencies: None
    """
    
    def __init__(self):
        """Initialize log return feature."""
        super().__init__(
            name="log_return",
            description="Daily log return",
            parameters={},
            dependencies=[],
        )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute log returns.
        
        Args:
            data: DataFrame with 'close' column
            **kwargs: Unused
        
        Returns:
            Series of log returns
        """
        try:
            returns = np.log(data['close'] / data['close'].shift(1))
            return returns
        except Exception as e:
            raise ComputationError(self.name, f"Failed to compute log returns: {e}")


class CumulativeReturn(Feature):
    """
    Cumulative return from start of period.
    
    Formula: (close - first_close) / first_close
    
    Parameters: None
    Dependencies: None
    """
    
    def __init__(self):
        """Initialize cumulative return feature."""
        super().__init__(
            name="cumulative_return",
            description="Cumulative return from period start",
            parameters={},
            dependencies=[],
        )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute cumulative returns.
        
        Args:
            data: DataFrame with 'close' column
            **kwargs: Unused
        
        Returns:
            Series of cumulative returns
        """
        try:
            first_close = data['close'].iloc[0]
            cum_return = (data['close'] - first_close) / first_close
            return cum_return
        except Exception as e:
            raise ComputationError(self.name, f"Failed to compute cumulative returns: {e}")


# ============================================================================
# Rolling Statistics
# ============================================================================

class RollingMean(Feature):
    """
    Configurable rolling mean of closing prices.
    
    Parameters:
        - window: Number of periods for rolling window (default: 20)
    
    Dependencies: None
    """
    
    def __init__(self, window: int = 20):
        """
        Initialize rolling mean feature.
        
        Args:
            window: Rolling window size in periods
        """
        super().__init__(
            name=f"rolling_mean_{window}",
            description=f"Rolling mean of close prices (window={window})",
            parameters={"window": window},
            dependencies=[],
        )
    
    def validate_parameters(self) -> None:
        """Validate window parameter."""
        window = self.parameters.get("window")
        if not isinstance(window, int) or window < 1:
            raise ParameterValidationError(
                self.name,
                f"window must be positive integer, got {window}"
            )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute rolling mean.
        
        Args:
            data: DataFrame with 'close' column
            **kwargs: Unused
        
        Returns:
            Series of rolling mean values
        """
        try:
            window = self.parameters["window"]
            rolling_mean = data['close'].rolling(window=window, min_periods=1).mean()
            return rolling_mean
        except Exception as e:
            raise ComputationError(self.name, f"Failed to compute rolling mean: {e}")


class RollingVolatility(Feature):
    """
    Configurable rolling volatility of returns.
    
    Calculated as standard deviation of log returns.
    
    Parameters:
        - window: Number of periods for rolling window (default: 20)
        - annualize: Whether to annualize volatility assuming 252 trading days (default: False)
    
    Dependencies: None
    """
    
    def __init__(self, window: int = 20, annualize: bool = False):
        """
        Initialize rolling volatility feature.
        
        Args:
            window: Rolling window size in periods
            annualize: Whether to annualize the volatility
        """
        annualize_str = "_annualized" if annualize else ""
        super().__init__(
            name=f"rolling_volatility_{window}{annualize_str}",
            description=f"Rolling volatility (window={window}, annualize={annualize})",
            parameters={
                "window": window,
                "annualize": annualize,
            },
            dependencies=[],
        )
    
    def validate_parameters(self) -> None:
        """Validate parameters."""
        window = self.parameters.get("window")
        if not isinstance(window, int) or window < 1:
            raise ParameterValidationError(
                self.name,
                f"window must be positive integer, got {window}"
            )
        
        annualize = self.parameters.get("annualize")
        if not isinstance(annualize, bool):
            raise ParameterValidationError(
                self.name,
                f"annualize must be bool, got {type(annualize)}"
            )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute rolling volatility.
        
        Args:
            data: DataFrame with 'close' column
            **kwargs: Unused
        
        Returns:
            Series of rolling volatility values
        """
        try:
            window = self.parameters["window"]
            annualize = self.parameters["annualize"]
            
            # Compute log returns
            returns = np.log(data['close'] / data['close'].shift(1))
            
            # Compute rolling standard deviation
            volatility = returns.rolling(window=window, min_periods=1).std()
            
            # Annualize if requested (252 trading days per year)
            if annualize:
                volatility = volatility * np.sqrt(252)
            
            return volatility
        except Exception as e:
            raise ComputationError(self.name, f"Failed to compute rolling volatility: {e}")


# ============================================================================
# Price-based Features
# ============================================================================

class PriceRange(Feature):
    """
    Daily price range as percentage of close.
    
    Formula: (high - low) / close * 100
    
    Parameters: None
    Dependencies: None
    """
    
    def __init__(self):
        """Initialize price range feature."""
        super().__init__(
            name="price_range_pct",
            description="Daily price range as percentage of close",
            parameters={},
            dependencies=[],
        )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute price range percentage.
        
        Args:
            data: DataFrame with 'high', 'low', 'close' columns
            **kwargs: Unused
        
        Returns:
            Series of price range percentages
        """
        try:
            price_range = (data['high'] - data['low']) / data['close'] * 100
            return price_range
        except Exception as e:
            raise ComputationError(self.name, f"Failed to compute price range: {e}")


class HighLowRatio(Feature):
    """
    Ratio of high to low price.
    
    Formula: high / low
    
    Parameters: None
    Dependencies: None
    """
    
    def __init__(self):
        """Initialize high/low ratio feature."""
        super().__init__(
            name="high_low_ratio",
            description="Ratio of high to low price",
            parameters={},
            dependencies=[],
        )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute high/low ratio.
        
        Args:
            data: DataFrame with 'high', 'low' columns
            **kwargs: Unused
        
        Returns:
            Series of high/low ratios
        """
        try:
            ratio = data['high'] / data['low']
            return ratio
        except Exception as e:
            raise ComputationError(self.name, f"Failed to compute high/low ratio: {e}")


class CloseToOpen(Feature):
    """
    Ratio of close to open price.
    
    Formula: close / open
    
    Parameters: None
    Dependencies: None
    """
    
    def __init__(self):
        """Initialize close/open ratio feature."""
        super().__init__(
            name="close_to_open",
            description="Ratio of close to open price",
            parameters={},
            dependencies=[],
        )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute close/open ratio.
        
        Args:
            data: DataFrame with 'close', 'open' columns
            **kwargs: Unused
        
        Returns:
            Series of close/open ratios
        """
        try:
            ratio = data['close'] / data['open']
            return ratio
        except Exception as e:
            raise ComputationError(self.name, f"Failed to compute close/open ratio: {e}")


# ============================================================================
# Volume-based Features
# ============================================================================

class RelativeVolume(Feature):
    """
    Volume relative to rolling mean volume.
    
    Formula: volume / rolling_mean(volume)
    
    Parameters:
        - window: Number of periods for rolling mean (default: 20)
    
    Dependencies: None
    """
    
    def __init__(self, window: int = 20):
        """
        Initialize relative volume feature.
        
        Args:
            window: Rolling window size for volume mean
        """
        super().__init__(
            name=f"relative_volume_{window}",
            description=f"Volume relative to {window}-period mean",
            parameters={"window": window},
            dependencies=[],
        )
    
    def validate_parameters(self) -> None:
        """Validate window parameter."""
        window = self.parameters.get("window")
        if not isinstance(window, int) or window < 1:
            raise ParameterValidationError(
                self.name,
                f"window must be positive integer, got {window}"
            )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute relative volume.
        
        Args:
            data: DataFrame with 'volume' column
            **kwargs: Unused
        
        Returns:
            Series of relative volume values
        """
        try:
            window = self.parameters["window"]
            mean_volume = data['volume'].rolling(window=window, min_periods=1).mean()
            # Avoid division by zero
            relative_vol = data['volume'] / mean_volume.replace(0, 1)
            return relative_vol
        except Exception as e:
            raise ComputationError(self.name, f"Failed to compute relative volume: {e}")


# ============================================================================
# Higher-level Features (with dependencies)
# ============================================================================

class VolatilityOfReturns(Feature):
    """
    Volatility computed from simple returns.
    
    Demonstrates feature dependencies - requires simple_return feature.
    
    Parameters:
        - window: Rolling window for volatility calculation (default: 20)
    
    Dependencies: ['simple_return']
    """
    
    def __init__(self, window: int = 20):
        """
        Initialize volatility of returns feature.
        
        Args:
            window: Rolling window size
        """
        super().__init__(
            name=f"volatility_of_returns_{window}",
            description=f"Volatility of returns (window={window})",
            parameters={"window": window},
            dependencies=["simple_return"],
        )
    
    def validate_parameters(self) -> None:
        """Validate window parameter."""
        window = self.parameters.get("window")
        if not isinstance(window, int) or window < 1:
            raise ParameterValidationError(
                self.name,
                f"window must be positive integer, got {window}"
            )
    
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Compute volatility of returns.
        
        Args:
            data: DataFrame with 'simple_return' column (from dependency)
            **kwargs: Unused
        
        Returns:
            Series of volatility values
        """
        try:
            window = self.parameters["window"]
            if 'simple_return' not in data.columns:
                raise ValueError("Missing 'simple_return' column - dependency not computed")
            
            volatility = data['simple_return'].rolling(window=window, min_periods=1).std()
            return volatility
        except Exception as e:
            raise ComputationError(self.name, f"Failed to compute volatility of returns: {e}")
