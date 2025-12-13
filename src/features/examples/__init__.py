"""
Example feature implementations.

Provides ready-to-use feature implementations demonstrating the framework:

Return Features:
- SimpleReturn: Daily log return
- LogReturn: Daily log return
- CumulativeReturn: Cumulative return from period start

Rolling Statistics:
- RollingMean: Configurable rolling mean
- RollingVolatility: Configurable rolling volatility

Price Features:
- PriceRange: Daily price range as percentage
- HighLowRatio: Ratio of high to low price
- CloseToOpen: Ratio of close to open price

Volume Features:
- RelativeVolume: Volume relative to rolling mean

Dependent Features:
- VolatilityOfReturns: Volatility of returns (depends on SimpleReturn)
"""

from .basic import (
    SimpleReturn,
    LogReturn,
    CumulativeReturn,
    RollingMean,
    RollingVolatility,
    PriceRange,
    HighLowRatio,
    CloseToOpen,
    RelativeVolume,
    VolatilityOfReturns,
)

__all__ = [
    # Return features
    "SimpleReturn",
    "LogReturn",
    "CumulativeReturn",
    # Rolling statistics
    "RollingMean",
    "RollingVolatility",
    # Price features
    "PriceRange",
    "HighLowRatio",
    "CloseToOpen",
    # Volume features
    "RelativeVolume",
    # Dependent features
    "VolatilityOfReturns",
]
