"""
Schema definitions for Parquet data validation.

Provides schema definitions for:
- OHLCV (Open, High, Low, Close, Volume) data
- Feature datasets
- Raw, cleaned, and features layers
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class SchemaEnforcementMode(str, Enum):
    """Schema enforcement modes for read/write operations."""
    STRICT = "strict"  # Fail if schema doesn't match exactly
    COERCE = "coerce"  # Try to coerce data types
    IGNORE = "ignore"  # Warn but don't fail


@dataclass
class ColumnSchema:
    """Definition of a single column in a dataset."""
    name: str
    data_type: str  # Parquet type: int32, int64, float, double, string, timestamp[ns], etc.
    nullable: bool = True
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "description": self.description,
        }


@dataclass
class DataSchema:
    """Schema definition for a dataset."""
    name: str
    version: str
    description: str
    columns: List[ColumnSchema]
    metadata: Optional[Dict[str, str]] = None
    
    def get_column_names(self) -> List[str]:
        """Get list of column names."""
        return [col.name for col in self.columns]
    
    def get_required_columns(self) -> List[str]:
        """Get list of non-nullable column names."""
        return [col.name for col in self.columns if not col.nullable]
    
    def get_optional_columns(self) -> List[str]:
        """Get list of nullable column names."""
        return [col.name for col in self.columns if col.nullable]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "columns": [col.to_dict() for col in self.columns],
            "metadata": self.metadata or {},
        }


# ============================================================================
# Predefined Schemas
# ============================================================================

# OHLCV Schema - used for raw and cleaned data layers
OHLCV_SCHEMA = DataSchema(
    name="ohlcv",
    version="1.0.0",
    description="Open, High, Low, Close, Volume candlestick data",
    columns=[
        ColumnSchema(
            name="timestamp",
            data_type="timestamp[ns]",
            nullable=False,
            description="Trade timestamp (UTC)"
        ),
        ColumnSchema(
            name="open",
            data_type="double",
            nullable=False,
            description="Opening price"
        ),
        ColumnSchema(
            name="high",
            data_type="double",
            nullable=False,
            description="Highest price"
        ),
        ColumnSchema(
            name="low",
            data_type="double",
            nullable=False,
            description="Lowest price"
        ),
        ColumnSchema(
            name="close",
            data_type="double",
            nullable=False,
            description="Closing price"
        ),
        ColumnSchema(
            name="volume",
            data_type="int64",
            nullable=False,
            description="Volume in contracts"
        ),
    ],
    metadata={
        "asset": "dynamic",
        "frequency": "configurable",
        "source": "market_data_feed",
    }
)

# Price Volatility Features Schema
PRICE_VOLATILITY_SCHEMA = DataSchema(
    name="price_volatility",
    version="1.0.0",
    description="Price volatility indicators and features",
    columns=[
        ColumnSchema(
            name="timestamp",
            data_type="timestamp[ns]",
            nullable=False,
            description="Feature timestamp (UTC)"
        ),
        ColumnSchema(
            name="volatility_30d",
            data_type="double",
            nullable=True,
            description="30-day rolling volatility"
        ),
        ColumnSchema(
            name="volatility_60d",
            data_type="double",
            nullable=True,
            description="60-day rolling volatility"
        ),
        ColumnSchema(
            name="volatility_ratio",
            data_type="double",
            nullable=True,
            description="Ratio of 30d to 60d volatility"
        ),
        ColumnSchema(
            name="price_range_pct",
            data_type="double",
            nullable=True,
            description="Daily price range as percentage"
        ),
    ],
    metadata={
        "category": "volatility",
        "window_30d": "rolling_window",
        "window_60d": "rolling_window",
    }
)

# Momentum Indicators Features Schema
MOMENTUM_INDICATORS_SCHEMA = DataSchema(
    name="momentum_indicators",
    version="1.0.0",
    description="Momentum and trend indicators",
    columns=[
        ColumnSchema(
            name="timestamp",
            data_type="timestamp[ns]",
            nullable=False,
            description="Feature timestamp (UTC)"
        ),
        ColumnSchema(
            name="rsi",
            data_type="double",
            nullable=True,
            description="Relative Strength Index (14-period)"
        ),
        ColumnSchema(
            name="macd",
            data_type="double",
            nullable=True,
            description="MACD line (12-26 EMA)"
        ),
        ColumnSchema(
            name="macd_signal",
            data_type="double",
            nullable=True,
            description="MACD signal line (9-period EMA)"
        ),
        ColumnSchema(
            name="momentum",
            data_type="double",
            nullable=True,
            description="Price momentum (10-period)"
        ),
    ],
    metadata={
        "category": "momentum",
        "rsi_period": "14",
        "macd_fast": "12",
        "macd_slow": "26",
    }
)

# Volume Profile Features Schema
VOLUME_PROFILE_SCHEMA = DataSchema(
    name="volume_profile",
    version="1.0.0",
    description="Volume profile and distribution metrics",
    columns=[
        ColumnSchema(
            name="timestamp",
            data_type="timestamp[ns]",
            nullable=False,
            description="Feature timestamp (UTC)"
        ),
        ColumnSchema(
            name="volume_profile_high",
            data_type="double",
            nullable=True,
            description="Price level with highest volume"
        ),
        ColumnSchema(
            name="volume_at_price_high",
            data_type="int64",
            nullable=True,
            description="Volume at highest price level"
        ),
        ColumnSchema(
            name="volume_concentration",
            data_type="double",
            nullable=True,
            description="Percentage of volume in top 5 price levels"
        ),
    ],
    metadata={
        "category": "volume",
        "granularity": "daily",
    }
)

# Registry of all available schemas
SCHEMA_REGISTRY = {
    "ohlcv": OHLCV_SCHEMA,
    "price_volatility": PRICE_VOLATILITY_SCHEMA,
    "momentum_indicators": MOMENTUM_INDICATORS_SCHEMA,
    "volume_profile": VOLUME_PROFILE_SCHEMA,
}


def get_schema(schema_name: str) -> DataSchema:
    """
    Retrieve a schema from the registry.
    
    Args:
        schema_name: Name of the schema (e.g., 'ohlcv', 'price_volatility')
    
    Returns:
        DataSchema object
    
    Raises:
        KeyError: If schema_name is not found in registry
    """
    if schema_name not in SCHEMA_REGISTRY:
        available = ", ".join(SCHEMA_REGISTRY.keys())
        raise KeyError(f"Schema '{schema_name}' not found. Available: {available}")
    return SCHEMA_REGISTRY[schema_name]


def register_schema(schema: DataSchema) -> None:
    """
    Register a new schema in the registry.
    
    Args:
        schema: DataSchema object to register
    """
    SCHEMA_REGISTRY[schema.name] = schema
