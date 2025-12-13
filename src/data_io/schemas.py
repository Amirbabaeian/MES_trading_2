"""
Schema definitions for data validation in the hybrid versioning system.

This module provides predefined schemas for common data types (OHLCV, features)
and utilities for schema management and validation.
"""

from typing import Dict, Any, Optional, List
import pyarrow as pa


# OHLCV Schema (Open, High, Low, Close, Volume)
# Standard schema for raw and cleaned market data
OHLCV_SCHEMA = pa.schema([
    pa.field("timestamp", pa.timestamp("us")),  # Microsecond precision
    pa.field("asset", pa.string()),              # Asset identifier (e.g., "MES", "ES", "VIX")
    pa.field("open", pa.float64()),
    pa.field("high", pa.float64()),
    pa.field("low", pa.float64()),
    pa.field("close", pa.float64()),
    pa.field("volume", pa.uint64()),
])

# Features Schema
# For engineered feature datasets with flexible structure
FEATURES_SCHEMA = pa.schema([
    pa.field("timestamp", pa.timestamp("us")),
    pa.field("asset", pa.string()),
    pa.field("feature_set", pa.string()),        # Identifier for feature set version
    pa.field("features", pa.struct([             # Flexible nested structure
        pa.field("sma_20", pa.float64(), nullable=True),
        pa.field("sma_50", pa.float64(), nullable=True),
        pa.field("ema_12", pa.float64(), nullable=True),
        pa.field("ema_26", pa.float64(), nullable=True),
        pa.field("rsi_14", pa.float64(), nullable=True),
        pa.field("macd", pa.float64(), nullable=True),
        pa.field("macd_signal", pa.float64(), nullable=True),
        pa.field("bollinger_upper", pa.float64(), nullable=True),
        pa.field("bollinger_lower", pa.float64(), nullable=True),
        pa.field("atr_14", pa.float64(), nullable=True),
    ])),
])


class SchemaRegistry:
    """
    Registry for managing and retrieving data schemas.
    
    Provides centralized management of predefined schemas and support for
    custom schema definitions.
    """
    
    def __init__(self):
        """Initialize schema registry with default schemas."""
        self._schemas: Dict[str, pa.Schema] = {
            "ohlcv": OHLCV_SCHEMA,
            "features": FEATURES_SCHEMA,
        }
    
    def register(self, name: str, schema: pa.Schema) -> None:
        """
        Register a new schema.
        
        Args:
            name: Unique identifier for the schema
            schema: PyArrow schema object
            
        Raises:
            ValueError: If schema name already exists
        """
        if name in self._schemas:
            raise ValueError(f"Schema '{name}' already registered")
        self._schemas[name] = schema
    
    def get(self, name: str) -> pa.Schema:
        """
        Retrieve a registered schema by name.
        
        Args:
            name: Schema identifier
            
        Returns:
            PyArrow schema object
            
        Raises:
            KeyError: If schema name not found
        """
        if name not in self._schemas:
            raise KeyError(f"Schema '{name}' not found in registry")
        return self._schemas[name]
    
    def list_schemas(self) -> List[str]:
        """Get list of all registered schema names."""
        return list(self._schemas.keys())
    
    def schema_to_dict(self, schema: pa.Schema) -> Dict[str, str]:
        """
        Convert PyArrow schema to dictionary representation.
        
        Args:
            schema: PyArrow schema object
            
        Returns:
            Dictionary mapping column names to data types
        """
        return {field.name: str(field.type) for field in schema}
    
    def validate_data_against_schema(
        self,
        data_schema: pa.Schema,
        target_schema: pa.Schema,
        strict: bool = False
    ) -> tuple[bool, Optional[str]]:
        """
        Validate if data schema matches target schema.
        
        Args:
            data_schema: Schema of the data being validated
            target_schema: Expected schema
            strict: If True, exact match required; if False, allow additional columns
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        target_names = {field.name for field in target_schema}
        data_names = {field.name for field in data_schema}
        
        # Check for missing required columns
        missing = target_names - data_names
        if missing:
            return False, f"Missing required columns: {sorted(missing)}"
        
        # Check for extra columns in strict mode
        if strict:
            extra = data_names - target_names
            if extra:
                return False, f"Extra columns not in schema: {sorted(extra)}"
        
        # Check column types
        target_types = {field.name: field.type for field in target_schema}
        data_types = {field.name: field.type for field in data_schema}
        
        type_mismatches = []
        for col_name in target_names:
            if col_name in data_types:
                if data_types[col_name] != target_types[col_name]:
                    type_mismatches.append(
                        f"{col_name}: expected {target_types[col_name]}, "
                        f"got {data_types[col_name]}"
                    )
        
        if type_mismatches:
            return False, "Type mismatches:\n  " + "\n  ".join(type_mismatches)
        
        return True, None


# Global schema registry instance
_default_registry = SchemaRegistry()


def get_schema(name: str) -> pa.Schema:
    """Get a schema from the default registry."""
    return _default_registry.get(name)


def register_schema(name: str, schema: pa.Schema) -> None:
    """Register a schema in the default registry."""
    _default_registry.register(name, schema)


def list_schemas() -> List[str]:
    """List all registered schemas."""
    return _default_registry.list_schemas()
