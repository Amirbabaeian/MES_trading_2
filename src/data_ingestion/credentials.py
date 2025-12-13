"""
Secure credential management system for data vendor API authentication.

This module provides:
- Credential schema definitions for each supported vendor
- Loading credentials from environment variables and configuration files
- Credential validation against vendor-specific schemas
- Secure logging with credential masking
- Connection testing utilities to verify credentials

Example:
    >>> loader = CredentialLoader()
    >>> creds = loader.load_credentials('polygon')
    >>> validator = PolygonCredentialValidator()
    >>> validator.validate(creds)
    >>> tester = CredentialTester()
    >>> tester.test_connection('polygon', creds)
"""

import os
import json
import re
import logging
from typing import Dict, Any, Optional, List, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path=None):
        """Fallback if python-dotenv is not installed."""
        pass

logger = logging.getLogger(__name__)


# ============================================================================
# Credential Schema Classes
# ============================================================================

@dataclass
class IBCredential:
    """Interactive Brokers credential schema."""
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    account: Optional[str] = None
    password: Optional[str] = None
    
    def required_fields(self) -> List[str]:
        """Return list of required fields for IB authentication."""
        return ['host', 'port', 'client_id']
    
    def sensitive_fields(self) -> List[str]:
        """Return list of sensitive fields that should be masked in logs."""
        return ['password', 'account']


@dataclass
class PolygonCredential:
    """Polygon.io credential schema."""
    api_key: Optional[str] = None
    
    def required_fields(self) -> List[str]:
        """Return list of required fields for Polygon authentication."""
        return ['api_key']
    
    def sensitive_fields(self) -> List[str]:
        """Return list of sensitive fields that should be masked in logs."""
        return ['api_key']


@dataclass
class DatabentoCredential:
    """Databento credential schema."""
    api_key: Optional[str] = None
    
    def required_fields(self) -> List[str]:
        """Return list of required fields for Databento authentication."""
        return ['api_key']
    
    def sensitive_fields(self) -> List[str]:
        """Return list of sensitive fields that should be masked in logs."""
        return ['api_key']


# Mapping of vendor names to credential classes
CREDENTIAL_SCHEMAS = {
    'ib': IBCredential,
    'interactive_brokers': IBCredential,
    'polygon': PolygonCredential,
    'databento': DatabentoCredential,
}


# ============================================================================
# Credential Validation
# ============================================================================

class CredentialValidator(ABC):
    """Base class for credential validation."""
    
    @abstractmethod
    def validate(self, credentials: Dict[str, Any]) -> None:
        """
        Validate credentials against schema.
        
        Args:
            credentials: Dictionary of credential key-value pairs
            
        Raises:
            ValueError: If validation fails
        """
        pass
    
    @abstractmethod
    def get_required_fields(self) -> List[str]:
        """Get list of required fields for this credential type."""
        pass
    
    @abstractmethod
    def get_sensitive_fields(self) -> List[str]:
        """Get list of sensitive fields that should be masked."""
        pass


class IBCredentialValidator(CredentialValidator):
    """Validator for Interactive Brokers credentials."""
    
    REQUIRED_FIELDS = ['host', 'port', 'client_id']
    SENSITIVE_FIELDS = ['password', 'account']
    
    def validate(self, credentials: Dict[str, Any]) -> None:
        """
        Validate Interactive Brokers credentials.
        
        Args:
            credentials: Dictionary with keys: host, port, client_id, account, password
            
        Raises:
            ValueError: If any required field is missing or invalid
        """
        missing = [f for f in self.REQUIRED_FIELDS if f not in credentials]
        if missing:
            raise ValueError(
                f"Interactive Brokers: Missing required fields: {', '.join(missing)}"
            )
        
        # Validate types
        try:
            host = credentials.get('host')
            if not isinstance(host, str) or not host.strip():
                raise ValueError("host must be a non-empty string")
            
            port = credentials.get('port')
            if not isinstance(port, int) or port < 1 or port > 65535:
                raise ValueError("port must be an integer between 1 and 65535")
            
            client_id = credentials.get('client_id')
            if not isinstance(client_id, int) or client_id < 0:
                raise ValueError("client_id must be a non-negative integer")
        except ValueError as e:
            raise ValueError(f"Interactive Brokers credential validation failed: {str(e)}")
    
    def get_required_fields(self) -> List[str]:
        """Get required fields."""
        return self.REQUIRED_FIELDS
    
    def get_sensitive_fields(self) -> List[str]:
        """Get sensitive fields."""
        return self.SENSITIVE_FIELDS


class PolygonCredentialValidator(CredentialValidator):
    """Validator for Polygon.io credentials."""
    
    REQUIRED_FIELDS = ['api_key']
    SENSITIVE_FIELDS = ['api_key']
    
    def validate(self, credentials: Dict[str, Any]) -> None:
        """
        Validate Polygon.io credentials.
        
        Args:
            credentials: Dictionary with key: api_key
            
        Raises:
            ValueError: If required field is missing or invalid
        """
        if 'api_key' not in credentials:
            raise ValueError("Polygon.io: Missing required field: api_key")
        
        api_key = credentials.get('api_key')
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Polygon.io: api_key must be a non-empty string")
    
    def get_required_fields(self) -> List[str]:
        """Get required fields."""
        return self.REQUIRED_FIELDS
    
    def get_sensitive_fields(self) -> List[str]:
        """Get sensitive fields."""
        return self.SENSITIVE_FIELDS


class DatabentoCredentialValidator(CredentialValidator):
    """Validator for Databento credentials."""
    
    REQUIRED_FIELDS = ['api_key']
    SENSITIVE_FIELDS = ['api_key']
    
    def validate(self, credentials: Dict[str, Any]) -> None:
        """
        Validate Databento credentials.
        
        Args:
            credentials: Dictionary with key: api_key
            
        Raises:
            ValueError: If required field is missing or invalid
        """
        if 'api_key' not in credentials:
            raise ValueError("Databento: Missing required field: api_key")
        
        api_key = credentials.get('api_key')
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Databento: api_key must be a non-empty string")
    
    def get_required_fields(self) -> List[str]:
        """Get required fields."""
        return self.REQUIRED_FIELDS
    
    def get_sensitive_fields(self) -> List[str]:
        """Get sensitive fields."""
        return self.SENSITIVE_FIELDS


# Mapping of vendor names to validator classes
CREDENTIAL_VALIDATORS = {
    'ib': IBCredentialValidator,
    'interactive_brokers': IBCredentialValidator,
    'polygon': PolygonCredentialValidator,
    'databento': DatabentoCredentialValidator,
}


# ============================================================================
# Secure Logging Utilities
# ============================================================================

class SensitiveDataMasker:
    """Utility for masking sensitive credentials in logs and error messages."""
    
    # Patterns to match and mask
    PATTERNS = [
        # API keys (common formats)
        (r'api[_-]?key[\'\":\s]*([a-zA-Z0-9_\-]{20,})', 'api_key'),
        (r'[\'\"]\w{20,}[\'\"]\s*(?:,|$)', 'key'),
        # Passwords
        (r'password[\'\":\s]*([^,\]}\'"]+)', 'password'),
        # Tokens
        (r'token[\'\":\s]*([a-zA-Z0-9_\-]{20,})', 'token'),
        # Basic auth credentials (username:password)
        (r'://([^:]+):([^@]+)@', 'basic_auth'),
    ]
    
    @staticmethod
    def mask_string(value: str, field_name: str = None) -> str:
        """
        Mask a sensitive string value.
        
        Args:
            value: The value to mask
            field_name: Optional field name for better masking
            
        Returns:
            Masked value
        """
        if not isinstance(value, str):
            return str(value)
        
        if len(value) <= 4:
            return '*' * len(value)
        
        # Show first 3 and last 1 characters
        return value[:3] + '*' * (len(value) - 4) + value[-1]
    
    @staticmethod
    def mask_dict(data: Dict[str, Any], sensitive_fields: List[str]) -> Dict[str, Any]:
        """
        Mask sensitive fields in a dictionary.
        
        Args:
            data: Dictionary to mask
            sensitive_fields: List of field names to mask
            
        Returns:
            Dictionary with sensitive fields masked
        """
        masked = {}
        for key, value in data.items():
            if key in sensitive_fields:
                if isinstance(value, str):
                    masked[key] = SensitiveDataMasker.mask_string(value, key)
                else:
                    masked[key] = '***'
            else:
                masked[key] = value
        return masked
    
    @staticmethod
    def mask_error_message(message: str) -> str:
        """
        Mask sensitive data in error messages.
        
        Args:
            message: Error message that may contain sensitive data
            
        Returns:
            Error message with sensitive data masked
        """
        result = message
        for pattern, field_name in SensitiveDataMasker.PATTERNS:
            result = re.sub(
                pattern,
                lambda m: f"{field_name}=***",
                result,
                flags=re.IGNORECASE
            )
        return result


# ============================================================================
# Credential Loading
# ============================================================================

class CredentialLoader:
    """
    Load credentials from multiple sources with priority ordering.
    
    Priority order:
    1. Environment variables
    2. Configuration files (JSON/YAML)
    3. Default values
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize credential loader.
        
        Args:
            env_file: Path to .env file to load (defaults to .env)
        """
        self.env_file = env_file or '.env'
        self._load_env_file()
    
    def _load_env_file(self) -> None:
        """Load environment variables from .env file if it exists."""
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
            logger.debug(f"Loaded environment variables from {self.env_file}")
    
    def load_credentials(
        self,
        vendor: str,
        config_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load credentials for a vendor from multiple sources.
        
        Priority:
        1. Environment variables
        2. Configuration file (if provided)
        3. Raises error if nothing found
        
        Args:
            vendor: Vendor name ('ib', 'polygon', 'databento')
            config_file: Optional path to JSON/YAML config file
            
        Returns:
            Dictionary of credentials
            
        Raises:
            ValueError: If vendor is not recognized
            FileNotFoundError: If config file is provided but not found
            EnvironmentError: If no credentials found in any source
        """
        vendor = vendor.lower().strip()
        
        # Validate vendor
        if vendor not in CREDENTIAL_SCHEMAS:
            raise ValueError(
                f"Unknown vendor: {vendor}. "
                f"Supported vendors: {', '.join(CREDENTIAL_SCHEMAS.keys())}"
            )
        
        # Try to load from environment variables first
        creds = self._load_from_env(vendor)
        if creds:
            logger.debug(f"Loaded {vendor} credentials from environment variables")
            return creds
        
        # Try to load from config file
        if config_file:
            creds = self._load_from_config(vendor, config_file)
            if creds:
                logger.debug(f"Loaded {vendor} credentials from {config_file}")
                return creds
        
        # No credentials found
        raise EnvironmentError(
            f"No credentials found for {vendor}. "
            f"Please set environment variables or provide a config file. "
            f"See docs/data_vendor_setup.md for setup instructions."
        )
    
    def _load_from_env(self, vendor: str) -> Optional[Dict[str, Any]]:
        """
        Load credentials from environment variables.
        
        Environment variable names follow pattern: {VENDOR}_{FIELD}
        Examples:
        - IB_HOST, IB_PORT, IB_CLIENT_ID, IB_ACCOUNT, IB_PASSWORD
        - POLYGON_API_KEY
        - DATABENTO_API_KEY
        
        Args:
            vendor: Vendor name
            
        Returns:
            Dictionary of credentials or None if not found
        """
        vendor_upper = vendor.upper()
        
        # Map vendor names to env var prefixes
        env_prefixes = {
            'ib': 'IB',
            'interactive_brokers': 'IB',
            'polygon': 'POLYGON',
            'databento': 'DATABENTO',
        }
        
        prefix = env_prefixes.get(vendor, vendor_upper)
        schema = CREDENTIAL_SCHEMAS[vendor]
        
        creds = {}
        schema_instance = schema()
        
        # Get all fields from schema
        schema_fields = schema_instance.__dataclass_fields__.keys()
        
        for field in schema_fields:
            env_var = f"{prefix}_{field.upper()}"
            value = os.getenv(env_var)
            if value is not None:
                # Try to convert to the appropriate type
                if hasattr(schema_instance, field):
                    field_type = schema_instance.__dataclass_fields__[field].type
                    # Handle Optional types
                    if hasattr(field_type, '__origin__'):
                        # This is a generic type like Optional[int]
                        if hasattr(field_type, '__args__'):
                            actual_type = field_type.__args__[0]
                            if actual_type == int:
                                try:
                                    creds[field] = int(value)
                                except ValueError:
                                    creds[field] = value
                            else:
                                creds[field] = value
                    elif field_type == int:
                        try:
                            creds[field] = int(value)
                        except ValueError:
                            creds[field] = value
                    else:
                        creds[field] = value
                else:
                    creds[field] = value
        
        return creds if creds else None
    
    def _load_from_config(self, vendor: str, config_file: str) -> Optional[Dict[str, Any]]:
        """
        Load credentials from JSON/YAML configuration file.
        
        Config file format:
        {
            "ib": {"host": "127.0.0.1", "port": 7497, ...},
            "polygon": {"api_key": "pk_..."},
            "databento": {"api_key": "db_..."}
        }
        
        Args:
            vendor: Vendor name
            config_file: Path to config file
            
        Returns:
            Dictionary of credentials or None if not found
            
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        try:
            with open(config_path) as f:
                if config_file.endswith('.json'):
                    config_data = json.load(f)
                else:
                    # Try YAML if installed
                    try:
                        import yaml
                        config_data = yaml.safe_load(f)
                    except ImportError:
                        logger.warning(
                            f"YAML parsing not available. "
                            f"Install PyYAML or use JSON config files."
                        )
                        return None
            
            # Look for vendor credentials in config
            vendor_lower = vendor.lower()
            if vendor_lower in config_data:
                return config_data[vendor_lower]
            
            # Also try without vendor key (single vendor config)
            if 'credentials' in config_data:
                return config_data['credentials']
            
            return None
        
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading config file {config_file}: {str(e)}")
            return None


# ============================================================================
# Connection Testing
# ============================================================================

class CredentialTester:
    """Test credentials by attempting connection to each vendor."""
    
    def test_connection(
        self,
        vendor: str,
        credentials: Dict[str, Any]
    ) -> bool:
        """
        Test credentials by attempting connection.
        
        Args:
            vendor: Vendor name ('ib', 'polygon', 'databento')
            credentials: Dictionary of credentials
            
        Returns:
            True if connection test succeeds
            
        Raises:
            ValueError: If vendor is not recognized
            ConnectionError: If connection test fails
        """
        vendor = vendor.lower().strip()
        
        if vendor not in ['ib', 'interactive_brokers', 'polygon', 'databento']:
            raise ValueError(f"Unknown vendor: {vendor}")
        
        # Validate credentials first
        validator_class = CREDENTIAL_VALIDATORS.get(vendor)
        if validator_class:
            validator = validator_class()
            try:
                validator.validate(credentials)
            except ValueError as e:
                raise ValueError(f"Credential validation failed: {str(e)}")
        
        # Test connection based on vendor
        if vendor in ['ib', 'interactive_brokers']:
            return self._test_ib_connection(credentials)
        elif vendor == 'polygon':
            return self._test_polygon_connection(credentials)
        elif vendor == 'databento':
            return self._test_databento_connection(credentials)
        
        return False
    
    def _test_ib_connection(self, credentials: Dict[str, Any]) -> bool:
        """
        Test Interactive Brokers connection.
        
        Args:
            credentials: IB credentials (host, port, client_id, etc.)
            
        Returns:
            True if connection succeeds
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            # Import the adapter here to avoid circular imports
            from .ib_provider import IBDataProvider
            
            provider = IBDataProvider(
                host=credentials.get('host', '127.0.0.1'),
                port=credentials.get('port', 7497),
                client_id=credentials.get('client_id', 1),
                api_key=credentials.get('account')
            )
            
            # Try to authenticate
            provider.authenticate()
            logger.info("Interactive Brokers connection test successful")
            return True
        
        except Exception as e:
            error_msg = f"Interactive Brokers connection test failed: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def _test_polygon_connection(self, credentials: Dict[str, Any]) -> bool:
        """
        Test Polygon.io connection.
        
        Args:
            credentials: Polygon credentials (api_key)
            
        Returns:
            True if connection succeeds
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            # Import the adapter here to avoid circular imports
            from .polygon_provider import PolygonDataProvider
            
            provider = PolygonDataProvider(api_key=credentials.get('api_key'))
            
            # Try to authenticate
            provider.authenticate()
            logger.info("Polygon.io connection test successful")
            return True
        
        except Exception as e:
            error_msg = f"Polygon.io connection test failed: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
    
    def _test_databento_connection(self, credentials: Dict[str, Any]) -> bool:
        """
        Test Databento connection.
        
        Args:
            credentials: Databento credentials (api_key)
            
        Returns:
            True if connection succeeds
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            # Import the adapter here to avoid circular imports
            from .databento_provider import DatabentoDataProvider
            
            provider = DatabentoDataProvider(api_key=credentials.get('api_key'))
            
            # Try to authenticate
            provider.authenticate()
            logger.info("Databento connection test successful")
            return True
        
        except Exception as e:
            error_msg = f"Databento connection test failed: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)


# ============================================================================
# Credential Manager (High-level interface)
# ============================================================================

class CredentialManager:
    """
    High-level credential management interface.
    
    Combines loading, validation, and testing in a single interface.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize credential manager.
        
        Args:
            env_file: Path to .env file (defaults to '.env')
        """
        self.loader = CredentialLoader(env_file)
        self.tester = CredentialTester()
        self.validators = CREDENTIAL_VALIDATORS
    
    def get_vendor_credentials(
        self,
        vendor: str,
        config_file: Optional[str] = None,
        test_connection: bool = False
    ) -> Dict[str, Any]:
        """
        Get and optionally validate credentials for a vendor.
        
        Args:
            vendor: Vendor name ('ib', 'polygon', 'databento')
            config_file: Optional path to config file
            test_connection: If True, test connection before returning
            
        Returns:
            Dictionary of credentials
            
        Raises:
            ValueError: If vendor unknown or credentials invalid
            EnvironmentError: If credentials not found
            ConnectionError: If test_connection=True and test fails
        """
        # Load credentials
        credentials = self.loader.load_credentials(vendor, config_file)
        
        # Validate credentials
        validator_class = self.validators.get(vendor)
        if validator_class:
            validator = validator_class()
            validator.validate(credentials)
        
        # Test connection if requested
        if test_connection:
            self.tester.test_connection(vendor, credentials)
        
        return credentials
    
    def get_all_vendor_credentials(
        self,
        config_file: Optional[str] = None,
        test_connections: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load credentials for all supported vendors.
        
        Fails gracefully if a vendor's credentials are not found.
        
        Args:
            config_file: Optional path to config file with multiple vendors
            test_connections: If True, test all connections
            
        Returns:
            Dictionary mapping vendor names to their credentials
        """
        all_creds = {}
        vendors = ['ib', 'polygon', 'databento']
        
        for vendor in vendors:
            try:
                creds = self.get_vendor_credentials(
                    vendor,
                    config_file,
                    test_connection=test_connections
                )
                all_creds[vendor] = creds
                logger.info(f"Successfully loaded credentials for {vendor}")
            except (ValueError, EnvironmentError) as e:
                logger.warning(f"Could not load credentials for {vendor}: {str(e)}")
            except ConnectionError as e:
                logger.error(f"Connection test failed for {vendor}: {str(e)}")
        
        return all_creds
