"""
Comprehensive tests for credential management system.

Tests cover:
- Credential schema definitions
- Credential validation
- Credential loading from environment and config files
- Secure logging with credential masking
- Connection testing utilities
- Error handling and edge cases
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data_ingestion.credentials import (
    IBCredential,
    PolygonCredential,
    DatabentoCredential,
    IBCredentialValidator,
    PolygonCredentialValidator,
    DatabentoCredentialValidator,
    CredentialLoader,
    CredentialTester,
    CredentialManager,
    SensitiveDataMasker,
)


# ============================================================================
# Tests for Credential Schemas
# ============================================================================

class TestCredentialSchemas:
    """Test credential schema dataclasses."""
    
    def test_ib_credential_schema(self):
        """Test Interactive Brokers credential schema."""
        cred = IBCredential(
            host="127.0.0.1",
            port=7497,
            client_id=1,
            account="DU123456",
            password="secret"
        )
        assert cred.host == "127.0.0.1"
        assert cred.port == 7497
        assert cred.client_id == 1
        assert cred.account == "DU123456"
        assert cred.password == "secret"
    
    def test_ib_credential_required_fields(self):
        """Test IB required fields."""
        cred = IBCredential()
        assert 'host' in cred.required_fields()
        assert 'port' in cred.required_fields()
        assert 'client_id' in cred.required_fields()
    
    def test_ib_credential_sensitive_fields(self):
        """Test IB sensitive fields."""
        cred = IBCredential()
        assert 'password' in cred.sensitive_fields()
        assert 'account' in cred.sensitive_fields()
    
    def test_polygon_credential_schema(self):
        """Test Polygon credential schema."""
        cred = PolygonCredential(api_key="pk_test_key_123")
        assert cred.api_key == "pk_test_key_123"
    
    def test_polygon_credential_required_fields(self):
        """Test Polygon required fields."""
        cred = PolygonCredential()
        assert 'api_key' in cred.required_fields()
    
    def test_polygon_credential_sensitive_fields(self):
        """Test Polygon sensitive fields."""
        cred = PolygonCredential()
        assert 'api_key' in cred.sensitive_fields()
    
    def test_databento_credential_schema(self):
        """Test Databento credential schema."""
        cred = DatabentoCredential(api_key="db_test_key_123")
        assert cred.api_key == "db_test_key_123"
    
    def test_databento_credential_required_fields(self):
        """Test Databento required fields."""
        cred = DatabentoCredential()
        assert 'api_key' in cred.required_fields()


# ============================================================================
# Tests for Credential Validators
# ============================================================================

class TestIBCredentialValidator:
    """Test Interactive Brokers credential validation."""
    
    def test_valid_credentials(self):
        """Test validation of valid IB credentials."""
        validator = IBCredentialValidator()
        credentials = {
            'host': '127.0.0.1',
            'port': 7497,
            'client_id': 1
        }
        # Should not raise
        validator.validate(credentials)
    
    def test_missing_host(self):
        """Test validation fails with missing host."""
        validator = IBCredentialValidator()
        credentials = {'port': 7497, 'client_id': 1}
        with pytest.raises(ValueError, match="Missing required fields"):
            validator.validate(credentials)
    
    def test_missing_port(self):
        """Test validation fails with missing port."""
        validator = IBCredentialValidator()
        credentials = {'host': '127.0.0.1', 'client_id': 1}
        with pytest.raises(ValueError, match="Missing required fields"):
            validator.validate(credentials)
    
    def test_missing_client_id(self):
        """Test validation fails with missing client_id."""
        validator = IBCredentialValidator()
        credentials = {'host': '127.0.0.1', 'port': 7497}
        with pytest.raises(ValueError, match="Missing required fields"):
            validator.validate(credentials)
    
    def test_invalid_host(self):
        """Test validation fails with invalid host."""
        validator = IBCredentialValidator()
        credentials = {'host': '', 'port': 7497, 'client_id': 1}
        with pytest.raises(ValueError, match="host must be a non-empty string"):
            validator.validate(credentials)
    
    def test_invalid_port(self):
        """Test validation fails with invalid port."""
        validator = IBCredentialValidator()
        credentials = {'host': '127.0.0.1', 'port': 99999, 'client_id': 1}
        with pytest.raises(ValueError, match="port must be"):
            validator.validate(credentials)
    
    def test_invalid_port_not_integer(self):
        """Test validation fails when port is not integer."""
        validator = IBCredentialValidator()
        credentials = {'host': '127.0.0.1', 'port': '7497', 'client_id': 1}
        with pytest.raises(ValueError, match="port must be"):
            validator.validate(credentials)
    
    def test_negative_client_id(self):
        """Test validation fails with negative client_id."""
        validator = IBCredentialValidator()
        credentials = {'host': '127.0.0.1', 'port': 7497, 'client_id': -1}
        with pytest.raises(ValueError, match="client_id must be"):
            validator.validate(credentials)
    
    def test_get_required_fields(self):
        """Test getting required fields."""
        validator = IBCredentialValidator()
        assert validator.get_required_fields() == ['host', 'port', 'client_id']
    
    def test_get_sensitive_fields(self):
        """Test getting sensitive fields."""
        validator = IBCredentialValidator()
        assert 'password' in validator.get_sensitive_fields()
        assert 'account' in validator.get_sensitive_fields()


class TestPolygonCredentialValidator:
    """Test Polygon credential validation."""
    
    def test_valid_credentials(self):
        """Test validation of valid Polygon credentials."""
        validator = PolygonCredentialValidator()
        credentials = {'api_key': 'pk_test_key_123'}
        # Should not raise
        validator.validate(credentials)
    
    def test_missing_api_key(self):
        """Test validation fails with missing API key."""
        validator = PolygonCredentialValidator()
        credentials = {}
        with pytest.raises(ValueError, match="Missing required field: api_key"):
            validator.validate(credentials)
    
    def test_empty_api_key(self):
        """Test validation fails with empty API key."""
        validator = PolygonCredentialValidator()
        credentials = {'api_key': ''}
        with pytest.raises(ValueError, match="api_key must be a non-empty string"):
            validator.validate(credentials)
    
    def test_non_string_api_key(self):
        """Test validation fails when API key is not string."""
        validator = PolygonCredentialValidator()
        credentials = {'api_key': 12345}
        with pytest.raises(ValueError, match="api_key must be a non-empty string"):
            validator.validate(credentials)
    
    def test_get_required_fields(self):
        """Test getting required fields."""
        validator = PolygonCredentialValidator()
        assert validator.get_required_fields() == ['api_key']
    
    def test_get_sensitive_fields(self):
        """Test getting sensitive fields."""
        validator = PolygonCredentialValidator()
        assert validator.get_sensitive_fields() == ['api_key']


class TestDatabentoCredentialValidator:
    """Test Databento credential validation."""
    
    def test_valid_credentials(self):
        """Test validation of valid Databento credentials."""
        validator = DatabentoCredentialValidator()
        credentials = {'api_key': 'db_test_key_123'}
        # Should not raise
        validator.validate(credentials)
    
    def test_missing_api_key(self):
        """Test validation fails with missing API key."""
        validator = DatabentoCredentialValidator()
        credentials = {}
        with pytest.raises(ValueError, match="Missing required field: api_key"):
            validator.validate(credentials)
    
    def test_empty_api_key(self):
        """Test validation fails with empty API key."""
        validator = DatabentoCredentialValidator()
        credentials = {'api_key': ''}
        with pytest.raises(ValueError, match="api_key must be a non-empty string"):
            validator.validate(credentials)


# ============================================================================
# Tests for Secure Logging / Data Masking
# ============================================================================

class TestSensitiveDataMasker:
    """Test sensitive data masking utilities."""
    
    def test_mask_string_short(self):
        """Test masking short strings."""
        masked = SensitiveDataMasker.mask_string("abc")
        assert masked == "***"
        assert len(masked) == len("abc")
    
    def test_mask_string_long(self):
        """Test masking long strings."""
        masked = SensitiveDataMasker.mask_string("pk_abcdefghijklmnopqrst")
        # Should show first 3 and last 1
        assert masked.startswith("pk_")
        assert masked.endswith("t")
        assert "****" in masked
    
    def test_mask_dict(self):
        """Test masking dictionary values."""
        data = {
            'api_key': 'pk_secret_key_123',
            'host': '127.0.0.1',
            'port': 7497
        }
        masked = SensitiveDataMasker.mask_dict(data, ['api_key'])
        
        # api_key should be masked
        assert '****' in masked['api_key']
        # Other fields should be unchanged
        assert masked['host'] == '127.0.0.1'
        assert masked['port'] == 7497
    
    def test_mask_dict_with_non_string_sensitive(self):
        """Test masking non-string sensitive values."""
        data = {
            'password': 12345,
            'username': 'user'
        }
        masked = SensitiveDataMasker.mask_dict(data, ['password'])
        
        assert masked['password'] == '***'
        assert masked['username'] == 'user'
    
    def test_mask_error_message(self):
        """Test masking error messages."""
        message = "Connection failed with api_key='pk_secret123' at host"
        masked = SensitiveDataMasker.mask_error_message(message)
        
        # Should mask the API key
        assert 'pk_secret123' not in masked
        assert '***' in masked


# ============================================================================
# Tests for Credential Loader
# ============================================================================

class TestCredentialLoader:
    """Test credential loading from multiple sources."""
    
    def test_load_from_environment(self):
        """Test loading credentials from environment variables."""
        with patch.dict(os.environ, {
            'POLYGON_API_KEY': 'pk_test_key_123'
        }):
            loader = CredentialLoader()
            creds = loader._load_from_env('polygon')
            
            assert creds is not None
            assert creds['api_key'] == 'pk_test_key_123'
    
    def test_load_from_environment_ib(self):
        """Test loading IB credentials from environment."""
        with patch.dict(os.environ, {
            'IB_HOST': '192.168.1.1',
            'IB_PORT': '7496',
            'IB_CLIENT_ID': '2'
        }):
            loader = CredentialLoader()
            creds = loader._load_from_env('ib')
            
            assert creds is not None
            assert creds['host'] == '192.168.1.1'
            assert creds['port'] == 7496
            assert creds['client_id'] == 2
    
    def test_load_from_environment_not_found(self):
        """Test loading returns None when credentials not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            loader = CredentialLoader()
            creds = loader._load_from_env('polygon')
            
            assert creds is None
    
    def test_load_from_json_config(self):
        """Test loading credentials from JSON config file."""
        config = {
            'polygon': {
                'api_key': 'pk_from_json_123'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_file = f.name
        
        try:
            loader = CredentialLoader()
            creds = loader._load_from_config('polygon', config_file)
            
            assert creds is not None
            assert creds['api_key'] == 'pk_from_json_123'
        finally:
            os.unlink(config_file)
    
    def test_load_credentials_priority_env_over_config(self):
        """Test that environment variables take priority over config files."""
        config = {
            'polygon': {
                'api_key': 'pk_from_config_123'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_file = f.name
        
        try:
            with patch.dict(os.environ, {
                'POLYGON_API_KEY': 'pk_from_env_123'
            }):
                loader = CredentialLoader()
                creds = loader.load_credentials('polygon', config_file)
                
                # Environment variable should win
                assert creds['api_key'] == 'pk_from_env_123'
        finally:
            os.unlink(config_file)
    
    def test_load_credentials_from_config_when_env_missing(self):
        """Test loading from config when env variables not found."""
        config = {
            'polygon': {
                'api_key': 'pk_from_config_456'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_file = f.name
        
        try:
            with patch.dict(os.environ, {}, clear=True):
                loader = CredentialLoader()
                creds = loader.load_credentials('polygon', config_file)
                
                assert creds['api_key'] == 'pk_from_config_456'
        finally:
            os.unlink(config_file)
    
    def test_load_credentials_unknown_vendor(self):
        """Test error when vendor is unknown."""
        loader = CredentialLoader()
        
        with pytest.raises(ValueError, match="Unknown vendor"):
            loader.load_credentials('unknown_vendor')
    
    def test_load_credentials_not_found(self):
        """Test error when no credentials found."""
        with patch.dict(os.environ, {}, clear=True):
            loader = CredentialLoader()
            
            with pytest.raises(EnvironmentError, match="No credentials found"):
                loader.load_credentials('polygon')
    
    def test_load_credentials_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        loader = CredentialLoader()
        
        with pytest.raises(FileNotFoundError):
            loader._load_from_config('polygon', '/nonexistent/file.json')


# ============================================================================
# Tests for Credential Tester
# ============================================================================

class TestCredentialTester:
    """Test connection testing utilities."""
    
    def test_test_connection_polygon(self):
        """Test Polygon connection testing."""
        credentials = {'api_key': 'pk_test_key_123'}
        
        with patch('src.data_ingestion.credentials.PolygonDataProvider') as mock_provider:
            mock_instance = MagicMock()
            mock_provider.return_value = mock_instance
            
            tester = CredentialTester()
            result = tester.test_connection('polygon', credentials)
            
            assert result is True
            mock_provider.assert_called_once_with(api_key='pk_test_key_123')
            mock_instance.authenticate.assert_called_once()
    
    def test_test_connection_unknown_vendor(self):
        """Test error with unknown vendor."""
        tester = CredentialTester()
        
        with pytest.raises(ValueError, match="Unknown vendor"):
            tester.test_connection('unknown', {})
    
    def test_test_connection_invalid_credentials(self):
        """Test error with invalid credentials."""
        credentials = {'api_key': ''}
        
        tester = CredentialTester()
        with pytest.raises(ValueError, match="Credential validation failed"):
            tester.test_connection('polygon', credentials)


# ============================================================================
# Tests for Credential Manager
# ============================================================================

class TestCredentialManager:
    """Test high-level credential management interface."""
    
    def test_get_vendor_credentials(self):
        """Test getting vendor credentials."""
        with patch.dict(os.environ, {
            'POLYGON_API_KEY': 'pk_test_key_123'
        }):
            manager = CredentialManager()
            creds = manager.get_vendor_credentials('polygon')
            
            assert creds['api_key'] == 'pk_test_key_123'
    
    def test_get_vendor_credentials_with_validation(self):
        """Test credentials are validated."""
        with patch.dict(os.environ, {
            'POLYGON_API_KEY': ''
        }):
            manager = CredentialManager()
            
            with pytest.raises(ValueError, match="api_key must be a non-empty string"):
                manager.get_vendor_credentials('polygon')
    
    def test_get_vendor_credentials_with_connection_test(self):
        """Test credentials are tested for connection."""
        with patch.dict(os.environ, {
            'POLYGON_API_KEY': 'pk_test_key_123'
        }):
            with patch('src.data_ingestion.credentials.PolygonDataProvider'):
                manager = CredentialManager()
                creds = manager.get_vendor_credentials(
                    'polygon',
                    test_connection=True
                )
                
                assert creds['api_key'] == 'pk_test_key_123'
    
    def test_get_all_vendor_credentials(self):
        """Test loading all vendor credentials."""
        with patch.dict(os.environ, {
            'POLYGON_API_KEY': 'pk_test_123',
            'DATABENTO_API_KEY': 'db_test_123'
        }):
            manager = CredentialManager()
            all_creds = manager.get_all_vendor_credentials()
            
            # Should load polygon and databento, but not IB (optional)
            assert 'polygon' in all_creds
            assert 'databento' in all_creds
            # IB is optional, may or may not be present
    
    def test_get_all_vendor_credentials_graceful_failure(self):
        """Test graceful handling of missing credentials."""
        with patch.dict(os.environ, {}, clear=True):
            manager = CredentialManager()
            all_creds = manager.get_all_vendor_credentials()
            
            # Should return empty dict when no credentials available
            assert isinstance(all_creds, dict)
            # All vendors should fail gracefully
            assert len(all_creds) == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestCredentialIntegration:
    """Integration tests for credential management workflow."""
    
    def test_full_workflow_env_to_provider(self):
        """Test full workflow from env variables to provider usage."""
        with patch.dict(os.environ, {
            'POLYGON_API_KEY': 'pk_integration_test_123'
        }):
            # Load credentials
            manager = CredentialManager()
            creds = manager.get_vendor_credentials('polygon')
            
            # Use credentials to create provider
            from src.data_ingestion import PolygonDataProvider
            provider = PolygonDataProvider(**creds)
            
            assert provider.api_key == 'pk_integration_test_123'
    
    def test_full_workflow_config_file(self):
        """Test full workflow using config file."""
        config = {
            'polygon': {'api_key': 'pk_config_test_123'}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_file = f.name
        
        try:
            with patch.dict(os.environ, {}, clear=True):
                manager = CredentialManager()
                creds = manager.get_vendor_credentials('polygon', config_file)
                
                assert creds['api_key'] == 'pk_config_test_123'
        finally:
            os.unlink(config_file)
    
    def test_credential_masking_in_logging(self):
        """Test that sensitive credentials are masked in logs."""
        credentials = {
            'api_key': 'pk_secret_password_123',
            'host': 'api.example.com'
        }
        
        sensitive_fields = ['api_key']
        masked = SensitiveDataMasker.mask_dict(credentials, sensitive_fields)
        
        # Verify masking works
        assert 'secret_password' not in str(masked)
        assert 'api.example.com' in str(masked)
