"""
Configuration system for cloud storage providers and sync settings.

Provides:
- YAML-based provider configuration
- Credential management
- Sync engine configuration
- Environment variable support
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import yaml

from .cloud_sync import CloudSyncEngine, ConflictStrategy, CloudStorage
from .providers.s3 import S3CloudStorage
from .providers.gcs import GCSCloudStorage
from .providers.azure import AzureCloudStorage

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for a cloud storage provider."""
    provider: str  # 's3', 'gcs', 'azure'
    bucket: str
    region: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = None
    storage_class: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "bucket": self.bucket,
            "region": self.region,
            "credentials": self.credentials,
            "storage_class": self.storage_class,
        }


@dataclass
class SyncConfig:
    """Configuration for sync engine."""
    storage_config: StorageConfig
    sync_state_db: Path
    conflict_strategy: ConflictStrategy = ConflictStrategy.TIMESTAMP_WINS
    max_workers: int = 4
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "storage_config": self.storage_config.to_dict(),
            "sync_state_db": str(self.sync_state_db),
            "conflict_strategy": self.conflict_strategy.value,
            "max_workers": self.max_workers,
            "retry_attempts": self.retry_attempts,
            "retry_delay_seconds": self.retry_delay_seconds,
        }


class CloudStorageFactory:
    """Factory for creating cloud storage providers."""
    
    @staticmethod
    def create(config: StorageConfig) -> CloudStorage:
        """
        Create a cloud storage provider from config.
        
        Args:
            config: StorageConfig object
        
        Returns:
            CloudStorage provider instance
        
        Raises:
            ValueError: If provider type is unknown
        """
        provider = config.provider.lower()
        
        if provider == "s3":
            creds = config.credentials or {}
            return S3CloudStorage(
                bucket=config.bucket,
                region=config.region or "us-east-1",
                aws_access_key_id=creds.get("aws_access_key_id"),
                aws_secret_access_key=creds.get("aws_secret_access_key"),
                storage_class=config.storage_class or "STANDARD",
            )
        
        elif provider == "gcs":
            creds = config.credentials or {}
            return GCSCloudStorage(
                bucket=config.bucket,
                project_id=creds.get("project_id"),
                credentials_path=creds.get("credentials_path"),
                storage_class=config.storage_class or "STANDARD",
            )
        
        elif provider == "azure":
            creds = config.credentials or {}
            return AzureCloudStorage(
                container=config.bucket,
                connection_string=creds.get("connection_string"),
                account_name=creds.get("account_name"),
                account_key=creds.get("account_key"),
                access_tier=config.storage_class or "hot",
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider}")


class ConfigManager:
    """Manages configuration loading and storage."""
    
    @staticmethod
    def load_yaml(config_path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        
        Returns:
            Configuration dictionary
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config is invalid YAML
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    @staticmethod
    def save_yaml(config: Dict[str, Any], config_path: Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to write YAML file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Saved configuration to {config_path}")
    
    @staticmethod
    def create_storage_config(config_dict: Dict[str, Any]) -> StorageConfig:
        """
        Create StorageConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            StorageConfig object
        
        Raises:
            KeyError: If required fields are missing
            ValueError: If configuration is invalid
        """
        # Replace environment variables
        config_dict = ConfigManager._substitute_env_vars(config_dict)
        
        provider = config_dict.get("provider", "").lower()
        bucket = config_dict.get("bucket")
        
        if not provider:
            raise ValueError("'provider' field is required")
        if not bucket:
            raise ValueError("'bucket' field is required")
        if provider not in {"s3", "gcs", "azure"}:
            raise ValueError(f"Unknown provider: {provider}")
        
        return StorageConfig(
            provider=provider,
            bucket=bucket,
            region=config_dict.get("region"),
            credentials=config_dict.get("credentials"),
            storage_class=config_dict.get("storage_class"),
        )
    
    @staticmethod
    def create_sync_config(config_dict: Dict[str, Any]) -> SyncConfig:
        """
        Create SyncConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            SyncConfig object
        """
        storage_config = ConfigManager.create_storage_config(
            config_dict.get("storage", {})
        )
        
        sync_state_db = Path(config_dict.get("sync_state_db", ".sync_state.db"))
        
        conflict_strategy_str = config_dict.get("conflict_strategy", "timestamp_wins")
        conflict_strategy = ConflictStrategy(conflict_strategy_str)
        
        return SyncConfig(
            storage_config=storage_config,
            sync_state_db=sync_state_db,
            conflict_strategy=conflict_strategy,
            max_workers=config_dict.get("max_workers", 4),
            retry_attempts=config_dict.get("retry_attempts", 3),
            retry_delay_seconds=config_dict.get("retry_delay_seconds", 5),
        )
    
    @staticmethod
    def _substitute_env_vars(config: Any) -> Any:
        """
        Recursively substitute environment variables in config.
        
        Format: ${VAR_NAME} or ${VAR_NAME:default_value}
        """
        if isinstance(config, dict):
            return {k: ConfigManager._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [ConfigManager._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Simple environment variable substitution
            import re
            pattern = r'\$\{([^}]+)\}'
            
            def replacer(match):
                var_spec = match.group(1)
                if ':' in var_spec:
                    var_name, default = var_spec.split(':', 1)
                    return os.getenv(var_name.strip(), default.strip())
                else:
                    return os.getenv(var_spec, match.group(0))
            
            return re.sub(pattern, replacer, config)
        else:
            return config


class SyncEngineBuilder:
    """Builder for creating configured sync engines."""
    
    @staticmethod
    def from_config_file(config_path: Path) -> CloudSyncEngine:
        """
        Create a sync engine from a configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        
        Returns:
            Configured CloudSyncEngine
        """
        config_dict = ConfigManager.load_yaml(config_path)
        return SyncEngineBuilder.from_config_dict(config_dict)
    
    @staticmethod
    def from_config_dict(config_dict: Dict[str, Any]) -> CloudSyncEngine:
        """
        Create a sync engine from a configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            Configured CloudSyncEngine
        """
        sync_config = ConfigManager.create_sync_config(config_dict)
        
        # Create cloud storage provider
        cloud_storage = CloudStorageFactory.create(sync_config.storage_config)
        cloud_storage.connect()
        
        # Create sync engine
        engine = CloudSyncEngine(
            cloud_storage=cloud_storage,
            sync_state_db=sync_config.sync_state_db,
            conflict_strategy=sync_config.conflict_strategy,
            max_workers=sync_config.max_workers,
            retry_attempts=sync_config.retry_attempts,
            retry_delay_seconds=sync_config.retry_delay_seconds,
        )
        
        logger.info(f"Created sync engine with {sync_config.storage_config.provider} backend")
        
        return engine
    
    @staticmethod
    def from_env() -> CloudSyncEngine:
        """
        Create a sync engine from environment variables and default config.
        
        Expected environment variables:
        - CLOUD_STORAGE_PROVIDER: s3, gcs, or azure
        - CLOUD_STORAGE_BUCKET: bucket/container name
        - CLOUD_STORAGE_REGION: (S3) region
        - CLOUD_STORAGE_CREDENTIALS_*: provider-specific credentials
        
        Returns:
            Configured CloudSyncEngine
        """
        provider = os.getenv("CLOUD_STORAGE_PROVIDER", "s3").lower()
        bucket = os.getenv("CLOUD_STORAGE_BUCKET")
        
        if not bucket:
            raise ValueError("CLOUD_STORAGE_BUCKET environment variable is required")
        
        config_dict = {
            "storage": {
                "provider": provider,
                "bucket": bucket,
                "region": os.getenv("CLOUD_STORAGE_REGION"),
            },
            "sync_state_db": os.getenv("SYNC_STATE_DB", ".sync_state.db"),
            "conflict_strategy": os.getenv("CONFLICT_STRATEGY", "timestamp_wins"),
            "max_workers": int(os.getenv("SYNC_MAX_WORKERS", "4")),
            "retry_attempts": int(os.getenv("SYNC_RETRY_ATTEMPTS", "3")),
            "retry_delay_seconds": int(os.getenv("SYNC_RETRY_DELAY", "5")),
        }
        
        return SyncEngineBuilder.from_config_dict(config_dict)
