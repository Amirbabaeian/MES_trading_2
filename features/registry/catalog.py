"""
Feature Registry Catalog - Core registry implementation for feature management.

This module provides the central catalog system for registering, storing, and 
managing feature metadata across the research platform.
"""

import json
import inspect
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Type
from enum import Enum
import threading


class FeatureStatus(Enum):
    """Status of a feature in its lifecycle."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    ARCHIVED = "archived"


@dataclass
class VersionInfo:
    """Version information for a feature."""
    version: str
    created_at: datetime
    created_by: Optional[str] = None
    changes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'changes': self.changes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VersionInfo':
        """Create from dictionary."""
        return cls(
            version=data['version'],
            created_at=datetime.fromisoformat(data['created_at']),
            created_by=data.get('created_by'),
            changes=data.get('changes')
        )


@dataclass
class FeatureMetadata:
    """
    Comprehensive metadata for a registered feature.
    
    Attributes:
        name: Unique identifier for the feature
        description: Human-readable description of what the feature computes
        computation_logic: High-level summary of the computation approach
        parameters: Dictionary of parameter names to their specifications
        data_dependencies: List of data sources or other features this depends on
        version_history: List of version information
        current_version: The active version of this feature
        status: Current lifecycle status (active, deprecated, etc.)
        category: Feature category for organization (e.g., "momentum", "volatility")
        assets: List of assets this feature applies to
        tags: Additional searchable tags
        author: Feature author/creator
        created_at: Timestamp when first registered
        updated_at: Timestamp of last update
        deprecation_info: Information about deprecation if applicable
        computation_cost: Estimated computational cost (low/medium/high)
        example_usage: Example code showing how to use the feature
        feature_class: Reference to the actual feature class (if applicable)
        feature_id: Unique hash identifier
    """
    name: str
    description: str
    computation_logic: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    data_dependencies: List[str] = field(default_factory=list)
    version_history: List[VersionInfo] = field(default_factory=list)
    current_version: str = "1.0.0"
    status: FeatureStatus = FeatureStatus.ACTIVE
    category: Optional[str] = None
    assets: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    author: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    deprecation_info: Optional[Dict[str, Any]] = None
    computation_cost: str = "medium"
    example_usage: Optional[str] = None
    feature_class: Optional[Type] = field(default=None, repr=False)
    feature_id: str = field(init=False)
    
    def __post_init__(self):
        """Generate unique feature ID based on name and version."""
        if not hasattr(self, 'feature_id') or not self.feature_id:
            id_string = f"{self.name}:{self.current_version}"
            self.feature_id = hashlib.md5(id_string.encode()).hexdigest()[:12]
        
        # Ensure tags is a set
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
        
        # Ensure status is FeatureStatus enum
        if isinstance(self.status, str):
            self.status = FeatureStatus(self.status)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        data = {
            'name': self.name,
            'description': self.description,
            'computation_logic': self.computation_logic,
            'parameters': self.parameters,
            'data_dependencies': self.data_dependencies,
            'version_history': [v.to_dict() for v in self.version_history],
            'current_version': self.current_version,
            'status': self.status.value,
            'category': self.category,
            'assets': self.assets,
            'tags': list(self.tags),
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'deprecation_info': self.deprecation_info,
            'computation_cost': self.computation_cost,
            'example_usage': self.example_usage,
            'feature_id': self.feature_id
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureMetadata':
        """Create metadata from dictionary."""
        # Convert version history
        version_history = [
            VersionInfo.from_dict(v) for v in data.get('version_history', [])
        ]
        
        return cls(
            name=data['name'],
            description=data['description'],
            computation_logic=data['computation_logic'],
            parameters=data.get('parameters', {}),
            data_dependencies=data.get('data_dependencies', []),
            version_history=version_history,
            current_version=data.get('current_version', '1.0.0'),
            status=FeatureStatus(data.get('status', 'active')),
            category=data.get('category'),
            assets=data.get('assets', []),
            tags=set(data.get('tags', [])),
            author=data.get('author'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            deprecation_info=data.get('deprecation_info'),
            computation_cost=data.get('computation_cost', 'medium'),
            example_usage=data.get('example_usage')
        )
    
    def add_version(self, version: str, changes: str, author: Optional[str] = None):
        """Add a new version to the version history."""
        version_info = VersionInfo(
            version=version,
            created_at=datetime.now(),
            created_by=author,
            changes=changes
        )
        self.version_history.append(version_info)
        self.current_version = version
        self.updated_at = datetime.now()


class FeatureRegistry:
    """
    Central registry for managing all features in the system.
    
    This is a singleton class that maintains the catalog of all registered
    features with their metadata. It provides methods for registration,
    retrieval, and persistence.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the registry."""
        if self._initialized:
            return
        
        self._features: Dict[str, FeatureMetadata] = {}
        self._features_by_id: Dict[str, FeatureMetadata] = {}
        self._version_map: Dict[str, Dict[str, FeatureMetadata]] = {}  # name -> version -> metadata
        self._categories: Dict[str, List[str]] = {}  # category -> feature names
        self._asset_map: Dict[str, List[str]] = {}  # asset -> feature names
        self._tag_map: Dict[str, List[str]] = {}  # tag -> feature names
        self._persistence_path: Optional[Path] = None
        self._auto_save: bool = False
        self._initialized = True
    
    def configure(self, persistence_path: Optional[str] = None, auto_save: bool = False):
        """
        Configure the registry settings.
        
        Args:
            persistence_path: Path to JSON file for persisting registry data
            auto_save: Whether to automatically save after each registration
        """
        if persistence_path:
            self._persistence_path = Path(persistence_path)
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self._auto_save = auto_save
    
    def register(
        self,
        name: str,
        description: str,
        computation_logic: str,
        feature_class: Optional[Type] = None,
        **kwargs
    ) -> FeatureMetadata:
        """
        Register a new feature in the catalog.
        
        Args:
            name: Unique feature name
            description: Human-readable description
            computation_logic: Summary of computation approach
            feature_class: Optional reference to the feature class
            **kwargs: Additional metadata fields
            
        Returns:
            FeatureMetadata object for the registered feature
            
        Raises:
            ValueError: If a feature with the same name already exists
        """
        if name in self._features:
            existing = self._features[name]
            if existing.current_version == kwargs.get('current_version', '1.0.0'):
                raise ValueError(
                    f"Feature '{name}' version {existing.current_version} already registered. "
                    "Use update() or register a new version."
                )
        
        # Create metadata
        metadata = FeatureMetadata(
            name=name,
            description=description,
            computation_logic=computation_logic,
            feature_class=feature_class,
            **kwargs
        )
        
        # Store in main registry
        self._features[name] = metadata
        self._features_by_id[metadata.feature_id] = metadata
        
        # Store in version map
        if name not in self._version_map:
            self._version_map[name] = {}
        self._version_map[name][metadata.current_version] = metadata
        
        # Update indices
        self._update_indices(metadata)
        
        # Auto-save if enabled
        if self._auto_save:
            self.save()
        
        return metadata
    
    def _update_indices(self, metadata: FeatureMetadata):
        """Update internal indices for efficient lookup."""
        # Category index
        if metadata.category:
            if metadata.category not in self._categories:
                self._categories[metadata.category] = []
            if metadata.name not in self._categories[metadata.category]:
                self._categories[metadata.category].append(metadata.name)
        
        # Asset index
        for asset in metadata.assets:
            if asset not in self._asset_map:
                self._asset_map[asset] = []
            if metadata.name not in self._asset_map[asset]:
                self._asset_map[asset].append(metadata.name)
        
        # Tag index
        for tag in metadata.tags:
            if tag not in self._tag_map:
                self._tag_map[tag] = []
            if metadata.name not in self._tag_map[tag]:
                self._tag_map[tag].append(metadata.name)
    
    def update(self, name: str, **kwargs) -> FeatureMetadata:
        """
        Update an existing feature's metadata.
        
        Args:
            name: Feature name to update
            **kwargs: Fields to update
            
        Returns:
            Updated FeatureMetadata object
            
        Raises:
            KeyError: If feature not found
        """
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not found in registry")
        
        metadata = self._features[name]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        metadata.updated_at = datetime.now()
        
        # Update indices
        self._update_indices(metadata)
        
        if self._auto_save:
            self.save()
        
        return metadata
    
    def get(self, name: str, version: Optional[str] = None) -> Optional[FeatureMetadata]:
        """
        Retrieve feature metadata by name and optionally version.
        
        Args:
            name: Feature name
            version: Optional specific version (defaults to current)
            
        Returns:
            FeatureMetadata if found, None otherwise
        """
        if version:
            return self._version_map.get(name, {}).get(version)
        return self._features.get(name)
    
    def get_by_id(self, feature_id: str) -> Optional[FeatureMetadata]:
        """Retrieve feature metadata by ID."""
        return self._features_by_id.get(feature_id)
    
    def list_all(self, include_deprecated: bool = False) -> List[FeatureMetadata]:
        """
        List all registered features.
        
        Args:
            include_deprecated: Whether to include deprecated features
            
        Returns:
            List of FeatureMetadata objects
        """
        features = list(self._features.values())
        if not include_deprecated:
            features = [f for f in features if f.status != FeatureStatus.DEPRECATED]
        return features
    
    def list_by_category(self, category: str) -> List[FeatureMetadata]:
        """List all features in a specific category."""
        feature_names = self._categories.get(category, [])
        return [self._features[name] for name in feature_names if name in self._features]
    
    def list_by_asset(self, asset: str) -> List[FeatureMetadata]:
        """List all features applicable to a specific asset."""
        feature_names = self._asset_map.get(asset, [])
        return [self._features[name] for name in feature_names if name in self._features]
    
    def list_by_tag(self, tag: str) -> List[FeatureMetadata]:
        """List all features with a specific tag."""
        feature_names = self._tag_map.get(tag, [])
        return [self._features[name] for name in feature_names if name in self._features]
    
    def list_categories(self) -> List[str]:
        """Get list of all categories."""
        return sorted(self._categories.keys())
    
    def list_assets(self) -> List[str]:
        """Get list of all assets."""
        return sorted(self._asset_map.keys())
    
    def list_tags(self) -> List[str]:
        """Get list of all tags."""
        return sorted(self._tag_map.keys())
    
    def exists(self, name: str, version: Optional[str] = None) -> bool:
        """Check if a feature exists in the registry."""
        if version:
            return name in self._version_map and version in self._version_map[name]
        return name in self._features
    
    def remove(self, name: str, version: Optional[str] = None):
        """
        Remove a feature from the registry.
        
        Args:
            name: Feature name
            version: Optional specific version to remove
        """
        if version:
            # Remove specific version
            if name in self._version_map and version in self._version_map[name]:
                metadata = self._version_map[name][version]
                del self._version_map[name][version]
                if metadata.feature_id in self._features_by_id:
                    del self._features_by_id[metadata.feature_id]
        else:
            # Remove all versions
            if name in self._features:
                metadata = self._features[name]
                del self._features[name]
                del self._features_by_id[metadata.feature_id]
                if name in self._version_map:
                    del self._version_map[name]
                
                # Clean up indices
                if metadata.category and metadata.category in self._categories:
                    self._categories[metadata.category] = [
                        n for n in self._categories[metadata.category] if n != name
                    ]
                
                for asset in metadata.assets:
                    if asset in self._asset_map:
                        self._asset_map[asset] = [
                            n for n in self._asset_map[asset] if n != name
                        ]
                
                for tag in metadata.tags:
                    if tag in self._tag_map:
                        self._tag_map[tag] = [
                            n for n in self._tag_map[tag] if n != name
                        ]
        
        if self._auto_save:
            self.save()
    
    def clear(self):
        """Clear all features from the registry."""
        self._features.clear()
        self._features_by_id.clear()
        self._version_map.clear()
        self._categories.clear()
        self._asset_map.clear()
        self._tag_map.clear()
    
    def save(self, path: Optional[str] = None):
        """
        Persist the registry to a JSON file.
        
        Args:
            path: Optional path override (uses configured path if not provided)
        """
        save_path = Path(path) if path else self._persistence_path
        if not save_path:
            raise ValueError("No persistence path configured")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        data = {
            'features': {name: metadata.to_dict() for name, metadata in self._features.items()},
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'feature_count': len(self._features)
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Optional[str] = None):
        """
        Load the registry from a JSON file.
        
        Args:
            path: Optional path override (uses configured path if not provided)
        """
        load_path = Path(path) if path else self._persistence_path
        if not load_path or not load_path.exists():
            return
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        # Clear existing data
        self.clear()
        
        # Load features
        for name, feature_data in data.get('features', {}).items():
            metadata = FeatureMetadata.from_dict(feature_data)
            self._features[name] = metadata
            self._features_by_id[metadata.feature_id] = metadata
            
            # Rebuild version map
            if name not in self._version_map:
                self._version_map[name] = {}
            self._version_map[name][metadata.current_version] = metadata
            
            # Rebuild indices
            self._update_indices(metadata)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the registry."""
        features = self.list_all(include_deprecated=True)
        
        status_counts = {}
        for status in FeatureStatus:
            status_counts[status.value] = sum(1 for f in features if f.status == status)
        
        return {
            'total_features': len(features),
            'active_features': status_counts.get(FeatureStatus.ACTIVE.value, 0),
            'deprecated_features': status_counts.get(FeatureStatus.DEPRECATED.value, 0),
            'experimental_features': status_counts.get(FeatureStatus.EXPERIMENTAL.value, 0),
            'categories': len(self._categories),
            'assets': len(self._asset_map),
            'tags': len(self._tag_map),
            'status_breakdown': status_counts
        }


def register_feature(
    name: Optional[str] = None,
    description: Optional[str] = None,
    computation_logic: Optional[str] = None,
    **metadata_kwargs
) -> Callable:
    """
    Decorator for automatic feature registration.
    
    Usage:
        @register_feature(
            name="momentum_10d",
            description="10-day momentum indicator",
            computation_logic="Returns over trailing 10 days",
            category="momentum",
            assets=["stocks", "futures"]
        )
        class Momentum10D:
            def compute(self, data):
                return data.pct_change(10)
    
    Args:
        name: Feature name (defaults to class name)
        description: Feature description (extracted from docstring if not provided)
        computation_logic: Computation summary
        **metadata_kwargs: Additional metadata fields
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type) -> Type:
        registry = FeatureRegistry()
        
        # Extract metadata from class if not provided
        feature_name = name or cls.__name__
        feature_desc = description or (cls.__doc__ or "").strip().split('\n')[0]
        
        # Try to extract computation logic from docstring or method
        if computation_logic:
            comp_logic = computation_logic
        else:
            # Look for compute method docstring
            if hasattr(cls, 'compute') and cls.compute.__doc__:
                comp_logic = cls.compute.__doc__.strip().split('\n')[0]
            else:
                comp_logic = "No computation logic provided"
        
        # Extract parameter information from compute method signature
        parameters = {}
        if hasattr(cls, 'compute'):
            sig = inspect.signature(cls.compute)
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                parameters[param_name] = {
                    'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                    'default': str(param.default) if param.default != inspect.Parameter.empty else None
                }
        
        # Register the feature
        registry.register(
            name=feature_name,
            description=feature_desc,
            computation_logic=comp_logic,
            feature_class=cls,
            parameters=parameters,
            **metadata_kwargs
        )
        
        # Add registry metadata to class
        cls._feature_name = feature_name
        cls._feature_metadata = registry.get(feature_name)
        
        return cls
    
    return decorator


def extract_metadata_from_class(feature_class: Type) -> Dict[str, Any]:
    """
    Extract metadata from a feature class through introspection.
    
    This utility function inspects a class to automatically extract:
    - Description from docstring
    - Parameters from method signatures
    - Type hints
    - Default values
    
    Args:
        feature_class: The feature class to inspect
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata = {
        'name': feature_class.__name__,
        'description': '',
        'computation_logic': '',
        'parameters': {},
        'data_dependencies': []
    }
    
    # Extract from docstring
    if feature_class.__doc__:
        doc_lines = feature_class.__doc__.strip().split('\n')
        metadata['description'] = doc_lines[0].strip()
        if len(doc_lines) > 1:
            metadata['computation_logic'] = ' '.join(
                line.strip() for line in doc_lines[1:] if line.strip()
            )
    
    # Extract parameters from compute method
    if hasattr(feature_class, 'compute'):
        sig = inspect.signature(feature_class.compute)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            metadata['parameters'][param_name] = {
                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any',
                'default': str(param.default) if param.default != inspect.Parameter.empty else None,
                'required': param.default == inspect.Parameter.empty
            }
    
    # Look for dependencies attribute
    if hasattr(feature_class, 'dependencies'):
        deps = getattr(feature_class, 'dependencies')
        if isinstance(deps, (list, tuple)):
            metadata['data_dependencies'] = list(deps)
    
    return metadata


# Global registry instance
_global_registry = FeatureRegistry()


def get_registry() -> FeatureRegistry:
    """Get the global feature registry instance."""
    return _global_registry
