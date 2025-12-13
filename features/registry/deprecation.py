"""
Feature Deprecation Workflow Management.

This module provides a comprehensive deprecation system for features, including
warnings, migration guidance, and version-aware lifecycle management.
"""

from typing import Optional, Dict, Any, Callable, Type
from datetime import datetime
from enum import Enum
import warnings
import functools

from .catalog import FeatureRegistry, FeatureMetadata, FeatureStatus, get_registry


class DeprecationLevel(Enum):
    """Severity level of deprecation."""
    WARNING = "warning"  # Feature is deprecated but still works normally
    STRICT = "strict"  # Feature shows prominent warnings
    PENDING_REMOVAL = "pending_removal"  # Feature will be removed soon
    REMOVED = "removed"  # Feature has been removed


class DeprecationPolicy(Enum):
    """Policy for handling deprecated features."""
    WARN = "warn"  # Issue warnings but allow usage
    PREVENT_NEW = "prevent_new"  # Prevent new usage, allow existing
    BLOCK = "block"  # Block all usage


class DeprecationWarning(UserWarning):
    """Warning category for feature deprecation."""
    pass


class FeatureDeprecationError(Exception):
    """Exception raised when attempting to use a removed feature."""
    pass


class FeatureDeprecationManager:
    """
    Manage feature deprecation lifecycle.
    
    Handles marking features as deprecated, issuing warnings, and managing
    the transition to replacement features.
    """
    
    def __init__(self, registry: Optional[FeatureRegistry] = None):
        """
        Initialize the deprecation manager.
        
        Args:
            registry: Feature registry (uses global if not provided)
        """
        self.registry = registry or get_registry()
        self._deprecation_policy = DeprecationPolicy.WARN
        self._warned_features = set()  # Track which features have been warned about
    
    def set_policy(self, policy: DeprecationPolicy):
        """
        Set the global deprecation policy.
        
        Args:
            policy: Deprecation policy to enforce
        """
        self._deprecation_policy = policy
    
    def deprecate_feature(
        self,
        feature_name: str,
        reason: str,
        deprecated_in: str,
        remove_in: Optional[str] = None,
        replacement: Optional[str] = None,
        migration_guide: Optional[str] = None,
        level: DeprecationLevel = DeprecationLevel.WARNING
    ):
        """
        Mark a feature as deprecated.
        
        Args:
            feature_name: Name of the feature to deprecate
            reason: Reason for deprecation
            deprecated_in: Version in which feature was deprecated
            remove_in: Version in which feature will be removed
            replacement: Name of replacement feature
            migration_guide: Instructions for migrating to replacement
            level: Deprecation severity level
        """
        feature = self.registry.get(feature_name)
        if not feature:
            raise ValueError(f"Feature '{feature_name}' not found in registry")
        
        # Update feature status
        self.registry.update(
            feature_name,
            status=FeatureStatus.DEPRECATED,
            deprecation_info={
                'reason': reason,
                'deprecated_in': deprecated_in,
                'remove_in': remove_in,
                'replacement': replacement,
                'migration_guide': migration_guide,
                'level': level.value,
                'deprecated_at': datetime.now().isoformat()
            }
        )
    
    def undeprecate_feature(self, feature_name: str):
        """
        Remove deprecation status from a feature.
        
        Args:
            feature_name: Name of the feature to undeprecate
        """
        feature = self.registry.get(feature_name)
        if not feature:
            raise ValueError(f"Feature '{feature_name}' not found in registry")
        
        self.registry.update(
            feature_name,
            status=FeatureStatus.ACTIVE,
            deprecation_info=None
        )
    
    def check_deprecation(
        self,
        feature_name: str,
        raise_on_removed: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a feature is deprecated and return deprecation info.
        
        Args:
            feature_name: Name of the feature to check
            raise_on_removed: Whether to raise exception for removed features
            
        Returns:
            Deprecation info dictionary if deprecated, None otherwise
            
        Raises:
            FeatureDeprecationError: If feature is removed and raise_on_removed is True
        """
        feature = self.registry.get(feature_name)
        if not feature:
            return None
        
        if feature.status != FeatureStatus.DEPRECATED:
            return None
        
        deprecation_info = feature.deprecation_info
        if not deprecation_info:
            return None
        
        level = DeprecationLevel(deprecation_info.get('level', 'warning'))
        
        # Check if feature has been removed
        if level == DeprecationLevel.REMOVED and raise_on_removed:
            replacement = deprecation_info.get('replacement')
            msg = f"Feature '{feature_name}' has been removed"
            if replacement:
                msg += f". Use '{replacement}' instead"
            raise FeatureDeprecationError(msg)
        
        return deprecation_info
    
    def warn_if_deprecated(self, feature_name: str, stacklevel: int = 2):
        """
        Issue a warning if the feature is deprecated.
        
        Args:
            feature_name: Name of the feature to check
            stacklevel: Stack level for warning (for proper source location)
        """
        deprecation_info = self.check_deprecation(feature_name, raise_on_removed=True)
        
        if not deprecation_info:
            return
        
        # Only warn once per feature per session (unless it's pending removal)
        level = DeprecationLevel(deprecation_info.get('level', 'warning'))
        if level != DeprecationLevel.PENDING_REMOVAL and feature_name in self._warned_features:
            return
        
        self._warned_features.add(feature_name)
        
        # Build warning message
        msg = self._build_warning_message(feature_name, deprecation_info)
        
        # Issue warning with appropriate category
        warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)
    
    def _build_warning_message(
        self,
        feature_name: str,
        deprecation_info: Dict[str, Any]
    ) -> str:
        """Build a comprehensive deprecation warning message."""
        lines = [f"Feature '{feature_name}' is deprecated"]
        
        reason = deprecation_info.get('reason')
        if reason:
            lines.append(f"Reason: {reason}")
        
        deprecated_in = deprecation_info.get('deprecated_in')
        if deprecated_in:
            lines.append(f"Deprecated in version: {deprecated_in}")
        
        remove_in = deprecation_info.get('remove_in')
        if remove_in:
            lines.append(f"Will be removed in version: {remove_in}")
        
        replacement = deprecation_info.get('replacement')
        if replacement:
            lines.append(f"Replacement: Use '{replacement}' instead")
        
        migration_guide = deprecation_info.get('migration_guide')
        if migration_guide:
            lines.append(f"Migration guide: {migration_guide}")
        
        return ". ".join(lines) + "."
    
    def prevent_usage_if_deprecated(
        self,
        feature_name: str,
        allow_existing: bool = False
    ):
        """
        Prevent usage of deprecated features based on policy.
        
        Args:
            feature_name: Name of the feature
            allow_existing: Whether to allow existing code (vs. new usage)
            
        Raises:
            FeatureDeprecationError: If usage is not allowed
        """
        deprecation_info = self.check_deprecation(feature_name, raise_on_removed=True)
        
        if not deprecation_info:
            return
        
        # Check policy
        if self._deprecation_policy == DeprecationPolicy.WARN:
            self.warn_if_deprecated(feature_name)
        
        elif self._deprecation_policy == DeprecationPolicy.PREVENT_NEW:
            if not allow_existing:
                replacement = deprecation_info.get('replacement')
                msg = f"New usage of deprecated feature '{feature_name}' is not allowed"
                if replacement:
                    msg += f". Use '{replacement}' instead"
                raise FeatureDeprecationError(msg)
            else:
                self.warn_if_deprecated(feature_name)
        
        elif self._deprecation_policy == DeprecationPolicy.BLOCK:
            replacement = deprecation_info.get('replacement')
            msg = f"Usage of deprecated feature '{feature_name}' is blocked"
            if replacement:
                msg += f". Use '{replacement}' instead"
            raise FeatureDeprecationError(msg)
    
    def get_deprecated_features(self) -> Dict[str, FeatureMetadata]:
        """
        Get all deprecated features.
        
        Returns:
            Dictionary mapping feature names to metadata
        """
        all_features = self.registry.list_all(include_deprecated=True)
        return {
            f.name: f
            for f in all_features
            if f.status == FeatureStatus.DEPRECATED
        }
    
    def get_features_pending_removal(self) -> Dict[str, FeatureMetadata]:
        """
        Get features marked for removal in upcoming versions.
        
        Returns:
            Dictionary mapping feature names to metadata
        """
        deprecated = self.get_deprecated_features()
        return {
            name: metadata
            for name, metadata in deprecated.items()
            if metadata.deprecation_info and metadata.deprecation_info.get('remove_in')
        }
    
    def generate_deprecation_report(self) -> str:
        """
        Generate a report of all deprecated features.
        
        Returns:
            Markdown formatted deprecation report
        """
        deprecated = self.get_deprecated_features()
        
        lines = [
            "# Deprecated Features Report",
            "",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            f"**Total Deprecated Features:** {len(deprecated)}",
            ""
        ]
        
        # Group by removal status
        pending_removal = self.get_features_pending_removal()
        no_removal_plan = {
            name: meta for name, meta in deprecated.items()
            if name not in pending_removal
        }
        
        if pending_removal:
            lines.append("## ⚠️ Pending Removal")
            lines.append("")
            lines.append("These features will be removed in upcoming versions:")
            lines.append("")
            
            for name, metadata in sorted(pending_removal.items()):
                lines.extend(self._format_deprecated_feature(name, metadata))
                lines.append("")
        
        if no_removal_plan:
            lines.append("## Deprecated (No Removal Date)")
            lines.append("")
            
            for name, metadata in sorted(no_removal_plan.items()):
                lines.extend(self._format_deprecated_feature(name, metadata))
                lines.append("")
        
        return "\n".join(lines)
    
    def _format_deprecated_feature(
        self,
        name: str,
        metadata: FeatureMetadata
    ) -> list:
        """Format a deprecated feature for the report."""
        lines = [f"### {name}"]
        
        if metadata.deprecation_info:
            info = metadata.deprecation_info
            
            if info.get('deprecated_in'):
                lines.append(f"**Deprecated in:** {info['deprecated_in']}")
            
            if info.get('remove_in'):
                lines.append(f"**Will be removed in:** {info['remove_in']}")
            
            if info.get('reason'):
                lines.append(f"**Reason:** {info['reason']}")
            
            if info.get('replacement'):
                lines.append(f"**Replacement:** `{info['replacement']}`")
            
            if info.get('migration_guide'):
                lines.append(f"**Migration Guide:** {info['migration_guide']}")
        
        lines.append("---")
        return lines
    
    def create_migration_plan(
        self,
        old_feature: str,
        new_feature: str,
        guide: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a migration plan from old feature to new feature.
        
        Args:
            old_feature: Name of deprecated feature
            new_feature: Name of replacement feature
            guide: Optional migration instructions
            
        Returns:
            Migration plan dictionary
        """
        old = self.registry.get(old_feature)
        new = self.registry.get(new_feature)
        
        if not old:
            raise ValueError(f"Old feature '{old_feature}' not found")
        if not new:
            raise ValueError(f"New feature '{new_feature}' not found")
        
        return {
            'old_feature': {
                'name': old.name,
                'version': old.current_version,
                'description': old.description
            },
            'new_feature': {
                'name': new.name,
                'version': new.current_version,
                'description': new.description
            },
            'migration_guide': guide or "No migration guide provided",
            'parameter_changes': self._compare_parameters(old, new),
            'created_at': datetime.now().isoformat()
        }
    
    def _compare_parameters(
        self,
        old: FeatureMetadata,
        new: FeatureMetadata
    ) -> Dict[str, Any]:
        """Compare parameters between old and new features."""
        old_params = set(old.parameters.keys())
        new_params = set(new.parameters.keys())
        
        return {
            'removed': list(old_params - new_params),
            'added': list(new_params - old_params),
            'common': list(old_params & new_params)
        }


def deprecated(
    reason: str,
    deprecated_in: str,
    remove_in: Optional[str] = None,
    replacement: Optional[str] = None,
    migration_guide: Optional[str] = None,
    level: DeprecationLevel = DeprecationLevel.WARNING
) -> Callable:
    """
    Decorator to mark a feature class or function as deprecated.
    
    Usage:
        @deprecated(
            reason="Replaced by more efficient implementation",
            deprecated_in="2.0.0",
            remove_in="3.0.0",
            replacement="NewFeature"
        )
        class OldFeature:
            def compute(self, data):
                return data.mean()
    
    Args:
        reason: Reason for deprecation
        deprecated_in: Version in which feature was deprecated
        remove_in: Version in which feature will be removed
        replacement: Name of replacement feature
        migration_guide: Migration instructions
        level: Deprecation severity level
        
    Returns:
        Decorator function
    """
    def decorator(obj: Type) -> Type:
        feature_name = getattr(obj, '_feature_name', obj.__name__)
        manager = FeatureDeprecationManager()
        
        # Mark feature as deprecated in registry if it exists
        if manager.registry.exists(feature_name):
            manager.deprecate_feature(
                feature_name=feature_name,
                reason=reason,
                deprecated_in=deprecated_in,
                remove_in=remove_in,
                replacement=replacement,
                migration_guide=migration_guide,
                level=level
            )
        
        # Wrap the class/function to issue warnings on usage
        if isinstance(obj, type):
            # It's a class - wrap __init__
            original_init = obj.__init__
            
            @functools.wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                manager.warn_if_deprecated(feature_name, stacklevel=2)
                return original_init(self, *args, **kwargs)
            
            obj.__init__ = wrapped_init
            obj._deprecated = True
            obj._deprecation_info = {
                'reason': reason,
                'deprecated_in': deprecated_in,
                'remove_in': remove_in,
                'replacement': replacement,
                'migration_guide': migration_guide,
                'level': level.value
            }
        else:
            # It's a function - wrap directly
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                manager.warn_if_deprecated(feature_name, stacklevel=2)
                return obj(*args, **kwargs)
            
            wrapper._deprecated = True
            wrapper._deprecation_info = {
                'reason': reason,
                'deprecated_in': deprecated_in,
                'remove_in': remove_in,
                'replacement': replacement,
                'migration_guide': migration_guide,
                'level': level.value
            }
            return wrapper
        
        return obj
    
    return decorator


def check_feature_deprecation(feature_name: str) -> Optional[Dict[str, Any]]:
    """
    Convenient function to check if a feature is deprecated.
    
    Args:
        feature_name: Name of the feature to check
        
    Returns:
        Deprecation info if deprecated, None otherwise
    """
    manager = FeatureDeprecationManager()
    return manager.check_deprecation(feature_name, raise_on_removed=False)


def warn_if_deprecated(feature_name: str):
    """
    Convenient function to issue deprecation warning.
    
    Args:
        feature_name: Name of the feature to check
    """
    manager = FeatureDeprecationManager()
    manager.warn_if_deprecated(feature_name, stacklevel=3)


def get_deprecated_features() -> Dict[str, FeatureMetadata]:
    """Get all deprecated features."""
    manager = FeatureDeprecationManager()
    return manager.get_deprecated_features()


def generate_deprecation_report() -> str:
    """Generate a deprecation report."""
    manager = FeatureDeprecationManager()
    return manager.generate_deprecation_report()
