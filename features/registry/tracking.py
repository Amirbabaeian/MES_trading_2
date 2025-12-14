"""
Feature Usage Tracking and Statistics.

This module tracks feature usage in backtests and research notebooks,
collecting statistics to identify usage patterns and underutilized features.
"""

from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import json
import threading

from .catalog import FeatureRegistry, FeatureMetadata, get_registry


@dataclass
class UsageEvent:
    """
    Record of a single feature usage event.
    
    Attributes:
        feature_name: Name of the feature used
        timestamp: When the feature was accessed
        context: Context of usage (e.g., 'backtest', 'notebook', 'production')
        user: Optional user identifier
        session_id: Optional session identifier
        metadata: Additional usage metadata
    """
    feature_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: str = "unknown"
    user: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'feature_name': self.feature_name,
            'timestamp': self.timestamp.isoformat(),
            'context': self.context,
            'user': self.user,
            'session_id': self.session_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UsageEvent':
        """Create from dictionary."""
        return cls(
            feature_name=data['feature_name'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            context=data.get('context', 'unknown'),
            user=data.get('user'),
            session_id=data.get('session_id'),
            metadata=data.get('metadata', {})
        )


@dataclass
class ValidationStats:
    """
    Validation statistics for a feature.
    
    Attributes:
        feature_name: Name of the feature
        total_validations: Total number of validation runs
        passed_validations: Number of successful validations
        failed_validations: Number of failed validations
        average_coverage: Average data coverage (0-1)
        quality_score: Overall quality score (0-1)
        last_validated: Timestamp of last validation
        validation_history: List of recent validation results
    """
    feature_name: str
    total_validations: int = 0
    passed_validations: int = 0
    failed_validations: int = 0
    average_coverage: float = 0.0
    quality_score: float = 0.0
    last_validated: Optional[datetime] = None
    validation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate validation success rate."""
        if self.total_validations == 0:
            return 0.0
        return self.passed_validations / self.total_validations
    
    def add_validation_result(
        self,
        passed: bool,
        coverage: float,
        quality: float,
        details: Optional[Dict[str, Any]] = None
    ):
        """Record a new validation result."""
        self.total_validations += 1
        if passed:
            self.passed_validations += 1
        else:
            self.failed_validations += 1
        
        # Update averages
        self.average_coverage = (
            (self.average_coverage * (self.total_validations - 1) + coverage) /
            self.total_validations
        )
        self.quality_score = (
            (self.quality_score * (self.total_validations - 1) + quality) /
            self.total_validations
        )
        
        self.last_validated = datetime.now()
        
        # Add to history (keep last 100)
        self.validation_history.append({
            'timestamp': self.last_validated.isoformat(),
            'passed': passed,
            'coverage': coverage,
            'quality': quality,
            'details': details or {}
        })
        if len(self.validation_history) > 100:
            self.validation_history = self.validation_history[-100:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'feature_name': self.feature_name,
            'total_validations': self.total_validations,
            'passed_validations': self.passed_validations,
            'failed_validations': self.failed_validations,
            'average_coverage': self.average_coverage,
            'quality_score': self.quality_score,
            'last_validated': self.last_validated.isoformat() if self.last_validated else None,
            'validation_history': self.validation_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationStats':
        """Create from dictionary."""
        return cls(
            feature_name=data['feature_name'],
            total_validations=data['total_validations'],
            passed_validations=data['passed_validations'],
            failed_validations=data['failed_validations'],
            average_coverage=data['average_coverage'],
            quality_score=data['quality_score'],
            last_validated=datetime.fromisoformat(data['last_validated']) if data.get('last_validated') else None,
            validation_history=data.get('validation_history', [])
        )


class UsageTracker:
    """
    Track feature usage across the system.
    
    Records when features are accessed, providing insights into usage patterns
    and helping identify underutilized features.
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
        """Initialize the usage tracker."""
        if self._initialized:
            return
        
        self._events: List[UsageEvent] = []
        self._usage_counts: Dict[str, int] = defaultdict(int)
        self._context_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._validation_stats: Dict[str, ValidationStats] = {}
        self._persistence_path: Optional[Path] = None
        self._auto_save: bool = False
        self._tracking_enabled: bool = True
        self._hooks: List[Callable[[UsageEvent], None]] = []
        self._initialized = True
    
    def configure(
        self,
        persistence_path: Optional[str] = None,
        auto_save: bool = False,
        enabled: bool = True
    ):
        """
        Configure the usage tracker.
        
        Args:
            persistence_path: Path to save usage data
            auto_save: Whether to auto-save after each event
            enabled: Whether tracking is enabled
        """
        if persistence_path:
            self._persistence_path = Path(persistence_path)
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self._auto_save = auto_save
        self._tracking_enabled = enabled
    
    def track(
        self,
        feature_name: str,
        context: str = "unknown",
        user: Optional[str] = None,
        session_id: Optional[str] = None,
        **metadata
    ):
        """
        Record a feature usage event.
        
        Args:
            feature_name: Name of the feature being used
            context: Usage context (e.g., 'backtest', 'notebook')
            user: Optional user identifier
            session_id: Optional session identifier
            **metadata: Additional metadata about the usage
        """
        if not self._tracking_enabled:
            return
        
        event = UsageEvent(
            feature_name=feature_name,
            context=context,
            user=user,
            session_id=session_id,
            metadata=metadata
        )
        
        self._events.append(event)
        self._usage_counts[feature_name] += 1
        self._context_counts[context][feature_name] += 1
        
        # Call hooks
        for hook in self._hooks:
            try:
                hook(event)
            except Exception:
                pass  # Don't let hook failures break tracking
        
        if self._auto_save:
            self.save()
    
    def add_hook(self, hook: Callable[[UsageEvent], None]):
        """
        Add a hook function to be called on each usage event.
        
        Args:
            hook: Callable that takes a UsageEvent
        """
        self._hooks.append(hook)
    
    def track_validation(
        self,
        feature_name: str,
        passed: bool,
        coverage: float = 1.0,
        quality: float = 1.0,
        **details
    ):
        """
        Record a feature validation result.
        
        Args:
            feature_name: Name of the feature validated
            passed: Whether validation passed
            coverage: Data coverage (0-1)
            quality: Quality score (0-1)
            **details: Additional validation details
        """
        if feature_name not in self._validation_stats:
            self._validation_stats[feature_name] = ValidationStats(feature_name=feature_name)
        
        self._validation_stats[feature_name].add_validation_result(
            passed=passed,
            coverage=coverage,
            quality=quality,
            details=details
        )
        
        if self._auto_save:
            self.save()
    
    def get_usage_count(self, feature_name: str) -> int:
        """Get total usage count for a feature."""
        return self._usage_counts.get(feature_name, 0)
    
    def get_usage_by_context(self, feature_name: str) -> Dict[str, int]:
        """Get usage counts by context for a feature."""
        counts = {}
        for context, feature_counts in self._context_counts.items():
            if feature_name in feature_counts:
                counts[context] = feature_counts[feature_name]
        return counts
    
    def get_validation_stats(self, feature_name: str) -> Optional[ValidationStats]:
        """Get validation statistics for a feature."""
        return self._validation_stats.get(feature_name)
    
    def get_events(
        self,
        feature_name: Optional[str] = None,
        context: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[UsageEvent]:
        """
        Retrieve usage events with filtering.
        
        Args:
            feature_name: Filter by feature name
            context: Filter by context
            start_time: Filter events after this time
            end_time: Filter events before this time
            limit: Maximum number of events to return
            
        Returns:
            List of matching usage events
        """
        events = self._events
        
        if feature_name:
            events = [e for e in events if e.feature_name == feature_name]
        
        if context:
            events = [e for e in events if e.context == context]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_most_used(self, limit: int = 10) -> List[tuple[str, int]]:
        """
        Get most frequently used features.
        
        Args:
            limit: Number of features to return
            
        Returns:
            List of (feature_name, count) tuples
        """
        return Counter(self._usage_counts).most_common(limit)
    
    def get_least_used(self, registry: Optional[FeatureRegistry] = None, limit: int = 10) -> List[tuple[str, int]]:
        """
        Get least frequently used features.
        
        Args:
            registry: Feature registry to check registered features
            limit: Number of features to return
            
        Returns:
            List of (feature_name, count) tuples
        """
        if registry is None:
            registry = get_registry()
        
        all_features = {f.name for f in registry.list_all()}
        feature_counts = [(name, self._usage_counts.get(name, 0)) for name in all_features]
        feature_counts.sort(key=lambda x: x[1])
        
        return feature_counts[:limit]
    
    def get_unused_features(self, registry: Optional[FeatureRegistry] = None) -> List[str]:
        """
        Get list of features that have never been used.
        
        Args:
            registry: Feature registry to check registered features
            
        Returns:
            List of unused feature names
        """
        if registry is None:
            registry = get_registry()
        
        all_features = {f.name for f in registry.list_all()}
        used_features = set(self._usage_counts.keys())
        
        return sorted(all_features - used_features)
    
    def get_usage_report(
        self,
        registry: Optional[FeatureRegistry] = None,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive usage report.
        
        Args:
            registry: Feature registry
            period_days: Number of days to include in report
            
        Returns:
            Dictionary with usage statistics
        """
        if registry is None:
            registry = get_registry()
        
        cutoff_time = datetime.now() - timedelta(days=period_days)
        recent_events = [e for e in self._events if e.timestamp >= cutoff_time]
        
        # Calculate statistics
        total_features = len(registry.list_all())
        used_features = len(set(e.feature_name for e in recent_events))
        total_events = len(recent_events)
        
        # Context breakdown
        context_breakdown = defaultdict(int)
        for event in recent_events:
            context_breakdown[event.context] += 1
        
        # Most used features
        feature_counts = Counter(e.feature_name for e in recent_events)
        most_used = feature_counts.most_common(10)
        
        # Validation summary
        validation_summary = {
            'features_validated': len(self._validation_stats),
            'total_validations': sum(s.total_validations for s in self._validation_stats.values()),
            'average_success_rate': (
                sum(s.success_rate for s in self._validation_stats.values()) /
                len(self._validation_stats) if self._validation_stats else 0
            ),
            'average_quality_score': (
                sum(s.quality_score for s in self._validation_stats.values()) /
                len(self._validation_stats) if self._validation_stats else 0
            )
        }
        
        return {
            'period_days': period_days,
            'total_features': total_features,
            'used_features': used_features,
            'unused_features': total_features - used_features,
            'total_events': total_events,
            'context_breakdown': dict(context_breakdown),
            'most_used': most_used,
            'validation_summary': validation_summary,
            'generated_at': datetime.now().isoformat()
        }
    
    def clear_old_events(self, days: int = 90):
        """
        Remove usage events older than specified days.
        
        Args:
            days: Keep events from the last N days
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        self._events = [e for e in self._events if e.timestamp >= cutoff_time]
        
        # Recalculate usage counts from remaining events
        self._usage_counts.clear()
        self._context_counts.clear()
        
        for event in self._events:
            self._usage_counts[event.feature_name] += 1
            self._context_counts[event.context][event.feature_name] += 1
    
    def save(self, path: Optional[str] = None):
        """
        Persist tracking data to file.
        
        Args:
            path: Optional path override
        """
        save_path = Path(path) if path else self._persistence_path
        if not save_path:
            raise ValueError("No persistence path configured")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'events': [e.to_dict() for e in self._events],
            'validation_stats': {
                name: stats.to_dict()
                for name, stats in self._validation_stats.items()
            },
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'total_events': len(self._events),
                'total_features_tracked': len(self._usage_counts)
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: Optional[str] = None):
        """
        Load tracking data from file.
        
        Args:
            path: Optional path override
        """
        load_path = Path(path) if path else self._persistence_path
        if not load_path or not load_path.exists():
            return
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        # Clear existing data
        self._events.clear()
        self._usage_counts.clear()
        self._context_counts.clear()
        self._validation_stats.clear()
        
        # Load events
        for event_data in data.get('events', []):
            event = UsageEvent.from_dict(event_data)
            self._events.append(event)
            self._usage_counts[event.feature_name] += 1
            self._context_counts[event.context][event.feature_name] += 1
        
        # Load validation stats
        for name, stats_data in data.get('validation_stats', {}).items():
            self._validation_stats[name] = ValidationStats.from_dict(stats_data)
    
    def clear(self):
        """Clear all tracking data."""
        self._events.clear()
        self._usage_counts.clear()
        self._context_counts.clear()
        self._validation_stats.clear()


def track_feature_usage(feature_name: str, context: str = "unknown", **metadata):
    """
    Convenient function to track feature usage.
    
    Args:
        feature_name: Name of the feature
        context: Usage context
        **metadata: Additional metadata
    """
    tracker = UsageTracker()
    tracker.track(feature_name, context, **metadata)


def track_validation(feature_name: str, passed: bool, coverage: float = 1.0, quality: float = 1.0, **details):
    """
    Convenient function to track validation results.
    
    Args:
        feature_name: Name of the feature
        passed: Whether validation passed
        coverage: Data coverage
        quality: Quality score
        **details: Additional details
    """
    tracker = UsageTracker()
    tracker.track_validation(feature_name, passed, coverage, quality, **details)


def get_usage_tracker() -> UsageTracker:
    """Get the global usage tracker instance."""
    return UsageTracker()


class TrackedFeature:
    """
    Decorator/wrapper for automatic usage tracking.
    
    Usage:
        @TrackedFeature(context="backtest")
        class MyFeature:
            def compute(self, data):
                return data.mean()
    """
    
    def __init__(self, context: str = "unknown", track_calls: bool = True):
        """
        Initialize tracked feature decorator.
        
        Args:
            context: Default context for tracking
            track_calls: Whether to track compute() calls
        """
        self.context = context
        self.track_calls = track_calls
    
    def __call__(self, cls):
        """Apply tracking to a class."""
        original_init = cls.__init__
        original_compute = getattr(cls, 'compute', None)
        
        def tracked_init(instance, *args, **kwargs):
            original_init(instance, *args, **kwargs)
            # Track instantiation
            feature_name = getattr(cls, '_feature_name', cls.__name__)
            track_feature_usage(feature_name, context=self.context, event='instantiation')
        
        def tracked_compute(instance, *args, **kwargs):
            # Track computation
            feature_name = getattr(cls, '_feature_name', cls.__name__)
            track_feature_usage(feature_name, context=self.context, event='compute')
            return original_compute(instance, *args, **kwargs)
        
        cls.__init__ = tracked_init
        if original_compute:
            cls.compute = tracked_compute
        
        return cls
