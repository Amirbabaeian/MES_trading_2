"""
Alert data structures and management for anomaly detection.

Provides structured alert generation, aggregation, and reporting for data quality
issues detected during validation. Alerts include severity levels, context, and
metadata for manual review workflows.

Features:
- Severity levels (critical, warning, info)
- Alert aggregation to prevent alert fatigue
- Context information (timestamp, asset, values)
- Batch alert processing
- Machine-readable output (DataFrame, JSON)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import pandas as pd
import json
from datetime import datetime


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"  # Data unusable, must fix
    WARNING = "warning"    # Suspicious but may be valid
    INFO = "info"          # Notable but likely valid


class AnomalyType(str, Enum):
    """Types of anomalies detected."""
    PRICE_GAP = "price_gap"
    PRICE_SPIKE = "price_spike"
    VOLUME_SPIKE = "volume_spike"
    VOLUME_ANOMALY = "volume_anomaly"
    INTRADAY_GAP = "intraday_gap"
    SESSION_BOUNDARY_GAP = "session_boundary_gap"


@dataclass
class Alert:
    """
    Represents a single anomaly detection alert.
    
    Attributes:
        severity: Alert severity level
        anomaly_type: Type of anomaly detected
        timestamp: Timestamp of the bar with anomaly
        bar_index: Index of the bar in the dataset
        asset_symbol: Asset symbol/name
        measured_value: Actual measured value
        expected_value: Expected value or range
        threshold_value: Threshold that was exceeded
        message: Human-readable alert message
        metadata: Additional context (dict)
        created_at: Timestamp when alert was generated
    """
    severity: AlertSeverity
    anomaly_type: AnomalyType
    timestamp: Any
    bar_index: int
    asset_symbol: str
    measured_value: float
    expected_value: Optional[float] = None
    threshold_value: Optional[float] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Set created_at to now if not provided."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        data = asdict(self)
        # Convert enums to strings
        data["severity"] = self.severity.value
        data["anomaly_type"] = self.anomaly_type.value
        # Convert timestamps to ISO format strings
        data["timestamp"] = str(self.timestamp) if self.timestamp is not None else None
        data["created_at"] = self.created_at.isoformat() if self.created_at else None
        return data
    
    def to_json(self) -> str:
        """Convert alert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AlertAggregator:
    """
    Aggregates alerts to prevent alert fatigue.
    
    Groups consecutive alerts of the same type within a threshold distance
    to reduce redundant alerts. Useful when multiple bars in a sequence have
    anomalies (e.g., a sustained volume spike).
    """
    
    def __init__(self, max_consecutive_bars: int = 3):
        """
        Initialize AlertAggregator.
        
        Args:
            max_consecutive_bars: Maximum consecutive bars to group together
        """
        self.max_consecutive_bars = max_consecutive_bars
    
    def aggregate(self, alerts: List[Alert]) -> List[Alert]:
        """
        Aggregate consecutive alerts of the same type.
        
        Args:
            alerts: List of Alert objects, ideally sorted by bar_index
            
        Returns:
            Aggregated list of alerts
        """
        if not alerts:
            return []
        
        # Sort by bar_index to ensure proper grouping
        sorted_alerts = sorted(alerts, key=lambda a: a.bar_index)
        
        aggregated = []
        current_group = [sorted_alerts[0]]
        
        for alert in sorted_alerts[1:]:
            # Check if this alert is part of the current group
            last_index = current_group[-1].bar_index
            is_same_type = alert.anomaly_type == current_group[0].anomaly_type
            is_consecutive = alert.bar_index - last_index <= self.max_consecutive_bars
            
            if is_same_type and is_consecutive:
                current_group.append(alert)
            else:
                # Finalize current group and start new one
                aggregated.append(self._aggregate_group(current_group))
                current_group = [alert]
        
        # Don't forget the last group
        aggregated.append(self._aggregate_group(current_group))
        
        return aggregated
    
    def _aggregate_group(self, group: List[Alert]) -> Alert:
        """
        Aggregate a group of consecutive alerts into a single alert.
        
        Args:
            group: List of consecutive alerts of the same type
            
        Returns:
            Single aggregated alert
        """
        if len(group) == 1:
            return group[0]
        
        # Use first alert as base
        base_alert = group[0]
        
        # Calculate aggregate metrics
        measured_values = [a.measured_value for a in group]
        bar_indices = [a.bar_index for a in group]
        timestamps = [a.timestamp for a in group]
        
        # Create aggregated alert
        aggregated = Alert(
            severity=base_alert.severity,
            anomaly_type=base_alert.anomaly_type,
            timestamp=timestamps[0],
            bar_index=bar_indices[0],
            asset_symbol=base_alert.asset_symbol,
            measured_value=measured_values[0],
            expected_value=base_alert.expected_value,
            threshold_value=base_alert.threshold_value,
            message=f"{base_alert.message} (and {len(group) - 1} more bars)",
            metadata={
                "original_count": len(group),
                "bar_indices": bar_indices,
                "measured_values": measured_values,
                "min_value": min(measured_values),
                "max_value": max(measured_values),
                "avg_value": sum(measured_values) / len(measured_values),
                **base_alert.metadata,
            },
            created_at=base_alert.created_at,
        )
        
        return aggregated


class AlertReport:
    """
    Generates reports from a collection of alerts.
    """
    
    @staticmethod
    def to_dataframe(alerts: List[Alert]) -> pd.DataFrame:
        """
        Convert alerts to DataFrame for analysis and reporting.
        
        Args:
            alerts: List of Alert objects
            
        Returns:
            DataFrame with alert information
        """
        if not alerts:
            return pd.DataFrame()
        
        records = [alert.to_dict() for alert in alerts]
        df = pd.DataFrame(records)
        
        # Convert severity and anomaly_type to categorical for better performance
        if "severity" in df.columns:
            df["severity"] = pd.Categorical(df["severity"])
        if "anomaly_type" in df.columns:
            df["anomaly_type"] = pd.Categorical(df["anomaly_type"])
        
        return df
    
    @staticmethod
    def group_by_severity(alerts: List[Alert]) -> Dict[AlertSeverity, List[Alert]]:
        """
        Group alerts by severity level.
        
        Args:
            alerts: List of Alert objects
            
        Returns:
            Dictionary mapping severity to list of alerts
        """
        grouped = {}
        for severity in AlertSeverity:
            grouped[severity] = [a for a in alerts if a.severity == severity]
        return grouped
    
    @staticmethod
    def group_by_type(alerts: List[Alert]) -> Dict[AnomalyType, List[Alert]]:
        """
        Group alerts by anomaly type.
        
        Args:
            alerts: List of Alert objects
            
        Returns:
            Dictionary mapping anomaly type to list of alerts
        """
        grouped = {}
        for atype in AnomalyType:
            grouped[atype] = [a for a in alerts if a.anomaly_type == atype]
        return grouped
    
    @staticmethod
    def summary(alerts: List[Alert]) -> Dict[str, Any]:
        """
        Generate summary statistics for alerts.
        
        Args:
            alerts: List of Alert objects
            
        Returns:
            Dictionary with summary information
        """
        if not alerts:
            return {
                "total_alerts": 0,
                "by_severity": {},
                "by_type": {},
            }
        
        grouped_severity = AlertReport.group_by_severity(alerts)
        grouped_type = AlertReport.group_by_type(alerts)
        
        return {
            "total_alerts": len(alerts),
            "by_severity": {
                severity.value: len(alert_list)
                for severity, alert_list in grouped_severity.items()
            },
            "by_type": {
                atype.value: len(alert_list)
                for atype, alert_list in grouped_type.items()
            },
            "first_alert_timestamp": min(a.timestamp for a in alerts) if alerts else None,
            "last_alert_timestamp": max(a.timestamp for a in alerts) if alerts else None,
        }
