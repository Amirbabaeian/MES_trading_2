"""
Validation pipeline orchestration and automation.

Orchestrates all validation checks in the correct order with blocking logic
and state management. Integrates results from schema, timestamp, missing bars,
and anomaly validators.

Features:
- Sequential validation orchestration
- Blocking logic based on severity
- Validation state tracking
- Error handling and recovery
- Progress logging
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
from enum import Enum
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

from .validation import DataValidator, ValidationResult, Violation
from .anomaly_detectors import (
    GapDetector,
    ZScoreSpikeDetector,
    IQRSpikeDetector,
    VolumeAnomalyDetector,
    GapDetectionConfig,
    SpikeDetectionConfig,
    VolumeDetectionConfig,
)
from .alerts import Alert, AlertAggregator, AlertSeverity, AnomalyType
from .reports import (
    ValidationReport,
    ValidatorResult,
    ValidatorType,
    DetailedIssue,
    SeverityLevel,
    ReportAggregator,
)
from .override import OverrideManager, ValidationMetadata, OverrideStatus

logger = logging.getLogger(__name__)


class ValidationState(str, Enum):
    """States of the validation pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED_BLOCKED = "failed-blocked"
    FAILED_OVERRIDDEN = "failed-overridden"


@dataclass
class PipelineConfig:
    """Configuration for validation pipeline."""
    enable_schema_validation: bool = True
    enable_timestamp_validation: bool = True
    enable_missing_bars_validation: bool = False
    enable_gap_detection: bool = True
    enable_spike_detection: bool = True
    enable_volume_detection: bool = True
    
    # Severity mappings
    schema_violations_severity: SeverityLevel = SeverityLevel.CRITICAL
    timestamp_violations_severity: SeverityLevel = SeverityLevel.CRITICAL
    missing_bars_severity: SeverityLevel = SeverityLevel.WARNING
    gap_detection_severity: SeverityLevel = SeverityLevel.WARNING
    spike_detection_severity: SeverityLevel = SeverityLevel.WARNING
    volume_detection_severity: SeverityLevel = SeverityLevel.INFO
    
    # Anomaly detection configs
    gap_config: Optional[GapDetectionConfig] = None
    spike_config: Optional[SpikeDetectionConfig] = None
    volume_config: Optional[VolumeDetectionConfig] = None
    
    # Override management
    allow_overrides: bool = True
    require_approval: bool = True
    
    # Output
    save_reports: bool = True
    report_dir: Optional[Path] = None


@dataclass
class PipelineResult:
    """Result from validation pipeline."""
    asset: str
    validation_state: ValidationState
    passed: bool
    report: Optional[ValidationReport] = None
    can_promote: bool = False
    blocking_validators: List[ValidatorType] = field(default_factory=list)
    override_count: int = 0


class ValidationOrchestrator:
    """
    Orchestrates the complete validation pipeline.
    
    Runs validators in sequence with appropriate blocking logic:
    1. Schema validation (blocks if fails)
    2. Timestamp validation (blocks if fails)
    3. Anomaly detection (warns if issues found)
    4. Missing bars detection (warns if gaps found)
    
    Generates comprehensive reports and tracks validation state.
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        override_manager: Optional[OverrideManager] = None,
    ):
        """
        Initialize ValidationOrchestrator.
        
        Args:
            config: PipelineConfig for customization
            override_manager: OverrideManager for override handling
        """
        self.config = config or PipelineConfig()
        self.override_manager = override_manager or OverrideManager()
        
        # Initialize validators
        self.schema_validator = DataValidator()
        self.gap_detector = GapDetector(config=self.config.gap_config)
        self.zscore_detector = ZScoreSpikeDetector(config=self.config.spike_config)
        self.iqr_detector = IQRSpikeDetector(config=self.config.spike_config)
        self.volume_detector = VolumeAnomalyDetector(config=self.config.volume_config)
        self.alert_aggregator = AlertAggregator()
    
    def validate(
        self,
        df: pd.DataFrame,
        asset: str,
    ) -> PipelineResult:
        """
        Run complete validation pipeline on a dataset.
        
        Args:
            df: OHLCV DataFrame to validate
            asset: Asset identifier
        
        Returns:
            PipelineResult with validation outcome
        """
        logger.info(f"Starting validation pipeline for {asset}")
        
        validation_timestamp = datetime.utcnow()
        state = ValidationState.RUNNING
        results = []
        
        try:
            # Schema validation (blocking)
            if self.config.enable_schema_validation:
                logger.info(f"  Running schema validation...")
                schema_result = self._validate_schema(df, asset)
                results.append(schema_result)
                if not schema_result.passed:
                    logger.warning(f"    Schema validation FAILED")
                    state = ValidationState.FAILED_BLOCKED
                    return self._create_result(
                        asset, state, results, validation_timestamp
                    )
            
            # Timestamp validation (blocking)
            if self.config.enable_timestamp_validation:
                logger.info(f"  Running timestamp validation...")
                timestamp_result = self._validate_timestamps(df, asset)
                results.append(timestamp_result)
                if not timestamp_result.passed:
                    logger.warning(f"    Timestamp validation FAILED")
                    state = ValidationState.FAILED_BLOCKED
                    return self._create_result(
                        asset, state, results, validation_timestamp
                    )
            
            # Anomaly detection (warnings)
            if self.config.enable_gap_detection or self.config.enable_spike_detection or self.config.enable_volume_detection:
                logger.info(f"  Running anomaly detection...")
                anomaly_result = self._detect_anomalies(df, asset)
                results.append(anomaly_result)
            
            # Missing bars detection (warnings)
            if self.config.enable_missing_bars_validation:
                logger.info(f"  Running missing bars validation...")
                missing_bars_result = self._detect_missing_bars(df, asset)
                results.append(missing_bars_result)
            
            # Check for blocking issues
            blocking_validators = [
                r.validator_type for r in results if r.has_blocking_issues()
            ]
            
            if blocking_validators:
                state = ValidationState.FAILED_BLOCKED
            else:
                state = ValidationState.PASSED
            
        except Exception as e:
            logger.error(f"Validation pipeline failed with error: {e}", exc_info=True)
            state = ValidationState.FAILED_BLOCKED
        
        return self._create_result(asset, state, results, validation_timestamp)
    
    def _validate_schema(self, df: pd.DataFrame, asset: str) -> ValidatorResult:
        """Run schema validation."""
        validation_result = self.schema_validator.validate(df)
        
        # Convert to ValidatorResult
        issues = []
        for violation in validation_result.violations:
            issue = DetailedIssue(
                asset=asset,
                bar_timestamp=violation.timestamp,
                bar_index=violation.bar_index,
                issue_type=violation.violation_type,
                severity=self.config.schema_violations_severity,
                message=violation.message,
                validator_type=ValidatorType.SCHEMA,
                context={
                    "column": violation.column,
                },
            )
            issues.append(issue)
        
        return ValidatorResult(
            validator_type=ValidatorType.SCHEMA,
            asset=asset,
            passed=validation_result.passed,
            total_bars_checked=validation_result.total_bars_checked,
            issues=issues,
            summary=validation_result.summary,
        )
    
    def _validate_timestamps(self, df: pd.DataFrame, asset: str) -> ValidatorResult:
        """Run timestamp validation."""
        validation_result = self.schema_validator.validate(df)
        
        # Filter for timestamp violations
        timestamp_violations = [
            v for v in validation_result.violations
            if v.violation_type == "timestamp"
        ]
        
        issues = []
        for violation in timestamp_violations:
            issue = DetailedIssue(
                asset=asset,
                bar_timestamp=violation.timestamp,
                bar_index=violation.bar_index,
                issue_type=violation.violation_type,
                severity=self.config.timestamp_violations_severity,
                message=violation.message,
                validator_type=ValidatorType.TIMESTAMP,
            )
            issues.append(issue)
        
        return ValidatorResult(
            validator_type=ValidatorType.TIMESTAMP,
            asset=asset,
            passed=len(issues) == 0,
            total_bars_checked=len(df),
            issues=issues,
            summary={"validation_stage": "timestamp"},
        )
    
    def _detect_anomalies(self, df: pd.DataFrame, asset: str) -> ValidatorResult:
        """Run anomaly detection (gaps, spikes, volume)."""
        alerts = []
        
        # Gap detection
        if self.config.enable_gap_detection:
            gap_alerts = self.gap_detector.detect(df, asset_symbol=asset)
            alerts.extend(gap_alerts)
        
        # Spike detection (Z-score)
        if self.config.enable_spike_detection:
            spike_alerts = self.zscore_detector.detect(df, asset_symbol=asset)
            alerts.extend(spike_alerts)
        
        # Volume anomalies
        if self.config.enable_volume_detection:
            volume_alerts = self.volume_detector.detect(df, asset_symbol=asset)
            alerts.extend(volume_alerts)
        
        # Aggregate alerts
        aggregated_alerts = self.alert_aggregator.aggregate(alerts)
        
        # Convert alerts to DetailedIssue
        issues = []
        for alert in aggregated_alerts:
            # Map alert severity to issue severity
            severity_map = {
                AlertSeverity.CRITICAL: SeverityLevel.CRITICAL,
                AlertSeverity.WARNING: SeverityLevel.WARNING,
                AlertSeverity.INFO: SeverityLevel.INFO,
            }
            
            issue = DetailedIssue(
                asset=asset,
                bar_timestamp=alert.timestamp,
                bar_index=alert.bar_index,
                issue_type=alert.anomaly_type.value,
                severity=severity_map.get(alert.severity, SeverityLevel.INFO),
                measured_value=alert.measured_value,
                expected_value=alert.expected_value,
                threshold_value=alert.threshold_value,
                message=alert.message,
                validator_type=ValidatorType.ANOMALY,
                metadata=alert.metadata,
            )
            issues.append(issue)
        
        return ValidatorResult(
            validator_type=ValidatorType.ANOMALY,
            asset=asset,
            passed=len(issues) == 0,
            total_bars_checked=len(df),
            issues=issues,
            summary={
                "total_alerts": len(aggregated_alerts),
                "by_severity": {
                    severity.value: len([a for a in aggregated_alerts if a.severity == severity])
                    for severity in AlertSeverity
                },
            },
        )
    
    def _detect_missing_bars(self, df: pd.DataFrame, asset: str) -> ValidatorResult:
        """Run missing bars detection."""
        # Placeholder - would integrate with MissingBarsValidator
        # For now, return a passing result
        return ValidatorResult(
            validator_type=ValidatorType.MISSING_BARS,
            asset=asset,
            passed=True,
            total_bars_checked=len(df),
            issues=[],
            summary={"missing_bars_found": 0},
        )
    
    def _create_result(
        self,
        asset: str,
        state: ValidationState,
        results: List[ValidatorResult],
        validation_timestamp: datetime,
    ) -> PipelineResult:
        """Create pipeline result from validation state and results."""
        # Create comprehensive report
        report = ReportAggregator.create_report(asset, results)
        
        # Determine if data can be promoted
        can_promote = state == ValidationState.PASSED
        
        # Check for overrides
        override_count = 0
        if state == ValidationState.FAILED_BLOCKED and self.config.allow_overrides:
            active_overrides = self.override_manager.get_active_overrides(asset)
            if active_overrides:
                can_promote = True
                state = ValidationState.FAILED_OVERRIDDEN
                override_count = len(active_overrides)
        
        # Create metadata
        metadata = ValidationMetadata(
            asset=asset,
            validation_timestamp=validation_timestamp,
            validation_state=state.value,
            passed=state == ValidationState.PASSED,
            total_issues=len(report.detailed_issues),
            critical_issues=len(report.get_critical_issues()),
            overrides=self.override_manager.get_overrides(asset),
            override_count=override_count,
        )
        
        # Save metadata if configured
        if self.config.save_reports:
            self.override_manager.save_metadata(asset, metadata)
        
        # Save reports if configured
        if self.config.save_reports and self.config.report_dir:
            self._save_reports(report)
        
        # Log result
        self._log_result(state, asset, report)
        
        return PipelineResult(
            asset=asset,
            validation_state=state,
            passed=state == ValidationState.PASSED,
            report=report,
            can_promote=can_promote,
            blocking_validators=report.summary.blocking_validators if report.summary else [],
            override_count=override_count,
        )
    
    def _save_reports(self, report: ValidationReport) -> None:
        """Save reports to disk."""
        from .reports import JSONReportGenerator, TextReportGenerator, HTMLReportGenerator
        
        if not self.config.report_dir:
            return
        
        report_dir = Path(self.config.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = report.validation_timestamp.strftime("%Y%m%d_%H%M%S")
        asset = report.asset.replace("/", "_")
        
        # Save JSON
        json_file = report_dir / f"{asset}_{timestamp}_validation.json"
        JSONReportGenerator.save(report, json_file)
        
        # Save text
        text_file = report_dir / f"{asset}_{timestamp}_validation.txt"
        TextReportGenerator.save(report, text_file)
        
        # Save HTML
        html_file = report_dir / f"{asset}_{timestamp}_validation.html"
        HTMLReportGenerator.save(report, html_file)
    
    def _log_result(
        self,
        state: ValidationState,
        asset: str,
        report: ValidationReport,
    ) -> None:
        """Log validation result."""
        if state == ValidationState.PASSED:
            logger.info(
                f"✓ Validation PASSED for {asset}: {report.summary.total_bars_validated} bars, "
                f"{report.summary.total_issues} issues"
            )
        elif state == ValidationState.FAILED_BLOCKED:
            blocking = [v.value for v in report.summary.blocking_validators]
            logger.error(
                f"✗ Validation FAILED for {asset}: {len(blocking)} blocking validators: {blocking}, "
                f"{report.summary.total_issues} issues"
            )
        elif state == ValidationState.FAILED_OVERRIDDEN:
            logger.warning(
                f"⚠ Validation OVERRIDDEN for {asset}: {report.summary.total_issues} issues, "
                f"can promote with explicit approval"
            )
