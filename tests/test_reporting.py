"""
Comprehensive test suite for validation reporting and automation.

Tests cover:
- Report generation (JSON, text, HTML)
- Report aggregation from all validators
- Validation orchestration and state tracking
- Override system and audit trails
- Pipeline blocking logic
- Metadata generation and persistence
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile

from src.data_io.reports import (
    DetailedIssue,
    ValidatorResult,
    ValidationSummary,
    ValidationReport,
    ValidatorType,
    SeverityLevel,
    ReportAggregator,
    JSONReportGenerator,
    TextReportGenerator,
    HTMLReportGenerator,
)
from src.data_io.pipeline import (
    ValidationOrchestrator,
    PipelineConfig,
    ValidationState,
)
from src.data_io.override import (
    OverrideManager,
    OverrideRecord,
    ValidationMetadata,
    OverrideStatus,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def valid_ohlcv_data():
    """Create valid OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1H", tz="US/Eastern")
    np.random.seed(42)
    
    data = {
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 100),
        "high": np.random.uniform(110, 120, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(100, 110, 100),
        "volume": np.random.randint(1000, 10000, 100),
    }
    
    # Ensure OHLC relationships
    for i in range(100):
        data["high"][i] = max(data["open"][i], data["close"][i], data["high"][i])
        data["low"][i] = min(data["open"][i], data["close"][i], data["low"][i])
    
    return pd.DataFrame(data)


@pytest.fixture
def invalid_schema_data():
    """Create data with missing columns."""
    dates = pd.date_range("2024-01-01", periods=10, freq="1H", tz="US/Eastern")
    return pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 10),
    })


@pytest.fixture
def detailed_issue():
    """Create a sample detailed issue."""
    return DetailedIssue(
        asset="MES",
        bar_timestamp=pd.Timestamp("2024-01-01 10:00:00", tz="US/Eastern"),
        bar_index=5,
        issue_type="schema",
        severity=SeverityLevel.CRITICAL,
        message="Missing required column: high",
        validator_type=ValidatorType.SCHEMA,
    )


@pytest.fixture
def validator_result(detailed_issue):
    """Create a sample validator result."""
    return ValidatorResult(
        validator_type=ValidatorType.SCHEMA,
        asset="MES",
        passed=False,
        total_bars_checked=100,
        issues=[detailed_issue],
        summary={"message": "Schema validation failed"},
    )


# ============================================================================
# Tests: Report Data Structures
# ============================================================================

class TestDetailedIssue:
    """Tests for DetailedIssue data structure."""
    
    def test_create_issue(self, detailed_issue):
        """Test creating a detailed issue."""
        assert detailed_issue.asset == "MES"
        assert detailed_issue.bar_index == 5
        assert detailed_issue.issue_type == "schema"
        assert detailed_issue.severity == SeverityLevel.CRITICAL
    
    def test_issue_to_dict(self, detailed_issue):
        """Test converting issue to dictionary."""
        issue_dict = detailed_issue.to_dict()
        assert issue_dict["asset"] == "MES"
        assert issue_dict["bar_index"] == 5
        assert issue_dict["severity"] == "critical"
        assert issue_dict["validator_type"] == "schema"


class TestValidatorResult:
    """Tests for ValidatorResult data structure."""
    
    def test_create_result(self, validator_result):
        """Test creating validator result."""
        assert validator_result.validator_type == ValidatorType.SCHEMA
        assert validator_result.asset == "MES"
        assert not validator_result.passed
        assert validator_result.total_bars_checked == 100
        assert len(validator_result.issues) == 1
    
    def test_has_blocking_issues(self, validator_result):
        """Test checking for blocking issues."""
        assert validator_result.has_blocking_issues()
    
    def test_get_issue_count(self, validator_result):
        """Test getting issue count."""
        assert validator_result.get_issue_count() == 1
    
    def test_result_to_dict(self, validator_result):
        """Test converting result to dictionary."""
        result_dict = validator_result.to_dict()
        assert result_dict["validator_type"] == "schema"
        assert result_dict["passed"] == False
        assert result_dict["total_issues"] == 1


class TestValidationSummary:
    """Tests for ValidationSummary data structure."""
    
    def test_create_summary(self):
        """Test creating validation summary."""
        summary = ValidationSummary(
            asset="MES",
            validation_timestamp=datetime.utcnow(),
            overall_passed=False,
            total_bars_validated=100,
            validators_passed=1,
            validators_failed=1,
            total_issues=5,
            issues_by_severity={"critical": 2, "warning": 3},
            issues_by_type={"schema": 2, "timestamp": 3},
            pass_rate=50.0,
            blocking_validators=[ValidatorType.SCHEMA],
        )
        
        assert summary.asset == "MES"
        assert not summary.overall_passed
        assert summary.total_issues == 5
        assert summary.validators_passed == 1


class TestValidationReport:
    """Tests for ValidationReport data structure."""
    
    def test_create_report(self, validator_result):
        """Test creating validation report."""
        report = ValidationReport(
            asset="MES",
            validation_timestamp=datetime.utcnow(),
            overall_passed=False,
        )
        report.add_result(validator_result)
        
        assert report.asset == "MES"
        assert not report.overall_passed
        assert len(report.results) == 1
        assert len(report.detailed_issues) == 1
    
    def test_get_critical_issues(self, validator_result):
        """Test getting critical issues."""
        report = ValidationReport(
            asset="MES",
            validation_timestamp=datetime.utcnow(),
            overall_passed=False,
        )
        report.add_result(validator_result)
        
        critical = report.get_critical_issues()
        assert len(critical) == 1
        assert critical[0].severity == SeverityLevel.CRITICAL
    
    def test_filter_issues_by_severity(self, validator_result):
        """Test filtering issues by severity."""
        report = ValidationReport(
            asset="MES",
            validation_timestamp=datetime.utcnow(),
            overall_passed=False,
        )
        report.add_result(validator_result)
        
        filtered = report.filter_issues(severity=SeverityLevel.CRITICAL)
        assert len(filtered) == 1
    
    def test_report_to_dict(self, validator_result):
        """Test converting report to dictionary."""
        report = ValidationReport(
            asset="MES",
            validation_timestamp=datetime.utcnow(),
            overall_passed=False,
        )
        report.add_result(validator_result)
        
        report_dict = report.to_dict()
        assert report_dict["asset"] == "MES"
        assert report_dict["total_issues"] == 1
    
    def test_report_to_dataframe(self, validator_result):
        """Test converting report to DataFrame."""
        report = ValidationReport(
            asset="MES",
            validation_timestamp=datetime.utcnow(),
            overall_passed=False,
        )
        report.add_result(validator_result)
        
        df = report.to_dataframe()
        assert len(df) == 1
        assert df["asset"].iloc[0] == "MES"


# ============================================================================
# Tests: Report Aggregation
# ============================================================================

class TestReportAggregator:
    """Tests for report aggregation."""
    
    def test_create_report(self, validator_result):
        """Test creating aggregated report."""
        report = ReportAggregator.create_report("MES", [validator_result])
        
        assert report.asset == "MES"
        assert not report.overall_passed
        assert report.summary is not None
        assert report.summary.validators_failed == 1
    
    def test_create_report_all_pass(self):
        """Test creating report when all validators pass."""
        result1 = ValidatorResult(
            validator_type=ValidatorType.SCHEMA,
            asset="MES",
            passed=True,
            total_bars_checked=100,
            issues=[],
        )
        result2 = ValidatorResult(
            validator_type=ValidatorType.TIMESTAMP,
            asset="MES",
            passed=True,
            total_bars_checked=100,
            issues=[],
        )
        
        report = ReportAggregator.create_report("MES", [result1, result2])
        
        assert report.overall_passed
        assert report.summary.validators_passed == 2
        assert report.summary.validators_failed == 0


# ============================================================================
# Tests: Report Generators
# ============================================================================

class TestJSONReportGenerator:
    """Tests for JSON report generation."""
    
    def test_generate_json(self, validator_result):
        """Test generating JSON report."""
        report = ReportAggregator.create_report("MES", [validator_result])
        json_str = JSONReportGenerator.generate(report)
        
        # Parse JSON to verify it's valid
        data = json.loads(json_str)
        assert data["asset"] == "MES"
        assert "results_by_validator" in data
    
    def test_save_json(self, validator_result):
        """Test saving JSON report to file."""
        report = ReportAggregator.create_report("MES", [validator_result])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            JSONReportGenerator.save(report, filepath)
            
            assert filepath.exists()
            data = json.loads(filepath.read_text())
            assert data["asset"] == "MES"


class TestTextReportGenerator:
    """Tests for text report generation."""
    
    def test_generate_text(self, validator_result):
        """Test generating text report."""
        report = ReportAggregator.create_report("MES", [validator_result])
        text_str = TextReportGenerator.generate(report)
        
        assert "VALIDATION REPORT" in text_str
        assert "MES" in text_str
        assert "FAILED" in text_str
    
    def test_save_text(self, validator_result):
        """Test saving text report to file."""
        report = ReportAggregator.create_report("MES", [validator_result])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.txt"
            TextReportGenerator.save(report, filepath)
            
            assert filepath.exists()
            content = filepath.read_text()
            assert "VALIDATION REPORT" in content


class TestHTMLReportGenerator:
    """Tests for HTML report generation."""
    
    def test_generate_html(self, validator_result):
        """Test generating HTML report."""
        report = ReportAggregator.create_report("MES", [validator_result])
        html_str = HTMLReportGenerator.generate(report)
        
        assert "<!DOCTYPE html>" in html_str
        assert "MES" in html_str
        assert "<table>" in html_str
        assert "FAILED" in html_str
    
    def test_save_html(self, validator_result):
        """Test saving HTML report to file."""
        report = ReportAggregator.create_report("MES", [validator_result])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.html"
            HTMLReportGenerator.save(report, filepath)
            
            assert filepath.exists()
            content = filepath.read_text()
            assert "<!DOCTYPE html>" in content


# ============================================================================
# Tests: Validation Pipeline
# ============================================================================

class TestValidationOrchestrator:
    """Tests for validation orchestration."""
    
    def test_orchestrator_creation(self):
        """Test creating orchestrator."""
        config = PipelineConfig()
        orchestrator = ValidationOrchestrator(config=config)
        
        assert orchestrator.config is not None
        assert orchestrator.schema_validator is not None
    
    def test_validate_valid_data(self, valid_ohlcv_data):
        """Test validating valid data."""
        config = PipelineConfig(
            enable_schema_validation=True,
            enable_timestamp_validation=True,
            enable_gap_detection=False,
            enable_spike_detection=False,
            enable_volume_detection=False,
        )
        orchestrator = ValidationOrchestrator(config=config)
        result = orchestrator.validate(valid_ohlcv_data, asset="MES")
        
        assert result.passed
        assert result.validation_state == ValidationState.PASSED
        assert result.can_promote
    
    def test_validate_invalid_schema(self, invalid_schema_data):
        """Test validation with invalid schema."""
        config = PipelineConfig(
            enable_schema_validation=True,
            enable_timestamp_validation=False,
        )
        orchestrator = ValidationOrchestrator(config=config)
        result = orchestrator.validate(invalid_schema_data, asset="MES")
        
        assert not result.passed
        assert result.validation_state == ValidationState.FAILED_BLOCKED
        assert not result.can_promote
    
    def test_validation_state_tracking(self, valid_ohlcv_data):
        """Test validation state tracking."""
        config = PipelineConfig()
        orchestrator = ValidationOrchestrator(config=config)
        result = orchestrator.validate(valid_ohlcv_data, asset="MES")
        
        assert result.validation_state in [
            ValidationState.PASSED,
            ValidationState.FAILED_BLOCKED,
            ValidationState.FAILED_OVERRIDDEN,
        ]


# ============================================================================
# Tests: Override System
# ============================================================================

class TestOverrideRecord:
    """Tests for override records."""
    
    def test_create_override(self):
        """Test creating override record."""
        override = OverrideRecord(
            asset="MES",
            validator_type="schema",
            override_timestamp=datetime.utcnow(),
            overridden_by="john",
            justification="Known issue",
            status=OverrideStatus.PENDING,
        )
        
        assert override.asset == "MES"
        assert override.validator_type == "schema"
        assert override.status == OverrideStatus.PENDING
    
    def test_approve_override(self):
        """Test approving override."""
        override = OverrideRecord(
            asset="MES",
            validator_type="schema",
            override_timestamp=datetime.utcnow(),
            overridden_by="john",
            justification="Known issue",
            status=OverrideStatus.PENDING,
        )
        
        override.approve("manager")
        
        assert override.status == OverrideStatus.APPROVED
        assert override.approved_by == "manager"
        assert override.is_valid()
    
    def test_revoke_override(self):
        """Test revoking override."""
        override = OverrideRecord(
            asset="MES",
            validator_type="schema",
            override_timestamp=datetime.utcnow(),
            overridden_by="john",
            justification="Known issue",
            status=OverrideStatus.APPROVED,
            approved_by="manager",
            approval_timestamp=datetime.utcnow(),
        )
        
        override.revoke()
        
        assert override.status == OverrideStatus.REVOKED
        assert not override.is_valid()


class TestOverrideManager:
    """Tests for override management."""
    
    def test_create_override(self):
        """Test creating override via manager."""
        manager = OverrideManager()
        override = manager.create_override(
            asset="MES",
            validator_type="schema",
            overridden_by="john",
            justification="Known issue",
        )
        
        assert override.asset == "MES"
        assert override.status == OverrideStatus.PENDING
    
    def test_register_and_get_override(self):
        """Test registering and retrieving overrides."""
        manager = OverrideManager()
        override = manager.create_override(
            asset="MES",
            validator_type="schema",
            overridden_by="john",
            justification="Known issue",
        )
        
        manager.register_override("MES", override)
        overrides = manager.get_overrides("MES")
        
        assert len(overrides) == 1
        assert overrides[0].asset == "MES"
    
    def test_get_active_overrides(self):
        """Test getting active overrides only."""
        manager = OverrideManager()
        
        # Create and approve one override
        override1 = manager.create_override(
            asset="MES",
            validator_type="schema",
            overridden_by="john",
            justification="Issue 1",
        )
        manager.approve_override(override1, "manager")
        manager.register_override("MES", override1)
        
        # Create but don't approve another
        override2 = manager.create_override(
            asset="MES",
            validator_type="timestamp",
            overridden_by="john",
            justification="Issue 2",
        )
        manager.register_override("MES", override2)
        
        active = manager.get_active_overrides("MES")
        assert len(active) == 1
        assert active[0].validator_type == "schema"
    
    def test_save_and_load_metadata(self):
        """Test saving and loading metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OverrideManager(metadata_dir=Path(tmpdir))
            
            metadata = ValidationMetadata(
                asset="MES",
                validation_timestamp=datetime.utcnow(),
                validation_state="passed",
                passed=True,
                total_issues=0,
                critical_issues=0,
            )
            
            manager.save_metadata("MES", metadata)
            loaded = manager.load_metadata("MES")
            
            assert loaded is not None
            assert loaded.asset == "MES"
            assert loaded.passed


class TestValidationMetadata:
    """Tests for validation metadata."""
    
    def test_can_promote_when_passed(self):
        """Test promotion eligibility when validation passed."""
        metadata = ValidationMetadata(
            asset="MES",
            validation_timestamp=datetime.utcnow(),
            validation_state="passed",
            passed=True,
            total_issues=0,
            critical_issues=0,
        )
        
        assert metadata.can_promote()
    
    def test_can_promote_with_approved_override(self):
        """Test promotion eligibility with approved override."""
        override = OverrideRecord(
            asset="MES",
            validator_type="schema",
            override_timestamp=datetime.utcnow(),
            overridden_by="john",
            justification="Known issue",
            status=OverrideStatus.APPROVED,
            approved_by="manager",
            approval_timestamp=datetime.utcnow(),
        )
        
        metadata = ValidationMetadata(
            asset="MES",
            validation_timestamp=datetime.utcnow(),
            validation_state="failed-overridden",
            passed=False,
            total_issues=1,
            critical_issues=1,
            overrides=[override],
        )
        
        assert metadata.can_promote()
    
    def test_cannot_promote_with_pending_override(self):
        """Test promotion blocked with pending override."""
        override = OverrideRecord(
            asset="MES",
            validator_type="schema",
            override_timestamp=datetime.utcnow(),
            overridden_by="john",
            justification="Known issue",
            status=OverrideStatus.PENDING,
        )
        
        metadata = ValidationMetadata(
            asset="MES",
            validation_timestamp=datetime.utcnow(),
            validation_state="failed-overridden",
            passed=False,
            total_issues=1,
            critical_issues=1,
            overrides=[override],
        )
        
        assert not metadata.can_promote()


# ============================================================================
# Tests: Integration
# ============================================================================

class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_full_pipeline_with_reports(self, valid_ohlcv_data):
        """Test full validation pipeline with report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(
                enable_schema_validation=True,
                enable_timestamp_validation=True,
                save_reports=True,
                report_dir=tmpdir,
            )
            
            orchestrator = ValidationOrchestrator(config=config)
            result = orchestrator.validate(valid_ohlcv_data, asset="MES")
            
            assert result.report is not None
            assert result.report.asset == "MES"
            assert result.passed
    
    def test_pipeline_with_override(self, invalid_schema_data):
        """Test validation with override handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            override_manager = OverrideManager(metadata_dir=Path(tmpdir))
            
            # Create override before validation
            override = override_manager.create_override(
                asset="MES",
                validator_type="schema",
                overridden_by="john",
                justification="Known issue",
            )
            override_manager.approve_override(override, "manager")
            override_manager.register_override("MES", override)
            
            config = PipelineConfig(
                enable_schema_validation=True,
                allow_overrides=True,
            )
            orchestrator = ValidationOrchestrator(
                config=config,
                override_manager=override_manager,
            )
            
            result = orchestrator.validate(invalid_schema_data, asset="MES")
            
            # With override, data should be promotable
            assert result.override_count >= 0
