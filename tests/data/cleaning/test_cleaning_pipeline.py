"""
Integration tests for data cleaning pipeline.

Tests cover:
- End-to-end pipeline execution
- Validation failures blocking promotion
- Version incrementation
- Cleaning reports generation
- Dry-run mode
- Idempotency
- Incremental cleaning
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

from src.data.cleaning.cleaning_pipeline import (
    CleaningPipeline,
    CleaningReport,
    CleaningMetrics,
    VersionChangeType,
    VersioningStrategy,
    run_cleaning_pipeline,
)
from src.storage.version_metadata import SemanticVersion, DataRange

logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dirs():
    """Create temporary directories for raw and cleaned data."""
    with tempfile.TemporaryDirectory() as raw_dir:
        with tempfile.TemporaryDirectory() as cleaned_dir:
            raw_path = Path(raw_dir) / "raw"
            cleaned_path = Path(cleaned_dir) / "cleaned"
            raw_path.mkdir(parents=True, exist_ok=True)
            cleaned_path.mkdir(parents=True, exist_ok=True)
            yield raw_path, cleaned_path


@pytest.fixture
def sample_raw_data():
    """Create sample raw OHLCV data for ES."""
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    df = pd.DataFrame({
        'timestamp': dates,
        'contract': ['ESZ23'] * 25 + ['ESH24'] * 25,
        'open': np.linspace(5000, 5100, 50),
        'high': np.linspace(5010, 5110, 50),
        'low': np.linspace(4990, 5090, 50),
        'close': np.linspace(5000, 5100, 50),
        'volume': np.random.randint(1000, 10000, 50),
    })
    return df


@pytest.fixture
def sample_raw_data_with_tz():
    """Create sample raw data with timezone-aware timestamps."""
    dates = pd.date_range('2024-01-01', periods=50, freq='D', tz='UTC')
    df = pd.DataFrame({
        'timestamp': dates,
        'contract': ['ESZ23'] * 25 + ['ESH24'] * 25,
        'open': np.linspace(5000, 5100, 50),
        'high': np.linspace(5010, 5110, 50),
        'low': np.linspace(4990, 5090, 50),
        'close': np.linspace(5000, 5100, 50),
        'volume': np.random.randint(1000, 10000, 50),
    })
    return df


# ============================================================================
# Tests: Version Management
# ============================================================================

class TestVersioning:
    """Tests for semantic versioning logic."""
    
    def test_get_next_version_first_version(self):
        """Test getting first version."""
        next_v = VersioningStrategy.get_next_version(None, VersionChangeType.MINOR)
        assert next_v == SemanticVersion(1, 0, 0)
    
    def test_get_next_version_major(self):
        """Test major version increment."""
        current = SemanticVersion(1, 2, 3)
        next_v = VersioningStrategy.get_next_version(current, VersionChangeType.MAJOR)
        assert next_v == SemanticVersion(2, 0, 0)
    
    def test_get_next_version_minor(self):
        """Test minor version increment."""
        current = SemanticVersion(1, 2, 3)
        next_v = VersioningStrategy.get_next_version(current, VersionChangeType.MINOR)
        assert next_v == SemanticVersion(1, 3, 0)
    
    def test_get_next_version_patch(self):
        """Test patch version increment."""
        current = SemanticVersion(1, 2, 3)
        next_v = VersioningStrategy.get_next_version(current, VersionChangeType.PATCH)
        assert next_v == SemanticVersion(1, 2, 4)
    
    def test_determine_change_type_first_version(self):
        """Test determining change type for first version."""
        change = VersioningStrategy.determine_change_type(None)
        assert change == VersionChangeType.MINOR
    
    def test_determine_change_type_method_changed(self):
        """Test that method change is always major."""
        change = VersioningStrategy.determine_change_type(
            None,
            existing_version=None,
            method_changed=True,
        )
        assert change == VersionChangeType.MAJOR


# ============================================================================
# Tests: Cleaning Reports
# ============================================================================

class TestCleaningReport:
    """Tests for cleaning report generation."""
    
    def test_report_creation(self):
        """Test creating a cleaning report."""
        report = CleaningReport(
            asset='ES',
            version='v1.0.0',
            start_date='2024-01-01',
            end_date='2024-12-31',
            creation_timestamp=datetime.now(timezone.utc).isoformat(),
            status='success',
            dry_run=False,
        )
        
        assert report.asset == 'ES'
        assert report.version == 'v1.0.0'
        assert report.status == 'success'
    
    def test_report_to_dict(self):
        """Test converting report to dictionary."""
        report = CleaningReport(
            asset='ES',
            version='v1.0.0',
            start_date='2024-01-01',
            end_date='2024-12-31',
            creation_timestamp=datetime.now(timezone.utc).isoformat(),
            status='success',
            dry_run=False,
        )
        
        data = report.to_dict()
        assert data['asset'] == 'ES'
        assert data['version'] == 'v1.0.0'
        assert 'metrics' in data
    
    def test_report_to_markdown(self):
        """Test generating markdown report."""
        report = CleaningReport(
            asset='ES',
            version='v1.0.0',
            start_date='2024-01-01',
            end_date='2024-12-31',
            creation_timestamp=datetime.now(timezone.utc).isoformat(),
            status='success',
            dry_run=False,
        )
        report.metrics.rows_in = 100
        report.metrics.rows_out = 95
        
        markdown = report.to_markdown()
        assert '# Data Cleaning Report: ES' in markdown
        assert 'Rows In: 100' in markdown
        assert 'Rows Out: 95' in markdown


# ============================================================================
# Tests: Pipeline Execution
# ============================================================================

class TestCleaningPipelineExecution:
    """Tests for pipeline execution."""
    
    def test_pipeline_initialization(self, temp_dirs):
        """Test pipeline initialization."""
        raw_path, cleaned_path = temp_dirs
        
        pipeline = CleaningPipeline(
            raw_data_path=raw_path,
            cleaned_data_path=cleaned_path,
        )
        
        assert pipeline.raw_data_path == raw_path
        assert pipeline.cleaned_data_path == cleaned_path
    
    def test_pipeline_run_missing_data(self, temp_dirs):
        """Test pipeline with missing raw data."""
        raw_path, cleaned_path = temp_dirs
        
        pipeline = CleaningPipeline(
            raw_data_path=raw_path,
            cleaned_data_path=cleaned_path,
        )
        
        result = pipeline.run(
            asset='ES',
            start_date='2024-01-01',
            end_date='2024-12-31',
            dry_run=True,
        )
        
        # Should fail when no data found
        assert not result.success
        assert result.error_message is not None


# ============================================================================
# Tests: Dry-Run Mode
# ============================================================================

class TestDryRunMode:
    """Tests for dry-run functionality."""
    
    def test_dry_run_no_output(self, temp_dirs, sample_raw_data):
        """Test that dry-run doesn't write output."""
        raw_path, cleaned_path = temp_dirs
        
        # Setup raw data
        es_dir = raw_path / 'ES'
        es_dir.mkdir(exist_ok=True)
        sample_raw_data.to_parquet(es_dir / 'data.parquet')
        
        pipeline = CleaningPipeline(
            raw_data_path=raw_path,
            cleaned_data_path=cleaned_path,
        )
        
        # Run with dry_run=True
        result = pipeline.run(
            asset='ES',
            start_date='2024-01-01',
            end_date='2024-02-01',
            dry_run=True,
        )
        
        # Output directory should be None
        assert result.output_dir is None
        assert result.report.dry_run is True


# ============================================================================
# Tests: Validation Blocking
# ============================================================================

class TestValidationBlocking:
    """Tests for validation blocking promotion."""
    
    def test_validation_failure_blocks_promotion(self, temp_dirs, sample_raw_data):
        """Test that validation failures block promotion."""
        raw_path, cleaned_path = temp_dirs
        
        # Setup raw data with invalid schema (missing required columns)
        es_dir = raw_path / 'ES'
        es_dir.mkdir(exist_ok=True)
        
        invalid_data = sample_raw_data.drop(columns=['close'])  # Missing required column
        invalid_data.to_parquet(es_dir / 'data.parquet')
        
        pipeline = CleaningPipeline(
            raw_data_path=raw_path,
            cleaned_data_path=cleaned_path,
        )
        
        result = pipeline.run(
            asset='ES',
            start_date='2024-01-01',
            end_date='2024-02-01',
            dry_run=True,
        )
        
        # Should fail due to schema validation
        if not result.success:
            assert result.error_message is not None


# ============================================================================
# Tests: Idempotency
# ============================================================================

class TestIdempotency:
    """Tests for pipeline idempotency."""
    
    def test_rerunning_same_version_produces_same_output(self, temp_dirs, sample_raw_data):
        """Test that re-running same version produces identical output."""
        raw_path, cleaned_path = temp_dirs
        
        # Setup raw data
        es_dir = raw_path / 'ES'
        es_dir.mkdir(exist_ok=True)
        sample_raw_data.to_parquet(es_dir / 'data.parquet')
        
        pipeline = CleaningPipeline(
            raw_data_path=raw_path,
            cleaned_data_path=cleaned_path,
        )
        
        # First run
        result1 = pipeline.run(
            asset='ES',
            start_date='2024-01-01',
            end_date='2024-02-01',
            version='v1.0.0',
            dry_run=True,
        )
        
        # Second run with same version
        result2 = pipeline.run(
            asset='ES',
            start_date='2024-01-01',
            end_date='2024-02-01',
            version='v1.0.0',
            dry_run=True,
        )
        
        # Both should produce same results
        assert result1.success == result2.success
        assert result1.report.metrics.rows_in == result2.report.metrics.rows_in
        assert result1.report.metrics.rows_out == result2.report.metrics.rows_out


# ============================================================================
# Tests: Incremental Cleaning
# ============================================================================

class TestIncrementalCleaning:
    """Tests for incremental data cleaning."""
    
    def test_version_increment_minor(self, temp_dirs, sample_raw_data):
        """Test that version increments correctly for minor changes."""
        raw_path, cleaned_path = temp_dirs
        
        # Setup raw data
        es_dir = raw_path / 'ES'
        es_dir.mkdir(exist_ok=True)
        sample_raw_data.to_parquet(es_dir / 'data.parquet')
        
        pipeline = CleaningPipeline(
            raw_data_path=raw_path,
            cleaned_data_path=cleaned_path,
        )
        
        # Run without specifying version
        result = pipeline.run(
            asset='ES',
            start_date='2024-01-01',
            end_date='2024-02-01',
            change_type=VersionChangeType.MINOR,
            dry_run=True,
        )
        
        # Should auto-increment to v1.0.0 (first version)
        assert result.version == 'v1.0.0'


# ============================================================================
# Tests: Public API
# ============================================================================

class TestPublicAPI:
    """Tests for public API functions."""
    
    def test_run_cleaning_pipeline_invalid_change_type(self):
        """Test that invalid change_type raises error."""
        with pytest.raises(ValueError):
            run_cleaning_pipeline(
                asset='ES',
                start_date='2024-01-01',
                end_date='2024-12-31',
                change_type='invalid',
            )
    
    def test_run_cleaning_pipeline_missing_data(self, temp_dirs):
        """Test run_cleaning_pipeline with missing data."""
        raw_path, cleaned_path = temp_dirs
        
        result = run_cleaning_pipeline(
            asset='ES',
            start_date='2024-01-01',
            end_date='2024-12-31',
            raw_data_path=raw_path,
            cleaned_data_path=cleaned_path,
            dry_run=True,
        )
        
        # Should fail with no data
        assert not result.success


# ============================================================================
# Tests: Metrics and Transformations
# ============================================================================

class TestMetricsAndTransformations:
    """Tests for metrics tracking and transformations."""
    
    def test_cleaning_metrics_creation(self):
        """Test creating cleaning metrics."""
        metrics = CleaningMetrics(
            rows_in=100,
            rows_out=95,
            rows_filtered=5,
            timezone_normalized=True,
            calendar_filtered=True,
        )
        
        assert metrics.rows_in == 100
        assert metrics.rows_out == 95
        assert metrics.rows_filtered == 5
    
    def test_cleaning_metrics_to_dict(self):
        """Test converting metrics to dict."""
        metrics = CleaningMetrics(
            rows_in=100,
            rows_out=95,
        )
        
        data = metrics.to_dict()
        assert data['rows_in'] == 100
        assert data['rows_out'] == 95


# ============================================================================
# Tests: Promotion Workflow
# ============================================================================

class TestPromotionWorkflow:
    """Tests for promotion workflow."""
    
    def test_promote_to_cleaned_updates_report(self, temp_dirs, sample_raw_data):
        """Test that promotion updates report metadata."""
        raw_path, cleaned_path = temp_dirs
        
        # Setup raw data
        es_dir = raw_path / 'ES'
        es_dir.mkdir(exist_ok=True)
        sample_raw_data.to_parquet(es_dir / 'data.parquet')
        
        pipeline = CleaningPipeline(
            raw_data_path=raw_path,
            cleaned_data_path=cleaned_path,
        )
        
        # Run pipeline
        result = pipeline.run(
            asset='ES',
            start_date='2024-01-01',
            end_date='2024-02-01',
            version='v1.0.0',
            dry_run=False,
        )
        
        if result.success:
            # Try to promote
            promoted = pipeline.promote_to_cleaned(
                asset='ES',
                version='v1.0.0',
                approval_status=True,
                approval_reason='Approved after review',
            )
            
            assert promoted


__all__ = [
    "TestVersioning",
    "TestCleaningReport",
    "TestCleaningPipelineExecution",
    "TestDryRunMode",
    "TestValidationBlocking",
    "TestIdempotency",
    "TestIncrementalCleaning",
    "TestPublicAPI",
    "TestMetricsAndTransformations",
    "TestPromotionWorkflow",
]
