# Task 13: Validation Reporting and Automation - Implementation Summary

## Overview
Implemented a comprehensive validation reporting system that aggregates quality checks, generates actionable reports, and automates the validation pipeline with blocking controls. The system integrates schema, timestamp, missing bars, and anomaly detection validators, orchestrates them in the correct sequence, and provides detailed reporting and override management capabilities.

## Key Components Implemented

### 1. Report Generation (`src/data_io/reports.py`)

#### Data Structures
- **DetailedIssue**: Represents a single issue found during validation
  - Asset identifier, bar timestamp/index
  - Issue type, severity level, validator type
  - Measured value, expected value, threshold
  - Context and metadata for troubleshooting
  - Serialization to dict/JSON

- **ValidatorResult**: Result from a single validator
  - Validator type (schema, timestamp, missing_bars, anomaly)
  - Pass/fail status, total bars checked
  - List of DetailedIssue objects
  - Summary statistics
  - Methods to query issues by severity and detect blocking issues

- **ValidationSummary**: Aggregate statistics across validators
  - Overall pass/fail status
  - Total bars validated across all validators
  - Count of validators passed/failed
  - Issues grouped by severity and type
  - Pass rate percentage
  - Identifies blocking validators
  - Serialization to dict

- **ValidationReport**: Comprehensive validation report
  - Aggregates all ValidatorResult objects
  - Combines all DetailedIssue objects
  - Includes ValidationSummary
  - Methods to filter/sort issues
  - Serialization to dict and DataFrame

#### Report Generators

1. **JSONReportGenerator**
   - Generates JSON format for programmatic access
   - Includes all validation details in machine-readable format
   - Methods: `generate()`, `save()`

2. **TextReportGenerator**
   - Human-readable text format with clear sections
   - Shows summary, per-validator results, and detailed issues
   - Grouped by severity (critical, warning, info)
   - Methods: `generate()`, `save()`

3. **HTMLReportGenerator**
   - Professional HTML report with styled tables
   - Visual indicators for pass/fail status
   - Color-coded severity levels (red=critical, yellow=warning, blue=info)
   - Responsive design with proper styling
   - Methods: `generate()`, `save()`

#### Report Aggregator
- **ReportAggregator**: Combines results from multiple validators
  - `create_report()`: Aggregates results and generates summary
  - Calculates statistics (pass rate, issue counts by severity/type)
  - Identifies blocking validators

### 2. Override System (`src/data_io/override.py`)

#### Override Management
- **OverrideStatus**: Enum for override states
  - APPROVED: Override is active
  - PENDING: Awaiting approval
  - REJECTED: Override was rejected
  - REVOKED: Previously approved override was revoked

- **OverrideRecord**: Records a manual override
  - Asset and validator being overridden
  - User who created override and justification
  - Status and approval tracking (by, timestamp)
  - Data version and affected bars for traceability
  - Metadata for additional context
  - Methods: `approve()`, `reject()`, `revoke()`, `is_valid()`

- **ValidationMetadata**: Validation run metadata
  - Asset identifier and validation timestamp
  - Validation state (pending, running, passed, failed-blocked, failed-overridden)
  - Pass status and issue counts (total, critical)
  - List of associated override records
  - `can_promote()`: Determines if data can be promoted based on state and overrides

#### Override Manager
- **OverrideManager**: Manages overrides with persistence
  - Create overrides with justification
  - Approve/reject/revoke overrides
  - Register and retrieve overrides
  - Get active (approved) overrides only
  - Save/load metadata to JSON files
  - Automatic directory creation for metadata storage

### 3. Validation Pipeline (`src/data_io/pipeline.py`)

#### Pipeline Configuration
- **PipelineConfig**: Customizable pipeline settings
  - Enable/disable each validator (schema, timestamp, missing bars, gaps, spikes, volume)
  - Severity mappings for each validator type
  - Anomaly detection configs (gap, spike, volume)
  - Override settings (allow, require approval)
  - Output settings (save reports, report directory)

#### Pipeline States
- **ValidationState**: Enum for pipeline states
  - PENDING: Not started
  - RUNNING: Currently validating
  - PASSED: All validators passed
  - FAILED_BLOCKED: Critical failures prevent promotion
  - FAILED_OVERRIDDEN: Failed but overridden, requires approval for promotion

#### Pipeline Result
- **PipelineResult**: Outcome of validation
  - Asset identifier
  - Validation state
  - Pass/fail status
  - Full ValidationReport object
  - Can promote flag (considering overrides)
  - List of blocking validators
  - Override count

#### Orchestrator
- **ValidationOrchestrator**: Orchestrates complete validation pipeline
  - Sequential execution of validators
  - Blocking logic: schema and timestamp failures block further validation
  - Anomaly detection runs non-blocking
  - Automatic error handling and recovery
  - Report generation and persistence
  - Progress logging

  **Validation Order:**
  1. Schema validation (blocks if fails)
  2. Timestamp validation (blocks if fails)
  3. Anomaly detection (warnings, no blocking)
  4. Missing bars detection (warnings, no blocking)

### 4. CLI Interface (`scripts/run_validation.py`)

Comprehensive command-line interface for validation operations.

#### Commands

1. **validate**: Run validation on data
   ```bash
   python scripts/run_validation.py validate --asset MES --file data.parquet
   ```
   - Options:
     - `--asset`: Asset identifier (required)
     - `--file`: Path to parquet/CSV file (required)
     - `--report-dir`: Output directory for reports
     - `--skip-*`: Skip specific validators (schema, timestamp, missing-bars, gaps, spikes, volume)

2. **report**: Generate validation report
   ```bash
   python scripts/run_validation.py report --asset MES --format html
   ```
   - Options:
     - `--asset`: Asset identifier (required)
     - `--format`: Report format (json, text, html)
     - `--metadata-dir`: Metadata storage directory

3. **override**: Manage validation overrides
   - **create**: Create new override
     ```bash
     python scripts/run_validation.py override create --asset MES --validator schema \
         --user john --reason "Known issue"
     ```
   - **approve**: Approve pending override
     ```bash
     python scripts/run_validation.py override approve --asset MES --validator schema --user manager
     ```
   - **revoke**: Revoke approved override
     ```bash
     python scripts/run_validation.py override revoke --asset MES --validator schema
     ```

4. **metadata**: Query validation metadata
   ```bash
   python scripts/run_validation.py metadata --asset MES
   ```
   - Displays all metadata in JSON format

### 5. Comprehensive Test Suite (`tests/test_reporting.py`)

#### Test Coverage
1. **Report Data Structures** (20 tests)
   - DetailedIssue creation and serialization
   - ValidatorResult aggregation and queries
   - ValidationSummary statistics
   - ValidationReport filtering and conversion

2. **Report Generators** (6 tests)
   - JSON generation and persistence
   - Text report generation and formatting
   - HTML report generation with proper structure

3. **Aggregation** (2 tests)
   - Multi-validator aggregation
   - Summary statistics calculation

4. **Validation Pipeline** (4 tests)
   - Orchestrator initialization
   - Valid data validation
   - Invalid schema handling
   - State tracking

5. **Override System** (8 tests)
   - Override record creation and state transitions
   - Manager operations (create, approve, revoke)
   - Active override filtering
   - Metadata persistence

6. **Integration Tests** (2 tests)
   - Full pipeline with reports
   - Pipeline with overrides

**Total: ~40+ comprehensive test cases**

## Features Implemented

### ✅ Report Generation
- [x] Aggregate results from all validators
- [x] Generate summary reports with per-validator status
- [x] Create detailed issue logs with context
- [x] Support JSON format (programmatic)
- [x] Support text format (human-readable)
- [x] Support HTML format (professional reports)
- [x] Support DataFrame conversion (analysis)
- [x] Include statistics (bars, issues, pass rate, severity breakdown)

### ✅ Validation Automation
- [x] Orchestrate validators in correct sequence
- [x] Schema validation (blocking on failure)
- [x] Timestamp validation (blocking on failure)
- [x] Anomaly detection (non-blocking warnings)
- [x] Missing bars detection (non-blocking warnings)
- [x] Implement blocking logic based on severity
- [x] Error handling and recovery

### ✅ State Management
- [x] Track validation states (pending, running, passed, failed-blocked, failed-overridden)
- [x] Determine promotion eligibility
- [x] Generate validation metadata files
- [x] Persist metadata to JSON

### ✅ Override System
- [x] Manual override with justification
- [x] Approval workflow (create → approve → apply)
- [x] Audit trail with timestamps and users
- [x] Multiple override statuses (pending, approved, rejected, revoked)
- [x] Override persistence and loading
- [x] Active override filtering

### ✅ Reporting
- [x] Issue filtering by severity, type, date range
- [x] Issue grouping and aggregation
- [x] Context information (surrounding bars, statistics)
- [x] Machine-readable output (JSON, DataFrame)
- [x] Human-readable output (text, HTML)
- [x] Professional HTML with styling

### ✅ CLI Interface
- [x] Validation execution
- [x] Report generation
- [x] Override management
- [x] Metadata queries
- [x] Flexible option handling
- [x] Clear usage documentation

## Data Flow

```
Raw Data
    ↓
ValidationOrchestrator.validate()
    ↓
┌─────────────────────────────────────┐
│ Schema Validation                   │ → BLOCKED if fails
│ (DataValidator)                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Timestamp Validation                │ → BLOCKED if fails
│ (DataValidator)                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Anomaly Detection                   │ → WARNING (no block)
│ (GapDetector, SpikeDetector, etc)   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Missing Bars Detection              │ → WARNING (no block)
│ (MissingBarsValidator)              │
└─────────────────────────────────────┘
    ↓
ReportAggregator.create_report()
    ↓
├─→ JSONReportGenerator → report.json
├─→ TextReportGenerator → report.txt
├─→ HTMLReportGenerator → report.html
└─→ ValidationMetadata → metadata.json
    ↓
OverrideManager (if needed)
    ├─→ create_override()
    ├─→ approve_override()
    └─→ save_metadata()
    ↓
PipelineResult (can_promote decision)
    ↓
Promotion to Cleaned Layer (or hold for override)
```

## Integration Points

1. **With Data Validation** (`src/data_io/validation.py`)
   - Uses DataValidator for schema and timestamp checks
   - Converts ValidationResult to ValidatorResult

2. **With Anomaly Detection** (`src/data_io/anomaly_detectors.py`)
   - Uses GapDetector, SpikeDetectors, VolumeAnomalyDetector
   - Converts Alert objects to DetailedIssue

3. **With Alert System** (`src/data_io/alerts.py`)
   - Uses AlertAggregator for grouping related alerts
   - Leverages AlertSeverity and AnomalyType enums

4. **With Data Cleaning Pipeline** (`Task #11`)
   - Validation must pass before data promotion
   - Provides detailed failure reasons and blocking info
   - Enables informed override decisions

## File Structure

```
src/data_io/
├── reports.py           # Report generation (800+ lines)
├── pipeline.py          # Pipeline orchestration (450+ lines)
├── override.py          # Override system (300+ lines)
├── validation.py        # (existing) Data validation
├── anomaly_detectors.py # (existing) Anomaly detection
└── alerts.py            # (existing) Alert management

scripts/
└── run_validation.py    # CLI interface (400+ lines)

tests/
└── test_reporting.py    # Test suite (500+ lines)

TASK_13_VALIDATION_REPORTING_SUMMARY.md  # This document
```

## Usage Examples

### Basic Validation
```python
from src.data_io.pipeline import ValidationOrchestrator, PipelineConfig
import pandas as pd

df = pd.read_parquet("data.parquet")
config = PipelineConfig(
    enable_schema_validation=True,
    enable_timestamp_validation=True,
    enable_gap_detection=True,
)
orchestrator = ValidationOrchestrator(config=config)
result = orchestrator.validate(df, asset="MES")

if result.passed:
    print("✓ Validation passed - data ready for promotion")
else:
    print(f"✗ Validation failed - blocking validators: {result.blocking_validators}")
    print(f"Total issues: {len(result.report.detailed_issues)}")
```

### Generate Reports
```python
from src.data_io.reports import (
    JSONReportGenerator,
    TextReportGenerator,
    HTMLReportGenerator,
)
from pathlib import Path

report_dir = Path("./reports")
report_dir.mkdir(exist_ok=True)

JSONReportGenerator.save(result.report, report_dir / "report.json")
TextReportGenerator.save(result.report, report_dir / "report.txt")
HTMLReportGenerator.save(result.report, report_dir / "report.html")
```

### Override Failures
```python
from src.data_io.override import OverrideManager

manager = OverrideManager()
override = manager.create_override(
    asset="MES",
    validator_type="schema",
    overridden_by="john",
    justification="Known data quality issue - data is still usable",
)

# Later, manager approves it
manager.approve_override(override, approved_by="qa_manager")
manager.register_override("MES", override)

# Check if data can be promoted
metadata = manager.load_metadata("MES")
if metadata.can_promote():
    print("Data can be promoted with override")
```

## Success Criteria Met

- [x] Reports correctly aggregate all validation results
- [x] Critical failures block data promotion (automation works)
- [x] Manual override workflow functions correctly with audit trail
- [x] Reports are actionable (easy to identify and locate issues)
- [x] HTML reports are readable and well-formatted
- [x] JSON reports contain all necessary information for downstream tooling
- [x] Validation metadata enables reproducibility
- [x] Pipeline handles errors gracefully

## Design Principles

1. **Separation of Concerns**
   - Reporting independent from validation logic
   - Override system independent from pipeline
   - CLI independent from core functionality

2. **Extensibility**
   - Easy to add new validator types
   - Easy to add new report generators
   - Easy to customize severity mappings

3. **Auditability**
   - Complete audit trail of overrides (who, when, why)
   - Metadata tracking for reproducibility
   - State transitions logged

4. **Usability**
   - CLI for non-programmers
   - Programmatic API for integration
   - Multiple report formats for different audiences
   - Clear error messages and state indicators

## Future Enhancements

- Webhook notifications for critical failures
- Database persistence for historical tracking
- Advanced filtering/querying of historical validations
- Integration with monitoring/alerting systems
- Custom report templates
- Automatic retry logic for transient failures
- Performance analytics (validation runtime tracking)
