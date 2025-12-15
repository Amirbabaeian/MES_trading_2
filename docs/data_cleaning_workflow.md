# Data Cleaning Workflow

## Overview

The data cleaning pipeline orchestrates all cleaning and validation steps to transform raw market data into high-quality, production-ready datasets. The pipeline maintains semantic versioning for reproducibility and includes quality gates to prevent promotion of invalid data.

**Key Features:**
- Automated orchestration of all cleaning steps
- Semantic versioning with automatic version increment
- Quality gates with validation blocking
- Comprehensive audit trail and reporting
- Dry-run mode for preview before commit
- Idempotent operations (re-runnable without side effects)
- Manual approval workflow for major versions
- Incremental cleaning (process only new data)

## Pipeline Architecture

### Processing Steps

The cleaning pipeline executes the following steps in sequence:

```
Raw Data
   ↓
1. Load Raw Data (from /raw/)
   ↓
2. Normalize Timezone (to US/Eastern)
   ↓
3. Filter Trading Hours (calendar-aware)
   ↓
4. Detect & Adjust Contract Rolls
   ↓
5. Validate Data Quality
   ↓
6. Generate Cleaning Report
   ↓
7. Promote to Cleaned Layer (/cleaned/vX/)
```

Each step is:
- **Idempotent**: Re-running produces identical output
- **Logged**: All transformations documented in report
- **Blockable**: Failures prevent promotion to next step
- **Auditable**: Detailed metrics and metadata captured

### Directory Structure

```
data/
├── raw/
│   ├── ES/
│   │   └── *.parquet
│   ├── MES/
│   │   └── *.parquet
│   └── VIX/
│       └── *.parquet
│
├── cleaned/
│   ├── v1.0.0/
│   │   ├── ES/
│   │   │   ├── ES_cleaned.parquet
│   │   │   ├── cleaning_report.json
│   │   │   └── cleaning_report.md
│   │   ├── MES/
│   │   └── VIX/
│   │
│   ├── v1.1.0/
│   │   └── ...
│   │
│   └── v2.0.0/
│       └── ...
```

## Semantic Versioning

The pipeline uses semantic versioning (major.minor.patch) to track data versions:

```
v{MAJOR}.{MINOR}.{PATCH}
```

### Version Increment Rules

| Event | Version Change | Example |
|-------|---|---------|
| **Major** | Breaking changes (method change) | v1.0.0 → v2.0.0 |
| **Minor** | New data added (extend range) | v1.0.0 → v1.1.0 |
| **Patch** | Data fixes (same range) | v1.0.0 → v1.0.1 |

### Change Types

**Major Version (Breaking Changes)**
- Adjustment method changed (backward_ratio ↔ panama_canal)
- Schema changes
- Historical data recalculated with different methodology
- ⚠️ **Requires Manual Approval** before promotion

**Minor Version (New Data)**
- New date range added (extend existing data)
- New asset or contract added
- Date ranges non-overlapping with previous version
- ✓ Auto-approved (can be overridden)

**Patch Version (Fixes)**
- Correcting data in existing date range
- Reprocessing overlapping period
- Bug fixes affecting same data range
- ✓ Auto-approved

## Running the Pipeline

### Command Line Interface

Basic usage:

```bash
python scripts/run_cleaning.py \
  --asset ES \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

With optional parameters:

```bash
python scripts/run_cleaning.py \
  --asset ES \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --version v1.2.0 \
  --change-type minor \
  --adjustment-method backward_ratio \
  --approve \
  --verbose
```

**Options:**

- `--asset {ES,MES,VIX}` - Asset symbol (required)
- `--start-date YYYY-MM-DD` - Start date (required)
- `--end-date YYYY-MM-DD` - End date (required)
- `--version VERSION` - Target version (auto-increment if not specified)
- `--change-type {major,minor,patch}` - Version increment type (default: minor)
- `--adjustment-method {backward_ratio,panama_canal}` - Contract roll method (default: backward_ratio)
- `--approve` - Approve version for promotion
- `--dry-run` - Preview changes without writing
- `--raw-data-path PATH` - Path to raw data (default: data/raw)
- `--cleaned-data-path PATH` - Path to cleaned data (default: data/cleaned)
- `-v, --verbose` - Verbose logging

### Python API

```python
from src.data.cleaning.cleaning_pipeline import run_cleaning_pipeline

# Run cleaning pipeline
result = run_cleaning_pipeline(
    asset='ES',
    start_date='2024-01-01',
    end_date='2024-12-31',
    version='v1.1.0',
    change_type='minor',
    approval_status=False,
    adjustment_method='backward_ratio',
    dry_run=False,
)

# Check results
if result.success:
    print(f"Version {result.version} created successfully")
    print(f"Output: {result.output_dir}")
    print(result.report.to_markdown())
else:
    print(f"Pipeline failed: {result.error_message}")
    for issue in result.report.blocking_issues:
        print(f"  - {issue}")
```

## Approval Workflow

### Standard Approval Process

```
Pipeline Run
    ↓
Validation Checks
    ↓
├─ PASS: Auto-approve minor/patch
│        Require approval for major
│
└─ FAIL: Block promotion
         Document blocking issues
         Return error
```

### Manual Approval for Major Versions

Major versions (breaking changes) require manual approval:

1. **Review Pipeline**: Check the cleaning report
   ```bash
   # View generated report
   cat data/cleaned/v2.0.0/ES/cleaning_report.md
   ```

2. **Approve or Reject**:
   ```python
   from src.data.cleaning.cleaning_pipeline import CleaningPipeline
   
   pipeline = CleaningPipeline(
       raw_data_path='data/raw',
       cleaned_data_path='data/cleaned',
   )
   
   # Approve
   pipeline.promote_to_cleaned(
       asset='ES',
       version='v2.0.0',
       approval_status=True,
       approval_reason='Methodology reviewed and approved by DataOps',
   )
   ```

3. **Promotion**: Upon approval, data is moved to final cleaned layer

## Cleaning Reports

Each cleaned version generates a comprehensive report documenting all operations.

### Report Contents

#### JSON Format (`cleaning_report.json`)
Structured format for programmatic access:
- Asset, version, date range
- Processing timestamps
- Metrics (rows in/out, filtered, etc.)
- Transformations applied
- Validation results
- Blocking issues
- Approval status

#### Markdown Format (`cleaning_report.md`)
Human-readable format for review:
- Executive summary
- Data metrics
- Validation results
- Transformation details
- Blocking issues (if any)
- Approval status

### Example Report

```markdown
# Data Cleaning Report: ES

**Version**: v1.1.0
**Status**: success
**Created**: 2024-01-15T10:30:00Z
**Dry Run**: False

## Data Range
- Start Date: 2024-01-01
- End Date: 2024-12-31

## Metrics
- Rows In: 250,000
- Rows Out: 248,500
- Rows Filtered: 1,500
- Timezone Normalized: True
- Calendar Filtered: True
- Contract Rolls Detected: 4
- Contract Rolls Adjusted: True

## Validation Results
- Passed: True
- Errors: 0
- Warnings: 3

## Transformations Applied

### Timezone Normalization
- Timestamp: 2024-01-15T10:30:00Z
- Description: Normalized timestamps to US/Eastern timezone
- Rows Affected: 250,000

### Trading Calendar Filter
- Timestamp: 2024-01-15T10:30:00Z
- Description: Filtered to trading hours only
- Rows Affected: 1,500

### Contract Roll Adjustment
- Timestamp: 2024-01-15T10:30:00Z
- Description: Detected and adjusted 4 contract rolls
- Rows Affected: 248,500

## Approval
- Approved: No
```

## Quality Gates

The pipeline enforces quality gates before promotion:

### Blocking Issues (CRITICAL)
- Missing required columns
- Invalid timestamp format/order
- Schema mismatches
- Data type errors
- **Result**: Promotion blocked, no data written

### Warnings (NON-BLOCKING)
- Gaps in timestamps
- Price anomalies (spikes)
- Volume anomalies
- **Result**: Logged in report, promotion allowed

### Validation Configuration

```python
from src.data_io.pipeline import PipelineConfig

config = PipelineConfig(
    enable_schema_validation=True,           # Required
    enable_timestamp_validation=True,        # Required
    enable_missing_bars_validation=False,    # Optional
    enable_gap_detection=True,               # Optional
    enable_spike_detection=True,             # Optional
    enable_volume_detection=False,           # Optional
)

# Use with pipeline
pipeline = CleaningPipeline(
    raw_data_path='data/raw',
    cleaned_data_path='data/cleaned',
    validation_config=config,
)
```

## Dry-Run Mode

Preview changes without writing to disk:

```bash
python scripts/run_cleaning.py \
  --asset ES \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --dry-run
```

Output:
```
CLEANING PIPELINE RESULT
================================================================================
Asset:     ES
Version:   v1.1.0
Success:   True
Status:    success
Dry Run:   True

Metrics:
  Rows In:                 250,000
  Rows Out:                248,500
  Rows Filtered:           1,500
  Timezone Normalized:     True
  Calendar Filtered:       True
  Contract Rolls Detected: 4
  Validation Passed:       True

(no output files written)
```

## Idempotency

The pipeline is designed to be idempotent - re-running with identical inputs produces identical outputs:

```python
# First run
result1 = run_cleaning_pipeline(
    asset='ES',
    start_date='2024-01-01',
    end_date='2024-12-31',
    version='v1.0.0',
)

# Second run (same inputs)
result2 = run_cleaning_pipeline(
    asset='ES',
    start_date='2024-01-01',
    end_date='2024-12-31',
    version='v1.0.0',
)

# Results are identical
assert result1.report.metrics == result2.report.metrics
```

This ensures:
- ✓ Consistency across multiple runs
- ✓ Reproducibility for research
- ✓ Safe re-execution for error recovery
- ✓ No cumulative side effects

## Incremental Cleaning

Process only new data since the last cleaned version:

```python
result = run_cleaning_pipeline(
    asset='ES',
    start_date='2024-02-01',  # New data after v1.0.0
    end_date='2024-03-31',
    change_type='minor',      # Extends existing data
    # version auto-increments to v1.1.0
)
```

The system:
1. Detects existing version v1.0.0
2. Determines this is new data (extends range)
3. Auto-increments to v1.1.0 (minor version)
4. Processes only new period
5. Saves to `/cleaned/v1.1.0/`

## Error Handling

### Common Errors

**Error: "No raw data found"**
- Cause: Raw data files don't exist for date range
- Solution: Check raw data directory and date ranges
  ```bash
  ls data/raw/ES/
  ```

**Error: "Schema validation failed"**
- Cause: Missing or invalid columns
- Solution: Check that raw data has required columns: timestamp, open, high, low, close, volume

**Error: "Validation failed - blocking issues detected"**
- Cause: Data quality issues prevent promotion
- Solution: Review blocking issues in report, fix raw data, re-run

**Error: "Version already exists"**
- Cause: Target version is already created
- Solution: Specify different version or delete existing version

### Recovery

For failed runs:

1. **Diagnose**: Check error message and cleaning report
2. **Fix Source**: Correct issues in raw data if needed
3. **Re-run**: Same version produces identical output (idempotent)
4. **Fallback**: Use previous version if needed
   ```bash
   # List available versions
   ls -la data/cleaned/
   
   # Use specific version in downstream processing
   python my_analysis.py --version v1.0.0
   ```

## Best Practices

### 1. Start with Dry-Run
Always preview changes first:
```bash
python scripts/run_cleaning.py \
  --asset ES \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --dry-run
```

### 2. Review Reports
Check the cleaning report before approval:
```bash
cat data/cleaned/v1.1.0/ES/cleaning_report.md
```

### 3. Incremental Processing
Process new data incrementally:
```bash
# Process Jan-Mar 2024
python scripts/run_cleaning.py \
  --asset ES \
  --start-date 2024-01-01 \
  --end-date 2024-03-31

# Later, add Apr-Jun 2024
python scripts/run_cleaning.py \
  --asset ES \
  --start-date 2024-04-01 \
  --end-date 2024-06-30
  # Auto-increments to v1.1.0
```

### 4. Approve Major Versions Carefully
Major versions indicate breaking changes - require review:
```python
# Change adjustment method: requires major version
result = run_cleaning_pipeline(
    asset='ES',
    start_date='2024-01-01',
    end_date='2024-12-31',
    change_type='major',  # Changed methodology
    adjustment_method='panama_canal',
    approval_status=False,  # Requires manual review
)

# After review, approve
if result.success:
    pipeline.promote_to_cleaned(
        asset='ES',
        version=result.version,
        approval_status=True,
        approval_reason='New methodology reviewed and approved',
    )
```

### 5. Use Version Control
Track changes to cleaning logic in Git:
```bash
git add src/data/cleaning/cleaning_pipeline.py
git commit -m "Update timezone handling logic"
# Changes automatically affect future runs
```

## Monitoring and Diagnostics

### Check Pipeline Status
```bash
# View latest version
ls -lah data/cleaned/ | tail -5

# Check report for specific version
cat data/cleaned/v1.0.0/ES/cleaning_report.json | jq .

# Count rows processed
cat data/cleaned/v1.0.0/ES/cleaning_report.json | jq '.metrics.rows_in'
```

### Performance Tuning
Monitor metrics across versions:
```python
import json
from pathlib import Path

# Compare versions
for v_dir in sorted(Path('data/cleaned').glob('v*')):
    for asset_dir in v_dir.glob('*'):
        report_path = asset_dir / 'cleaning_report.json'
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            print(f"{v_dir.name}/{asset_dir.name}: {report['metrics']['rows_out']:,} rows")
```

## Integration with Downstream Systems

### Using Cleaned Data in Backtest
```python
from src.storage.version_manager import VersionManager

# Load specific version
vm = VersionManager('data/cleaned', layer='cleaned')
metadata = vm.get_version('ES', 'v1.0.0')

# Use in backtest
df = pd.read_parquet(f'data/cleaned/v1.0.0/ES/ES_cleaned.parquet')
```

### Latest Version
```python
# Automatically use latest version
vm = VersionManager('data/cleaned', layer='cleaned')
latest = vm.get_latest_version('ES')
df = pd.read_parquet(f'data/cleaned/{latest.version}/ES/ES_cleaned.parquet')
```

## Troubleshooting Guide

### Pipeline Hangs
- Check available disk space
- Verify network connectivity (for cloud storage)
- Check system resource limits

### Out of Memory
- Process smaller date ranges
- Reduce number of assets per run
- Increase system RAM

### Slow Performance
- Check disk I/O (SSD recommended)
- Parallelize by asset
- Consider data compression

## Further Reading

- [Semantic Versioning](https://semver.org/)
- [Data Validation Framework](../src/data_io/README.md)
- [Trading Calendar Integration](../src/data/cleaning/trading_calendar.py)
- [Contract Roll Handling](../src/data/cleaning/contract_rolls.py)
- [Timezone Management](../src/data/cleaning/timezone_utils.py)
