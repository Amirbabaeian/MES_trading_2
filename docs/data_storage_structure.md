# Data Storage Structure - Hybrid Versioning System

## Overview

This document defines the directory structure and naming conventions for the hybrid versioning data storage system. The system is designed to support immutable raw data ingestion, validated/normalized cleaned data with semantic versioning, and computed feature sets with deterministic reproducibility.

**Storage Provider**: Cloud-agnostic paths (AWS S3, Google Cloud Storage, Azure Blob compatible). Examples use S3-style paths.

**Asset Scope**: MES, ES, VIX (equity index futures and volatility index)

---

## Directory Hierarchy

### 1. Raw Layer (Date-Based Versioning)

**Purpose**: Immutable storage of ingested data. Each ingestion date creates a new directory.

**Structure**: `/raw/{asset}/{YYYY-MM-DD}/`

**Characteristics**:
- One directory per ingestion date
- Data is immutable once written
- No modification or re-versioning
- Supports rollback to any historical ingestion
- Each date directory is independent

**Examples**:
```
s3://trading-data-bucket/raw/
├── MES/
│   ├── 2024-01-15/
│   │   ├── ohlcv.parquet
│   │   ├── metadata.json
│   │   └── tick_data.parquet
│   ├── 2024-01-16/
│   │   ├── ohlcv.parquet
│   │   ├── metadata.json
│   │   └── tick_data.parquet
│   └── 2024-01-17/
│       ├── ohlcv.parquet
│       ├── metadata.json
│       └── tick_data.parquet
├── ES/
│   ├── 2024-01-15/
│   │   ├── ohlcv.parquet
│   │   └── metadata.json
│   └── 2024-01-16/
│       ├── ohlcv.parquet
│       └── metadata.json
└── VIX/
    ├── 2024-01-15/
    │   ├── ohlcv.parquet
    │   └── metadata.json
    └── 2024-01-16/
        ├── ohlcv.parquet
        └── metadata.json
```

**File Naming Conventions**:
- All lowercase with underscores: `ohlcv.parquet`, `tick_data.parquet`
- Descriptive names indicating content type
- No version suffixes (date directory provides versioning)
- Parquet format for all tabular data

---

### 2. Cleaned Layer (Semantic Versioning)

**Purpose**: Validated, normalized, and consolidated data with consistent schema. Cleaned data aggregates multiple raw ingestions and applies standardized processing.

**Structure**: `/cleaned/v{X}/{asset}/`

**Characteristics**:
- Major version increments (v1, v2, v3, ...) for breaking changes
- Each asset maintains its own version stream
- All data under a version has consistent schema
- Multiple versions coexist for backward compatibility
- Represents "canonical" cleaned state for a version

**Examples**:
```
s3://trading-data-bucket/cleaned/
├── v1/
│   ├── MES/
│   │   ├── ohlcv.parquet
│   │   ├── metadata.json
│   │   └── quality_metrics.json
│   ├── ES/
│   │   ├── ohlcv.parquet
│   │   └── metadata.json
│   └── VIX/
│       ├── ohlcv.parquet
│       └── metadata.json
├── v2/
│   ├── MES/
│   │   ├── ohlcv.parquet
│   │   ├── metadata.json
│   │   └── quality_metrics.json
│   ├── ES/
│   │   ├── ohlcv.parquet
│   │   └── metadata.json
│   └── VIX/
│       ├── ohlcv.parquet
│       └── metadata.json
└── v3/
    ├── MES/
    │   ├── ohlcv.parquet
    │   ├── metadata.json
    │   └── quality_metrics.json
    └── ...
```

**File Naming Conventions**:
- Asset-specific files: `ohlcv.parquet`, `quality_metrics.json`
- Metadata always: `metadata.json`
- All lowercase with underscores
- No date suffixes (version metadata handles temporal aspects)

**Partitioned Data Support** (if applicable):
```
s3://trading-data-bucket/cleaned/v1/MES/
├── ohlcv/
│   ├── date=2024-01-01/
│   │   ├── 00000_000.parquet
│   │   └── 00001_000.parquet
│   ├── date=2024-01-02/
│   │   └── 00000_000.parquet
│   └── ...
├── metadata.json
└── _manifest
```

---

### 3. Features Layer (Semantic Versioning)

**Purpose**: Computed features and derived datasets from cleaned data. Features have explicit definitions and version-specific computation logic.

**Structure**: `/features/v{Y}/{feature_set}/`

**Characteristics**:
- Major version increments for breaking changes in computation logic or definitions
- Feature sets are independent and composable
- Each feature set contains derived data for all relevant assets
- Versions support A/B testing and gradual rollout
- Reproducibility via explicit feature definitions

**Examples**:
```
s3://trading-data-bucket/features/
├── v1/
│   ├── price_volatility/
│   │   ├── MES.parquet
│   │   ├── ES.parquet
│   │   ├── VIX.parquet
│   │   ├── metadata.json
│   │   └── feature_definitions.json
│   ├── momentum_indicators/
│   │   ├── MES.parquet
│   │   ├── ES.parquet
│   │   ├── VIX.parquet
│   │   ├── metadata.json
│   │   └── feature_definitions.json
│   └── volume_profile/
│       ├── MES.parquet
│       ├── ES.parquet
│       ├── metadata.json
│       └── feature_definitions.json
├── v2/
│   ├── price_volatility/
│   │   ├── MES.parquet
│   │   ├── ES.parquet
│   │   ├── VIX.parquet
│   │   ├── metadata.json
│   │   └── feature_definitions.json
│   └── momentum_indicators/
│       ├── MES.parquet
│       ├── ES.parquet
│       ├── VIX.parquet
│       ├── metadata.json
│       └── feature_definitions.json
└── v3/
    └── ...
```

**File Naming Conventions**:
- Feature set directories use snake_case: `price_volatility`, `momentum_indicators`
- Asset files include asset name: `MES.parquet`, `ES.parquet`
- Metadata: `metadata.json`
- Feature definitions: `feature_definitions.json`
- All lowercase with underscores

**Partitioned Data Support** (if applicable):
```
s3://trading-data-bucket/features/v1/price_volatility/
├── MES/
│   ├── date=2024-01-01/
│   │   └── 00000_000.parquet
│   ├── date=2024-01-02/
│   │   └── 00000_000.parquet
│   └── ...
├── ES/
│   └── ...
├── metadata.json
└── feature_definitions.json
```

---

## Complete Directory Tree Example

```
s3://trading-data-bucket/
├── raw/
│   ├── MES/
│   │   ├── 2024-01-15/
│   │   │   ├── ohlcv.parquet
│   │   │   ├── tick_data.parquet
│   │   │   └── metadata.json
│   │   ├── 2024-01-16/
│   │   │   ├── ohlcv.parquet
│   │   │   ├── tick_data.parquet
│   │   │   └── metadata.json
│   │   └── 2024-01-17/
│   │       ├── ohlcv.parquet
│   │       ├── tick_data.parquet
│   │       └── metadata.json
│   ├── ES/
│   │   ├── 2024-01-15/
│   │   │   ├── ohlcv.parquet
│   │   │   └── metadata.json
│   │   ├── 2024-01-16/
│   │   │   ├── ohlcv.parquet
│   │   │   └── metadata.json
│   │   └── 2024-01-17/
│   │       ├── ohlcv.parquet
│   │       └── metadata.json
│   └── VIX/
│       ├── 2024-01-15/
│       │   ├── ohlcv.parquet
│       │   └── metadata.json
│       ├── 2024-01-16/
│       │   ├── ohlcv.parquet
│       │   └── metadata.json
│       └── 2024-01-17/
│           ├── ohlcv.parquet
│           └── metadata.json
├── cleaned/
│   ├── v1/
│   │   ├── MES/
│   │   │   ├── ohlcv.parquet
│   │   │   ├── quality_metrics.json
│   │   │   └── metadata.json
│   │   ├── ES/
│   │   │   ├── ohlcv.parquet
│   │   │   └── metadata.json
│   │   └── VIX/
│   │       ├── ohlcv.parquet
│   │       └── metadata.json
│   ├── v2/
│   │   ├── MES/
│   │   │   ├── ohlcv.parquet
│   │   │   ├── quality_metrics.json
│   │   │   └── metadata.json
│   │   ├── ES/
│   │   │   ├── ohlcv.parquet
│   │   │   └── metadata.json
│   │   └── VIX/
│   │       ├── ohlcv.parquet
│   │       └── metadata.json
│   └── v3/
│       ├── MES/
│       │   ├── ohlcv.parquet
│       │   ├── quality_metrics.json
│       │   └── metadata.json
│       ├── ES/
│       │   ├── ohlcv.parquet
│       │   └── metadata.json
│       └── VIX/
│           ├── ohlcv.parquet
│           └── metadata.json
└── features/
    ├── v1/
    │   ├── price_volatility/
    │   │   ├── MES.parquet
    │   │   ├── ES.parquet
    │   │   ├── VIX.parquet
    │   │   ├── metadata.json
    │   │   └── feature_definitions.json
    │   ├── momentum_indicators/
    │   │   ├── MES.parquet
    │   │   ├── ES.parquet
    │   │   ├── VIX.parquet
    │   │   ├── metadata.json
    │   │   └── feature_definitions.json
    │   └── volume_profile/
    │       ├── MES.parquet
    │       ├── ES.parquet
    │       ├── metadata.json
    │       └── feature_definitions.json
    ├── v2/
    │   ├── price_volatility/
    │   │   ├── MES.parquet
    │   │   ├── ES.parquet
    │   │   ├── VIX.parquet
    │   │   ├── metadata.json
    │   │   └── feature_definitions.json
    │   └── momentum_indicators/
    │       ├── MES.parquet
    │       ├── ES.parquet
    │       ├── VIX.parquet
    │       ├── metadata.json
    │       └── feature_definitions.json
    └── v3/
        └── ...
```

---

## Naming Convention Standards

### General Rules
- **Case**: All lowercase (filenames and directories)
- **Separators**: Underscores (`_`) for multi-word names, no hyphens or spaces
- **Descriptiveness**: Names must clearly indicate content type and purpose
- **Special Characters**: No dots, slashes, or special characters except underscore
- **Path Length**: Keep individual path segments under 64 characters

### Examples
| Layer | Type | ✓ Valid | ✗ Invalid |
|-------|------|---------|-----------|
| Raw | File | `ohlcv.parquet`, `tick_data.parquet` | `OHLCV.parquet`, `ohlcv_data.parquet` |
| Cleaned | File | `quality_metrics.json`, `ohlcv.parquet` | `QualityMetrics.json`, `ohlcv-metrics.json` |
| Features | Directory | `price_volatility`, `momentum_indicators` | `PriceVolatility`, `price-volatility` |
| Features | File | `MES.parquet`, `feature_definitions.json` | `MES_features.parquet`, `definitions.json` |

### Cloud Provider Path Variations

**AWS S3**:
```
s3://bucket-name/raw/MES/2024-01-15/ohlcv.parquet
```

**Google Cloud Storage (GCS)**:
```
gs://bucket-name/raw/MES/2024-01-15/ohlcv.parquet
```

**Azure Blob Storage**:
```
abfs://container-name@account.dfs.core.windows.net/raw/MES/2024-01-15/ohlcv.parquet
```

**Local File System** (development/testing):
```
file:///data/trading/raw/MES/2024-01-15/ohlcv.parquet
```

---

## Design Rationale

### Why Date-Based Raw Versioning?
- **Immutability**: Each date creates a separate, immutable snapshot
- **Auditability**: Easy to trace data lineage to source ingestion date
- **Rollback**: Support for recovery to any historical state
- **Scalability**: Independent date directories prevent lock contention
- **Simplicity**: Date is universal identifier, no manual versioning needed

### Why Semantic Versioning for Cleaned/Features?
- **Breaking Changes**: Major versions for schema or logic changes
- **Backward Compatibility**: Multiple versions coexist for dependent systems
- **Clarity**: Version number communicates change scope
- **Migration**: Clear upgrade path (v1 → v2 → v3)
- **A/B Testing**: Support for parallel version evaluation

### Why Parquet Format?
- **Compression**: Efficient storage and reduced bandwidth
- **Columnar**: Optimized for analytical queries
- **Schema Evolution**: Supports nested structures and flexible additions
- **Compatibility**: Wide support across data platforms
- **Reproducibility**: Binary identical across platforms

### Why Separate Feature Sets?
- **Composability**: Mix and match features for different strategies
- **Independent Versioning**: Features evolve independently
- **Reusability**: Features used across multiple models
- **Testability**: Easier to validate feature computation in isolation

### Why Asset-Per-File in Features?
- **Simplicity**: Clean mapping of asset to feature data
- **Parallel Processing**: Assets can be processed independently
- **Partial Updates**: Can update one asset without touching others
- **Access Patterns**: Users typically request asset-specific feature data

---

## Summary

The hybrid versioning structure balances:
- **Immutability** (raw layer) for audit trails and historical accuracy
- **Reproducibility** (cleaned and features) through explicit semantic versioning
- **Flexibility** (coexisting versions) for gradual migration and A/B testing
- **Scalability** (independent asset/version directories) for parallel processing
- **Clarity** (consistent naming) for programmatic access and human readability
