# Versioning Semantics - Hybrid Versioning System

## Overview

This document defines the versioning strategies, increment rules, compatibility policies, and migration paths for each data layer in the hybrid versioning system.

**Key Principle**: Versioning reflects the type of change—date versioning for immutable snapshots, semantic versioning for reproducible transformations.

---

## Layer 1: Raw Data Versioning

### Versioning Strategy: Date-Based (Immutable)

**Format**: `/raw/{asset}/{YYYY-MM-DD}/`

**Rule**: Create a new directory for each data ingestion, regardless of content changes.

### Version Increment Rules

**When to Create a New Date Directory**:
1. **Daily Ingestion**: Standard case—one directory per trading day
2. **Intra-Day Updates**: Special handling required
   - If data is corrected within the same day, use the same date directory
   - Overwrite files with corrected versions
   - Update metadata to document correction
   - Note: This breaks true immutability; see alternative below
3. **Alternative: Intra-Day Snapshots** (Recommended for strict immutability)
   - Create subdirectories with timestamps: `2024-01-15/10_30_00/`, `2024-01-15/16_00_00/`
   - Preserves immutability at each snapshot level
   - Enables rollback to intermediate states

### Immutability Guarantees

**Once Written**:
- Files in a date directory are never modified
- Corrections create new date directories
- Historical access always returns identical data

**Exceptions** (Documented):
- Metadata corrections (timestamps, format, not data values)
- Parquet file compression/optimization (if identical after decompression)
- These exceptions must be logged in metadata

### Compatibility Rules

- **Always Compatible**: Any raw data can be re-processed with newer cleaning logic
- **No Migration Needed**: Raw data requires no transformation between versions
- **Rollback**: Always possible to any historical date by re-running cleaning pipeline

### Metadata Requirements

- `metadata.json` at each date directory
- Required fields: `ingestion_timestamp`, `record_count`, `data_quality_flags`
- See `schema/metadata_schema.json` for complete specification

### Examples

**Standard daily ingestion**:
```
raw/MES/2024-01-15/ohlcv.parquet  ← Ingested 2024-01-15
raw/MES/2024-01-16/ohlcv.parquet  ← Ingested 2024-01-16
raw/MES/2024-01-17/ohlcv.parquet  ← Ingested 2024-01-17
```

**Intra-day snapshot (strict immutability)**:
```
raw/MES/2024-01-15/09_30_00/ohlcv.parquet  ← Market open
raw/MES/2024-01-15/16_00_00/ohlcv.parquet  ← Market close
raw/MES/2024-01-15/20_00_00/ohlcv.parquet  ← After-hours
```

---

## Layer 2: Cleaned Data Versioning

### Versioning Strategy: Semantic (Breaking Changes)

**Format**: `/cleaned/v{X}/{asset}/`

**Rule**: Increment major version when cleaning logic or schema changes in breaking way.

### Version Increment Rules

#### When to Increment Version (v1 → v2)

**Breaking Changes** (Require version increment):
1. **Schema Changes**:
   - Add new required column (consumers need to handle)
   - Remove existing column (consumer code breaks)
   - Change column data type (e.g., float → int)
   - Change column semantics (e.g., "price" is now "price_adjusted")

2. **Data Transformation Changes**:
   - Change normalization logic (e.g., decimal places, unit conversion)
   - Change null handling policy
   - Change outlier filtering rules
   - Change timestamp alignment (e.g., UTC → EST)

3. **Composition Changes**:
   - Change which raw data sources are included
   - Change aggregation level (e.g., per-contract to per-symbol)
   - Change data availability windows

4. **Quality Assurance Changes**:
   - Change validation rules that reject records
   - Change imputation strategies
   - Change duplicate detection/removal logic

#### When NOT to Increment Version (v1.x patch)

**Non-Breaking Changes** (Maintain version):
- Fix data corruption in existing records
- Reprocess raw data with same cleaning logic
- Update metadata/documentation
- Optimize file format/compression
- Add optional metadata fields

**Note**: This system uses major versioning only (no minor/patch versions). All non-breaking changes are regenerated within the same major version.

### Version Lifecycle

```
DESIGN (v1)
    ↓
IMPLEMENT (v1)
    ↓
VALIDATE (v1) → Found issue? → FIX BUG (v1, same files)
    ↓
PRODUCTION (v1)
    ↓
EVOLUTION (identified breaking change) → CREATE v2
    ↓
PARALLEL OPERATION (v1 and v2 coexist)
    ↓
MIGRATION → Consumers move to v2
    ↓
SUNSET (v1 deprecated after migration)
```

### Compatibility Rules

**Within Version**: All data under v1 has identical schema and transformation logic
- Safe to combine data from v1/MES, v1/ES, v1/VIX

**Across Versions**: v2 data is NOT backward compatible with v1 consumers
- Applications expecting v1 schema will fail on v2 data
- Requires explicit consumer code updates

**Parallel Operation**: v1 and v2 coexist during migration period
- Allow gradual consumer migration
- No forced cutover
- Both versions can be queried independently

### Migration Paths

#### From v1 to v2
1. **Phase 1: Coexistence** (Week 1-2)
   - v2 created and populated
   - v1 remains available for existing consumers
   - New projects can choose v2

2. **Phase 2: Dual Consumption** (Week 3-4)
   - Key applications migrate to v2
   - Test v2 performance in production
   - Validate results against v1

3. **Phase 3: Cutover** (Week 5-6)
   - Remaining applications updated to v2
   - Point all automated pipelines to v2
   - Schedule v1 deprecation

4. **Phase 4: Archive** (Week 7+)
   - v1 moved to archive storage
   - Keep accessible for historical analysis
   - Remove from primary access path

#### Backward Compatibility Guarantees
- **Guaranteed**: All raw data can be re-cleaned with any version
- **Not Guaranteed**: Exact byte-for-byte data match (recomputation may differ)

### Example Version Transitions

**v1 → v2: Add new quality metric**
```
v1/MES/ohlcv.parquet          Schema: [timestamp, open, high, low, close, volume]
    ↓ (break in logic)
v2/MES/ohlcv.parquet          Schema: [timestamp, open, high, low, close, volume, vwap]
                              Action: Requires v2 to read new column
                              Status: Major version bump
```

**v1 → v1: Fix data corruption (same version)**
```
v1/MES/ohlcv.parquet (corrupt values)
    ↓ (non-breaking fix)
v1/MES/ohlcv.parquet (corrected, same schema)
                              Action: Reprocessed with same logic
                              Status: No version change
```

**v1 → v2: Change outlier removal policy**
```
v1/MES/2024-01-15/          100,000 records (outliers included)
    ↓ (logic change)
v2/MES/                      98,500 records (outliers removed)
                              Action: Data count differs, different schema interpretation
                              Status: Major version bump
```

---

## Layer 3: Features Versioning

### Versioning Strategy: Semantic (Breaking Changes)

**Format**: `/features/v{Y}/{feature_set}/`

**Rule**: Increment major version when feature computation or definitions change in breaking way.

### Version Increment Rules

#### When to Increment Version (v1 → v2)

**Breaking Changes** (Require version increment):
1. **Computation Logic Changes**:
   - Change algorithm (e.g., SMA vs EMA for momentum)
   - Change parameters (e.g., window size 20 → 30 days)
   - Change input data source/version (e.g., raw → cleaned v2)
   - Change mathematical formula

2. **Output Schema Changes**:
   - Add required columns
   - Remove columns
   - Change data types
   - Change column naming/semantics

3. **Definition Changes**:
   - Redefine what constitutes a feature
   - Change feature semantics (e.g., "volatility" calculation method)

4. **Time Window Changes**:
   - Change feature computation window (e.g., 30-day → 60-day)
   - Change data availability (e.g., 5-year history → 10-year)

#### When NOT to Increment Version

**Non-Breaking Changes** (Maintain version):
- Bug fix in computation (incorrect result → correct result, same schema)
- Recompute with same logic using new underlying cleaned data
- Add optional metadata fields
- Update documentation/definitions

### Version Lifecycle

```
FEATURE DESIGN (v1)
    ↓
IMPLEMENTATION (v1)
    ↓
VALIDATION & BACKTESTING (v1)
    ↓
PRODUCTION DEPLOYMENT (v1)
    ↓
ANALYSIS (identify improvement) → CREATE v2
    ↓
PARALLEL COMPUTATION (v1 and v2 coexist)
    ↓
A/B TESTING (models use v1 or v2)
    ↓
WINNER SELECTION
    ↓
MIGRATION → Models move to v2
    ↓
SUNSET (v1 deprecated)
```

### Compatibility Rules

**Within Feature Set**: All assets in a version have identical computation
- `features/v1/price_volatility/MES.parquet` uses same logic as `.../ES.parquet`
- Safe to compare features across assets

**Across Feature Sets**: Feature sets within a version are independent
- Can combine `v1/price_volatility` + `v1/momentum_indicators`
- Careful about version alignment (prefer all from same major version)

**Across Versions**: v2 features are NOT backward compatible
- Models trained on v1 features may not work with v2 features
- Different statistical properties
- Requires retraining

**Dependencies**: Features may depend on specific cleaned data version
- Feature v1 requires cleaned v1 (or compatible version)
- Document this explicitly in `feature_definitions.json`
- Breaking change if underlying cleaned version becomes unavailable

### Migration Paths

#### From Feature v1 to v2
1. **Phase 1: Parallel Computation** (Week 1-2)
   - Compute v2 features alongside v1
   - No model changes yet
   - Store both versions

2. **Phase 2: Validation** (Week 3-4)
   - Compare v1 and v2 outputs
   - Backtest models with v2 features
   - Measure performance impact

3. **Phase 3: A/B Testing** (Week 5-8)
   - Deploy models using v1 and v2 in parallel
   - Monitor live trading performance
   - Gather statistical evidence

4. **Phase 4: Gradual Migration** (Week 9-12)
   - Increase allocation to v2-based models
   - Monitor for market regime changes
   - Maintain v1 as fallback

5. **Phase 5: Complete Cutover** (Week 13+)
   - All models use v2
   - Archive v1 computation results
   - Keep v1 definitions for historical analysis

#### Backward Compatibility Guarantees
- **Guaranteed**: Features can be recomputed with newer code (if parameters same)
- **Guaranteed**: Features from different assets in same version are comparable
- **Not Guaranteed**: Exact match with original computation (floating point differences)

### Example Version Transitions

**v1 → v2: Change volatility calculation window**
```
v1/price_volatility/         Uses 30-day rolling volatility window
    ↓
v2/price_volatility/         Uses 60-day rolling volatility window
                              Action: Different results, different semantics
                              Status: Major version bump
                              Impact: Models must be retrained
```

**v1 → v1: Fix calculation bug (same version)**
```
v1/price_volatility/         Wrong calculation producing NaNs
    ↓ (bug fix)
v1/price_volatility/         Correct calculation, same window
                              Action: Recomputed with corrected logic
                              Status: No version change
                              Impact: Automated recomputation, no model changes
```

**v1 → v2: Add new derived feature column**
```
v1/momentum_indicators/      Columns: [timestamp, rsi, macd]
    ↓
v2/momentum_indicators/      Columns: [timestamp, rsi, macd, signal_line]
                              Action: Schema extended with new column
                              Status: Major version bump
                              Impact: Consumers must handle new column
```

---

## Decision Framework: When to Version Up?

### The Breaking Change Test

Ask these questions:
1. **Schema Change?** Will existing code that reads this data break?
   - YES → Version up
   - NO → Check next question

2. **Different Results?** Do consumers expect same values?
   - YES (same results expected) → Version up
   - NO (results expected to differ) → Check next question

3. **Dependent Systems Affected?** Will any downstream system be surprised?
   - YES → Version up
   - NO → Keep same version

### Examples Applying the Framework

**Scenario A: Add optional metadata field**
- Schema Change? NO (optional field, no break)
- Different Results? NO (data values same)
- Dependent Systems? NO
- **Decision**: Keep same version ✓

**Scenario B: Change normalization from [0,1] to [-1,1]**
- Schema Change? NO (same columns)
- Different Results? YES (completely different values)
- Dependent Systems? YES (models expect [0,1])
- **Decision**: Version up ✓

**Scenario C: Reprocess raw data with updated cleaning rules**
- Schema Change? YES (new columns added)
- Different Results? YES
- Dependent Systems? YES
- **Decision**: Version up ✓

**Scenario D: Fix data type from float to double for precision**
- Schema Change? YES (technically)
- Different Results? NO (same values, just more precision)
- Dependent Systems? NO (double is superset of float)
- **Decision**: Version up (to be safe) ✓

---

## Versioning Best Practices

### 1. Document Every Version

- Create `CHANGELOG.md` in each versioned directory
- List breaking changes explicitly
- Include: date, author, description, impact

### 2. Plan Major Versions

- Design v2 before v1 reaches production
- Identify known future breaking changes
- Plan migration path proactively

### 3. Minimize Major Versions

- Batch multiple breaking changes into one major version
- Avoid frequent version bumps
- Deprecation takes time and resources

### 4. Communicate Changes

- Announce version deprecation schedule
- Provide migration guides
- Support multiple versions during transition

### 5. Automate Compatibility Checks

- Detect schema changes automatically
- Flag breaking changes in CI/CD
- Prevent accidental incompatible commits

### 6. Version Dependencies

- List which cleaned version features depend on
- Update when cleaned version changes
- Validate transitivity of dependencies

### 7. Archive Gracefully

- Keep old versions accessible for historical queries
- Move to cheaper storage after migration
- Maintain reference data for reproducibility

---

## Summary

**Raw Layer**: Date-based immutability
- New directory per ingestion date
- No versioning needed—date is version
- Always safe to re-process with any cleaning logic

**Cleaned Layer**: Semantic versioning with breaking changes
- Major version increments (v1, v2, ...) only
- Change schema or transformation logic → version up
- Multiple versions coexist during migration

**Features Layer**: Semantic versioning with breaking changes
- Major version increments (v1, v2, ...) only
- Change computation logic or output schema → version up
- A/B testing supported via parallel versions

**Key Rule**: If downstream systems will be surprised by changes, version up.
