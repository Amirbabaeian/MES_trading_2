# Cloud Storage Synchronization Utilities

Comprehensive utilities for hybrid cloud architecture enabling team members to work locally while sharing data through cloud storage.

## Features

### Cloud Provider Support
- **AWS S3**: Full implementation with multipart uploads, streaming downloads, and storage class management
- **Google Cloud Storage (GCS)**: Complete implementation with support for different storage classes
- **Azure Blob Storage**: Full implementation with access tier management
- Pluggable architecture for adding additional providers

### Bidirectional Sync
- **Upload**: Push local data to cloud with progress tracking
- **Download**: Pull cloud data to local machine
- **Bidirectional Sync**: Smart sync that:
  - Only transfers new/modified files
  - Compares timestamps and checksums
  - Detects and resolves conflicts
  - Preserves directory structure

### Incremental Transfers
- **State Tracking**: SQLite database tracks sync history
- **Resume Support**: Interrupted transfers can be resumed
- **Checksum Verification**: MD5 checksums ensure data integrity
- **Smart Sync**: Only transfers changed files based on:
  - File modification timestamps
  - File checksums/etags
  - Directory structure changes

### Conflict Resolution
Multiple strategies for handling simultaneous modifications:
- **timestamp_wins**: Newest file wins
- **local_wins**: Local file always wins
- **cloud_wins**: Cloud file always wins
- **manual**: Block sync and require manual resolution
- **version_preserve**: Keep both versions with conflict suffix

### Large File Handling
- Multipart uploads for files >100MB
- Streaming downloads for efficient memory usage
- File chunking (5MB chunks by default)
- Progress callbacks for transfer monitoring

### Error Handling & Reliability
- Automatic retry logic with exponential backoff
- Transient failure recovery (network issues, throttling)
- Comprehensive error logging
- Operation tracking and audit trails

## Architecture

```
CloudStorage (abstract interface)
├── S3CloudStorage (AWS S3)
├── GCSCloudStorage (Google Cloud Storage)
└── AzureCloudStorage (Azure Blob Storage)

CloudSyncEngine
├── Bidirectional sync orchestration
├── Conflict detection and resolution
├── Progress reporting
└── State management

SyncStateDatabase
├── SQLite-based state tracking
├── Operation history
├── Conflict logs
└── Statistics

Configuration
├── YAML-based config
├── Environment variable support
└── Provider factory
```

## Quick Start

### Installation

Required dependencies:
```bash
# For all providers
pip install boto3 google-cloud-storage azure-storage-blob PyYAML

# Or specific providers
pip install boto3          # AWS S3
pip install google-cloud-storage  # GCS
pip install azure-storage-blob    # Azure
```

### Basic Usage

#### 1. Configuration

Create `config/storage_config.yaml`:

```yaml
storage:
  provider: s3
  bucket: my-data-bucket
  region: us-east-1
  credentials:
    aws_access_key_id: ${AWS_ACCESS_KEY_ID}
    aws_secret_access_key: ${AWS_SECRET_ACCESS_KEY}

sync_state_db: .sync_state.db
conflict_strategy: timestamp_wins
max_workers: 4
retry_attempts: 3
```

#### 2. Upload Local Data

```bash
python scripts/sync_data.py upload \
  --config config/storage_config.yaml \
  --local-dir data/raw \
  --cloud-prefix raw/
```

#### 3. Download Cloud Data

```bash
python scripts/sync_data.py download \
  --config config/storage_config.yaml \
  --cloud-prefix cleaned/ \
  --local-dir data/cleaned
```

#### 4. Bidirectional Sync

```bash
python scripts/sync_data.py sync \
  --config config/storage_config.yaml \
  --local-dir data/ \
  --cloud-prefix data/
```

#### 5. View Status and History

```bash
python scripts/sync_data.py status --config config/storage_config.yaml
```

#### 6. Retry Failed Transfers

```bash
python scripts/sync_data.py retry --config config/storage_config.yaml
```

## Programmatic Usage

### Basic Upload

```python
from src.storage.config import SyncEngineBuilder
from pathlib import Path

# Create engine from config file
engine = SyncEngineBuilder.from_config_file(Path("config/storage_config.yaml"))

# Upload directory
progress = engine.upload_directory(
    local_dir=Path("data/raw"),
    cloud_prefix="raw/",
)

print(f"Uploaded {progress.completed_files} files")
print(f"Transfer speed: {progress.transfer_speed_mbps:.2f} MB/s")

engine.cloud_storage.disconnect()
```

### Bidirectional Sync with Conflict Handling

```python
from src.storage.config import SyncEngineBuilder
from pathlib import Path

engine = SyncEngineBuilder.from_config_file(Path("config/storage_config.yaml"))

# Perform sync
progress, conflicts = engine.sync_bidirectional(
    local_dir=Path("data"),
    cloud_prefix="data/",
)

# Handle conflicts
if conflicts:
    print(f"Found {len(conflicts)} conflicts:")
    for conflict in conflicts:
        print(f"  {conflict.file_path}: {conflict.conflict_reason}")
        print(f"    Resolution: {conflict.resolution}")

engine.cloud_storage.disconnect()
```

### Download with Progress Tracking

```python
from src.storage.config import SyncEngineBuilder
from pathlib import Path

def progress_callback(bytes_transferred):
    """Called periodically during transfer."""
    print(f"Transferred {bytes_transferred:,} bytes")

engine = SyncEngineBuilder.from_config_file(Path("config/storage_config.yaml"))

progress = engine.download_directory(
    cloud_prefix="data/",
    local_dir=Path("data/local"),
)

engine.cloud_storage.disconnect()
```

### Direct Provider Usage

```python
from src.storage.providers.s3 import S3CloudStorage
from pathlib import Path

# Create S3 provider
s3 = S3CloudStorage(
    bucket="my-bucket",
    region="us-east-1",
)
s3.connect()

# Upload a file
etag = s3.upload_file(
    local_path=Path("file.parquet"),
    cloud_path="data/file.parquet",
)

# Download a file
checksum = s3.download_file(
    cloud_path="data/file.parquet",
    local_path=Path("downloaded_file.parquet"),
)

# List files
files = s3.list_files(prefix="data/")
for file in files:
    print(f"{file.path} ({file.size} bytes)")

s3.disconnect()
```

## Configuration Details

### Storage Config Options

```yaml
storage:
  provider: s3|gcs|azure           # Required
  bucket: bucket-or-container-name # Required
  region: region-name              # S3 only
  storage_class: tier-name         # Optional
  credentials:                     # Optional, supports env vars
    # Provider-specific credentials
```

### Sync Config Options

```yaml
sync_state_db: .sync_state.db                    # State database path
conflict_strategy: timestamp_wins|local_wins|...  # Conflict strategy
max_workers: 4                                   # Concurrent transfers
retry_attempts: 3                                # Retry count
retry_delay_seconds: 5                           # Delay between retries
```

### Environment Variables

All credentials and paths support environment variable substitution:

```yaml
# Format: ${VAR_NAME} or ${VAR_NAME:default}
credentials:
  aws_access_key_id: ${AWS_ACCESS_KEY_ID}
  aws_secret_access_key: ${AWS_SECRET_ACCESS_KEY:default_key}
```

## Advanced Features

### Incremental Sync

The system tracks file state in SQLite:

```python
from src.storage.sync_state import SyncStateDatabase

db = SyncStateDatabase(Path(".sync_state.db"))

# Get sync statistics
stats = db.get_sync_statistics()
print(f"Synced bytes: {stats['total_bytes_synced']}")
print(f"Failed operations: {stats.get('recent_failures_24h', 0)}")

# Get failed records for retry
failed = db.get_failed_records()
for record in failed:
    print(f"Retry: {record.local_path}")
```

### Custom Conflict Resolution

```python
from src.storage.cloud_sync import ConflictStrategy, CloudSyncEngine
from src.storage.config import SyncEngineBuilder

# Create engine with MANUAL conflict strategy
engine = CloudSyncEngine(
    cloud_storage=provider,
    sync_state_db=Path(".sync_state.db"),
    conflict_strategy=ConflictStrategy.MANUAL,
)

progress, conflicts = engine.sync_bidirectional(
    local_dir=Path("data"),
    cloud_prefix="data/",
)

# Manually resolve conflicts
for conflict in conflicts:
    if should_keep_local(conflict.file_path):
        conflict.resolution = ConflictStrategy.LOCAL_WINS
    else:
        conflict.resolution = ConflictStrategy.CLOUD_WINS
```

### Custom Progress Monitoring

```python
from src.storage.providers.s3 import S3CloudStorage
from pathlib import Path

s3 = S3CloudStorage(bucket="my-bucket")
s3.connect()

bytes_transferred = 0

def progress_callback(bytes_chunk):
    global bytes_transferred
    bytes_transferred += bytes_chunk
    print(f"Progress: {bytes_transferred:,} bytes")

s3.upload_file(
    local_path=Path("large_file.parquet"),
    cloud_path="data/large_file.parquet",
    progress_callback=progress_callback,
)

s3.disconnect()
```

## Cloud Provider Setup

### AWS S3

```bash
# Configure AWS credentials
aws configure
# OR set environment variables
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# Create S3 bucket
aws s3 mb s3://my-trading-data --region us-east-1
```

### Google Cloud Storage

```bash
# Install gcloud CLI and authenticate
gcloud auth login
gcloud config set project MY_PROJECT

# Create service account
gcloud iam service-accounts create my-sync-account
gcloud iam service-accounts keys create service-account.json \
  --iam-account=my-sync-account@MY_PROJECT.iam.gserviceaccount.com

# Grant permissions
gcloud projects add-iam-policy-binding MY_PROJECT \
  --member=serviceAccount:my-sync-account@MY_PROJECT.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin

# Create GCS bucket
gsutil mb gs://my-gcs-bucket
```

### Azure Blob Storage

```bash
# Login with Azure CLI
az login

# Create storage account
az storage account create \
  --name mystorageaccount \
  --resource-group myresourcegroup

# Create container
az storage container create \
  --name mycontainer \
  --account-name mystorageaccount

# Get connection string
az storage account show-connection-string \
  --name mystorageaccount
```

## Performance Optimization

### Multipart Upload Configuration

```python
# Large files automatically use multipart upload
# Chunk size is configurable per provider:

s3.DEFAULT_CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB chunks
```

### Concurrent Transfers

```yaml
# Increase workers for faster sync on high-bandwidth connections
max_workers: 8  # Default is 4

# Decrease for limited connections
max_workers: 2
```

### Storage Class Selection

```yaml
# For infrequently accessed data
storage_class: STANDARD_IA  # S3
storage_class: NEARLINE     # GCS
storage_class: cool         # Azure

# For archival
storage_class: GLACIER      # S3
storage_class: ARCHIVE      # GCS
storage_class: archive      # Azure
```

## Troubleshooting

### Connection Issues

```python
try:
    engine = SyncEngineBuilder.from_config_file("config/storage_config.yaml")
except RuntimeError as e:
    print(f"Connection failed: {e}")
    # Check credentials and bucket name
```

### Retry Failed Transfers

```bash
# View failed operations
python scripts/sync_data.py status --verbose

# Retry automatically
python scripts/sync_data.py retry --config config/storage_config.yaml
```

### Clean Up Old State

```python
from src.storage.sync_state import SyncStateDatabase
from pathlib import Path

db = SyncStateDatabase(Path(".sync_state.db"))

# Delete records older than 30 days
deleted = db.cleanup_old_records(days_old=30)
print(f"Deleted {deleted} old records")
```

## File Organization

```
src/storage/
├── __init__.py              # Package exports
├── cloud_sync.py            # Core sync engine
├── sync_state.py            # SQLite state tracking
├── config.py                # Configuration management
├── providers/
│   ├── __init__.py
│   ├── s3.py               # AWS S3 implementation
│   ├── gcs.py              # Google Cloud Storage
│   └── azure.py            # Azure Blob Storage
└── README.md                # This file

scripts/
├── sync_data.py             # CLI utility

config/
├── storage_config.yaml      # Configuration template

tests/
└── storage/
    ├── __init__.py
    └── test_cloud_sync.py   # Comprehensive tests
```

## Testing

```bash
# Run all storage tests
pytest tests/storage/ -v

# Run specific test
pytest tests/storage/test_cloud_sync.py::TestUploadDirectory -v

# With coverage
pytest tests/storage/ --cov=src/storage
```

## Performance Benchmarks

Typical performance (depends on network and file sizes):

| Operation | Speed | Notes |
|-----------|-------|-------|
| Upload small files (<1MB) | ~10-20 files/sec | Network limited |
| Upload large files (100MB+) | ~50-100 MB/s | With 4 workers |
| Download small files | ~10-20 files/sec | Network limited |
| Download large files | ~50-100 MB/s | With streaming |
| List operations | ~1000 files/sec | Cloud API limited |

## Security Considerations

### Credential Management

- Never commit credentials to version control
- Use environment variables or credential files outside repo
- Rotate access keys regularly
- Use IAM roles when possible

### Data Encryption

- Enable encryption at rest for cloud storage
- Use encrypted connections (HTTPS/TLS) - default
- Consider client-side encryption for sensitive data

### Access Control

- Use cloud provider IAM to limit access
- Implement bucket policies to prevent public access
- Enable versioning for accidental deletion recovery
- Monitor access logs

## Limitations and Future Work

- No delta/block-level sync (full file transfer)
- No built-in compression during transfer (cloud providers handle this)
- SQLite state DB not suitable for distributed teams (consider DynamoDB/Firestore)
- No built-in bandwidth throttling
- No scheduled sync (use cron/scheduled tasks)

## Contributing

To add a new cloud provider:

1. Implement `CloudStorage` abstract interface
2. Add provider class to `src/storage/providers/`
3. Update `CloudStorageFactory.create()`
4. Add tests to `tests/storage/test_cloud_sync.py`
5. Update documentation

## License

Part of the trading data pipeline project.
