# Provider Configuration Templates

This directory contains configuration templates for data provider adapters. Each template documents the required and optional settings for the corresponding adapter.

## Files

- `ib_config.yaml` - Interactive Brokers adapter configuration
- `polygon_config.yaml` - Polygon.io adapter configuration
- `databento_config.yaml` - Databento adapter configuration

## Using Configuration Templates

### 1. Interactive Brokers

Copy `ib_config.yaml` and update with your TWS/Gateway settings:

```yaml
connection:
  host: "127.0.0.1"        # Your TWS/Gateway host
  port: 7497               # 7497 for live, 7498 for paper
  account_id: "DU123456"   # Your IB account ID (REQUIRED)
  client_id: 1             # Unique client ID
```

Python usage:
```python
from src.data_ingestion.ib_provider import IBDataProvider

provider = IBDataProvider(
    account_id="DU123456",
    host="127.0.0.1",
    port=7497,
    client_id=1
)
provider.authenticate()
```

**Requirements**:
- TWS (Trader Workstation) or IB Gateway must be running
- Enable API access in TWS: File > API > Settings
- Check "Enable ActiveX and Socket Clients"

---

### 2. Polygon.io

Copy `polygon_config.yaml` and update with your API key:

```yaml
authentication:
  api_key: "YOUR_API_KEY_HERE"  # Get from https://polygon.io/dashboard
```

Python usage:
```python
from src.data_ingestion.polygon_provider import PolygonDataProvider

provider = PolygonDataProvider(api_key="your_api_key_here")
provider.authenticate()
```

**How to Get API Key**:
1. Visit https://polygon.io
2. Sign up for a free or paid account
3. Go to Dashboard > API Keys
4. Copy your API key

**Pricing Tiers**:
- Free: 5 requests/minute, ~2 years of history
- Starter: 30 requests/minute
- Professional: 600 requests/minute
- Enterprise: Custom limits

Adjust `rate_limiting.requests_per_second` based on your tier.

---

### 3. Databento

Copy `databento_config.yaml` and update with your credentials:

```yaml
authentication:
  api_key: "YOUR_API_KEY"
  client: "your_email@example.com"
  
dataset:
  default: "GLBX"  # CME futures
```

Python usage:
```python
from src.data_ingestion.databento_provider import DatabentoDataProvider

provider = DatabentoDataProvider(
    api_key="your_api_key",
    client="your_email@example.com"
)
provider.authenticate()
```

**How to Get Credentials**:
1. Visit https://databento.com
2. Sign up for an account
3. Go to Account Settings to get API key
4. Client ID is typically your registered email

**Available Datasets**:
- `GLBX` - CME/CBOT global derivatives (ES, NQ, YM, GC, CL, etc.)
- `XNAS` - NASDAQ equities
- `XNYS` - NYSE equities
- `OPRA` - Options data
- `CRYO/ERGO` - Cryptocurrency

---

## Environment Variables

For security, use environment variables instead of hardcoding credentials in configuration files:

### Polygon.io
```bash
export POLYGON_API_KEY="your_api_key_here"
```

```python
import os
api_key = os.getenv("POLYGON_API_KEY")
provider = PolygonDataProvider(api_key=api_key)
```

### Databento
```bash
export DATABENTO_API_KEY="your_api_key"
export DATABENTO_CLIENT="your_email@example.com"
```

```python
import os
provider = DatabentoDataProvider(
    api_key=os.getenv("DATABENTO_API_KEY"),
    client=os.getenv("DATABENTO_CLIENT")
)
```

---

## Quick Start Examples

### Example 1: Fetch Data from Interactive Brokers

```python
from src.data_ingestion.ib_provider import IBDataProvider
from datetime import datetime

# Create provider with your account
provider = IBDataProvider(
    account_id="DU123456",
    host="127.0.0.1",
    port=7497
)

# Authenticate (TWS/Gateway must be running)
provider.authenticate()

# Fetch ES futures daily data
df = provider.fetch_ohlcv(
    symbol="ES",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    timeframe="1D"
)

print(f"Fetched {len(df)} bars")
print(df.head())

provider.disconnect()
```

### Example 2: Fetch Data from Polygon.io

```python
from src.data_ingestion.polygon_provider import PolygonDataProvider
from datetime import datetime
import os

# Create provider
provider = PolygonDataProvider(
    api_key=os.getenv("POLYGON_API_KEY")
)

# Authenticate
provider.authenticate()

# Fetch Apple stock daily data
df = provider.fetch_ohlcv(
    symbol="AAPL",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2023, 12, 31),
    timeframe="1D"
)

print(f"Fetched {len(df)} bars of AAPL")
print(df.tail())

# Get available symbols
symbols = provider.get_available_symbols()
print(f"Available symbols: {symbols[:10]}")  # First 10
```

### Example 3: Fetch Data from Databento with Context Manager

```python
from src.data_ingestion.databento_provider import DatabentoDataProvider
from datetime import datetime
import os

# Use context manager for automatic cleanup
with DatabentoDataProvider(
    api_key=os.getenv("DATABENTO_API_KEY"),
    client=os.getenv("DATABENTO_CLIENT")
) as provider:
    # Fetch ES data
    df = provider.fetch_ohlcv(
        symbol="ES",
        start_date=datetime(2023, 6, 1),
        end_date=datetime(2023, 6, 30),
        timeframe="1H"
    )
    
    print(f"Fetched {len(df)} hourly bars")
    print(df.head())

# provider.disconnect() called automatically
```

---

## Configuration Best Practices

### 1. Security
- Never commit real credentials to version control
- Use environment variables for sensitive data
- Use `.gitignore` to exclude credential files:
  ```
  config/*.local.yaml
  config/*.prod.yaml
  .env
  ```

### 2. Rate Limiting
- Adjust `requests_per_second` based on your API plan
- Start conservative and increase if needed
- Monitor 429 (rate limit) errors in logs

### 3. Retry Policy
- Increase `max_retries` for unstable networks
- Increase `max_delay_seconds` for APIs with strict rate limits
- Use exponential backoff for transient errors

### 4. Testing
- Use default `seed` for reproducible synthetic data
- Change `seed` for different synthetic data variations
- Replace with real credentials for integration testing

---

## Troubleshooting

### Interactive Brokers
- **Error**: "Connection refused"
  - Ensure TWS or IB Gateway is running
  - Check host and port settings
  - Verify firewall allows connections

- **Error**: "Account not found"
  - Verify `account_id` matches your IB account
  - Check permissions for API access

### Polygon.io
- **Error**: "Invalid API key"
  - Verify API key from dashboard
  - Check key has not expired
  - Ensure API key is copied correctly (no spaces)

- **Error**: "429 Too Many Requests"
  - Reduce `requests_per_second`
  - Upgrade to higher tier for more requests
  - Wait longer between requests

### Databento
- **Error**: "Authentication failed"
  - Verify API key is correct
  - Check client ID (email) is registered
  - Ensure subscription to required datasets

- **Error**: "Dataset not available"
  - Verify dataset is in `enabled` list
  - Check subscription includes dataset
  - Use different dataset (e.g., XNAS instead of GLBX)

---

## References

- [Full Adapter Implementation Guide](../adapter_implementation_guide.md)
- [Data Provider Interface Specification](../data_provider_interface.md)
- [Interactive Brokers API Documentation](https://interactivebrokers.com/api)
- [Polygon.io API Docs](https://polygon.io/docs)
- [Databento API Docs](https://databento.com/docs)
