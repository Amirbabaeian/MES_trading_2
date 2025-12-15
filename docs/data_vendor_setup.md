# Data Vendor Credential Setup Guide

This guide provides step-by-step instructions for setting up credentials for each supported data vendor. Following these instructions, you should be able to set up any vendor in under 10 minutes.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Interactive Brokers (IB)](#interactive-brokers-ib)
3. [Polygon.io](#polygonio)
4. [Databento](#databento)
5. [Credential Storage Options](#credential-storage-options)
6. [Testing Your Credentials](#testing-your-credentials)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

The credential system supports multiple storage methods:

1. **Environment Variables** (Recommended for production)
   - Store credentials in a `.env` file
   - Supported: IB, Polygon, Databento

2. **Configuration Files** (Recommended for development)
   - Store credentials in `config/credentials.json`
   - Supported: JSON format with vendor-specific sections

3. **Programmatically** (For advanced use)
   - Pass credentials directly to provider constructors

### Initialize Your Credentials

```bash
# Copy the example files to your actual files
cp .env.example .env
cp config/credentials.example.json config/credentials.json

# Edit .env or config/credentials.json with your actual credentials
# WARNING: Never commit .env or credentials.json to version control!
```

---

## Interactive Brokers (IB)

Interactive Brokers provides access to stocks, futures, options, forex, and more through their Trader Workstation (TWS) or Gateway application.

### Requirements

- **Account**: Interactive Brokers account (requires funding)
- **Software**: TWS (Trader Workstation) or IB Gateway installed and running
- **Time**: ~5 minutes to set up

### Step 1: Get Your IB Credentials

1. **Obtain Account Number**
   - Log in to your IB account at https://www.interactivebrokers.com
   - Go to Account Management → Account Settings
   - Your account number is displayed (e.g., `DU123456`)

2. **Set Your Password**
   - Use your IB login password (or create an API password if your broker supports it)
   - Ensure your account is configured to allow API connections

### Step 2: Install and Configure TWS/Gateway

1. **Download TWS or IB Gateway**
   - TWS: https://www.interactivebrokers.com/en/trading/platforms/trader-workstation.php
   - IB Gateway (lighter, recommended): https://www.interactivebrokers.com/en/trading/platforms/ib-gateway.php

2. **Configure for API Connections**
   - Open TWS/Gateway
   - Go to Edit → Settings → API
   - Enable the API and configure:
     - **Socket Port**: `7497` (paper trading) or `7496` (live trading)
     - **Enable read-only API**: Unchecked (we need write access for orders)
     - **Allow connections from IP**: Ensure `127.0.0.1` is allowed

3. **Keep TWS/Gateway Running**
   - TWS/Gateway must be running for API connections to work
   - Consider setting it to start automatically on system startup

### Step 3: Configure Credentials

**Option A: Environment Variables (.env)**

Edit your `.env` file:

```bash
IB_HOST=127.0.0.1
IB_PORT=7497           # 7497 for paper, 7496 for live
IB_CLIENT_ID=1         # Must be unique (1-2000)
IB_ACCOUNT=DU123456    # Your account number
IB_PASSWORD=your_password
```

**Option B: Configuration File**

Edit `config/credentials.json`:

```json
{
  "ib": {
    "host": "127.0.0.1",
    "port": 7497,
    "client_id": 1,
    "account": "DU123456",
    "password": "your_password"
  }
}
```

### Step 4: Test Your Connection

```python
from src.data_ingestion import CredentialManager

# Load and test credentials
manager = CredentialManager()
try:
    creds = manager.get_vendor_credentials('ib', test_connection=True)
    print("✓ Interactive Brokers connection successful!")
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

### Security Notes for IB

- Never commit `.env` file with real credentials to version control
- Use environment variables in production, not config files
- IB doesn't require a real API key; credentials are your account credentials
- For additional security, consider creating a dedicated API-only account with limited permissions

---

## Polygon.io

Polygon.io provides REST API access to stock market data including stocks, ETFs, options, forex, and cryptocurrency.

### Requirements

- **Account**: Free Polygon.io account
- **API Key**: Available after registration
- **Time**: ~2 minutes to set up

### Step 1: Create a Polygon.io Account

1. **Sign Up**
   - Go to https://polygon.io
   - Click "Sign Up" (or "Start Free" on pricing page)
   - Use your email address

2. **Verify Email**
   - Check your email inbox for verification link
   - Click the link to activate your account

3. **Log In**
   - Go to https://polygon.io/dashboard
   - Log in with your email and password

### Step 2: Get Your API Key

1. **Navigate to API Keys**
   - In the Polygon.io dashboard, click on "API Keys" in the left sidebar
   - Your default API key is displayed

2. **Copy Your API Key**
   - The key starts with `pk_` and is a long alphanumeric string
   - Example: `pk_aHR0cHM6Ly9hcGkucG9seWdvbi5pby8...`
   - Keep this key secret!

### Step 3: Understand Rate Limits

Different subscription tiers have different rate limits:

- **Free Tier**: 5 API calls per minute
- **Starter**: $99/month - 30 calls per minute
- **Professional**: $499/month - 600 calls per minute
- **Enterprise**: Custom limits

See your account dashboard for your current tier and limits.

### Step 4: Configure Credentials

**Option A: Environment Variables (.env)**

Edit your `.env` file:

```bash
POLYGON_API_KEY=pk_your_api_key_here
```

**Option B: Configuration File**

Edit `config/credentials.json`:

```json
{
  "polygon": {
    "api_key": "pk_your_api_key_here"
  }
}
```

### Step 5: Test Your Connection

```python
from src.data_ingestion import CredentialManager

manager = CredentialManager()
try:
    creds = manager.get_vendor_credentials('polygon', test_connection=True)
    print("✓ Polygon.io connection successful!")
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

### Security Notes for Polygon

- Your API key grants access to your Polygon.io account
- Never share your API key publicly
- Never commit `.env` file with real credentials to version control
- Use environment variables in production, not config files
- You can rotate API keys in your account settings if compromised

### API Limitations

- **Historical Data**: Available based on your subscription tier
- **Real-Time Data**: Not available through REST API (requires upgrade)
- **Rate Limiting**: Enforced per-minute based on tier
- **Data Adjustments**: Polygon applies dividend and split adjustments by default

---

## Databento

Databento provides high-quality market data for equities, futures, options, and cryptocurrency through REST and native streaming protocols.

### Requirements

- **Account**: Free Databento community account or paid plan
- **API Key**: Available after registration
- **Time**: ~3 minutes to set up

### Step 1: Create a Databento Account

1. **Sign Up**
   - Go to https://databento.com
   - Click "Try Databento" or go to login page
   - Sign up with email, GitHub, or other options

2. **Verify Your Account**
   - Check your email for verification
   - Complete any required profile setup

3. **Log In**
   - Go to https://databento.com (dashboard available after login)

### Step 2: Get Your API Key

1. **Navigate to API Credentials**
   - In your Databento dashboard, find "Settings" or "API Keys"
   - Your API key should be displayed

2. **Copy Your API Key**
   - The key typically starts with `db_` 
   - Example: `db_Ym9iLnNtaXRoQGV4YW1wbGUuY29tOm1hc3Rlcg`
   - Keep this key secret!

### Step 3: Understand Your Plan

Databento offers different tiers:

- **Community**: Free tier with limited symbols and historical depth
- **Basic**: $299/month - Broad market coverage
- **Professional**: $999+/month - Full market coverage
- **Enterprise**: Custom pricing

Check your account page for current plan limits.

### Step 4: Configure Credentials

**Option A: Environment Variables (.env)**

Edit your `.env` file:

```bash
DATABENTO_API_KEY=db_your_api_key_here
```

**Option B: Configuration File**

Edit `config/credentials.json`:

```json
{
  "databento": {
    "api_key": "db_your_api_key_here"
  }
}
```

### Step 5: Test Your Connection

```python
from src.data_ingestion import CredentialManager

manager = CredentialManager()
try:
    creds = manager.get_vendor_credentials('databento', test_connection=True)
    print("✓ Databento connection successful!")
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

### Security Notes for Databento

- Your API key grants access to all your Databento data
- Never share your API key publicly
- Never commit `.env` file with real credentials to version control
- Use environment variables in production
- You can generate multiple API keys and revoke old ones in settings

### Data Coverage

- **Equities**: US stocks and ETFs
- **Futures**: CME and other exchanges
- **Options**: Standard options contracts
- **Crypto**: Major cryptocurrency pairs
- **Historical Data**: Varies by plan and symbol

---

## Credential Storage Options

### 1. Environment Variables (.env File)

**Best for**: Development and quick testing

```bash
# .env file
POLYGON_API_KEY=pk_...
DATABENTO_API_KEY=db_...
IB_HOST=127.0.0.1
```

**Pros:**
- Simple and quick
- No file parsing needed
- Works across languages/frameworks

**Cons:**
- Less structured than config files
- Not ideal for multiple environments
- Can leak in environment printouts

### 2. JSON Configuration File

**Best for**: Development with multiple vendors

```json
{
  "ib": {
    "host": "127.0.0.1",
    "port": 7497,
    "account": "DU123456",
    "password": "..."
  },
  "polygon": {
    "api_key": "pk_..."
  },
  "databento": {
    "api_key": "db_..."
  }
}
```

**Pros:**
- Structured and organized
- Supports all vendor configurations
- Easy to validate schema

**Cons:**
- Must keep out of version control
- Less secure than environment variables
- File needs to be found/loaded

### 3. Python Code (Not Recommended for Production)

```python
# Only for development/testing
from src.data_ingestion import IBDataProvider

provider = IBDataProvider(
    host="127.0.0.1",
    port=7497,
    account="DU123456",
    password="..."  # BAD! Never hardcode credentials
)
```

**Only use this for:**
- Testing with mock data
- Development environments
- Never in production code

---

## Testing Your Credentials

### Quick Test for All Vendors

```python
from src.data_ingestion import CredentialManager

manager = CredentialManager()

# Load credentials for a specific vendor
try:
    polygon_creds = manager.get_vendor_credentials('polygon')
    print("✓ Polygon credentials loaded")
except Exception as e:
    print(f"✗ Polygon error: {e}")

# Test all vendors (continues on failure)
all_creds = manager.get_all_vendor_credentials(test_connections=False)
print(f"Loaded credentials for: {list(all_creds.keys())}")
```

### Test Connection Before Using Provider

```python
from src.data_ingestion import CredentialManager

manager = CredentialManager()

# This loads credentials AND tests the connection
try:
    creds = manager.get_vendor_credentials(
        'polygon',
        test_connection=True  # Attempts actual connection
    )
    print("✓ Connection successful, ready to use")
except ConnectionError as e:
    print(f"✗ Connection test failed: {e}")
except ValueError as e:
    print(f"✗ Validation error: {e}")
```

### Validate Credentials Without Testing Connection

```python
from src.data_ingestion import (
    PolygonCredentialValidator,
    SensitiveDataMasker
)

validator = PolygonCredentialValidator()

creds = {'api_key': 'pk_...'}

try:
    validator.validate(creds)
    
    # Show credentials with sensitive data masked
    masked = SensitiveDataMasker.mask_dict(
        creds,
        validator.get_sensitive_fields()
    )
    print(f"✓ Credentials valid: {masked}")
    
except ValueError as e:
    print(f"✗ Validation error: {e}")
```

---

## Troubleshooting

### Issue: "No credentials found for [vendor]"

**Solution:**
1. Verify environment variables are set correctly:
   ```bash
   echo $POLYGON_API_KEY  # Should print your API key
   ```
2. Or verify config file exists at `config/credentials.json`
3. Check file paths are correct if using custom config file

### Issue: "Missing required fields: [field names]"

**Solution:**
- Check you've provided all required fields for the vendor
- For IB: host, port, client_id (account/password optional for mock mode)
- For Polygon/Databento: api_key
- Review the `.env.example` or `config/credentials.example.json` for required fields

### Issue: "Interactive Brokers connection test failed"

**Solution:**
1. Verify TWS or IB Gateway is running
2. Check the port is correct (7497 for paper, 7496 for live)
3. Verify API connections are enabled in TWS/Gateway settings
4. Ensure your account allows API connections
5. Check firewall rules allow localhost connections
6. Verify client_id is unique (1-2000)

### Issue: "Polygon.io connection test failed"

**Solution:**
1. Verify API key is correct (should start with `pk_`)
2. Ensure API key is not expired or revoked in dashboard
3. Check your internet connection
4. Verify you have an active Polygon.io account
5. Check for any account suspension or quota limits

### Issue: "Databento connection test failed"

**Solution:**
1. Verify API key is correct (usually starts with `db_`)
2. Ensure API key is not revoked in account settings
3. Check your internet connection
4. Verify your Databento account is active
5. Check your subscription tier allows API access

### Issue: Credentials visible in logs or error messages

**Solution:**
- The system automatically masks sensitive credentials in logs
- If you see unmasked credentials, you may be using them directly
- Always use `CredentialManager` or `SensitiveDataMasker` for handling credentials
- Never print credential objects directly: `print(creds)` will expose them
- Use masking: `masked = SensitiveDataMasker.mask_dict(creds, sensitive_fields)`

---

## Security Best Practices

1. **Never Commit Credentials**
   ```bash
   # Add to .gitignore
   .env
   config/credentials.json
   config/credentials.*.json
   ```

2. **Use Environment Variables in Production**
   - Store credentials in your deployment environment variables
   - Use CI/CD secrets management (GitHub Secrets, GitLab CI/CD, etc.)

3. **Rotate Credentials Regularly**
   - Change passwords every 90 days
   - Rotate API keys every 6-12 months
   - Immediately rotate if compromised

4. **Use Strong Passwords**
   - For IB: Use a strong, unique password
   - Enable 2FA on all vendor accounts

5. **Limit Permissions**
   - IB: Use API-only accounts with minimal permissions
   - Polygon/Databento: Use read-only API keys if available

6. **Monitor Account Activity**
   - Check vendor dashboards for unusual activity
   - Review API usage logs periodically
   - Set up alerts for quota limits

---

## Getting Help

If you encounter issues beyond this guide:

1. **Check the vendor documentation**
   - IB: https://ibkr.info/article/2170
   - Polygon: https://polygon.io/docs/stocks/getting-started
   - Databento: https://docs.databento.com/

2. **Review project issues**
   - Check GitHub issues for similar problems
   - Search existing discussions

3. **Verify requirements**
   - Ensure you meet all account requirements for each vendor
   - Verify your system meets technical requirements
   - Check you have sufficient permissions/subscriptions

4. **Enable debug logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

---

## Summary

You now have all the information needed to set up credentials for any supported vendor:

- **Interactive Brokers**: 5-10 minutes (requires TWS/Gateway)
- **Polygon.io**: 2-3 minutes
- **Databento**: 2-3 minutes

After setup, test your credentials with the included testing utilities to verify everything is working correctly.
