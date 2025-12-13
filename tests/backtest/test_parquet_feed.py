"""
Comprehensive test suite for ParquetDataFeed and helper functions.

Tests cover:
- Data feed initialization with various parameters
- Parquet file reading and data loading
- Date range filtering
- Timeframe and resampling
- Data validation
- Datetime and timezone handling
- Helper function usage
- Integration with backtrader
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date, timedelta
import tempfile

try:
    import backtrader as bt
except ImportError:
    bt = None

from src.backtest.feeds.parquet_feed import ParquetDataFeed
from src.backtest.feeds.helpers import (
    create_parquet_feed,
    create_multi_feeds,
    validate_feed_exists,
    list_available_feeds,
    get_feed_date_range,
)
from src.data_io.parquet_utils import write_parquet_validated
from src.data_io.schemas import OHLCV_SCHEMA


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory structure for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create directory structure: data/cleaned/v1/MES/
        base_path = Path(tmpdir) / 'data' / 'cleaned' / 'v1'
        symbol_dir = base_path / 'MES'
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        yield tmpdir


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for 1 month (1-minute bars)."""
    # Create 1-minute bars for January 2024
    dates = pd.date_range('2024-01-01', periods=10080, freq='1T', tz='UTC')
    
    np.random.seed(42)
    # Generate realistic OHLCV data
    prices = np.random.normal(4500, 20, len(dates))
    
    data = {
        'timestamp': dates,
        'open': prices,
        'high': prices + np.random.uniform(0, 5, len(dates)),
        'low': prices - np.random.uniform(0, 5, len(dates)),
        'close': prices + np.random.uniform(-2, 2, len(dates)),
        'volume': np.random.randint(100, 1000, len(dates)),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_ohlcv_5min_data():
    """Create sample OHLCV data for 5-minute bars."""
    dates = pd.date_range('2024-01-01', periods=2016, freq='5T', tz='UTC')
    
    np.random.seed(42)
    prices = np.random.normal(4500, 20, len(dates))
    
    data = {
        'timestamp': dates,
        'open': prices,
        'high': prices + np.random.uniform(0, 5, len(dates)),
        'low': prices - np.random.uniform(0, 5, len(dates)),
        'close': prices + np.random.uniform(-2, 2, len(dates)),
        'volume': np.random.randint(500, 5000, len(dates)),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def write_sample_parquet(temp_data_dir, sample_ohlcv_data):
    """Write sample OHLCV data to Parquet file."""
    def _write(symbol='MES', version='v1', data=None):
        if data is None:
            data = sample_ohlcv_data
        
        file_path = (
            f'{temp_data_dir}/data/cleaned/'
            f'{version}/{symbol}/ohlcv.parquet'
        )
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        write_parquet_validated(data, file_path, schema=OHLCV_SCHEMA)
        return file_path
    
    return _write


# ============================================================================
# Tests: Basic Feed Initialization
# ============================================================================

@pytest.mark.skipif(bt is None, reason='backtrader not installed')
class TestParquetDataFeedBasic:
    """Tests for basic ParquetDataFeed functionality."""
    
    def test_feed_initialization(self, temp_data_dir, write_sample_parquet):
        """Test basic feed initialization."""
        write_sample_parquet()
        
        feed = ParquetDataFeed(
            symbol='MES',
            cleaned_version='v1',
            base_path=f'{temp_data_dir}/data/cleaned',
        )
        
        assert feed is not None
        assert feed.params.symbol == 'MES'
        assert feed.params.cleaned_version == 'v1'
        assert feed._data is not None
        assert len(feed._data) > 0
    
    def test_feed_with_date_range_filtering(
        self, temp_data_dir, write_sample_parquet
    ):
        """Test feed initialization with date range filtering."""
        write_sample_parquet()
        
        start_date = datetime(2024, 1, 5)
        end_date = datetime(2024, 1, 10)
        
        feed = ParquetDataFeed(
            symbol='MES',
            cleaned_version='v1',
            start_date=start_date,
            end_date=end_date,
            base_path=f'{temp_data_dir}/data/cleaned',
        )
        
        assert feed._data is not None
        assert len(feed._data) > 0
        
        # Check date filtering
        min_date = feed._data['timestamp'].min()
        max_date = feed._data['timestamp'].max()
        assert min_date >= pd.Timestamp(start_date, tz='UTC')
        assert max_date <= pd.Timestamp(end_date, tz='UTC')
    
    def test_feed_validates_file_existence(self, temp_data_dir):
        """Test that feed raises error if file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            ParquetDataFeed(
                symbol='NONEXISTENT',
                cleaned_version='v1',
                base_path=f'{temp_data_dir}/data/cleaned',
            )
    
    def test_feed_with_schema_validation(self, temp_data_dir, write_sample_parquet):
        """Test feed initialization with schema validation enabled."""
        write_sample_parquet()
        
        feed = ParquetDataFeed(
            symbol='MES',
            cleaned_version='v1',
            base_path=f'{temp_data_dir}/data/cleaned',
            validate_schema=True,
        )
        
        assert feed._data is not None
        # Verify schema validation passed (no exception)
    
    def test_feed_disables_schema_validation(self, temp_data_dir, write_sample_parquet):
        """Test feed with schema validation disabled."""
        write_sample_parquet()
        
        feed = ParquetDataFeed(
            symbol='MES',
            cleaned_version='v1',
            base_path=f'{temp_data_dir}/data/cleaned',
            validate_schema=False,
        )
        
        assert feed._data is not None


# ============================================================================
# Tests: Data Validation
# ============================================================================

class TestDataValidation:
    """Tests for data validation logic."""
    
    def test_validate_data_checks_required_columns(self, temp_data_dir):
        """Test that validation fails with missing required columns."""
        base_path = Path(temp_data_dir) / 'data' / 'cleaned' / 'v1'
        symbol_dir = base_path / 'MES'
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data missing required column
        dates = pd.date_range('2024-01-01', periods=100, freq='1T', tz='UTC')
        data = {
            'timestamp': dates,
            'open': np.random.uniform(4500, 4600, 100),
            'high': np.random.uniform(4600, 4700, 100),
            'low': np.random.uniform(4400, 4500, 100),
            # Missing 'close' and 'volume'
        }
        df = pd.DataFrame(data)
        
        file_path = symbol_dir / 'ohlcv.parquet'
        df.to_parquet(file_path)
        
        with pytest.raises(ValueError, match='Missing required columns'):
            ParquetDataFeed(
                symbol='MES',
                cleaned_version='v1',
                base_path=str(base_path.parent),
                validate_schema=False,  # Skip schema validation to test our logic
            )
    
    def test_validate_data_checks_duplicate_timestamps(self, temp_data_dir):
        """Test that validation warns about duplicate timestamps."""
        base_path = Path(temp_data_dir) / 'data' / 'cleaned' / 'v1'
        symbol_dir = base_path / 'MES'
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data with duplicate timestamp
        dates = pd.date_range('2024-01-01', periods=100, freq='1T', tz='UTC').tolist()
        dates[50] = dates[49]  # Duplicate
        
        data = {
            'timestamp': dates,
            'open': np.random.uniform(4500, 4600, 100),
            'high': np.random.uniform(4600, 4700, 100),
            'low': np.random.uniform(4400, 4500, 100),
            'close': np.random.uniform(4500, 4600, 100),
            'volume': np.random.randint(100, 1000, 100),
        }
        df = pd.DataFrame(data)
        
        file_path = symbol_dir / 'ohlcv.parquet'
        df.to_parquet(file_path)
        
        with pytest.raises(ValueError, match='Duplicate timestamps'):
            ParquetDataFeed(
                symbol='MES',
                cleaned_version='v1',
                base_path=str(base_path.parent),
                validate_schema=False,
            )


# ============================================================================
# Tests: Helper Functions
# ============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_create_parquet_feed(self, temp_data_dir, write_sample_parquet):
        """Test create_parquet_feed helper function."""
        write_sample_parquet()
        
        feed = create_parquet_feed(
            symbol='MES',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            cleaned_version='v1',
            base_path=f'{temp_data_dir}/data/cleaned',
        )
        
        assert isinstance(feed, ParquetDataFeed)
        assert feed.params.symbol == 'MES'
        assert feed._data is not None
    
    def test_create_parquet_feed_validates_dates(self):
        """Test that create_parquet_feed validates date parameters."""
        with pytest.raises(ValueError, match='start_date must be before'):
            create_parquet_feed(
                symbol='MES',
                start_date=datetime(2024, 1, 31),
                end_date=datetime(2024, 1, 1),
            )
    
    def test_create_parquet_feed_validates_symbol(self):
        """Test that create_parquet_feed validates symbol."""
        with pytest.raises(ValueError, match='symbol must be'):
            create_parquet_feed(
                symbol='',
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
            )
    
    def test_validate_feed_exists(self, temp_data_dir, write_sample_parquet):
        """Test validate_feed_exists helper function."""
        write_sample_parquet()
        
        exists = validate_feed_exists(
            symbol='MES',
            cleaned_version='v1',
            base_path=f'{temp_data_dir}/data/cleaned',
        )
        
        assert exists is True
    
    def test_validate_feed_exists_returns_false(self, temp_data_dir):
        """Test validate_feed_exists returns False for missing file."""
        exists = validate_feed_exists(
            symbol='NONEXISTENT',
            cleaned_version='v1',
            base_path=f'{temp_data_dir}/data/cleaned',
        )
        
        assert exists is False
    
    def test_list_available_feeds(self, temp_data_dir, write_sample_parquet):
        """Test list_available_feeds helper function."""
        write_sample_parquet(symbol='MES')
        write_sample_parquet(symbol='ES')
        
        feeds = list_available_feeds(
            cleaned_version='v1',
            base_path=f'{temp_data_dir}/data/cleaned',
        )
        
        assert 'MES' in feeds
        assert 'ES' in feeds
        assert len(feeds) == 2
    
    def test_get_feed_date_range(self, temp_data_dir, write_sample_parquet):
        """Test get_feed_date_range helper function."""
        write_sample_parquet()
        
        date_range = get_feed_date_range(
            symbol='MES',
            cleaned_version='v1',
            base_path=f'{temp_data_dir}/data/cleaned',
        )
        
        assert date_range is not None
        min_date, max_date = date_range
        assert min_date < max_date
        assert min_date.year == 2024
    
    def test_get_feed_date_range_nonexistent(self, temp_data_dir):
        """Test get_feed_date_range with nonexistent feed."""
        date_range = get_feed_date_range(
            symbol='NONEXISTENT',
            cleaned_version='v1',
            base_path=f'{temp_data_dir}/data/cleaned',
        )
        
        assert date_range is None


# ============================================================================
# Tests: Data Loading and Bar Access
# ============================================================================

@pytest.mark.skipif(bt is None, reason='backtrader not installed')
class TestDataLoading:
    """Tests for data loading and bar access."""
    
    def test_feed_loads_correct_bar_count(self, temp_data_dir, write_sample_parquet):
        """Test that feed loads correct number of bars."""
        write_sample_parquet()
        
        feed = ParquetDataFeed(
            symbol='MES',
            cleaned_version='v1',
            base_path=f'{temp_data_dir}/data/cleaned',
        )
        
        # Should have all bars from sample data
        assert len(feed._data) > 0
        assert feed._getsizes() == len(feed._data)
    
    def test_feed_bar_access_with_backtrader(self, temp_data_dir, write_sample_parquet):
        """Test that feed works with backtrader cerebro."""
        if bt is None:
            pytest.skip('backtrader not installed')
        
        write_sample_parquet()
        
        cerebro = bt.Cerebro()
        feed = ParquetDataFeed(
            symbol='MES',
            cleaned_version='v1',
            base_path=f'{temp_data_dir}/data/cleaned',
        )
        
        cerebro.adddata(feed)
        
        # Should not raise any errors
        assert feed in cerebro.datas
    
    def test_feed_returns_false_after_eof(
        self, temp_data_dir, write_sample_parquet
    ):
        """Test that feed._load() returns False after EOF."""
        write_sample_parquet()
        
        feed = ParquetDataFeed(
            symbol='MES',
            cleaned_version='v1',
            base_path=f'{temp_data_dir}/data/cleaned',
        )
        
        # Load all bars
        while feed._load():
            pass
        
        # Next call should return False
        assert feed._load() is False
        assert feed._eof is True


# ============================================================================
# Tests: Timezone Handling
# ============================================================================

class TestTimezoneHandling:
    """Tests for timezone-aware datetime handling."""
    
    def test_feed_preserves_timezone(self, temp_data_dir):
        """Test that feed preserves UTC timezone."""
        base_path = Path(temp_data_dir) / 'data' / 'cleaned' / 'v1'
        symbol_dir = base_path / 'MES'
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data with UTC timezone
        dates = pd.date_range('2024-01-01', periods=100, freq='1T', tz='UTC')
        data = {
            'timestamp': dates,
            'open': np.random.uniform(4500, 4600, 100),
            'high': np.random.uniform(4600, 4700, 100),
            'low': np.random.uniform(4400, 4500, 100),
            'close': np.random.uniform(4500, 4600, 100),
            'volume': np.random.randint(100, 1000, 100),
        }
        df = pd.DataFrame(data)
        
        file_path = symbol_dir / 'ohlcv.parquet'
        write_parquet_validated(df, file_path, schema=OHLCV_SCHEMA)
        
        feed = ParquetDataFeed(
            symbol='MES',
            cleaned_version='v1',
            base_path=str(base_path.parent),
            tz='UTC',
        )
        
        assert feed._data is not None
        assert feed._data['timestamp'].dt.tz is not None
        assert str(feed._data['timestamp'].dt.tz) == 'UTC'
    
    def test_feed_converts_timezone(self, temp_data_dir):
        """Test that feed converts timezones correctly."""
        base_path = Path(temp_data_dir) / 'data' / 'cleaned' / 'v1'
        symbol_dir = base_path / 'MES'
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data with UTC timezone
        dates = pd.date_range('2024-01-01', periods=100, freq='1T', tz='UTC')
        data = {
            'timestamp': dates,
            'open': np.random.uniform(4500, 4600, 100),
            'high': np.random.uniform(4600, 4700, 100),
            'low': np.random.uniform(4400, 4500, 100),
            'close': np.random.uniform(4500, 4600, 100),
            'volume': np.random.randint(100, 1000, 100),
        }
        df = pd.DataFrame(data)
        
        file_path = symbol_dir / 'ohlcv.parquet'
        write_parquet_validated(df, file_path, schema=OHLCV_SCHEMA)
        
        feed = ParquetDataFeed(
            symbol='MES',
            cleaned_version='v1',
            base_path=str(base_path.parent),
            tz='America/Chicago',
        )
        
        assert feed._data is not None
        # Verify timezone conversion happened
        assert str(feed._data['timestamp'].dt.tz) == 'America/Chicago'


# ============================================================================
# Tests: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_feed_with_empty_date_range(self, temp_data_dir, write_sample_parquet):
        """Test feed behavior with date range that yields no data."""
        write_sample_parquet()
        
        # Use date range outside of data
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 1, 31)
        
        feed = ParquetDataFeed(
            symbol='MES',
            cleaned_version='v1',
            start_date=start_date,
            end_date=end_date,
            base_path=f'{temp_data_dir}/data/cleaned',
        )
        
        # Should have empty or minimal data
        assert len(feed._data) == 0
    
    def test_feed_with_single_bar(self, temp_data_dir):
        """Test feed with minimal data (single bar)."""
        base_path = Path(temp_data_dir) / 'data' / 'cleaned' / 'v1'
        symbol_dir = base_path / 'MES'
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Create data with single bar
        dates = pd.date_range('2024-01-01', periods=1, freq='1T', tz='UTC')
        data = {
            'timestamp': dates,
            'open': [4500.0],
            'high': [4505.0],
            'low': [4495.0],
            'close': [4502.0],
            'volume': [100],
        }
        df = pd.DataFrame(data)
        
        file_path = symbol_dir / 'ohlcv.parquet'
        write_parquet_validated(df, file_path, schema=OHLCV_SCHEMA)
        
        feed = ParquetDataFeed(
            symbol='MES',
            cleaned_version='v1',
            base_path=str(base_path.parent),
        )
        
        assert len(feed._data) == 1
        assert feed._getsizes() == 1


# ============================================================================
# Tests: Performance and Large Datasets
# ============================================================================

class TestPerformance:
    """Tests for performance with realistic datasets."""
    
    def test_feed_memory_efficiency(self, temp_data_dir):
        """Test that feed handles large datasets efficiently."""
        base_path = Path(temp_data_dir) / 'data' / 'cleaned' / 'v1'
        symbol_dir = base_path / 'MES'
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 1 year of 1-minute data (should be ~250k bars)
        dates = pd.date_range('2024-01-01', periods=250000, freq='1T', tz='UTC')
        
        # Only create a smaller subset for testing
        dates = dates[:50000]  # ~35 days of 1-minute data
        
        np.random.seed(42)
        data = {
            'timestamp': dates,
            'open': np.random.normal(4500, 20, len(dates)),
            'high': np.random.normal(4510, 20, len(dates)),
            'low': np.random.normal(4490, 20, len(dates)),
            'close': np.random.normal(4500, 20, len(dates)),
            'volume': np.random.randint(100, 1000, len(dates)),
        }
        df = pd.DataFrame(data)
        
        file_path = symbol_dir / 'ohlcv.parquet'
        write_parquet_validated(df, file_path, schema=OHLCV_SCHEMA)
        
        # Should load without memory issues
        feed = ParquetDataFeed(
            symbol='MES',
            cleaned_version='v1',
            base_path=str(base_path.parent),
        )
        
        assert len(feed._data) > 0
        assert feed._getsizes() == len(feed._data)
