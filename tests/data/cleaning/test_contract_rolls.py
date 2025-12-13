"""
Comprehensive test suite for contract rolls functionality.

Tests cover:
- Contract metadata (symbols, expirations, specs)
- Roll detection in OHLCV data
- Backward ratio adjustment
- Panama canal method
- Price continuity verification
- Edge cases (first bar at roll, no rolls, missing data)
- Audit trail and metadata tracking
"""

import pytest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List

from src.data.cleaning.contract_rolls import (
    RollPoint,
    AdjustmentMethod,
    get_contract_spec,
    get_expiration_months,
    get_contract_symbol,
    parse_contract_symbol,
    calculate_expiration_date,
    detect_contract_rolls,
    identify_active_contract,
    backward_ratio_adjustment,
    panama_canal_method,
    verify_price_continuity,
    get_supported_assets,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_es_data_single_roll():
    """Create sample ES data with a single contract roll."""
    # Create data for ESZ23 (Dec 2023) and ESH24 (March 2024)
    dates_z23 = pd.date_range("2024-01-01", periods=50, freq="D", tz="US/Eastern")
    dates_h24 = pd.date_range("2024-03-18", periods=50, freq="D", tz="US/Eastern")
    
    # Old contract (closing prices 100-105)
    prices_z23 = np.linspace(100, 105, 50)
    
    # New contract at roll point
    # If old close is ~105, and we roll with ratio 1.05, new contract starts at ~100
    prices_h24 = np.linspace(100, 105, 50)
    
    df = pd.DataFrame({
        'timestamp': pd.concat([
            pd.Series(dates_z23),
            pd.Series(dates_h24)
        ]).reset_index(drop=True),
        'contract': ['ESZ23'] * 50 + ['ESH24'] * 50,
        'open': np.concatenate([prices_z23 - 0.5, prices_h24 - 0.5]),
        'high': np.concatenate([prices_z23 + 1, prices_h24 + 1]),
        'low': np.concatenate([prices_z23 - 1, prices_h24 - 1]),
        'close': np.concatenate([prices_z23, prices_h24]),
        'volume': np.random.randint(1000, 10000, 100),
    })
    
    return df


@pytest.fixture
def sample_es_data_multiple_rolls():
    """Create sample ES data with multiple contract rolls."""
    # ESZ23 -> ESH24 -> ESM24 -> ESU24
    dates_z23 = pd.date_range("2024-01-01", periods=30, freq="D", tz="US/Eastern")
    dates_h24 = pd.date_range("2024-03-18", periods=30, freq="D", tz="US/Eastern")
    dates_m24 = pd.date_range("2024-06-17", periods=30, freq="D", tz="US/Eastern")
    dates_u24 = pd.date_range("2024-09-16", periods=30, freq="D", tz="US/Eastern")
    
    # Different price levels for each contract
    prices_z23 = np.linspace(100, 102, 30)
    prices_h24 = np.linspace(100, 102, 30)  # Same level after adjustment
    prices_m24 = np.linspace(100, 102, 30)
    prices_u24 = np.linspace(100, 102, 30)
    
    df = pd.DataFrame({
        'timestamp': pd.concat([
            pd.Series(dates_z23),
            pd.Series(dates_h24),
            pd.Series(dates_m24),
            pd.Series(dates_u24)
        ]).reset_index(drop=True),
        'contract': ['ESZ23'] * 30 + ['ESH24'] * 30 + ['ESM24'] * 30 + ['ESU24'] * 30,
        'open': np.concatenate([prices_z23 - 0.5, prices_h24 - 0.5, prices_m24 - 0.5, prices_u24 - 0.5]),
        'high': np.concatenate([prices_z23 + 1, prices_h24 + 1, prices_m24 + 1, prices_u24 + 1]),
        'low': np.concatenate([prices_z23 - 1, prices_h24 - 1, prices_m24 - 1, prices_u24 - 1]),
        'close': np.concatenate([prices_z23, prices_h24, prices_m24, prices_u24]),
        'volume': np.random.randint(1000, 10000, 120),
    })
    
    return df


@pytest.fixture
def sample_vix_data_single_roll():
    """Create sample VIX data with a single contract roll."""
    # VIXF24 (Jan 2024) and VIXG24 (Feb 2024)
    dates_f24 = pd.date_range("2024-01-01", periods=20, freq="D", tz="US/Eastern")
    dates_g24 = pd.date_range("2024-02-01", periods=20, freq="D", tz="US/Eastern")
    
    prices_f24 = np.linspace(15, 20, 20)
    prices_g24 = np.linspace(15, 20, 20)
    
    df = pd.DataFrame({
        'timestamp': pd.concat([
            pd.Series(dates_f24),
            pd.Series(dates_g24)
        ]).reset_index(drop=True),
        'contract': ['VIXF24'] * 20 + ['VIXG24'] * 20,
        'open': np.concatenate([prices_f24 - 0.5, prices_g24 - 0.5]),
        'high': np.concatenate([prices_f24 + 1, prices_g24 + 1]),
        'low': np.concatenate([prices_f24 - 1, prices_g24 - 1]),
        'close': np.concatenate([prices_f24, prices_g24]),
        'volume': np.random.randint(100, 1000, 40),
    })
    
    return df


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame."""
    return pd.DataFrame({
        'timestamp': pd.DatetimeIndex([], tz='US/Eastern'),
        'contract': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': [],
    })


# ============================================================================
# Tests: Contract Metadata
# ============================================================================

class TestContractMetadata:
    """Tests for contract metadata functions."""
    
    def test_get_supported_assets(self):
        """Test that we can get supported assets."""
        assets = get_supported_assets()
        assert 'ES' in assets
        assert 'MES' in assets
        assert 'VIX' in assets
    
    def test_get_contract_spec_es(self):
        """Test getting ES contract specification."""
        spec = get_contract_spec('ES')
        assert spec['name'] == 'E-mini S&P 500 Futures'
        assert spec['expiration_cycle'] == 'quarterly'
        assert spec['multiplier'] == 50.0
    
    def test_get_contract_spec_mes(self):
        """Test getting MES contract specification."""
        spec = get_contract_spec('MES')
        assert spec['name'] == 'E-mini S&P 500 Micro Futures'
        assert spec['expiration_cycle'] == 'quarterly'
        assert spec['multiplier'] == 5.0
    
    def test_get_contract_spec_vix(self):
        """Test getting VIX contract specification."""
        spec = get_contract_spec('VIX')
        assert spec['name'] == 'Volatility Index Futures'
        assert spec['expiration_cycle'] == 'monthly'
        assert spec['multiplier'] == 100.0
    
    def test_get_contract_spec_invalid_asset(self):
        """Test that invalid asset raises error."""
        with pytest.raises(ValueError):
            get_contract_spec('INVALID')
    
    def test_get_expiration_months_es(self):
        """Test ES expiration months."""
        months = get_expiration_months('ES')
        assert months == [3, 6, 9, 12]  # Quarterly
    
    def test_get_expiration_months_vix(self):
        """Test VIX expiration months."""
        months = get_expiration_months('VIX')
        assert len(months) == 12  # Monthly
        assert months == list(range(1, 13))
    
    def test_get_contract_symbol_es(self):
        """Test contract symbol generation for ES."""
        symbol = get_contract_symbol('ES', 2024, 3)
        assert symbol == 'ESH24'  # March 2024
        
        symbol = get_contract_symbol('ES', 2024, 6)
        assert symbol == 'ESM24'  # June 2024
        
        symbol = get_contract_symbol('ES', 2024, 9)
        assert symbol == 'ESU24'  # September 2024
        
        symbol = get_contract_symbol('ES', 2024, 12)
        assert symbol == 'ESZ24'  # December 2024
    
    def test_get_contract_symbol_vix(self):
        """Test contract symbol generation for VIX."""
        symbol = get_contract_symbol('VIX', 2024, 1)
        assert symbol == 'VIXF24'  # January 2024
    
    def test_get_contract_symbol_invalid_month(self):
        """Test that invalid month raises error."""
        with pytest.raises(ValueError):
            get_contract_symbol('ES', 2024, 13)
    
    def test_parse_contract_symbol(self):
        """Test parsing contract symbols."""
        asset, year, month = parse_contract_symbol('ESH24')
        assert asset == 'ES'
        assert year == 2024
        assert month == 3  # H = March
        
        asset, year, month = parse_contract_symbol('VIXZ25')
        assert asset == 'VIX'
        assert year == 2025
        assert month == 12  # Z = December
    
    def test_calculate_expiration_date_es(self):
        """Test expiration date calculation for ES."""
        exp = calculate_expiration_date(2024, 3, 'ES')
        # ES: Third Friday of expiration month
        # March 2024: First Friday is March 1, third Friday is March 15
        assert exp.day == 15
        assert exp.month == 3
        assert exp.year == 2024
    
    def test_calculate_expiration_date_vix(self):
        """Test expiration date calculation for VIX."""
        # VIX: 30 days before third Friday of following month
        exp = calculate_expiration_date(2024, 1, 'VIX')
        # February 2024: Third Friday is Feb 16
        # 30 days before = Jan 17
        assert exp.month == 1
        assert exp.year == 2024
        assert 15 <= exp.day <= 17  # Approximately 30 days before third Friday


# ============================================================================
# Tests: Roll Detection
# ============================================================================

class TestRollDetection:
    """Tests for contract roll detection."""
    
    def test_detect_single_roll(self, sample_es_data_single_roll):
        """Test detection of a single contract roll."""
        rolls = detect_contract_rolls(sample_es_data_single_roll, 'ES')
        assert len(rolls) == 1
        assert rolls[0].old_contract == 'ESZ23'
        assert rolls[0].new_contract == 'ESH24'
    
    def test_detect_multiple_rolls(self, sample_es_data_multiple_rolls):
        """Test detection of multiple contract rolls."""
        rolls = detect_contract_rolls(sample_es_data_multiple_rolls, 'ES')
        assert len(rolls) == 3
        assert rolls[0].old_contract == 'ESZ23'
        assert rolls[0].new_contract == 'ESH24'
        assert rolls[1].old_contract == 'ESH24'
        assert rolls[1].new_contract == 'ESM24'
        assert rolls[2].old_contract == 'ESM24'
        assert rolls[2].new_contract == 'ESU24'
    
    def test_detect_no_rolls_single_contract(self):
        """Test that no rolls are detected when only one contract."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D", tz="US/Eastern")
        df = pd.DataFrame({
            'timestamp': dates,
            'contract': ['ESZ23'] * 50,
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.randint(1000, 10000, 50),
        })
        
        rolls = detect_contract_rolls(df, 'ES')
        assert len(rolls) == 0
    
    def test_detect_rolls_empty_dataframe(self, empty_dataframe):
        """Test detection with empty DataFrame."""
        rolls = detect_contract_rolls(empty_dataframe, 'ES')
        assert len(rolls) == 0
    
    def test_detect_rolls_missing_contract_column(self, sample_es_data_single_roll):
        """Test that missing contract column raises error."""
        df = sample_es_data_single_roll.drop('contract', axis=1)
        with pytest.raises(KeyError):
            detect_contract_rolls(df, 'ES')
    
    def test_identify_active_contract(self, sample_es_data_multiple_rolls):
        """Test identifying the active (most recent) contract."""
        active = identify_active_contract(sample_es_data_multiple_rolls, 'ES')
        assert active == 'ESU24'
    
    def test_identify_active_contract_empty_dataframe(self, empty_dataframe):
        """Test that empty DataFrame raises error."""
        with pytest.raises(ValueError):
            identify_active_contract(empty_dataframe, 'ES')


# ============================================================================
# Tests: Backward Ratio Adjustment
# ============================================================================

class TestBackwardRatioAdjustment:
    """Tests for backward ratio adjustment method."""
    
    def test_backward_ratio_single_roll(self, sample_es_data_single_roll):
        """Test backward ratio adjustment with a single roll."""
        # Detect rolls first
        rolls = detect_contract_rolls(sample_es_data_single_roll, 'ES')
        
        # Apply adjustment
        adjusted_df, metadata = backward_ratio_adjustment(
            sample_es_data_single_roll, rolls
        )
        
        # Verify adjustment was applied
        assert metadata['method'] == 'backward_ratio'
        assert metadata['total_rolls'] == 1
        assert len(metadata['adjustments']) == 1
        
        # Check that old contract prices were adjusted
        old_contract_mask = adjusted_df['contract'] == 'ESZ23'
        new_contract_mask = adjusted_df['contract'] == 'ESH24'
        
        # Old contract should have different prices (due to ratio adjustment)
        old_close = sample_es_data_single_roll[old_contract_mask]['close'].iloc[-1]
        new_close = sample_es_data_single_roll[new_contract_mask]['close'].iloc[0]
        
        adjusted_old_close = adjusted_df[old_contract_mask]['close'].iloc[-1]
        adjusted_new_close = adjusted_df[new_contract_mask]['close'].iloc[0]
        
        # Last old close should be adjusted, first new close unchanged
        ratio = new_close / old_close
        assert abs(adjusted_old_close - old_close * ratio) < 0.01
    
    def test_backward_ratio_multiple_rolls(self, sample_es_data_multiple_rolls):
        """Test backward ratio adjustment with multiple rolls."""
        rolls = detect_contract_rolls(sample_es_data_multiple_rolls, 'ES')
        adjusted_df, metadata = backward_ratio_adjustment(
            sample_es_data_multiple_rolls, rolls
        )
        
        assert metadata['total_rolls'] == 3
        assert len(metadata['adjustments']) == 3
        
        # Verify all adjustments were applied
        for adjustment in metadata['adjustments']:
            assert 'roll_date' in adjustment
            assert 'roll_ratio' in adjustment
            assert 'rows_adjusted' in adjustment
            assert adjustment['rows_adjusted'] > 0
    
    def test_backward_ratio_no_rolls(self, sample_es_data_single_roll):
        """Test that no adjustment occurs with empty roll list."""
        adjusted_df, metadata = backward_ratio_adjustment(
            sample_es_data_single_roll, []
        )
        
        assert metadata['total_rolls'] == 0
        assert len(metadata['adjustments']) == 0
        
        # Data should be unchanged
        pd.testing.assert_frame_equal(sample_es_data_single_roll, adjusted_df)
    
    def test_backward_ratio_ohlc_adjustment(self, sample_es_data_single_roll):
        """Test that all OHLC prices are adjusted equally."""
        rolls = detect_contract_rolls(sample_es_data_single_roll, 'ES')
        adjusted_df, _ = backward_ratio_adjustment(
            sample_es_data_single_roll, rolls
        )
        
        # Get the adjustment ratio
        roll_ratio = rolls[0].roll_ratio
        
        # Get old contract data
        old_contract_mask = sample_es_data_single_roll['contract'] == 'ESZ23'
        
        # Check that OHLC are adjusted by the same ratio
        for col in ['open', 'high', 'low', 'close']:
            original = sample_es_data_single_roll[old_contract_mask][col].iloc[-1]
            adjusted = adjusted_df[old_contract_mask][col].iloc[-1]
            expected = original * roll_ratio
            assert abs(adjusted - expected) < 0.01
    
    def test_backward_ratio_volume_preserved(self, sample_es_data_single_roll):
        """Test that volume is not adjusted."""
        rolls = detect_contract_rolls(sample_es_data_single_roll, 'ES')
        adjusted_df, _ = backward_ratio_adjustment(
            sample_es_data_single_roll, rolls
        )
        
        # Volume should be identical
        pd.testing.assert_series_equal(
            sample_es_data_single_roll['volume'],
            adjusted_df['volume'],
            check_names=True
        )
    
    def test_backward_ratio_missing_column(self, sample_es_data_single_roll):
        """Test that missing OHLC column raises error."""
        rolls = detect_contract_rolls(sample_es_data_single_roll, 'ES')
        df_missing = sample_es_data_single_roll.drop('close', axis=1)
        
        with pytest.raises(KeyError):
            backward_ratio_adjustment(df_missing, rolls)


# ============================================================================
# Tests: Panama Canal Method
# ============================================================================

class TestPanamaCanalMethod:
    """Tests for panama canal method (concatenation without adjustment)."""
    
    def test_panama_canal_no_adjustment(self, sample_es_data_single_roll):
        """Test that panama canal method doesn't adjust prices."""
        rolls = detect_contract_rolls(sample_es_data_single_roll, 'ES')
        adjusted_df, metadata = panama_canal_method(
            sample_es_data_single_roll, rolls
        )
        
        # Prices should be unchanged
        pd.testing.assert_frame_equal(
            sample_es_data_single_roll[['open', 'high', 'low', 'close']],
            adjusted_df[['open', 'high', 'low', 'close']]
        )
    
    def test_panama_canal_tracks_gaps(self, sample_es_data_single_roll):
        """Test that panama canal method tracks price gaps."""
        rolls = detect_contract_rolls(sample_es_data_single_roll, 'ES')
        _, metadata = panama_canal_method(
            sample_es_data_single_roll, rolls
        )
        
        assert metadata['method'] == 'panama_canal'
        assert len(metadata['roll_info']) == 1
        
        gap_info = metadata['roll_info'][0]
        assert 'old_close' in gap_info
        assert 'new_close' in gap_info
        assert 'gap_pct' in gap_info
    
    def test_panama_canal_empty_rolls(self, sample_es_data_single_roll):
        """Test panama canal with no rolls."""
        adjusted_df, metadata = panama_canal_method(
            sample_es_data_single_roll, []
        )
        
        assert metadata['total_rolls'] == 0
        pd.testing.assert_frame_equal(sample_es_data_single_roll, adjusted_df)


# ============================================================================
# Tests: Price Continuity Verification
# ============================================================================

class TestPriceContinuityVerification:
    """Tests for price continuity verification."""
    
    def test_verify_continuity_after_adjustment(self, sample_es_data_single_roll):
        """Test that adjusted prices maintain continuity."""
        rolls = detect_contract_rolls(sample_es_data_single_roll, 'ES')
        adjusted_df, _ = backward_ratio_adjustment(
            sample_es_data_single_roll, rolls
        )
        
        continuity = verify_price_continuity(adjusted_df, rolls)
        
        assert continuity['passed'] is True
        assert continuity['total_rolls'] == 1
        assert len(continuity['gaps']) == 1
        
        gap = continuity['gaps'][0]
        assert gap['within_tolerance'] is True
        assert gap['gap_pct'] < 0.1
    
    def test_verify_continuity_panama_canal(self, sample_es_data_single_roll):
        """Test continuity verification with panama canal method."""
        rolls = detect_contract_rolls(sample_es_data_single_roll, 'ES')
        adjusted_df, _ = panama_canal_method(
            sample_es_data_single_roll, rolls
        )
        
        continuity = verify_price_continuity(
            adjusted_df, rolls, tolerance_pct=100.0
        )
        
        # Panama canal may have larger gaps, so we use 100% tolerance
        assert continuity['total_rolls'] == 1
    
    def test_verify_continuity_no_rolls(self, sample_es_data_single_roll):
        """Test continuity with no rolls."""
        continuity = verify_price_continuity(sample_es_data_single_roll, [])
        
        assert continuity['passed'] is True
        assert continuity['total_rolls'] == 0
        assert len(continuity['gaps']) == 0
    
    def test_verify_continuity_custom_tolerance(self, sample_es_data_single_roll):
        """Test continuity with custom tolerance."""
        rolls = detect_contract_rolls(sample_es_data_single_roll, 'ES')
        adjusted_df, _ = backward_ratio_adjustment(
            sample_es_data_single_roll, rolls
        )
        
        # Very tight tolerance
        continuity = verify_price_continuity(
            adjusted_df, rolls, tolerance_pct=0.01
        )
        
        # Should still pass for backward ratio adjusted data
        assert continuity['passed'] is True


# ============================================================================
# Tests: Roll Point Data Class
# ============================================================================

class TestRollPoint:
    """Tests for RollPoint data class."""
    
    def test_rollpoint_creation(self):
        """Test creating a RollPoint."""
        roll = RollPoint(
            roll_date=pd.Timestamp('2024-03-18'),
            old_contract='ESZ23',
            new_contract='ESH24',
            old_contract_close=105.0,
            new_contract_close=100.0,
            roll_ratio=100.0 / 105.0
        )
        
        assert roll.old_contract == 'ESZ23'
        assert roll.new_contract == 'ESH24'
        assert abs(roll.roll_ratio - (100.0 / 105.0)) < 0.0001
    
    def test_rollpoint_to_dict(self):
        """Test converting RollPoint to dictionary."""
        roll = RollPoint(
            roll_date=pd.Timestamp('2024-03-18'),
            old_contract='ESZ23',
            new_contract='ESH24',
            old_contract_close=105.0,
            new_contract_close=100.0,
            roll_ratio=100.0 / 105.0
        )
        
        roll_dict = roll.to_dict()
        assert roll_dict['old_contract'] == 'ESZ23'
        assert roll_dict['new_contract'] == 'ESH24'
        assert isinstance(roll_dict['roll_date'], str)


# ============================================================================
# Tests: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_zero_close_price_on_roll(self):
        """Test handling of zero close price (should not cause division errors)."""
        dates = pd.date_range("2024-01-01", periods=2, freq="D", tz="US/Eastern")
        df = pd.DataFrame({
            'timestamp': dates,
            'contract': ['ESZ23', 'ESH24'],
            'open': [100.0, 100.0],
            'high': [110.0, 110.0],
            'low': [90.0, 90.0],
            'close': [0.0, 100.0],  # Zero close on roll
            'volume': [1000, 1000],
        })
        
        # Should not raise error
        rolls = detect_contract_rolls(df, 'ES')
        assert len(rolls) == 1
        # Roll ratio should be set to 1.0 when old close is 0
        assert rolls[0].roll_ratio == 1.0
    
    def test_missing_data_between_rolls(self):
        """Test handling of missing data between contracts."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="US/Eastern")
        df = pd.DataFrame({
            'timestamp': dates,
            'contract': ['ESZ23'] * 40 + ['ESH24'] * 60,
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.randint(1000, 10000, 100),
        })
        
        rolls = detect_contract_rolls(df, 'ES')
        assert len(rolls) == 1
        
        # Adjustment should work
        adjusted_df, metadata = backward_ratio_adjustment(df, rolls)
        assert metadata['total_rolls'] == 1
    
    def test_vix_specific_detection(self, sample_vix_data_single_roll):
        """Test roll detection with VIX contracts."""
        rolls = detect_contract_rolls(sample_vix_data_single_roll, 'VIX')
        assert len(rolls) == 1
        assert rolls[0].old_contract == 'VIXF24'
        assert rolls[0].new_contract == 'VIXG24'
    
    def test_adjustment_method_enum(self):
        """Test AdjustmentMethod enum."""
        assert AdjustmentMethod.BACKWARD_RATIO.value == 'backward_ratio'
        assert AdjustmentMethod.PANAMA_CANAL.value == 'panama_canal'
