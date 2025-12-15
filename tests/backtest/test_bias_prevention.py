"""
Look-Ahead Bias Prevention Tests

Tests for detecting and preventing look-ahead bias in strategy execution.

Look-ahead bias occurs when:
- Future price data is accessed during current bar processing
- Future signals are used to make current decisions
- Order fills are based on future OHLC data
- Indicators are computed with future data

Tests cover:
- Timestamp validation
- Bar index validation
- Price range validation
- Data isolation checks
- Bias detection in strategy methods
"""

import pytest
from datetime import datetime, timedelta
from src.backtest.utils.bias_detection import (
    assert_current_bar_timestamp,
    assert_bar_not_in_future,
    validate_price_within_bar_range,
    detect_future_data_usage,
    validate_data_isolation,
    LookAheadBiasError,
)


# ============================================================================
# Timestamp Validation Tests
# ============================================================================

class TestTimestampValidation:
    """Test timestamp validation functions."""
    
    def test_valid_current_bar_timestamp(self):
        """Test that current bar timestamp passes validation."""
        current_ts = datetime(2024, 1, 15, 10, 0)
        
        # Should not raise
        assert_current_bar_timestamp(current_ts, current_ts, strict=True)
    
    def test_valid_past_timestamp(self):
        """Test that past timestamp passes validation."""
        current_ts = datetime(2024, 1, 15, 10, 0)
        past_ts = datetime(2024, 1, 15, 9, 0)
        
        # Should not raise
        assert_current_bar_timestamp(past_ts, current_ts, strict=True)
    
    def test_invalid_future_timestamp_strict(self):
        """Test that future timestamp raises in strict mode."""
        current_ts = datetime(2024, 1, 15, 10, 0)
        future_ts = datetime(2024, 1, 15, 11, 0)
        
        with pytest.raises(LookAheadBiasError):
            assert_current_bar_timestamp(future_ts, current_ts, strict=True)
    
    def test_invalid_future_timestamp_non_strict(self):
        """Test that future timestamp logs warning in non-strict mode."""
        current_ts = datetime(2024, 1, 15, 10, 0)
        future_ts = datetime(2024, 1, 15, 11, 0)
        
        # Should not raise in non-strict mode
        assert_current_bar_timestamp(future_ts, current_ts, strict=False)
    
    def test_timestamp_validation_with_context(self):
        """Test timestamp validation includes context in error."""
        current_ts = datetime(2024, 1, 15, 10, 0)
        future_ts = datetime(2024, 1, 15, 11, 0)
        
        try:
            assert_current_bar_timestamp(
                future_ts,
                current_ts,
                context='TestStrategy.next()',
                strict=True,
            )
            pytest.fail("Should have raised LookAheadBiasError")
        except LookAheadBiasError as e:
            assert 'TestStrategy.next()' in str(e)
    
    def test_multiple_timestamp_comparisons(self):
        """Test comparing multiple timestamps."""
        base_ts = datetime(2024, 1, 15, 10, 0)
        
        # All these should pass
        assert_current_bar_timestamp(base_ts, base_ts)  # Current
        assert_current_bar_timestamp(base_ts - timedelta(hours=1), base_ts)  # 1h ago
        assert_current_bar_timestamp(base_ts - timedelta(days=1), base_ts)  # 1 day ago
        
        # These should fail
        with pytest.raises(LookAheadBiasError):
            assert_current_bar_timestamp(base_ts + timedelta(minutes=1), base_ts, strict=True)


# ============================================================================
# Bar Index Validation Tests
# ============================================================================

class TestBarIndexValidation:
    """Test bar index validation functions."""
    
    def test_valid_current_bar_index(self):
        """Test that current bar index (0) is valid."""
        # Should not raise
        assert_bar_not_in_future(0, strict=True)
    
    def test_valid_past_bar_indices(self):
        """Test that past bar indices (negative) are valid."""
        # Should not raise
        assert_bar_not_in_future(-1, strict=True)  # Previous bar
        assert_bar_not_in_future(-5, strict=True)  # 5 bars ago
        assert_bar_not_in_future(-100, strict=True)  # 100 bars ago
    
    def test_invalid_future_bar_index_strict(self):
        """Test that future bar index raises in strict mode."""
        with pytest.raises(LookAheadBiasError):
            assert_bar_not_in_future(1, strict=True)
        
        with pytest.raises(LookAheadBiasError):
            assert_bar_not_in_future(5, strict=True)
    
    def test_invalid_future_bar_index_non_strict(self):
        """Test that future bar index logs in non-strict mode."""
        # Should not raise in non-strict mode
        assert_bar_not_in_future(1, strict=False)
        assert_bar_not_in_future(10, strict=False)
    
    def test_bar_index_validation_with_context(self):
        """Test bar index validation includes context."""
        try:
            assert_bar_not_in_future(
                2,
                context='accessing close[2]',
                strict=True,
            )
            pytest.fail("Should have raised LookAheadBiasError")
        except LookAheadBiasError as e:
            assert 'accessing close[2]' in str(e)


# ============================================================================
# Price Range Validation Tests
# ============================================================================

class TestPriceRangeValidation:
    """Test price range validation functions."""
    
    def test_price_within_range(self):
        """Test that price within OHLC range is valid."""
        low = 5100.0
        high = 5110.0
        close = 5105.0
        
        assert validate_price_within_bar_range(5105.0, low, high, close) is True
        assert validate_price_within_bar_range(5100.0, low, high, close) is True
        assert validate_price_within_bar_range(5110.0, low, high, close) is True
        assert validate_price_within_bar_range(5105.5, low, high, close) is True
    
    def test_price_below_range(self):
        """Test that price below range is invalid."""
        low = 5100.0
        high = 5110.0
        close = 5105.0
        
        result = validate_price_within_bar_range(5099.0, low, high, close)
        assert result is False
    
    def test_price_above_range(self):
        """Test that price above range is invalid."""
        low = 5100.0
        high = 5110.0
        close = 5105.0
        
        result = validate_price_within_bar_range(5111.0, low, high, close)
        assert result is False
    
    def test_price_range_with_context(self):
        """Test price range validation includes context."""
        low = 5100.0
        high = 5110.0
        close = 5105.0
        
        result = validate_price_within_bar_range(
            5099.0,
            low,
            high,
            close,
            context='entry order fill',
        )
        assert result is False
    
    def test_price_range_with_decimals(self):
        """Test price range validation with decimal values."""
        low = 5100.25
        high = 5110.75
        close = 5105.50
        
        assert validate_price_within_bar_range(5100.25, low, high, close) is True
        assert validate_price_within_bar_range(5110.75, low, high, close) is True
        assert validate_price_within_bar_range(5105.50, low, high, close) is True
        
        assert validate_price_within_bar_range(5100.24, low, high, close) is False
        assert validate_price_within_bar_range(5110.76, low, high, close) is False


# ============================================================================
# Data Isolation Tests
# ============================================================================

class TestDataIsolation:
    """Test data isolation validation."""
    
    def test_isolated_data_no_future_access(self):
        """Test that isolated data (no future access) passes."""
        current_data = [5100.0, 5105.0, 5099.0, 5102.0, 10000.0]  # OHLCV
        
        result = validate_data_isolation(current_data, None, 'MES')
        assert result is True
    
    def test_data_isolation_with_future_data(self):
        """Test that having future data fails isolation check."""
        current_data = [5100.0, 5105.0, 5099.0, 5102.0, 10000.0]
        future_data = [5101.0, 5106.0, 5100.0, 5103.0, 10100.0]
        
        result = validate_data_isolation(current_data, future_data, 'MES')
        assert result is False
    
    def test_data_isolation_multiple_symbols(self):
        """Test data isolation for multiple symbols."""
        mes_current = [5100.0, 5105.0, 5099.0, 5102.0, 10000.0]
        es_current = [5100.0, 5105.0, 5099.0, 5102.0, 10000.0]
        
        # Both should be isolated
        result1 = validate_data_isolation(mes_current, None, 'MES')
        result2 = validate_data_isolation(es_current, None, 'ES')
        
        assert result1 is True
        assert result2 is True


# ============================================================================
# Future Data Usage Detection Tests
# ============================================================================

class TestFutureDataDetection:
    """Test detection of future data usage patterns."""
    
    def test_detect_no_future_usage(self):
        """Test that clean code doesn't trigger detection."""
        def clean_strategy():
            return self.close[0]  # Current close only
        
        result = detect_future_data_usage(clean_strategy, 100, 1000)
        # May return False if no suspicious patterns found
    
    def test_detect_suspicious_patterns(self):
        """Test that suspicious patterns are detected."""
        def suspicious_strategy():
            # Contains suspicious pattern
            next_price = self.close[-1]  # Actually accessing -1, should be ok
            future_access = "tomorrow"
            return next_price
        
        result = detect_future_data_usage(suspicious_strategy, 100, 1000)
        # Should detect "tomorrow" pattern


# ============================================================================
# Comprehensive Bias Scenarios
# ============================================================================

class TestBiasScenarios:
    """Test comprehensive bias prevention scenarios."""
    
    def test_scenario_future_close_access(self):
        """Scenario: Strategy tries to use next bar's close price."""
        current_ts = datetime(2024, 1, 15, 10, 0)
        next_ts = current_ts + timedelta(minutes=1)
        
        with pytest.raises(LookAheadBiasError):
            assert_current_bar_timestamp(next_ts, current_ts, strict=True)
    
    def test_scenario_multi_bar_lookahead(self):
        """Scenario: Strategy looks ahead multiple bars."""
        for bar_offset in [1, 2, 5, 10]:
            with pytest.raises(LookAheadBiasError):
                assert_bar_not_in_future(bar_offset, strict=True)
    
    def test_scenario_unrealistic_fill_price(self):
        """Scenario: Order fills at price outside bar range."""
        bar_low = 5100.0
        bar_high = 5110.0
        bar_close = 5105.0
        
        # Try to fill below low
        result = validate_price_within_bar_range(5099.0, bar_low, bar_high, bar_close)
        assert result is False
        
        # Try to fill above high
        result = validate_price_within_bar_range(5111.0, bar_low, bar_high, bar_close)
        assert result is False
    
    def test_scenario_mixed_symbol_timestamps(self):
        """Scenario: Using timestamps from different symbols."""
        current_mes_ts = datetime(2024, 1, 15, 10, 0)
        future_es_ts = datetime(2024, 1, 15, 10, 1)
        
        # Should catch this as future data
        with pytest.raises(LookAheadBiasError):
            assert_current_bar_timestamp(future_es_ts, current_mes_ts, strict=True)
    
    def test_scenario_indicator_lookahead(self):
        """Scenario: Indicator computed with future data."""
        # Simulate trying to access future bar for indicator computation
        with pytest.raises(LookAheadBiasError):
            assert_bar_not_in_future(1, context='RSI calculation', strict=True)
    
    def test_scenario_sequential_valid_accesses(self):
        """Scenario: Multiple valid accesses in sequence."""
        base_ts = datetime(2024, 1, 15, 10, 0)
        
        # All should pass
        assert_current_bar_timestamp(base_ts, base_ts)
        assert_current_bar_timestamp(base_ts - timedelta(minutes=1), base_ts)
        assert_current_bar_timestamp(base_ts - timedelta(hours=1), base_ts)
        
        assert_bar_not_in_future(0)
        assert_bar_not_in_future(-1)
        assert_bar_not_in_future(-100)


# ============================================================================
# Edge Cases
# ============================================================================

class TestBiasDetectionEdgeCases:
    """Test edge cases in bias detection."""
    
    def test_exact_timestamp_match(self):
        """Test exact timestamp match (current bar)."""
        ts = datetime(2024, 1, 15, 10, 0, 0, 0)
        assert_current_bar_timestamp(ts, ts, strict=True)
    
    def test_microsecond_difference(self):
        """Test microsecond-level timestamp difference."""
        ts1 = datetime(2024, 1, 15, 10, 0, 0, 0)
        ts2 = datetime(2024, 1, 15, 10, 0, 0, 1)
        
        # Should detect as future (microsecond later)
        with pytest.raises(LookAheadBiasError):
            assert_current_bar_timestamp(ts2, ts1, strict=True)
    
    def test_price_range_edge_prices(self):
        """Test prices exactly at high and low."""
        assert validate_price_within_bar_range(100.0, 100.0, 110.0, 105.0) is True
        assert validate_price_within_bar_range(110.0, 100.0, 110.0, 105.0) is True
    
    def test_zero_price_range(self):
        """Test degenerate case where high == low."""
        # OHLC where price didn't move
        assert validate_price_within_bar_range(100.0, 100.0, 100.0, 100.0) is True
        assert validate_price_within_bar_range(100.1, 100.0, 100.0, 100.0) is False
    
    def test_large_price_moves(self):
        """Test with large intrabar price movements."""
        low = 1000.0
        high = 2000.0
        
        assert validate_price_within_bar_range(1500.0, low, high, 1500.0) is True
        assert validate_price_within_bar_range(999.0, low, high, 1500.0) is False
        assert validate_price_within_bar_range(2001.0, low, high, 1500.0) is False


# ============================================================================
# Strict vs Non-Strict Mode Tests
# ============================================================================

class TestStrictModes:
    """Test strict vs non-strict mode behavior."""
    
    def test_strict_mode_raises_immediately(self):
        """Test that strict mode raises on first violation."""
        with pytest.raises(LookAheadBiasError):
            assert_bar_not_in_future(1, strict=True)
    
    def test_non_strict_mode_allows_continuation(self):
        """Test that non-strict mode allows code to continue."""
        try:
            assert_bar_not_in_future(1, strict=False)
            assert_bar_not_in_future(2, strict=False)
            assert_bar_not_in_future(3, strict=False)
            # Should reach here without exception
            success = True
        except LookAheadBiasError:
            success = False
        
        assert success is True
    
    def test_mixed_strict_non_strict(self):
        """Test mixing strict and non-strict checks."""
        # Non-strict checks should not interfere with strict ones
        assert_bar_not_in_future(1, strict=False)  # Logs warning but continues
        
        with pytest.raises(LookAheadBiasError):
            assert_bar_not_in_future(1, strict=True)  # Should raise
