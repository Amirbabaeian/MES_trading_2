"""
Comprehensive test suite for anomaly detection.

Tests cover:
- Price gap detection (absolute and percentage thresholds)
- Z-score spike detection
- IQR spike detection
- Volume anomaly detection
- Alert generation and aggregation
- Batch processing
- Edge cases and configuration
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data_io.anomaly_detectors import (
    GapDetector,
    GapDetectionConfig,
    ZScoreSpikeDetector,
    IQRSpikeDetector,
    VolumeAnomalyDetector,
    AnomalyDetector,
    SpikeDetectionConfig,
    VolumeDetectionConfig,
)
from src.data_io.alerts import (
    Alert,
    AlertSeverity,
    AnomalyType,
    AlertAggregator,
    AlertReport,
)
from src.data_io.calendar_utils import TradingCalendar, MarketType, MarketHours, get_calendar
from datetime import time


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def valid_ohlcv_data():
    """Create valid OHLCV data with normal patterns."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1H", tz="US/Eastern")
    np.random.seed(42)
    
    # Create realistic OHLCV data
    bases = np.cumsum(np.random.normal(0, 0.5, 100)) + 100
    
    data = {
        "timestamp": dates,
        "open": bases,
        "close": bases + np.random.normal(0, 0.3, 100),
        "high": bases + np.abs(np.random.normal(1, 0.5, 100)),
        "low": bases - np.abs(np.random.normal(1, 0.5, 100)),
        "volume": np.random.randint(1000, 10000, 100),
    }
    
    df = pd.DataFrame(data)
    
    # Ensure OHLC relationships
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    
    return df


@pytest.fixture
def data_with_price_gap():
    """Create data with abnormal price gap."""
    dates = pd.date_range("2024-01-01", periods=50, freq="1H", tz="US/Eastern")
    
    # Normal data
    bases = np.full(50, 100.0)
    
    # Introduce a large gap at index 25
    bases[25] = 110.0  # $10 gap
    
    data = {
        "timestamp": dates,
        "open": bases,
        "close": bases,
        "high": bases + 0.5,
        "low": bases - 0.5,
        "volume": np.full(50, 5000),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def data_with_price_spike():
    """Create data with price spike outlier."""
    dates = pd.date_range("2024-01-01", periods=50, freq="1H", tz="US/Eastern")
    
    # Normal data around 100
    bases = np.full(50, 100.0) + np.random.normal(0, 0.5, 50)
    
    # Introduce a spike at index 25
    bases[25] = 105.0  # 5% spike
    
    data = {
        "timestamp": dates,
        "open": bases,
        "close": bases,
        "high": bases + 0.5,
        "low": bases - 0.5,
        "volume": np.full(50, 5000),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def data_with_volume_spike():
    """Create data with abnormal volume spike."""
    dates = pd.date_range("2024-01-01", periods=50, freq="1H", tz="US/Eastern")
    
    volumes = np.full(50, 5000)
    volumes[25] = 50000  # 10x spike
    
    data = {
        "timestamp": dates,
        "open": np.full(50, 100.0),
        "close": np.full(50, 100.0),
        "high": np.full(50, 100.5),
        "low": np.full(50, 99.5),
        "volume": volumes,
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def data_with_zero_volume():
    """Create data with zero volume bars."""
    dates = pd.date_range("2024-01-01", periods=20, freq="1H", tz="US/Eastern")
    
    data = {
        "timestamp": dates,
        "open": np.full(20, 100.0),
        "close": np.full(20, 100.0),
        "high": np.full(20, 100.5),
        "low": np.full(20, 99.5),
        "volume": np.full(20, 5000),
    }
    
    df = pd.DataFrame(data)
    df.loc[5, "volume"] = 0
    df.loc[15, "volume"] = 0
    
    return df


@pytest.fixture
def nyse_calendar():
    """Get NYSE calendar for testing."""
    return get_calendar("NYSE")


# ============================================================================
# Tests: Gap Detection
# ============================================================================

class TestGapDetection:
    """Tests for price gap detection."""
    
    def test_detect_absolute_gap(self, data_with_price_gap):
        """Test detection of gap exceeding absolute threshold."""
        config = GapDetectionConfig(absolute_threshold=5.0, percent_threshold=10.0)
        detector = GapDetector(config)
        
        alerts = detector.detect(data_with_price_gap)
        
        assert len(alerts) > 0
        assert alerts[0].anomaly_type in [AnomalyType.INTRADAY_GAP, AnomalyType.SESSION_BOUNDARY_GAP]
        assert alerts[0].measured_value > 5.0
    
    def test_detect_percentage_gap(self):
        """Test detection of gap exceeding percentage threshold."""
        dates = pd.date_range("2024-01-01", periods=20, freq="1H", tz="US/Eastern")
        
        # 5% gap at index 10
        bases = np.full(20, 100.0)
        bases[10] = 105.0
        
        data = {
            "timestamp": dates,
            "open": bases,
            "close": bases,
            "high": bases + 0.5,
            "low": bases - 0.5,
            "volume": np.full(20, 5000),
        }
        
        df = pd.DataFrame(data)
        config = GapDetectionConfig(absolute_threshold=10.0, percent_threshold=2.0)
        detector = GapDetector(config)
        
        alerts = detector.detect(df)
        
        # Should detect the 5% gap
        assert len(alerts) > 0
    
    def test_no_alert_for_small_gap(self):
        """Test that small gaps don't trigger alerts."""
        dates = pd.date_range("2024-01-01", periods=20, freq="1H", tz="US/Eastern")
        
        # 0.5% gap (below threshold)
        bases = np.full(20, 100.0)
        bases[10] = 100.5
        
        data = {
            "timestamp": dates,
            "open": bases,
            "close": bases,
            "high": bases + 0.1,
            "low": bases - 0.1,
            "volume": np.full(20, 5000),
        }
        
        df = pd.DataFrame(data)
        config = GapDetectionConfig(absolute_threshold=5.0, percent_threshold=2.0)
        detector = GapDetector(config)
        
        alerts = detector.detect(df)
        
        assert len(alerts) == 0
    
    def test_gap_detector_with_calendar(self, data_with_price_gap, nyse_calendar):
        """Test gap detection with calendar support."""
        config = GapDetectionConfig(
            absolute_threshold=5.0,
            flag_session_boundaries=False
        )
        detector = GapDetector(config, nyse_calendar)
        
        alerts = detector.detect(data_with_price_gap)
        
        # Should still detect gaps
        assert len(alerts) > 0


# ============================================================================
# Tests: Z-Score Spike Detection
# ============================================================================

class TestZScoreSpikeDetection:
    """Tests for z-score based spike detection."""
    
    def test_detect_zscore_spike(self, data_with_price_spike):
        """Test detection of z-score spike."""
        config = SpikeDetectionConfig(
            zscore_sigma=2.0,
            zscore_window=20,
            min_history=5,
        )
        detector = ZScoreSpikeDetector(config)
        
        alerts = detector.detect(data_with_price_spike)
        
        # Should detect the spike
        assert len(alerts) > 0
        assert alerts[0].anomaly_type == AnomalyType.PRICE_SPIKE
    
    def test_zscore_different_price_columns(self):
        """Test z-score detection on different price columns."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1H", tz="US/Eastern")
        
        # Create data with spike in high prices
        highs = np.full(50, 100.5)
        highs[25] = 110.0  # spike in high
        
        data = {
            "timestamp": dates,
            "open": np.full(50, 100.0),
            "close": np.full(50, 100.0),
            "high": highs,
            "low": np.full(50, 99.5),
            "volume": np.full(50, 5000),
        }
        
        df = pd.DataFrame(data)
        config = SpikeDetectionConfig(zscore_sigma=2.0, zscore_window=10)
        detector = ZScoreSpikeDetector(config)
        
        # Test detection on high prices
        alerts_high = detector.detect(df, price_column="high")
        
        assert len(alerts_high) > 0
    
    def test_zscore_minimum_history_requirement(self):
        """Test that z-score detection respects minimum history."""
        dates = pd.date_range("2024-01-01", periods=3, freq="1H", tz="US/Eastern")
        
        data = {
            "timestamp": dates,
            "open": [100.0, 100.5, 105.0],
            "close": [100.0, 100.5, 105.0],
            "high": [100.5, 101.0, 105.5],
            "low": [99.5, 100.0, 104.5],
            "volume": [5000, 5000, 5000],
        }
        
        df = pd.DataFrame(data)
        config = SpikeDetectionConfig(min_history=10)
        detector = ZScoreSpikeDetector(config)
        
        alerts = detector.detect(df)
        
        # Should not detect anything (insufficient history)
        assert len(alerts) == 0


# ============================================================================
# Tests: IQR Spike Detection
# ============================================================================

class TestIQRSpikeDetection:
    """Tests for IQR-based spike detection."""
    
    def test_detect_iqr_spike(self, data_with_price_spike):
        """Test detection of IQR spike."""
        config = SpikeDetectionConfig(
            iqr_multiplier=1.5,
            iqr_window=20,
            min_history=5,
        )
        detector = IQRSpikeDetector(config)
        
        alerts = detector.detect(data_with_price_spike)
        
        # Should detect the spike
        assert len(alerts) > 0
        assert alerts[0].anomaly_type == AnomalyType.PRICE_SPIKE
    
    def test_iqr_robustness_to_outliers(self):
        """Test that IQR is more robust than z-score."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1H", tz="US/Eastern")
        
        # Normal data with one extreme outlier
        prices = np.full(50, 100.0)
        prices[25] = 200.0  # Extreme outlier
        
        data = {
            "timestamp": dates,
            "open": prices,
            "close": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "volume": np.full(50, 5000),
        }
        
        df = pd.DataFrame(data)
        
        config = SpikeDetectionConfig(
            iqr_multiplier=1.5,
            iqr_window=20,
            min_history=5,
        )
        detector = IQRSpikeDetector(config)
        
        alerts = detector.detect(df)
        
        # IQR should still detect the outlier
        assert len(alerts) > 0
    
    def test_iqr_different_multipliers(self):
        """Test IQR detection with different multipliers."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1H", tz="US/Eastern")
        
        prices = np.full(50, 100.0)
        prices[25] = 105.0
        
        data = {
            "timestamp": dates,
            "open": prices,
            "close": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "volume": np.full(50, 5000),
        }
        
        df = pd.DataFrame(data)
        
        # Test with strict multiplier (1.0)
        config_strict = SpikeDetectionConfig(iqr_multiplier=1.0, iqr_window=10)
        detector_strict = IQRSpikeDetector(config_strict)
        alerts_strict = detector_strict.detect(df)
        
        # Test with loose multiplier (3.0)
        config_loose = SpikeDetectionConfig(iqr_multiplier=3.0, iqr_window=10)
        detector_loose = IQRSpikeDetector(config_loose)
        alerts_loose = detector_loose.detect(df)
        
        # Stricter multiplier should find more (or same) alerts
        assert len(alerts_strict) >= len(alerts_loose)


# ============================================================================
# Tests: Volume Anomaly Detection
# ============================================================================

class TestVolumeAnomalyDetection:
    """Tests for volume anomaly detection."""
    
    def test_detect_high_volume_spike(self, data_with_volume_spike):
        """Test detection of abnormally high volume."""
        config = VolumeDetectionConfig(
            high_threshold_sigma=2.0,
            baseline_window=20,
            min_history=5,
        )
        detector = VolumeAnomalyDetector(config)
        
        alerts = detector.detect(data_with_volume_spike)
        
        # Should detect high volume
        assert len(alerts) > 0
        assert any(a.anomaly_type == AnomalyType.VOLUME_SPIKE for a in alerts)
    
    def test_detect_zero_volume(self, data_with_zero_volume):
        """Test detection of zero volume bars."""
        config = VolumeDetectionConfig(flag_zero_volume=True)
        detector = VolumeAnomalyDetector(config)
        
        alerts = detector.detect(data_with_zero_volume)
        
        # Should detect 2 zero volume bars
        zero_alerts = [a for a in alerts if "zero" in a.message.lower()]
        assert len(zero_alerts) >= 2
    
    def test_ignore_zero_volume_if_disabled(self, data_with_zero_volume):
        """Test that zero volume check can be disabled."""
        config = VolumeDetectionConfig(flag_zero_volume=False)
        detector = VolumeAnomalyDetector(config)
        
        alerts = detector.detect(data_with_zero_volume)
        
        # Should not detect zero volume
        zero_alerts = [a for a in alerts if "zero" in a.message.lower()]
        assert len(zero_alerts) == 0
    
    def test_detect_low_volume(self):
        """Test detection of abnormally low volume."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1H", tz="US/Eastern")
        
        volumes = np.full(50, 5000.0)
        volumes[25] = 500  # 10x lower
        
        data = {
            "timestamp": dates,
            "open": np.full(50, 100.0),
            "close": np.full(50, 100.0),
            "high": np.full(50, 100.5),
            "low": np.full(50, 99.5),
            "volume": volumes,
        }
        
        df = pd.DataFrame(data)
        config = VolumeDetectionConfig(
            low_threshold_sigma=-2.0,
            baseline_window=20,
            min_history=5,
        )
        detector = VolumeAnomalyDetector(config)
        
        alerts = detector.detect(df)
        
        # Should detect low volume
        assert len(alerts) > 0


# ============================================================================
# Tests: Alert Generation and Management
# ============================================================================

class TestAlertGeneration:
    """Tests for alert generation and properties."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            anomaly_type=AnomalyType.PRICE_GAP,
            timestamp="2024-01-01 10:00:00",
            bar_index=5,
            asset_symbol="ES",
            measured_value=10.0,
            expected_value=5.0,
            message="Price gap detected",
        )
        
        assert alert.severity == AlertSeverity.WARNING
        assert alert.anomaly_type == AnomalyType.PRICE_GAP
        assert alert.bar_index == 5
        assert alert.created_at is not None
    
    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = Alert(
            severity=AlertSeverity.CRITICAL,
            anomaly_type=AnomalyType.VOLUME_SPIKE,
            timestamp="2024-01-01 10:00:00",
            bar_index=10,
            asset_symbol="MES",
            measured_value=100000,
            message="Volume spike",
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict["severity"] == "critical"
        assert alert_dict["anomaly_type"] == "volume_spike"
        assert alert_dict["bar_index"] == 10
        assert alert_dict["asset_symbol"] == "MES"
    
    def test_alert_to_json(self):
        """Test converting alert to JSON."""
        alert = Alert(
            severity=AlertSeverity.INFO,
            anomaly_type=AnomalyType.PRICE_SPIKE,
            timestamp="2024-01-01 10:00:00",
            bar_index=5,
            asset_symbol="TEST",
            measured_value=50.0,
            message="Test alert",
        )
        
        json_str = alert.to_json()
        
        assert "severity" in json_str
        assert "price_spike" in json_str
        assert "TEST" in json_str


# ============================================================================
# Tests: Alert Aggregation
# ============================================================================

class TestAlertAggregation:
    """Tests for alert aggregation."""
    
    def test_aggregate_consecutive_alerts(self):
        """Test aggregating consecutive alerts of the same type."""
        alerts = [
            Alert(
                severity=AlertSeverity.WARNING,
                anomaly_type=AnomalyType.PRICE_GAP,
                timestamp=f"2024-01-01 {i:02d}:00:00",
                bar_index=i,
                asset_symbol="TEST",
                measured_value=5.0,
                message="Gap",
            )
            for i in range(25, 28)
        ]
        
        aggregator = AlertAggregator(max_consecutive_bars=3)
        aggregated = aggregator.aggregate(alerts)
        
        # Should be aggregated into one alert
        assert len(aggregated) == 1
        assert aggregated[0].metadata["original_count"] == 3
    
    def test_no_aggregation_for_different_types(self):
        """Test that different anomaly types aren't aggregated."""
        alerts = [
            Alert(
                severity=AlertSeverity.WARNING,
                anomaly_type=AnomalyType.PRICE_GAP,
                timestamp="2024-01-01 10:00:00",
                bar_index=5,
                asset_symbol="TEST",
                measured_value=5.0,
                message="Gap",
            ),
            Alert(
                severity=AlertSeverity.WARNING,
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                timestamp="2024-01-01 11:00:00",
                bar_index=6,
                asset_symbol="TEST",
                measured_value=100000,
                message="Volume spike",
            ),
        ]
        
        aggregator = AlertAggregator()
        aggregated = aggregator.aggregate(alerts)
        
        # Should not be aggregated
        assert len(aggregated) == 2
    
    def test_aggregation_respects_gap_threshold(self):
        """Test that aggregation respects maximum consecutive bar threshold."""
        alerts = [
            Alert(
                severity=AlertSeverity.WARNING,
                anomaly_type=AnomalyType.PRICE_GAP,
                timestamp=f"2024-01-01 {i:02d}:00:00",
                bar_index=i,
                asset_symbol="TEST",
                measured_value=5.0,
                message="Gap",
            )
            for i in [10, 11, 12, 20]  # Gap at 8 bars
        ]
        
        aggregator = AlertAggregator(max_consecutive_bars=3)
        aggregated = aggregator.aggregate(alerts)
        
        # Should be aggregated into 2 groups
        assert len(aggregated) == 2


# ============================================================================
# Tests: Alert Reporting
# ============================================================================

class TestAlertReporting:
    """Tests for alert reporting and analysis."""
    
    def test_alerts_to_dataframe(self):
        """Test converting alerts to DataFrame."""
        alerts = [
            Alert(
                severity=AlertSeverity.WARNING,
                anomaly_type=AnomalyType.PRICE_GAP,
                timestamp="2024-01-01 10:00:00",
                bar_index=5,
                asset_symbol="TEST",
                measured_value=5.0,
                message="Gap",
            ),
            Alert(
                severity=AlertSeverity.CRITICAL,
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                timestamp="2024-01-01 11:00:00",
                bar_index=6,
                asset_symbol="TEST",
                measured_value=100000,
                message="Volume spike",
            ),
        ]
        
        df = AlertReport.to_dataframe(alerts)
        
        assert len(df) == 2
        assert "severity" in df.columns
        assert "anomaly_type" in df.columns
        assert "bar_index" in df.columns
    
    def test_group_alerts_by_severity(self):
        """Test grouping alerts by severity."""
        alerts = [
            Alert(
                severity=AlertSeverity.CRITICAL,
                anomaly_type=AnomalyType.PRICE_GAP,
                timestamp="2024-01-01 10:00:00",
                bar_index=5,
                asset_symbol="TEST",
                measured_value=5.0,
            ),
            Alert(
                severity=AlertSeverity.WARNING,
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                timestamp="2024-01-01 11:00:00",
                bar_index=6,
                asset_symbol="TEST",
                measured_value=100000,
            ),
            Alert(
                severity=AlertSeverity.CRITICAL,
                anomaly_type=AnomalyType.PRICE_SPIKE,
                timestamp="2024-01-01 12:00:00",
                bar_index=7,
                asset_symbol="TEST",
                measured_value=105.0,
            ),
        ]
        
        grouped = AlertReport.group_by_severity(alerts)
        
        assert len(grouped[AlertSeverity.CRITICAL]) == 2
        assert len(grouped[AlertSeverity.WARNING]) == 1
        assert len(grouped[AlertSeverity.INFO]) == 0
    
    def test_group_alerts_by_type(self):
        """Test grouping alerts by anomaly type."""
        alerts = [
            Alert(
                severity=AlertSeverity.WARNING,
                anomaly_type=AnomalyType.PRICE_GAP,
                timestamp="2024-01-01 10:00:00",
                bar_index=5,
                asset_symbol="TEST",
                measured_value=5.0,
            ),
            Alert(
                severity=AlertSeverity.WARNING,
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                timestamp="2024-01-01 11:00:00",
                bar_index=6,
                asset_symbol="TEST",
                measured_value=100000,
            ),
            Alert(
                severity=AlertSeverity.WARNING,
                anomaly_type=AnomalyType.PRICE_GAP,
                timestamp="2024-01-01 12:00:00",
                bar_index=7,
                asset_symbol="TEST",
                measured_value=3.0,
            ),
        ]
        
        grouped = AlertReport.group_by_type(alerts)
        
        assert len(grouped[AnomalyType.PRICE_GAP]) == 2
        assert len(grouped[AnomalyType.VOLUME_SPIKE]) == 1
    
    def test_alert_summary(self):
        """Test generating alert summary."""
        alerts = [
            Alert(
                severity=AlertSeverity.CRITICAL,
                anomaly_type=AnomalyType.PRICE_GAP,
                timestamp="2024-01-01 10:00:00",
                bar_index=5,
                asset_symbol="TEST",
                measured_value=5.0,
            ),
            Alert(
                severity=AlertSeverity.WARNING,
                anomaly_type=AnomalyType.VOLUME_SPIKE,
                timestamp="2024-01-01 11:00:00",
                bar_index=6,
                asset_symbol="TEST",
                measured_value=100000,
            ),
        ]
        
        summary = AlertReport.summary(alerts)
        
        assert summary["total_alerts"] == 2
        assert summary["by_severity"]["critical"] == 1
        assert summary["by_severity"]["warning"] == 1
        assert summary["by_type"]["price_gap"] == 1
        assert summary["by_type"]["volume_spike"] == 1


# ============================================================================
# Tests: Main Anomaly Detector
# ============================================================================

class TestAnomalyDetector:
    """Tests for the main AnomalyDetector orchestrator."""
    
    def test_detect_all_anomalies(self, valid_ohlcv_data):
        """Test detecting all types of anomalies."""
        # Inject various anomalies
        df = valid_ohlcv_data.copy()
        
        # Add price gap
        df.loc[25, "close"] = df.loc[25, "close"] + 5.0
        
        # Add volume spike
        df.loc[50, "volume"] = 100000
        
        # Add zero volume
        df.loc[75, "volume"] = 0
        
        detector = AnomalyDetector(
            enable_gap_detection=True,
            enable_zscore_spike_detection=True,
            enable_iqr_spike_detection=True,
            enable_volume_detection=True,
            aggregate_alerts=False,
        )
        
        alerts = detector.detect(df, "TEST")
        
        # Should detect multiple anomalies
        assert len(alerts) > 0
    
    def test_selective_detection(self, valid_ohlcv_data):
        """Test running only selected detectors."""
        df = valid_ohlcv_data.copy()
        df.loc[25, "close"] = df.loc[25, "close"] + 5.0
        
        # Only gap detection
        detector = AnomalyDetector(
            enable_gap_detection=True,
            enable_zscore_spike_detection=False,
            enable_iqr_spike_detection=False,
            enable_volume_detection=False,
        )
        
        alerts = detector.detect(df, "TEST")
        
        # Should only have gap alerts (if any)
        for alert in alerts:
            assert alert.anomaly_type in [AnomalyType.INTRADAY_GAP, AnomalyType.SESSION_BOUNDARY_GAP]
    
    def test_batch_detection(self, valid_ohlcv_data):
        """Test batch processing of multiple assets."""
        datasets = {
            "TEST1": valid_ohlcv_data.copy(),
            "TEST2": valid_ohlcv_data.copy(),
        }
        
        detector = AnomalyDetector(
            aggregate_alerts=False,
        )
        
        results = detector.detect_batch(datasets)
        
        assert len(results) == 2
        assert "TEST1" in results
        assert "TEST2" in results
    
    def test_configuration_respected(self, valid_ohlcv_data):
        """Test that configuration parameters are respected."""
        df = valid_ohlcv_data.copy()
        
        # Create detector with strict gap thresholds
        strict_config = GapDetectionConfig(
            absolute_threshold=100.0,  # Very high threshold
        )
        
        detector_strict = AnomalyDetector(
            gap_config=strict_config,
            enable_gap_detection=True,
            enable_zscore_spike_detection=False,
            enable_iqr_spike_detection=False,
            enable_volume_detection=False,
        )
        
        alerts_strict = detector_strict.detect(df, "TEST")
        
        # Should be very few alerts with high threshold
        assert len(alerts_strict) == 0


# ============================================================================
# Tests: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        detector = AnomalyDetector()
        alerts = detector.detect(df)
        
        assert len(alerts) == 0
    
    def test_single_bar(self):
        """Test handling of single bar."""
        dates = pd.date_range("2024-01-01", periods=1, freq="1H", tz="US/Eastern")
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": [100.0],
            "close": [100.0],
            "high": [100.5],
            "low": [99.5],
            "volume": [5000],
        })
        
        detector = AnomalyDetector()
        alerts = detector.detect(df)
        
        # Should handle gracefully
        assert isinstance(alerts, list)
    
    def test_flat_prices(self):
        """Test handling of flat prices (no volatility)."""
        dates = pd.date_range("2024-01-01", periods=50, freq="1H", tz="US/Eastern")
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.full(50, 100.0),
            "close": np.full(50, 100.0),
            "high": np.full(50, 100.0),
            "low": np.full(50, 100.0),
            "volume": np.full(50, 5000),
        })
        
        detector = AnomalyDetector()
        alerts = detector.detect(df)
        
        # Should handle flat prices without error
        assert isinstance(alerts, list)
    
    def test_nan_values(self):
        """Test handling of NaN values."""
        dates = pd.date_range("2024-01-01", periods=20, freq="1H", tz="US/Eastern")
        
        data = {
            "timestamp": dates,
            "open": [100.0] * 20,
            "close": [100.0] * 20,
            "high": [100.5] * 20,
            "low": [99.5] * 20,
            "volume": [5000.0] * 20,
        }
        
        df = pd.DataFrame(data)
        df.loc[5, "close"] = np.nan
        
        detector = AnomalyDetector()
        
        # Should handle NaN gracefully
        try:
            alerts = detector.detect(df)
            assert isinstance(alerts, list)
        except Exception as e:
            pytest.fail(f"Detector should handle NaN values: {e}")
