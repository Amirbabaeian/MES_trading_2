"""
Anomaly detection validators for market data quality assessment.

Detects data quality issues including:
- Price gaps (abnormal price jumps between bars)
- Price spikes (extreme outliers using z-score and IQR methods)
- Volume anomalies (unusual volume patterns)

Provides configurable detection thresholds and session boundary handling.
Designed to flag suspicious data requiring manual review before promotion.

Features:
- Configurable gap detection (absolute $ and percentage thresholds)
- Z-score based spike detection
- IQR-based spike detection
- Volume baseline comparison
- Session boundary gap handling
- Alert generation with severity levels
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import logging

from .alerts import Alert, AlertSeverity, AnomalyType, AlertAggregator
from .calendar_utils import TradingCalendar, get_calendar

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class GapDetectionConfig:
    """Configuration for price gap detection."""
    absolute_threshold: float = 5.0  # Dollar amount threshold (e.g., $5)
    percent_threshold: float = 2.0   # Percentage threshold (e.g., 2%)
    flag_session_boundaries: bool = False  # Flag gaps at market open/close
    use_close_to_close: bool = True  # Compare close-to-close vs open-to-previous-close
    

@dataclass
class SpikeDetectionConfig:
    """Configuration for price spike detection."""
    zscore_sigma: float = 3.0  # Z-score threshold (3σ or 5σ)
    zscore_window: int = 20    # Rolling window size for z-score
    iqr_multiplier: float = 1.5  # IQR multiplier (typically 1.5 or 3.0)
    iqr_window: int = 20       # Rolling window size for IQR
    apply_to_returns: bool = False  # Apply to returns instead of prices
    min_history: int = 5       # Minimum bars needed for detection


@dataclass
class VolumeDetectionConfig:
    """Configuration for volume anomaly detection."""
    high_threshold_sigma: float = 2.5  # Z-score for high volume detection
    low_threshold_sigma: float = -2.0  # Z-score for low volume detection
    baseline_window: int = 20   # Bars to use for volume baseline
    min_history: int = 5        # Minimum bars needed for detection
    flag_zero_volume: bool = True  # Flag bars with zero volume
    consider_session_open: bool = True  # Account for higher volume at open


# ============================================================================
# Gap Detection
# ============================================================================

class GapDetector:
    """
    Detects abnormal price gaps between consecutive bars.
    
    Identifies price jumps that exceed configured thresholds,
    distinguishing between intraday gaps and session boundary gaps.
    """
    
    def __init__(
        self,
        config: Optional[GapDetectionConfig] = None,
        calendar: Optional[TradingCalendar] = None,
    ):
        """
        Initialize GapDetector.
        
        Args:
            config: GapDetectionConfig instance
            calendar: TradingCalendar for session boundary detection
        """
        self.config = config or GapDetectionConfig()
        self.calendar = calendar
    
    def detect(
        self,
        df: pd.DataFrame,
        asset_symbol: str = "UNKNOWN",
    ) -> List[Alert]:
        """
        Detect price gaps in OHLCV data.
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
            asset_symbol: Symbol/identifier for the asset
            
        Returns:
            List of Alert objects for detected gaps
        """
        alerts = []
        
        if len(df) < 2:
            return alerts
        
        # Calculate price changes
        if self.config.use_close_to_close:
            # Compare close to previous close
            price_changes = df["close"].diff()
            reference_prices = df["close"].shift()
        else:
            # Compare open to previous close
            price_changes = df["open"] - df["close"].shift()
            reference_prices = df["close"].shift()
        
        # Calculate absolute and percentage gaps
        abs_gaps = price_changes.abs()
        pct_gaps = (abs_gaps / reference_prices).abs() * 100
        
        for i in range(1, len(df)):
            abs_gap = abs_gaps.iloc[i]
            pct_gap = pct_gaps.iloc[i]
            
            # Check if gap exceeds thresholds
            exceeds_absolute = abs_gap > self.config.absolute_threshold
            exceeds_percent = pct_gap > self.config.percent_threshold
            
            if not (exceeds_absolute or exceeds_percent):
                continue
            
            # Determine if this is a session boundary gap
            is_session_boundary = self._is_session_boundary_gap(df, i)
            
            if is_session_boundary and not self.config.flag_session_boundaries:
                continue
            
            # Create alert
            anomaly_type = (
                AnomalyType.SESSION_BOUNDARY_GAP 
                if is_session_boundary 
                else AnomalyType.INTRADAY_GAP
            )
            
            severity = (
                AlertSeverity.WARNING 
                if is_session_boundary 
                else AlertSeverity.CRITICAL
            )
            
            message = (
                f"Price gap of ${abs_gap:.2f} ({pct_gap:.2f}%) "
                f"from {reference_prices.iloc[i]:.2f} to {df['close'].iloc[i]:.2f}"
            )
            
            alert = Alert(
                severity=severity,
                anomaly_type=anomaly_type,
                timestamp=df["timestamp"].iloc[i],
                bar_index=i,
                asset_symbol=asset_symbol,
                measured_value=abs_gap,
                expected_value=self.config.absolute_threshold,
                threshold_value=self.config.absolute_threshold,
                message=message,
                metadata={
                    "gap_amount": float(abs_gap),
                    "gap_percent": float(pct_gap),
                    "previous_price": float(reference_prices.iloc[i]),
                    "current_price": float(df["close"].iloc[i]),
                    "is_session_boundary": is_session_boundary,
                },
            )
            alerts.append(alert)
        
        return alerts
    
    def _is_session_boundary_gap(self, df: pd.DataFrame, bar_index: int) -> bool:
        """
        Check if a gap occurs at a session boundary (market close/open).
        
        Args:
            df: DataFrame with OHLCV data
            bar_index: Index of the bar to check
            
        Returns:
            True if gap occurs at session boundary
        """
        if self.calendar is None:
            return False
        
        prev_timestamp = df["timestamp"].iloc[bar_index - 1]
        curr_timestamp = df["timestamp"].iloc[bar_index]
        
        # Check if timestamps are on different dates
        if prev_timestamp.date() != curr_timestamp.date():
            return True
        
        # Could also check for gaps within extended hours, but for now
        # we focus on overnight gaps
        return False


# ============================================================================
# Spike Detection (Z-Score)
# ============================================================================

class ZScoreSpikeDetector:
    """
    Detects price spikes using z-score method.
    
    Identifies extreme outliers that deviate significantly from the
    rolling mean (typically 3σ or 5σ).
    """
    
    def __init__(
        self,
        config: Optional[SpikeDetectionConfig] = None,
    ):
        """
        Initialize ZScoreSpikeDetector.
        
        Args:
            config: SpikeDetectionConfig instance
        """
        self.config = config or SpikeDetectionConfig()
    
    def detect(
        self,
        df: pd.DataFrame,
        asset_symbol: str = "UNKNOWN",
        price_column: str = "close",
    ) -> List[Alert]:
        """
        Detect price spikes using z-score method.
        
        Args:
            df: DataFrame with OHLCV data
            asset_symbol: Symbol/identifier for the asset
            price_column: Column to analyze (open, close, high, low)
            
        Returns:
            List of Alert objects for detected spikes
        """
        alerts = []
        
        if len(df) < self.config.min_history:
            return alerts
        
        # Get price data
        if self.config.apply_to_returns:
            prices = df[price_column].pct_change().fillna(0)
        else:
            prices = df[price_column]
        
        # Calculate rolling mean and std
        rolling_mean = prices.rolling(window=self.config.zscore_window).mean()
        rolling_std = prices.rolling(window=self.config.zscore_window).std()
        
        # Calculate z-scores
        zscores = (prices - rolling_mean) / rolling_std
        
        # Detect spikes (absolute z-score > threshold)
        for i in range(self.config.min_history, len(df)):
            zscore = zscores.iloc[i]
            
            if pd.isna(zscore):
                continue
            
            if abs(zscore) > self.config.zscore_sigma:
                message = (
                    f"Price spike detected: {price_column} = {prices.iloc[i]:.4f} "
                    f"(z-score: {zscore:.2f}, threshold: ±{self.config.zscore_sigma:.1f}σ)"
                )
                
                alert = Alert(
                    severity=AlertSeverity.WARNING,
                    anomaly_type=AnomalyType.PRICE_SPIKE,
                    timestamp=df["timestamp"].iloc[i],
                    bar_index=i,
                    asset_symbol=asset_symbol,
                    measured_value=float(prices.iloc[i]),
                    expected_value=float(rolling_mean.iloc[i]),
                    threshold_value=self.config.zscore_sigma,
                    message=message,
                    metadata={
                        "price": float(prices.iloc[i]),
                        "rolling_mean": float(rolling_mean.iloc[i]),
                        "rolling_std": float(rolling_std.iloc[i]),
                        "zscore": float(zscore),
                        "method": "zscore",
                        "price_column": price_column,
                    },
                )
                alerts.append(alert)
        
        return alerts


# ============================================================================
# Spike Detection (IQR)
# ============================================================================

class IQRSpikeDetector:
    """
    Detects price spikes using Interquartile Range (IQR) method.
    
    Identifies outliers as values beyond Q1 - 1.5*IQR and Q3 + 1.5*IQR.
    More robust to extreme outliers than z-score method.
    """
    
    def __init__(
        self,
        config: Optional[SpikeDetectionConfig] = None,
    ):
        """
        Initialize IQRSpikeDetector.
        
        Args:
            config: SpikeDetectionConfig instance
        """
        self.config = config or SpikeDetectionConfig()
    
    def detect(
        self,
        df: pd.DataFrame,
        asset_symbol: str = "UNKNOWN",
        price_column: str = "close",
    ) -> List[Alert]:
        """
        Detect price spikes using IQR method.
        
        Args:
            df: DataFrame with OHLCV data
            asset_symbol: Symbol/identifier for the asset
            price_column: Column to analyze (open, close, high, low)
            
        Returns:
            List of Alert objects for detected spikes
        """
        alerts = []
        
        if len(df) < self.config.min_history:
            return alerts
        
        # Get price data
        if self.config.apply_to_returns:
            prices = df[price_column].pct_change().fillna(0)
        else:
            prices = df[price_column]
        
        # Calculate rolling IQR
        for i in range(self.config.min_history, len(df)):
            # Get window
            window_start = max(0, i - self.config.iqr_window + 1)
            window_data = prices.iloc[window_start:i]
            
            if len(window_data) < self.config.min_history:
                continue
            
            # Calculate quartiles
            q1 = window_data.quantile(0.25)
            q3 = window_data.quantile(0.75)
            iqr = q3 - q1
            
            # Define bounds
            lower_bound = q1 - self.config.iqr_multiplier * iqr
            upper_bound = q3 + self.config.iqr_multiplier * iqr
            
            current_price = prices.iloc[i]
            
            # Check if price is outside bounds
            if current_price < lower_bound or current_price > upper_bound:
                is_high = current_price > upper_bound
                bound = upper_bound if is_high else lower_bound
                direction = "above" if is_high else "below"
                
                message = (
                    f"Price spike detected: {price_column} = {current_price:.4f} "
                    f"is {direction} IQR bound ({bound:.4f})"
                )
                
                alert = Alert(
                    severity=AlertSeverity.WARNING,
                    anomaly_type=AnomalyType.PRICE_SPIKE,
                    timestamp=df["timestamp"].iloc[i],
                    bar_index=i,
                    asset_symbol=asset_symbol,
                    measured_value=float(current_price),
                    expected_value=float((q1 + q3) / 2),  # Median
                    threshold_value=float(bound),
                    message=message,
                    metadata={
                        "price": float(current_price),
                        "q1": float(q1),
                        "q3": float(q3),
                        "iqr": float(iqr),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "method": "iqr",
                        "price_column": price_column,
                    },
                )
                alerts.append(alert)
        
        return alerts


# ============================================================================
# Volume Anomaly Detection
# ============================================================================

class VolumeAnomalyDetector:
    """
    Detects unusual volume patterns and anomalies.
    
    Identifies both abnormally high volume (potential market events)
    and abnormally low volume (potential data issues).
    """
    
    def __init__(
        self,
        config: Optional[VolumeDetectionConfig] = None,
    ):
        """
        Initialize VolumeAnomalyDetector.
        
        Args:
            config: VolumeDetectionConfig instance
        """
        self.config = config or VolumeDetectionConfig()
    
    def detect(
        self,
        df: pd.DataFrame,
        asset_symbol: str = "UNKNOWN",
    ) -> List[Alert]:
        """
        Detect volume anomalies in OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            asset_symbol: Symbol/identifier for the asset
            
        Returns:
            List of Alert objects for detected volume anomalies
        """
        alerts = []
        
        if len(df) < self.config.min_history:
            return alerts
        
        volumes = df["volume"]
        
        # Check for zero volume
        if self.config.flag_zero_volume:
            for i in range(len(df)):
                if volumes.iloc[i] == 0 or pd.isna(volumes.iloc[i]):
                    message = f"Zero or missing volume detected"
                    alert = Alert(
                        severity=AlertSeverity.CRITICAL,
                        anomaly_type=AnomalyType.VOLUME_ANOMALY,
                        timestamp=df["timestamp"].iloc[i],
                        bar_index=i,
                        asset_symbol=asset_symbol,
                        measured_value=0.0,
                        expected_value=None,
                        threshold_value=0.0,
                        message=message,
                        metadata={
                            "volume": float(volumes.iloc[i]) if pd.notna(volumes.iloc[i]) else 0,
                            "reason": "zero_or_missing",
                        },
                    )
                    alerts.append(alert)
                    continue
        
        # Calculate rolling baseline for volume
        for i in range(self.config.min_history, len(df)):
            # Get window (exclude current bar to avoid lookahead bias)
            window_start = max(0, i - self.config.baseline_window)
            window_data = volumes.iloc[window_start:i]
            
            if len(window_data) < self.config.min_history:
                continue
            
            # Calculate baseline statistics
            baseline_mean = window_data.mean()
            baseline_std = window_data.std()
            
            if baseline_std == 0:
                continue
            
            current_volume = volumes.iloc[i]
            
            # Calculate z-score
            zscore = (current_volume - baseline_mean) / baseline_std
            
            # Check for high volume anomaly
            if zscore > self.config.high_threshold_sigma:
                message = (
                    f"High volume spike: {current_volume:.0f} "
                    f"(z-score: {zscore:.2f}, baseline: {baseline_mean:.0f})"
                )
                
                alert = Alert(
                    severity=AlertSeverity.INFO,
                    anomaly_type=AnomalyType.VOLUME_SPIKE,
                    timestamp=df["timestamp"].iloc[i],
                    bar_index=i,
                    asset_symbol=asset_symbol,
                    measured_value=float(current_volume),
                    expected_value=float(baseline_mean),
                    threshold_value=self.config.high_threshold_sigma,
                    message=message,
                    metadata={
                        "volume": float(current_volume),
                        "baseline_mean": float(baseline_mean),
                        "baseline_std": float(baseline_std),
                        "zscore": float(zscore),
                        "direction": "high",
                    },
                )
                alerts.append(alert)
            
            # Check for low volume anomaly
            elif zscore < self.config.low_threshold_sigma:
                message = (
                    f"Low volume anomaly: {current_volume:.0f} "
                    f"(z-score: {zscore:.2f}, baseline: {baseline_mean:.0f})"
                )
                
                alert = Alert(
                    severity=AlertSeverity.WARNING,
                    anomaly_type=AnomalyType.VOLUME_ANOMALY,
                    timestamp=df["timestamp"].iloc[i],
                    bar_index=i,
                    asset_symbol=asset_symbol,
                    measured_value=float(current_volume),
                    expected_value=float(baseline_mean),
                    threshold_value=self.config.low_threshold_sigma,
                    message=message,
                    metadata={
                        "volume": float(current_volume),
                        "baseline_mean": float(baseline_mean),
                        "baseline_std": float(baseline_std),
                        "zscore": float(zscore),
                        "direction": "low",
                    },
                )
                alerts.append(alert)
        
        return alerts


# ============================================================================
# Main Anomaly Detector Orchestrator
# ============================================================================

class AnomalyDetector:
    """
    Main orchestrator for comprehensive anomaly detection.
    
    Runs multiple detection methods (gap, spike, volume) and
    aggregates results for reporting.
    """
    
    def __init__(
        self,
        gap_config: Optional[GapDetectionConfig] = None,
        spike_config: Optional[SpikeDetectionConfig] = None,
        volume_config: Optional[VolumeDetectionConfig] = None,
        calendar: Optional[TradingCalendar] = None,
        enable_gap_detection: bool = True,
        enable_zscore_spike_detection: bool = True,
        enable_iqr_spike_detection: bool = True,
        enable_volume_detection: bool = True,
        aggregate_alerts: bool = True,
        aggregation_window: int = 3,
    ):
        """
        Initialize AnomalyDetector.
        
        Args:
            gap_config: Configuration for gap detection
            spike_config: Configuration for spike detection
            volume_config: Configuration for volume detection
            calendar: TradingCalendar for session boundary handling
            enable_gap_detection: Whether to run gap detection
            enable_zscore_spike_detection: Whether to run z-score spike detection
            enable_iqr_spike_detection: Whether to run IQR spike detection
            enable_volume_detection: Whether to run volume detection
            aggregate_alerts: Whether to aggregate consecutive alerts
            aggregation_window: Maximum consecutive bars to group together
        """
        self.gap_config = gap_config or GapDetectionConfig()
        self.spike_config = spike_config or SpikeDetectionConfig()
        self.volume_config = volume_config or VolumeDetectionConfig()
        self.calendar = calendar
        
        self.enable_gap_detection = enable_gap_detection
        self.enable_zscore_spike_detection = enable_zscore_spike_detection
        self.enable_iqr_spike_detection = enable_iqr_spike_detection
        self.enable_volume_detection = enable_volume_detection
        self.aggregate_alerts = aggregate_alerts
        self.aggregation_window = aggregation_window
        
        # Initialize detectors
        self.gap_detector = GapDetector(self.gap_config, self.calendar)
        self.zscore_detector = ZScoreSpikeDetector(self.spike_config)
        self.iqr_detector = IQRSpikeDetector(self.spike_config)
        self.volume_detector = VolumeAnomalyDetector(self.volume_config)
        self.aggregator = AlertAggregator(aggregation_window)
    
    def detect(
        self,
        df: pd.DataFrame,
        asset_symbol: str = "UNKNOWN",
    ) -> List[Alert]:
        """
        Run all enabled anomaly detectors on the data.
        
        Args:
            df: DataFrame with OHLCV data
            asset_symbol: Symbol/identifier for the asset
            
        Returns:
            List of Alert objects, aggregated if enabled
        """
        all_alerts = []
        
        # Run gap detection
        if self.enable_gap_detection:
            gap_alerts = self.gap_detector.detect(df, asset_symbol)
            all_alerts.extend(gap_alerts)
            logger.debug(f"Gap detection found {len(gap_alerts)} alerts")
        
        # Run z-score spike detection
        if self.enable_zscore_spike_detection:
            for col in ["close", "high", "low", "open"]:
                if col in df.columns:
                    zscore_alerts = self.zscore_detector.detect(df, asset_symbol, col)
                    all_alerts.extend(zscore_alerts)
            logger.debug(f"Z-score spike detection found {len(zscore_alerts)} alerts")
        
        # Run IQR spike detection
        if self.enable_iqr_spike_detection:
            for col in ["close", "high", "low", "open"]:
                if col in df.columns:
                    iqr_alerts = self.iqr_detector.detect(df, asset_symbol, col)
                    all_alerts.extend(iqr_alerts)
            logger.debug(f"IQR spike detection found {len(iqr_alerts)} alerts")
        
        # Run volume detection
        if self.enable_volume_detection:
            volume_alerts = self.volume_detector.detect(df, asset_symbol)
            all_alerts.extend(volume_alerts)
            logger.debug(f"Volume detection found {len(volume_alerts)} alerts")
        
        # Sort by bar index for aggregation
        all_alerts.sort(key=lambda a: a.bar_index)
        
        # Aggregate if enabled
        if self.aggregate_alerts and len(all_alerts) > 0:
            all_alerts = self.aggregator.aggregate(all_alerts)
            logger.debug(f"Alert aggregation reduced {len(all_alerts)} alerts")
        
        return all_alerts
    
    def detect_batch(
        self,
        datasets: Dict[str, pd.DataFrame],
    ) -> Dict[str, List[Alert]]:
        """
        Run anomaly detection on multiple datasets.
        
        Args:
            datasets: Dictionary mapping asset symbol to DataFrame
            
        Returns:
            Dictionary mapping asset symbol to list of Alert objects
        """
        results = {}
        for symbol, df in datasets.items():
            results[symbol] = self.detect(df, symbol)
        return results
