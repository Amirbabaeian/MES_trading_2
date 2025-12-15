"""
Progress tracking and state persistence for data ingestion jobs.

This module provides utilities for tracking ingestion progress and persisting
state to enable resumable jobs that can recover from interruptions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)


class ProgressState:
    """Represents the state of a data ingestion job."""
    
    def __init__(
        self,
        asset: str,
        timeframe: str,
        last_fetched_timestamp: Optional[datetime] = None,
        last_fetch_time: Optional[datetime] = None,
        total_bars_fetched: int = 0,
        total_errors: int = 0,
        status: str = "pending"  # pending, in_progress, completed, failed
    ):
        """
        Initialize progress state.
        
        Args:
            asset: Asset symbol (e.g., 'ES', 'MES', 'VIX').
            timeframe: Timeframe specification (e.g., '1min', '5min', '1D').
            last_fetched_timestamp: Latest timestamp successfully ingested.
            last_fetch_time: When the last fetch occurred.
            total_bars_fetched: Total bars fetched so far.
            total_errors: Total errors encountered.
            status: Current status of the ingestion.
        """
        self.asset = asset
        self.timeframe = timeframe
        self.last_fetched_timestamp = last_fetched_timestamp
        self.last_fetch_time = last_fetch_time
        self.total_bars_fetched = total_bars_fetched
        self.total_errors = total_errors
        self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        return {
            'asset': self.asset,
            'timeframe': self.timeframe,
            'last_fetched_timestamp': (
                self.last_fetched_timestamp.isoformat()
                if self.last_fetched_timestamp else None
            ),
            'last_fetch_time': (
                self.last_fetch_time.isoformat()
                if self.last_fetch_time else None
            ),
            'total_bars_fetched': self.total_bars_fetched,
            'total_errors': self.total_errors,
            'status': self.status,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ProgressState':
        """Create ProgressState from dictionary."""
        return ProgressState(
            asset=data['asset'],
            timeframe=data['timeframe'],
            last_fetched_timestamp=(
                datetime.fromisoformat(data['last_fetched_timestamp'])
                if data.get('last_fetched_timestamp') else None
            ),
            last_fetch_time=(
                datetime.fromisoformat(data['last_fetch_time'])
                if data.get('last_fetch_time') else None
            ),
            total_bars_fetched=data.get('total_bars_fetched', 0),
            total_errors=data.get('total_errors', 0),
            status=data.get('status', 'pending'),
        )


class ProgressTracker:
    """
    Tracks progress of data ingestion jobs with state persistence.
    
    Maintains a state file that records the last successful fetch timestamp
    for each asset/timeframe combination. This enables:
    - Resumable jobs (can restart from last successful point)
    - Incremental updates (fetch only new data)
    - Progress monitoring (track overall completion)
    """
    
    def __init__(self, state_file: str = ".ingestion_state.json"):
        """
        Initialize progress tracker.
        
        Args:
            state_file: Path to JSON file for persisting state.
        """
        self.state_file = Path(state_file)
        self.states: Dict[str, ProgressState] = {}
        self._load_state()
    
    def _load_state(self) -> None:
        """Load state from file if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    for key, state_data in data.items():
                        self.states[key] = ProgressState.from_dict(state_data)
                logger.info(f"Loaded progress state from {self.state_file}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load progress state: {e}. Starting fresh.")
                self.states = {}
        else:
            logger.debug(f"No existing progress state file at {self.state_file}")
    
    def _save_state(self) -> None:
        """Persist state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                data = {key: state.to_dict() for key, state in self.states.items()}
                json.dump(data, f, indent=2)
            logger.debug(f"Saved progress state to {self.state_file}")
        except IOError as e:
            logger.error(f"Failed to save progress state: {e}")
    
    def _make_key(self, asset: str, timeframe: str) -> str:
        """Create a unique key for asset/timeframe combination."""
        return f"{asset}_{timeframe}"
    
    def get_state(self, asset: str, timeframe: str) -> ProgressState:
        """
        Get progress state for asset/timeframe.
        
        Args:
            asset: Asset symbol.
            timeframe: Timeframe specification.
            
        Returns:
            ProgressState for the asset/timeframe combination.
        """
        key = self._make_key(asset, timeframe)
        if key not in self.states:
            self.states[key] = ProgressState(asset, timeframe)
        return self.states[key]
    
    def update_state(
        self,
        asset: str,
        timeframe: str,
        last_fetched_timestamp: Optional[datetime] = None,
        bars_fetched: int = 0,
        error_occurred: bool = False,
        status: str = "completed"
    ) -> None:
        """
        Update progress state for an asset/timeframe.
        
        Args:
            asset: Asset symbol.
            timeframe: Timeframe specification.
            last_fetched_timestamp: Latest timestamp ingested.
            bars_fetched: Number of bars fetched in this update.
            error_occurred: Whether an error occurred.
            status: Status to set ('pending', 'in_progress', 'completed', 'failed').
        """
        key = self._make_key(asset, timeframe)
        state = self.get_state(asset, timeframe)
        
        if last_fetched_timestamp:
            state.last_fetched_timestamp = last_fetched_timestamp
        
        state.last_fetch_time = datetime.now(tz=pytz.UTC)
        state.total_bars_fetched += bars_fetched
        
        if error_occurred:
            state.total_errors += 1
        
        state.status = status
        
        self.states[key] = state
        self._save_state()
        
        logger.info(
            f"Updated progress: {asset}/{timeframe} - "
            f"status={status}, bars={bars_fetched}, total={state.total_bars_fetched}"
        )
    
    def get_all_states(self) -> List[ProgressState]:
        """Get all tracked states."""
        return list(self.states.values())
    
    def reset_state(self, asset: str, timeframe: str) -> None:
        """Reset state for asset/timeframe (useful for restarting from scratch)."""
        key = self._make_key(asset, timeframe)
        if key in self.states:
            self.states[key] = ProgressState(asset, timeframe)
            self._save_state()
            logger.info(f"Reset progress state for {asset}/{timeframe}")
    
    def reset_all(self) -> None:
        """Reset all tracked states."""
        self.states = {}
        self._save_state()
        logger.info("Reset all progress states")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all progress states."""
        total_bars = sum(s.total_bars_fetched for s in self.states.values())
        total_errors = sum(s.total_errors for s in self.states.values())
        completed = sum(1 for s in self.states.values() if s.status == "completed")
        failed = sum(1 for s in self.states.values() if s.status == "failed")
        
        return {
            'total_assets_tracked': len(self.states),
            'total_bars_fetched': total_bars,
            'total_errors': total_errors,
            'completed_count': completed,
            'failed_count': failed,
            'pending_count': len(self.states) - completed - failed,
            'assets': [
                {
                    'key': self._make_key(s.asset, s.timeframe),
                    'asset': s.asset,
                    'timeframe': s.timeframe,
                    'status': s.status,
                    'bars_fetched': s.total_bars_fetched,
                    'errors': s.total_errors,
                    'last_fetched': s.last_fetched_timestamp.isoformat() if s.last_fetched_timestamp else None,
                }
                for s in self.states.values()
            ]
        }
