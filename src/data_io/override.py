"""
Manual override system for validation failures.

Allows authorized users to override blocking validation failures with justification.
Creates audit trails for reproducibility and accountability.

Features:
- Override records with timestamps and justification
- User tracking and authorization checks
- Audit trail generation
- Override state persistence
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OverrideStatus(str, Enum):
    """Status of an override."""
    APPROVED = "approved"
    PENDING = "pending"
    REJECTED = "rejected"
    REVOKED = "revoked"


@dataclass
class OverrideRecord:
    """
    Records a manual override of a validation failure.
    
    Attributes:
        asset: Asset being overridden
        validator_type: Validator being overridden
        override_timestamp: When override was created
        overridden_by: User who created the override
        justification: Reason for override
        approved_by: User who approved the override (if applicable)
        approval_timestamp: When override was approved
        status: Current override status
        data_version: Version of data being overridden
        affected_bars: List of affected bar indices
        metadata: Additional context
    """
    asset: str
    validator_type: str
    override_timestamp: datetime
    overridden_by: str
    justification: str
    status: OverrideStatus
    data_version: Optional[str] = None
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    affected_bars: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def approve(self, approved_by: str) -> None:
        """Approve this override."""
        self.approved_by = approved_by
        self.approval_timestamp = datetime.utcnow()
        self.status = OverrideStatus.APPROVED
    
    def reject(self) -> None:
        """Reject this override."""
        self.status = OverrideStatus.REJECTED
    
    def revoke(self) -> None:
        """Revoke a previously approved override."""
        self.status = OverrideStatus.REVOKED
        self.approval_timestamp = None
        self.approved_by = None
    
    def is_valid(self) -> bool:
        """Check if override is currently valid (approved and not revoked)."""
        return self.status == OverrideStatus.APPROVED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["override_timestamp"] = self.override_timestamp.isoformat()
        data["approval_timestamp"] = (
            self.approval_timestamp.isoformat()
            if self.approval_timestamp
            else None
        )
        return data


@dataclass
class ValidationMetadata:
    """
    Metadata about a validation run.
    
    Attributes:
        asset: Asset being validated
        validation_timestamp: When validation occurred
        validation_state: Current state of validation
        passed: Whether validation passed
        total_issues: Total issues found
        critical_issues: Count of critical issues
        overrides: List of override records
        override_count: Number of overrides applied
    """
    asset: str
    validation_timestamp: datetime
    validation_state: str  # pending, running, passed, failed-blocked, failed-overridden
    passed: bool
    total_issues: int
    critical_issues: int
    overrides: List[OverrideRecord] = field(default_factory=list)
    override_count: int = 0
    
    def can_promote(self) -> bool:
        """Check if data can be promoted to cleaned layer."""
        if self.validation_state == "passed":
            return True
        if self.validation_state == "failed-overridden":
            # Check all overrides are approved
            return all(o.is_valid() for o in self.overrides)
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "asset": self.asset,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "validation_state": self.validation_state,
            "passed": self.passed,
            "total_issues": self.total_issues,
            "critical_issues": self.critical_issues,
            "override_count": self.override_count,
            "overrides": [o.to_dict() for o in self.overrides],
        }


class OverrideManager:
    """
    Manages validation overrides with persistence and audit trail.
    """
    
    def __init__(self, metadata_dir: Optional[Path] = None):
        """
        Initialize OverrideManager.
        
        Args:
            metadata_dir: Directory to store override/metadata files
        """
        self.metadata_dir = metadata_dir or Path("./validation_metadata")
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.overrides: Dict[str, List[OverrideRecord]] = {}
    
    def create_override(
        self,
        asset: str,
        validator_type: str,
        overridden_by: str,
        justification: str,
        data_version: Optional[str] = None,
        affected_bars: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OverrideRecord:
        """
        Create an override record.
        
        Args:
            asset: Asset identifier
            validator_type: Validator being overridden
            overridden_by: User creating the override
            justification: Reason for override
            data_version: Version of data (for tracking)
            affected_bars: Bar indices affected by override
            metadata: Additional context
        
        Returns:
            OverrideRecord object
        """
        override = OverrideRecord(
            asset=asset,
            validator_type=validator_type,
            override_timestamp=datetime.utcnow(),
            overridden_by=overridden_by,
            justification=justification,
            status=OverrideStatus.PENDING,
            data_version=data_version,
            affected_bars=affected_bars or [],
            metadata=metadata or {},
        )
        
        logger.info(
            f"Created override for {asset}/{validator_type} by {overridden_by}: {justification}"
        )
        
        return override
    
    def approve_override(
        self,
        override: OverrideRecord,
        approved_by: str,
    ) -> None:
        """Approve an override."""
        override.approve(approved_by)
        logger.info(
            f"Approved override for {override.asset}/{override.validator_type} by {approved_by}"
        )
    
    def register_override(self, asset: str, override: OverrideRecord) -> None:
        """Register an override for an asset."""
        if asset not in self.overrides:
            self.overrides[asset] = []
        self.overrides[asset].append(override)
    
    def get_active_overrides(self, asset: str) -> List[OverrideRecord]:
        """Get all active (approved) overrides for an asset."""
        if asset not in self.overrides:
            return []
        return [o for o in self.overrides[asset] if o.is_valid()]
    
    def get_overrides(self, asset: str) -> List[OverrideRecord]:
        """Get all overrides for an asset."""
        return self.overrides.get(asset, [])
    
    def save_metadata(
        self,
        asset: str,
        metadata: ValidationMetadata,
    ) -> None:
        """
        Save validation metadata to file.
        
        Args:
            asset: Asset identifier
            metadata: ValidationMetadata object
        """
        metadata_file = self.metadata_dir / f"{asset}_validation_metadata.json"
        json_data = json.dumps(metadata.to_dict(), indent=2, default=str)
        metadata_file.write_text(json_data, encoding='utf-8')
        logger.info(f"Saved validation metadata to {metadata_file}")
    
    def load_metadata(self, asset: str) -> Optional[ValidationMetadata]:
        """
        Load validation metadata from file.
        
        Args:
            asset: Asset identifier
        
        Returns:
            ValidationMetadata object or None if not found
        """
        metadata_file = self.metadata_dir / f"{asset}_validation_metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            data = json.loads(metadata_file.read_text(encoding='utf-8'))
            
            # Parse override records
            overrides = [
                OverrideRecord(
                    asset=o["asset"],
                    validator_type=o["validator_type"],
                    override_timestamp=datetime.fromisoformat(o["override_timestamp"]),
                    overridden_by=o["overridden_by"],
                    justification=o["justification"],
                    status=OverrideStatus(o["status"]),
                    data_version=o.get("data_version"),
                    approved_by=o.get("approved_by"),
                    approval_timestamp=(
                        datetime.fromisoformat(o["approval_timestamp"])
                        if o.get("approval_timestamp")
                        else None
                    ),
                    affected_bars=o.get("affected_bars", []),
                    metadata=o.get("metadata", {}),
                )
                for o in data.get("overrides", [])
            ]
            
            return ValidationMetadata(
                asset=data["asset"],
                validation_timestamp=datetime.fromisoformat(data["validation_timestamp"]),
                validation_state=data["validation_state"],
                passed=data["passed"],
                total_issues=data["total_issues"],
                critical_issues=data["critical_issues"],
                overrides=overrides,
                override_count=data.get("override_count", 0),
            )
        except Exception as e:
            logger.error(f"Failed to load metadata for {asset}: {e}")
            return None
