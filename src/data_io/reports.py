"""
Comprehensive validation reporting system.

Aggregates results from all validators (schema, timestamp, missing bars, anomalies)
and generates actionable reports in multiple formats (HTML, JSON, text, DataFrame).

Features:
- Aggregate results from all validator types
- Summary reports with pass/fail status per validator
- Detailed issue logs with context
- Multiple output formats (HTML, JSON, text, DataFrame)
- Statistics and severity breakdown
- Issue filtering and sorting
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import pandas as pd
import json
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Structures
# ============================================================================

class ValidatorType(str, Enum):
    """Types of validators in the pipeline."""
    SCHEMA = "schema"
    TIMESTAMP = "timestamp"
    MISSING_BARS = "missing_bars"
    ANOMALY = "anomaly"


class SeverityLevel(str, Enum):
    """Severity levels for issues."""
    CRITICAL = "critical"  # Blocks promotion
    WARNING = "warning"    # Logs but allows continuation
    INFO = "info"          # Informational only


@dataclass
class DetailedIssue:
    """
    Represents a single detailed issue found during validation.
    
    Attributes:
        asset: Asset identifier
        bar_timestamp: Timestamp of problematic bar
        bar_index: Index of problematic bar
        issue_type: Type of issue (schema, timestamp, gap, spike, etc.)
        severity: Severity level
        measured_value: Actual measured value
        expected_value: Expected value
        threshold_value: Threshold that was exceeded
        message: Detailed human-readable message
        validator_type: Which validator detected this
        context: Dict with surrounding bar info or other context
        metadata: Additional metadata
    """
    asset: str
    bar_timestamp: Optional[Any]
    bar_index: int
    issue_type: str
    severity: SeverityLevel
    message: str
    validator_type: ValidatorType
    measured_value: Optional[float] = None
    expected_value: Optional[float] = None
    threshold_value: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        data = asdict(self)
        data["severity"] = self.severity.value
        data["validator_type"] = self.validator_type.value
        data["bar_timestamp"] = str(self.bar_timestamp) if self.bar_timestamp else None
        return data


@dataclass
class ValidatorResult:
    """
    Result from a single validator.
    
    Attributes:
        validator_type: Type of validator
        asset: Asset being validated
        passed: Whether validation passed
        total_bars_checked: Number of bars checked
        issues: List of detailed issues
        summary: Summary statistics and messages
    """
    validator_type: ValidatorType
    asset: str
    passed: bool
    total_bars_checked: int
    issues: List[DetailedIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def get_issue_count(self) -> int:
        """Get total number of issues."""
        return len(self.issues)
    
    def get_issues_by_severity(self) -> Dict[SeverityLevel, List[DetailedIssue]]:
        """Get issues grouped by severity."""
        grouped = {}
        for severity in SeverityLevel:
            grouped[severity] = [i for i in self.issues if i.severity == severity]
        return grouped
    
    def has_blocking_issues(self) -> bool:
        """Check if validator has critical (blocking) issues."""
        return any(i.severity == SeverityLevel.CRITICAL for i in self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "validator_type": self.validator_type.value,
            "asset": self.asset,
            "passed": self.passed,
            "total_bars_checked": self.total_bars_checked,
            "total_issues": self.get_issue_count(),
            "issues_by_severity": {
                severity.value: len(issues)
                for severity, issues in self.get_issues_by_severity().items()
            },
            "summary": self.summary,
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class ValidationSummary:
    """
    Summary statistics across all validators for an asset.
    
    Attributes:
        asset: Asset identifier
        validation_timestamp: When validation occurred
        overall_passed: Whether all validators passed
        total_bars_validated: Total bars checked
        validators_passed: Count of passed validators
        validators_failed: Count of failed validators
        total_issues: Total issues across all validators
        issues_by_severity: Count of issues by severity
        issues_by_type: Count of issues by type
        pass_rate: Percentage of validators that passed
        blocking_validators: Which validators have blocking issues
        results_by_validator: Results keyed by validator type
    """
    asset: str
    validation_timestamp: datetime
    overall_passed: bool
    total_bars_validated: int
    validators_passed: int
    validators_failed: int
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues_by_type: Dict[str, int]
    pass_rate: float
    blocking_validators: List[ValidatorType]
    results_by_validator: Dict[ValidatorType, ValidatorResult] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "asset": self.asset,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "overall_passed": self.overall_passed,
            "total_bars_validated": self.total_bars_validated,
            "validators_passed": self.validators_passed,
            "validators_failed": self.validators_failed,
            "total_issues": self.total_issues,
            "issues_by_severity": self.issues_by_severity,
            "issues_by_type": self.issues_by_type,
            "pass_rate": round(self.pass_rate, 4),
            "blocking_validators": [v.value for v in self.blocking_validators],
        }


@dataclass
class ValidationReport:
    """
    Comprehensive validation report aggregating all validators.
    
    Attributes:
        asset: Asset being validated
        validation_timestamp: When validation occurred
        overall_passed: Whether all validators passed
        results: List of ValidatorResult objects
        summary: ValidationSummary with aggregate statistics
        detailed_issues: List of all DetailedIssue objects
    """
    asset: str
    validation_timestamp: datetime
    overall_passed: bool
    results: List[ValidatorResult] = field(default_factory=list)
    summary: Optional[ValidationSummary] = None
    detailed_issues: List[DetailedIssue] = field(default_factory=list)
    
    def add_result(self, result: ValidatorResult) -> None:
        """Add a validator result to the report."""
        self.results.append(result)
        self.detailed_issues.extend(result.issues)
    
    def get_critical_issues(self) -> List[DetailedIssue]:
        """Get all critical issues."""
        return [i for i in self.detailed_issues if i.severity == SeverityLevel.CRITICAL]
    
    def get_warning_issues(self) -> List[DetailedIssue]:
        """Get all warning issues."""
        return [i for i in self.detailed_issues if i.severity == SeverityLevel.WARNING]
    
    def get_info_issues(self) -> List[DetailedIssue]:
        """Get all info issues."""
        return [i for i in self.detailed_issues if i.severity == SeverityLevel.INFO]
    
    def get_issues_by_type(self) -> Dict[str, List[DetailedIssue]]:
        """Get issues grouped by type."""
        grouped = {}
        for issue in self.detailed_issues:
            if issue.issue_type not in grouped:
                grouped[issue.issue_type] = []
            grouped[issue.issue_type].append(issue)
        return grouped
    
    def filter_issues(
        self,
        severity: Optional[SeverityLevel] = None,
        issue_type: Optional[str] = None,
        bar_index_range: Optional[Tuple[int, int]] = None,
    ) -> List[DetailedIssue]:
        """
        Filter issues by various criteria.
        
        Args:
            severity: Filter by severity level
            issue_type: Filter by issue type
            bar_index_range: Filter by bar index range (min, max inclusive)
        
        Returns:
            Filtered list of issues
        """
        filtered = self.detailed_issues
        
        if severity is not None:
            filtered = [i for i in filtered if i.severity == severity]
        
        if issue_type is not None:
            filtered = [i for i in filtered if i.issue_type == issue_type]
        
        if bar_index_range is not None:
            min_idx, max_idx = bar_index_range
            filtered = [i for i in filtered if min_idx <= i.bar_index <= max_idx]
        
        return filtered
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "asset": self.asset,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "overall_passed": self.overall_passed,
            "total_issues": len(self.detailed_issues),
            "critical_issues": len(self.get_critical_issues()),
            "warning_issues": len(self.get_warning_issues()),
            "info_issues": len(self.get_info_issues()),
            "summary": self.summary.to_dict() if self.summary else None,
            "results_by_validator": {r.validator_type.value: r.to_dict() for r in self.results},
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert detailed issues to DataFrame for analysis.
        
        Returns:
            DataFrame with issue information
        """
        if not self.detailed_issues:
            return pd.DataFrame()
        
        records = [issue.to_dict() for issue in self.detailed_issues]
        df = pd.DataFrame(records)
        
        # Convert enum columns to categorical
        if "severity" in df.columns:
            df["severity"] = pd.Categorical(df["severity"])
        if "validator_type" in df.columns:
            df["validator_type"] = pd.Categorical(df["validator_type"])
        
        return df


# ============================================================================
# Report Aggregator
# ============================================================================

class ReportAggregator:
    """
    Aggregates validation results into comprehensive reports.
    """
    
    @staticmethod
    def create_report(
        asset: str,
        results: List[ValidatorResult],
    ) -> ValidationReport:
        """
        Create a comprehensive validation report.
        
        Args:
            asset: Asset identifier
            results: List of ValidatorResult objects from all validators
        
        Returns:
            ValidationReport with aggregated results and summary
        """
        validation_timestamp = datetime.utcnow()
        
        # Create report
        report = ValidationReport(
            asset=asset,
            validation_timestamp=validation_timestamp,
            overall_passed=True,
            results=results,
        )
        
        # Add all issues from results
        for result in results:
            report.add_result(result)
            if not result.passed:
                report.overall_passed = False
        
        # Create summary
        report.summary = ReportAggregator._create_summary(asset, validation_timestamp, results)
        
        return report
    
    @staticmethod
    def _create_summary(
        asset: str,
        validation_timestamp: datetime,
        results: List[ValidatorResult],
    ) -> ValidationSummary:
        """Create validation summary."""
        validators_passed = sum(1 for r in results if r.passed)
        validators_failed = len(results) - validators_passed
        overall_passed = validators_failed == 0
        
        # Aggregate issues by severity and type
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)
        
        issues_by_severity = {}
        for severity in SeverityLevel:
            count = sum(1 for i in all_issues if i.severity == severity)
            issues_by_severity[severity.value] = count
        
        issues_by_type = {}
        for issue in all_issues:
            issues_by_type[issue.issue_type] = issues_by_type.get(issue.issue_type, 0) + 1
        
        # Identify blocking validators
        blocking_validators = [
            r.validator_type for r in results if r.has_blocking_issues()
        ]
        
        # Calculate statistics
        total_bars = sum(r.total_bars_checked for r in results)
        total_issues = len(all_issues)
        pass_rate = (validators_passed / len(results) * 100) if results else 100.0
        
        return ValidationSummary(
            asset=asset,
            validation_timestamp=validation_timestamp,
            overall_passed=overall_passed,
            total_bars_validated=total_bars,
            validators_passed=validators_passed,
            validators_failed=validators_failed,
            total_issues=total_issues,
            issues_by_severity=issues_by_severity,
            issues_by_type=issues_by_type,
            pass_rate=pass_rate,
            blocking_validators=blocking_validators,
            results_by_validator={r.validator_type: r for r in results},
        )


# ============================================================================
# Report Generators
# ============================================================================

class JSONReportGenerator:
    """Generates JSON format reports."""
    
    @staticmethod
    def generate(report: ValidationReport) -> str:
        """
        Generate JSON report.
        
        Args:
            report: ValidationReport object
        
        Returns:
            JSON string representation of report
        """
        return json.dumps(report.to_dict(), indent=2, default=str)
    
    @staticmethod
    def save(report: ValidationReport, filepath: Path) -> None:
        """Save JSON report to file."""
        json_content = JSONReportGenerator.generate(report)
        filepath.write_text(json_content, encoding='utf-8')
        logger.info(f"Saved JSON report to {filepath}")


class TextReportGenerator:
    """Generates human-readable text format reports."""
    
    @staticmethod
    def generate(report: ValidationReport) -> str:
        """
        Generate text report.
        
        Args:
            report: ValidationReport object
        
        Returns:
            Human-readable text report
        """
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"VALIDATION REPORT - {report.asset}")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {report.validation_timestamp.isoformat()}")
        lines.append(f"Overall Status: {'✓ PASSED' if report.overall_passed else '✗ FAILED'}")
        lines.append("")
        
        # Summary
        if report.summary:
            lines.append("-" * 80)
            lines.append("SUMMARY")
            lines.append("-" * 80)
            summary = report.summary
            lines.append(f"Total Bars Validated: {summary.total_bars_validated}")
            lines.append(f"Validators Passed: {summary.validators_passed}/{len(report.results)}")
            lines.append(f"Pass Rate: {summary.pass_rate:.2f}%")
            lines.append(f"Total Issues: {summary.total_issues}")
            lines.append("")
            
            lines.append("Issues by Severity:")
            for severity in SeverityLevel:
                count = summary.issues_by_severity.get(severity.value, 0)
                lines.append(f"  {severity.value.upper()}: {count}")
            
            if summary.blocking_validators:
                lines.append("")
                lines.append("BLOCKING VALIDATORS:")
                for validator in summary.blocking_validators:
                    lines.append(f"  - {validator.value}")
            lines.append("")
        
        # Validator Results
        lines.append("-" * 80)
        lines.append("VALIDATOR RESULTS")
        lines.append("-" * 80)
        for result in report.results:
            status = "✓ PASSED" if result.passed else "✗ FAILED"
            lines.append(f"{result.validator_type.value.upper()}: {status}")
            lines.append(f"  Bars Checked: {result.total_bars_checked}")
            lines.append(f"  Issues: {result.get_issue_count()}")
            if result.summary:
                for key, value in result.summary.items():
                    lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Detailed Issues (if any)
        if report.detailed_issues:
            lines.append("-" * 80)
            lines.append("DETAILED ISSUES")
            lines.append("-" * 80)
            
            # Group by severity
            for severity in [SeverityLevel.CRITICAL, SeverityLevel.WARNING, SeverityLevel.INFO]:
                issues = [i for i in report.detailed_issues if i.severity == severity]
                if issues:
                    lines.append(f"\n{severity.value.upper()} ({len(issues)} issues):")
                    for issue in issues:
                        lines.append(f"  Bar {issue.bar_index} ({issue.bar_timestamp}):")
                        lines.append(f"    Type: {issue.issue_type}")
                        lines.append(f"    Validator: {issue.validator_type.value}")
                        lines.append(f"    Message: {issue.message}")
                        if issue.measured_value is not None:
                            lines.append(f"    Measured: {issue.measured_value}")
                        if issue.expected_value is not None:
                            lines.append(f"    Expected: {issue.expected_value}")
                        lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    @staticmethod
    def save(report: ValidationReport, filepath: Path) -> None:
        """Save text report to file."""
        text_content = TextReportGenerator.generate(report)
        filepath.write_text(text_content, encoding='utf-8')
        logger.info(f"Saved text report to {filepath}")


class HTMLReportGenerator:
    """Generates HTML format reports with styling and tables."""
    
    @staticmethod
    def generate(report: ValidationReport) -> str:
        """
        Generate HTML report.
        
        Args:
            report: ValidationReport object
        
        Returns:
            HTML string representation of report
        """
        # Build table rows for summary
        summary_rows = ""
        if report.summary:
            s = report.summary
            summary_rows = f"""
            <tr>
                <td>Total Bars Validated</td>
                <td>{s.total_bars_validated}</td>
            </tr>
            <tr>
                <td>Validators Passed</td>
                <td>{s.validators_passed}/{len(report.results)}</td>
            </tr>
            <tr>
                <td>Pass Rate</td>
                <td>{s.pass_rate:.2f}%</td>
            </tr>
            <tr>
                <td>Total Issues</td>
                <td class="count">{s.total_issues}</td>
            </tr>
            <tr class="critical">
                <td>Critical Issues</td>
                <td>{s.issues_by_severity.get('critical', 0)}</td>
            </tr>
            <tr class="warning">
                <td>Warning Issues</td>
                <td>{s.issues_by_severity.get('warning', 0)}</td>
            </tr>
            <tr class="info">
                <td>Info Issues</td>
                <td>{s.issues_by_severity.get('info', 0)}</td>
            </tr>
            """
        
        # Build validator results table
        validator_rows = ""
        for result in report.results:
            status_class = "passed" if result.passed else "failed"
            status_text = "✓ PASSED" if result.passed else "✗ FAILED"
            validator_rows += f"""
            <tr class="{status_class}">
                <td>{result.validator_type.value}</td>
                <td>{status_text}</td>
                <td>{result.total_bars_checked}</td>
                <td>{result.get_issue_count()}</td>
            </tr>
            """
        
        # Build issues table
        issues_rows = ""
        if report.detailed_issues:
            for issue in report.detailed_issues:
                severity_class = issue.severity.value.lower()
                issues_rows += f"""
            <tr class="{severity_class}">
                <td>{issue.bar_index}</td>
                <td>{issue.bar_timestamp if issue.bar_timestamp else 'N/A'}</td>
                <td>{issue.issue_type}</td>
                <td>{issue.severity.value}</td>
                <td>{issue.validator_type.value}</td>
                <td>{issue.message}</td>
            </tr>
            """
        
        overall_status = "PASSED" if report.overall_passed else "FAILED"
        overall_class = "passed" if report.overall_passed else "failed"
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Validation Report - {report.asset}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-left: 4px solid #007bff;
            padding-left: 10px;
        }}
        .header {{
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .status {{
            font-size: 18px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .status.passed {{
            color: #28a745;
        }}
        .status.failed {{
            color: #dc3545;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th {{
            background-color: #007bff;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f9f9f9;
        }}
        tr.passed {{
            background-color: #d4edda;
        }}
        tr.failed {{
            background-color: #f8d7da;
        }}
        tr.critical {{
            background-color: #f8d7da;
        }}
        tr.warning {{
            background-color: #fff3cd;
        }}
        tr.info {{
            background-color: #d1ecf1;
        }}
        .timestamp {{
            color: #666;
            font-size: 14px;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Validation Report - {report.asset}</h1>
        
        <div class="header">
            <div class="timestamp">Generated: {report.validation_timestamp.isoformat()}</div>
            <div class="status {overall_class}">Overall Status: {overall_status}</div>
        </div>
        
        <h2>Summary</h2>
        <table>
            <tbody>
                {summary_rows}
            </tbody>
        </table>
        
        <h2>Validator Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Validator</th>
                    <th>Status</th>
                    <th>Bars Checked</th>
                    <th>Issues</th>
                </tr>
            </thead>
            <tbody>
                {validator_rows}
            </tbody>
        </table>
        
        <h2>Detailed Issues</h2>
        {f'''
        <table>
            <thead>
                <tr>
                    <th>Bar Index</th>
                    <th>Timestamp</th>
                    <th>Issue Type</th>
                    <th>Severity</th>
                    <th>Validator</th>
                    <th>Message</th>
                </tr>
            </thead>
            <tbody>
                {issues_rows if issues_rows else '<tr><td colspan="6">No issues found</td></tr>'}
            </tbody>
        </table>
        ''' if report.detailed_issues else '<p>No issues found</p>'}
        
        <div class="footer">
            <p>This report was automatically generated by the validation pipeline.</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    @staticmethod
    def save(report: ValidationReport, filepath: Path) -> None:
        """Save HTML report to file."""
        html_content = HTMLReportGenerator.generate(report)
        filepath.write_text(html_content, encoding='utf-8')
        logger.info(f"Saved HTML report to {filepath}")
