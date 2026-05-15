"""DVC integration — log datamend reports as DVC metrics and plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datamend.core.contract import ContractReport
    from datamend.core.drift import DriftReport
    from datamend.core.repair import RepairReport
    from datamend.pipeline import PipelineResult


def save_repair_metrics(
    report: RepairReport,
    *,
    metrics_path: str = "datamend_repair_metrics.json",
) -> None:
    """Write AutoRepair metrics to a DVC-trackable JSON metrics file.

    Args:
        report: RepairReport from ``datamend.repair()``.
        metrics_path: Path for the DVC metrics JSON file.
    """
    metrics = {
        "mend_score_before": report.mend_score_before,
        "mend_score_after": report.mend_score_after,
        "issues_found": report.total_issues_found,
        "rows_affected": report.total_rows_affected,
        "duration_s": report.duration_seconds,
    }
    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_drift_metrics(
    report: DriftReport,
    *,
    metrics_path: str = "datamend_drift_metrics.json",
    plots_path: str = "datamend_drift_plots.json",
) -> None:
    """Write DriftRadar metrics and column-level drift scores for DVC tracking.

    Args:
        report: DriftReport from ``datamend.drift()``.
        metrics_path: Path for the summary metrics JSON file.
        plots_path: Path for the per-column drift scores (DVC plots format).
    """
    summary = {
        "mend_score": report.mend_score,
        "columns_drifted": len(report.columns_drifted),
        "overall_drifted": report.overall_drifted,
    }
    Path(metrics_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plots = [
        {
            "column": col,
            "drift_score": r.drift_score,
            "psi": r.psi,
            "jsd": r.jsd,
            "severity": r.severity,
        }
        for col, r in report.column_results.items()
    ]
    Path(plots_path).write_text(json.dumps(plots, indent=2, default=str), encoding="utf-8")


def save_contract_metrics(
    report: ContractReport,
    *,
    metrics_path: str = "datamend_contract_metrics.json",
) -> None:
    """Write DataContract validation metrics to a DVC-trackable file.

    Args:
        report: ContractReport from ``datamend.validate()``.
        metrics_path: Output path.
    """
    metrics = {
        "mend_score": report.mend_score,
        "passed": report.passed,
        "violations": len(report.violations),
        "warnings": len(report.warnings),
        "columns_checked": report.columns_checked,
    }
    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_pipeline_result(
    result: PipelineResult,
    *,
    output_dir: str = "datamend_dvc",
) -> None:
    """Save all pipeline reports to a directory for DVC tracking.

    Args:
        result: PipelineResult from ``MendPipeline.transform()``.
        output_dir: Directory to write all metric and plot files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_repair_metrics(
        result.repair_report,
        metrics_path=str(Path(output_dir) / "repair_metrics.json"),
    )
    if result.contract_report:
        save_contract_metrics(
            result.contract_report,
            metrics_path=str(Path(output_dir) / "contract_metrics.json"),
        )
    if result.drift_report:
        save_drift_metrics(
            result.drift_report,
            metrics_path=str(Path(output_dir) / "drift_metrics.json"),
            plots_path=str(Path(output_dir) / "drift_plots.json"),
        )
    summary = {
        "overall_mend_score": result.overall_mend_score,
        "timestamp": result.timestamp,
    }
    (Path(output_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
