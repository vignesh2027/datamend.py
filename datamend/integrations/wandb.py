"""Weights & Biases integration — log datamend reports to W&B runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datamend.core.contract import ContractReport
    from datamend.core.drift import DriftReport
    from datamend.core.repair import RepairReport
    from datamend.pipeline import PipelineResult


def _get_wandb() -> Any:
    """Import and return the wandb module or raise a helpful error."""
    try:
        import wandb
        return wandb
    except ImportError as exc:
        raise ImportError("wandb is required: pip install datamend[wandb]") from exc


def log_repair(
    report: RepairReport,
    *,
    step: int | None = None,
    prefix: str = "datamend/repair",
) -> None:
    """Log AutoRepair metrics to the active W&B run.

    Args:
        report: RepairReport from ``datamend.repair()``.
        step: Optional W&B step index.
        prefix: Metric name prefix.
    """
    wandb = _get_wandb()
    wandb.log(
        {
            f"{prefix}/mend_score_before": report.mend_score_before,
            f"{prefix}/mend_score_after": report.mend_score_after,
            f"{prefix}/issues_found": report.total_issues_found,
            f"{prefix}/rows_affected": report.total_rows_affected,
            f"{prefix}/duration_s": report.duration_seconds,
        },
        step=step,
    )


def log_contract(
    report: ContractReport,
    *,
    step: int | None = None,
    prefix: str = "datamend/contract",
) -> None:
    """Log DataContract metrics to the active W&B run.

    Args:
        report: ContractReport from ``datamend.validate()``.
        step: Optional W&B step index.
        prefix: Metric name prefix.
    """
    wandb = _get_wandb()
    wandb.log(
        {
            f"{prefix}/mend_score": report.mend_score,
            f"{prefix}/violations": len(report.violations),
            f"{prefix}/warnings": len(report.warnings),
            f"{prefix}/passed": int(report.passed),
        },
        step=step,
    )


def log_drift(
    report: DriftReport,
    *,
    step: int | None = None,
    prefix: str = "datamend/drift",
) -> None:
    """Log DriftRadar metrics to the active W&B run.

    Args:
        report: DriftReport from ``datamend.drift()``.
        step: Optional W&B step index.
        prefix: Metric name prefix.
    """
    wandb = _get_wandb()
    metrics: dict[str, Any] = {
        f"{prefix}/mend_score": report.mend_score,
        f"{prefix}/columns_drifted": len(report.columns_drifted),
        f"{prefix}/overall_drifted": int(report.overall_drifted),
    }
    for col, result in report.column_results.items():
        safe = col.replace(" ", "_")[:40]
        metrics[f"{prefix}/{safe}/drift_score"] = result.drift_score
        if result.psi is not None:
            metrics[f"{prefix}/{safe}/psi"] = result.psi
        if result.jsd is not None:
            metrics[f"{prefix}/{safe}/jsd"] = result.jsd
    wandb.log(metrics, step=step)


def log_pipeline_result(
    result: PipelineResult,
    *,
    step: int | None = None,
) -> None:
    """Log a full MendPipeline result to the active W&B run.

    Args:
        result: PipelineResult from ``MendPipeline.transform()``.
        step: Optional W&B step index.
    """
    wandb = _get_wandb()
    wandb.log({"datamend/overall_mend_score": result.overall_mend_score}, step=step)
    log_repair(result.repair_report, step=step)
    if result.contract_report:
        log_contract(result.contract_report, step=step)
    if result.drift_report:
        log_drift(result.drift_report, step=step)
