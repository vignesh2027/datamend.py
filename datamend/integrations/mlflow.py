"""MLflow integration — log datamend reports as MLflow run artifacts and metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from datamend.core.contract import ContractReport
    from datamend.core.drift import DriftReport
    from datamend.core.repair import RepairReport
    from datamend.core.trace import TraceReport
    from datamend.pipeline import PipelineResult


def log_repair(
    report: "RepairReport",
    *,
    run_id: Optional[str] = None,
    prefix: str = "datamend.repair",
) -> None:
    """Log AutoRepair metrics and the repair report JSON to an active MLflow run.

    Args:
        report: RepairReport from ``datamend.repair()``.
        run_id: Optional MLflow run ID. If None, uses the currently active run.
        prefix: Metric name prefix.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("MLflow is required: pip install datamend[mlflow]") from exc

    with _maybe_run(run_id):
        mlflow.log_metrics(
            {
                f"{prefix}.mend_score_before": report.mend_score_before,
                f"{prefix}.mend_score_after": report.mend_score_after,
                f"{prefix}.issues_found": report.total_issues_found,
                f"{prefix}.rows_affected": report.total_rows_affected,
                f"{prefix}.duration_s": report.duration_seconds,
            }
        )
        import json, tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_repair_report.json", delete=False
        ) as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
            tmp_path = fh.name
        try:
            mlflow.log_artifact(tmp_path, artifact_path="datamend")
        finally:
            os.unlink(tmp_path)


def log_contract(
    report: "ContractReport",
    *,
    run_id: Optional[str] = None,
    prefix: str = "datamend.contract",
) -> None:
    """Log DataContract validation metrics to MLflow.

    Args:
        report: ContractReport from ``datamend.validate()``.
        run_id: Optional MLflow run ID.
        prefix: Metric name prefix.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("MLflow is required: pip install datamend[mlflow]") from exc

    with _maybe_run(run_id):
        mlflow.log_metrics(
            {
                f"{prefix}.mend_score": report.mend_score,
                f"{prefix}.violations": len(report.violations),
                f"{prefix}.warnings": len(report.warnings),
                f"{prefix}.passed": int(report.passed),
            }
        )
        import json, tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_contract_report.json", delete=False
        ) as fh:
            fh.write(report.to_json())
            tmp_path = fh.name
        try:
            mlflow.log_artifact(tmp_path, artifact_path="datamend")
        finally:
            os.unlink(tmp_path)


def log_drift(
    report: "DriftReport",
    *,
    run_id: Optional[str] = None,
    prefix: str = "datamend.drift",
) -> None:
    """Log DriftRadar metrics to MLflow.

    Args:
        report: DriftReport from ``datamend.drift()``.
        run_id: Optional MLflow run ID.
        prefix: Metric name prefix.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("MLflow is required: pip install datamend[mlflow]") from exc

    with _maybe_run(run_id):
        mlflow.log_metrics(
            {
                f"{prefix}.mend_score": report.mend_score,
                f"{prefix}.columns_drifted": len(report.columns_drifted),
                f"{prefix}.overall_drifted": int(report.overall_drifted),
            }
        )
        for col, result in report.column_results.items():
            safe_col = col.replace(" ", "_")[:40]
            metrics: dict[str, float] = {f"{prefix}.{safe_col}.drift_score": result.drift_score}
            if result.psi is not None:
                metrics[f"{prefix}.{safe_col}.psi"] = result.psi
            if result.jsd is not None:
                metrics[f"{prefix}.{safe_col}.jsd"] = result.jsd
            mlflow.log_metrics(metrics)


def log_pipeline_result(
    result: "PipelineResult",
    *,
    run_id: Optional[str] = None,
) -> None:
    """Log a full MendPipeline result (all four pillars) to MLflow.

    Args:
        result: PipelineResult from ``MendPipeline.transform()``.
        run_id: Optional MLflow run ID.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("MLflow is required: pip install datamend[mlflow]") from exc

    with _maybe_run(run_id):
        mlflow.log_metric("datamend.overall_mend_score", result.overall_mend_score)
        log_repair(result.repair_report, run_id=run_id)
        if result.contract_report:
            log_contract(result.contract_report, run_id=run_id)
        if result.drift_report:
            log_drift(result.drift_report, run_id=run_id)


def _maybe_run(run_id: Optional[str]) -> Any:
    """Context manager that either uses an existing run or the active run."""
    import mlflow

    if run_id:
        return mlflow.start_run(run_id=run_id, nested=True)
    return _NullContext()


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *args: Any) -> None:
        pass
