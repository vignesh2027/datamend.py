"""MendPipeline — unified pipeline chaining all four datamend pillars."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from datamend.core.contract import DataContract, ContractReport
from datamend.core.drift import DriftRadar, DriftReport
from datamend.core.repair import AutoRepair, RepairReport
from datamend.core.trace import FailureTrace, TraceReport


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Aggregated result from a full MendPipeline run."""

    repaired_df: pd.DataFrame
    repair_report: RepairReport
    contract_report: Optional[ContractReport]
    drift_report: Optional[DriftReport]
    trace_report: Optional[TraceReport]
    overall_mend_score: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary(self) -> str:
        """Return a multi-pillar summary string."""
        lines = [
            "=" * 65,
            "datamend MendPipeline — Full Health Report",
            "=" * 65,
            f"  Overall MendScore   : {self.overall_mend_score:.1f}/100",
            "",
            "  [Pillar 1] AutoRepair",
            f"    Issues fixed      : {self.repair_report.total_issues_found}",
            f"    MendScore change  : {self.repair_report.mend_score_before:.1f} → {self.repair_report.mend_score_after:.1f}",
        ]
        if self.contract_report:
            status = "PASSED" if self.contract_report.passed else "FAILED"
            lines += [
                "",
                f"  [Pillar 2] DataContract — {status}",
                f"    Violations        : {len(self.contract_report.violations)}",
                f"    MendScore         : {self.contract_report.mend_score:.1f}",
            ]
        if self.drift_report:
            status = "DRIFT" if self.drift_report.overall_drifted else "STABLE"
            lines += [
                "",
                f"  [Pillar 3] DriftRadar — {status}",
                f"    Columns drifted   : {len(self.drift_report.columns_drifted)}",
                f"    MendScore (drift) : {self.drift_report.mend_score:.1f}",
            ]
        if self.trace_report:
            lines += [
                "",
                "  [Pillar 4] FailureTrace",
                f"    Suspicious rows   : {len(self.trace_report.suspicious_rows)}",
                f"    MendScore         : {self.trace_report.mend_score:.1f}",
            ]
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict (excluding the DataFrame)."""
        result: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "overall_mend_score": round(self.overall_mend_score, 2),
            "repair": self.repair_report.to_dict(),
        }
        if self.contract_report:
            result["contract"] = self.contract_report.to_dict()
        if self.drift_report:
            result["drift"] = self.drift_report.to_dict()
        if self.trace_report:
            result["trace"] = self.trace_report.to_dict()
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# MendPipeline
# ---------------------------------------------------------------------------


class MendPipeline:
    """Unified pipeline that chains AutoRepair → DataContract → DriftRadar → FailureTrace.

    Fit once on training data and call on any new batch to get a complete
    health report, auto-repaired data, contract validation, drift score,
    and failure trace.

    Args:
        repair_strategy: Imputation strategy for AutoRepair.
        null_threshold: Contract null rate threshold (0–1).
        drift_alpha: Significance level for drift tests.
        psi_buckets: Buckets for PSI computation.
        top_k_trace: Top columns to surface in FailureTrace.
        enable_repair: Run AutoRepair. Default True.
        enable_contract: Run DataContract validation. Default True.
        enable_drift: Run DriftRadar. Default True.
        enable_trace: Run FailureTrace (requires a model). Default True.
        fast_mode: Enable fast mode (sampling) for large datasets.
        verbose: Print pillar summaries to stdout.
        plugins: Custom repair plugin instances.
    """

    def __init__(
        self,
        repair_strategy: str = "auto",
        null_threshold: float = 0.05,
        drift_alpha: float = 0.05,
        psi_buckets: int = 10,
        top_k_trace: int = 10,
        enable_repair: bool = True,
        enable_contract: bool = True,
        enable_drift: bool = True,
        enable_trace: bool = True,
        fast_mode: bool = False,
        verbose: bool = True,
        plugins: Optional[List[Any]] = None,
    ) -> None:
        self.enable_repair = enable_repair
        self.enable_contract = enable_contract
        self.enable_drift = enable_drift
        self.enable_trace = enable_trace
        self.verbose = verbose

        self._repair_engine = AutoRepair(
            strategy=repair_strategy,
            fast_mode=fast_mode,
            plugins=plugins or [],
            verbose=False,
        )
        self._contract = DataContract(null_threshold=null_threshold)
        self._drift_radar = DriftRadar(psi_buckets=psi_buckets, alpha=drift_alpha, verbose=False)
        self._tracer = FailureTrace(top_k=top_k_trace, verbose=False)

        self._train_df_clean: Optional[pd.DataFrame] = None
        self._fitted = False

    def fit(self, train_df: pd.DataFrame) -> "MendPipeline":
        """Fit the pipeline on clean training data.

        Repairs the training data, learns its DataContract, and stores a
        clean reference for drift comparison.

        Args:
            train_df: Raw training DataFrame.

        Returns:
            Self, for method chaining.
        """
        if self.enable_repair:
            clean, _ = self._repair_engine.fit_transform(train_df)
        else:
            clean = train_df.copy()

        if self.enable_contract:
            self._contract.fit(clean)

        self._train_df_clean = clean
        self._fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        model: Optional[Any] = None,
        predictions: Optional[Any] = None,
        ground_truth: Optional[Any] = None,
    ) -> PipelineResult:
        """Run all enabled pillars on new data and return a PipelineResult.

        Args:
            df: New batch DataFrame (production or validation data).
            model: Optional model for FailureTrace.
            predictions: Optional predictions for FailureTrace.
            ground_truth: Optional true labels for FailureTrace.

        Returns:
            PipelineResult with repaired data and all pillar reports.
        """
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted first. Call .fit(train_df).")

        # Pillar 1 — AutoRepair
        if self.enable_repair:
            repaired_df, repair_report = self._repair_engine.fit_transform(df)
        else:
            repaired_df = df.copy()
            from datamend.core.repair import RepairAction
            repair_report = RepairReport(
                total_issues_found=0,
                total_rows_affected=0,
                actions=[],
                columns_repaired=[],
                duration_seconds=0.0,
                mend_score_before=100.0,
                mend_score_after=100.0,
            )

        # Pillar 2 — DataContract
        contract_report: Optional[ContractReport] = None
        if self.enable_contract and self._contract._fitted:
            contract_report = self._contract.validate(repaired_df)

        # Pillar 3 — DriftRadar
        drift_report: Optional[DriftReport] = None
        if self.enable_drift and self._train_df_clean is not None:
            shared = list(set(self._train_df_clean.columns) & set(repaired_df.columns))
            if shared:
                drift_report = self._drift_radar.detect(
                    self._train_df_clean, repaired_df, columns=shared
                )

        # Pillar 4 — FailureTrace
        trace_report: Optional[TraceReport] = None
        if self.enable_trace and model is not None and predictions is not None:
            trace_report = self._tracer.trace(model, repaired_df, predictions, ground_truth)

        overall_score = self._compute_overall_mend_score(
            repair_report, contract_report, drift_report, trace_report
        )

        result = PipelineResult(
            repaired_df=repaired_df,
            repair_report=repair_report,
            contract_report=contract_report,
            drift_report=drift_report,
            trace_report=trace_report,
            overall_mend_score=overall_score,
        )

        if self.verbose:
            try:
                from rich.console import Console
                from rich.panel import Panel
                console = Console()
                console.print(Panel(result.summary(), style="bold magenta"))
            except ImportError:
                print(result.summary())

        return result

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        prod_df: Optional[pd.DataFrame] = None,
        model: Optional[Any] = None,
        predictions: Optional[Any] = None,
        ground_truth: Optional[Any] = None,
    ) -> PipelineResult:
        """Fit on training data and immediately transform production data.

        Args:
            train_df: Raw training DataFrame.
            prod_df: Production DataFrame to analyse. If None, analyses train_df itself.
            model: Optional model for FailureTrace.
            predictions: Optional predictions.
            ground_truth: Optional true labels.

        Returns:
            PipelineResult for prod_df (or train_df if prod_df is None).
        """
        self.fit(train_df)
        target = prod_df if prod_df is not None else train_df
        return self.transform(target, model=model, predictions=predictions, ground_truth=ground_truth)

    @staticmethod
    def _compute_overall_mend_score(
        repair: RepairReport,
        contract: Optional[ContractReport],
        drift: Optional[DriftReport],
        trace: Optional[TraceReport],
    ) -> float:
        """Compute a weighted overall MendScore from all pillar reports."""
        scores: List[Tuple[float, float]] = []  # (score, weight)
        scores.append((repair.mend_score_after, 0.35))
        if contract is not None:
            scores.append((contract.mend_score, 0.30))
        if drift is not None:
            # Invert: high drift MendScore = bad; pipeline score = low
            scores.append((max(0.0, 100.0 - drift.mend_score), 0.20))
        if trace is not None:
            scores.append((max(0.0, 100.0 - trace.mend_score), 0.15))

        total_weight = sum(w for _, w in scores)
        if total_weight == 0:
            return 100.0
        return sum(s * w for s, w in scores) / total_weight
