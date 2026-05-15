"""FailureTrace engine — row-level and column-level model failure attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RowFailure:
    """Diagnosis for a single suspicious row."""

    row_index: Any
    suspicion_score: float          # 0–100; higher = more likely failure
    top_columns: list[str]          # columns most responsible for this row's anomaly
    data_quality_score: float       # 0–100; 0 = terrible quality for this row
    model_confidence: float | None  # if available from model
    reason: str


@dataclass
class ColumnAttribution:
    """Column-level root-cause attribution."""

    column: str
    importance_score: float     # 0–100; model + data-quality combined
    data_quality_contribution: float
    model_contribution: float
    anomaly_rate: float         # fraction of rows where this column is anomalous


@dataclass
class TraceReport:
    """Complete FailureTrace result."""

    total_rows: int
    suspicious_rows: list[RowFailure]
    column_attributions: list[ColumnAttribution]
    mend_score: float           # 0=no failures traced, 100=severe widespread failures
    top_failure_columns: list[str]
    data_quality_failure_pct: float
    model_failure_pct: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict."""
        return {
            "timestamp": self.timestamp,
            "total_rows": self.total_rows,
            "suspicious_row_count": len(self.suspicious_rows),
            "mend_score": round(self.mend_score, 2),
            "top_failure_columns": self.top_failure_columns,
            "data_quality_failure_pct": round(self.data_quality_failure_pct, 2),
            "model_failure_pct": round(self.model_failure_pct, 2),
            "column_attributions": [
                {
                    "column": ca.column,
                    "importance_score": round(ca.importance_score, 2),
                    "data_quality_contribution": round(ca.data_quality_contribution, 2),
                    "model_contribution": round(ca.model_contribution, 2),
                    "anomaly_rate": round(ca.anomaly_rate, 4),
                }
                for ca in self.column_attributions
            ],
            "suspicious_rows": [
                {
                    "row_index": r.row_index,
                    "suspicion_score": round(r.suspicion_score, 2),
                    "top_columns": r.top_columns,
                    "data_quality_score": round(r.data_quality_score, 2),
                    "model_confidence": r.model_confidence,
                    "reason": r.reason,
                }
                for r in self.suspicious_rows[:50]
            ],
        }

    def summary(self) -> str:
        """Return a human-readable failure trace summary."""
        lines = [
            "=" * 65,
            "datamend FailureTrace Report",
            "=" * 65,
            f"  Total rows analysed     : {self.total_rows}",
            f"  Suspicious rows found   : {len(self.suspicious_rows)}",
            f"  MendScore (failures)    : {self.mend_score:.1f}/100",
            f"  Data quality failures   : {self.data_quality_failure_pct:.1f}%",
            f"  Model confidence issues : {self.model_failure_pct:.1f}%",
            "",
            "  Top failure-causing columns:",
        ]
        for ca in self.column_attributions[:self.top_failure_columns.__len__()]:
            lines.append(
                f"    {ca.column}: importance={ca.importance_score:.1f}, "
                f"dq_contrib={ca.data_quality_contribution:.1f}, "
                f"model_contrib={ca.model_contribution:.1f}, "
                f"anomaly_rate={ca.anomaly_rate:.2%}"
            )
        if self.suspicious_rows:
            lines.append("")
            lines.append("  Top 5 most suspicious rows:")
            for r in self.suspicious_rows[:5]:
                lines.append(
                    f"    Row {r.row_index}: score={r.suspicion_score:.1f}, "
                    f"reason='{r.reason}', columns={r.top_columns}"
                )
        lines.append("=" * 65)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_model_type(model: Any) -> str:
    """Identify the model framework."""
    module = type(model).__module__
    if "sklearn" in module:
        return "sklearn"
    if "xgboost" in module:
        return "xgboost"
    if "lightgbm" in module:
        return "lightgbm"
    if "torch" in module:
        return "torch"
    return "unknown"


def _get_feature_importances(
    model: Any, df: pd.DataFrame, model_type: str
) -> np.ndarray | None:
    """Extract normalised feature importances from a model (0–1 per feature)."""
    try:
        if hasattr(model, "feature_importances_"):
            fi = np.array(model.feature_importances_, dtype=float)
            total = fi.sum()
            return fi / total if total > 0 else fi
        if hasattr(model, "coef_"):
            coef = np.abs(np.array(model.coef_)).flatten()
            coef = coef[: df.shape[1]]  # guard against multi-class
            total = coef.sum()
            return coef / total if total > 0 else coef
        if model_type == "torch":
            return None  # PyTorch requires autograd; use surrogate below
    except Exception:
        return None
    return None


def _surrogate_importances(
    model: Any, df: pd.DataFrame, predictions: np.ndarray
) -> np.ndarray:
    """Approximate feature importances via a surrogate decision tree.

    Trains a DecisionTreeRegressor/Classifier on (X, predictions) and reads
    its feature_importances_ as a proxy for the black-box model.
    """
    try:
        from sklearn.tree import DecisionTreeRegressor  # type: ignore

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] == 0 or len(df) < 10:
            return np.ones(df.shape[1]) / max(df.shape[1], 1)

        x_vals = numeric_df.fillna(numeric_df.median()).values
        y = predictions.astype(float) if predictions.ndim == 1 else predictions[:, 0].astype(float)
        surrogate = DecisionTreeRegressor(max_depth=5, random_state=42)
        surrogate.fit(x_vals, y[: len(x_vals)])
        fi = surrogate.feature_importances_
        # Map back to full column space
        full_fi = np.zeros(df.shape[1])
        numeric_idx = [df.columns.get_loc(c) for c in numeric_df.columns]
        for i, idx in enumerate(numeric_idx):
            full_fi[idx] = fi[i] if i < len(fi) else 0.0
        total = full_fi.sum()
        return full_fi / total if total > 0 else full_fi
    except ImportError:
        return np.ones(df.shape[1]) / max(df.shape[1], 1)
    except Exception:
        return np.ones(df.shape[1]) / max(df.shape[1], 1)


def _row_data_quality_score(row: pd.Series, global_stats: dict[str, Any]) -> float:
    """Score the data quality of a single row (0=terrible, 100=perfect)."""
    issues = 0
    checks = 0

    for i, col in enumerate(row.index):
        raw = row.iloc[i]
        val = raw.item() if hasattr(raw, "item") else raw
        stats_col = global_stats.get(str(col), {})
        checks += 1

        if pd.isna(val):
            issues += 1
            continue

        if isinstance(val, (int, float)):
            mean_v = stats_col.get("mean")
            std_v = stats_col.get("std")
            if mean_v is not None and std_v is not None and std_v > 0:
                z = abs((val - mean_v) / std_v)
                if z > 4.0:
                    issues += 0.5

    if checks == 0:
        return 100.0
    return max(0.0, 100.0 * (1.0 - issues / checks))


def _predict_confidence(model: Any, df: pd.DataFrame, model_type: str) -> np.ndarray | None:
    """Extract per-row model confidence scores (probability of top class or abs score)."""
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)
            return proba.max(axis=1)
        if hasattr(model, "predict"):
            raw = model.predict(df)
            if np.issubdtype(raw.dtype, np.floating):
                # Normalise regression scores: use distance from median as inverse confidence
                median = float(np.median(raw))
                std = float(np.std(raw)) or 1.0
                z = np.abs((raw - median) / std)
                return np.clip(1.0 - z / 5.0, 0.0, 1.0)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# FailureTrace engine
# ---------------------------------------------------------------------------


class FailureTrace:
    """Root-cause attribution for model prediction failures.

    Identifies suspicious rows and the columns most responsible for failures,
    combining data-quality anomaly detection with model-level attribution
    (feature importances or surrogate modelling for black-box models).

    Args:
        top_k: Number of top columns to surface in the report.
        verbose: Print summary to stdout.
    """

    def __init__(self, top_k: int = 10, verbose: bool = True) -> None:
        self.top_k = top_k
        self.verbose = verbose
        self._model: Any | None = None
        self._train_df: pd.DataFrame | None = None
        self._train_labels: Any | None = None

    def fit(
        self,
        model: Any,
        train_df: pd.DataFrame,
        train_labels: Any | None = None,
    ) -> FailureTrace:
        """Store a fitted model and optional training data for later tracing.

        Enables the ``fit / trace`` pattern::

            tracer = FailureTrace().fit(model, X_train, y_train)
            report = tracer.trace(X_prod, predictions)

        Args:
            model: Any sklearn-compatible, XGBoost, LightGBM, or PyTorch model.
            train_df: Training DataFrame (used as reference for anomaly scoring).
            train_labels: Optional training labels.

        Returns:
            Self, for method chaining.
        """
        self._model = model
        self._train_df = train_df
        self._train_labels = train_labels
        return self

    def trace(
        self,
        model: Any,
        df: pd.DataFrame,
        predictions: Any,
        ground_truth: Any | None = None,
    ) -> TraceReport:
        """Run the full failure trace.

        Args:
            model: Any sklearn-compatible, XGBoost, LightGBM, or PyTorch model.
            df: Input DataFrame used to generate *predictions*.
            predictions: Model output (array-like).
            ground_truth: Optional true labels for supervised attribution.

        Returns:
            TraceReport with suspicious rows and column-level root-cause attribution.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")

        predictions_arr = np.asarray(predictions)
        ground_truth_arr = np.asarray(ground_truth) if ground_truth is not None else None

        model_type = _detect_model_type(model)

        # --- Feature importances ---
        fi = _get_feature_importances(model, df, model_type)
        if fi is None or len(fi) != df.shape[1]:
            fi = _surrogate_importances(model, df, predictions_arr)

        # Pad or trim fi to match column count
        fi = np.pad(fi, (0, df.shape[1] - len(fi))) if len(fi) < df.shape[1] else fi[:df.shape[1]]

        # --- Global column statistics for data-quality checks ---
        global_stats: dict[str, Any] = {}
        for col in df.columns:
            series = df[col].dropna()
            if pd.api.types.is_numeric_dtype(series):
                global_stats[str(col)] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()) if len(series) > 1 else 0.0,
                }

        # --- Per-column anomaly rates ---
        anomaly_rates: dict[str, float] = {}
        for col in df.columns:
            series = df[col]
            n = len(series)
            if n == 0:
                anomaly_rates[str(col)] = 0.0
                continue
            anomalies = series.isnull().sum()
            if pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                if len(clean) > 3:
                    median = clean.median()
                    mad = np.median(np.abs(clean - median))
                    if mad > 0:
                        z = np.abs(0.6745 * (clean - median) / mad)
                        anomalies += int((z > 3.5).sum())
            anomaly_rates[str(col)] = float(anomalies / n)

        # --- Model confidence per row ---
        confidences = _predict_confidence(model, df, model_type)

        # --- Per-row suspicion scoring ---
        row_failures: list[RowFailure] = []
        index_list = list(df.index)
        for row_pos, idx in enumerate(index_list):
            row = df.iloc[row_pos]  # always use integer position to avoid get_loc issues
            dq_score = _row_data_quality_score(row, global_stats)

            # Build row-level column anomaly scores
            col_anomalies: dict[str, float] = {}
            for i, col in enumerate(df.columns):
                val = row.iloc[i]
                score = 0.0
                scalar_val = val.item() if hasattr(val, "item") else val
                if pd.isna(scalar_val):
                    score = 1.0
                elif isinstance(scalar_val, (int, float)):
                    col_stats = global_stats.get(str(col))
                    if col_stats and col_stats["std"] > 0:
                        z = abs((scalar_val - col_stats["mean"]) / col_stats["std"])
                        score = min(z / 4.0, 1.0)
                col_anomalies[str(col)] = score

            # Weight by feature importance
            weighted_anomaly = float(
                sum(col_anomalies[str(col)] * fi[i] for i, col in enumerate(df.columns))
            )

            model_conf = None
            model_suspicion = 0.0
            if confidences is not None and row_pos < len(confidences):
                model_conf = float(confidences[row_pos])
                model_suspicion = 1.0 - model_conf

            dq_suspicion = 1.0 - dq_score / 100.0
            suspicion_score = (0.5 * dq_suspicion + 0.3 * weighted_anomaly + 0.2 * model_suspicion) * 100

            if ground_truth_arr is not None and row_pos < len(ground_truth_arr) and row_pos < len(predictions_arr):
                    error = abs(
                        float(predictions_arr[row_pos]) - float(ground_truth_arr[row_pos])
                    )
                    suspicion_score = min(suspicion_score + error * 10, 100.0)

            top_cols = sorted(
                col_anomalies.keys(),
                key=lambda c: col_anomalies[c],
                reverse=True,
            )[: 3]

            reason_parts = []
            if dq_suspicion > 0.3:
                reason_parts.append("data quality issues")
            if model_suspicion > 0.3:
                reason_parts.append("low model confidence")
            if weighted_anomaly > 0.3:
                reason_parts.append("feature anomalies")
            reason = "; ".join(reason_parts) if reason_parts else "no strong signal"

            if suspicion_score > 20:
                row_failures.append(
                    RowFailure(
                        row_index=idx,
                        suspicion_score=suspicion_score,
                        top_columns=top_cols,
                        data_quality_score=dq_score,
                        model_confidence=model_conf,
                        reason=reason,
                    )
                )

        row_failures.sort(key=lambda r: r.suspicion_score, reverse=True)

        # --- Column attributions ---
        column_attributions: list[ColumnAttribution] = []
        for i, col in enumerate(df.columns):
            model_contrib = float(fi[i]) * 100
            dq_contrib = anomaly_rates.get(str(col), 0.0) * 100
            importance = 0.6 * model_contrib + 0.4 * dq_contrib

            column_attributions.append(
                ColumnAttribution(
                    column=str(col),
                    importance_score=importance,
                    data_quality_contribution=dq_contrib,
                    model_contribution=model_contrib,
                    anomaly_rate=anomaly_rates.get(str(col), 0.0),
                )
            )
        column_attributions.sort(key=lambda ca: ca.importance_score, reverse=True)
        top_cols_list = [ca.column for ca in column_attributions[: self.top_k]]

        total_rows = len(df)
        dq_failure_pct = (
            sum(1 for r in row_failures if r.data_quality_score < 70) / max(total_rows, 1) * 100
        )
        model_failure_pct = (
            sum(1 for r in row_failures if r.model_confidence is not None and r.model_confidence < 0.5)
            / max(total_rows, 1) * 100
        )
        mend_score = min(len(row_failures) / max(total_rows, 1) * 100 * 2, 100.0)

        report = TraceReport(
            total_rows=total_rows,
            suspicious_rows=row_failures,
            column_attributions=column_attributions[: self.top_k],
            mend_score=mend_score,
            top_failure_columns=top_cols_list,
            data_quality_failure_pct=dq_failure_pct,
            model_failure_pct=model_failure_pct,
        )

        if self.verbose:
            try:
                from rich.console import Console
                from rich.panel import Panel
                console = Console()
                console.print(Panel(report.summary(), style="bold red"))
            except ImportError:
                print(report.summary())

        return report
