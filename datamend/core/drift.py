"""DriftRadar engine — statistical drift detection with MendScore."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ColumnDriftResult:
    """Drift analysis result for a single column."""

    column: str
    dtype: str
    psi: Optional[float]          # Population Stability Index (numeric)
    ks_stat: Optional[float]      # Kolmogorov-Smirnov statistic (numeric)
    ks_pvalue: Optional[float]    # KS p-value
    chi2_stat: Optional[float]    # Chi-square statistic (categorical)
    chi2_pvalue: Optional[float]  # Chi-square p-value
    jsd: Optional[float]          # Jensen-Shannon divergence (0–1)
    drift_score: float            # Composite 0–100 (higher = more drift)
    drifted: bool                 # True if statistically significant drift detected
    severity: str                 # "none" | "low" | "medium" | "high" | "critical"


@dataclass
class DriftReport:
    """Full DriftRadar result for a pair of DataFrames."""

    mend_score: float             # 0=no drift (stable), 100=severe drift
    overall_drifted: bool
    columns_drifted: List[str]
    column_results: Dict[str, ColumnDriftResult]
    train_shape: Tuple[int, int]
    prod_shape: Tuple[int, int]
    alpha: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict."""
        return {
            "timestamp": self.timestamp,
            "mend_score": round(self.mend_score, 2),
            "overall_drifted": self.overall_drifted,
            "columns_drifted": self.columns_drifted,
            "train_shape": self.train_shape,
            "prod_shape": self.prod_shape,
            "column_results": {
                col: {
                    "dtype": r.dtype,
                    "psi": round(r.psi, 4) if r.psi is not None else None,
                    "ks_stat": round(r.ks_stat, 4) if r.ks_stat is not None else None,
                    "ks_pvalue": round(r.ks_pvalue, 4) if r.ks_pvalue is not None else None,
                    "chi2_stat": round(r.chi2_stat, 4) if r.chi2_stat is not None else None,
                    "chi2_pvalue": round(r.chi2_pvalue, 4) if r.chi2_pvalue is not None else None,
                    "jsd": round(r.jsd, 4) if r.jsd is not None else None,
                    "drift_score": round(r.drift_score, 2),
                    "drifted": r.drifted,
                    "severity": r.severity,
                }
                for col, r in self.column_results.items()
            },
        }

    def summary(self) -> str:
        """Return a human-readable drift summary."""
        status = "DRIFT DETECTED" if self.overall_drifted else "STABLE"
        lines = [
            "=" * 65,
            f"datamend DriftRadar Report — {status}",
            "=" * 65,
            f"  MendScore (drift)   : {self.mend_score:.1f}/100  (0=stable, 100=critical)",
            f"  Columns drifted     : {len(self.columns_drifted)}/{len(self.column_results)}",
            f"  Train shape         : {self.train_shape}",
            f"  Production shape    : {self.prod_shape}",
            f"  Significance (α)    : {self.alpha}",
            "",
            "  Column-level drift:",
        ]
        for col, result in self.column_results.items():
            flag = "DRIFT" if result.drifted else "ok"
            metrics: List[str] = [f"severity={result.severity}", f"score={result.drift_score:.1f}"]
            if result.psi is not None:
                metrics.append(f"PSI={result.psi:.4f}")
            if result.ks_stat is not None:
                metrics.append(f"KS={result.ks_stat:.4f}")
            if result.jsd is not None:
                metrics.append(f"JSD={result.jsd:.4f}")
            lines.append(f"    [{flag}] {col}: {', '.join(metrics)}")
        lines.append("=" * 65)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------


def _compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Compute Population Stability Index (PSI).

    PSI < 0.1  → no significant drift
    PSI < 0.2  → moderate drift
    PSI ≥ 0.2  → significant drift
    """
    eps = 1e-6
    breakpoints = np.linspace(0, 100, buckets + 1)
    bucket_edges = np.percentile(expected, breakpoints)
    bucket_edges = np.unique(bucket_edges)
    if len(bucket_edges) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bucket_edges)
    actual_counts, _ = np.histogram(actual, bins=bucket_edges)

    expected_pct = (expected_counts + eps) / (len(expected) + eps * len(expected_counts))
    actual_pct = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return max(0.0, psi)


def _compute_jsd(p: np.ndarray, q: np.ndarray, n_bins: int = 50) -> float:
    """Compute Jensen-Shannon Divergence (JSD) in [0, 1] for continuous data.

    Both arrays are first histogrammed onto the same bin edges.
    """
    eps = 1e-10
    combined = np.concatenate([p, q])
    mn, mx = combined.min(), combined.max()
    if mn == mx:
        return 0.0
    bins = np.linspace(mn, mx, n_bins + 1)
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_norm = p_hist / p_hist.sum()
    q_norm = q_hist / q_hist.sum()
    m = 0.5 * (p_norm + q_norm)
    jsd = 0.5 * np.sum(p_norm * np.log(p_norm / m)) + 0.5 * np.sum(q_norm * np.log(q_norm / m))
    return float(np.clip(jsd, 0.0, 1.0))


def _compute_jsd_categorical(
    train_series: pd.Series, prod_series: pd.Series
) -> float:
    """Compute JSD for categorical distributions."""
    eps = 1e-10
    all_cats = set(train_series.dropna().unique()) | set(prod_series.dropna().unique())
    if not all_cats:
        return 0.0
    cats = sorted(all_cats, key=str)
    train_counts = train_series.value_counts().reindex(cats, fill_value=0).astype(float) + eps
    prod_counts = prod_series.value_counts().reindex(cats, fill_value=0).astype(float) + eps
    p = train_counts.values / train_counts.sum()
    q = prod_counts.values / prod_counts.sum()
    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(np.clip(jsd, 0.0, 1.0))


def _psi_to_severity(psi: float) -> str:
    """Convert PSI score to a severity label."""
    if psi < 0.1:
        return "none"
    if psi < 0.2:
        return "low"
    if psi < 0.25:
        return "medium"
    if psi < 0.5:
        return "high"
    return "critical"


def _drift_score_from_metrics(
    psi: Optional[float],
    ks_stat: Optional[float],
    jsd: Optional[float],
    chi2_normed: Optional[float],
) -> float:
    """Compute a composite drift score (0–100) from multiple metrics."""
    components: List[float] = []
    if psi is not None:
        components.append(min(psi / 0.5, 1.0) * 100)
    if ks_stat is not None:
        components.append(ks_stat * 100)
    if jsd is not None:
        components.append(jsd * 100)
    if chi2_normed is not None:
        components.append(min(chi2_normed, 1.0) * 100)
    if not components:
        return 0.0
    return float(np.mean(components))


# ---------------------------------------------------------------------------
# DriftRadar engine
# ---------------------------------------------------------------------------


class DriftRadar:
    """Detect statistical drift between training and production DataFrames.

    Uses PSI, Kolmogorov-Smirnov test, chi-square test, and Jensen-Shannon
    Divergence. Emits a composite MendScore that summarises overall drift
    urgency on a 0–100 scale.

    Args:
        psi_buckets: Number of buckets for PSI computation.
        alpha: Significance level for KS and chi-square tests.
        verbose: Print drift summary to stdout.
    """

    def __init__(
        self,
        psi_buckets: int = 10,
        alpha: float = 0.05,
        verbose: bool = True,
    ) -> None:
        self.psi_buckets = psi_buckets
        self.alpha = alpha
        self.verbose = verbose
        self._reference_df: Optional[pd.DataFrame] = None

    def fit(self, reference_df: pd.DataFrame) -> "DriftRadar":
        """Store a reference DataFrame for later drift comparison.

        Enables the ``fit / detect`` pattern::

            radar = DriftRadar().fit(train_df)
            report = radar.detect(prod_df)

        Args:
            reference_df: Reference (training) DataFrame.

        Returns:
            Self, for method chaining.
        """
        if not isinstance(reference_df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(reference_df).__name__}")
        self._reference_df = reference_df
        return self

    def detect(
        self,
        train_df: pd.DataFrame,
        prod_df: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
    ) -> DriftReport:
        """Detect drift between *train_df* and *prod_df*.

        When called after :meth:`fit`, *prod_df* can be the only positional
        argument and *train_df* acts as the production frame::

            radar = DriftRadar().fit(reference_df)
            report = radar.detect(prod_df)

        Or both DataFrames can be passed directly::

            report = DriftRadar().detect(train_df, prod_df)

        Args:
            train_df: Reference (training) DataFrame — or production DataFrame
                      when called after :meth:`fit`.
            prod_df: Current (production) DataFrame. If omitted, the DataFrame
                     stored by :meth:`fit` is used as the reference and *train_df*
                     is treated as the production frame.
            columns: Subset of columns to analyse. Defaults to all shared columns.

        Returns:
            DriftReport with per-column results and overall MendScore.
        """
        # Support fit/detect pattern: detect(prod_df) after fit(reference_df)
        if prod_df is None:
            if self._reference_df is None:
                raise ValueError(
                    "prod_df is required when no reference was stored via .fit()."
                )
            prod_df = train_df
            train_df = self._reference_df

        if not isinstance(train_df, pd.DataFrame) or not isinstance(prod_df, pd.DataFrame):
            raise TypeError("Both arguments must be pandas DataFrames.")

        shared_cols = list(set(train_df.columns) & set(prod_df.columns))
        if columns is not None:
            shared_cols = [c for c in columns if c in shared_cols]
        if not shared_cols:
            raise ValueError("No shared columns found between train_df and prod_df.")

        column_results: Dict[str, ColumnDriftResult] = {}
        for col in shared_cols:
            result = self._analyse_column(
                train_df[col].dropna(), prod_df[col].dropna(), col
            )
            column_results[col] = result

        drifted_cols = [col for col, r in column_results.items() if r.drifted]
        drift_scores = [r.drift_score for r in column_results.values()]
        mend_score = float(np.mean(drift_scores)) if drift_scores else 0.0
        overall_drifted = len(drifted_cols) > 0

        report = DriftReport(
            mend_score=mend_score,
            overall_drifted=overall_drifted,
            columns_drifted=drifted_cols,
            column_results=column_results,
            train_shape=train_df.shape,
            prod_shape=prod_df.shape,
            alpha=self.alpha,
        )

        if self.verbose:
            try:
                from rich.console import Console
                from rich.panel import Panel
                console = Console()
                console.print(Panel(report.summary(), style="bold cyan"))
            except ImportError:
                print(report.summary())

        return report

    def _analyse_column(
        self,
        train_series: pd.Series,
        prod_series: pd.Series,
        col: str,
    ) -> ColumnDriftResult:
        """Run drift analysis for a single column."""
        dtype = str(train_series.dtype)
        is_numeric = pd.api.types.is_numeric_dtype(train_series)

        psi = ks_stat = ks_pvalue = chi2_stat = chi2_pvalue = jsd = None
        chi2_normed = None

        if len(train_series) < 5 or len(prod_series) < 5:
            return ColumnDriftResult(
                column=col, dtype=dtype, psi=None, ks_stat=None, ks_pvalue=None,
                chi2_stat=None, chi2_pvalue=None, jsd=None,
                drift_score=0.0, drifted=False, severity="none",
            )

        if is_numeric:
            train_arr = train_series.to_numpy(dtype=float)
            prod_arr = prod_series.to_numpy(dtype=float)

            psi = _compute_psi(train_arr, prod_arr, self.psi_buckets)
            ks_stat, ks_pvalue = stats.ks_2samp(train_arr, prod_arr)
            jsd = _compute_jsd(train_arr, prod_arr)

            drifted = (psi >= 0.2) or (ks_pvalue is not None and ks_pvalue < self.alpha and ks_stat > 0.1)
            severity = _psi_to_severity(psi)

        else:
            jsd = _compute_jsd_categorical(train_series, prod_series)
            all_cats = sorted(
                set(train_series.unique()) | set(prod_series.unique()), key=str
            )
            train_counts = (
                train_series.value_counts().reindex(all_cats, fill_value=0).values
            )
            prod_counts = (
                prod_series.value_counts().reindex(all_cats, fill_value=0).values
            )
            if train_counts.sum() > 0 and prod_counts.sum() > 0 and len(all_cats) > 1:
                try:
                    chi2_stat, chi2_pvalue = stats.chisquare(
                        prod_counts + 1,
                        f_exp=(train_counts + 1)
                        * prod_counts.sum()
                        / train_counts.sum(),
                    )
                    chi2_stat = float(chi2_stat)
                    chi2_pvalue = float(chi2_pvalue)
                    n = prod_counts.sum()
                    k = len(all_cats)
                    chi2_normed = float(chi2_stat) / max(n * (k - 1), 1)
                except Exception:
                    chi2_stat = chi2_pvalue = None

            drifted = (
                (jsd > 0.2)
                or (chi2_pvalue is not None and chi2_pvalue < self.alpha and jsd > 0.1)
            )
            severity = "none" if jsd < 0.1 else "low" if jsd < 0.2 else "medium" if jsd < 0.35 else "high" if jsd < 0.5 else "critical"

        drift_score = _drift_score_from_metrics(psi, ks_stat, jsd, chi2_normed)

        return ColumnDriftResult(
            column=col,
            dtype=dtype,
            psi=psi,
            ks_stat=ks_stat,
            ks_pvalue=ks_pvalue,
            chi2_stat=chi2_stat,
            chi2_pvalue=chi2_pvalue,
            jsd=jsd,
            drift_score=drift_score,
            drifted=drifted,
            severity=severity,
        )
