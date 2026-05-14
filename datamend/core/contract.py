"""DataContract engine — define, enforce, and export data contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ContractViolationError(Exception):
    """Raised when a DataFrame fails contract validation and raise_on_failure=True."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ColumnSpec:
    """Schema and statistical specification for a single column."""

    name: str
    dtype: str
    nullable: bool
    null_rate: float
    min_val: Optional[float]
    max_val: Optional[float]
    mean_val: Optional[float]
    std_val: Optional[float]
    cardinality: int
    allowed_values: Optional[List[Any]]  # for low-cardinality categoricals
    percentiles: Optional[Dict[str, float]]  # p5, p25, p50, p75, p95
    distribution_params: Optional[Dict[str, float]]  # fitted normal params


@dataclass
class ContractViolation:
    """A single contract violation for a column."""

    column: str
    violation_type: str
    expected: Any
    observed: Any
    severity: str  # "WARNING" | "ERROR"
    message: str


@dataclass
class ContractReport:
    """Full result of validating a DataFrame against a DataContract."""

    passed: bool
    violations: List[ContractViolation]
    warnings: List[ContractViolation]
    columns_checked: int
    columns_failed: List[str]
    mend_score: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to plain dict."""
        return {
            "timestamp": self.timestamp,
            "passed": self.passed,
            "mend_score": round(self.mend_score, 2),
            "columns_checked": self.columns_checked,
            "columns_failed": self.columns_failed,
            "violations": [
                {
                    "column": v.column,
                    "violation_type": v.violation_type,
                    "expected": str(v.expected),
                    "observed": str(v.observed),
                    "severity": v.severity,
                    "message": v.message,
                }
                for v in self.violations
            ],
            "warnings": [
                {
                    "column": w.column,
                    "violation_type": w.violation_type,
                    "expected": str(w.expected),
                    "observed": str(w.observed),
                    "severity": w.severity,
                    "message": w.message,
                }
                for w in self.warnings
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Return a human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            "=" * 60,
            f"datamend DataContract Report — {status}",
            "=" * 60,
            f"  Columns checked : {self.columns_checked}",
            f"  Columns failed  : {len(self.columns_failed)}",
            f"  MendScore       : {self.mend_score:.1f}/100",
            "",
        ]
        if self.violations:
            lines.append("  ERRORS:")
            for v in self.violations:
                lines.append(f"    [{v.column}] {v.message}")
        if self.warnings:
            lines.append("  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    [{w.column}] {w.message}")
        if not self.violations and not self.warnings:
            lines.append("  All checks passed.")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DataContract
# ---------------------------------------------------------------------------


class DataContract:
    """Generate and enforce data contracts from reference DataFrames.

    Args:
        name: Human-readable contract identifier.
        null_threshold: Maximum allowed null rate per column (0–1).
        drift_threshold: Distribution shift threshold before flagging as violation.
        strict: If True, any single ERROR violation marks the whole contract as failed.
    """

    def __init__(
        self,
        name: str = "default",
        null_threshold: float = 0.05,
        drift_threshold: float = 0.1,
        strict: bool = False,
    ) -> None:
        self.name = name
        self.null_threshold = null_threshold
        self.drift_threshold = drift_threshold
        self.strict = strict
        self._specs: Dict[str, ColumnSpec] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DataContract":
        """Learn the schema and statistics of a reference DataFrame.

        Args:
            df: Clean reference DataFrame (training data).

        Returns:
            Self, for method chaining.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        self._specs = {}
        for col in df.columns:
            self._specs[col] = self._build_spec(df[col], col)
        self._fitted = True
        return self

    def _build_spec(self, series: pd.Series, col: str) -> ColumnSpec:
        """Compute a ColumnSpec from a single Series."""
        dtype = str(series.dtype)
        null_rate = float(series.isnull().mean())
        nullable = null_rate > 0
        cardinality = int(series.nunique())
        is_numeric = pd.api.types.is_numeric_dtype(series)

        min_val = max_val = mean_val = std_val = None
        percentiles: Optional[Dict[str, float]] = None
        distribution_params: Optional[Dict[str, float]] = None
        allowed_values: Optional[List[Any]] = None

        if is_numeric:
            clean = series.dropna()
            if len(clean) > 0:
                min_val = float(clean.min())
                max_val = float(clean.max())
                mean_val = float(clean.mean())
                std_val = float(clean.std()) if len(clean) > 1 else 0.0
                percentiles = {
                    "p5": float(clean.quantile(0.05)),
                    "p25": float(clean.quantile(0.25)),
                    "p50": float(clean.quantile(0.50)),
                    "p75": float(clean.quantile(0.75)),
                    "p95": float(clean.quantile(0.95)),
                }
                if len(clean) >= 8:
                    mu, sigma = float(clean.mean()), float(clean.std())
                    distribution_params = {"mu": mu, "sigma": sigma}
        else:
            if cardinality <= 50:
                allowed_values = [
                    v for v in series.dropna().unique().tolist() if v is not None
                ]

        return ColumnSpec(
            name=col,
            dtype=dtype,
            nullable=nullable,
            null_rate=null_rate,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            std_val=std_val,
            cardinality=cardinality,
            allowed_values=allowed_values,
            percentiles=percentiles,
            distribution_params=distribution_params,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
        raise_on_failure: bool = False,
    ) -> ContractReport:
        """Validate a DataFrame against this contract.

        Args:
            df: DataFrame to validate.
            raise_on_failure: Raise ContractViolationError if any ERROR-level violation found.

        Returns:
            ContractReport with full violation detail.
        """
        if not self._fitted:
            raise RuntimeError("Contract has not been fitted. Call .fit(df) first.")

        violations: List[ContractViolation] = []
        warnings: List[ContractViolation] = []
        failed_cols: List[str] = []

        # Schema: check for missing expected columns
        missing_cols = set(self._specs.keys()) - set(df.columns)
        for col in missing_cols:
            violations.append(
                ContractViolation(
                    column=col,
                    violation_type="MISSING_COLUMN",
                    expected="present",
                    observed="absent",
                    severity="ERROR",
                    message=f"Column '{col}' expected but missing from DataFrame.",
                )
            )
            failed_cols.append(col)

        # Check extra columns (warning only)
        extra_cols = set(df.columns) - set(self._specs.keys())
        for col in extra_cols:
            warnings.append(
                ContractViolation(
                    column=col,
                    violation_type="EXTRA_COLUMN",
                    expected="absent",
                    observed="present",
                    severity="WARNING",
                    message=f"Unexpected column '{col}' not in contract.",
                )
            )

        # Per-column checks
        for col, spec in self._specs.items():
            if col not in df.columns:
                continue
            series = df[col]
            col_violations, col_warnings = self._check_column(series, col, spec)
            violations.extend(col_violations)
            warnings.extend(col_warnings)
            if col_violations:
                failed_cols.append(col)

        passed = len(violations) == 0 if self.strict else len(violations) == 0

        # Compute a contract mend score
        total_checks = len(self._specs) * 4
        failed_checks = len(violations)
        mend_score = max(0.0, 100.0 * (1.0 - failed_checks / max(total_checks, 1)))

        report = ContractReport(
            passed=passed,
            violations=violations,
            warnings=warnings,
            columns_checked=len(self._specs),
            columns_failed=list(set(failed_cols)),
            mend_score=mend_score,
        )

        if raise_on_failure and not passed:
            raise ContractViolationError(report.summary())

        return report

    def _check_column(
        self,
        series: pd.Series,
        col: str,
        spec: ColumnSpec,
    ) -> Tuple[List[ContractViolation], List[ContractViolation]]:
        """Run all checks for a single column. Returns (violations, warnings)."""
        violations: List[ContractViolation] = []
        warnings: List[ContractViolation] = []

        # Null rate check
        obs_null_rate = float(series.isnull().mean())
        if not spec.nullable and obs_null_rate > 0:
            violations.append(ContractViolation(
                column=col, violation_type="NULL_RATE",
                expected="0.0", observed=f"{obs_null_rate:.4f}", severity="ERROR",
                message=f"'{col}' expected non-nullable but has {obs_null_rate:.2%} nulls.",
            ))
        elif obs_null_rate > self.null_threshold:
            violations.append(ContractViolation(
                column=col, violation_type="NULL_RATE",
                expected=f"≤{self.null_threshold:.2%}", observed=f"{obs_null_rate:.2%}",
                severity="ERROR",
                message=f"'{col}' null rate {obs_null_rate:.2%} exceeds threshold {self.null_threshold:.2%}.",
            ))

        # Dtype compatibility
        current_dtype = str(series.dtype)
        expected_dtype = spec.dtype
        if not _dtypes_compatible(current_dtype, expected_dtype):
            violations.append(ContractViolation(
                column=col, violation_type="DTYPE",
                expected=expected_dtype, observed=current_dtype, severity="ERROR",
                message=f"'{col}' dtype mismatch: expected {expected_dtype}, got {current_dtype}.",
            ))

        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_numeric and spec.min_val is not None and spec.max_val is not None:
            clean = series.dropna()

            # Range check
            obs_min = float(clean.min()) if len(clean) > 0 else None
            obs_max = float(clean.max()) if len(clean) > 0 else None
            tol = max(abs(spec.max_val - spec.min_val) * 0.5, 1.0)

            if obs_min is not None and obs_min < spec.min_val - tol:
                violations.append(ContractViolation(
                    column=col, violation_type="RANGE_VIOLATION",
                    expected=f"≥{spec.min_val:.4g}", observed=f"{obs_min:.4g}", severity="ERROR",
                    message=f"'{col}' min value {obs_min:.4g} is far below expected {spec.min_val:.4g}.",
                ))
            if obs_max is not None and obs_max > spec.max_val + tol:
                violations.append(ContractViolation(
                    column=col, violation_type="RANGE_VIOLATION",
                    expected=f"≤{spec.max_val:.4g}", observed=f"{obs_max:.4g}", severity="ERROR",
                    message=f"'{col}' max value {obs_max:.4g} is far above expected {spec.max_val:.4g}.",
                ))

            # Distribution drift (KS test)
            if spec.distribution_params and len(clean) >= 8:
                ref_mu = spec.distribution_params["mu"]
                ref_sigma = spec.distribution_params["sigma"]
                if ref_sigma > 0:
                    ref_sample = np.random.default_rng(42).normal(
                        ref_mu, ref_sigma, min(len(clean), 5000)
                    )
                    ks_stat, ks_p = stats.ks_2samp(
                        clean.values[:5000], ref_sample
                    )
                    if ks_stat > self.drift_threshold and ks_p < 0.05:
                        warnings.append(ContractViolation(
                            column=col, violation_type="DISTRIBUTION_DRIFT",
                            expected=f"KS≤{self.drift_threshold:.2f}",
                            observed=f"KS={ks_stat:.4f} (p={ks_p:.4f})",
                            severity="WARNING",
                            message=(
                                f"'{col}' shows distribution drift (KS={ks_stat:.4f}, p={ks_p:.4f}). "
                                "Check for feature drift."
                            ),
                        ))

        # Cardinality check (categorical)
        if spec.allowed_values is not None:
            obs_unique = set(series.dropna().unique().tolist())
            new_vals = obs_unique - set(spec.allowed_values)
            if new_vals:
                violations.append(ContractViolation(
                    column=col, violation_type="CARDINALITY_VIOLATION",
                    expected=f"values in {spec.allowed_values[:5]}...",
                    observed=f"new values: {list(new_vals)[:5]}",
                    severity="ERROR",
                    message=(
                        f"'{col}' contains {len(new_vals)} unseen category values: "
                        f"{list(new_vals)[:3]}."
                    ),
                ))

        return violations, warnings

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the contract to a JSON file.

        Args:
            path: File path ending in ``.json``.
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted contract.")
        data = {
            "name": self.name,
            "null_threshold": self.null_threshold,
            "drift_threshold": self.drift_threshold,
            "strict": self.strict,
            "columns": {
                col: asdict(spec) for col, spec in self._specs.items()
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "DataContract":
        """Load a previously saved contract from a JSON file.

        Args:
            path: Path to a ``.json`` file saved by :meth:`save`.

        Returns:
            Fitted DataContract instance.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        dc = cls(
            name=data["name"],
            null_threshold=data["null_threshold"],
            drift_threshold=data["drift_threshold"],
            strict=data["strict"],
        )
        specs: Dict[str, ColumnSpec] = {}
        for col, spec_dict in data["columns"].items():
            specs[col] = ColumnSpec(**spec_dict)
        dc._specs = specs
        dc._fitted = True
        return dc

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"DataContract(name='{self.name}', columns={len(self._specs)}, "
            f"null_threshold={self.null_threshold}, status={status})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dtypes_compatible(current: str, expected: str) -> bool:
    """Check dtype compatibility with tolerance for numeric sub-types."""
    numeric_types = {"int8", "int16", "int32", "int64", "float16", "float32", "float64"}
    current_base = current.rstrip("0123456789").rstrip("_")
    expected_base = expected.rstrip("0123456789").rstrip("_")

    if current == expected:
        return True
    if current_base in {"int", "float"} and expected_base in {"int", "float"}:
        return True  # numeric sub-type changes are compatible
    if "datetime" in current and "datetime" in expected:
        return True
    if current in numeric_types and expected in numeric_types:
        return True
    return False
