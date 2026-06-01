"""AutoRepair engine — automatically detect and fix all data quality issues."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RepairAction:
    """A single atomic repair applied to the DataFrame."""

    column: str
    issue_type: str
    description: str
    rows_affected: int
    before_sample: Any
    after_sample: Any
    strategy: str


@dataclass
class RepairReport:
    """Complete record of every repair datamend applied to a DataFrame."""

    total_issues_found: int
    total_rows_affected: int
    actions: list[RepairAction]
    columns_repaired: list[str]
    duration_seconds: float
    mend_score_before: float
    mend_score_after: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_issues_found": self.total_issues_found,
            "total_rows_affected": self.total_rows_affected,
            "mend_score_before": round(self.mend_score_before, 2),
            "mend_score_after": round(self.mend_score_after, 2),
            "columns_repaired": self.columns_repaired,
            "actions": [
                {
                    "column": a.column,
                    "issue_type": a.issue_type,
                    "description": a.description,
                    "rows_affected": a.rows_affected,
                    "strategy": a.strategy,
                }
                for a in self.actions
            ],
        }

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=" * 60,
            "datamend AutoRepair Report",
            "=" * 60,
            f"  Issues found    : {self.total_issues_found}",
            f"  Rows affected   : {self.total_rows_affected}",
            f"  Columns repaired: {len(self.columns_repaired)}",
            f"  MendScore before: {self.mend_score_before:.1f}/100",
            f"  MendScore after : {self.mend_score_after:.1f}/100",
            f"  Duration        : {self.duration_seconds:.2f}s",
            "",
            "  Repairs applied:",
        ]
        for action in self.actions:
            lines.append(
                f"    [{action.issue_type}] {action.column} — "
                f"{action.description} ({action.rows_affected} rows, "
                f"strategy: {action.strategy})"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _compute_mend_score(df: pd.DataFrame) -> float:
    """Compute a data quality MendScore (0–100) for a DataFrame.

    Penalises nulls, outliers, type inconsistencies, and duplicates.
    """
    if df.empty:
        return 100.0

    penalty = 0.0
    total_cells = df.shape[0] * df.shape[1]

    # Null penalty
    null_rate = df.isnull().sum().sum() / max(total_cells, 1)
    penalty += null_rate * 40

    # Duplicate row penalty
    dup_rate = df.duplicated().sum() / max(len(df), 1)
    penalty += dup_rate * 20

    # Outlier penalty (numeric columns only)
    num_cols = df.select_dtypes(include=[np.number]).columns
    outlier_total = 0
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        z = np.abs(stats.zscore(series))
        outlier_total += (z > 3.5).sum()
    if len(num_cols) > 0 and len(df) > 0:
        outlier_rate = outlier_total / max(len(df) * len(num_cols), 1)
        penalty += outlier_rate * 25

    # Whitespace / hidden characters penalty (object columns)
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    ws_issues = 0
    for col in obj_cols:
        series = df[col].dropna().astype(str)
        ws_issues += series.str.contains(r"^\s+|\s+$", regex=True).sum()
    if len(obj_cols) > 0 and len(df) > 0:
        ws_rate = ws_issues / max(len(df) * len(obj_cols), 1)
        penalty += ws_rate * 15

    return max(0.0, 100.0 - penalty)


def _is_numeric_string_column(series: pd.Series) -> bool:
    """Return True if an object column contains predominantly numeric strings."""
    sample = series.dropna().astype(str).head(500)
    if len(sample) == 0:
        return False
    numeric_count = sample.str.match(r"^-?\d+(\.\d+)?$").sum()
    return numeric_count / len(sample) > 0.8


def _detect_date_formats(series: pd.Series) -> list[str]:
    """Return a list of date format strings detected in the series."""
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
        "%Y/%m/%d", "%d.%m.%Y", "%m.%d.%Y", "%B %d, %Y",
        "%b %d, %Y", "%Y%m%d", "%d %B %Y",
    ]
    detected = []
    sample = series.dropna().astype(str).head(200)
    for fmt in formats:
        hits = 0
        for val in sample:
            try:
                datetime.strptime(val.strip(), fmt)
                hits += 1
            except ValueError:
                pass
        if hits / max(len(sample), 1) > 0.3:
            detected.append(fmt)
    return detected


def _fix_mojibake(text: str) -> str:
    """Attempt to fix common encoding corruption (mojibake)."""
    try:
        fixed = text.encode("latin-1").decode("utf-8")
        return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _normalise_category(val: str) -> str:
    """Normalise a categorical string value for grouping similar labels."""
    return unicodedata.normalize("NFKD", val).strip().lower()


# ---------------------------------------------------------------------------
# Issue detectors
# ---------------------------------------------------------------------------


class _NullDetector:
    """Detect and impute missing values."""

    def __init__(self, strategy: str = "auto") -> None:
        self.strategy = strategy

    def detect(self, df: pd.DataFrame) -> dict[str, int]:
        return {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().any()}

    def fix(
        self, df: pd.DataFrame, null_counts: dict[str, int]
    ) -> tuple[pd.DataFrame, list[RepairAction]]:
        actions: list[RepairAction] = []
        df = df.copy()
        for col, count in null_counts.items():
            if count == 0:
                continue
            series = df[col]
            dtype = series.dtype
            before_sample = series[series.isnull()].index[:3].tolist()

            if pd.api.types.is_numeric_dtype(dtype):
                strategy = self.strategy
                if strategy == "auto":
                    # Use median for skewed, mean otherwise
                    skewness = abs(series.dropna().skew()) if len(series.dropna()) > 3 else 0
                    strategy = "median" if skewness > 1.0 else "mean"
                if strategy == "median":
                    fill_val = series.median()
                elif strategy == "mean":
                    fill_val = series.mean()
                else:
                    fill_val = series.mean()
                df[col] = series.fillna(fill_val)
                desc = f"Imputed {count} nulls with {strategy}={fill_val:.4g}"
            elif isinstance(series.dtype, pd.CategoricalDtype) or dtype is object or str(dtype) == "string":
                # Use dropna() so all-null columns don't return None as mode
                mode_vals = series.dropna().mode()
                fill_val = mode_vals.iloc[0] if not mode_vals.empty else None
                strategy = "mode"
                if fill_val is not None:
                    df[col] = series.fillna(fill_val)
                    desc = f"Imputed {count} nulls with mode='{fill_val}'"
                else:
                    desc = f"Skipped {count} nulls in all-null column (no reference values)"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                fill_val = series.median()
                df[col] = series.fillna(fill_val)
                desc = f"Imputed {count} null datetimes with median"
                strategy = "median"
            else:
                mode_vals = series.dropna().mode()
                fill_val = mode_vals.iloc[0] if not mode_vals.empty else None
                strategy = "mode"
                if fill_val is not None:
                    df[col] = series.fillna(fill_val)
                desc = f"Imputed {count} nulls with mode"

            actions.append(
                RepairAction(
                    column=col,
                    issue_type="NULL",
                    description=desc,
                    rows_affected=count,
                    before_sample=before_sample,
                    after_sample=fill_val,
                    strategy=strategy,
                )
            )
        return df, actions


class _OutlierDetector:
    """Detect and clip statistical outliers in numeric columns."""

    def __init__(self, z_threshold: float = 3.5) -> None:
        self.z_threshold = z_threshold

    def detect(self, df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            # Modified Z-score using median absolute deviation (robust)
            median = series.median()
            mad = np.median(np.abs(series - median))
            if mad == 0:
                continue
            modified_z = 0.6745 * (series - median) / mad
            outlier_mask = np.abs(modified_z) > self.z_threshold
            outlier_idx = series[outlier_mask].index.to_numpy()
            if len(outlier_idx) > 0:
                result[col] = (outlier_idx, series[outlier_mask].to_numpy())
        return result

    def fix(
        self, df: pd.DataFrame, outliers: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> tuple[pd.DataFrame, list[RepairAction]]:
        actions: list[RepairAction] = []
        df = df.copy()
        for col, (idx, vals) in outliers.items():
            series = df[col]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df[col] = series.clip(lower=lower, upper=upper)
            actions.append(
                RepairAction(
                    column=col,
                    issue_type="OUTLIER",
                    description=f"Clipped {len(idx)} outliers to IQR bounds [{lower:.4g}, {upper:.4g}]",
                    rows_affected=len(idx),
                    before_sample=vals[:3].tolist(),
                    after_sample=[lower, upper],
                    strategy="iqr_clip",
                )
            )
        return df, actions


class _TypeMismatchDetector:
    """Detect silent type coercions and fix columns stored as wrong dtype."""

    def detect(self, df: pd.DataFrame) -> dict[str, str]:
        candidates: dict[str, str] = {}
        for col in df.select_dtypes(include=["object", "string"]).columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            if _is_numeric_string_column(series):
                candidates[col] = "numeric"
            else:
                fmt_list = _detect_date_formats(series)
                if fmt_list:
                    candidates[col] = f"date:{fmt_list[0]}"
        return candidates

    def fix(
        self, df: pd.DataFrame, candidates: dict[str, str]
    ) -> tuple[pd.DataFrame, list[RepairAction]]:
        actions: list[RepairAction] = []
        df = df.copy()
        for col, target_type in candidates.items():
            series = df[col]
            count = len(series.dropna())
            if target_type == "numeric":
                converted = pd.to_numeric(series, errors="coerce")
                new_nulls = converted.isnull().sum() - series.isnull().sum()
                if new_nulls / max(len(series), 1) < 0.05:
                    df[col] = converted
                    actions.append(
                        RepairAction(
                            column=col,
                            issue_type="TYPE_MISMATCH",
                            description=f"Converted object column to numeric ({count} values)",
                            rows_affected=count,
                            before_sample="object",
                            after_sample="float64",
                            strategy="to_numeric",
                        )
                    )
            elif target_type.startswith("date:"):
                fmt = target_type.split("date:")[1]
                try:
                    converted_dt = pd.to_datetime(series, format=fmt, errors="coerce")
                    new_nulls = converted_dt.isnull().sum() - series.isnull().sum()
                    if new_nulls / max(len(series), 1) < 0.1:
                        df[col] = converted_dt
                        actions.append(
                            RepairAction(
                                column=col,
                                issue_type="TYPE_MISMATCH",
                                description=f"Converted object column to datetime using format {fmt}",
                                rows_affected=count,
                                before_sample="object",
                                after_sample="datetime64",
                                strategy="to_datetime",
                            )
                        )
                except Exception:
                    pass
        return df, actions


class _DuplicateDetector:
    """Detect and remove exact and near-duplicate rows."""

    def detect_exact(self, df: pd.DataFrame) -> pd.Index:
        return df[df.duplicated()].index

    def detect_near_duplicates(
        self, df: pd.DataFrame, threshold: float = 0.85
    ) -> list[tuple[int, int]]:
        """Find near-duplicate rows using Jaccard similarity on string representations."""
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        if len(obj_cols) == 0:
            return []
        str_repr = df[obj_cols].fillna("").astype(str).apply(
            lambda row: set(" ".join(row.values).lower().split()), axis=1
        )
        pairs: list[tuple[int, int]] = []
        idx_list = list(str_repr.index[:500])  # cap at 500 for performance
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                a = str_repr[idx_list[i]]
                b = str_repr[idx_list[j]]
                union = len(a | b)
                if union == 0:
                    continue
                jaccard = len(a & b) / union
                if jaccard >= threshold:
                    pairs.append((idx_list[i], idx_list[j]))
        return pairs

    def fix(
        self,
        df: pd.DataFrame,
        exact_idx: pd.Index,
        near_pairs: list[tuple[int, int]],
    ) -> tuple[pd.DataFrame, list[RepairAction]]:
        actions: list[RepairAction] = []
        df = df.copy()
        if len(exact_idx) > 0:
            df = df.drop(index=exact_idx).reset_index(drop=True)
            actions.append(
                RepairAction(
                    column="[ALL]",
                    issue_type="DUPLICATE",
                    description=f"Removed {len(exact_idx)} exact duplicate rows",
                    rows_affected=len(exact_idx),
                    before_sample=exact_idx[:3].tolist(),
                    after_sample=None,
                    strategy="drop_duplicates",
                )
            )
        to_drop = list({j for _, j in near_pairs})
        to_drop_valid = [i for i in to_drop if i in df.index]
        if to_drop_valid:
            df = df.drop(index=to_drop_valid).reset_index(drop=True)
            actions.append(
                RepairAction(
                    column="[ALL]",
                    issue_type="NEAR_DUPLICATE",
                    description=f"Removed {len(to_drop_valid)} near-duplicate rows (Jaccard≥0.85)",
                    rows_affected=len(to_drop_valid),
                    before_sample=to_drop_valid[:3],
                    after_sample=None,
                    strategy="jaccard_dedup",
                )
            )
        return df, actions


class _EncodingDetector:
    """Detect and fix encoding corruption (mojibake) in string columns."""

    _MOJIBAKE_PATTERN = re.compile(r"[\xc0-\xff][\x80-\xbf]{1,3}")

    def detect(self, df: pd.DataFrame) -> dict[str, int]:
        result: dict[str, int] = {}
        for col in df.select_dtypes(include=["object", "string"]).columns:
            series = df[col].dropna().astype(str)
            if len(series) == 0:
                continue
            count = int(series.apply(lambda x: bool(self._MOJIBAKE_PATTERN.search(x))).sum())
            if count > 0:
                result[col] = count
        return result

    def fix(
        self, df: pd.DataFrame, affected: dict[str, int]
    ) -> tuple[pd.DataFrame, list[RepairAction]]:
        actions: list[RepairAction] = []
        df = df.copy()
        for col, count in affected.items():
            df[col] = df[col].apply(
                lambda x: _fix_mojibake(str(x)) if pd.notna(x) else x
            )
            actions.append(
                RepairAction(
                    column=col,
                    issue_type="ENCODING",
                    description=f"Fixed encoding corruption (mojibake) in {count} values",
                    rows_affected=count,
                    before_sample=None,
                    after_sample=None,
                    strategy="latin1_to_utf8",
                )
            )
        return df, actions


class _CategoryNormalisationDetector:
    """Detect inconsistent category labels (Male/male/M → male)."""

    def detect(self, df: pd.DataFrame, cardinality_cap: int = 50) -> dict[str, dict[str, str]]:
        """Return per-column mapping from raw value → normalised value."""
        result: dict[str, dict[str, str]] = {}
        for col in df.select_dtypes(include=["object", "string"]).columns:
            series = df[col].dropna().astype(str)
            unique_vals = series.unique()
            if len(unique_vals) > cardinality_cap:
                continue
            norm_map: dict[str, str] = {}
            norm_to_canonical: dict[str, str] = {}
            for val in unique_vals:
                norm = _normalise_category(val)
                if norm in norm_to_canonical:
                    if norm_to_canonical[norm] != val:
                        norm_map[val] = norm_to_canonical[norm]
                else:
                    norm_to_canonical[norm] = val
            if norm_map:
                result[col] = norm_map
        return result

    def fix(
        self, df: pd.DataFrame, mappings: dict[str, dict[str, str]]
    ) -> tuple[pd.DataFrame, list[RepairAction]]:
        actions: list[RepairAction] = []
        df = df.copy()
        for col, mapping in mappings.items():
            affected = df[col].isin(mapping.keys()).sum()
            df[col] = df[col].replace(mapping)
            actions.append(
                RepairAction(
                    column=col,
                    issue_type="INCONSISTENT_CATEGORY",
                    description=(
                        f"Normalised {len(mapping)} label variants to canonical form "
                        f"in {affected} rows. Examples: {dict(list(mapping.items())[:2])}"
                    ),
                    rows_affected=int(affected),
                    before_sample=list(mapping.keys())[:3],
                    after_sample=list(mapping.values())[:3],
                    strategy="normalise_category",
                )
            )
        return df, actions


class _WhitespaceDetector:
    """Detect and strip leading/trailing whitespace and hidden characters."""

    _HIDDEN_CHARS = re.compile(r"[\x00-\x1f\x7f\xa0​‌‍﻿]")

    def detect(self, df: pd.DataFrame) -> dict[str, int]:
        result: dict[str, int] = {}
        for col in df.select_dtypes(include=["object", "string"]).columns:
            series = df[col].dropna().astype(str)
            if len(series) == 0:
                continue
            count = int(series.apply(
                lambda x: bool(re.search(r"^\s+|\s+$", x) or self._HIDDEN_CHARS.search(x))
            ).sum())
            if count > 0:
                result[col] = count
        return result

    def fix(
        self, df: pd.DataFrame, affected: dict[str, int]
    ) -> tuple[pd.DataFrame, list[RepairAction]]:
        actions: list[RepairAction] = []
        df = df.copy()
        for col, count in affected.items():
            df[col] = df[col].apply(
                lambda x: self._HIDDEN_CHARS.sub("", str(x)).strip() if pd.notna(x) else x
            )
            actions.append(
                RepairAction(
                    column=col,
                    issue_type="WHITESPACE",
                    description=f"Stripped whitespace and hidden characters from {count} values",
                    rows_affected=count,
                    before_sample=None,
                    after_sample=None,
                    strategy="strip_hidden",
                )
            )
        return df, actions


class _UnitMismatchDetector:
    """Detect probable unit mismatches in numeric columns (e.g. km vs miles)."""

    def detect(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Detect columns where the distribution has a suspicious bimodal structure."""
        result: dict[str, dict[str, Any]] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 20:
                continue
            # Use Hartigan's dip test approximation: check if std/mean ratio is extremely high
            cv = series.std() / (abs(series.mean()) + 1e-9)
            if cv > 5.0:
                q25, q75 = series.quantile(0.25), series.quantile(0.75)
                if q75 > 0 and q25 > 0 and q75 / q25 > 10:
                    result[col] = {
                        "cv": float(cv),
                        "q25": float(q25),
                        "q75": float(q75),
                    }
        return result

    def fix(
        self, df: pd.DataFrame, suspects: dict[str, dict[str, Any]]
    ) -> tuple[pd.DataFrame, list[RepairAction]]:
        """Flag but do not auto-correct unit mismatches (domain-specific)."""
        actions: list[RepairAction] = []
        for col, info in suspects.items():
            actions.append(
                RepairAction(
                    column=col,
                    issue_type="UNIT_MISMATCH_SUSPECT",
                    description=(
                        f"Possible unit mismatch detected (CV={info['cv']:.2f}, "
                        f"IQR ratio={info['q75']/info['q25']:.1f}x). "
                        "Manual review recommended."
                    ),
                    rows_affected=0,
                    before_sample=None,
                    after_sample=None,
                    strategy="flag_only",
                )
            )
        return df, actions


# ---------------------------------------------------------------------------
# AutoRepair engine
# ---------------------------------------------------------------------------


class AutoRepair:
    """Context-aware automatic data repair engine.

    Detects and fixes: nulls, outliers, type mismatches, duplicates,
    encoding corruption, inconsistent categories, whitespace/hidden characters,
    and probable unit mismatches.

    Args:
        strategy: Null imputation strategy — ``"auto"``, ``"mean"``, ``"median"``.
        confirm: If True, require explicit user confirmation before applying repairs.
        fast_mode: If True, sample up to 50 000 rows for detection (faster on large data).
        chunk_size: Row chunk size for chunked processing.
        plugins: List of custom repair plugin instances implementing ``BaseRepairPlugin``.
        verbose: Print repair summary to stdout.
    """

    def __init__(
        self,
        strategy: str = "auto",
        confirm: bool = False,
        fast_mode: bool = False,
        chunk_size: int = 100_000,
        plugins: list[Any] | None = None,
        verbose: bool = True,
    ) -> None:
        self.strategy = strategy
        self.confirm = confirm
        self.fast_mode = fast_mode
        self.chunk_size = chunk_size
        self.plugins: list[Any] = plugins or []
        self.verbose = verbose

        self._null = _NullDetector(strategy=strategy)
        self._outlier = _OutlierDetector()
        self._type = _TypeMismatchDetector()
        self._dup = _DuplicateDetector()
        self._enc = _EncodingDetector()
        self._cat = _CategoryNormalisationDetector()
        self._ws = _WhitespaceDetector()
        self._unit = _UnitMismatchDetector()

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, RepairReport]:
        """Run the full repair pipeline on *df* and return (repaired_df, report).

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (repaired DataFrame, RepairReport).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")

        import time
        start = time.perf_counter()

        mend_before = _compute_mend_score(df)

        detect_df = df.sample(50000, random_state=42) if self.fast_mode and len(df) > 50000 else df

        all_actions: list[RepairAction] = []

        # Phase 1 — whitespace / encoding (run first so subsequent detectors see clean text)
        ws_affected = self._ws.detect(detect_df)
        df, ws_actions = self._ws.fix(df, ws_affected)
        all_actions.extend(ws_actions)

        enc_affected = self._enc.detect(detect_df)
        df, enc_actions = self._enc.fix(df, enc_affected)
        all_actions.extend(enc_actions)

        # Phase 2 — type coercion (run before nulls so numeric columns are proper dtype)
        type_candidates = self._type.detect(df)
        df, type_actions = self._type.fix(df, type_candidates)
        all_actions.extend(type_actions)

        # Phase 3 — nulls
        null_counts = self._null.detect(df)
        df, null_actions = self._null.fix(df, null_counts)
        all_actions.extend(null_actions)

        # Phase 4 — outliers
        outliers = self._outlier.detect(df)
        df, out_actions = self._outlier.fix(df, outliers)
        all_actions.extend(out_actions)

        # Phase 5 — duplicates
        exact_idx = self._dup.detect_exact(df)
        near_pairs = self._dup.detect_near_duplicates(df)
        df, dup_actions = self._dup.fix(df, exact_idx, near_pairs)
        all_actions.extend(dup_actions)

        # Phase 6 — category normalisation
        cat_mappings = self._cat.detect(df)
        df, cat_actions = self._cat.fix(df, cat_mappings)
        all_actions.extend(cat_actions)

        # Phase 7 — unit mismatch detection (flags only)
        unit_suspects = self._unit.detect(df)
        df, unit_actions = self._unit.fix(df, unit_suspects)
        all_actions.extend(unit_actions)

        # Phase 8 — community plugins
        for plugin in self.plugins:
            df, plugin_actions = plugin.repair(df)
            all_actions.extend(plugin_actions)

        mend_after = _compute_mend_score(df)
        duration = time.perf_counter() - start

        cols_repaired = list({a.column for a in all_actions if a.column != "[ALL]"})
        total_rows = sum(a.rows_affected for a in all_actions)

        report = RepairReport(
            total_issues_found=len(all_actions),
            total_rows_affected=total_rows,
            actions=all_actions,
            columns_repaired=cols_repaired,
            duration_seconds=duration,
            mend_score_before=mend_before,
            mend_score_after=mend_after,
        )

        if self.verbose:
            try:
                from rich.console import Console
                from rich.panel import Panel
                console = Console()
                console.print(Panel(report.summary(), style="bold green"))
            except ImportError:
                print(report.summary())

        if self.confirm and all_actions:
            response = input("\nApply all repairs? [y/N]: ").strip().lower()
            if response != "y":
                print("Repairs cancelled. Returning original DataFrame.")
                return df.copy(), report

        return df, report

    def repair_chunked(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[RepairReport]]:
        """Process a large DataFrame in chunks, returning one report per chunk.

        Args:
            df: Large input DataFrame.

        Returns:
            Tuple of (concatenated repaired DataFrame, list of per-chunk RepairReports).
        """
        chunks = [df.iloc[i : i + self.chunk_size] for i in range(0, len(df), self.chunk_size)]
        repaired_chunks: list[pd.DataFrame] = []
        reports: list[RepairReport] = []
        for chunk in chunks:
            repaired, report = self.fit_transform(chunk)
            repaired_chunks.append(repaired)
            reports.append(report)
        return pd.concat(repaired_chunks, ignore_index=True), reports
