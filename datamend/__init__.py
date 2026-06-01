"""
datamend — The unified data repair, validation, drift detection, and failure tracing library.

Four pillars, one API:
  datamend.repair(df)                    → AutoRepair
  datamend.contract(df)                  → DataContract generation
  datamend.validate(df, contract)        → DataContract enforcement
  datamend.drift(train_df, prod_df)      → DriftRadar
  datamend.trace(model, df, predictions) → FailureTrace
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from datamend.core.contract import ContractReport, DataContract
from datamend.core.drift import DriftRadar, DriftReport
from datamend.core.repair import AutoRepair, RepairReport
from datamend.core.trace import FailureTrace, TraceReport
from datamend.pipeline import MendPipeline
from datamend.report import MendReport

__version__ = "1.1.2"
__author__ = "Vignesh"
__license__ = "MIT"

__all__ = [
    "repair",
    "contract",
    "validate",
    "drift",
    "trace",
    "AutoRepair",
    "DataContract",
    "DriftRadar",
    "FailureTrace",
    "MendPipeline",
    "MendReport",
    "RepairReport",
    "ContractReport",
    "DriftReport",
    "TraceReport",
    "__version__",
]


def repair(
    df: pd.DataFrame,
    *,
    strategy: str = "auto",
    confirm: bool = False,
    fast_mode: bool = False,
    chunk_size: int = 100_000,
    plugins: list[Any] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, RepairReport]:
    """Automatically detect and repair all data quality issues in a DataFrame.

    Args:
        df: Input DataFrame with dirty data.
        strategy: Imputation strategy — ``"auto"``, ``"mean"``, ``"median"``, ``"knn"``.
        confirm: If True, require explicit confirmation before applying repairs (production mode).
        fast_mode: If True, use sampling for large datasets to speed up detection.
        chunk_size: Number of rows per chunk for chunked processing on large datasets.
        plugins: Optional list of custom repair plugin instances.
        verbose: If True, print a repair summary to stdout.

    Returns:
        Tuple of (repaired_df, RepairReport).
    """
    engine = AutoRepair(
        strategy=strategy,
        confirm=confirm,
        fast_mode=fast_mode,
        chunk_size=chunk_size,
        plugins=plugins or [],
        verbose=verbose,
    )
    return engine.fit_transform(df)


def contract(
    df: pd.DataFrame,
    *,
    name: str = "default",
    null_threshold: float = 0.05,
    drift_threshold: float = 0.1,
    strict: bool = False,
) -> DataContract:
    """Generate a DataContract from a clean reference DataFrame.

    Args:
        df: Clean reference DataFrame (typically training data).
        name: Identifier for this contract.
        null_threshold: Maximum allowed null rate per column (0–1).
        drift_threshold: Maximum allowed distribution drift before violation.
        strict: If True, any single violation marks the entire contract as failed.

    Returns:
        DataContract instance that can be saved and reused.
    """
    dc = DataContract(
        name=name,
        null_threshold=null_threshold,
        drift_threshold=drift_threshold,
        strict=strict,
    )
    dc.fit(df)
    return dc


def validate(
    df: pd.DataFrame,
    data_contract: DataContract,
    *,
    raise_on_failure: bool = False,
) -> ContractReport:
    """Validate a DataFrame against a previously generated DataContract.

    Args:
        df: DataFrame to validate (typically production or new batch data).
        data_contract: Contract generated from ``datamend.contract()``.
        raise_on_failure: If True, raise ``ContractViolationError`` on any failure.

    Returns:
        ContractReport with all violations and a pass/fail verdict.
    """
    return data_contract.validate(df, raise_on_failure=raise_on_failure)


def drift(
    train_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    psi_buckets: int = 10,
    alpha: float = 0.05,
    verbose: bool = True,
) -> DriftReport:
    """Detect statistical drift between training data and production data.

    Args:
        train_df: Reference DataFrame (training distribution).
        prod_df: Current DataFrame (production distribution).
        columns: Subset of columns to check. If None, all shared columns are checked.
        psi_buckets: Number of bins used for PSI calculation.
        alpha: Significance level for KS and chi-square tests.
        verbose: If True, print a drift summary with MendScore to stdout.

    Returns:
        DriftReport with per-column drift scores and an overall MendScore.
    """
    radar = DriftRadar(psi_buckets=psi_buckets, alpha=alpha, verbose=verbose)
    return radar.detect(train_df, prod_df, columns=columns)


def trace(
    model: Any,
    df: pd.DataFrame,
    predictions: Any,
    *,
    ground_truth: Any | None = None,
    top_k: int = 10,
    verbose: bool = True,
) -> TraceReport:
    """Trace the root cause of model prediction failures at row and column level.

    Args:
        model: Any sklearn-compatible model, XGBoost, LightGBM, or PyTorch model.
        df: Input DataFrame used for predictions.
        predictions: Model predictions (array-like).
        ground_truth: Optional true labels for supervised failure attribution.
        top_k: Number of top contributing columns to surface in the report.
        verbose: If True, print a failure trace summary to stdout.

    Returns:
        TraceReport with suspicious rows, root-cause columns, and MendScore.
    """
    tracer = FailureTrace(top_k=top_k, verbose=verbose)
    return tracer.trace(model, df, predictions, ground_truth=ground_truth)
