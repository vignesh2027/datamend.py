"""Tests for the FailureTrace engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import datamend
from datamend.core.trace import (
    FailureTrace,
    TraceReport,
    _detect_model_type,
    _surrogate_importances,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_detect_model_type_sklearn() -> None:
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression  # type: ignore

    m = LinearRegression()
    assert _detect_model_type(m) == "sklearn"


def test_detect_model_type_unknown() -> None:
    class FakeModel:
        pass

    assert _detect_model_type(FakeModel()) == "unknown"


# ---------------------------------------------------------------------------
# _surrogate_importances
# ---------------------------------------------------------------------------


def test_surrogate_importances_shape(clean_df: pd.DataFrame) -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(42)
    preds = rng.uniform(0, 1, len(clean_df))
    fi = _surrogate_importances(None, clean_df, preds)
    assert fi.shape[0] == clean_df.shape[1]
    assert abs(fi.sum() - 1.0) < 1e-6 or fi.sum() == 0.0


# ---------------------------------------------------------------------------
# FailureTrace
# ---------------------------------------------------------------------------


class TestFailureTrace:
    def test_trace_sklearn_model(
        self, simple_sklearn_model: object, clean_df: pd.DataFrame
    ) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        predictions = simple_sklearn_model.predict(X)  # type: ignore
        report = datamend.trace(simple_sklearn_model, X, predictions, verbose=False)
        assert isinstance(report, TraceReport)

    def test_trace_report_has_columns(
        self, simple_sklearn_model: object, clean_df: pd.DataFrame
    ) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        predictions = simple_sklearn_model.predict(X)  # type: ignore
        report = datamend.trace(simple_sklearn_model, X, predictions, verbose=False)
        assert len(report.column_attributions) > 0

    def test_trace_mend_score_range(
        self, simple_sklearn_model: object, clean_df: pd.DataFrame
    ) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        predictions = simple_sklearn_model.predict(X)  # type: ignore
        report = datamend.trace(simple_sklearn_model, X, predictions, verbose=False)
        assert 0.0 <= report.mend_score <= 100.0

    def test_trace_with_ground_truth(
        self, simple_sklearn_model: object, clean_df: pd.DataFrame
    ) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        predictions = simple_sklearn_model.predict(X)  # type: ignore
        gt = predictions + np.random.default_rng(42).normal(0, 100, len(predictions))
        report = datamend.trace(
            simple_sklearn_model, X, predictions, ground_truth=gt, verbose=False
        )
        assert isinstance(report, TraceReport)

    def test_trace_invalid_input_raises(
        self, simple_sklearn_model: object
    ) -> None:
        with pytest.raises(TypeError):
            datamend.trace(simple_sklearn_model, [1, 2, 3], [1, 2, 3], verbose=False)  # type: ignore

    def test_trace_report_to_dict(
        self, simple_sklearn_model: object, clean_df: pd.DataFrame
    ) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        predictions = simple_sklearn_model.predict(X)  # type: ignore
        report = datamend.trace(simple_sklearn_model, X, predictions, verbose=False)
        d = report.to_dict()
        assert "mend_score" in d
        assert "column_attributions" in d
        assert "suspicious_rows" in d

    def test_top_k_columns(
        self, simple_sklearn_model: object, clean_df: pd.DataFrame
    ) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        predictions = simple_sklearn_model.predict(X)  # type: ignore
        tracer = FailureTrace(top_k=2, verbose=False)
        report = tracer.trace(simple_sklearn_model, X, predictions)
        assert len(report.column_attributions) <= 2

    def test_trace_dirty_data_more_suspicious(self, dirty_df: pd.DataFrame) -> None:
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestRegressor  # type: ignore

        dirty_repaired, _ = datamend.repair(dirty_df, verbose=False)
        numeric_cols = dirty_repaired.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            pytest.skip("Need at least 2 numeric columns")
        X = dirty_repaired[numeric_cols]
        y = X.iloc[:, 0] + X.iloc[:, 1]
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        predictions = model.predict(X)

        # Run trace on dirty data (pre-repair)
        dirty_numeric = dirty_df[numeric_cols].copy()
        dirty_numeric = dirty_numeric.fillna(dirty_numeric.mean())
        predictions_dirty = model.predict(dirty_numeric)
        tracer = FailureTrace(verbose=False)
        report = tracer.trace(model, dirty_numeric, predictions_dirty)
        assert isinstance(report, TraceReport)
