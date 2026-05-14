"""Tests for the DriftRadar engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import datamend
from datamend.core.drift import (
    DriftRadar,
    DriftReport,
    _compute_jsd,
    _compute_jsd_categorical,
    _compute_psi,
    _psi_to_severity,
)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def test_psi_same_distribution() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 1000)
    b = rng.normal(0, 1, 1000)
    psi = _compute_psi(a, b)
    assert psi < 0.1, f"PSI should be low for same distribution, got {psi}"


def test_psi_different_distribution() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 1000)
    b = rng.normal(5, 1, 1000)
    psi = _compute_psi(a, b)
    assert psi >= 0.1, f"PSI should be high for different distributions, got {psi}"


def test_jsd_identical() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 500)
    jsd = _compute_jsd(a, a.copy())
    assert jsd < 0.05


def test_jsd_very_different() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 500)
    b = rng.normal(10, 1, 500)
    jsd = _compute_jsd(a, b)
    assert jsd > 0.1


def test_jsd_categorical_same() -> None:
    s = pd.Series(["A", "B", "C", "A", "B"] * 50)
    jsd = _compute_jsd_categorical(s, s.copy())
    assert jsd < 0.01


def test_jsd_categorical_different() -> None:
    s1 = pd.Series(["A", "A", "A", "B"] * 50)
    s2 = pd.Series(["C", "C", "C", "D"] * 50)
    jsd = _compute_jsd_categorical(s1, s2)
    assert jsd > 0.3


def test_psi_severity_mapping() -> None:
    assert _psi_to_severity(0.05) == "none"
    assert _psi_to_severity(0.15) == "low"
    assert _psi_to_severity(0.22) == "medium"
    assert _psi_to_severity(0.35) == "high"
    assert _psi_to_severity(0.6) == "critical"


# ---------------------------------------------------------------------------
# DriftRadar
# ---------------------------------------------------------------------------


class TestDriftRadar:
    def test_stable_data_no_drift(
        self, train_df: pd.DataFrame, prod_stable_df: pd.DataFrame
    ) -> None:
        report = datamend.drift(train_df, prod_stable_df, verbose=False)
        assert isinstance(report, DriftReport)
        assert not report.overall_drifted or len(report.columns_drifted) < 2

    def test_drifted_data_detected(
        self, train_df: pd.DataFrame, prod_drifted_df: pd.DataFrame
    ) -> None:
        report = datamend.drift(train_df, prod_drifted_df, verbose=False)
        assert report.overall_drifted
        assert len(report.columns_drifted) >= 1

    def test_mend_score_range(
        self, train_df: pd.DataFrame, prod_drifted_df: pd.DataFrame
    ) -> None:
        report = datamend.drift(train_df, prod_drifted_df, verbose=False)
        assert 0.0 <= report.mend_score <= 100.0

    def test_column_results_populated(
        self, train_df: pd.DataFrame, prod_stable_df: pd.DataFrame
    ) -> None:
        report = datamend.drift(train_df, prod_stable_df, verbose=False)
        assert len(report.column_results) == len(train_df.columns)

    def test_columns_subset(
        self, train_df: pd.DataFrame, prod_stable_df: pd.DataFrame
    ) -> None:
        report = datamend.drift(
            train_df, prod_stable_df, columns=["feature_a"], verbose=False
        )
        assert "feature_a" in report.column_results
        assert "feature_b" not in report.column_results

    def test_invalid_input_raises(self) -> None:
        with pytest.raises(TypeError):
            DriftRadar().detect("not_a_df", pd.DataFrame())  # type: ignore

    def test_no_shared_columns_raises(self) -> None:
        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"b": [1, 2, 3]})
        with pytest.raises(ValueError, match="No shared columns"):
            DriftRadar().detect(df1, df2)

    def test_report_to_dict(
        self, train_df: pd.DataFrame, prod_drifted_df: pd.DataFrame
    ) -> None:
        report = datamend.drift(train_df, prod_drifted_df, verbose=False)
        d = report.to_dict()
        assert "mend_score" in d
        assert "column_results" in d
        assert "overall_drifted" in d

    def test_report_summary_string(
        self, train_df: pd.DataFrame, prod_stable_df: pd.DataFrame
    ) -> None:
        report = datamend.drift(train_df, prod_stable_df, verbose=False)
        summary = report.summary()
        assert "DriftRadar" in summary
        assert "MendScore" in summary

    def test_numeric_column_has_psi(
        self, train_df: pd.DataFrame, prod_drifted_df: pd.DataFrame
    ) -> None:
        report = datamend.drift(train_df, prod_drifted_df, verbose=False)
        result = report.column_results["feature_a"]
        assert result.psi is not None
        assert result.ks_stat is not None

    def test_categorical_column_has_chi2(
        self, train_df: pd.DataFrame, prod_drifted_df: pd.DataFrame
    ) -> None:
        report = datamend.drift(train_df, prod_drifted_df, verbose=False)
        result = report.column_results["category"]
        assert result.jsd is not None

    def test_small_series_no_crash(self) -> None:
        df1 = pd.DataFrame({"x": [1.0, 2.0]})
        df2 = pd.DataFrame({"x": [3.0, 4.0]})
        report = DriftRadar(verbose=False).detect(df1, df2)
        assert isinstance(report, DriftReport)
