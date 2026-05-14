"""Tests for the AutoRepair engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import datamend
from datamend.core.repair import (
    AutoRepair,
    RepairReport,
    _compute_mend_score,
    _NullDetector,
    _OutlierDetector,
    _DuplicateDetector,
    _CategoryNormalisationDetector,
    _WhitespaceDetector,
    _EncodingDetector,
    _TypeMismatchDetector,
)


# ---------------------------------------------------------------------------
# _compute_mend_score
# ---------------------------------------------------------------------------


def test_mend_score_clean_df(clean_df: pd.DataFrame) -> None:
    score = _compute_mend_score(clean_df)
    assert 70.0 <= score <= 100.0, f"Expected score ≥70 for clean df, got {score}"


def test_mend_score_dirty_df(dirty_df: pd.DataFrame) -> None:
    clean_score = _compute_mend_score(dirty_df.dropna())
    dirty_score = _compute_mend_score(dirty_df)
    assert dirty_score <= clean_score + 5


def test_mend_score_empty_df() -> None:
    assert _compute_mend_score(pd.DataFrame()) == 100.0


def test_mend_score_all_nulls() -> None:
    df = pd.DataFrame({"a": [None, None, None], "b": [None, None, None]})
    score = _compute_mend_score(df)
    assert score < 50.0


# ---------------------------------------------------------------------------
# _NullDetector
# ---------------------------------------------------------------------------


class TestNullDetector:
    def test_detect_nulls(self) -> None:
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, None, None], "c": [1, 2, 3]})
        nd = _NullDetector()
        result = nd.detect(df)
        assert "a" in result
        assert result["a"] == 1
        assert "b" in result
        assert result["b"] == 3
        assert "c" not in result

    def test_fix_numeric_mean(self) -> None:
        df = pd.DataFrame({"val": [1.0, None, 3.0, None, 5.0]})
        nd = _NullDetector(strategy="mean")
        null_counts = nd.detect(df)
        fixed_df, actions = nd.fix(df, null_counts)
        assert fixed_df["val"].isnull().sum() == 0
        assert len(actions) == 1
        assert actions[0].issue_type == "NULL"

    def test_fix_numeric_median(self) -> None:
        df = pd.DataFrame({"val": [1.0, None, 3.0]})
        nd = _NullDetector(strategy="median")
        null_counts = nd.detect(df)
        fixed_df, actions = nd.fix(df, null_counts)
        assert fixed_df["val"].isnull().sum() == 0

    def test_fix_categorical_mode(self) -> None:
        df = pd.DataFrame({"cat": ["a", "a", None, "b", "a", None]})
        nd = _NullDetector()
        null_counts = nd.detect(df)
        fixed_df, actions = nd.fix(df, null_counts)
        assert fixed_df["cat"].isnull().sum() == 0
        assert fixed_df["cat"].iloc[2] == "a"

    def test_fix_no_nulls(self) -> None:
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        nd = _NullDetector()
        null_counts = nd.detect(df)
        fixed_df, actions = nd.fix(df, null_counts)
        assert actions == []


# ---------------------------------------------------------------------------
# _OutlierDetector
# ---------------------------------------------------------------------------


class TestOutlierDetector:
    def test_detect_outliers(self) -> None:
        rng = np.random.default_rng(42)
        normal = rng.normal(0, 1, 100).tolist()
        normal += [100.0, -100.0, 200.0]
        df = pd.DataFrame({"vals": normal})
        od = _OutlierDetector(z_threshold=3.0)
        result = od.detect(df)
        assert "vals" in result
        idx, vals = result["vals"]
        assert len(idx) >= 2

    def test_fix_outliers(self) -> None:
        rng = np.random.default_rng(42)
        normal = rng.normal(50, 5, 100).tolist()
        normal[0] = 10_000.0
        df = pd.DataFrame({"x": normal})
        od = _OutlierDetector()
        outliers = od.detect(df)
        fixed_df, actions = od.fix(df, outliers)
        assert fixed_df["x"].max() < 1000.0
        assert len(actions) == 1
        assert actions[0].issue_type == "OUTLIER"

    def test_no_outliers_no_actions(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0] * 5})
        od = _OutlierDetector()
        outliers = od.detect(df)
        _, actions = od.fix(df, outliers)
        assert actions == []


# ---------------------------------------------------------------------------
# _DuplicateDetector
# ---------------------------------------------------------------------------


class TestDuplicateDetector:
    def test_detect_exact_duplicates(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 1, 2], "b": ["x", "y", "z", "x", "y"]})
        dd = _DuplicateDetector()
        idx = dd.detect_exact(df)
        assert len(idx) == 2

    def test_fix_exact_duplicates(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
        dd = _DuplicateDetector()
        exact_idx = dd.detect_exact(df)
        fixed_df, actions = dd.fix(df, exact_idx, [])
        assert len(fixed_df) == 3
        assert not fixed_df.duplicated().any()

    def test_no_duplicates(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        dd = _DuplicateDetector()
        idx = dd.detect_exact(df)
        assert len(idx) == 0


# ---------------------------------------------------------------------------
# _CategoryNormalisationDetector
# ---------------------------------------------------------------------------


class TestCategoryNormalisation:
    def test_detect_inconsistent(self) -> None:
        df = pd.DataFrame({"gender": ["male", "Male", "MALE", "female", "Female"]})
        det = _CategoryNormalisationDetector()
        mappings = det.detect(df)
        assert "gender" in mappings
        assert len(mappings["gender"]) > 0

    def test_fix_normalises_values(self) -> None:
        df = pd.DataFrame({"gender": ["male", "Male", "MALE", "female", "Female"] * 10})
        det = _CategoryNormalisationDetector()
        mappings = det.detect(df)
        fixed_df, actions = det.fix(df, mappings)
        unique_vals = fixed_df["gender"].unique()
        assert len(unique_vals) <= 2

    def test_high_cardinality_skipped(self) -> None:
        vals = [f"val_{i}" for i in range(100)]
        df = pd.DataFrame({"col": vals})
        det = _CategoryNormalisationDetector()
        mappings = det.detect(df)
        assert "col" not in mappings


# ---------------------------------------------------------------------------
# _WhitespaceDetector
# ---------------------------------------------------------------------------


class TestWhitespaceDetector:
    def test_detect_whitespace(self) -> None:
        df = pd.DataFrame({"name": ["  Alice", "Bob  ", "\tCharlie", "Dave"]})
        wd = _WhitespaceDetector()
        result = wd.detect(df)
        assert "name" in result
        assert result["name"] >= 3

    def test_fix_strips_whitespace(self) -> None:
        df = pd.DataFrame({"name": ["  Alice", "Bob  ", "Charlie"]})
        wd = _WhitespaceDetector()
        affected = wd.detect(df)
        fixed_df, actions = wd.fix(df, affected)
        assert fixed_df["name"].iloc[0] == "Alice"
        assert fixed_df["name"].iloc[1] == "Bob"


# ---------------------------------------------------------------------------
# _TypeMismatchDetector
# ---------------------------------------------------------------------------


class TestTypeMismatchDetector:
    def test_detect_numeric_string_column(self) -> None:
        df = pd.DataFrame({"val": ["1.5", "2.3", "4.1", "5.0"] * 50})
        det = _TypeMismatchDetector()
        candidates = det.detect(df)
        assert "val" in candidates
        assert candidates["val"] == "numeric"

    def test_fix_converts_to_numeric(self) -> None:
        df = pd.DataFrame({"val": ["1.0", "2.0", "3.0", "4.0", "5.0"] * 40})
        det = _TypeMismatchDetector()
        candidates = det.detect(df)
        fixed_df, actions = det.fix(df, candidates)
        assert pd.api.types.is_numeric_dtype(fixed_df["val"])
        assert len(actions) == 1


# ---------------------------------------------------------------------------
# AutoRepair end-to-end
# ---------------------------------------------------------------------------


class TestAutoRepair:
    def test_repair_dirty_df(self, dirty_df: pd.DataFrame) -> None:
        repaired, report = datamend.repair(dirty_df, verbose=False)
        assert isinstance(repaired, pd.DataFrame)
        assert isinstance(report, RepairReport)
        assert repaired["age"].isnull().sum() == 0

    def test_repair_increases_score(self, dirty_df: pd.DataFrame) -> None:
        repaired, report = datamend.repair(dirty_df, verbose=False)
        assert report.mend_score_after >= report.mend_score_before

    def test_repair_returns_dataframe(self, clean_df: pd.DataFrame) -> None:
        repaired, report = datamend.repair(clean_df, verbose=False)
        assert isinstance(repaired, pd.DataFrame)
        assert len(repaired) > 0

    def test_repair_report_has_actions(self, dirty_df: pd.DataFrame) -> None:
        _, report = datamend.repair(dirty_df, verbose=False)
        assert len(report.actions) > 0

    def test_repair_preserves_shape_approx(self, clean_df: pd.DataFrame) -> None:
        repaired, _ = datamend.repair(clean_df, verbose=False)
        assert repaired.shape[1] == clean_df.shape[1]

    def test_repair_invalid_input_raises(self) -> None:
        with pytest.raises(TypeError):
            engine = AutoRepair(verbose=False)
            engine.fit_transform([1, 2, 3])  # type: ignore

    def test_repair_report_to_dict(self, dirty_df: pd.DataFrame) -> None:
        _, report = datamend.repair(dirty_df, verbose=False)
        d = report.to_dict()
        assert "total_issues_found" in d
        assert "actions" in d
        assert "mend_score_after" in d

    def test_repair_chunked(self, dirty_df: pd.DataFrame) -> None:
        engine = AutoRepair(verbose=False, chunk_size=50)
        repaired, reports = engine.repair_chunked(dirty_df)
        assert isinstance(repaired, pd.DataFrame)
        assert len(reports) > 1

    def test_fast_mode(self, dirty_df: pd.DataFrame) -> None:
        repaired, report = datamend.repair(dirty_df, fast_mode=True, verbose=False)
        assert isinstance(repaired, pd.DataFrame)

    def test_strategies(self, dirty_df: pd.DataFrame) -> None:
        for strategy in ["mean", "median"]:
            repaired, _ = datamend.repair(dirty_df, strategy=strategy, verbose=False)
            assert repaired["age"].isnull().sum() == 0
