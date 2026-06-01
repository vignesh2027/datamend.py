"""Large-scale and edge-case stress tests for all four datamend pillars.

These tests go beyond unit tests: they use large DataFrames (10k–100k rows),
adversarial inputs, extreme distributions, and validate every documented
API pattern actually works end-to-end without error.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dirty_df(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate a large dirty DataFrame with every kind of quality issue."""
    rng = np.random.default_rng(seed)
    n_nulls = n // 10

    age = rng.integers(18, 65, n).astype(float)
    age[rng.choice(n, n_nulls, replace=False)] = np.nan
    age[rng.choice(n, 50, replace=False)] = 999  # outliers

    salary = rng.normal(60_000, 12_000, n)
    salary[rng.choice(n, n_nulls, replace=False)] = np.nan
    salary[rng.choice(n, 30, replace=False)] = -99_999  # outliers

    dept_vals = ["Engineering", "HR", "Data Science", "Finance", "Marketing"]
    dept = rng.choice(dept_vals, n)
    dept = dept.astype(object)
    dept[rng.choice(n, n_nulls, replace=False)] = None
    # add casing inconsistency
    mask = rng.random(n) < 0.05
    dept[mask] = np.where(mask, "engineering", dept)[mask]

    tenure = rng.integers(0, 20, n).astype(float)
    tenure[rng.choice(n, n_nulls // 2, replace=False)] = np.nan

    score = rng.uniform(0, 100, n)

    df = pd.DataFrame({"age": age, "salary": salary, "dept": dept,
                       "tenure": tenure, "score": score})

    # Add exact duplicates (5%)
    dup_count = n // 20
    dup_rows = df.sample(dup_count, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # Add whitespace noise
    df["dept"] = df["dept"].apply(
        lambda x: f"  {x}  " if x is not None and rng.random() < 0.03 else x
    )

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def _make_numeric_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.exponential(2, n),
        "c": rng.uniform(-10, 10, n),
        "d": rng.integers(0, 100, n).astype(float),
    })


# ---------------------------------------------------------------------------
# Pillar 1 — AutoRepair stress tests
# ---------------------------------------------------------------------------


class TestAutoRepairStress:
    def test_large_dataframe_10k(self) -> None:
        from datamend.core.repair import AutoRepair
        df = _make_dirty_df(10_000)
        _, report = AutoRepair(verbose=False).fit_transform(df)
        assert report.total_issues_found > 0
        assert report.mend_score_after > report.mend_score_before
        assert 0 <= report.mend_score_before <= 100
        assert 0 <= report.mend_score_after <= 100

    def test_large_dataframe_50k(self) -> None:
        from datamend.core.repair import AutoRepair
        df = _make_dirty_df(50_000)
        repaired, report = AutoRepair(verbose=False, fast_mode=True).fit_transform(df)
        assert len(repaired) > 0
        assert report.mend_score_after >= report.mend_score_before

    def test_empty_dataframe(self) -> None:
        from datamend.core.repair import AutoRepair
        df = pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=object)})
        repaired, report = AutoRepair(verbose=False).fit_transform(df)
        assert len(repaired) == 0
        assert report.total_issues_found == 0

    def test_all_nulls_column(self) -> None:
        from datamend.core.repair import AutoRepair
        df = pd.DataFrame({"x": [None] * 100, "y": range(100)})
        repaired, report = AutoRepair(verbose=False).fit_transform(df)
        # x column can't be imputed meaningfully (mode/mean of all-null is NaN)
        # Should not crash; may leave nulls or fill with fallback
        assert len(repaired) > 0

    def test_single_row(self) -> None:
        from datamend.core.repair import AutoRepair
        df = pd.DataFrame({"a": [1.0], "b": ["hello"]})
        repaired, report = AutoRepair(verbose=False).fit_transform(df)
        assert len(repaired) == 1

    def test_single_column(self) -> None:
        from datamend.core.repair import AutoRepair
        df = pd.DataFrame({"x": [1, 2, None, 4, 5, 999, 6, 7, 8, 9]})
        repaired, report = AutoRepair(verbose=False).fit_transform(df)
        assert repaired["x"].isnull().sum() == 0

    def test_all_duplicates(self) -> None:
        from datamend.core.repair import AutoRepair
        df = pd.DataFrame({"a": [1] * 100, "b": ["x"] * 100})
        repaired, report = AutoRepair(verbose=False).fit_transform(df)
        # All rows are duplicates; after dedup only 1 should remain
        assert len(repaired) == 1

    def test_no_issues(self) -> None:
        from datamend.core.repair import AutoRepair
        df = pd.DataFrame({"a": range(100), "b": [float(i) for i in range(100)]})
        _, report = AutoRepair(verbose=False).fit_transform(df)
        # score before should already be high
        assert report.mend_score_before > 50

    def test_non_dataframe_raises(self) -> None:
        from datamend.core.repair import AutoRepair
        with pytest.raises(TypeError):
            AutoRepair(verbose=False).fit_transform([1, 2, 3])  # type: ignore[arg-type]

    def test_mend_score_range(self) -> None:
        from datamend.core.repair import AutoRepair
        for seed in range(5):
            df = _make_dirty_df(500, seed=seed)
            _, report = AutoRepair(verbose=False).fit_transform(df)
            assert 0.0 <= report.mend_score_before <= 100.0
            assert 0.0 <= report.mend_score_after <= 100.0

    def test_chunked_processing(self) -> None:
        import math
        from datamend.core.repair import AutoRepair
        df = _make_dirty_df(1_000)
        chunk_size = 200
        engine = AutoRepair(verbose=False, chunk_size=chunk_size)
        repaired, reports = engine.repair_chunked(df)
        assert len(repaired) > 0
        assert len(reports) == math.ceil(len(df) / chunk_size)

    def test_categorical_imputation(self) -> None:
        from datamend.core.repair import AutoRepair
        df = pd.DataFrame({"cat": ["a", "a", "a", None, "b", None, "a"] * 10})
        repaired, _ = AutoRepair(verbose=False).fit_transform(df)
        assert repaired["cat"].isnull().sum() == 0

    def test_datetime_column_survives(self) -> None:
        from datamend.core.repair import AutoRepair
        df = pd.DataFrame({
            "ts": pd.date_range("2023-01-01", periods=50, freq="D"),
            "val": range(50),
        })
        repaired, _ = AutoRepair(verbose=False).fit_transform(df)
        assert len(repaired) == 50

    def test_whitespace_stripping(self) -> None:
        from datamend.core.repair import AutoRepair
        df = pd.DataFrame({"name": ["  Alice  ", "Bob\t", " Charlie", "Dave"]})
        repaired, report = AutoRepair(verbose=False).fit_transform(df)
        assert all(not v.startswith(" ") and not v.endswith(" ")
                   for v in repaired["name"] if isinstance(v, str))

    def test_type_coercion_numeric_strings(self) -> None:
        from datamend.core.repair import AutoRepair
        df = pd.DataFrame({"val": ["1.0", "2.0", "3.0", "4.0", "5.0"] * 20})
        repaired, report = AutoRepair(verbose=False).fit_transform(df)
        assert pd.api.types.is_numeric_dtype(repaired["val"])

    def test_report_serialisation(self) -> None:
        from datamend.core.repair import AutoRepair
        df = _make_dirty_df(200)
        _, report = AutoRepair(verbose=False).fit_transform(df)
        d = report.to_dict()
        assert "mend_score_before" in d
        assert "mend_score_after" in d
        assert "actions" in d
        assert isinstance(d["actions"], list)


# ---------------------------------------------------------------------------
# Pillar 2 — DataContract stress tests
# ---------------------------------------------------------------------------


class TestDataContractStress:
    def test_fit_validate_clean(self) -> None:
        from datamend.core.contract import DataContract
        df = _make_numeric_df(1_000)
        dc = DataContract()
        dc.fit(df)
        report = dc.validate(df)
        # Clean data fitted on itself should mostly pass
        assert report.mend_score > 0

    def test_schema_dict_api(self) -> None:
        from datamend.core.contract import DataContract
        schema = {
            "age": {"dtype": "numeric", "min": 0, "max": 120, "nullable": False},
            "salary": {"dtype": "numeric", "min": 0, "nullable": True},
            "dept": {"dtype": "categorical",
                     "allowed_values": ["Eng", "HR", "DS"], "nullable": False},
        }
        df_train = pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "salary": [50000, 60000, 70000, 80000, 90000],
            "dept": ["Eng", "HR", "DS", "Eng", "HR"],
        })
        dc = DataContract(schema)
        dc.fit(df_train)
        report = dc.validate(df_train)
        assert report.columns_checked == 3

    def test_string_first_arg_backward_compat(self) -> None:
        from datamend.core.contract import DataContract
        dc = DataContract("my_contract")
        assert dc.name == "my_contract"

    def test_fit_validate_convenience(self) -> None:
        from datamend.core.contract import DataContract
        df_train = _make_numeric_df(500, seed=1)
        df_prod = _make_numeric_df(200, seed=2)
        dc = DataContract()
        report = dc.fit_validate(df_train, df_prod)
        assert report.columns_checked == df_train.shape[1]

    def test_validate_before_fit_raises(self) -> None:
        from datamend.core.contract import DataContract
        dc = DataContract()
        with pytest.raises(RuntimeError, match="not been fitted"):
            dc.validate(pd.DataFrame({"a": [1, 2, 3]}))

    def test_missing_column_detected(self) -> None:
        from datamend.core.contract import DataContract
        train = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        prod = pd.DataFrame({"a": [1, 2, 3]})  # missing 'b'
        dc = DataContract().fit(train)
        report = dc.validate(prod)
        assert any(v.violation_type == "MISSING_COLUMN" for v in report.violations)

    def test_extra_column_is_warning(self) -> None:
        from datamend.core.contract import DataContract
        train = pd.DataFrame({"a": [1, 2, 3]})
        prod = pd.DataFrame({"a": [1, 2, 3], "extra": [7, 8, 9]})
        dc = DataContract().fit(train)
        report = dc.validate(prod)
        assert any(w.violation_type == "EXTRA_COLUMN" for w in report.warnings)

    def test_null_rate_violation(self) -> None:
        from datamend.core.contract import DataContract
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        prod = pd.DataFrame({"x": [None, None, None, 4.0, 5.0]})  # 60% null
        dc = DataContract(null_threshold=0.05).fit(train)
        report = dc.validate(prod)
        assert len(report.violations) > 0

    def test_save_load_roundtrip(self) -> None:
        from datamend.core.contract import DataContract
        df = _make_numeric_df(500)
        dc = DataContract().fit(df)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dc.save(path)
            dc2 = DataContract.load(path)
            assert set(dc2._specs.keys()) == set(dc._specs.keys())
            report = dc2.validate(df)
            assert report.columns_checked == df.shape[1]
        finally:
            os.unlink(path)

    def test_large_dataframe_100_cols(self) -> None:
        from datamend.core.contract import DataContract
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(0, 1, (500, 100)),
                          columns=[f"f{i}" for i in range(100)])
        dc = DataContract().fit(df)
        report = dc.validate(df)
        assert report.columns_checked == 100

    def test_categorical_allowed_values_violation(self) -> None:
        from datamend.core.contract import DataContract
        train = pd.DataFrame({"cat": ["a", "b", "c"] * 10})
        prod = pd.DataFrame({"cat": ["a", "b", "d", "e"]})  # d, e are new
        dc = DataContract().fit(train)
        report = dc.validate(prod)
        assert any(v.violation_type == "CARDINALITY_VIOLATION" for v in report.violations)

    def test_raise_on_failure(self) -> None:
        from datamend.core.contract import DataContract, ContractViolationError
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        prod = pd.DataFrame({"x": [None, None, None]})
        dc = DataContract(null_threshold=0.0).fit(train)
        with pytest.raises(ContractViolationError):
            dc.validate(prod, raise_on_failure=True)

    def test_report_to_json(self) -> None:
        from datamend.core.contract import DataContract
        df = _make_numeric_df(100)
        dc = DataContract().fit(df)
        report = dc.validate(df)
        j = report.to_json()
        parsed = json.loads(j)
        assert "mend_score" in parsed
        assert "violations" in parsed


# ---------------------------------------------------------------------------
# Pillar 3 — DriftRadar stress tests
# ---------------------------------------------------------------------------


class TestDriftRadarStress:
    def test_stable_identical_data(self) -> None:
        from datamend.core.drift import DriftRadar
        df = _make_numeric_df(1_000)
        report = DriftRadar(verbose=False).detect(df, df.copy())
        assert not report.overall_drifted
        assert report.mend_score < 20

    def test_severe_drift_detected(self) -> None:
        from datamend.core.drift import DriftRadar
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"x": rng.normal(0, 1, 1_000),
                               "y": rng.normal(5, 1, 1_000)})
        prod = pd.DataFrame({"x": rng.normal(10, 1, 1_000),
                              "y": rng.normal(15, 1, 1_000)})
        report = DriftRadar(verbose=False).detect(train, prod)
        assert report.overall_drifted
        assert report.mend_score > 30

    def test_fit_detect_pattern(self) -> None:
        from datamend.core.drift import DriftRadar
        rng = np.random.default_rng(1)
        train = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(3, 0.5, 500)})
        prod = pd.DataFrame({"a": rng.normal(0.1, 1, 200), "b": rng.normal(3.1, 0.5, 200)})
        radar = DriftRadar(verbose=False).fit(train)
        report = radar.detect(prod)
        assert report.train_shape[0] == 500
        assert report.prod_shape[0] == 200

    def test_detect_without_fit_two_args(self) -> None:
        from datamend.core.drift import DriftRadar
        df_a = _make_numeric_df(300, seed=0)
        df_b = _make_numeric_df(300, seed=5)
        report = DriftRadar(verbose=False).detect(df_a, df_b)
        assert len(report.column_results) == df_a.shape[1]

    def test_detect_single_arg_without_fit_raises(self) -> None:
        from datamend.core.drift import DriftRadar
        with pytest.raises(ValueError, match="prod_df is required"):
            DriftRadar(verbose=False).detect(pd.DataFrame({"a": [1, 2, 3]}))

    def test_no_shared_columns_raises(self) -> None:
        from datamend.core.drift import DriftRadar
        train = pd.DataFrame({"a": [1, 2, 3]})
        prod = pd.DataFrame({"b": [4, 5, 6]})
        with pytest.raises(ValueError, match="No shared columns"):
            DriftRadar(verbose=False).detect(train, prod)

    def test_categorical_drift(self) -> None:
        from datamend.core.drift import DriftRadar
        rng = np.random.default_rng(2)
        cats_train = rng.choice(["A", "B", "C"], 500, p=[0.6, 0.3, 0.1])
        cats_prod = rng.choice(["A", "B", "C"], 500, p=[0.1, 0.3, 0.6])
        train = pd.DataFrame({"cat": cats_train})
        prod = pd.DataFrame({"cat": cats_prod})
        report = DriftRadar(verbose=False).detect(train, prod)
        assert "cat" in report.column_results

    def test_mixed_types(self) -> None:
        from datamend.core.drift import DriftRadar
        rng = np.random.default_rng(3)
        train = pd.DataFrame({
            "num": rng.normal(0, 1, 500),
            "cat": rng.choice(["x", "y"], 500),
        })
        prod = pd.DataFrame({
            "num": rng.normal(0.5, 1.5, 200),
            "cat": rng.choice(["x", "y"], 200, p=[0.2, 0.8]),
        })
        report = DriftRadar(verbose=False).detect(train, prod)
        assert len(report.column_results) == 2

    def test_column_filter(self) -> None:
        from datamend.core.drift import DriftRadar
        df = _make_numeric_df(300)
        report = DriftRadar(verbose=False).detect(df, df.copy(), columns=["a", "b"])
        assert list(report.column_results.keys()) == ["a", "b"]

    def test_large_dataset_100k(self) -> None:
        from datamend.core.drift import DriftRadar
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"x": rng.normal(0, 1, 100_000),
                               "y": rng.exponential(1, 100_000)})
        prod = pd.DataFrame({"x": rng.normal(0.3, 1, 50_000),
                              "y": rng.exponential(1.2, 50_000)})
        report = DriftRadar(verbose=False).detect(train, prod)
        assert len(report.column_results) == 2

    def test_report_serialisation(self) -> None:
        from datamend.core.drift import DriftRadar
        df = _make_numeric_df(200)
        report = DriftRadar(verbose=False).detect(df, df.copy())
        d = report.to_dict()
        assert "mend_score" in d
        assert "column_results" in d
        j = json.dumps(d, default=str)
        parsed = json.loads(j)
        assert "column_results" in parsed

    def test_severity_labels(self) -> None:
        from datamend.core.drift import DriftRadar
        rng = np.random.default_rng(7)
        train = pd.DataFrame({"x": rng.normal(0, 1, 1_000)})
        prod = pd.DataFrame({"x": rng.normal(10, 1, 1_000)})
        report = DriftRadar(verbose=False).detect(train, prod)
        result = report.column_results["x"]
        assert result.severity in {"none", "low", "medium", "high", "critical"}

    def test_mend_score_range(self) -> None:
        from datamend.core.drift import DriftRadar
        df = _make_numeric_df(300)
        for shift in [0, 1, 5, 10]:
            rng = np.random.default_rng(shift)
            prod = df + shift
            report = DriftRadar(verbose=False).detect(df, prod)
            assert 0.0 <= report.mend_score <= 100.0


# ---------------------------------------------------------------------------
# Pillar 4 — FailureTrace stress tests
# ---------------------------------------------------------------------------


class TestFailureTraceStress:
    @pytest.fixture()
    def rf_model(self) -> Any:
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.standard_normal(500), "b": rng.standard_normal(500), "c": rng.standard_normal(500)})
        y = (X["a"] + X["b"] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=20, random_state=0)
        model.fit(X, y)
        return model

    def test_basic_trace(self, rf_model: Any) -> None:
        from datamend.core.trace import FailureTrace
        rng = np.random.default_rng(1)
        X_prod = pd.DataFrame({"a": rng.standard_normal(100), "b": rng.standard_normal(100), "c": rng.standard_normal(100)})
        preds = rf_model.predict(X_prod)
        report = FailureTrace(verbose=False).trace(rf_model, X_prod, preds)
        assert report.total_rows == 100
        assert 0.0 <= report.mend_score <= 100.0

    def test_fit_trace_pattern(self, rf_model: Any) -> None:
        from datamend.core.trace import FailureTrace
        rng = np.random.default_rng(2)
        X_train = pd.DataFrame({"a": rng.standard_normal(300), "b": rng.standard_normal(300), "c": rng.standard_normal(300)})
        y_train = (X_train["a"] > 0).astype(int)
        X_prod = pd.DataFrame({"a": rng.standard_normal(50), "b": rng.standard_normal(50), "c": rng.standard_normal(50)})
        preds = rf_model.predict(X_prod)
        tracer = FailureTrace(verbose=False).fit(rf_model, X_train, y_train)
        report = tracer.trace(rf_model, X_prod, preds)
        assert report.total_rows == 50

    def test_with_ground_truth(self, rf_model: Any) -> None:
        from datamend.core.trace import FailureTrace
        rng = np.random.default_rng(3)
        X_prod = pd.DataFrame({"a": rng.standard_normal(80), "b": rng.standard_normal(80), "c": rng.standard_normal(80)})
        preds = rf_model.predict(X_prod)
        truth = (X_prod["a"] + X_prod["b"] > 0).astype(int).to_numpy()
        report = FailureTrace(verbose=False).trace(rf_model, X_prod, preds, ground_truth=truth)
        assert len(report.column_attributions) > 0

    def test_large_dataset_10k(self, rf_model: Any) -> None:
        from datamend.core.trace import FailureTrace
        rng = np.random.default_rng(4)
        X_prod = pd.DataFrame({"a": rng.standard_normal(10_000), "b": rng.standard_normal(10_000),
                                "c": rng.standard_normal(10_000)})
        preds = rf_model.predict(X_prod)
        report = FailureTrace(verbose=False).trace(rf_model, X_prod, preds)
        assert report.total_rows == 10_000
        assert 0 <= report.mend_score <= 100

    def test_duplicate_index(self, rf_model: Any) -> None:
        from datamend.core.trace import FailureTrace
        rng = np.random.default_rng(5)
        X_prod = pd.DataFrame({"a": rng.standard_normal(50), "b": rng.standard_normal(50), "c": rng.standard_normal(50)},
                               index=list(range(25)) + list(range(25)))  # duplicate index!
        preds = rf_model.predict(X_prod)
        report = FailureTrace(verbose=False).trace(rf_model, X_prod, preds)
        assert report.total_rows == 50  # no crash

    def test_non_dataframe_raises(self, rf_model: Any) -> None:
        from datamend.core.trace import FailureTrace
        with pytest.raises(TypeError):
            FailureTrace(verbose=False).trace(rf_model, [[1, 2, 3]], [0])  # type: ignore

    def test_column_attributions_sorted(self, rf_model: Any) -> None:
        from datamend.core.trace import FailureTrace
        rng = np.random.default_rng(6)
        X_prod = pd.DataFrame({"a": rng.standard_normal(200), "b": rng.standard_normal(200), "c": rng.standard_normal(200)})
        preds = rf_model.predict(X_prod)
        report = FailureTrace(verbose=False).trace(rf_model, X_prod, preds)
        scores = [ca.importance_score for ca in report.column_attributions]
        assert scores == sorted(scores, reverse=True)

    def test_report_serialisation(self, rf_model: Any) -> None:
        from datamend.core.trace import FailureTrace
        rng = np.random.default_rng(7)
        X_prod = pd.DataFrame({"a": rng.standard_normal(100), "b": rng.standard_normal(100), "c": rng.standard_normal(100)})
        preds = rf_model.predict(X_prod)
        report = FailureTrace(verbose=False).trace(rf_model, X_prod, preds)
        d = report.to_dict()
        assert "total_rows" in d
        assert "column_attributions" in d
        assert "suspicious_rows" in d
        j = json.dumps(d, default=str)
        parsed = json.loads(j)
        assert parsed["total_rows"] == 100

    def test_with_nulls_in_prod(self, rf_model: Any) -> None:
        from datamend.core.trace import FailureTrace
        rng = np.random.default_rng(8)
        X_prod = pd.DataFrame({"a": rng.standard_normal(100), "b": rng.standard_normal(100), "c": rng.standard_normal(100)})
        X_prod.loc[::5, "a"] = np.nan  # add 20% nulls
        # fill for prediction, trace with nulls
        X_pred = X_prod.fillna(0)
        preds = rf_model.predict(X_pred)
        # FailureTrace should handle nulls in df without crashing
        report = FailureTrace(verbose=False).trace(rf_model, X_prod, preds)
        assert report.total_rows == 100


# ---------------------------------------------------------------------------
# MendReport (HTML) stress tests
# ---------------------------------------------------------------------------


class TestMendReportStress:
    def test_all_four_pillars(self) -> None:
        from sklearn.ensemble import RandomForestClassifier
        from datamend.core.repair import AutoRepair
        from datamend.core.contract import DataContract
        from datamend.core.drift import DriftRadar
        from datamend.core.trace import FailureTrace
        from datamend.report import MendReport

        rng = np.random.default_rng(0)
        df = _make_dirty_df(500)
        _, r1 = AutoRepair(verbose=False).fit_transform(df)

        train_clean = pd.DataFrame({"a": rng.standard_normal(200), "b": rng.standard_normal(200)})
        prod_clean = pd.DataFrame({"a": rng.standard_normal(100), "b": rng.standard_normal(100)})

        r2 = DataContract().fit_validate(train_clean, prod_clean)
        r3 = DriftRadar(verbose=False).detect(train_clean, prod_clean)

        X_tr = pd.DataFrame({"a": rng.standard_normal(150), "b": rng.standard_normal(150)})
        y_tr = (X_tr["a"] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_tr, y_tr)
        preds = model.predict(prod_clean)
        r4 = FailureTrace(verbose=False).trace(model, prod_clean, preds)

        # Test both naming conventions
        rep1 = MendReport(repair=r1, contract=r2, drift=r3, trace=r4)
        rep2 = MendReport(repair_report=r1, contract_report=r2, drift_report=r3, trace_report=r4)

        for rep in [rep1, rep2]:
            html = rep.to_html()
            assert len(html) > 2000
            assert "AutoRepair" in html
            assert "DataContract" in html
            assert "DriftRadar" in html
            assert "FailureTrace" in html
            j = json.loads(rep.to_json())
            assert "repair" in j and "contract" in j

    def test_partial_pillars(self) -> None:
        from datamend.core.repair import AutoRepair
        from datamend.report import MendReport

        df = _make_dirty_df(100)
        _, r1 = AutoRepair(verbose=False).fit_transform(df)
        rep = MendReport(repair=r1)
        html = rep.to_html()
        assert "AutoRepair" in html

    def test_to_html_saves_file(self) -> None:
        from datamend.core.repair import AutoRepair
        from datamend.report import MendReport

        df = _make_dirty_df(100)
        _, r1 = AutoRepair(verbose=False).fit_transform(df)
        rep = MendReport(repair=r1)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            html = rep.to_html(path)
            assert os.path.exists(path)
            with open(path, encoding="utf-8") as fh:
                on_disk = fh.read()
            assert on_disk == html
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# MendPipeline stress tests
# ---------------------------------------------------------------------------


class TestMendPipelineStress:
    def test_full_pipeline_no_model(self) -> None:
        from datamend import MendPipeline
        df = _make_dirty_df(1_000)
        pipe = MendPipeline(verbose=False, enable_trace=False)
        result = pipe.fit_transform(df)
        assert 0 <= result.overall_mend_score <= 100
        assert result.repair_report is not None
        assert result.contract_report is not None
        assert result.drift_report is not None

    def test_full_pipeline_with_model(self) -> None:
        from sklearn.ensemble import RandomForestClassifier
        from datamend import MendPipeline

        rng = np.random.default_rng(42)
        train = pd.DataFrame({"a": rng.standard_normal(300), "b": rng.standard_normal(300)})
        y_train = (train["a"] > 0).astype(int)
        prod = pd.DataFrame({"a": rng.standard_normal(100), "b": rng.standard_normal(100)})
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(train, y_train)
        preds = model.predict(prod)

        pipe = MendPipeline(verbose=False)
        pipe.fit(train)
        result = pipe.transform(prod, model=model, predictions=preds)
        assert result.trace_report is not None

    def test_pipeline_repair_only(self) -> None:
        from datamend import MendPipeline
        df = _make_dirty_df(500)
        pipe = MendPipeline(
            verbose=False,
            enable_contract=False,
            enable_drift=False,
            enable_trace=False,
        )
        result = pipe.fit_transform(df)
        assert result.contract_report is None
        assert result.drift_report is None
        assert result.trace_report is None

    def test_transform_before_fit_raises(self) -> None:
        from datamend import MendPipeline
        pipe = MendPipeline(verbose=False)
        with pytest.raises(RuntimeError, match="fitted"):
            pipe.transform(pd.DataFrame({"a": [1, 2, 3]}))

    def test_to_json(self) -> None:
        from datamend import MendPipeline
        df = _make_dirty_df(200)
        pipe = MendPipeline(verbose=False, enable_trace=False)
        result = pipe.fit_transform(df)
        j = result.to_json()
        parsed = json.loads(j)
        assert "overall_mend_score" in parsed
        assert "repair" in parsed

    def test_large_dataset_50k(self) -> None:
        from datamend import MendPipeline
        df = _make_dirty_df(50_000)
        pipe = MendPipeline(verbose=False, enable_trace=False, fast_mode=True)
        result = pipe.fit_transform(df)
        assert result.overall_mend_score >= 0


# ---------------------------------------------------------------------------
# Top-level convenience API
# ---------------------------------------------------------------------------


class TestTopLevelAPI:
    def test_datamend_repair(self) -> None:
        import datamend
        df = _make_dirty_df(200)
        repaired, report = datamend.repair(df, verbose=False)
        assert isinstance(repaired, pd.DataFrame)
        assert report.total_issues_found >= 0

    def test_datamend_contract_validate(self) -> None:
        import datamend
        train = _make_numeric_df(300)
        prod = _make_numeric_df(100, seed=99)
        dc = datamend.contract(train)
        report = datamend.validate(prod, dc)
        assert report.columns_checked == train.shape[1]

    def test_datamend_drift(self) -> None:
        import datamend
        train = _make_numeric_df(500)
        prod = _make_numeric_df(200, seed=7)
        report = datamend.drift(train, prod, verbose=False)
        assert 0 <= report.mend_score <= 100

    def test_datamend_trace(self) -> None:
        from sklearn.ensemble import RandomForestClassifier
        import datamend

        rng = np.random.default_rng(0)
        X = pd.DataFrame({"a": rng.standard_normal(200), "b": rng.standard_normal(200)})
        y = (X["a"] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        preds = model.predict(X)
        report = datamend.trace(model, X, preds, verbose=False)
        assert report.total_rows == 200

    def test_version_string(self) -> None:
        import datamend
        assert datamend.__version__ == "1.1.3"
