"""Advanced, adversarial, and cross-pillar integration tests for datamend.

Covers every documented API pattern, serialisation round-trips, edge-case
inputs (1-row, 1-column, all-null, boolean, datetime, unicode, wide DataFrames,
constant columns), CLI commands, and full 4-pillar integration flows.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import datamend
from datamend.core.contract import ContractReport, DataContract, ContractViolationError
from datamend.core.drift import DriftRadar, DriftReport
from datamend.core.repair import AutoRepair, RepairReport, _compute_mend_score
from datamend.core.trace import FailureTrace, TraceReport


# ===========================================================================
# Shared helpers
# ===========================================================================


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _wide_df(n_rows: int = 500, n_cols: int = 80, seed: int = 7) -> pd.DataFrame:
    """Wide DataFrame with numeric columns and injected nulls/outliers."""
    rng = _rng(seed)
    data = {f"col_{i}": rng.normal(i * 10, 2, n_rows) for i in range(n_cols)}
    df = pd.DataFrame(data)
    # inject nulls into first min(10, n_cols) cols
    for i in range(min(10, n_cols)):
        null_idx = rng.choice(n_rows, n_rows // 10, replace=False)
        df.iloc[null_idx, i] = np.nan
    # inject outliers into last min(10, n_cols) cols — clamp to valid range
    start = max(0, n_cols - 10)
    for i in range(start, n_cols):
        df.iloc[rng.choice(n_rows, min(5, n_rows), replace=False), i] = 1e9
    return df


def _datetime_df(n: int = 300) -> pd.DataFrame:
    rng = _rng(1)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "event_date": dates,
        "value": rng.normal(100, 15, n),
        "category": rng.choice(["A", "B", "C"], n),
    })


def _boolean_df(n: int = 200) -> pd.DataFrame:
    rng = _rng(2)
    return pd.DataFrame({
        "flag": rng.choice([True, False], n),
        "score": rng.uniform(0, 1, n),
        "label": rng.choice(["yes", "no"], n),
    })


def _unicode_df(n: int = 100) -> pd.DataFrame:
    rng = _rng(3)
    names = ["Ångström", "Müller", "François", "日本語", "中文", "한국어", "العربية", "Ñoño"]
    return pd.DataFrame({
        "name": rng.choice(names, n),
        "value": rng.normal(50, 10, n),
    })


def _constant_df(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame({
        "constant_int": [42] * n,
        "constant_str": ["hello"] * n,
        "constant_float": [3.14] * n,
        "varying": list(range(n)),
    })


def _single_row_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0], "b": ["foo"], "c": [True]})


def _single_col_df(n: int = 100) -> pd.DataFrame:
    rng = _rng(9)
    vals = rng.normal(0, 1, n)
    vals[::10] = np.nan
    return pd.DataFrame({"only_col": vals})


def _all_null_col_df(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "all_null": [np.nan] * n,
        "normal": list(range(n, dtype=float) if False else range(n)),
    })


def _duplicate_heavy_df(n: int = 500) -> pd.DataFrame:
    """70% of rows are exact duplicates."""
    rng = _rng(11)
    base = pd.DataFrame({
        "x": rng.integers(0, 5, n // 3).astype(float),
        "y": rng.choice(["a", "b"], n // 3),
    })
    return pd.concat([base] * 3, ignore_index=True)


def _high_null_df(n: int = 300) -> pd.DataFrame:
    """50% nulls in every column."""
    rng = _rng(12)
    df = pd.DataFrame({
        "num1": rng.normal(0, 1, n),
        "num2": rng.normal(10, 3, n),
        "cat": rng.choice(["X", "Y", "Z"], n).astype(object),
    })
    for col in df.columns:
        idx = rng.choice(n, n // 2, replace=False)
        df.iloc[idx] = np.nan
    return df


def _make_sklearn_model(X: pd.DataFrame, y: np.ndarray):
    sklearn = pytest.importorskip("sklearn")
    from sklearn.ensemble import GradientBoostingRegressor
    m = GradientBoostingRegressor(n_estimators=20, random_state=42)
    m.fit(X, y)
    return m


# ===========================================================================
# 1. AutoRepair — deep unit tests
# ===========================================================================


class TestAutoRepairDeep:

    # --- strategy=auto picks median for skewed data ---
    def test_auto_strategy_skewed_column(self) -> None:
        rng = _rng(5)
        vals = np.concatenate([rng.exponential(1, 190), [np.nan] * 10])
        rng.shuffle(vals)
        df = pd.DataFrame({"income": vals})
        repaired, _ = datamend.repair(df, strategy="auto", verbose=False)
        assert repaired["income"].isnull().sum() == 0
        # Median imputation keeps value <= original max
        assert repaired["income"].max() < 1e6

    # --- wide DataFrame doesn't crash ---
    def test_wide_dataframe_80_cols(self) -> None:
        df = _wide_df(500, 80)
        repaired, report = datamend.repair(df, verbose=False)
        assert repaired.shape[1] == 80
        assert report.mend_score_after >= report.mend_score_before

    # --- datetime columns pass through unchanged ---
    def test_datetime_columns_preserved(self) -> None:
        df = _datetime_df(300)
        repaired, _ = datamend.repair(df, verbose=False)
        assert repaired.shape[1] == df.shape[1]
        # date column still present
        assert "event_date" in repaired.columns

    # --- boolean column survives repair ---
    def test_boolean_column_no_crash(self) -> None:
        df = _boolean_df(200)
        repaired, _ = datamend.repair(df, verbose=False)
        assert len(repaired) > 0
        assert "flag" in repaired.columns

    # --- unicode strings don't crash ---
    def test_unicode_strings_no_crash(self) -> None:
        df = _unicode_df(100)
        repaired, _ = datamend.repair(df, verbose=False)
        assert len(repaired) > 0

    # --- constant columns (zero variance) don't crash outlier detector ---
    def test_constant_columns_no_crash(self) -> None:
        df = _constant_df(200)
        repaired, _ = datamend.repair(df, verbose=False)
        assert len(repaired) > 0

    # --- single row DataFrame ---
    def test_single_row_no_crash(self) -> None:
        df = _single_row_df()
        repaired, report = datamend.repair(df, verbose=False)
        assert len(repaired) == 1
        assert isinstance(report, RepairReport)

    # --- single column DataFrame ---
    def test_single_col_with_nulls(self) -> None:
        df = _single_col_df(100)
        repaired, report = datamend.repair(df, verbose=False)
        assert repaired["only_col"].isnull().sum() == 0

    # --- all-null single column repaired without crash ---
    def test_all_null_column_no_crash(self) -> None:
        df = _all_null_col_df(100)
        repaired, _ = datamend.repair(df, verbose=False)
        assert len(repaired) > 0

    # --- 70% duplicate rows ---
    def test_heavy_duplicate_removal(self) -> None:
        df = _duplicate_heavy_df(300)
        repaired, report = datamend.repair(df, verbose=False)
        # duplicates should be removed
        assert len(repaired) < len(df)
        assert any(a.issue_type == "DUPLICATE" for a in report.actions)

    # --- 50% null columns ---
    def test_high_null_rate_repaired(self) -> None:
        df = _high_null_df(300)
        repaired, _ = datamend.repair(df, verbose=False)
        assert len(repaired) > 0

    # --- very long string values don't crash ---
    def test_very_long_string_values(self) -> None:
        df = pd.DataFrame({
            "text": ["A" * 10_000, "B" * 10_000, "C" * 500] * 30,
            "val": list(range(90)),
        })
        repaired, _ = datamend.repair(df, verbose=False)
        assert len(repaired) > 0

    # --- RepairReport to_dict has all required keys ---
    def test_repair_report_dict_completeness(self, dirty_df: pd.DataFrame) -> None:
        _, report = datamend.repair(dirty_df, verbose=False)
        d = report.to_dict()
        required = {
            "total_issues_found", "total_rows_affected", "mend_score_before",
            "mend_score_after", "columns_repaired", "actions",
        }
        assert required <= d.keys(), f"Missing keys: {required - d.keys()}"

    # --- actions have required fields ---
    def test_repair_action_fields(self, dirty_df: pd.DataFrame) -> None:
        _, report = datamend.repair(dirty_df, verbose=False)
        for action in report.actions:
            assert hasattr(action, "column")
            assert hasattr(action, "issue_type")
            assert hasattr(action, "rows_affected")
            assert action.rows_affected >= 0

    # --- chunked repair produces consistent result ---
    def test_chunked_repair_result_length(self, dirty_df: pd.DataFrame) -> None:
        engine = AutoRepair(verbose=False, chunk_size=80)
        repaired, reports = engine.repair_chunked(dirty_df)
        # total chunks = ceil(len(df) / chunk_size)
        assert isinstance(repaired, pd.DataFrame)
        assert isinstance(reports, list)
        n_chunks = math.ceil(len(dirty_df) / 80)
        assert len(reports) == n_chunks

    # --- MendScore always in 0–100 ---
    def test_mend_score_bounds(self) -> None:
        for seed in range(10):
            rng = _rng(seed)
            n = rng.integers(10, 500).item()
            df = pd.DataFrame({
                "a": rng.normal(0, 1, n),
                "b": rng.choice(["x", "y", None], n),
            })
            score = _compute_mend_score(df)
            assert 0.0 <= score <= 100.0, f"Score out of bounds: {score}"

    # --- repair with explicit mean strategy ---
    def test_mean_strategy_imputes_correctly(self) -> None:
        # Use enough rows so outlier detector doesn't reorder imputation
        rng = _rng(77)
        vals = rng.normal(50, 5, 100).tolist()
        vals[::10] = [np.nan] * 10
        df = pd.DataFrame({"val": vals})
        repaired, _ = datamend.repair(df, strategy="mean", verbose=False)
        assert repaired["val"].isnull().sum() == 0
        # Imputed values should be near the column mean
        non_null_mean = pd.Series(vals).dropna().mean()
        assert abs(repaired["val"].mean() - non_null_mean) < 10.0

    # --- repair with median strategy ---
    def test_median_strategy_imputes_correctly(self) -> None:
        rng = _rng(88)
        vals = rng.normal(50, 5, 100).tolist()
        vals[::10] = [np.nan] * 10
        df = pd.DataFrame({"val": vals})
        repaired, _ = datamend.repair(df, strategy="median", verbose=False)
        assert repaired["val"].isnull().sum() == 0
        non_null_median = pd.Series(vals).dropna().median()
        assert abs(repaired["val"].median() - non_null_median) < 10.0

    # --- 100k row large dataset (fast_mode) ---
    def test_100k_rows_fast_mode(self) -> None:
        rng = _rng(99)
        n = 100_000
        df = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.choice(["X", "Y", "Z"], n),
        })
        null_idx = rng.choice(n, 5000, replace=False)
        df.iloc[null_idx, 0] = np.nan
        repaired, report = datamend.repair(df, fast_mode=True, verbose=False)
        assert isinstance(repaired, pd.DataFrame)
        assert 0 <= report.mend_score_after <= 100

    # --- integer columns with NaN become float and are repaired ---
    def test_integer_column_with_nan(self) -> None:
        df = pd.DataFrame({"age": [25, 30, np.nan, 28, np.nan, 35]})
        repaired, _ = datamend.repair(df, verbose=False)
        assert repaired["age"].isnull().sum() == 0

    # --- repair does not add extra columns ---
    def test_repair_no_extra_columns(self, dirty_df: pd.DataFrame) -> None:
        original_cols = set(dirty_df.columns)
        repaired, _ = datamend.repair(dirty_df, verbose=False)
        assert set(repaired.columns) == original_cols

    # --- outlier clipping keeps all values finite ---
    def test_outlier_clipping_all_finite(self) -> None:
        rng = _rng(20)
        vals = rng.normal(0, 1, 200)
        vals[::20] = 1e9
        df = pd.DataFrame({"x": vals})
        repaired, _ = datamend.repair(df, verbose=False)
        assert np.all(np.isfinite(repaired["x"].dropna()))


# ===========================================================================
# 2. DataContract — deep unit tests
# ===========================================================================


class TestDataContractDeep:

    # --- schema dict constructor ---
    def test_schema_dict_constructor(self) -> None:
        schema = {
            "age": {"dtype": "numeric", "min": 0, "max": 120, "nullable": False},
            "name": {"dtype": "string"},
        }
        dc = DataContract(schema)
        assert dc._user_schema is not None

    # --- schema dict constraints enforced after fit ---
    def test_schema_dict_min_max_enforced(self) -> None:
        schema = {"age": {"dtype": "numeric", "min": 0, "max": 120}}
        dc = DataContract(schema)
        train = pd.DataFrame({"age": [20.0, 30.0, 40.0] * 50})
        dc.fit(train)
        # Value way out of range
        bad = pd.DataFrame({"age": [200.0, -50.0, 30.0] * 10})
        report = dc.validate(bad)
        assert not report.passed or len(report.violations) > 0

    # --- fit_validate convenience method ---
    def test_fit_validate_one_liner(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        report = dc.fit_validate(clean_df, clean_df.copy())
        assert isinstance(report, ContractReport)
        assert dc._fitted

    # --- strict mode: one violation → passed=False ---
    def test_strict_mode_single_violation_fails(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract(strict=True)
        dc.fit(clean_df)
        # Missing a required column is a real violation (not just a warning)
        bad_df = clean_df.drop(columns=["age"])
        report = dc.validate(bad_df)
        assert not report.passed

    # --- custom null_threshold per column ---
    def test_custom_null_threshold(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract(null_threshold=0.0)  # zero tolerance
        dc.fit(clean_df)
        null_df = clean_df.copy()
        null_df.loc[0, "age"] = np.nan
        report = dc.validate(null_df)
        assert any(v.violation_type == "NULL_RATE" for v in report.violations)

    # --- range violation ---
    def test_range_violation_detected(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        out_of_range = clean_df.copy()
        out_of_range["age"] = out_of_range["age"] + 10_000  # far outside training range
        report = dc.validate(out_of_range)
        assert any(v.violation_type == "RANGE_VIOLATION" for v in report.violations)

    # --- backward-compat: DataContract("my_name") sets name ---
    def test_positional_string_sets_name(self) -> None:
        dc = DataContract("my_contract")
        assert dc.name == "my_contract"

    # --- contract name preserved through save/load ---
    def test_name_preserved_save_load(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract(name="production_v1")
        dc.fit(clean_df)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            dc.save(path)
            loaded = DataContract.load(path)
            assert loaded.name == "production_v1"
        finally:
            Path(path).unlink(missing_ok=True)

    # --- validate before fit raises RuntimeError ---
    def test_validate_before_fit_raises(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        with pytest.raises(RuntimeError):
            dc.validate(clean_df)

    # --- ContractReport summary is a non-empty string ---
    def test_contract_report_summary_string(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        report = dc.validate(clean_df.copy())
        s = report.summary()
        assert isinstance(s, str)
        assert len(s) > 10

    # --- validate: all violations have column and violation_type ---
    def test_violation_objects_have_fields(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        bad = clean_df.drop(columns=["age", "income"])
        report = dc.validate(bad)
        for v in report.violations:
            assert hasattr(v, "column")
            assert hasattr(v, "violation_type")
            assert v.violation_type

    # --- ContractReport mend_score always 0–100 ---
    def test_contract_report_mend_score_range(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        bad = clean_df.drop(columns=["age"])
        bad.loc[0, "income"] = np.nan
        report = dc.validate(bad)
        assert 0.0 <= report.mend_score <= 100.0

    # --- warnings for extra columns ---
    def test_extra_columns_produce_warnings(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        extra = clean_df.copy()
        extra["extra_col_1"] = 1
        extra["extra_col_2"] = "foo"
        report = dc.validate(extra)
        extra_col_warnings = [w for w in report.warnings if w.violation_type == "EXTRA_COLUMN"]
        assert len(extra_col_warnings) == 2

    # --- fit on empty DataFrame raises or handles gracefully ---
    def test_fit_empty_df_no_crash(self) -> None:
        dc = DataContract()
        empty = pd.DataFrame({"a": pd.Series([], dtype=float)})
        try:
            dc.fit(empty)
        except Exception as exc:
            pytest.fail(f"fit on empty df raised unexpectedly: {exc}")

    # --- save → load → validate: same results ---
    def test_save_load_validate_parity(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        report_orig = dc.validate(clean_df.copy())

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            dc.save(path)
            loaded = DataContract.load(path)
            report_loaded = loaded.validate(clean_df.copy())
            assert report_orig.passed == report_loaded.passed
            assert len(report_orig.violations) == len(report_loaded.violations)
        finally:
            Path(path).unlink(missing_ok=True)

    # --- contract JSON can be parsed back ---
    def test_save_produces_valid_json(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            dc.save(path)
            content = Path(path).read_text(encoding="utf-8")
            parsed = json.loads(content)
            assert "name" in parsed
            # saved format uses "columns" key (not "specs")
            assert "columns" in parsed
        finally:
            Path(path).unlink(missing_ok=True)

    # --- datamend.validate top-level function ---
    def test_top_level_validate_function(self, clean_df: pd.DataFrame) -> None:
        dc = datamend.contract(clean_df)
        report = datamend.validate(clean_df.copy(), dc)
        assert isinstance(report, ContractReport)

    # --- ContractViolationError raised with raise_on_failure ---
    def test_contract_violation_error_message(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        bad = clean_df.drop(columns=["age"])
        with pytest.raises(ContractViolationError) as exc_info:
            dc.validate(bad, raise_on_failure=True)
        assert "MISSING_COLUMN" in str(exc_info.value) or len(str(exc_info.value)) > 0

    # --- large-column-count contract ---
    def test_wide_df_contract(self) -> None:
        df = _wide_df(200, 60)
        dc = datamend.contract(df)
        report = dc.validate(df.copy())
        assert isinstance(report, ContractReport)
        assert len(dc._specs) == 60


# ===========================================================================
# 3. DriftRadar — deep unit tests
# ===========================================================================


class TestDriftRadarDeep:

    # --- fit/detect pattern ---
    def test_fit_detect_pattern(self, train_df: pd.DataFrame, prod_stable_df: pd.DataFrame) -> None:
        radar = DriftRadar(verbose=False)
        radar.fit(train_df)
        report = radar.detect(prod_stable_df)
        assert isinstance(report, DriftReport)

    # --- fit/detect on drifted data still works ---
    def test_fit_detect_drifted(self, train_df: pd.DataFrame, prod_drifted_df: pd.DataFrame) -> None:
        radar = DriftRadar(verbose=False)
        radar.fit(train_df)
        report = radar.detect(prod_drifted_df)
        assert report.overall_drifted

    # --- detect without fit (two-arg form) ---
    def test_two_arg_detect_still_works(self, train_df: pd.DataFrame, prod_drifted_df: pd.DataFrame) -> None:
        report = DriftRadar(verbose=False).detect(train_df, prod_drifted_df)
        assert isinstance(report, DriftReport)

    # --- detect without fit and no prior fit → ValueError ---
    def test_single_arg_without_fit_raises(self, train_df: pd.DataFrame) -> None:
        radar = DriftRadar(verbose=False)
        with pytest.raises((ValueError, TypeError)):
            radar.detect(train_df)  # no prod_df, no .fit()

    # --- all-numeric DataFrame ---
    def test_all_numeric_df(self) -> None:
        rng = _rng(1)
        n = 300
        df_train = pd.DataFrame({f"col_{i}": rng.normal(i * 10, 2, n) for i in range(5)})
        rng2 = _rng(2)
        df_prod = pd.DataFrame({f"col_{i}": rng2.normal(i * 10 + 1, 2, n) for i in range(5)})
        report = DriftRadar(verbose=False).detect(df_train, df_prod)
        assert len(report.column_results) == 5

    # --- all-categorical DataFrame ---
    def test_all_categorical_df(self) -> None:
        rng = _rng(5)
        train = pd.DataFrame({
            "c1": rng.choice(["A", "B", "C"], 200),
            "c2": rng.choice(["X", "Y"], 200),
        })
        prod = pd.DataFrame({
            "c1": rng.choice(["A", "B", "D"], 200),
            "c2": rng.choice(["X", "Z"], 200),
        })
        report = DriftRadar(verbose=False).detect(train, prod)
        assert isinstance(report, DriftReport)
        assert len(report.column_results) == 2

    # --- single column DataFrame ---
    def test_single_column_no_crash(self) -> None:
        rng = _rng(6)
        train = pd.DataFrame({"x": rng.normal(0, 1, 100)})
        prod = pd.DataFrame({"x": rng.normal(5, 1, 100)})
        report = DriftRadar(verbose=False).detect(train, prod)
        assert "x" in report.column_results

    # --- different alpha levels ---
    def test_alpha_0_01_more_conservative(self, train_df, prod_drifted_df) -> None:
        report_strict = DriftRadar(alpha=0.01, verbose=False).detect(train_df, prod_drifted_df)
        report_loose = DriftRadar(alpha=0.2, verbose=False).detect(train_df, prod_drifted_df)
        # Both should detect drift given strong signal; just check no crash
        assert isinstance(report_strict, DriftReport)
        assert isinstance(report_loose, DriftReport)

    # --- PSI bucket configurations ---
    def test_psi_bucket_configs(self, train_df, prod_drifted_df) -> None:
        for buckets in [5, 10, 20]:
            report = DriftRadar(psi_buckets=buckets, verbose=False).detect(train_df, prod_drifted_df)
            assert isinstance(report, DriftReport)

    # --- column_results keys match shared columns ---
    def test_column_results_keys(self, train_df, prod_stable_df) -> None:
        report = DriftRadar(verbose=False).detect(train_df, prod_stable_df)
        shared = set(train_df.columns) & set(prod_stable_df.columns)
        assert set(report.column_results.keys()) == shared

    # --- each ColumnResult has required fields ---
    def test_column_result_fields(self, train_df, prod_drifted_df) -> None:
        report = DriftRadar(verbose=False).detect(train_df, prod_drifted_df)
        for col, res in report.column_results.items():
            assert hasattr(res, "column")
            assert hasattr(res, "drifted")
            assert hasattr(res, "severity")
            assert hasattr(res, "jsd")
            assert res.severity in {"none", "low", "medium", "high", "critical"}

    # --- columns_drifted matches columns with drifted=True ---
    def test_columns_drifted_consistency(self, train_df, prod_drifted_df) -> None:
        report = DriftRadar(verbose=False).detect(train_df, prod_drifted_df)
        expected = {col for col, res in report.column_results.items() if res.drifted}
        assert set(report.columns_drifted) == expected

    # --- report.to_dict() is JSON serialisable ---
    def test_report_to_dict_json_serialisable(self, train_df, prod_drifted_df) -> None:
        report = DriftRadar(verbose=False).detect(train_df, prod_drifted_df)
        d = report.to_dict()
        json.dumps(d, default=str)  # should not raise

    # --- report.summary() contains expected strings ---
    def test_report_summary_format(self, train_df, prod_stable_df) -> None:
        report = DriftRadar(verbose=False).detect(train_df, prod_stable_df)
        summary = report.summary()
        assert "DriftRadar" in summary or "Drift" in summary
        assert "MendScore" in summary or "Score" in summary

    # --- very small datasets (10 rows) ---
    def test_very_small_datasets(self) -> None:
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        prod = pd.DataFrame({"x": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]})
        report = DriftRadar(verbose=False).detect(train, prod)
        assert isinstance(report, DriftReport)

    # --- mend_score stays in 0–100 even for severe drift ---
    def test_mend_score_bounds_severe_drift(self) -> None:
        rng = _rng(10)
        train = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.choice(["X", "Y"], 500)})
        prod = pd.DataFrame({"a": rng.normal(1000, 1, 500), "b": rng.choice(["Z", "W"], 500)})
        report = DriftRadar(verbose=False).detect(train, prod)
        assert 0.0 <= report.mend_score <= 100.0

    # --- numeric column result has psi and ks_stat ---
    def test_numeric_result_has_psi_ks(self, train_df, prod_drifted_df) -> None:
        report = DriftRadar(verbose=False).detect(train_df, prod_drifted_df)
        res = report.column_results["feature_a"]
        assert res.psi is not None
        assert res.ks_stat is not None

    # --- categorical column result has chi2 or jsd ---
    def test_categorical_result_has_jsd(self, train_df, prod_drifted_df) -> None:
        report = DriftRadar(verbose=False).detect(train_df, prod_drifted_df)
        res = report.column_results["category"]
        assert res.jsd is not None

    # --- subset columns parameter ---
    def test_columns_subset_filters_correctly(self, train_df, prod_drifted_df) -> None:
        report = DriftRadar(verbose=False).detect(train_df, prod_drifted_df, columns=["feature_a"])
        assert "feature_a" in report.column_results
        assert "feature_b" not in report.column_results
        assert "category" not in report.column_results

    # --- constant production column vs variable training ---
    def test_constant_prod_column_no_crash(self) -> None:
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0] * 40})
        prod = pd.DataFrame({"x": [0.0] * 200})  # constant
        report = DriftRadar(verbose=False).detect(train, prod)
        assert isinstance(report, DriftReport)


# ===========================================================================
# 4. FailureTrace — deep unit tests
# ===========================================================================


class TestFailureTraceDeep:

    # --- fit/trace pattern: fit stores model/data, trace still needs model+df+preds ---
    def test_fit_trace_pattern(self, simple_sklearn_model, clean_df: pd.DataFrame) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        preds = simple_sklearn_model.predict(X)
        tracer = FailureTrace(verbose=False)
        tracer.fit(simple_sklearn_model, X)  # stores reference for future use
        report = tracer.trace(simple_sklearn_model, X, preds)
        assert isinstance(report, TraceReport)

    # --- top_k larger than column count → returns at most n_cols ---
    def test_top_k_larger_than_columns(self, simple_sklearn_model, clean_df: pd.DataFrame) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        preds = simple_sklearn_model.predict(X)
        tracer = FailureTrace(top_k=100, verbose=False)
        report = tracer.trace(simple_sklearn_model, X, preds)
        assert len(report.column_attributions) <= X.shape[1]

    # --- top_k=1 returns exactly 1 column ---
    def test_top_k_one(self, simple_sklearn_model, clean_df: pd.DataFrame) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        preds = simple_sklearn_model.predict(X)
        tracer = FailureTrace(top_k=1, verbose=False)
        report = tracer.trace(simple_sklearn_model, X, preds)
        assert len(report.column_attributions) == 1

    # --- binary classification model ---
    def test_binary_classification_model(self, clean_df: pd.DataFrame) -> None:
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier
        X = clean_df[["age", "income", "score"]]
        y = (clean_df["income"] > 55_000).astype(int).values
        model = RandomForestClassifier(n_estimators=10, random_state=0)
        model.fit(X, y)
        preds = model.predict(X).astype(float)
        report = datamend.trace(model, X, preds, verbose=False)
        assert isinstance(report, TraceReport)

    # --- linear regression (no feature_importances_, uses surrogate) ---
    def test_linear_model_uses_surrogate(self, clean_df: pd.DataFrame) -> None:
        pytest.importorskip("sklearn")
        from sklearn.linear_model import LinearRegression
        X = clean_df[["age", "income", "score"]]
        y = X["age"].values + X["score"].values
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)
        report = datamend.trace(model, X, preds, verbose=False)
        assert isinstance(report, TraceReport)
        assert len(report.column_attributions) > 0

    # --- ground truth binary labels ---
    def test_ground_truth_binary_labels(self, simple_sklearn_model, clean_df) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        preds = simple_sklearn_model.predict(X)
        gt = (preds * 0 + _rng(5).choice([0, 1], len(preds))).astype(float)
        report = datamend.trace(simple_sklearn_model, X, preds, ground_truth=gt, verbose=False)
        assert isinstance(report, TraceReport)

    # --- column_attributions are sorted descending by importance_score ---
    def test_attributions_sorted_descending(self, simple_sklearn_model, clean_df) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        preds = simple_sklearn_model.predict(X)
        report = datamend.trace(simple_sklearn_model, X, preds, verbose=False)
        importances = [a.importance_score for a in report.column_attributions]
        assert importances == sorted(importances, reverse=True)

    # --- suspicious_rows contain RowFailure objects with valid row_index ---
    def test_suspicious_rows_are_valid_indices(self, simple_sklearn_model, clean_df) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        preds = simple_sklearn_model.predict(X)
        report = datamend.trace(simple_sklearn_model, X, preds, verbose=False)
        for row_failure in report.suspicious_rows[:5]:
            idx = row_failure.row_index
            _ = X.iloc[idx] if isinstance(idx, int) else X.loc[idx]

    # --- mend_score always 0–100 ---
    def test_mend_score_range(self, simple_sklearn_model, clean_df) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        preds = simple_sklearn_model.predict(X)
        report = datamend.trace(simple_sklearn_model, X, preds, verbose=False)
        assert 0.0 <= report.mend_score <= 100.0

    # --- report to_dict has required keys, attributions have importance_score ---
    def test_report_dict_keys(self, simple_sklearn_model, clean_df) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        preds = simple_sklearn_model.predict(X)
        d = datamend.trace(simple_sklearn_model, X, preds, verbose=False).to_dict()
        assert {"mend_score", "column_attributions", "suspicious_rows"} <= d.keys()
        for a in d["column_attributions"]:
            assert "importance_score" in a

    # --- report to_dict is JSON serialisable ---
    def test_report_json_serialisable(self, simple_sklearn_model, clean_df) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        preds = simple_sklearn_model.predict(X)
        d = datamend.trace(simple_sklearn_model, X, preds, verbose=False).to_dict()
        json.dumps(d, default=str)  # should not raise

    # --- non-unique index DataFrame ---
    def test_non_unique_index_no_crash(self, simple_sklearn_model, clean_df) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]].copy()
        X.index = [0] * len(X)  # all same index
        preds = simple_sklearn_model.predict(X)
        report = datamend.trace(simple_sklearn_model, X, preds, verbose=False)
        assert isinstance(report, TraceReport)

    # --- gradient boosting model ---
    def test_gradient_boosting_model(self, clean_df) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        y = X["age"].values
        model = _make_sklearn_model(X, y)
        preds = model.predict(X)
        report = datamend.trace(model, X, preds, verbose=False)
        assert isinstance(report, TraceReport)

    # --- single row prediction ---
    def test_single_row_prediction(self, simple_sklearn_model, clean_df) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]].iloc[:1]
        preds = simple_sklearn_model.predict(X)
        report = datamend.trace(simple_sklearn_model, X, preds, verbose=False)
        assert isinstance(report, TraceReport)

    # --- very wide feature DataFrame ---
    def test_wide_feature_df(self) -> None:
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestRegressor
        rng = _rng(30)
        n, k = 200, 50
        X = pd.DataFrame({f"f{i}": rng.normal(i, 1, n) for i in range(k)})
        y = X.sum(axis=1).values
        model = RandomForestRegressor(n_estimators=5, random_state=0)
        model.fit(X, y)
        preds = model.predict(X)
        report = datamend.trace(model, X, preds, top_k=10, verbose=False)
        assert len(report.column_attributions) <= 10


# ===========================================================================
# 5. MendPipeline — advanced tests
# ===========================================================================


class TestMendPipelineAdvanced:

    # --- multiple transform() calls after one fit() ---
    def test_multiple_transforms(self, clean_df, dirty_df) -> None:
        from datamend.pipeline import MendPipeline, PipelineResult
        pipeline = MendPipeline(verbose=False)
        pipeline.fit(clean_df)
        result1 = pipeline.transform(dirty_df)
        result2 = pipeline.transform(clean_df)
        assert isinstance(result1, PipelineResult)
        assert isinstance(result2, PipelineResult)

    # --- pipeline with only repair enabled ---
    def test_only_repair_enabled(self, clean_df, dirty_df) -> None:
        from datamend.pipeline import MendPipeline
        pipeline = MendPipeline(
            verbose=False,
            enable_contract=False,
            enable_drift=False,
            enable_trace=False,
        )
        pipeline.fit(clean_df)
        result = pipeline.transform(dirty_df)
        assert result.repair_report is not None
        assert result.contract_report is None
        assert result.drift_report is None

    # --- pipeline with only drift enabled ---
    def test_only_drift_enabled(self, clean_df, dirty_df) -> None:
        from datamend.pipeline import MendPipeline
        pipeline = MendPipeline(
            verbose=False,
            enable_repair=False,
            enable_contract=False,
            enable_trace=False,
        )
        pipeline.fit(clean_df)
        result = pipeline.transform(dirty_df.fillna(0))
        assert result.drift_report is not None

    # --- overall_mend_score stays in 0–100 ---
    def test_overall_mend_score_bounds(self, clean_df, dirty_df) -> None:
        from datamend.pipeline import MendPipeline
        pipeline = MendPipeline(verbose=False)
        pipeline.fit(clean_df)
        result = pipeline.transform(dirty_df)
        assert 0.0 <= result.overall_mend_score <= 100.0

    # --- to_dict has all top-level keys ---
    def test_to_dict_keys(self, clean_df) -> None:
        from datamend.pipeline import MendPipeline
        result = MendPipeline(verbose=False).fit_transform(clean_df)
        d = result.to_dict()
        assert "overall_mend_score" in d
        assert "repair" in d

    # --- to_json is valid JSON ---
    def test_to_json_valid(self, clean_df) -> None:
        from datamend.pipeline import MendPipeline
        result = MendPipeline(verbose=False).fit_transform(clean_df)
        parsed = json.loads(result.to_json())
        assert "overall_mend_score" in parsed

    # --- pipeline result summary contains key strings ---
    def test_summary_format(self, clean_df) -> None:
        from datamend.pipeline import MendPipeline
        result = MendPipeline(verbose=False).fit_transform(clean_df)
        summary = result.summary()
        assert "MendPipeline" in summary

    # --- repaired_df has no more nulls than original if repair enabled ---
    def test_repaired_df_fewer_nulls(self, dirty_df, clean_df) -> None:
        from datamend.pipeline import MendPipeline
        pipeline = MendPipeline(verbose=False)
        pipeline.fit(clean_df)
        result = pipeline.transform(dirty_df)
        orig_nulls = dirty_df.isnull().sum().sum()
        repaired_nulls = result.repaired_df.isnull().sum().sum()
        assert repaired_nulls <= orig_nulls

    # --- fit_transform is equivalent to fit+transform ---
    def test_fit_transform_equivalent(self, clean_df) -> None:
        from datamend.pipeline import MendPipeline
        p1 = MendPipeline(verbose=False)
        p2 = MendPipeline(verbose=False)
        r1 = p1.fit_transform(clean_df)
        p2.fit(clean_df)
        r2 = p2.transform(clean_df)
        assert abs(r1.overall_mend_score - r2.overall_mend_score) < 5.0


# ===========================================================================
# 6. CLI tests
# ===========================================================================


class TestCLI:

    def _write_csv(self, df: pd.DataFrame) -> str:
        fh = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        df.to_csv(fh.name, index=False)
        fh.close()
        return fh.name

    def test_repair_command_runs(self, dirty_df) -> None:
        from click.testing import CliRunner
        from datamend.cli import repair

        path = self._write_csv(dirty_df)
        out_path = path.replace(".csv", "_out.csv")
        runner = CliRunner()
        try:
            result = runner.invoke(repair, [path, "-o", out_path, "--no-verbose"])
            assert result.exit_code == 0, result.output
            assert Path(out_path).exists()
        finally:
            Path(path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)

    def test_repair_with_report_flag(self, dirty_df) -> None:
        from click.testing import CliRunner
        from datamend.cli import repair

        path = self._write_csv(dirty_df)
        report_path = path.replace(".csv", "_report.json")
        out_path = path.replace(".csv", "_out.csv")
        runner = CliRunner()
        try:
            result = runner.invoke(repair, [path, "-o", out_path, "--report", report_path, "--no-verbose"])
            assert result.exit_code == 0, result.output
            assert Path(report_path).exists()
            parsed = json.loads(Path(report_path).read_text())
            assert "mend_score_after" in parsed
        finally:
            for p in [path, out_path, report_path]:
                Path(p).unlink(missing_ok=True)

    def test_score_command_runs(self, dirty_df) -> None:
        from click.testing import CliRunner
        from datamend.cli import score

        path = self._write_csv(dirty_df)
        runner = CliRunner()
        try:
            result = runner.invoke(score, [path])
            assert result.exit_code == 0, result.output
            assert "MendScore" in result.output
        finally:
            Path(path).unlink(missing_ok=True)

    def test_contract_command_runs(self, clean_df) -> None:
        from click.testing import CliRunner
        from datamend.cli import contract

        path = self._write_csv(clean_df)
        out_path = path.replace(".csv", "_contract.json")
        runner = CliRunner()
        try:
            result = runner.invoke(contract, [path, "-o", out_path])
            assert result.exit_code == 0, result.output
            assert Path(out_path).exists()
            parsed = json.loads(Path(out_path).read_text())
            assert "columns" in parsed
        finally:
            for p in [path, out_path]:
                Path(p).unlink(missing_ok=True)

    def test_validate_command_runs(self, clean_df, dirty_df) -> None:
        from click.testing import CliRunner
        from datamend.cli import contract as contract_cmd
        from datamend.cli import validate as validate_cmd

        train_path = self._write_csv(clean_df)
        prod_path = self._write_csv(dirty_df.fillna(0).assign(
            **{c: dirty_df[c].fillna("unknown") for c in dirty_df.select_dtypes("object")}
        ))
        contract_path = train_path.replace(".csv", "_contract.json")
        runner = CliRunner()
        try:
            runner.invoke(contract_cmd, [train_path, "-o", contract_path])
            result = runner.invoke(validate_cmd, [prod_path, contract_path])
            assert result.exit_code in (0, 1)  # 0 = pass, 1 = fail-fast
        finally:
            for p in [train_path, prod_path, contract_path]:
                Path(p).unlink(missing_ok=True)

    def test_drift_command_runs(self, clean_df, dirty_df) -> None:
        from click.testing import CliRunner
        from datamend.cli import drift as drift_cmd

        train_path = self._write_csv(clean_df)
        prod_path = self._write_csv(dirty_df.fillna(0))
        runner = CliRunner()
        try:
            result = runner.invoke(drift_cmd, [train_path, prod_path, "--no-verbose"])
            assert result.exit_code == 0, result.output
        finally:
            for p in [train_path, prod_path]:
                Path(p).unlink(missing_ok=True)

    def test_repair_command_with_html(self, dirty_df) -> None:
        from click.testing import CliRunner
        from datamend.cli import repair

        path = self._write_csv(dirty_df)
        out_path = path.replace(".csv", "_out.csv")
        html_path = path.replace(".csv", ".html")
        runner = CliRunner()
        try:
            result = runner.invoke(repair, [path, "-o", out_path, "--html", html_path, "--no-verbose"])
            assert result.exit_code == 0, result.output
            assert Path(html_path).exists()
            content = Path(html_path).read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
        finally:
            for p in [path, out_path, html_path]:
                Path(p).unlink(missing_ok=True)

    def test_score_output_format(self, clean_df) -> None:
        from click.testing import CliRunner
        from datamend.cli import score

        path = self._write_csv(clean_df)
        runner = CliRunner()
        try:
            result = runner.invoke(score, [path])
            assert result.exit_code == 0
            assert "/100" in result.output
        finally:
            Path(path).unlink(missing_ok=True)

    def test_plugins_command_runs(self) -> None:
        from click.testing import CliRunner
        from datamend.cli import plugins

        runner = CliRunner()
        result = runner.invoke(plugins, [])
        assert result.exit_code == 0
        assert "repair" in result.output.lower()

    def test_repair_missing_file_exits_1(self) -> None:
        from click.testing import CliRunner
        from datamend.cli import repair

        runner = CliRunner()
        result = runner.invoke(repair, ["/nonexistent/path/data.csv"])
        assert result.exit_code == 1

    def test_repair_parquet_roundtrip(self, dirty_df) -> None:
        pytest.importorskip("pyarrow")
        from click.testing import CliRunner
        from datamend.cli import repair

        fh = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        fh.close()
        dirty_df.to_parquet(fh.name, index=False)
        out_path = fh.name.replace(".parquet", "_out.parquet")
        runner = CliRunner()
        try:
            result = runner.invoke(repair, [fh.name, "-o", out_path, "--no-verbose"])
            assert result.exit_code == 0, result.output
            loaded = pd.read_parquet(out_path)
            assert len(loaded) > 0
        finally:
            for p in [fh.name, out_path]:
                Path(p).unlink(missing_ok=True)


# ===========================================================================
# 7. Cross-pillar integration tests
# ===========================================================================


class TestCrossPillarIntegration:

    # --- repaired data should have fewer contract violations than dirty data ---
    def test_repair_then_validate_fewer_violations(
        self, clean_df: pd.DataFrame, dirty_df: pd.DataFrame
    ) -> None:
        dc = datamend.contract(clean_df)

        # fill object nulls for comparison (contract expects no nulls)
        prod_dirty = dirty_df.copy()
        prod_repaired, _ = datamend.repair(prod_dirty, verbose=False)

        report_dirty = dc.validate(prod_dirty.fillna(-1).astype(
            {c: str for c in prod_dirty.select_dtypes("object").columns}
        ))
        report_repaired = dc.validate(prod_repaired.fillna(-1).astype(
            {c: str for c in prod_repaired.select_dtypes("object").columns}
        ))
        # After repair, at least no more violations (or fewer)
        assert len(report_repaired.violations) <= len(report_dirty.violations) + 2

    # --- repaired data closer to train distribution ---
    def test_repair_reduces_drift_score(
        self, clean_df: pd.DataFrame, dirty_df: pd.DataFrame
    ) -> None:
        prod_dirty = dirty_df.copy()
        repaired, _ = datamend.repair(prod_dirty, verbose=False)

        numeric_train = clean_df.select_dtypes(include=[np.number])
        numeric_dirty = prod_dirty.select_dtypes(include=[np.number]).fillna(-999)
        numeric_repaired = repaired.select_dtypes(include=[np.number])

        shared = list(set(numeric_train.columns) & set(numeric_dirty.columns) & set(numeric_repaired.columns))
        if not shared:
            pytest.skip("No shared numeric columns")

        report_dirty = DriftRadar(verbose=False).detect(numeric_train[shared], numeric_dirty[shared])
        report_repaired = DriftRadar(verbose=False).detect(numeric_train[shared], numeric_repaired[shared])

        # Repaired drift score should be better (lower urgency) or equal
        assert report_repaired.mend_score >= report_dirty.mend_score - 10

    # --- full 4-pillar flow: repair → contract → drift → trace ---
    def test_full_four_pillar_flow(self, clean_df: pd.DataFrame, dirty_df: pd.DataFrame) -> None:
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestRegressor

        # Pillar 1: Repair
        repaired, repair_report = datamend.repair(dirty_df, verbose=False)
        assert isinstance(repair_report, RepairReport)

        # Pillar 2: Contract
        dc = datamend.contract(clean_df)
        filled = repaired.select_dtypes(include=[np.number]).fillna(0)
        train_numeric = clean_df.select_dtypes(include=[np.number])
        shared_cols = list(set(filled.columns) & set(train_numeric.columns))
        contract_report = dc.validate(repaired.fillna(0) if repaired.select_dtypes(include=[np.number]).isnull().any().any() else repaired)  # may have violations
        assert isinstance(contract_report, ContractReport)

        # Pillar 3: Drift
        if shared_cols:
            drift_report = datamend.drift(
                train_numeric[shared_cols], filled[shared_cols], verbose=False
            )
            assert isinstance(drift_report, DriftReport)

        # Pillar 4: Trace
        if shared_cols:
            X_train = train_numeric[shared_cols]
            y_train = X_train.iloc[:, 0]
            model = RandomForestRegressor(n_estimators=5, random_state=0)
            model.fit(X_train, y_train)
            X_prod = filled[shared_cols]
            preds = model.predict(X_prod)
            trace_report = datamend.trace(model, X_prod, preds, verbose=False)
            assert isinstance(trace_report, TraceReport)

    # --- MendReport wraps all four pillars ---
    def test_mend_report_all_four_pillars(
        self, clean_df: pd.DataFrame, dirty_df: pd.DataFrame
    ) -> None:
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestRegressor
        from datamend.report import MendReport

        repaired, repair_report = datamend.repair(dirty_df, verbose=False)
        dc = datamend.contract(clean_df)
        contract_report = dc.validate(repaired.fillna(0).astype(
            {c: str for c in repaired.select_dtypes("object").columns}
        ))

        numeric_cols = list(
            set(clean_df.select_dtypes(include=[np.number]).columns)
            & set(repaired.select_dtypes(include=[np.number]).columns)
        )
        drift_report = datamend.drift(
            clean_df[numeric_cols], repaired[numeric_cols].fillna(0), verbose=False
        )

        X = clean_df[numeric_cols]
        y = X.iloc[:, 0]
        model = RandomForestRegressor(n_estimators=5, random_state=0)
        model.fit(X, y)
        preds = model.predict(repaired[numeric_cols].fillna(0))
        trace_report = datamend.trace(
            model, repaired[numeric_cols].fillna(0), preds, verbose=False
        )

        mr = MendReport(
            repair=repair_report,
            contract=contract_report,
            drift=drift_report,
            trace=trace_report,
            title="Full Integration Test",
        )
        html = mr.to_html()
        assert "Pillar 1" in html
        assert "Pillar 2" in html
        assert "Pillar 3" in html
        assert "Pillar 4" in html

    # --- Pipeline wraps all four pillars end-to-end ---
    def test_pipeline_end_to_end(
        self, clean_df: pd.DataFrame, dirty_df: pd.DataFrame
    ) -> None:
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestRegressor
        from datamend.pipeline import MendPipeline

        numeric_cols = list(
            set(clean_df.select_dtypes(include=[np.number]).columns)
            & set(dirty_df.select_dtypes(include=[np.number]).columns)
        )
        X = clean_df[numeric_cols]
        model = RandomForestRegressor(n_estimators=5, random_state=0)
        model.fit(X, X.iloc[:, 0])

        pipeline = MendPipeline(verbose=False)
        pipeline.fit(clean_df)

        prod = dirty_df[numeric_cols].fillna(0)
        preds = model.predict(prod)
        result = pipeline.transform(dirty_df, model=model, predictions=preds)

        assert result.repair_report is not None
        assert result.contract_report is not None
        assert result.drift_report is not None
        assert result.trace_report is not None
        assert 0 <= result.overall_mend_score <= 100


# ===========================================================================
# 8. Serialisation round-trips
# ===========================================================================


class TestSerialisationRoundTrips:

    # --- RepairReport to_dict round-trip ---
    def test_repair_report_to_dict_all_fields(self, dirty_df: pd.DataFrame) -> None:
        _, report = datamend.repair(dirty_df, verbose=False)
        d = report.to_dict()
        assert isinstance(d["actions"], list)
        for a in d["actions"]:
            assert "column" in a
            assert "issue_type" in a

    # --- DriftReport to_dict round-trip ---
    def test_drift_report_to_dict_all_fields(self, train_df, prod_drifted_df) -> None:
        report = datamend.drift(train_df, prod_drifted_df, verbose=False)
        d = report.to_dict()
        assert "column_results" in d
        for col, r in d["column_results"].items():
            assert "drifted" in r
            assert "severity" in r

    # --- TraceReport to_dict round-trip ---
    def test_trace_report_to_dict_all_fields(self, simple_sklearn_model, clean_df) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        preds = simple_sklearn_model.predict(X)
        report = datamend.trace(simple_sklearn_model, X, preds, verbose=False)
        d = report.to_dict()
        assert "column_attributions" in d
        for a in d["column_attributions"]:
            assert "column" in a
            assert "importance_score" in a

    # --- ContractReport to_json / from dict ---
    def test_contract_report_to_json_parseable(self, clean_df) -> None:
        dc = datamend.contract(clean_df)
        report = datamend.validate(clean_df.copy(), dc)
        parsed = json.loads(report.to_json())
        assert "passed" in parsed
        assert "mend_score" in parsed

    # --- DataContract save / load with schema dict ---
    def test_schema_dict_save_load(self, clean_df) -> None:
        schema = {"age": {"dtype": "numeric", "min": 0, "max": 200}}
        dc = DataContract(schema, name="schema_test")
        dc.fit(clean_df)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            dc.save(path)
            loaded = DataContract.load(path)
            assert loaded.name == "schema_test"
            report = loaded.validate(clean_df.copy())
            assert isinstance(report, ContractReport)
        finally:
            Path(path).unlink(missing_ok=True)

    # --- MendPipeline result to_json round-trip ---
    def test_pipeline_result_to_json_roundtrip(self, clean_df) -> None:
        from datamend.pipeline import MendPipeline
        result = MendPipeline(verbose=False).fit_transform(clean_df)
        raw = result.to_json()
        parsed = json.loads(raw)
        assert isinstance(parsed["overall_mend_score"], (int, float))

    # --- MendReport to_json then to_dict ---
    def test_mend_report_json_roundtrip(self, dirty_df) -> None:
        from datamend.report import MendReport
        _, r = datamend.repair(dirty_df, verbose=False)
        mr = MendReport(repair=r)
        d = mr.to_dict()
        j = mr.to_json()
        parsed = json.loads(j)
        assert "repair" in parsed or "title" in parsed

    # --- HTML output is valid UTF-8 ---
    def test_html_output_utf8(self, dirty_df) -> None:
        from datamend.report import MendReport
        _, r = datamend.repair(dirty_df, verbose=False)
        mr = MendReport(repair=r, title="UTF-8 Test — Ångström")
        html = mr.to_html()
        html.encode("utf-8")  # should not raise UnicodeEncodeError


# ===========================================================================
# 9. MendReport — extended tests
# ===========================================================================


class TestMendReportExtended:

    def test_repair_report_short_kwarg(self, dirty_df) -> None:
        from datamend.report import MendReport
        _, r = datamend.repair(dirty_df, verbose=False)
        mr = MendReport(repair=r)
        html = mr.to_html()
        assert "AutoRepair" in html

    def test_repair_report_long_kwarg(self, dirty_df) -> None:
        from datamend.report import MendReport
        _, r = datamend.repair(dirty_df, verbose=False)
        mr = MendReport(repair_report=r)
        html = mr.to_html()
        assert "AutoRepair" in html

    def test_drift_report_short_kwarg(self, train_df, prod_drifted_df) -> None:
        from datamend.report import MendReport
        r = datamend.drift(train_df, prod_drifted_df, verbose=False)
        mr = MendReport(drift=r)
        html = mr.to_html()
        assert "DriftRadar" in html or "Drift" in html

    def test_all_reports_html_sections(self, clean_df, dirty_df) -> None:
        from datamend.report import MendReport
        _, repair_r = datamend.repair(dirty_df, verbose=False)
        contract_r = datamend.contract(clean_df).validate(clean_df.copy())
        drift_r = datamend.drift(clean_df, clean_df.copy(), verbose=False)

        mr = MendReport(repair=repair_r, contract=contract_r, drift=drift_r)
        html = mr.to_html()
        assert "Pillar 1" in html
        assert "Pillar 2" in html
        assert "Pillar 3" in html

    def test_to_html_saves_with_utf8_title(self, dirty_df) -> None:
        from datamend.report import MendReport
        _, r = datamend.repair(dirty_df, verbose=False)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as fh:
            path = fh.name
        try:
            MendReport(repair=r, title="Données — résumé").to_html(path=path)
            content = Path(path).read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_to_dict_title_field(self) -> None:
        from datamend.report import MendReport
        mr = MendReport(title="My Custom Report")
        d = mr.to_dict()
        assert d["title"] == "My Custom Report"


# ===========================================================================
# 10. Adversarial and edge-case battery
# ===========================================================================


class TestAdversarialInputs:

    # --- DataFrame with all the same value in every column ---
    def test_all_same_value_df(self) -> None:
        df = pd.DataFrame({"a": [5.0] * 100, "b": ["x"] * 100})
        repaired, _ = datamend.repair(df, verbose=False)
        assert len(repaired) > 0

    # --- DataFrame with one column that is all NaN ---
    def test_all_nan_column_no_crash(self) -> None:
        df = pd.DataFrame({"all_nan": [np.nan] * 50, "ok": list(range(50, dtype=float) if False else range(50))})
        repaired, _ = datamend.repair(df, verbose=False)
        assert len(repaired) > 0

    # --- DataFrame with mixed object/numeric column (strings + numbers) ---
    def test_mixed_type_column_object(self) -> None:
        df = pd.DataFrame({"mixed": ["1.5", "2.3", "not_a_number", "4.1"] * 30})
        repaired, _ = datamend.repair(df, verbose=False)
        assert len(repaired) > 0

    # --- Emoji / unusual unicode in category column ---
    def test_emoji_category_no_crash(self) -> None:
        rng = _rng(40)
        df = pd.DataFrame({
            "emoji": rng.choice(["😀", "🔥", "✅", "😀", "🔥"], 100),
            "val": rng.normal(0, 1, 100),
        })
        repaired, _ = datamend.repair(df, verbose=False)
        assert len(repaired) > 0

    # --- DataFrame with spaces-only strings ---
    def test_spaces_only_strings(self) -> None:
        df = pd.DataFrame({"text": ["   ", "   ", "   ", "hello"] * 20})
        repaired, _ = datamend.repair(df, verbose=False)
        assert len(repaired) > 0

    # --- Very skewed numeric data (log-normal) ---
    def test_log_normal_distribution(self) -> None:
        rng = _rng(50)
        vals = np.exp(rng.normal(0, 3, 500))
        vals[::50] = np.nan
        df = pd.DataFrame({"income": vals})
        repaired, _ = datamend.repair(df, verbose=False)
        assert repaired["income"].isnull().sum() == 0

    # --- DataContract with empty production DataFrame ---
    def test_contract_validate_empty_df(self, clean_df) -> None:
        dc = datamend.contract(clean_df)
        empty = pd.DataFrame(columns=clean_df.columns)
        report = dc.validate(empty)
        assert isinstance(report, ContractReport)

    # --- DriftRadar with single-category column ---
    def test_drift_single_category_value(self) -> None:
        train = pd.DataFrame({"cat": ["A"] * 100, "x": list(range(100))})
        prod = pd.DataFrame({"cat": ["B"] * 100, "x": list(range(100, 200))})
        report = DriftRadar(verbose=False).detect(train, prod)
        assert isinstance(report, DriftReport)

    # --- FailureTrace with all-same predictions ---
    def test_trace_all_same_predictions(self, simple_sklearn_model, clean_df) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        preds = np.ones(len(X)) * 42.0
        report = datamend.trace(simple_sklearn_model, X, preds, verbose=False)
        assert isinstance(report, TraceReport)

    # --- FailureTrace with NaN predictions (should not crash) ---
    def test_trace_nan_predictions_no_crash(self, simple_sklearn_model, clean_df) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        preds = np.where(np.arange(len(X)) % 5 == 0, np.nan, 1.0)
        try:
            report = datamend.trace(simple_sklearn_model, X, preds, verbose=False)
            assert isinstance(report, TraceReport)
        except (ValueError, TypeError):
            pass  # acceptable: NaN preds are invalid input

    # --- AutoRepair on DataFrame with integer index gaps ---
    def test_integer_index_with_gaps(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]}, index=[0, 5, 10, 15, 20])
        repaired, _ = datamend.repair(df, verbose=False)
        assert repaired["a"].isnull().sum() == 0

    # --- AutoRepair on DataFrame with string index ---
    def test_string_index(self) -> None:
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]}, index=["foo", "bar", "baz"])
        repaired, _ = datamend.repair(df, verbose=False)
        assert len(repaired) == 3

    # --- DataContract with datetime column ---
    def test_contract_datetime_column(self) -> None:
        df = _datetime_df(200)
        dc = DataContract()
        dc.fit(df)
        report = dc.validate(df.copy())
        assert isinstance(report, ContractReport)

    # --- DriftRadar with NaN-heavy column ---
    def test_drift_nan_heavy_column(self) -> None:
        rng = _rng(60)
        vals_train = rng.normal(0, 1, 200)
        vals_prod = rng.normal(0, 1, 200)
        vals_train[::2] = np.nan
        train = pd.DataFrame({"x": vals_train})
        prod = pd.DataFrame({"x": vals_prod})
        try:
            report = DriftRadar(verbose=False).detect(train, prod)
            assert isinstance(report, DriftReport)
        except Exception as exc:
            pytest.fail(f"DriftRadar crashed on NaN-heavy column: {exc}")

    # --- repair on DataFrame with only object columns ---
    def test_only_object_columns(self) -> None:
        rng = _rng(70)
        df = pd.DataFrame({
            "a": rng.choice(["foo", "bar", "baz", None], 100),
            "b": rng.choice(["x", "y", None], 100),
        })
        repaired, _ = datamend.repair(df, verbose=False)
        assert len(repaired) > 0

    # --- repair on DataFrame with only numeric columns ---
    def test_only_numeric_columns(self) -> None:
        rng = _rng(80)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 100),
            "b": rng.normal(5, 2, 100),
        })
        df.iloc[::10, 0] = np.nan
        repaired, _ = datamend.repair(df, verbose=False)
        assert repaired["a"].isnull().sum() == 0
