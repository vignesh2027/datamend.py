"""Tests for the DataContract engine."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import datamend
from datamend.core.contract import (
    ContractReport,
    ContractViolationError,
    DataContract,
    _dtypes_compatible,
)


# ---------------------------------------------------------------------------
# _dtypes_compatible
# ---------------------------------------------------------------------------


def test_dtypes_compatible_same() -> None:
    assert _dtypes_compatible("float64", "float64")


def test_dtypes_compatible_numeric_subtypes() -> None:
    assert _dtypes_compatible("int32", "int64")
    assert _dtypes_compatible("float32", "float64")


def test_dtypes_compatible_numeric_vs_string() -> None:
    assert not _dtypes_compatible("float64", "object")


def test_dtypes_compatible_datetime() -> None:
    assert _dtypes_compatible("datetime64[ns]", "datetime64[ns]")


# ---------------------------------------------------------------------------
# DataContract.fit
# ---------------------------------------------------------------------------


class TestDataContractFit:
    def test_fit_sets_specs(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        assert dc._fitted
        assert set(dc._specs.keys()) == set(clean_df.columns)

    def test_fit_numeric_specs(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        spec = dc._specs["age"]
        assert spec.min_val is not None
        assert spec.max_val is not None
        assert spec.mean_val is not None

    def test_fit_categorical_specs(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        spec = dc._specs["gender"]
        assert spec.allowed_values is not None
        assert len(spec.allowed_values) <= 50

    def test_fit_invalid_input_raises(self) -> None:
        dc = DataContract()
        with pytest.raises(TypeError):
            dc.fit("not_a_dataframe")  # type: ignore

    def test_datamend_contract_function(self, clean_df: pd.DataFrame) -> None:
        dc = datamend.contract(clean_df)
        assert dc._fitted
        assert isinstance(dc, DataContract)


# ---------------------------------------------------------------------------
# DataContract.validate
# ---------------------------------------------------------------------------


class TestDataContractValidate:
    def test_validate_clean_df_passes(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        report = dc.validate(clean_df.copy())
        assert isinstance(report, ContractReport)

    def test_validate_missing_column(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        bad_df = clean_df.drop(columns=["age"])
        report = dc.validate(bad_df)
        assert not report.passed
        assert any(v.violation_type == "MISSING_COLUMN" for v in report.violations)

    def test_validate_high_null_rate(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract(null_threshold=0.01)
        dc.fit(clean_df)
        null_df = clean_df.copy()
        null_df.loc[:50, "age"] = np.nan
        report = dc.validate(null_df)
        assert any(v.violation_type == "NULL_RATE" for v in report.violations)

    def test_validate_new_categories(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        new_df = clean_df.copy()
        new_df.loc[0, "gender"] = "ALIEN"
        report = dc.validate(new_df)
        assert any(v.violation_type == "CARDINALITY_VIOLATION" for v in report.violations)

    def test_validate_unfitted_raises(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        with pytest.raises(RuntimeError, match="fitted"):
            dc.validate(clean_df)

    def test_validate_raise_on_failure(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        bad_df = clean_df.drop(columns=["age"])
        with pytest.raises(ContractViolationError):
            dc.validate(bad_df, raise_on_failure=True)

    def test_validate_extra_columns_warning(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        extra_df = clean_df.copy()
        extra_df["new_col"] = 1
        report = dc.validate(extra_df)
        assert any(w.violation_type == "EXTRA_COLUMN" for w in report.warnings)

    def test_datamend_validate_function(self, clean_df: pd.DataFrame) -> None:
        dc = datamend.contract(clean_df)
        report = datamend.validate(clean_df.copy(), dc)
        assert isinstance(report, ContractReport)

    def test_contract_report_to_dict(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        report = dc.validate(clean_df.copy())
        d = report.to_dict()
        assert "passed" in d
        assert "violations" in d
        assert "mend_score" in d

    def test_contract_report_to_json(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        report = dc.validate(clean_df.copy())
        j = report.to_json()
        parsed = json.loads(j)
        assert "passed" in parsed


# ---------------------------------------------------------------------------
# DataContract persistence
# ---------------------------------------------------------------------------


class TestDataContractPersistence:
    def test_save_and_load(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract(name="test_contract")
        dc.fit(clean_df)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            dc.save(path)
            loaded = DataContract.load(path)
            assert loaded.name == "test_contract"
            assert set(loaded._specs.keys()) == set(dc._specs.keys())
            assert loaded._fitted
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_unfitted_raises(self) -> None:
        dc = DataContract()
        with pytest.raises(RuntimeError):
            dc.save("/tmp/test.json")

    def test_loaded_contract_validates(self, clean_df: pd.DataFrame) -> None:
        dc = DataContract()
        dc.fit(clean_df)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            dc.save(path)
            loaded = DataContract.load(path)
            report = loaded.validate(clean_df.copy())
            assert isinstance(report, ContractReport)
        finally:
            Path(path).unlink(missing_ok=True)
