"""Tests for MendPipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datamend.pipeline import MendPipeline, PipelineResult


class TestMendPipeline:
    def test_fit_and_transform(
        self, clean_df: pd.DataFrame, dirty_df: pd.DataFrame
    ) -> None:
        pipeline = MendPipeline(verbose=False)
        pipeline.fit(clean_df)
        result = pipeline.transform(dirty_df)
        assert isinstance(result, PipelineResult)

    def test_fit_transform_single_call(self, clean_df: pd.DataFrame) -> None:
        pipeline = MendPipeline(verbose=False)
        result = pipeline.fit_transform(clean_df)
        assert isinstance(result, PipelineResult)

    def test_result_has_repaired_df(
        self, clean_df: pd.DataFrame, dirty_df: pd.DataFrame
    ) -> None:
        pipeline = MendPipeline(verbose=False)
        pipeline.fit(clean_df)
        result = pipeline.transform(dirty_df)
        assert isinstance(result.repaired_df, pd.DataFrame)
        assert len(result.repaired_df) > 0

    def test_result_has_contract_report(
        self, clean_df: pd.DataFrame, dirty_df: pd.DataFrame
    ) -> None:
        pipeline = MendPipeline(verbose=False, enable_drift=False, enable_trace=False)
        pipeline.fit(clean_df)
        result = pipeline.transform(dirty_df)
        assert result.contract_report is not None

    def test_result_has_drift_report(
        self, clean_df: pd.DataFrame, dirty_df: pd.DataFrame
    ) -> None:
        pipeline = MendPipeline(verbose=False, enable_contract=False, enable_trace=False)
        pipeline.fit(clean_df)
        result = pipeline.transform(dirty_df)
        assert result.drift_report is not None

    def test_overall_mend_score_range(self, clean_df: pd.DataFrame) -> None:
        pipeline = MendPipeline(verbose=False)
        result = pipeline.fit_transform(clean_df)
        assert 0.0 <= result.overall_mend_score <= 100.0

    def test_transform_before_fit_raises(self, clean_df: pd.DataFrame) -> None:
        pipeline = MendPipeline(verbose=False)
        with pytest.raises(RuntimeError, match="fitted"):
            pipeline.transform(clean_df)

    def test_disabled_pillars(self, clean_df: pd.DataFrame) -> None:
        pipeline = MendPipeline(
            verbose=False,
            enable_repair=False,
            enable_contract=False,
            enable_drift=False,
            enable_trace=False,
        )
        pipeline.fit(clean_df)
        result = pipeline.transform(clean_df)
        assert result.contract_report is None
        assert result.drift_report is None
        assert result.trace_report is None

    def test_pipeline_with_sklearn_model(
        self, clean_df: pd.DataFrame, simple_sklearn_model: object
    ) -> None:
        pytest.importorskip("sklearn")
        X = clean_df[["age", "income", "score"]]
        predictions = simple_sklearn_model.predict(X)  # type: ignore
        pipeline = MendPipeline(verbose=False)
        pipeline.fit(clean_df)
        result = pipeline.transform(
            clean_df, model=simple_sklearn_model, predictions=predictions
        )
        assert result.trace_report is not None

    def test_result_summary(self, clean_df: pd.DataFrame) -> None:
        pipeline = MendPipeline(verbose=False)
        result = pipeline.fit_transform(clean_df)
        summary = result.summary()
        assert "MendPipeline" in summary
        assert "AutoRepair" in summary

    def test_result_to_dict(self, clean_df: pd.DataFrame) -> None:
        pipeline = MendPipeline(verbose=False)
        result = pipeline.fit_transform(clean_df)
        d = result.to_dict()
        assert "overall_mend_score" in d
        assert "repair" in d

    def test_result_to_json(self, clean_df: pd.DataFrame) -> None:
        import json
        pipeline = MendPipeline(verbose=False)
        result = pipeline.fit_transform(clean_df)
        j = result.to_json()
        parsed = json.loads(j)
        assert "overall_mend_score" in parsed
