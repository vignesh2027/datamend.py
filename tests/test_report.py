"""Tests for MendReport and HTML dashboard generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datamend.report import MendReport, _render_html_dashboard


class TestMendReport:
    def test_to_dict_empty(self) -> None:
        mr = MendReport(title="Test")
        d = mr.to_dict()
        assert d["title"] == "Test"

    def test_to_json_valid(self) -> None:
        import json
        mr = MendReport(title="Test")
        j = mr.to_json()
        parsed = json.loads(j)
        assert parsed["title"] == "Test"

    def test_to_html_returns_string(self) -> None:
        mr = MendReport(title="Test")
        html = mr.to_html()
        assert isinstance(html, str)
        assert "datamend" in html
        assert "<!DOCTYPE html>" in html

    def test_to_html_saves_file(self) -> None:
        mr = MendReport(title="Test")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as fh:
            path = fh.name
        try:
            mr.to_html(path=path)
            content = Path(path).read_text(encoding="utf-8")
            assert "datamend" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_to_html_with_repair_report(self, dirty_df: pd.DataFrame) -> None:
        import datamend
        _, repair_report = datamend.repair(dirty_df, verbose=False)
        mr = MendReport(repair=repair_report, title="Repair Test")
        html = mr.to_html()
        assert "AutoRepair" in html
        assert "Pillar 1" in html

    def test_to_html_with_all_reports(
        self, clean_df: pd.DataFrame, dirty_df: pd.DataFrame
    ) -> None:
        import datamend
        _, repair_report = datamend.repair(dirty_df, verbose=False)
        contract = datamend.contract(clean_df)
        contract_report = datamend.validate(dirty_df.fillna(0), contract)
        drift_report = datamend.drift(clean_df, dirty_df.fillna(0), verbose=False)
        mr = MendReport(
            repair=repair_report,
            contract=contract_report,
            drift=drift_report,
            title="Full Report",
        )
        html = mr.to_html()
        assert "Pillar 1" in html
        assert "Pillar 2" in html
        assert "Pillar 3" in html


def test_render_html_dashboard_empty() -> None:
    html = _render_html_dashboard({}, title="Empty")
    assert "datamend" in html
    assert "<!DOCTYPE html>" in html


def test_render_html_dashboard_with_data() -> None:
    data = {
        "repair": {
            "mend_score_before": 60.0,
            "mend_score_after": 90.0,
            "total_issues_found": 5,
            "total_rows_affected": 12,
            "columns_repaired": ["a", "b"],
            "actions": [],
        }
    }
    html = _render_html_dashboard(data, title="Test Dashboard")
    assert "AutoRepair" in html
    assert "90.0" in html
