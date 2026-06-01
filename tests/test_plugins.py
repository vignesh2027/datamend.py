"""Tests for the plugin system."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import pytest

from datamend.core.repair import RepairAction
from datamend.plugins.base import (
    BaseRepairPlugin,
    PluginRegistry,
    register_plugin,
    get_registry,
)


# ---------------------------------------------------------------------------
# Concrete plugin implementations for testing
# ---------------------------------------------------------------------------


class _UpperCasePlugin(BaseRepairPlugin):
    """Test plugin: uppercases all string values."""

    name = "uppercase"
    description = "Uppercase all string values (test plugin)"

    def repair(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[RepairAction]]:
        df = df.copy()
        actions: List[RepairAction] = []
        for col in df.select_dtypes(include=["object", "string"]).columns:
            count = df[col].dropna().shape[0]
            df[col] = df[col].str.upper()
            actions.append(
                RepairAction(
                    column=col,
                    issue_type="TEST_UPPERCASE",
                    description=f"Uppercased {count} values",
                    rows_affected=count,
                    before_sample=None,
                    after_sample=None,
                    strategy="uppercase",
                )
            )
        return df, actions


class _NoOpPlugin(BaseRepairPlugin):
    name = "noop"
    description = "Does nothing (test plugin)"

    def repair(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[RepairAction]]:
        return df, []


# ---------------------------------------------------------------------------
# PluginRegistry
# ---------------------------------------------------------------------------


class TestPluginRegistry:
    def test_register_repair_plugin(self) -> None:
        registry = PluginRegistry()
        registry.register(_UpperCasePlugin)
        assert "uppercase" in registry.list_plugins()["repair"]

    def test_register_duplicate_raises(self) -> None:
        registry = PluginRegistry()
        registry.register(_UpperCasePlugin)
        with pytest.raises(KeyError):
            registry.register(_UpperCasePlugin, overwrite=False)

    def test_register_duplicate_overwrite(self) -> None:
        registry = PluginRegistry()
        registry.register(_UpperCasePlugin)
        registry.register(_UpperCasePlugin, overwrite=True)
        assert "uppercase" in registry.list_plugins()["repair"]

    def test_register_invalid_class_raises(self) -> None:
        class NotAPlugin:
            pass

        registry = PluginRegistry()
        with pytest.raises(ValueError):
            registry.register(NotAPlugin)  # type: ignore

    def test_get_repair_plugins(self) -> None:
        registry = PluginRegistry()
        registry.register(_UpperCasePlugin)
        plugins = registry.get_repair_plugins()
        assert len(plugins) == 1
        assert isinstance(plugins[0], BaseRepairPlugin)

    def test_list_plugins_empty(self) -> None:
        registry = PluginRegistry()
        result = registry.list_plugins()
        assert result["repair"] == []

    def test_plugin_runs_in_autorepair(self) -> None:
        import datamend

        df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
        plugin = _UpperCasePlugin()
        repaired, report = datamend.repair(df, plugins=[plugin], verbose=False)
        assert repaired["name"].iloc[0] == "ALICE"

    def test_noop_plugin_no_actions(self) -> None:
        import datamend

        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        plugin = _NoOpPlugin()
        _, report = datamend.repair(df, plugins=[plugin], verbose=False)
        # Noop plugin should not add actions
        noop_actions = [a for a in report.actions if a.issue_type == "NOOP"]
        assert len(noop_actions) == 0


# ---------------------------------------------------------------------------
# register_plugin decorator
# ---------------------------------------------------------------------------


def test_register_plugin_decorator() -> None:
    registry = PluginRegistry()

    @register_plugin
    class _DecoratedPlugin(BaseRepairPlugin):
        name = "decorated_test"
        description = "Decorated plugin"

        def repair(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[RepairAction]]:
            return df, []

    # Should be registered in the global registry
    global_registry = get_registry()
    assert "decorated_test" in global_registry.list_plugins()["repair"]
