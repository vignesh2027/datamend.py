"""Plugin base classes and registry for the datamend plugin system.

Community plugins must subclass one of the four base classes and be registered
via :func:`register_plugin` or the ``datamend.plugins`` entry-point group.
"""

from __future__ import annotations

import abc
import importlib.metadata
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd

from datamend.core.repair import RepairAction


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class BaseRepairPlugin(abc.ABC):
    """Base class for custom data repair strategies.

    Subclass this and implement :meth:`repair` to add a new repair strategy
    that will be called during the AutoRepair pipeline phase 8.

    Example::

        class MyRepairPlugin(BaseRepairPlugin):
            name = "my_repair"
            description = "Fixes a custom domain-specific issue."

            def repair(self, df):
                # ... your logic ...
                return df, []
    """

    name: str = ""
    description: str = ""
    version: str = "0.1.0"
    author: str = ""

    @abc.abstractmethod
    def repair(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[RepairAction]]:
        """Apply the custom repair to *df*.

        Args:
            df: Input DataFrame (may already have been partially repaired).

        Returns:
            Tuple of (modified DataFrame, list of RepairActions describing changes).
        """

    def validate_config(self) -> bool:
        """Validate any plugin-specific configuration. Return True if valid."""
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}', version='{self.version}')"


class BaseValidatorPlugin(abc.ABC):
    """Base class for custom DataContract validators.

    Subclass this and implement :meth:`validate` to add new contract checks
    beyond the built-in schema, range, null, cardinality, and drift checks.
    """

    name: str = ""
    description: str = ""
    version: str = "0.1.0"
    author: str = ""

    @abc.abstractmethod
    def validate(
        self, df: pd.DataFrame, column: str, reference_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Validate a column in *df*.

        Args:
            df: DataFrame being validated.
            column: Column name to validate.
            reference_stats: Statistics computed during contract fitting.

        Returns:
            List of violation dicts with keys ``column``, ``violation_type``,
            ``expected``, ``observed``, ``severity``, ``message``.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}', version='{self.version}')"


class BaseDriftDetectorPlugin(abc.ABC):
    """Base class for custom drift detection algorithms.

    Subclass this to implement drift detection beyond PSI, KS, chi-square,
    and JSD. The plugin receives two Series (reference and current) and
    returns a drift score and a flag.
    """

    name: str = ""
    description: str = ""
    version: str = "0.1.0"
    author: str = ""

    @abc.abstractmethod
    def detect(
        self,
        reference: pd.Series,
        current: pd.Series,
        column: str,
    ) -> Dict[str, Any]:
        """Detect drift between *reference* and *current* for one column.

        Args:
            reference: Training (reference) Series.
            current: Production (current) Series.
            column: Column name.

        Returns:
            Dict with at minimum: ``drift_score`` (float, 0–100),
            ``drifted`` (bool), ``method`` (str), ``details`` (dict).
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}', version='{self.version}')"


class BaseTracerPlugin(abc.ABC):
    """Base class for custom failure tracers.

    Subclass this to extend FailureTrace with domain-specific anomaly scoring
    or attribution methods.
    """

    name: str = ""
    description: str = ""
    version: str = "0.1.0"
    author: str = ""

    @abc.abstractmethod
    def score_rows(
        self,
        model: Any,
        df: pd.DataFrame,
        predictions: Any,
        ground_truth: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Score rows for suspicion of prediction failure.

        Args:
            model: The model that produced *predictions*.
            df: Input DataFrame.
            predictions: Model predictions (array-like).
            ground_truth: Optional true labels.

        Returns:
            List of row-level dicts with keys ``row_index``, ``suspicion_score``,
            ``top_columns``, ``reason``.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name='{self.name}', version='{self.version}')"


# ---------------------------------------------------------------------------
# Plugin registry
# ---------------------------------------------------------------------------


class PluginRegistry:
    """Central registry for all datamend plugins.

    Plugins can be registered programmatically via :func:`register_plugin`
    or auto-discovered from the ``datamend.plugins`` setuptools entry-point group.
    """

    def __init__(self) -> None:
        self._repair: Dict[str, Type[BaseRepairPlugin]] = {}
        self._validators: Dict[str, Type[BaseValidatorPlugin]] = {}
        self._drift: Dict[str, Type[BaseDriftDetectorPlugin]] = {}
        self._tracers: Dict[str, Type[BaseTracerPlugin]] = {}

    def register(
        self,
        plugin_cls: Type[Any],
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a plugin class.

        Args:
            plugin_cls: Plugin class (must subclass one of the four bases).
            overwrite: If True, overwrite an existing plugin with the same name.

        Raises:
            ValueError: If the class is not a recognised plugin base class.
            KeyError: If a plugin with the same name is already registered and
                ``overwrite=False``.
        """
        name = getattr(plugin_cls, "name", "") or plugin_cls.__name__

        def _register(registry: Dict[str, Any]) -> None:
            if name in registry and not overwrite:
                raise KeyError(
                    f"Plugin '{name}' is already registered. Pass overwrite=True to replace it."
                )
            registry[name] = plugin_cls

        if issubclass(plugin_cls, BaseRepairPlugin):
            _register(self._repair)
        elif issubclass(plugin_cls, BaseValidatorPlugin):
            _register(self._validators)
        elif issubclass(plugin_cls, BaseDriftDetectorPlugin):
            _register(self._drift)
        elif issubclass(plugin_cls, BaseTracerPlugin):
            _register(self._tracers)
        else:
            raise ValueError(
                f"{plugin_cls.__name__} must subclass one of: BaseRepairPlugin, "
                "BaseValidatorPlugin, BaseDriftDetectorPlugin, BaseTracerPlugin."
            )

    def list_plugins(self) -> Dict[str, List[str]]:
        """Return a summary of all registered plugins grouped by type."""
        return {
            "repair": list(self._repair.keys()),
            "validators": list(self._validators.keys()),
            "drift": list(self._drift.keys()),
            "tracers": list(self._tracers.keys()),
        }

    def get_repair_plugins(self) -> List[BaseRepairPlugin]:
        """Instantiate and return all registered repair plugins."""
        return [cls() for cls in self._repair.values()]

    def get_validator_plugins(self) -> List[BaseValidatorPlugin]:
        """Instantiate and return all registered validator plugins."""
        return [cls() for cls in self._validators.values()]

    def get_drift_plugins(self) -> List[BaseDriftDetectorPlugin]:
        """Instantiate and return all registered drift detector plugins."""
        return [cls() for cls in self._drift.values()]

    def get_tracer_plugins(self) -> List[BaseTracerPlugin]:
        """Instantiate and return all registered tracer plugins."""
        return [cls() for cls in self._tracers.values()]

    def auto_discover(self, group: str = "datamend.plugins") -> int:
        """Auto-discover and register plugins from the given entry-point group.

        Packages that implement datamend plugins should declare entry-points
        under the group ``datamend.plugins`` in their ``pyproject.toml``:

        .. code-block:: toml

            [project.entry-points."datamend.plugins"]
            my_plugin = "my_package.plugins:MyRepairPlugin"

        Args:
            group: Entry-point group name.

        Returns:
            Number of plugins discovered and registered.
        """
        count = 0
        try:
            eps = importlib.metadata.entry_points(group=group)
            for ep in eps:
                try:
                    plugin_cls = ep.load()
                    self.register(plugin_cls, overwrite=False)
                    count += 1
                except Exception as exc:
                    import warnings
                    warnings.warn(
                        f"Failed to load datamend plugin '{ep.name}': {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        except Exception:
            pass
        return count

    def __repr__(self) -> str:
        total = sum(len(v) for v in self.list_plugins().values())
        return f"PluginRegistry(total={total}, {self.list_plugins()})"


# ---------------------------------------------------------------------------
# Module-level registry instance and convenience functions
# ---------------------------------------------------------------------------

_REGISTRY = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Return the global datamend plugin registry."""
    return _REGISTRY


def register_plugin(
    plugin_cls: Type[Any],
    *,
    overwrite: bool = False,
) -> Type[Any]:
    """Register a plugin class in the global registry.

    Can be used as a decorator::

        @register_plugin
        class MyRepairPlugin(BaseRepairPlugin):
            name = "my_repair"
            ...

    Args:
        plugin_cls: Plugin class to register.
        overwrite: If True, overwrite an existing plugin with the same name.

    Returns:
        The plugin class (unchanged), enabling decorator usage.
    """
    _REGISTRY.register(plugin_cls, overwrite=overwrite)
    return plugin_cls
