# Plugin Guide

datamend's plugin system lets the community extend every pillar with custom strategies.

## Plugin types

| Base class | Pillar | Purpose |
|:--|:--|:--|
| `BaseRepairPlugin` | AutoRepair | Custom repair strategies |
| `BaseValidatorPlugin` | DataContract | Custom validators |
| `BaseDriftDetectorPlugin` | DriftRadar | Custom drift algorithms |
| `BaseTracerPlugin` | FailureTrace | Custom failure attributors |

## Writing a repair plugin

```python
from datamend.plugins.base import BaseRepairPlugin, register_plugin
from datamend.core.repair import RepairAction
from typing import List, Tuple
import pandas as pd

@register_plugin
class PhoneNormalisationPlugin(BaseRepairPlugin):
    name = "phone_normalise"
    description = "Normalises phone numbers to E.164 format."
    version = "0.1.0"
    author = "Your Name"

    def repair(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[RepairAction]]:
        df = df.copy()
        actions = []
        for col in df.select_dtypes(include=["object"]).columns:
            if "phone" not in col.lower():
                continue
            import re
            original_count = df[col].notna().sum()
            df[col] = df[col].apply(self._normalise)
            actions.append(RepairAction(
                column=col,
                issue_type="PHONE_NORMALISE",
                description=f"Normalised {original_count} phone numbers",
                rows_affected=original_count,
                before_sample=None,
                after_sample=None,
                strategy="e164_normalise",
            ))
        return df, actions

    def _normalise(self, val):
        if pd.isna(val):
            return val
        digits = re.sub(r"\D", "", str(val))
        return f"+{digits}" if digits else val
```

Use it:

```python
import datamend
from my_package import PhoneNormalisationPlugin

clean_df, report = datamend.repair(df, plugins=[PhoneNormalisationPlugin()])
```

## Publishing a plugin

Add to your `pyproject.toml`:

```toml
[project.entry-points."datamend.plugins"]
phone_normalise = "my_package.plugins:PhoneNormalisationPlugin"
```

datamend auto-discovers it via `PluginRegistry.auto_discover()`.

## Plugin API Reference

::: datamend.plugins.base.BaseRepairPlugin
::: datamend.plugins.base.BaseValidatorPlugin
::: datamend.plugins.base.BaseDriftDetectorPlugin
::: datamend.plugins.base.BaseTracerPlugin
::: datamend.plugins.base.PluginRegistry
::: datamend.plugins.base.register_plugin
