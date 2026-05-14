# Contributing to datamend

Thank you for your interest in improving datamend. Every contribution matters — whether it is a bug report, a repair strategy, a new drift algorithm, documentation, or a test.

## Ways to contribute

| Type | What to do |
|:--|:--|
| Bug report | Open an issue with a minimal reproducible example |
| Feature request | Open a discussion before opening a PR |
| New repair strategy | Implement `BaseRepairPlugin` and open a PR |
| New validator | Implement `BaseValidatorPlugin` and open a PR |
| New drift detector | Implement `BaseDriftDetectorPlugin` and open a PR |
| New tracer | Implement `BaseTracerPlugin` and open a PR |
| Documentation | Edit files in `docs/` and open a PR |
| Tests | Add tests in `tests/` targeting uncovered code paths |

## Development setup

```bash
git clone https://github.com/vignesh2027/datamend.py.git
cd datamend.py
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running the tests

```bash
pytest                            # Run all tests with coverage
pytest tests/test_repair.py       # Run a specific test file
pytest -k "test_repair"           # Run tests matching a pattern
```

Coverage must remain above 90% on all PRs.

## Code style

datamend uses `ruff` for linting and formatting:

```bash
ruff check datamend/              # Lint
ruff format datamend/             # Format
mypy datamend/                    # Type check
```

All public functions, classes, and methods must have Google-style docstrings with Args and Returns sections. All public API must have full type hints.

## Writing a repair plugin

```python
from datamend.plugins.base import BaseRepairPlugin
from datamend.core.repair import RepairAction
from typing import List, Tuple
import pandas as pd

class MyRepairPlugin(BaseRepairPlugin):
    name = "my_repair"
    description = "Short description of what this repairs."
    version = "0.1.0"
    author = "Your Name"

    def repair(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[RepairAction]]:
        # 1. Copy the DataFrame — never mutate in place
        df = df.copy()
        actions: List[RepairAction] = []

        # 2. Detect the issue
        # 3. Fix the issue
        # 4. Log a RepairAction for every change

        return df, actions
```

Plugins must:
- Never mutate the input DataFrame in place
- Return a `RepairAction` for every change they make
- Handle errors gracefully (do not crash the pipeline)
- Have unit tests in `tests/`

## Writing a drift detector plugin

```python
from datamend.plugins.base import BaseDriftDetectorPlugin
from typing import Any, Dict
import pandas as pd

class MyDriftPlugin(BaseDriftDetectorPlugin):
    name = "my_drift"
    description = "Custom drift detector."

    def detect(self, reference: pd.Series, current: pd.Series, column: str) -> Dict[str, Any]:
        drift_score = ...  # your logic
        return {
            "drift_score": drift_score,  # 0–100
            "drifted": drift_score > 20,
            "method": "my_method",
            "details": {},
        }
```

## Submitting a PR

1. Fork the repository and create a branch: `git checkout -b feature/my-repair-plugin`
2. Write your code, tests, and docstrings
3. Run `pytest` and confirm coverage is above 90%
4. Run `ruff check datamend/` and `mypy datamend/`
5. Open a PR against `main` with a clear description of what the change does and why

## Publishing a plugin as a separate package

If your plugin is domain-specific (e.g. medical, financial), publish it as a standalone package and declare the datamend entry-point:

```toml
[project.entry-points."datamend.plugins"]
my_plugin = "my_package.plugins:MyRepairPlugin"
```

datamend will auto-discover it when the package is installed.

## Code of conduct

Be respectful and constructive. datamend is a community project and everyone is welcome.
