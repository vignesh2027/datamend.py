# Installation

datamend supports Python 3.9+ on Windows, macOS, and Linux.

## Minimum install (core only)

```bash
pip install datamend
```

Core dependencies: `pandas`, `numpy`, `scipy`, `click`, `rich`, `jinja2`, `pydantic`.

## With optional integrations

```bash
pip install "datamend[sklearn]"    # FailureTrace with sklearn models
pip install "datamend[xgboost]"    # XGBoost support
pip install "datamend[lightgbm]"   # LightGBM support
pip install "datamend[torch]"      # PyTorch support
pip install "datamend[mlflow]"     # MLflow integration
pip install "datamend[wandb]"      # Weights & Biases integration
pip install "datamend[dvc]"        # DVC integration
pip install "datamend[all]"        # All optional dependencies
```

## Development install

```bash
git clone https://github.com/vignesh2027/datamend.py.git
cd datamend.py
pip install -e ".[dev]"
pytest
```

## Verify

```python
import datamend
print(datamend.__version__)
```
