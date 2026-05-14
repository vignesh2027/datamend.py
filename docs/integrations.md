# Integrations Guide

datamend integrates with MLflow, Weights & Biases, and DVC so data quality and drift history are tracked alongside your model experiments.

## MLflow

```bash
pip install "datamend[mlflow]"
```

```python
import mlflow
import datamend
from datamend.integrations import mlflow as dm_mlflow

with mlflow.start_run():
    clean_df, repair_report = datamend.repair(df)
    dm_mlflow.log_repair(repair_report)

    contract = datamend.contract(clean_df)
    contract_report = datamend.validate(prod_df, contract)
    dm_mlflow.log_contract(contract_report)

    drift_report = datamend.drift(clean_df, prod_df)
    dm_mlflow.log_drift(drift_report)
```

### Logged metrics

| Metric | Value |
|:--|:--|
| `datamend.repair.mend_score_before` | MendScore before repair |
| `datamend.repair.mend_score_after` | MendScore after repair |
| `datamend.contract.passed` | 1.0 or 0.0 |
| `datamend.drift.mend_score` | Overall drift score |
| `datamend.drift.<col>.psi` | Per-column PSI |

## Weights & Biases

```bash
pip install "datamend[wandb]"
```

```python
import wandb
import datamend
from datamend.integrations import wandb as dm_wandb

with wandb.init(project="my-project"):
    _, repair_report = datamend.repair(df)
    dm_wandb.log_repair(repair_report, step=epoch)

    drift_report = datamend.drift(train_df, prod_df)
    dm_wandb.log_drift(drift_report, step=epoch)
```

## DVC

```bash
pip install "datamend[dvc]"
```

```python
from datamend.integrations import dvc as dm_dvc
from datamend.pipeline import MendPipeline

pipeline = MendPipeline()
result = pipeline.fit_transform(train_df, prod_df)

dm_dvc.save_pipeline_result(result, output_dir="datamend_dvc")
```

Then track with DVC:

```bash
dvc metrics show datamend_dvc/drift_metrics.json
dvc plots show datamend_dvc/drift_plots.json
```
