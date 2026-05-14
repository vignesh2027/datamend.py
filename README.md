<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1e293b,100:0f172a&height=200&section=header&text=datamend&fontSize=72&fontColor=6366f1&fontAlignY=45&desc=Repair.%20Validate.%20Detect%20Drift.%20Trace%20Failures.&descAlignY=70&descSize=18&descColor=94a3b8&animation=fadeIn" width="100%"/>

[![PyPI version](https://img.shields.io/pypi/v/datamend?style=for-the-badge&color=6366f1&labelColor=0d1117&logo=pypi&logoColor=white)](https://pypi.org/project/datamend)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=0d1117)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge&labelColor=0d1117)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/vignesh2027/datamend.py/ci.yml?branch=main&style=for-the-badge&label=tests&labelColor=0d1117)](https://github.com/vignesh2027/datamend.py/actions)
[![Coverage](https://img.shields.io/badge/coverage-90%25%2B-22c55e?style=for-the-badge&labelColor=0d1117)](https://github.com/vignesh2027/datamend.py)
[![OS](https://img.shields.io/badge/OS-Windows%20%7C%20Mac%20%7C%20Linux-06b6d4?style=for-the-badge&labelColor=0d1117)](https://github.com/vignesh2027/datamend.py)

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=20&duration=3000&pause=800&color=6366F1&center=true&vCenter=true&width=900&lines=60-80%25+of+every+ML+project+is+wasted+on+dirty+data;datamend+solves+it+in+one+line+of+code;AutoRepair+%E2%80%94+DataContract+%E2%80%94+DriftRadar+%E2%80%94+FailureTrace;The+missing+infrastructure+layer+for+every+ML+pipeline)](https://github.com/vignesh2027/datamend.py)

</div>

---

## The Problem Nobody Has Fully Solved

Every data scientist and ML engineer faces the same expensive cycle — every single day:

1. Get raw data. It is dirty.
2. Spend **days** cleaning it manually with pandas and custom scripts.
3. Build a model. It works in training.
4. It **silently breaks in production** because prod data looks different.
5. You have no idea which rows failed, which columns caused it, or why.
6. Repeat forever.

> **This costs the global data industry over $78 million every day** in wasted engineer hours, failed deployments, and broken pipelines.

pandas doesn't fix data. Great Expectations only validates. Evidently only detects drift. SHAP only explains outputs. **Nothing does all four together in one unified, intelligent API.**

**datamend does all four in one line of code each.**

---

## Five Lines That Replace Days of Work

```python
import datamend

clean_df, repair_report   = datamend.repair(df)                    # Fix everything
contract                  = datamend.contract(clean_df)            # Define your standard
violations                = datamend.validate(prod_df, contract)   # Enforce it in prod
drift_report              = datamend.drift(clean_df, prod_df)      # Catch distribution shift
failure_report            = datamend.trace(model, prod_df, preds)  # Diagnose failures
```

Or chain all four in a single production-ready pipeline:

```python
from datamend import MendPipeline

pipeline = MendPipeline()
pipeline.fit(train_df)                          # Learn the standard from training data

result = pipeline.transform(                    # Apply to any new batch
    prod_df,
    model=my_model,
    predictions=preds,
)

print(result.overall_mend_score)               # 0–100 overall health score
result.repair_report.summary()                 # What was fixed and why
result.contract_report.summary()               # What violated the contract
result.drift_report.summary()                  # Which features drifted and by how much
result.trace_report.summary()                  # Which rows and columns caused failures
```

---

## The Four Pillars

<table>
<tr>
<td width="25%" valign="top">

### Pillar 1
## AutoRepair

Automatically detects and fixes:
- Nulls (smart context-aware imputation)
- Outliers (modified Z-score + IQR clipping)
- Type mismatches and silent coercions
- Exact and near-duplicate rows (Jaccard similarity)
- Encoding corruption and mojibake
- Inconsistent categories (`Male` / `male` / `M`)
- Date format inconsistencies
- Whitespace and hidden characters
- Unit mismatch suspects

Every change logged with before/after sample, strategy used, and rows affected.

</td>
<td width="25%" valign="top">

### Pillar 2
## DataContract

Define your data standard once from clean training data. Enforce it on every batch forever.

Catches:
- Schema violations (missing / extra columns)
- Null rate violations
- Range violations
- Cardinality violations (new unseen categories)
- Distribution shift (KS test)
- Dtype mismatches

Saves and loads as JSON. Human-readable reports and machine-readable JSON output.

</td>
<td width="25%" valign="top">

### Pillar 3
## DriftRadar

Detects statistical drift between training and production with:
- **PSI** — Population Stability Index
- **KS test** — Kolmogorov-Smirnov for continuous features
- **Chi-square** — for categorical features
- **Jensen-Shannon Divergence** — symmetric distribution distance
- **MendScore** — composite 0–100 drift urgency score

Column-level attribution shows exactly which features drifted and by how much.

</td>
<td width="25%" valign="top">

### Pillar 4
## FailureTrace

Explains why predictions fail at the row and column level.

- Works with any sklearn-compatible model, XGBoost, LightGBM, PyTorch
- Uses feature importances directly if available
- Falls back to surrogate DecisionTree for black-box models
- Row-level suspicion scores (0–100)
- Column-level root-cause attribution
- Separates data-quality contribution from model contribution

</td>
</tr>
</table>

---

## Benchmark: datamend vs. Everything Else

| Capability | pandas | Great Expectations | Evidently | SHAP | **datamend** |
|:--|:--:|:--:|:--:|:--:|:--:|
| Auto-repair dirty data | ❌ | ❌ | ❌ | ❌ | ✅ |
| Data contract generation | ❌ | ✅ | ❌ | ❌ | ✅ |
| Contract validation | ❌ | ✅ | ❌ | ❌ | ✅ |
| Drift detection (PSI) | ❌ | ❌ | ✅ | ❌ | ✅ |
| Drift detection (KS + chi2 + JSD) | ❌ | ❌ | Partial | ❌ | ✅ |
| Row-level failure attribution | ❌ | ❌ | ❌ | ❌ | ✅ |
| Column-level root cause | ❌ | ❌ | ❌ | Partial | ✅ |
| Unified pipeline API | ❌ | ❌ | ❌ | ❌ | ✅ |
| MendScore (single health metric) | ❌ | ❌ | ❌ | ❌ | ✅ |
| HTML dashboard | ❌ | ❌ | ✅ | ❌ | ✅ |
| CLI | ❌ | ❌ | ❌ | ❌ | ✅ |
| Plugin system | ❌ | ❌ | ❌ | ❌ | ✅ |
| MLflow / W&B / DVC hooks | ❌ | ❌ | Partial | ❌ | ✅ |
| Core deps (pandas + numpy + scipy only) | — | No | No | No | ✅ |

---

## Installation

```bash
pip install datamend
```

With optional integrations:

```bash
pip install "datamend[sklearn]"      # FailureTrace with sklearn models
pip install "datamend[mlflow]"       # MLflow logging
pip install "datamend[wandb]"        # Weights & Biases logging
pip install "datamend[all]"          # Everything
```

**Requirements:** Python 3.9+, Windows / macOS / Linux.
**Core dependencies:** pandas, numpy, scipy, click, rich, jinja2, pydantic.

---

## The MendScore

Every datamend function returns a **MendScore** — a single number from 0 to 100 that tells you how healthy your data is, how much it has drifted, or how severe your model failures are.

| Score range | Meaning |
|:--|:--|
| **90–100** | Excellent. Production-ready. |
| **70–89** | Good. Minor issues. |
| **50–69** | Moderate problems. Repair recommended before deployment. |
| **30–49** | Serious issues. Model reliability at risk. |
| **0–29** | Critical. Do not deploy. |

```python
clean_df, report = datamend.repair(df)
print(f"MendScore: {report.mend_score_after:.1f}/100")
```

From the CLI:

```bash
datamend score mydata.csv
# MendScore: 47.3/100
```

---

## CLI Reference

```bash
# Repair a CSV file
datamend repair data.csv -o clean.csv --html dashboard.html

# Generate a DataContract from training data
datamend contract training.csv -o my_contract.json

# Validate production data against the contract
datamend validate prod.csv my_contract.json --fail-fast

# Detect drift between two datasets
datamend drift training.csv production.csv --report drift.json --html drift.html

# Get a MendScore for any file
datamend score mydata.csv

# Serve an HTML dashboard from any report JSON
datamend dashboard repair_report.json --port 8899
```

---

## HTML Dashboard

Every report can be exported as a fully self-contained, dark-mode HTML dashboard:

```python
from datamend.report import MendReport

mr = MendReport(
    repair=repair_report,
    contract=contract_report,
    drift=drift_report,
    trace=trace_report,
    title="My Production Health Report",
)

mr.to_html("health_dashboard.html")  # Save to file
mr.serve(port=8899)                  # Or serve it live
```

---

## Integrations

### MLflow

```python
from datamend.integrations import mlflow as dm_mlflow

with mlflow.start_run():
    clean_df, repair_report = datamend.repair(df)
    dm_mlflow.log_repair(repair_report)         # Logs metrics + artifact
    dm_mlflow.log_drift(drift_report)           # Logs per-column PSI, KS, JSD
    dm_mlflow.log_pipeline_result(result)       # Logs everything at once
```

### Weights & Biases

```python
from datamend.integrations import wandb as dm_wandb
import wandb

with wandb.init(project="my-ml-project"):
    dm_wandb.log_repair(repair_report)
    dm_wandb.log_drift(drift_report)
    dm_wandb.log_pipeline_result(result)
```

### DVC

```python
from datamend.integrations import dvc as dm_dvc

dm_dvc.save_pipeline_result(result, output_dir="datamend_dvc")
# Creates: datamend_dvc/repair_metrics.json
#          datamend_dvc/drift_metrics.json
#          datamend_dvc/drift_plots.json
#          datamend_dvc/summary.json
```

---

## Plugin System

Extend datamend with custom repair strategies, validators, drift detectors, and tracers:

```python
from datamend.plugins.base import BaseRepairPlugin, register_plugin
from datamend.core.repair import RepairAction

@register_plugin
class MyDomainRepairPlugin(BaseRepairPlugin):
    name = "my_domain_repair"
    description = "Fixes domain-specific issues in medical data."

    def repair(self, df):
        # Your logic here
        return df, [RepairAction(
            column="blood_pressure",
            issue_type="DOMAIN_FIX",
            description="Normalised BP format",
            rows_affected=12,
            before_sample="120/80mmHg",
            after_sample=120.0,
            strategy="regex_parse",
        )]

# Use it
clean_df, report = datamend.repair(df, plugins=[MyDomainRepairPlugin()])
```

Publish your plugin as a package with the entry-point group `datamend.plugins` and datamend will auto-discover it.

---

## Advanced Usage

### Async / large datasets

```python
engine = datamend.AutoRepair(chunk_size=50_000, fast_mode=True)
repaired, reports = engine.repair_chunked(huge_df)
```

### Polars DataFrames

```python
import polars as pl

polars_df = pl.read_csv("data.csv")
pandas_df = polars_df.to_pandas()
clean_df, report = datamend.repair(pandas_df)
result_polars = pl.from_pandas(clean_df)
```

### Production-safe repair (with confirmation)

```python
clean_df, report = datamend.repair(df, confirm=True)
# → Prints full repair plan and asks: "Apply all repairs? [y/N]:"
```

---

## Why datamend Saves 10–40 Hours Per Week

The average data team spends:
- **3–8 hours/week** manually cleaning data with custom pandas scripts
- **2–5 hours/week** debugging why a model failed on a specific batch
- **2–4 hours/week** writing and maintaining data validation rules
- **1–3 hours/week** checking for data drift after deployments

datamend automates all four. That is **8–20 hours/week per engineer**, every week, forever.

---

## Contributing

datamend welcomes contributions of all kinds — repair strategies, validators, drift detectors, tracers, integrations, and documentation.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution guide including how to write and publish a plugin.

---

## License

MIT © Vignesh. See [LICENSE](LICENSE).

---

<div align="center">

**Built to solve the single most painful and expensive problem in data science.**

[GitHub](https://github.com/vignesh2027/datamend.py) · [PyPI](https://pypi.org/project/datamend) · [Issues](https://github.com/vignesh2027/datamend.py/issues) · [Discussions](https://github.com/vignesh2027/datamend.py/discussions)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:1e293b,100:0d1117&height=100&section=footer" width="100%"/>

</div>
