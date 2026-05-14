# Quickstart

Get datamend running in under 5 minutes.

## Installation

```bash
pip install datamend
pip install "datamend[sklearn]"   # for FailureTrace with sklearn models
```

## 1. AutoRepair

```python
import pandas as pd
import datamend

df = pd.read_csv("raw_data.csv")

clean_df, report = datamend.repair(df)

print(report.summary())
# Issues found    : 47
# Rows affected   : 312
# MendScore before: 52.3/100
# MendScore after : 91.7/100
```

The repair report logs every change: what was fixed, why, how many rows, and which strategy was used.

## 2. DataContract

Generate a contract from your clean training data:

```python
contract = datamend.contract(clean_df, name="my_training_contract")
contract.save("my_contract.json")
```

Validate production data against it:

```python
loaded_contract = datamend.DataContract.load("my_contract.json")
report = datamend.validate(prod_df, loaded_contract)

if not report.passed:
    print(report.summary())
    # [age] NULL_RATE — null rate 12.3% exceeds threshold 5.0%
    # [gender] CARDINALITY_VIOLATION — new values: ['UNKNOWN']
```

## 3. DriftRadar

```python
drift_report = datamend.drift(train_df, prod_df)

print(drift_report.mend_score)       # 0=stable, 100=critical drift
print(drift_report.columns_drifted)  # ['feature_a', 'income']
```

## 4. FailureTrace

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier().fit(X_train, y_train)
predictions = model.predict(X_prod)

trace_report = datamend.trace(model, X_prod, predictions)

print(trace_report.top_failure_columns)   # ['income', 'age']
print(len(trace_report.suspicious_rows))  # 23
```

## 5. Unified MendPipeline

Chain everything together:

```python
from datamend import MendPipeline

pipeline = MendPipeline()
pipeline.fit(train_df)

result = pipeline.transform(prod_df, model=model, predictions=predictions)

print(f"Overall MendScore: {result.overall_mend_score:.1f}/100")
result.repair_report.summary()
result.drift_report.summary()
```

## Export an HTML Dashboard

```python
from datamend.report import MendReport

mr = MendReport(
    repair=result.repair_report,
    contract=result.contract_report,
    drift=result.drift_report,
    trace=result.trace_report,
)
mr.to_html("health_dashboard.html")
mr.serve(port=8899)   # Opens in browser
```
