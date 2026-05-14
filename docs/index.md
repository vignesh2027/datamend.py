# datamend

**Repair. Validate. Detect Drift. Trace Failures.**

The unified Python library that solves the four most painful problems in production data science — in one line of code each.

## What is datamend?

```python
import datamend

clean_df, repair_report   = datamend.repair(df)
contract                  = datamend.contract(clean_df)
violations                = datamend.validate(prod_df, contract)
drift_report              = datamend.drift(clean_df, prod_df)
failure_report            = datamend.trace(model, prod_df, predictions)
```

datamend is built on four pillars:

| Pillar | What it does | One-liner |
|:--|:--|:--|
| **AutoRepair** | Detects and fixes dirty data automatically | `datamend.repair(df)` |
| **DataContract** | Generates and enforces data schemas | `datamend.contract(df)` / `datamend.validate(df, contract)` |
| **DriftRadar** | Detects feature drift with PSI, KS, JSD | `datamend.drift(train_df, prod_df)` |
| **FailureTrace** | Root-cause attribution for model failures | `datamend.trace(model, df, preds)` |

## Install

```bash
pip install datamend
```

## Next steps

- [Quickstart](tutorials/quickstart.md) — run all four pillars in under 5 minutes
- [CLI Guide](tutorials/cli.md) — use datamend from the terminal without writing Python
- [API Reference](api/repair.md) — full API documentation
- [Plugin Guide](plugins.md) — extend datamend with custom strategies
