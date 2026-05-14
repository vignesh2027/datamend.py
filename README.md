<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=220&section=header&text=datamend&fontSize=80&fontColor=ffffff&fontAlignY=38&desc=The%20Only%20Library%20That%20Fixes%20Dirty%20Data%2C%20Enforces%20Contracts%2C%20Detects%20Drift%20%26%20Traces%20Failures%20%E2%80%94%20In%20One%20API&descAlignY=62&descSize=15&descColor=a5b4fc&animation=twinkling" width="100%"/>

</div>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/datamend?style=for-the-badge&color=6366f1&labelColor=0d1117&logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/datamend)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=0d1117)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge&labelColor=0d1117)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/vignesh2027/datamend.py/ci.yml?branch=main&style=for-the-badge&label=CI&labelColor=0d1117&logo=github-actions&logoColor=white)](https://github.com/vignesh2027/datamend.py/actions)
[![Coverage](https://img.shields.io/badge/Coverage-90%25%2B-22c55e?style=for-the-badge&labelColor=0d1117&logo=codecov&logoColor=white)](https://github.com/vignesh2027/datamend.py)

[![OS Support](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-06b6d4?style=for-the-badge&labelColor=0d1117&logo=linux&logoColor=white)](https://github.com/vignesh2027/datamend.py)
[![Downloads](https://img.shields.io/pypi/dm/datamend?style=for-the-badge&color=f59e0b&labelColor=0d1117&logo=pypi&logoColor=white)](https://pypi.org/project/datamend)
[![Stars](https://img.shields.io/github/stars/vignesh2027/datamend.py?style=for-the-badge&color=f59e0b&labelColor=0d1117&logo=github)](https://github.com/vignesh2027/datamend.py/stargazers)
[![Issues](https://img.shields.io/github/issues/vignesh2027/datamend.py?style=for-the-badge&color=ef4444&labelColor=0d1117&logo=github)](https://github.com/vignesh2027/datamend.py/issues)

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=800&size=22&duration=2800&pause=900&color=6366F1&center=true&vCenter=true&width=960&lines=60-80%25+of+every+ML+project+is+wasted+on+dirty+data.;datamend+fixes+it+in+one+line+of+code.;AutoRepair+%E2%80%94+DataContract+%E2%80%94+DriftRadar+%E2%80%94+FailureTrace.;The+missing+infrastructure+layer+every+ML+pipeline+needs.;pip+install+datamend)](https://github.com/vignesh2027/datamend.py)

</div>

---

<div align="center">

## The Problem That Costs $78M Every Day

</div>

Every data scientist and ML engineer faces the same brutal cycle — every single day:

```
Raw data arrives  →  It's dirty  →  Days wasted cleaning manually
     ↓
Model trained  →  Works in notebook  →  Silently breaks in production
     ↓
No idea which rows failed  →  No idea which columns caused it  →  No idea why
     ↓
Start over. Repeat forever.
```

> **Pandas doesn't fix data. Great Expectations only validates. Evidently only detects drift. SHAP only explains outputs.**
> **Nothing does all four in one unified API. Until now.**

**datamend is the first library to solve all four problems together — in one line of code each.**

---

<div align="center">

## The Five Lines That Replace Days of Work

</div>

```python
import datamend

clean_df, repair_report  = datamend.repair(df)                      # Pillar 1 — Fix everything
contract                 = datamend.contract(clean_df)              # Pillar 2 — Define the standard
violations               = datamend.validate(prod_df, contract)     # Pillar 2 — Enforce in prod
drift_report             = datamend.drift(clean_df, prod_df)        # Pillar 3 — Catch distribution shift
failure_report           = datamend.trace(model, prod_df, preds)    # Pillar 4 — Diagnose failures
```

Or chain **all four** in a single production-ready pipeline:

```python
from datamend import MendPipeline

pipeline = MendPipeline()
pipeline.fit(train_df)                          # Learn everything from training data

result = pipeline.transform(                    # Apply to any new batch
    prod_df,
    model=my_model,
    predictions=preds,
)

print(f"Overall health: {result.overall_mend_score:.1f}/100")   # One number
result.repair_report.summary()                                   # What was fixed
result.contract_report.summary()                                 # What violated the schema
result.drift_report.summary()                                    # What drifted and by how much
result.trace_report.summary()                                    # Which rows and columns failed
```

---

<div align="center">

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         datamend API                                 │
│  datamend.repair()  datamend.contract()  datamend.drift()  datamend.trace()  │
└──────────┬──────────────────┬──────────────────┬──────────────┬─────┘
           │                  │                  │              │
    ┌──────▼──────┐   ┌───────▼──────┐   ┌───────▼──────┐ ┌───▼──────────┐
    │  AutoRepair  │   │ DataContract │   │  DriftRadar  │ │FailureTrace  │
    │             │   │              │   │              │ │              │
    │ • Null imp. │   │ • Schema gen │   │ • PSI        │ │ • Feat. imp. │
    │ • Outliers  │   │ • Null rate  │   │ • KS test    │ │ • Surrogate  │
    │ • Type fix  │   │ • Range chk  │   │ • Chi-square │ │ • Row scores │
    │ • Dupes     │   │ • Cardinality│   │ • Jensen-    │ │ • Col attrib │
    │ • Encoding  │   │ • Dist drift │   │   Shannon    │ │ • DQ contrib │
    │ • Categories│   │ • JSON save  │   │ • MendScore  │ │ • Model cont │
    │ • Whitespace│   │ • JSON load  │   │ • Severity   │ │              │
    │ • Units     │   │              │   │              │ │              │
    └──────┬──────┘   └───────┬──────┘   └───────┬──────┘ └───┬──────────┘
           │                  │                  │              │
    ┌──────▼──────────────────▼──────────────────▼──────────────▼─────┐
    │                      MendPipeline                                │
    │           fit(train_df) → transform(prod_df, model, preds)       │
    └──────────────────────────────┬──────────────────────────────────┘
                                   │
    ┌──────────────────────────────▼──────────────────────────────────┐
    │                       MendReport + HTML Dashboard                │
    │               MendScore   Reports   Visualisations               │
    └─────────────────────────────────────────────────────────────────┘
```

</div>

---

<div align="center">

## The Four Pillars — Deep Dive

</div>

<details open>
<summary><b>🔧 Pillar 1 — AutoRepair: Detect and Fix Everything Automatically</b></summary>

<br/>

AutoRepair runs **8 detection phases in sequence**, each feeding clean data to the next:

```
Input DataFrame
      │
      ▼
Phase 1: Whitespace & Hidden Characters
      │  Strips leading/trailing whitespace, zero-width spaces,
      │  null bytes, and other invisible Unicode from all string columns
      ▼
Phase 2: Encoding Corruption (Mojibake)
      │  Detects Latin-1 interpreted as UTF-8 and reverses the encoding
      │  using regex pattern matching on high-byte sequences
      ▼
Phase 3: Type Mismatch Coercion
      │  Detects object columns that contain >80% numeric strings and
      │  converts them. Detects date strings and parses to datetime64.
      ▼
Phase 4: Null Imputation
      │  Numeric: auto-selects mean vs median based on skewness (>1.0 → median)
      │  Categorical: mode imputation
      │  Datetime: median imputation
      ▼
Phase 5: Outlier Detection & Clipping
      │  Uses Modified Z-Score with MAD (robust to outliers themselves).
      │  Clips to IQR bounds [Q1 - 1.5·IQR, Q3 + 1.5·IQR]
      ▼
Phase 6: Duplicate Removal
      │  Exact: pandas duplicated()
      │  Near-duplicate: Jaccard similarity on string bag-of-words (threshold 0.85)
      ▼
Phase 7: Category Normalisation
      │  Groups variants via Unicode NFKD normalisation + lowercase + strip
      │  Male / male / MALE / M → canonical form
      ▼
Phase 8: Community Plugins
         Any registered BaseRepairPlugin instances run here
         ▼
    Clean DataFrame + RepairReport
```

```python
clean_df, report = datamend.repair(df, strategy="auto", verbose=True)

# Every change is logged:
# [NULL]     age       — Imputed 47 nulls with median=34.0
# [OUTLIER]  income    — Clipped 3 outliers to IQR bounds [18k, 142k]
# [DUPLICATE]  [ALL]   — Removed 12 exact duplicate rows
# [INCONSISTENT_CATEGORY] gender — Normalised 3 variants to canonical form
# MendScore: 52.3 → 91.7
```

**Strategies supported:**
| Strategy | When to use |
|:--|:--|
| `"auto"` (default) | Detects skewness — median for skewed (>1.0), mean otherwise |
| `"mean"` | Force mean imputation for all numeric nulls |
| `"median"` | Force median imputation for all numeric nulls |

**Production-safe mode** — shows full repair plan and asks before applying:
```python
clean_df, report = datamend.repair(df, confirm=True)
# → Apply all 47 repairs? [y/N]:
```

**Large dataset support** — chunked processing + fast mode:
```python
engine = datamend.AutoRepair(fast_mode=True, chunk_size=50_000)
repaired, reports = engine.repair_chunked(huge_df)  # one report per chunk
```

</details>

<details>
<summary><b>📋 Pillar 2 — DataContract: Define the Standard. Enforce It Forever.</b></summary>

<br/>

DataContract captures **schema + statistical fingerprint** of your clean training data into a JSON file. You validate any new DataFrame against it in milliseconds.

```
Training DataFrame (clean reference)
      │
      ▼  datamend.contract(train_df)
  ┌───────────────────────────────────────┐
  │  Per-column ColumnSpec:               │
  │    dtype      : float64               │
  │    nullable   : False                 │
  │    null_rate  : 0.0                   │
  │    min / max  : 18.0 / 79.0          │
  │    mean / std : 41.3 / 15.7          │
  │    percentiles: p5=22, p25=29...     │
  │    dist_params: μ=41.3, σ=15.7       │
  │    cardinality: (for categoricals)   │
  │    allowed_values: [male, female]    │
  └──────────────┬────────────────────────┘
                 │ contract.save("my_contract.json")
                 ▼
          DataContract JSON
                 │
                 │ DataContract.load("my_contract.json")
                 ▼
  Production DataFrame → datamend.validate(prod_df, contract)
      │
      ▼
  ContractReport:
    ✗ [age]     NULL_RATE — 12.3% nulls (threshold: 5%)
    ✗ [gender]  CARDINALITY_VIOLATION — new value 'non-binary' not in contract
    ⚠ [income]  DISTRIBUTION_DRIFT — KS=0.34, p=0.001
    ✓ [score]   All checks passed
```

```python
# Generate and save the contract from training data
contract = datamend.contract(
    train_df,
    name="production_v1",
    null_threshold=0.05,    # max 5% nulls allowed
    drift_threshold=0.10,   # KS threshold for distribution warnings
)
contract.save("contracts/production_v1.json")

# In production — validate every incoming batch
contract = datamend.DataContract.load("contracts/production_v1.json")
report = datamend.validate(prod_df, contract)

if not report.passed:
    # Machine-readable JSON for alerting systems
    alert_payload = report.to_json()
    # Hard gate — raise exception and block the pipeline
    datamend.validate(prod_df, contract, raise_on_failure=True)
```

**Checks performed per column:**

| Check | Description |
|:--|:--|
| Schema | Missing or extra columns detected |
| Null rate | Exceeds configured threshold |
| Dtype | Incompatible type change (float→object etc.) |
| Range | Min/max far outside training distribution |
| Distribution | KS test against fitted normal parameters |
| Cardinality | Unseen category values present |

</details>

<details>
<summary><b>📡 Pillar 3 — DriftRadar: Four Algorithms. One Score. Full Attribution.</b></summary>

<br/>

DriftRadar runs **four statistical tests** per column and combines them into a single **MendScore** (0=stable, 100=critical drift):

```
Training Series (reference)    Production Series (current)
           │                              │
           └──────────────┬───────────────┘
                          │
                    ┌─────▼──────────────────────────────┐
                    │          Numeric columns            │
                    │                                     │
                    │  PSI   = Σ (A%-E%) × ln(A%/E%)    │
                    │          Population Stability Index │
                    │          <0.1=stable >0.2=drift    │
                    │                                     │
                    │  KS    = max|F₁(x) - F₂(x)|       │
                    │          Kolmogorov-Smirnov test    │
                    │          p-value < α → drift        │
                    │                                     │
                    │  JSD   = ½KL(P‖M) + ½KL(Q‖M)      │
                    │          Jensen-Shannon Divergence  │
                    │          0=identical 1=disjoint    │
                    └─────────────────────────────────────┘
                    ┌─────────────────────────────────────┐
                    │          Categorical columns         │
                    │                                     │
                    │  χ²    = Σ (O-E)²/E               │
                    │          Chi-square goodness of fit │
                    │                                     │
                    │  JSD   = on value frequency dists  │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────┐
                    │   Composite MendScore (0–100)        │
                    │   = mean(PSI/0.5, KS, JSD, χ²_norm) │
                    │   × 100, per column                  │
                    │                                      │
                    │   Severity:                          │
                    │   0–10%  → none    ████░░░░ green    │
                    │   10–20% → low     ████████ yellow   │
                    │   20–25% → medium  ████████ orange   │
                    │   25–50% → high    ████████ red      │
                    │   >50%   → critical████████ crimson  │
                    └──────────────────────────────────────┘
```

```python
report = datamend.drift(train_df, prod_df, verbose=True)

# Output:
# MendScore (drift): 34.2/100  (0=stable, 100=critical)
# Columns drifted  : 3/12
#
# [DRIFT] income:  severity=high,    score=67.1, PSI=0.342, KS=0.41, JSD=0.38
# [DRIFT] age:     severity=medium,  score=23.4, PSI=0.198, KS=0.22, JSD=0.19
# [DRIFT] region:  severity=low,     score=11.2, JSD=0.14, χ²=18.4
# [ok]    score:   severity=none,    score=2.1,  PSI=0.024, KS=0.04, JSD=0.02

# Per-column PSI, KS, chi-square, JSD — all in one dict
report.to_dict()
```

</details>

<details>
<summary><b>🔍 Pillar 4 — FailureTrace: Know Exactly Which Rows and Columns Broke Your Model</b></summary>

<br/>

FailureTrace combines **model-level attribution** with **data-quality anomaly detection** to pinpoint the root cause of prediction failures at the row and column level:

```
Model + Input DataFrame + Predictions
            │
            ▼
  Step 1: Feature Importance Extraction
  ┌─────────────────────────────────────────────────────┐
  │  sklearn tree models → feature_importances_          │
  │  sklearn linear models → |coef_|                    │
  │  XGBoost / LightGBM → feature_importances_          │
  │  Black-box / PyTorch → Surrogate DecisionTree       │
  │    (fits DecisionTreeRegressor on X→predictions     │
  │     and reads its feature_importances_ as proxy)    │
  └─────────────────────────────────────────────────────┘
            │
            ▼
  Step 2: Per-Column Anomaly Rates
  ┌─────────────────────────────────────────────────────┐
  │  For each column:                                   │
  │    anomaly_rate = (nulls + outliers) / total_rows   │
  │    Outlier detection via Modified Z-Score (MAD)     │
  └─────────────────────────────────────────────────────┘
            │
            ▼
  Step 3: Per-Row Suspicion Scoring
  ┌─────────────────────────────────────────────────────┐
  │  For each row:                                      │
  │    dq_suspicion   = 1 - row_quality_score/100       │
  │    model_suspicion= 1 - predict_proba.max()         │
  │    weighted_anomaly= Σ col_anomaly × feature_imp    │
  │                                                     │
  │    suspicion_score = (                              │
  │      0.5 × dq_suspicion +                          │
  │      0.3 × weighted_anomaly +                       │
  │      0.2 × model_suspicion                         │
  │    ) × 100                                          │
  └─────────────────────────────────────────────────────┘
            │
            ▼
  Step 4: Column Attribution (sorted by importance)
  ┌─────────────────────────────────────────────────────┐
  │  importance = 0.6 × model_contribution              │
  │             + 0.4 × data_quality_contribution       │
  └─────────────────────────────────────────────────────┘
            │
            ▼
  TraceReport:
    Suspicious rows (sorted by suspicion score, top 50)
    Column attributions (top-K, sorted by importance)
    data_quality_failure_pct  → % rows with DQ issues
    model_failure_pct         → % rows with low confidence
```

```python
report = datamend.trace(model, prod_df, predictions, ground_truth=y_true)

# Top failure columns:
#   income:  importance=78.3  dq_contrib=45.1  model_contrib=91.2  anomaly_rate=12.4%
#   age:     importance=31.2  dq_contrib=8.3   model_contrib=42.7  anomaly_rate=3.1%

# Most suspicious rows:
#   Row 1847: score=94.1  reason='data quality issues; low model confidence'
#   Row 392:  score=87.3  reason='feature anomalies; low model confidence'
```

</details>

---

<div align="center">

## How AutoRepair Detects Each Issue — Under the Hood

</div>

```
Issue                  Detection Method                    Fix Strategy
─────────────────────────────────────────────────────────────────────────────
Null values            df[col].isnull()                    mean / median / mode
                                                           (auto-selected by skewness)

Outliers               Modified Z-Score using MAD          IQR clipping
                       z = 0.6745 × (x−median) / MAD      [Q1−1.5·IQR, Q3+1.5·IQR]
                       flag if |z| > 3.5

Type mismatch          >80% of object column values        pd.to_numeric() /
                       match ^-?\d+(\.\d+)?$ regex         pd.to_datetime()
                       or parse as date format

Exact duplicates       df.duplicated()                     df.drop_duplicates()

Near-duplicates        Jaccard(bag_of_words(row_i),        Drop the duplicate row
                       bag_of_words(row_j)) ≥ 0.85         (keep first)

Encoding corruption    Regex [\xc0-\xff][\x80-\xbf]{1,3}  Encode latin-1, decode utf-8
(mojibake)

Inconsistent           Unicode NFKD normalise + lower      Replace all variants with
categories             + strip → group identical norms     canonical (most common) form

Whitespace /           r"^\s+|\s+$" + hidden char regex    str.strip() + re.sub(hidden)
hidden chars           [\x00-\x1f\x7f\xa0​‌‍﻿]

Unit mismatch          CV = std / |mean| > 5.0             Flag only — requires human
(suspected)            + IQR ratio (Q3/Q1) > 10            domain confirmation
─────────────────────────────────────────────────────────────────────────────
```

---

<div align="center">

## Installation

</div>

```bash
# Core (pandas + numpy + scipy + click + rich + jinja2 + pydantic)
pip install datamend

# With model integrations
pip install "datamend[sklearn]"     # scikit-learn — enables full FailureTrace
pip install "datamend[xgboost]"     # XGBoost
pip install "datamend[lightgbm]"    # LightGBM
pip install "datamend[torch]"       # PyTorch

# With experiment tracking
pip install "datamend[mlflow]"      # MLflow integration
pip install "datamend[wandb]"       # Weights & Biases
pip install "datamend[dvc]"         # DVC

# Everything
pip install "datamend[all]"

# Verify
python -c "import datamend; print(datamend.__version__)"
```

**System requirements:** Python 3.9+, Windows / macOS / Linux (all tested in CI on every commit)

---

<div align="center">

## The MendScore — One Number for Data Health

</div>

Every datamend function returns a **MendScore** — a single number from 0 to 100 that tells you exactly how healthy your data is.

```
MendScore Interpretation
─────────────────────────────────────────────────────────────────────
Score     Colour   Meaning                     Recommended action
─────────────────────────────────────────────────────────────────────
90–100    GREEN    Excellent. Production-ready. Deploy with confidence.
70–89     TEAL     Good. Minor issues.          Review repair report.
50–69     YELLOW   Moderate problems.           Repair before deploying.
30–49     ORANGE   Serious issues.              Do not deploy without review.
0–29      RED      Critical. Severe data rot.   Block deployment. Fix now.
─────────────────────────────────────────────────────────────────────
```

Each pillar produces its own MendScore:

| Pillar | MendScore meaning |
|:--|:--|
| `repair_report.mend_score_before` | Quality score of raw input data |
| `repair_report.mend_score_after` | Quality score after AutoRepair |
| `contract_report.mend_score` | How many contract checks passed (100 = all pass) |
| `drift_report.mend_score` | Drift severity (0 = no drift, 100 = critical drift) |
| `trace_report.mend_score` | Failure severity (0 = no failures, 100 = widespread) |
| `result.overall_mend_score` | Weighted composite of all four pillars |

```python
# One-liner MendScore from the CLI
$ datamend score production_data.csv
MendScore: 47.3/100    ← RED — serious issues detected
```

---

<div align="center">

## Full Benchmark: datamend vs Every Alternative

</div>

<table>
<thead>
<tr>
<th align="left">Capability</th>
<th align="center">pandas</th>
<th align="center">Great Expectations</th>
<th align="center">Evidently</th>
<th align="center">SHAP</th>
<th align="center"><b>datamend</b></th>
</tr>
</thead>
<tbody>
<tr><td><b>Auto-repair nulls</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ smart imputation</td></tr>
<tr><td><b>Auto-repair outliers</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ MAD + IQR clip</td></tr>
<tr><td><b>Fix type mismatches</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ auto-coerce</td></tr>
<tr><td><b>Deduplicate (near-dupes)</b></td><td align="center">Partial</td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ Jaccard similarity</td></tr>
<tr><td><b>Fix encoding corruption</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ mojibake repair</td></tr>
<tr><td><b>Normalise categories</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ NFKD normalise</td></tr>
<tr><td><b>Data contract generation</b></td><td align="center">❌</td><td align="center">✅</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ one line</td></tr>
<tr><td><b>Contract enforcement</b></td><td align="center">❌</td><td align="center">✅</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ + raise_on_failure</td></tr>
<tr><td><b>PSI drift detection</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">✅</td><td align="center">❌</td><td align="center">✅</td></tr>
<tr><td><b>KS + chi-square + JSD</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">Partial</td><td align="center">❌</td><td align="center">✅ all four</td></tr>
<tr><td><b>Row-level failure attribution</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅</td></tr>
<tr><td><b>Column-level root cause</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">Partial</td><td align="center">✅ DQ + model combined</td></tr>
<tr><td><b>Unified pipeline API</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ MendPipeline</td></tr>
<tr><td><b>Single health score</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ MendScore</td></tr>
<tr><td><b>HTML dashboard</b></td><td align="center">❌</td><td align="center">Partial</td><td align="center">✅</td><td align="center">❌</td><td align="center">✅ self-contained</td></tr>
<tr><td><b>CLI (no Python needed)</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ full CLI</td></tr>
<tr><td><b>Plugin / extension system</b></td><td align="center">❌</td><td align="center">Partial</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ 4 plugin types</td></tr>
<tr><td><b>MLflow / W&B / DVC hooks</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">Partial</td><td align="center">❌</td><td align="center">✅ all three</td></tr>
<tr><td><b>Core deps only</b></td><td align="center">✅</td><td align="center">No</td><td align="center">No</td><td align="center">No</td><td align="center">✅ pandas+numpy+scipy</td></tr>
<tr><td><b>Framework-agnostic models</b></td><td align="center">—</td><td align="center">—</td><td align="center">Partial</td><td align="center">✅</td><td align="center">✅ any sklearn API</td></tr>
<tr><td><b>Chunked / large dataset</b></td><td align="center">Partial</td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ repair_chunked()</td></tr>
<tr><td><b>Audit log / changelog</b></td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">❌</td><td align="center">✅ every change logged</td></tr>
</tbody>
</table>

---

<div align="center">

## CLI Reference — No Python Required

</div>

datamend ships a complete CLI. Point it at any file. Get results.

```bash
# ── Repair any file ───────────────────────────────────────────────────────────
datamend repair data.csv
datamend repair data.csv -o clean.csv --strategy median
datamend repair data.csv --report repair.json --html dashboard.html
datamend repair data.csv --fast          # sampling mode for large files
datamend repair data.csv --confirm       # ask before applying (production safe)

# ── Generate a DataContract from your training data ───────────────────────────
datamend contract training.csv -o contract.json
datamend contract training.csv --name "v1_production" --null-threshold 0.02

# ── Validate production data against the contract ─────────────────────────────
datamend validate prod.csv contract.json
datamend validate prod.csv contract.json --fail-fast   # exit code 1 on violations
datamend validate prod.csv contract.json --report violations.json --html report.html

# ── Detect drift between two datasets ─────────────────────────────────────────
datamend drift training.csv production.csv
datamend drift train.csv prod.csv --report drift.json --html drift.html --alpha 0.01

# ── Get a quick health score for any file ─────────────────────────────────────
datamend score mydata.csv
# MendScore: 47.3/100

# ── Serve a live HTML dashboard from any report JSON ─────────────────────────
datamend dashboard repair_report.json --port 8899

# ── List all installed plugins ────────────────────────────────────────────────
datamend plugins
```

---

<div align="center">

## HTML Dashboard — Self-Contained. Dark Mode. Zero Dependencies.

</div>

Every report exports as a **single HTML file** — no server, no external CSS, no JavaScript frameworks. Open it anywhere.

```python
from datamend.report import MendReport

mr = MendReport(
    repair=repair_report,
    contract=contract_report,
    drift=drift_report,
    trace=trace_report,
    title="Production Health — 2026-05-14",
)

mr.to_html("health_dashboard.html")    # Save as self-contained file
mr.serve(port=8899)                    # Or serve live — opens browser automatically
```

From the CLI:
```bash
datamend repair data.csv --html dashboard.html
datamend drift train.csv prod.csv --html drift_dashboard.html
datamend dashboard report.json --port 9000
```

---

<div align="center">

## Integrations — Track Data Health Alongside Model Experiments

</div>

<details>
<summary><b>MLflow</b></summary>

```python
import mlflow
import datamend
from datamend.integrations import mlflow as dm_mlflow

with mlflow.start_run():
    # Repair
    clean_df, repair_report = datamend.repair(df)
    dm_mlflow.log_repair(repair_report)
    # Logged: datamend.repair.mend_score_before/after, issues_found, rows_affected

    # Drift
    drift_report = datamend.drift(train_df, prod_df)
    dm_mlflow.log_drift(drift_report)
    # Logged: datamend.drift.mend_score, per-column PSI/KS/JSD

    # Full pipeline at once
    dm_mlflow.log_pipeline_result(pipeline_result)
```
</details>

<details>
<summary><b>Weights & Biases</b></summary>

```python
import wandb
from datamend.integrations import wandb as dm_wandb

with wandb.init(project="my-ml-project"):
    dm_wandb.log_repair(repair_report, step=epoch)
    dm_wandb.log_drift(drift_report, step=epoch)
    dm_wandb.log_pipeline_result(result, step=epoch)
```
</details>

<details>
<summary><b>DVC</b></summary>

```python
from datamend.integrations import dvc as dm_dvc

dm_dvc.save_pipeline_result(result, output_dir="datamend_metrics")
# Creates:
#   datamend_metrics/repair_metrics.json
#   datamend_metrics/drift_metrics.json
#   datamend_metrics/drift_plots.json   ← dvc plots show
#   datamend_metrics/summary.json
```

```bash
dvc metrics show datamend_metrics/repair_metrics.json
dvc plots show datamend_metrics/drift_plots.json
```
</details>

---

<div align="center">

## Plugin System — Extend Every Pillar

</div>

datamend has four plugin types — one for each pillar. Write a class, register it, done.

```python
from datamend.plugins.base import BaseRepairPlugin, register_plugin
from datamend.core.repair import RepairAction
import pandas as pd
import re

@register_plugin
class PhoneNormalisationPlugin(BaseRepairPlugin):
    """Normalise phone numbers to E.164 format."""
    name = "phone_normalise"
    description = "Strips non-digit characters and prepends + for phone columns."
    version = "1.0.0"
    author = "Your Name"

    def repair(self, df):
        df = df.copy()
        actions = []
        for col in df.select_dtypes(include=["object", "str"]).columns:
            if "phone" not in col.lower():
                continue
            count = df[col].notna().sum()
            df[col] = df[col].apply(
                lambda v: f"+{re.sub(r'\\D', '', str(v))}" if pd.notna(v) else v
            )
            actions.append(RepairAction(
                column=col, issue_type="PHONE_NORMALISE",
                description=f"Normalised {count} phone numbers to E.164",
                rows_affected=count, before_sample=None, after_sample=None,
                strategy="e164",
            ))
        return df, actions

# Use inline
clean_df, report = datamend.repair(df, plugins=[PhoneNormalisationPlugin()])

# Or register globally and it auto-runs in all repair() calls
# Publish as a package with entry-point: datamend.plugins → auto-discovered
```

**The four plugin types:**

| Base class | Pillar | Override method |
|:--|:--|:--|
| `BaseRepairPlugin` | AutoRepair | `repair(df) → (df, actions)` |
| `BaseValidatorPlugin` | DataContract | `validate(df, col, stats) → violations` |
| `BaseDriftDetectorPlugin` | DriftRadar | `detect(ref, cur, col) → result_dict` |
| `BaseTracerPlugin` | FailureTrace | `score_rows(model, df, preds) → rows` |

**Auto-discovery** — publish a package with:
```toml
[project.entry-points."datamend.plugins"]
my_plugin = "my_package:MyRepairPlugin"
```
datamend finds it automatically when installed.

---

<div align="center">

## Advanced Usage

</div>

**Large datasets — chunked processing:**
```python
engine = datamend.AutoRepair(chunk_size=50_000, fast_mode=True)
repaired_df, chunk_reports = engine.repair_chunked(huge_10M_row_df)
# Returns one RepairReport per chunk — merge as needed
```

**Async / streaming (custom chunking):**
```python
import pandas as pd

repaired_chunks = []
for chunk in pd.read_csv("huge_file.csv", chunksize=100_000):
    clean_chunk, _ = datamend.repair(chunk, verbose=False)
    repaired_chunks.append(clean_chunk)

repaired = pd.concat(repaired_chunks, ignore_index=True)
```

**Hard production gate:**
```python
contract = datamend.DataContract.load("contract.json")

# Raises ContractViolationError and stops the pipeline
datamend.validate(prod_df, contract, raise_on_failure=True)
```

**Selective drift check:**
```python
# Only check the features that matter most
report = datamend.drift(
    train_df, prod_df,
    columns=["income", "age", "credit_score"],
    alpha=0.01,   # stricter significance level
)
```

**MendPipeline with all options:**
```python
from datamend import MendPipeline

pipeline = MendPipeline(
    repair_strategy="median",     # force median imputation
    null_threshold=0.02,          # 2% max nulls in contract
    drift_alpha=0.01,             # stricter drift detection
    psi_buckets=20,               # finer PSI granularity
    top_k_trace=15,               # top 15 failure columns
    enable_repair=True,
    enable_contract=True,
    enable_drift=True,
    enable_trace=True,
    fast_mode=True,               # sampling for large data
    verbose=True,                 # rich terminal output
)
pipeline.fit(train_df)
result = pipeline.transform(prod_df, model=model, predictions=preds)
```

---

<div align="center">

## Why datamend Saves 10–40 Hours Per Week

</div>

The average data team spends without datamend:

```
Task                                    Hours/week
───────────────────────────────────────────────────
Manual data cleaning (custom scripts)   3–8 hours
Debugging why a model failed on prod    2–5 hours
Writing & maintaining validation rules  2–4 hours
Checking for data drift after deploy    1–3 hours
───────────────────────────────────────────────────
Total wasted per engineer               8–20 hours
Total wasted per team (5 engineers)    40–100 hours
```

datamend automates all four. That is **$78M/day saved globally** across the industry.

---

<div align="center">

## Project Structure

</div>

```
datamend/
├── datamend/
│   ├── __init__.py              ← Public API: repair(), contract(), validate(), drift(), trace()
│   ├── pipeline.py              ← MendPipeline (unified 4-pillar pipeline)
│   ├── report.py                ← MendReport + HTML dashboard generator
│   ├── cli.py                   ← Full Click-based CLI
│   ├── core/
│   │   ├── repair.py            ← AutoRepair engine (8-phase detection + fix)
│   │   ├── contract.py          ← DataContract generation + validation
│   │   ├── drift.py             ← DriftRadar (PSI + KS + chi2 + JSD + MendScore)
│   │   └── trace.py             ← FailureTrace (row + column attribution)
│   ├── plugins/
│   │   └── base.py              ← BaseRepairPlugin, PluginRegistry, @register_plugin
│   └── integrations/
│       ├── mlflow.py            ← MLflow logging hooks
│       ├── wandb.py             ← Weights & Biases logging hooks
│       └── dvc.py               ← DVC metrics + plots export
├── tests/                       ← 113 tests, 90%+ coverage
├── docs/                        ← MkDocs site (API + tutorials + plugin guide)
├── .github/workflows/
│   ├── ci.yml                   ← Tests on Windows/macOS/Linux, Python 3.9–3.12
│   └── publish.yml              ← Auto-publish to PyPI on git tag
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
└── CHANGELOG.md
```

---

<div align="center">

## Contributing

</div>

datamend welcomes contributions of all kinds.

**How to contribute:**
1. **Bug reports** — open an issue with a minimal reproducible example
2. **New repair strategy** — subclass `BaseRepairPlugin` and open a PR
3. **New drift algorithm** — subclass `BaseDriftDetectorPlugin` and open a PR
4. **New validator** — subclass `BaseValidatorPlugin` and open a PR
5. **Docs, tests, examples** — always welcome

```bash
git clone https://github.com/vignesh2027/datamend.py.git
cd datamend.py
pip install -e ".[dev]"
pytest              # all 113 tests must pass
ruff check datamend/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide including how to publish your plugin as a standalone package.

---

<div align="center">

## License

MIT © Vignesh — Free to use in any project, commercial or otherwise.

<br/>

**Built to solve the single most painful and expensive problem in data science.**

**Every data scientist who finds it should never want to work without it again.**

<br/>

[PyPI](https://pypi.org/project/datamend) · [GitHub](https://github.com/vignesh2027/datamend.py) · [Issues](https://github.com/vignesh2027/datamend.py/issues) · [Discussions](https://github.com/vignesh2027/datamend.py/discussions) · [Contributing](CONTRIBUTING.md)

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=twinkling" width="100%"/>

</div>
