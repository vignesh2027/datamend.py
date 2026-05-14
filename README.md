<div align="center">

<!-- Hero Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=220&section=header&text=datamend&fontSize=80&fontColor=fff&animation=twinkling&fontAlignY=35&desc=The%20Unified%20Data%20Repair%20%E2%80%A2%20Validation%20%E2%80%A2%20Drift%20Detection%20%E2%80%A2%20Failure%20Tracing%20Library&descAlignY=58&descSize=18" width="100%"/>

<!-- Typing SVG -->
<a href="https://github.com/vignesh2027/datamend.py">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&multiline=true&repeat=true&width=800&height=100&lines=pip+install+datamend;repaired%2C+report+%3D+datamend.repair(df);One+library.+Four+pillars.+Zero+dirty+data." alt="Typing SVG" />
</a>

<br/>

<!-- Badges Row 1 -->
<a href="https://pypi.org/project/datamend/"><img src="https://img.shields.io/pypi/v/datamend?style=for-the-badge&logo=pypi&logoColor=white&color=6366f1&labelColor=1e1b4b" alt="PyPI"/></a>
<a href="https://pypi.org/project/datamend/"><img src="https://img.shields.io/pypi/dm/datamend?style=for-the-badge&logo=pypi&logoColor=white&color=7c3aed&labelColor=1e1b4b" alt="Downloads"/></a>
<a href="https://pypi.org/project/datamend/"><img src="https://img.shields.io/pypi/pyversions/datamend?style=for-the-badge&logo=python&logoColor=white&color=0ea5e9&labelColor=1e1b4b" alt="Python"/></a>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge&labelColor=1e1b4b" alt="License"/>

<br/>

<!-- Badges Row 2 -->
<img src="https://img.shields.io/badge/Tests-113%20Passing-22c55e?style=for-the-badge&logo=pytest&logoColor=white&labelColor=1e1b4b" alt="Tests"/>
<img src="https://img.shields.io/badge/Coverage-94%25-22c55e?style=for-the-badge&logo=codecov&logoColor=white&labelColor=1e1b4b" alt="Coverage"/>
<img src="https://img.shields.io/badge/Code%20Style-Ruff-f97316?style=for-the-badge&labelColor=1e1b4b" alt="Ruff"/>
<img src="https://img.shields.io/badge/Type%20Checked-mypy-0ea5e9?style=for-the-badge&labelColor=1e1b4b" alt="mypy"/>

<br/><br/>

**[📖 Docs](https://vignesh2027.github.io/datamend.py)** &nbsp;•&nbsp;
**[🚀 PyPI](https://pypi.org/project/datamend/)** &nbsp;•&nbsp;
**[🐛 Issues](https://github.com/vignesh2027/datamend.py/issues)** &nbsp;•&nbsp;
**[💬 Discussions](https://github.com/vignesh2027/datamend.py/discussions)** &nbsp;•&nbsp;
**[📝 Changelog](CHANGELOG.md)**

</div>

---

<div align="center">

## ✦ Why datamend? ✦

</div>

> Real-world data is never clean. Nulls sneak in. Distributions shift. Models fail silently on corrupted inputs.  
> **datamend** is the single library that catches, fixes, validates, monitors, and traces every data quality issue — **automatically** — so your ML pipeline never breaks from bad data again.

<br/>

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   WITHOUT datamend              WITH datamend                       │
│   ─────────────────             ──────────────                      │
│   ❌ Nulls → model crashes      ✅ Auto-imputed before fit           │
│   ❌ Drift undetected           ✅ PSI + KS test every batch         │
│   ❌ Contract violations        ✅ Schema enforced at the gate        │
│   ❌ Hours debugging            ✅ Row-level failure attribution       │
│   ❌ 5 different libraries      ✅ One unified API                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

</div>

---

## 📦 Installation

```bash
# Core (repair, contract, drift, trace)
pip install datamend

# With scikit-learn + XGBoost support
pip install "datamend[sklearn,xgboost]"

# With experiment tracking
pip install "datamend[mlflow,wandb]"

# Everything
pip install "datamend[all]"
```

> **Requires:** Python ≥ 3.9 · pandas ≥ 1.5 · numpy ≥ 1.23 · scipy ≥ 1.9

---

## ⚡ 60-Second Demo

```python
import pandas as pd
import datamend

df = pd.read_csv("production_data.csv")   # messy real-world data

# ── Pillar 1: Auto-repair everything ──────────────────────────────────
repaired, report = datamend.repair(df)
print(report.summary())
# ✔ Fixed 247 nulls · Removed 31 duplicates · Clipped 19 outliers
# ✔ MendScore: 54.2 → 96.8  (+42.6 pts)

# ── Pillar 2: Enforce your data contract ──────────────────────────────
contract = datamend.contract(train_df)
violations = datamend.validate(repaired, contract)
# ✔ 0 violations · Contract PASSED

# ── Pillar 3: Detect drift vs training data ───────────────────────────
drift = datamend.drift(train_df, repaired)
print(drift.summary())
# ⚠ 'income' drifted  PSI=0.38  KS p=0.001

# ── Pillar 4: Trace model failures to root columns ────────────────────
trace = datamend.trace(model, repaired, predictions)
print(trace.summary())
# ⚠ Top suspicious rows: [1042, 887, 3310]  Top column: 'income'
```

---

<div align="center">

## 🏛️ The Four Pillars of datamend

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        ║
║   │  AutoRepair  │──▶│ DataContract │──▶│  DriftRadar  │──▶ 📊  ║
║   │  Pillar  1   │   │  Pillar  2   │   │  Pillar  3   │        ║
║   └──────────────┘   └──────────────┘   └──────────────┘        ║
║          │                  │                  │                 ║
║          └──────────────────┼──────────────────┘                 ║
║                             │                                    ║
║                             ▼                                    ║
║                   ┌──────────────────┐                           ║
║                   │  FailureTrace    │                           ║
║                   │   Pillar  4      │                           ║
║                   └──────────────────┘                           ║
║                             │                                    ║
║                             ▼                                    ║
║               MendScore  ▓▓▓▓▓▓▓▓▓▓▓▓▓  96.8/100               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

</div>

---

## 🔧 Pillar 1 — AutoRepair

<div align="center">

> **"Tell datamend to fix it. It will."**

</div>

AutoRepair is an **8-phase intelligent repair engine** that detects and heals over 15 distinct categories of data corruption using statistics-driven algorithms — no configuration needed.

<br/>

### 🔍 The 8-Phase Detection Pipeline

```
 RAW DATAFRAME IN
        │
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Phase 1 ── NULL DETECTION & IMPUTATION                 │
 │                                                         │
 │   skewness > 1.0?  ──YES──▶  Median imputation         │
 │        │                                                │
 │        NO                                               │
 │        ▼                                                │
 │   Mean imputation  (for numeric)                        │
 │   Mode imputation  (for categorical)                    │
 └──────────────────────────┬──────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Phase 2 ── OUTLIER DETECTION (Modified Z-Score / MAD)  │
 │                                                         │
 │   MAD = median(|Xi - median(X)|)                        │
 │   Modified Z = 0.6745 × (Xi - median) / MAD            │
 │                                                         │
 │   |Z| > 3.5?  ──YES──▶  IQR clip to [Q1-1.5×IQR,      │
 │                                        Q3+1.5×IQR]     │
 └──────────────────────────┬──────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Phase 3 ── TYPE MISMATCH DETECTION                     │
 │                                                         │
 │   >80% match r"^\s*-?\d+(\.\d+)?\s*$"?                 │
 │        ──YES──▶  coerce column to float64               │
 │                                                         │
 │   >60% match ISO-8601 / common date patterns?           │
 │        ──YES──▶  coerce to datetime64                   │
 └──────────────────────────┬──────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Phase 4 ── DUPLICATE DETECTION & REMOVAL               │
 │                                                         │
 │   Exact:  pandas .duplicated(keep='first')              │
 │                                                         │
 │   Near-duplicate (Jaccard ≥ 0.85):                      │
 │     token-set similarity across string columns          │
 └──────────────────────────┬──────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Phase 5 ── ENCODING CORRUPTION (Mojibake) REPAIR       │
 │                                                         │
 │   Regex: [\xc0-\xff][\x80-\xbf]{1,3}                   │
 │        ──YES──▶  encode latin-1, decode utf-8           │
 └──────────────────────────┬──────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Phase 6 ── CATEGORY NORMALISATION                      │
 │                                                         │
 │   NFKD + lower + strip whitespace                       │
 │   "  New York  " → "new york"                           │
 │   "Nono" → "nono"  (unicode canonical)                  │
 └──────────────────────────┬──────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Phase 7 ── WHITESPACE & HIDDEN CHARACTER REMOVAL       │
 │                                                         │
 │   Remove: zero-width spaces, soft hyphens, BOM, \r, \t  │
 │   Strip invisible unicode control characters            │
 └──────────────────────────┬──────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │  Phase 8 ── UNIT MISMATCH DETECTION                     │
 │                                                         │
 │   CV > 5.0  AND  IQR ratio > 10?                        │
 │        ──YES──▶  flag column as suspect unit mix        │
 │   (salary: 50000 mixed with 50.0 = same row anomaly)    │
 └──────────────────────────┬──────────────────────────────┘
                            │
                            ▼
   REPAIRED DATAFRAME  ·  RepairReport  ·  MendScore
```

<br/>

### 📊 What Each Detector Catches

| Phase | Issue Type | Detection Algorithm | Fix Strategy |
|-------|-----------|---------------------|-------------|
| 1 | Null / NaN values | Column-wise null rate | Mean / Median / Mode imputation |
| 2 | Outliers | Modified Z-score (MAD) | IQR-bounded clipping |
| 3 | Type mismatches | Regex coverage ≥ 80% | dtype coercion |
| 4 | Exact duplicates | pandas `.duplicated()` | Keep first, drop rest |
| 4 | Near-duplicates | Jaccard token similarity ≥ 0.85 | Drop near-clone rows |
| 5 | Mojibake encoding | `[\xc0-\xff][\x80-\xbf]` regex | latin-1 → utf-8 re-encode |
| 6 | Category noise | NFKD unicode normalisation | Lowercase canonical form |
| 7 | Whitespace / invisible chars | Unicode control char regex | Strip to clean string |
| 8 | Unit mismatch | CV > 5.0 + IQR ratio > 10 | Flag + warn |

<br/>

### 💡 Usage Examples

```python
import datamend

# ── Simple one-liner ──────────────────────────────────────────────
repaired, report = datamend.repair(df)

# ── With specific strategy ────────────────────────────────────────
repaired, report = datamend.repair(df, strategy="median", verbose=True)

# ── For large datasets (10M+ rows, chunked processing) ────────────
from datamend import AutoRepair
engine = AutoRepair(strategy="auto", fast_mode=True)
repaired, report = engine.repair_chunked(df, chunk_size=500_000)

# ── Inspect what was fixed ────────────────────────────────────────
for action in report.actions:
    print(f"[{action.column}] {action.issue_type}: {action.description}")
    print(f"  Rows affected: {action.rows_affected}")

# ── Full repair report ────────────────────────────────────────────
print(report.summary())
print(f"MendScore: {report.mend_score_before:.1f} → {report.mend_score_after:.1f}")
```

<br/>

### 🧮 MendScore — The Data Health Metric

datamend computes a composite **MendScore (0–100)** that tells you exactly how healthy your data is:

```
MendScore = 100
   - 40 × null_rate          ← nulls hurt the most
   - 20 × duplicate_rate     ← dupes skew aggregations
   - 25 × outlier_rate       ← outliers corrupt models
   - 15 × whitespace_rate    ← silent model confusion
```

| Score Range | Health Grade | Interpretation |
|------------|--------------|----------------|
| 95 – 100 | 🟢 Excellent | Production-ready, no action needed |
| 85 – 94 | 🟡 Good | Minor issues, acceptable for most models |
| 70 – 84 | 🟠 Fair | Noticeable problems, repair recommended |
| 50 – 69 | 🔴 Poor | Significant corruption, repair required |
| 0 – 49 | ⛔ Critical | Severe data quality issues, stop pipeline |

---

## 📋 Pillar 2 — DataContract

<div align="center">

> **"Define what clean data looks like. Enforce it forever."**

</div>

DataContract learns the statistical fingerprint of your training data and validates every new batch against it — catching schema violations, null rate explosions, distribution shifts, and cardinality mismatches before they reach your model.

<br/>

### 🔍 Contract Fitting & Validation Flow

```
 TRAINING DATA (clean)
        │
        ▼
 ┌─────────────────────────────────────────────────────────┐
 │  DataContract.fit(train_df)                             │
 │                                                         │
 │  For each column, learns:                               │
 │    dtype          ← expected data type                  │
 │    nullable       ← is null allowed?                    │
 │    null_rate      ← acceptable null fraction            │
 │    min / max      ← numeric range bounds                │
 │    mean / std     ← distribution centre + spread        │
 │    percentiles    ← p5, p25, p50, p75, p95             │
 │    allowed_values ← set of valid categories             │
 │    cardinality    ← number of unique values             │
 │    distribution   ← KS-ready empirical CDF             │
 └──────────────────────────┬──────────────────────────────┘
                            │  contract.save("contract.json")
                            ▼
                    ┌───────────────┐
                    │ contract.json │  ← version-controlled
                    └───────┬───────┘
                            │  DataContract.load("contract.json")
                            ▼
 ┌─────────────────────────────────────────────────────────┐
 │  DataContract.validate(new_df)                          │
 │                                                         │
 │  Check 1: Missing columns?     ──FAIL──▶ CRITICAL       │
 │  Check 2: Extra columns?       ──WARN──▶ LOW            │
 │  Check 3: Null rate exceeded?  ──FAIL──▶ HIGH           │
 │  Check 4: dtype mismatch?      ──FAIL──▶ HIGH           │
 │  Check 5: Values out of range? ──FAIL──▶ MEDIUM         │
 │  Check 6: KS distribution?     ──FAIL──▶ MEDIUM         │
 │  Check 7: Cardinality shifted? ──WARN──▶ LOW            │
 └──────────────────────────┬──────────────────────────────┘
                            │
                            ▼
              ContractReport  ·  violations[]  ·  passed?
```

<br/>

### 💡 Usage Examples

```python
import datamend

# ── Fit contract on clean training data ───────────────────────────
contract = datamend.contract(train_df)
contract.save("contracts/v1.json")   # version control this!

# ── Load and validate production batch ───────────────────────────
contract = datamend.contract.load("contracts/v1.json")
report = datamend.validate(prod_df, contract)

if not report.passed:
    for v in report.violations:
        print(f"[{v.severity}] {v.column}: {v.message}")
        print(f"  Expected: {v.expected}  |  Got: {v.observed}")

# ── Raise exception on violation (for strict pipelines) ───────────
try:
    datamend.validate(prod_df, contract, raise_on_failure=True)
except datamend.ContractViolationError as e:
    # Block the pipeline, alert the team
    alert_slack(str(e))

# ── Using DataContract class directly ────────────────────────────
from datamend import DataContract
contract = DataContract(null_threshold=0.02)  # max 2% nulls allowed
contract.fit(train_df)
report = contract.validate(prod_df)
print(report.summary())
```

<br/>

### 🆚 DataContract vs Great Expectations vs Pandera

| Feature | **datamend** | Great Expectations | Pandera |
|---------|:---:|:---:|:---:|
| Auto-learn from data | ✅ | ❌ (manual) | ❌ (manual) |
| Statistical distribution check | ✅ KS-test | ❌ | ❌ |
| JSON persistence | ✅ | ✅ (JSON/YAML) | ✅ (YAML) |
| Setup lines of code | **2** | ~20 | ~10 |
| Integrated repair | ✅ | ❌ | ❌ |
| MendScore health metric | ✅ | ❌ | ❌ |
| Drift detection built-in | ✅ | ❌ | ❌ |

---

## 📡 Pillar 3 — DriftRadar

<div align="center">

> **"Know before your model knows it's broken."**

</div>

DriftRadar runs four independent statistical tests on every feature column and combines them into a single drift verdict with severity scoring — giving you early warning before degraded model performance becomes visible.

<br/>

### 🔍 Multi-Test Drift Detection Pipeline

```
 TRAINING DATA  ──────────────────────────────────────────┐
                                                           │
 PRODUCTION DATA ─────────────────────────────────────────┤
                                                           │
                                                           ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                  DriftRadar.detect()                                │
 │                                                                     │
 │   For each column:                                                  │
 │                                                                     │
 │   ┌──────────────────────────────────────────────────────────────┐  │
 │   │  Test 1: PSI  (Population Stability Index)                   │  │
 │   │                                                              │  │
 │   │   1. Build percentile-based bins on training data            │  │
 │   │   2. Count actual% and expected% per bin                     │  │
 │   │   3. PSI = Sum (actual% - expected%) x ln(actual%/expected%) │  │
 │   │                                                              │  │
 │   │   PSI < 0.10  ──▶  Stable                                   │  │
 │   │   PSI 0.10–0.25  ──▶  Slight shift (monitor)                │  │
 │   │   PSI > 0.25  ──▶  Significant drift (alert!)               │  │
 │   └──────────────────────────────────────────────────────────────┘  │
 │                                                                     │
 │   ┌──────────────────────────────────────────────────────────────┐  │
 │   │  Test 2: KS Test  (Kolmogorov-Smirnov, continuous columns)   │  │
 │   │                                                              │  │
 │   │   D = max|F_train(x) - F_prod(x)|   (max CDF distance)      │  │
 │   │   p-value < alpha (0.05)  ──▶  Distributions differ          │  │
 │   └──────────────────────────────────────────────────────────────┘  │
 │                                                                     │
 │   ┌──────────────────────────────────────────────────────────────┐  │
 │   │  Test 3: Chi-Square  (categorical columns)                   │  │
 │   │                                                              │  │
 │   │   Compare observed vs expected category frequencies          │  │
 │   │   p-value < alpha  ──▶  Category distribution shifted        │  │
 │   └──────────────────────────────────────────────────────────────┘  │
 │                                                                     │
 │   ┌──────────────────────────────────────────────────────────────┐  │
 │   │  Test 4: JSD  (Jensen-Shannon Divergence)                    │  │
 │   │                                                              │  │
 │   │   JSD(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), M = (P+Q)/2     │  │
 │   │   0 = identical  ·  1 = maximally different                 │  │
 │   └──────────────────────────────────────────────────────────────┘  │
 │                                                                     │
 │   Combined Drift Score = 0.40xPSI + 0.25xKS + 0.20xJSD + 0.15xX2 │
 │                                                                     │
 └──────────────────────────────────────────────┬──────────────────────┘
                                                │
                                                ▼
              DriftReport  ·  per-column results  ·  MendScore
```

<br/>

### 📊 Drift Severity Thresholds

| PSI Value | Severity | Recommended Action |
|-----------|----------|-------------------|
| < 0.10 | ✅ None | No action needed |
| 0.10 – 0.20 | 🟡 Low | Monitor closely |
| 0.20 – 0.25 | 🟠 Medium | Investigate source |
| 0.25 – 0.50 | 🔴 High | Retrain model soon |
| > 0.50 | ⛔ Critical | Stop serving, retrain now |

<br/>

### 💡 Usage Examples

```python
import datamend

# ── Basic drift detection ─────────────────────────────────────────
report = datamend.drift(train_df, prod_df)
print(report.summary())

# ── Only check specific columns ───────────────────────────────────
report = datamend.drift(train_df, prod_df, columns=["age", "income", "tenure"])

# ── Inspect each column's drift metrics ──────────────────────────
for col, result in report.column_results.items():
    if result.drifted:
        print(f"[DRIFT] {col}")
        print(f"  PSI={result.psi:.3f}  KS p={result.ks_pvalue:.4f}")
        print(f"  JSD={result.jsd:.3f}  Severity: {result.severity}")

# ── With custom significance level ───────────────────────────────
from datamend import DriftRadar
radar = DriftRadar(psi_buckets=20, alpha=0.01, verbose=True)
report = radar.detect(train_df, prod_df)

# ── Only numeric or only categorical ─────────────────────────────
numeric_cols = prod_df.select_dtypes("number").columns.tolist()
report = datamend.drift(train_df, prod_df, columns=numeric_cols)
```

<br/>

### 🆚 DriftRadar vs Evidently vs NannyML

| Feature | **datamend** | Evidently | NannyML |
|---------|:---:|:---:|:---:|
| PSI (numeric drift) | ✅ | ✅ | ✅ |
| KS test | ✅ | ✅ | ✅ |
| Chi-Square | ✅ | ✅ | ❌ |
| Jensen-Shannon Divergence | ✅ | ❌ | ❌ |
| Combined drift score | ✅ | ❌ | ✅ |
| Integrated repair pipeline | ✅ | ❌ | ❌ |
| HTML dashboard (offline) | ✅ | ✅ | ✅ |
| Zero server / zero cloud | ✅ | ✅ | ❌ |
| Setup complexity | **2 lines** | ~10 lines | ~15 lines |

---

## 🔬 Pillar 4 — FailureTrace

<div align="center">

> **"Your model failed. Which rows? Which columns? Why?"**

</div>

FailureTrace provides **row-level and column-level attribution** of model failures. It combines data-quality signals with model confidence estimates and surrogate model explanations to surface the exact rows and features causing predictions to go wrong.

<br/>

### 🔍 Failure Attribution Pipeline

```
 MODEL + DATAFRAME + PREDICTIONS
              │
              ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 1: Feature Importance (Column Attribution)                    │
 │                                                                     │
 │  Native importances?  ──YES──▶  sklearn .feature_importances_       │
 │       │                         xgboost .feature_importances_       │
 │       │                         lightgbm .feature_importances_      │
 │       │                         torch .weight.abs().mean()          │
 │       NO                                                            │
 │       ▼                                                             │
 │  Surrogate:  DecisionTreeRegressor(X, predictions)                  │
 │              → extract .feature_importances_                        │
 └──────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 2: Data Quality Score (Per Row)                               │
 │                                                                     │
 │  dq_score = 1.0                                                     │
 │    - 0.3 x has_any_null                                             │
 │    - 0.3 x is_outlier  (modified Z-score)                          │
 │    - 0.2 x has_encoding_issue                                       │
 │    - 0.2 x has_type_mismatch                                        │
 │                                                                     │
 │  dq_suspicion = 1.0 - dq_score                                     │
 └──────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 3: Model Confidence Score (Per Row)                           │
 │                                                                     │
 │  Classifier:  confidence = 1 - max(predict_proba(row))              │
 │               (low confidence = high suspicion)                     │
 │                                                                     │
 │  Regressor:   confidence from normalized absolute residuals         │
 │                                                                     │
 │  model_suspicion = 1.0 - confidence                                 │
 └──────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 4: Composite Suspicion Score (Per Row)                        │
 │                                                                     │
 │  suspicion = 0.50 x dq_suspicion                                   │
 │            + 0.30 x weighted_anomaly_score                          │
 │            + 0.20 x model_suspicion                                 │
 │                                                                     │
 │  Top-K rows by suspicion score = "suspicious rows"                 │
 └──────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 5: Column Attribution Score (Per Column)                      │
 │                                                                     │
 │  col_score = 0.6 x model_importance                                 │
 │            + 0.4 x data_quality_contribution                        │
 │                                                                     │
 │  Sorted descending → top columns driving failures                   │
 └──────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
         TraceReport  ·  suspicious_rows[]  ·  column_attributions{}
```

<br/>

### 💡 Usage Examples

```python
import datamend

# ── Basic failure trace ───────────────────────────────────────────
report = datamend.trace(model, df, predictions)
print(report.summary())

# ── With ground truth (shows actual errors) ───────────────────────
report = datamend.trace(model, df, predictions, ground_truth=y_true)

# ── Inspect suspicious rows ───────────────────────────────────────
for row in report.suspicious_rows[:5]:
    print(f"Row {row.row_index}  suspicion={row.suspicion_score:.3f}")
    print(f"  Top cols: {row.top_columns}")
    print(f"  DQ score: {row.data_quality_score:.3f}")
    print(f"  Reason: {row.reason}")

# ── Inspect which columns drive failures ──────────────────────────
for col, attr in sorted(report.column_attributions.items(),
                        key=lambda x: -x[1].importance_score):
    print(f"{col}: importance={attr.importance_score:.3f}  "
          f"anomaly_rate={attr.anomaly_rate:.3f}")

# ── Works with sklearn, XGBoost, LightGBM, PyTorch ───────────────
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
report_sk = datamend.trace(rf_model, df, preds)
report_xgb = datamend.trace(xgb_model, df, preds)
```

<br/>

### 🆚 FailureTrace vs SHAP vs LIME

| Feature | **datamend** | SHAP | LIME |
|---------|:---:|:---:|:---:|
| Row-level suspicion score | ✅ | ❌ | ❌ |
| Data quality × model signal | ✅ | ❌ | ❌ |
| Zero-configuration | ✅ | ❌ (needs tree explainer) | ❌ |
| Works on black-box models | ✅ | ⚠ (KernelSHAP slow) | ✅ |
| Column attribution | ✅ | ✅ | ✅ |
| Integrated pipeline | ✅ | ❌ | ❌ |
| HTML dashboard output | ✅ | ❌ | ❌ |

---

## 🚀 MendPipeline — All Four Pillars, One Call

For production ML systems, `MendPipeline` chains all four pillars into a single, stateful object:

```python
from datamend import MendPipeline

# ── Fit on clean training data (once) ────────────────────────────
pipeline = MendPipeline(
    repair_strategy="auto",
    null_threshold=0.05,
    drift_alpha=0.05,
    psi_buckets=10,
    top_k_trace=10,
    verbose=True,
)
pipeline.fit(train_df)

# ── Run on every production batch ────────────────────────────────
result = pipeline.transform(
    prod_df,
    model=model,
    predictions=preds,
    ground_truth=y_true,    # optional
)

# ── Full report ───────────────────────────────────────────────────
print(result.summary())
# =================================================================
# datamend MendPipeline — Full Health Report
# =================================================================
#   Overall MendScore   : 91.4/100
#
#   [Pillar 1] AutoRepair
#     Issues fixed      : 142
#     MendScore change  : 54.2 → 96.8
#
#   [Pillar 2] DataContract — PASSED
#     Violations        : 0
#     MendScore         : 98.0
#
#   [Pillar 3] DriftRadar — STABLE
#     Columns drifted   : 0
#     MendScore (drift) : 4.2
#
#   [Pillar 4] FailureTrace
#     Suspicious rows   : 3
#     MendScore         : 87.1

# ── Export repaired data ──────────────────────────────────────────
result.repaired_df.to_parquet("clean_batch.parquet")

# ── Serialize to JSON ─────────────────────────────────────────────
result.to_json()
```

<br/>

### Overall MendScore Formula

```
Overall MendScore =
    0.35 x repair_score_after
  + 0.30 x contract_score
  + 0.20 x (100 - drift_score)    ← inverted: low drift = good
  + 0.15 x (100 - trace_score)    ← inverted: low failures = good
```

---

## 🖥️ HTML Dashboard

datamend generates a self-contained, **single-file dark-mode HTML dashboard** — no server, no internet, no dependencies:

```python
from datamend import MendReport

# Build report from individual pillar outputs
report = MendReport(
    repair_report=repair_report,
    contract_report=contract_report,
    drift_report=drift_report,
    trace_report=trace_report,
)

# Write dashboard to disk
report.to_html("dashboard.html")

# Or launch a live server in your browser
report.serve(port=8080, open_browser=True)
```

**Dashboard sections:**

```
┌────────────────────────────────────────────────────────────┐
│  datamend Dashboard                           MendScore 96 │
├────────────┬────────────┬────────────┬────────────────────┤
│ AutoRepair │  Contract  │ DriftRadar │  FailureTrace       │
│  Fixes: 142│  PASSED ✓  │  STABLE ✓  │  Rows: 3           │
├────────────┴────────────┴────────────┴────────────────────┤
│  Repair Actions Table   (sortable, filterable)            │
├────────────────────────────────────────────────────────────┤
│  Contract Violations    (severity colour-coded)           │
├────────────────────────────────────────────────────────────┤
│  Drift Results          (per-column PSI/KS/JSD)           │
├────────────────────────────────────────────────────────────┤
│  Column Attribution     (importance scores bar chart)     │
└────────────────────────────────────────────────────────────┘
```

---

## 💻 CLI Reference

datamend ships a full command-line interface:

```bash
# ── Repair ────────────────────────────────────────────────────────
datamend repair data.csv -o repaired.csv --strategy median --verbose
datamend repair data.parquet -o clean.parquet --fast

# ── Validate against a contract ───────────────────────────────────
datamend validate data.csv --contract contracts/v1.json
datamend contract data.csv -o contracts/v1.json   # fit contract

# ── Detect drift ──────────────────────────────────────────────────
datamend drift train.csv prod.csv --alpha 0.01 --columns age income

# ── Score data quality ────────────────────────────────────────────
datamend score data.csv           # prints MendScore

# ── Generate HTML dashboard ───────────────────────────────────────
datamend dashboard data.csv -o report.html --open

# ── List registered plugins ───────────────────────────────────────
datamend plugins list

# ── Supported formats: CSV · Parquet · JSON · Excel (.xlsx) ───────
datamend repair data.xlsx -o clean.xlsx
```

---

## 🔌 Plugin System

Build custom repair logic and plug it in with a decorator:

```python
from datamend.plugins.base import BaseRepairPlugin, register_plugin
from datamend.core.repair import RepairAction
import pandas as pd

@register_plugin
class ClipNegativePlugin(BaseRepairPlugin):
    name = "clip_negative"
    description = "Clips all negative values in numeric columns to 0"

    def repair(self, df):
        df = df.copy()
        actions = []
        for col in df.select_dtypes("number").columns:
            mask = df[col] < 0
            count = mask.sum()
            if count > 0:
                df.loc[mask, col] = 0
                actions.append(RepairAction(
                    column=col,
                    issue_type="NEGATIVE_VALUE",
                    description=f"Clipped {count} negative values to 0",
                    rows_affected=int(count),
                    before_sample=None, after_sample=None,
                    strategy="clip_negative",
                ))
        return df, actions

# ── Use your plugin ───────────────────────────────────────────────
repaired, report = datamend.repair(df, plugins=[ClipNegativePlugin()])
```

**Plugin auto-discovery** via entry points:

```toml
# In your pyproject.toml
[project.entry-points."datamend.plugins"]
my_plugin = "my_package.plugins:MyPlugin"
```

---

## 🔗 Integrations

### MLflow

```python
from datamend.integrations.mlflow import log_repair, log_drift, log_pipeline_result
import mlflow

with mlflow.start_run():
    repaired, repair_report = datamend.repair(df)
    log_repair(repair_report)           # logs MendScore, issue counts as metrics

    pipeline_result = pipeline.transform(prod_df, model=model, predictions=preds)
    log_pipeline_result(pipeline_result)  # logs all 4 pillars + artifacts
```

### Weights & Biases

```python
from datamend.integrations.wandb import log_repair, log_drift

import wandb
wandb.init(project="my-ml-project")

repaired, repair_report = datamend.repair(df)
log_repair(repair_report)      # logs to current wandb run

drift_report = datamend.drift(train_df, prod_df)
log_drift(drift_report)
```

### DVC

```python
from datamend.integrations.dvc import save_repair_metrics, save_pipeline_result

repaired, report = datamend.repair(df)
save_repair_metrics(report, path="metrics/repair.json")    # git + dvc tracked

result = pipeline.transform(prod_df, model=model, predictions=preds)
save_pipeline_result(result, path="metrics/pipeline.json")
```

---

## ⚙️ Advanced Usage

<details>
<summary><b>🔹 Async / Concurrent Processing</b></summary>

```python
import asyncio
import datamend

async def process_batch(df):
    loop = asyncio.get_event_loop()
    # Run blocking repair in a thread pool
    repaired, report = await loop.run_in_executor(
        None, lambda: datamend.repair(df, verbose=False)
    )
    return repaired, report

# Process multiple batches concurrently
tasks = [process_batch(batch) for batch in batches]
results = await asyncio.gather(*tasks)
```

</details>

<details>
<summary><b>🔹 Large Dataset — Chunked Mode</b></summary>

```python
from datamend import AutoRepair

# Handles 50M+ rows without memory blowup
engine = AutoRepair(strategy="median", fast_mode=True)
repaired, report = engine.repair_chunked(
    df,
    chunk_size=1_000_000,   # process 1M rows at a time
)
print(f"Total rows processed: {len(repaired):,}")
print(f"MendScore: {report.mend_score_after:.1f}")
```

</details>

<details>
<summary><b>🔹 Production-Safe Selective Repair</b></summary>

```python
# Repair only specific columns (e.g., don't touch ID columns)
from datamend import AutoRepair

engine = AutoRepair(strategy="auto")
subset = df[["age", "income", "score"]].copy()
repaired_subset, report = engine.fit_transform(subset)

# Merge back into original frame
df[["age", "income", "score"]] = repaired_subset
```

</details>

<details>
<summary><b>🔹 Selective Drift Monitoring</b></summary>

```python
# Monitor only numeric features for drift (skip ID/timestamp cols)
numeric_cols = [c for c in prod_df.select_dtypes("number").columns
                if c not in ["id", "timestamp", "row_num"]]

report = datamend.drift(train_df, prod_df, columns=numeric_cols)

# Send alert if any column is critical
critical = [c for c, r in report.column_results.items()
            if r.severity == "critical"]
if critical:
    send_pagerduty_alert(f"Critical drift: {critical}")
```

</details>

<details>
<summary><b>🔹 Custom DataContract Rules</b></summary>

```python
from datamend import DataContract

# Strict contract: 0% nulls, max 10% cardinality change
contract = DataContract(
    null_threshold=0.0,        # zero nulls allowed
)
contract.fit(train_df)

# Save with metadata
import json
contract_dict = json.loads(contract.to_json())
contract_dict["version"] = "1.2.0"
contract_dict["fitted_on"] = "2024-01-15"
with open("contract_v1.2.json", "w") as f:
    json.dump(contract_dict, f, indent=2)
```

</details>

---

## 📊 Benchmark

Measured on a 100,000-row · 20-column dataset (MacBook Pro M2, Python 3.11):

| Task | **datamend** | pandas manual | Great Expectations | Evidently | SHAP |
|------|:-----------:|:------------:|:-----------------:|:---------:|:----:|
| Null imputation | **0.12s** | 0.08s | N/A | N/A | N/A |
| Outlier detection + fix | **0.31s** | ~1.2s manual | N/A | N/A | N/A |
| Duplicate removal | **0.09s** | 0.07s | N/A | N/A | N/A |
| Full data repair | **0.61s** | ~4s manual | N/A | N/A | N/A |
| Contract fit | **0.18s** | N/A | ~2.1s | N/A | N/A |
| Contract validate | **0.11s** | N/A | ~0.9s | N/A | N/A |
| Drift detection (10 cols) | **0.29s** | N/A | N/A | ~0.8s | N/A |
| Failure trace (RF model) | **1.14s** | N/A | N/A | N/A | ~8.2s |
| **Full pipeline** | **2.1s** | ~7s+ combined | N/A | N/A | N/A |

> Benchmarks are indicative. Performance varies by data shape, column types, and hardware.

---

## 🏗️ Architecture & Project Structure

```
datamend/
│
├── datamend/                      ← library package
│   ├── __init__.py                ← top-level API (repair, contract, drift, trace)
│   ├── pipeline.py                ← MendPipeline (all 4 pillars unified)
│   ├── report.py                  ← MendReport + HTML dashboard generator
│   ├── cli.py                     ← Click CLI (repair/validate/drift/score/dashboard)
│   │
│   ├── core/
│   │   ├── repair.py              ← AutoRepair — 8-phase engine (15+ detectors)
│   │   ├── contract.py            ← DataContract — fit / validate / persist
│   │   ├── drift.py               ← DriftRadar — PSI + KS + chi² + JSD
│   │   └── trace.py               ← FailureTrace — row + column attribution
│   │
│   ├── plugins/
│   │   └── base.py                ← BaseRepairPlugin, PluginRegistry, @register_plugin
│   │
│   └── integrations/
│       ├── mlflow.py              ← MLflow metrics + artifact logging
│       ├── wandb.py               ← W&B metrics logging
│       └── dvc.py                 ← DVC-tracked JSON metrics
│
├── tests/                         ← 113 tests, 94% coverage
│   ├── conftest.py                ← shared fixtures
│   ├── test_repair.py             ← 32 tests
│   ├── test_contract.py           ← 22 tests
│   ├── test_drift.py              ← 19 tests
│   ├── test_trace.py              ← 11 tests
│   ├── test_pipeline.py           ← 12 tests
│   ├── test_report.py             ← 8 tests
│   └── test_plugins.py            ← 9 tests
│
├── .github/
│   ├── workflows/ci.yml           ← Tests: ubuntu/windows/macos × py3.9–3.12
│   └── workflows/publish.yml      ← PyPI trusted publish on v*.*.* tags
│
├── pyproject.toml
└── README.md
```

---

## 🧪 Running Tests

```bash
git clone https://github.com/vignesh2027/datamend.py.git
cd datamend.py

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run all 113 tests with coverage
pytest tests/ -v --cov=datamend --cov-report=term-missing

# Run a single pillar
pytest tests/test_repair.py -v
pytest tests/test_drift.py -v
```

---

## ⏱️ Time Saved Per Week

| Task | Manual time | With datamend | Saved |
|------|------------|---------------|-------|
| Null imputation per dataset | ~25 min | < 1 sec | **25 min** |
| Outlier detection & fix | ~45 min | < 1 sec | **45 min** |
| Schema validation setup | ~2 hours | 2 lines | **2 hours** |
| Drift monitoring setup | ~3 hours | 1 line | **3 hours** |
| Debugging model failures | ~4 hours | 2 sec | **~4 hours** |
| **Total per week** | **~10+ hours** | **< 5 seconds** | **10 hours** |

---

## 📋 Requirements

| Package | Version | Why |
|---------|---------|-----|
| pandas | ≥ 1.5.0 | Core DataFrame operations |
| numpy | ≥ 1.23.0 | Numerical computations |
| scipy | ≥ 1.9.0 | KS test, chi-square, statistical tests |
| click | ≥ 8.0.0 | CLI framework |
| rich | ≥ 13.0.0 | Beautiful terminal output |
| jinja2 | ≥ 3.1.0 | HTML dashboard templating |
| pydantic | ≥ 2.0.0 | Data validation models |

**Optional extras:**

```bash
pip install "datamend[sklearn]"   # scikit-learn integration
pip install "datamend[xgboost]"   # XGBoost native importances
pip install "datamend[lightgbm]"  # LightGBM native importances
pip install "datamend[torch]"     # PyTorch layer attribution
pip install "datamend[mlflow]"    # MLflow experiment tracking
pip install "datamend[wandb]"     # Weights & Biases logging
pip install "datamend[dvc]"       # DVC metric tracking
pip install "datamend[all]"       # Everything
```

---

## 🗺️ Roadmap

- [x] AutoRepair — 8-phase repair engine
- [x] DataContract — statistical contract learning
- [x] DriftRadar — PSI + KS + chi² + JSD
- [x] FailureTrace — surrogate row attribution
- [x] MendPipeline — unified 4-pillar pipeline
- [x] CLI — repair / validate / drift / score / dashboard
- [x] HTML dashboard — self-contained dark-mode output
- [x] MLflow / W&B / DVC integrations
- [x] Plugin system with entry-point discovery
- [x] PyPI release (0.1.0)
- [ ] Async native support (0.2.0)
- [ ] Polars DataFrame support (0.2.0)
- [ ] Time-series drift (CUSUM / ADWIN) (0.3.0)
- [ ] REST API server mode (0.3.0)
- [ ] Grafana plugin for MendScore dashboards (0.4.0)
- [ ] AutoML-style repair strategy search (0.5.0)

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss the change, then submit a PR.

```bash
# Fork and clone
git clone https://github.com/<your-username>/datamend.py.git

# Install dev dependencies
pip install -e ".[dev]"

# Run the full test suite before submitting
pytest tests/ -v
ruff check datamend/
mypy datamend/
```

---

## 📄 License

**MIT** — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with care by [Vignesh](https://github.com/vignesh2027)**

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
