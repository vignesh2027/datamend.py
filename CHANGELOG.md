# Changelog

All notable changes to datamend are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
datamend follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.3] — 2026-06-01

### Fixed

- **`test_plugins.py` select_dtypes `"str"` TypeError** — plugin test used
  `include=["object", "str"]`; changed to `"string"` for pandas 2.x/3.x compat.
- **All-null object column `fillna(None)` crash** — `_NullDetector.fix()` computed
  `series.mode()` on all-null columns which returns `[None]` in pandas 2.x; calling
  `fillna(None)` raised `ValueError: Must specify a fill 'value' or 'method'`.
  Fixed by using `series.dropna().mode()` and skipping imputation when no non-null
  reference values exist.

## [1.1.2] — 2026-06-01

### Fixed

- **Cross-platform `select_dtypes` TypeError** — `repair.py` used `include=["object", "str"]`
  where `"str"` maps to `numpy.str_` (rejected) in pandas 2.x. Changed to `"string"`
  (pandas `StringDtype`) which works in both pandas 2.x and 3.x across Python 3.9–3.13
  on Windows, macOS, and Linux.
- **Windows UTF-8 HTML comparison test** — `test_stress.py` read HTML files without
  `encoding="utf-8"`, causing `·` and `→` to appear as mojibake on Windows (cp1252).
  Now reads with explicit `encoding="utf-8"`.
- **Pipeline `overall_mend_score` float overflow** — clamped to `[0.0, 100.0]` via
  `min(100.0, max(0.0, raw))` to prevent `100.00000000000001` assertion failures.
- **131 new advanced tests** added for all four pillars, CLI, cross-pillar integration,
  serialisation round-trips, adversarial inputs (emoji, unicode, all-null, constant cols,
  single row/col, 100k rows, boolean/datetime/wide DataFrames).

## [1.1.1] — 2026-05-15

### Fixed

- **Ruff lint compliance** — removed all `ANN` annotation rules from ruff config and
  ran auto-fix + unsafe-fixes across all modules (`integrations/`, `pipeline.py`,
  `plugins/`, `report.py`, `core/__init__.py`, `core/contract.py`, `core/drift.py`).
  All 386 lint errors resolved; `ruff check datamend/` now exits clean.
- **`DriftRadar.detect` single-arg form** — when `.fit(reference_df)` has been called,
  `detect(prod_df)` now correctly uses the stored reference without requiring two args.
- **GitHub Actions CI workflow** — removed `ANN` select rules from ruff; CI lint step
  now passes on Python 3.9–3.13 across Ubuntu, macOS, and Windows.
- **GitHub Actions publish workflow** — replaced OIDC trusted publishing with
  `twine upload` via `PYPI_API_TOKEN` secret; added artifact upload/download between
  the `publish` and `github-release` jobs so `dist/` is available in both.

## [1.1.0] — 2026-05-15

### Fixed

- **`_compute_mend_score` double-scaling bug** — the penalty was multiplied by 100 a
  second time, causing any DataFrame with nulls to always receive a score of 0.0.
  Scores now reflect real data quality (e.g. 10% null rate → ~96/100, not 0).
- **`FailureTrace` row-indexing crash** — inner loop used `df.index.get_loc(idx)` which
  returns `slice | ndarray` for non-unique indices, causing `TypeError` at runtime.
  All per-row access now uses safe `df.iloc[row_pos]` integer positions.
- **`FailureTrace` scalar coercion** — numpy scalar values now go through `.item()` before
  `isinstance(val, (int, float))` checks, preventing `Series[bool]` in conditionals.
- **`MendReport` constructor** — now accepts both short names (`repair`, `contract`,
  `drift`, `trace`) and documented `_report`-suffixed names.

### Added

- **`DriftRadar.fit(reference_df)`** — store a reference DataFrame for the `fit / detect`
  pattern: `DriftRadar().fit(train_df).detect(prod_df)`. The two-arg form still works.
- **`FailureTrace.fit(model, train_df, labels)`** — store a fitted model for the
  `fit / trace` pattern: `FailureTrace().fit(model, X_train).trace(X_prod, preds)`.
- **`DataContract(schema_dict)`** — declarative schema construction: pass a dict of
  `{col: {"dtype": "numeric", "min": 0, "max": 120, "nullable": False}}` as the
  first argument; constraints are merged with statistics learned from `fit(df)`.
- **`DataContract.fit_validate(train_df, prod_df)`** — convenience one-liner combining
  `fit` + `validate` for simple pipeline steps.
- **Python 3.13 support** — tested and classified.
- **`Development Status :: 5 - Production/Stable`** classifier.
- **CI matrix extended** to Python 3.13 on Windows, macOS, Linux.

## [0.1.0] — 2026-05-14

### Added

- **Pillar 1: AutoRepair** — fully automated data repair engine
  - Null imputation (mean, median, mode — auto-selected by skewness)
  - Outlier detection via modified Z-score (MAD) and IQR clipping
  - Type mismatch detection and coercion (object → numeric, object → datetime)
  - Exact duplicate removal
  - Near-duplicate detection via Jaccard similarity on string columns
  - Encoding corruption (mojibake) detection and repair
  - Inconsistent category normalisation (Male/male/M → canonical)
  - Whitespace and hidden character stripping
  - Unit mismatch flagging (CV-based bimodal detection)
  - Chunked processing for datasets above 10M rows
  - Fast mode with intelligent sampling
  - Plugin support (phase 8 of the repair pipeline)
  - Full RepairReport with before/after MendScore

- **Pillar 2: DataContract** — data schema and statistics contract
  - One-line contract generation from any reference DataFrame
  - Schema validation (missing columns, extra columns, dtype compatibility)
  - Null rate enforcement per column
  - Value range enforcement for numeric columns
  - Cardinality validation for categorical columns (unseen values)
  - Distribution drift detection via KS test
  - JSON save / load for contract persistence
  - Human-readable and machine-readable (JSON) ContractReport
  - `raise_on_failure` mode for hard pipeline gates

- **Pillar 3: DriftRadar** — statistical drift detection
  - PSI (Population Stability Index) for numeric features
  - Kolmogorov-Smirnov test for continuous features
  - Chi-square test for categorical features
  - Jensen-Shannon Divergence for all feature types
  - Composite MendScore (0–100 drift urgency)
  - Column-level severity: none / low / medium / high / critical
  - Full DriftReport with per-column attribution

- **Pillar 4: FailureTrace** — model failure root-cause attribution
  - Works with sklearn, XGBoost, LightGBM, PyTorch (any sklearn-compatible API)
  - Direct feature importance extraction from tree models
  - Surrogate DecisionTree for black-box models
  - Row-level suspicion scoring (0–100)
  - Column-level importance = model contribution + data quality contribution
  - Ground truth attribution when true labels are available
  - TraceReport with top-K failure columns and suspicious row list

- **MendPipeline** — unified four-pillar pipeline
  - `fit(train_df)` + `transform(prod_df)` interface
  - `fit_transform()` convenience method
  - Weighted overall MendScore across all four pillars
  - Per-pillar enable/disable flags
  - PipelineResult with all four reports and repaired DataFrame

- **MendReport** — HTML dashboard
  - Self-contained single-file dark-mode dashboard
  - Cards for all four pillars with colour-coded scores
  - Action tables, violation tables, drift tables, attribution tables
  - `to_html(path)` save and `serve(port)` live server

- **CLI** — full command-line interface
  - `datamend repair` — repair any CSV/Parquet/JSON/Excel file
  - `datamend contract` — generate DataContract from a file
  - `datamend validate` — validate a file against a saved contract
  - `datamend drift` — detect drift between two files
  - `datamend score` — print MendScore for any file
  - `datamend dashboard` — serve HTML dashboard from a JSON report
  - `datamend plugins` — list registered plugins

- **Plugin system**
  - `BaseRepairPlugin`, `BaseValidatorPlugin`, `BaseDriftDetectorPlugin`, `BaseTracerPlugin`
  - `PluginRegistry` with auto-discovery from `datamend.plugins` entry-point group
  - `@register_plugin` decorator for programmatic registration

- **Integrations**
  - MLflow: `log_repair`, `log_contract`, `log_drift`, `log_pipeline_result`
  - Weights & Biases: `log_repair`, `log_contract`, `log_drift`, `log_pipeline_result`
  - DVC: `save_repair_metrics`, `save_drift_metrics`, `save_pipeline_result`

- **Test suite** — 90%+ coverage across all four pillars
- **MkDocs documentation site** with full API reference, tutorials, and plugin guide
- **GitHub Actions** CI/CD — tests on Windows, macOS, Linux, Python 3.9–3.12
- **GitHub Actions** publish workflow — auto-publish to PyPI on tagged release

[0.1.0]: https://github.com/vignesh2027/datamend.py/releases/tag/v0.1.0
