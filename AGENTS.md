# AGENTS.md — AutoML repository instructions

## Project
Python AutoML pipelines for CatBoost and LogisticRegression in `GeneralizedClassifierPipeline/`. Preserve existing public APIs and training behavior unless fixing a confirmed bug. Record code changes in `CHANGES.md` with tests and limitations.

## Current task: build a local tabular benchmark collection
Use this Kaggle discussion as the exact source of candidate competitions:
https://www.kaggle.com/discussions/general/217130

The ten competitions are:
- mercedes-benz-greener-manufacturing
- santander-value-prediction-challenge
- allstate-claims-severity
- bnp-paribas-cardif-claims-management
- santander-customer-transaction-prediction
- santander-customer-satisfaction
- porto-seguro-safe-driver-prediction
- ieee-fraud-detection
- walmart-recruiting-trip-type-classification
- otto-group-product-classification-challenge

### Execution
1. Verify each competition's official Kaggle page, data availability, access requirements, terms/license, target, task type, competition metric, and file layout. Do not guess metadata.
2. Download datasets where access is authorized and technically possible, using Kaggle CLI/API credentials already configured locally. Never request, print, store, or commit credentials. Do not bypass competition consent/access requirements. Report unavailable datasets with explicit reasons.
3. Create `benchmarks/` with one subdirectory per competition. Put authorized raw downloaded files in `benchmarks/<slug>/raw/`, and create `benchmarks/<slug>/README.md` documenting source URL, task type, target, official metric, rows/columns (measured), train/test structure, feature types, missing values, data size, relevant license/terms, download status, and usage notes. Use 'unknown' when not verifiable.
4. Create `benchmarks/README.md` with a comparison table for all ten datasets and instructions to obtain/reproduce the data. Add a machine-readable manifest (JSON or YAML) with verified dataset metadata, status, and local file paths.
5. Add a small reproducible download/validation script if useful: verify expected files, load samples, validate target presence, and report schema. Avoid heavy EDA or model training in this task.
6. Kaggle competition test sets commonly lack labels. For local evaluation, produce deterministic train/validation split definitions derived only from labeled training data (fixed seed, stratification for classification where possible); never claim Kaggle private leaderboard metrics are locally reproducible. Avoid target leakage and preserve any official temporal/group structure.
7. Do not commit large/raw competition datasets to Git; add suitable `.gitignore` rules, but keep metadata, documentation, download scripts, and manifests under version control. Respect licensing and redistribution restrictions. Do not silently replace missing data with synthetic data.
8. Validate download results and scripts; distinguish successful, partial, blocked, and unverified items. Do not misreport downloads as successful.

### Deliverables
- `benchmarks/` populated with available local data, individual dataset READMEs, index and manifest.
- Reproducible retrieval/validation workflow.
- Update `CHANGES.md` with exact changes, dataset status, tests run, and unresolved issues.
- Final response in Russian: brief summary plus any datasets requiring manual Kaggle access.
