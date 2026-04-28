# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Unsupervised clustering of diabetic patient encounters from the UCI "Diabetes 130-US hospitals 1999-2008" dataset. The supervised label `readmitted` is **held out** during clustering and used only as `y_hidden` for external cluster validation (ARI, Jaccard). Assignment brief: `doc/Trabalho Pratico Clustering.pdf`.

All analysis lives in a single notebook: `analysis.ipynb`. A standalone script `_clustering_check.py` reproduces the full pipeline (preproc + clustering + metrics) without Jupyter, useful for CI-style dry runs.

## Environment

- Python 3.11+ venv at `.venv/`. Activate with `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows).
- `requirements.txt` only lists `ucimlrepo`. The notebook also requires: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `prince` (for FAMD), and `ydata-profiling` (EDA reports).
- Data is checked into `data/diabetes+130-us+hospitals+for+years+1999-2008/` — no download step needed.
- EDA HTML output goes to `reports/`.

## Running

```bash
# Notebook (primary artifact)
jupyter lab analysis.ipynb   # restart-and-run-all if anything upstream changes

# Standalone pipeline check (no Jupyter needed)
python _clustering_check.py
```

There is no test suite, linter config, or build system.

## Pipeline (analysis.ipynb)

Cells run top-to-bottom and depend on prior cell state. Two parts:

### Part I — EDA and preprocessing (cells 0-26)

1. **Load + null normalization** — `?` -> `NaN`.
2. **Filter expired/hospice** — `discharge_disposition_id` in {11,13,14,19,20,21} removed (cannot be readmitted by construction).
3. **Dedup** — first encounter per `patient_nbr`; drop `encounter_id`, `patient_nbr`, `weight`, `payer_code`.
4. **Administrative ID mapping** — `admission_type_id`, `discharge_disposition_id`, `admission_source_id` mapped from int codes to clinical category strings (e.g. "Emergency", "Home", "Referral") then originals dropped. These are categorical, not ordinal.
5. **ICD-9 grouping** — `diag_1/2/3` collapsed into 9 clinical buckets via `group_icd9()`.
6. **Explicit imputation** — `A1Cresult`/`max_glu_serum` NaN -> `"NotMeasured"` (NaN is informative: exam not ordered). `medical_specialty`/`race` NaN -> `"Unknown"`. Three `gender="Unknown/Invalid"` rows dropped.
7. **Near-constant drop** — any column with mode share >= 0.99 removed (replaces the old "top-8 medications" heuristic).
8. **Skew transform** — `log1p()` on `number_outpatient/emergency/inpatient`. `age` mapped ordinally (0-9).
9. **Encoding** — `ColumnTransformer`: `StandardScaler` on continuous, `OneHotEncoder(drop='first')` on categorical (dummies NOT rescaled). Result: `X_enc` (~70k x 164).
10. **Label hold-out** — `readmitted` copied to `y_hidden` and dropped from feature frame before any clustering.
11. **Dimensionality reduction** — **FAMD** (`prince.FAMD`, 20 components) is the primary embedding for mixed data. PCA (90% variance) kept as fallback comparison. Both saved.
12. **Outlier scenarios** — Two substrates: `X_full` (all rows, for density methods) and `X_no_iso` (post-`IsolationForest(contamination=0.05)`, for K-Means). Stratified 5k sample `X_hier` for agglomerative (O(n^2) constraint).

### Part II — Clustering algorithms and evaluation (cells 27-43)

13. **Evaluation helpers** — `evaluate()` computes: silhouette, Davies-Bouldin, Calinski-Harabasz, intra-cluster variance, connectivity (Handl & Knowles), ARI, pairwise Jaccard.
14. **K-Means++ sweep** — K=2..10 on `X_famd_no`. Best K chosen by silhouette (K=3).
15. **K-Means++ vs random init** — comparison at best K (converge to same solution with n_init=10).
16. **DBSCAN** — eps swept via k-distance percentiles (p80-p95) on `X_famd_no`, min_samples=10.
17. **Agglomerative** — single/complete/average linkage on stratified 5k sample from `X_no_iso`.
18. **Stability** — bootstrap subamostragem (B=5, 50% each), ARI between bootstrap assignments.
19. **Comparative table** — all algorithms ranked. Winner: **K-Means++ k=3** (balanced, interpretable clusters; silhueta modest but non-degenerate unlike agglomerative).
20. **Cluster characterization** — per-cluster distributions of continuous features, discharge destination, primary diagnosis, and readmission rates.

## Key invariants

- **`y_hidden` row-alignment**: after any row-filtering step (outlier removal, dedup, gender filter), `y_hidden` must be re-aligned with the surviving index. Breaking this silently corrupts all external validation metrics.
- **`readmitted` must never enter the feature frame** fed to clustering — it is external validation only.
- **FAMD vs PCA**: FAMD is primary because the data is majority categorical. PCA on one-hot encoded data is a fallback, not the default path.
- **Encoding split**: continuous features get `StandardScaler`; categorical get `OneHotEncoder` without rescaling. Rescaling dummies by inverse frequency gives unjustified weight to rare categories.

## Conventions

- Notebook prose and print statements mix Portuguese and English — follow the surrounding cell rather than normalizing.
- `GEMINI.md` predates the rename from `sample.ipynb` to `analysis.ipynb` and lists a `models/` dir that doesn't exist; treat it as historical context, not ground truth.
