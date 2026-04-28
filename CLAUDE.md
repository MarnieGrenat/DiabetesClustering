# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Unsupervised clustering of diabetic patient encounters from the UCI "Diabetes 130-US hospitals 1999–2008" dataset. The supervised label `readmitted` is **held out** during clustering and used only as `y_hidden` for external cluster validation (Jaccard, etc.). Assignment brief: `doc/Trabalho Prático Clustering.pdf`.

All analysis lives in a single notebook: `analysis.ipynb`. There is no application code, CLI, or test suite.

## Environment

- Python 3.11 venv at `.venv/`. Activate with `source .venv/bin/activate`.
- `requirements.txt` is intentionally minimal (`ucimlrepo`); pandas, numpy, scikit-learn, matplotlib, seaborn, and ydata-profiling are used in the notebook and must be installed manually if the venv is rebuilt.
- Data is checked into `data/diabetes+130-us+hospitals+for+years+1999-2008/` (`diabetic_data.csv`, `IDS_mapping.csv`) — no download step needed.
- EDA output goes to `reports/` (e.g. `diabetes_profile_before.html`).

## Pipeline shape (analysis.ipynb)

The cells run top-to-bottom and depend on prior cell state. Order matters; restart-and-run-all if anything upstream changes. Stages:

1. **Load + null normalization** — `?` → `NaN`.
2. **Preprocessing** — drop `encounter_id`; keep first encounter per `patient_nbr` then drop it; drop `weight`/`payer_code` (heavy missingness); fill `medical_specialty` with `"Unknown"`.
3. **ICD-9 grouping** — `diag_1/2/3` collapsed into ~9 clinical buckets (`Circulatory`, `Respiratory`, `Diabetes`, …, `Other`) via `group_icd9`. V/E codes go to `Other`.
4. **Medication pruning** — keep only the top-8 most-prescribed drugs (by count of values ≠ `'No'`); drop the rest.
5. **Label hold-out** — `readmitted` is copied to `y_hidden` and dropped from the feature frame **before** any clustering.
6. **Encoding/scaling** — `age` mapped ordinally on its sorted bin order; remaining categoricals via `pd.get_dummies(drop_first=True)`; `StandardScaler` over the full frame.
7. **Outlier removal + dim. reduction** — `IsolationForest(contamination=0.05, random_state=42)` filters rows; `y_hidden` is re-aligned with the surviving index. `PCA(n_components=0.90)` reduces dimensionality.
8. **Clustering** — downstream of step 7 (work in progress).

When modifying the notebook, preserve the `y_hidden` row-alignment after any row-filtering step, and keep `readmitted` out of the feature frame fed to clustering.

## Conventions

- Notebook prose and print statements mix Portuguese and English — follow the surrounding cell rather than normalizing.
- `GEMINI.md` predates the rename from `sample.ipynb` to `analysis.ipynb` and lists a `models/` dir that doesn't exist; treat it as historical context, not ground truth.
