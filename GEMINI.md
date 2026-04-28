# Diabetes Clustering Analysis - Project Context

## Project Overview
This project focuses on identifying patient profiles among diabetic individuals hospitalized in 130 US hospitals (1999–2008) using unsupervised learning (clustering). While the dataset was originally designed for predicting 30-day readmission (supervised), this analysis discards the labels during the clustering phase and uses them only for external validation.

### Main Technologies
- **Python 3**: Primary programming language.
- **Pandas**: Data manipulation and analysis.
- **ydata-profiling**: Used for automated Exploratory Data Analysis (EDA).
- **UCI ML Repository**: Source of the dataset (`diabetes+130-us+hospitals+for+years+1999-2008`).

## Project Structure
- `sample.ipynb`: The main Jupyter notebook containing exploratory data analysis and clustering logic.
- `data/`: Contains the raw dataset and mapping files.
    - `diabetic_data.csv`: The primary dataset with ~101k instances and 50 features.
    - `IDS_mapping.csv`: Mapping for categorical identifiers (admission type, discharge disposition, etc.).
- `models/`: Intended directory for storing trained clustering models (currently empty).
- `requirements.txt`: Minimal dependency list (currently only contains `ucimlrepo`).

## Building and Running
### Prerequisites
- Python 3.12+ (as inferred from the `.venv` path in notebook errors).
- Virtual environment (`.venv` is already present in the workspace).

### Installation
1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # Note: Additional dependencies like pandas and ydata-profiling may be required.
   pip install pandas ydata-profiling
   ```

### Execution
The analysis is primarily performed within the `sample.ipynb` notebook. Use a Jupyter-compatible editor (like PyCharm, VS Code, or Jupyter Lab) to run the cells.

## Development Conventions
- **Exploratory Workflow**: Heavy use of notebooks for data exploration and visualization.
- **Data Privacy**: The dataset is anonymized, but standard healthcare data handling practices should be followed if expanded.
- **Validation Strategy**: Use "hidden" labels (`readmitted`) for external validation of clusters using metrics like Jaccard index or Range-in-range comparison.

## TODO / Future Work
- [ ] Fix dependency issues in `sample.ipynb` (specifically `pkg_resources` / `setuptools` related to `ydata-profiling`).
- [ ] Implement the clustering logic (K-Means, Hierarchical, etc.).
- [ ] Generate the EDA report in the `./report/` directory.
- [ ] Populate `requirements.txt` with all necessary libraries.
