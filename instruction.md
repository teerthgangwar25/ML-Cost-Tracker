# ML Experiment Cost Tracker — Project Brief

## What we are building

A **local, fully free** ML portfolio project called the **ML Experiment Cost Tracker**.

It connects to MLflow and visualizes the cost vs. accuracy tradeoffs across all your
experiments. The core ML insight is a **Pareto-front analysis** — surfacing the runs
where no other run is both cheaper AND more accurate. It also uses SHAP to explain
which hyperparameters drove efficiency.

This runs 100% locally. No paid APIs, no cloud storage, no external data purchases.
It is designed to be a professional portfolio piece for a Data Science / ML Engineer role.

---

## Project constraints (non-negotiable)

- Free only — no paid services, no cloud billing APIs
- Runs entirely on localhost
- Data is self-generated (your own MLflow experiment runs)
- Storage is local SQLite + Parquet files
- Dashboard runs on Streamlit (localhost)/ changed to css and all

---

## Full folder structure

```
ml-cost-tracker/
├── data/
│   ├── raw/              # MLflow exports, any CSVs
│   ├── processed/        # Feature-engineered Parquet files
│   └── .dvc/             # DVC tracking metadata
├── src/
│   ├── ingestion/
│   │   ├── mlflow_connector.py     # Pulls runs via MLflow Python API
│   │   ├── optuna_connector.py     # Reads Optuna study DB
│   │   └── cost_config.yaml        # Instance type → $/hr mapping
│   ├── processing/
│   │   ├── run_parser.py           # Normalizes MLflow run schema
│   │   ├── cost_calculator.py      # duration × rate = estimated cost
│   │   └── feature_builder.py      # Builds ML-ready dataframe
│   ├── analysis/
│   │   ├── pareto.py               # Pareto-front algorithm (THE core ML logic)
│   │   ├── importance.py           # Random Forest + SHAP on hyperparams
│   │   └── budget_optimizer.py     # Given cost cap → best accuracy achievable
│   └── dashboard/
│       └── app.py                  # Streamlit entry point
├── notebooks/
│   └── eda.ipynb                   # Exploratory analysis
├── dvc.yaml                        # DVC pipeline definition
├── params.yaml                     # Instance cost rates (you maintain this)
└── requirements.txt
```

---

## Architecture layers (top to bottom)

### Layer 1 — Data ingestion
- MLflow connector → pulls run ID, start/end time, params, metrics, tags
- Optuna connector → reads hyperparameter trial history from optuna.db
- System metrics → psutil (CPU/RAM) + gputil (GPU memory, utilization)
- Cost config → a simple YAML file you write manually with $/hr estimates

### Layer 2 — Data processing
- Run parser → normalizes run schema across MLflow versions
- Cost calculator → `cost = (end_time - start_time) in hours × instance_rate`
- Feature builder → produces a clean Pandas dataframe: one row per run
- DVC versioning → tracks processed datasets as versioned artifacts (local remote)

### Layer 3 — ML analysis core (the brain of the project)
- **Pareto-front engine** → finds runs where no other run dominates on both cost and accuracy
- **Hyperparam importance** → trains a Random Forest on runs, applies SHAP to rank params
- **Budget optimizer** → given a max cost, returns the best model achievable within it
- **Trend analyzer** → cost-efficiency improvement over time as experiments evolved

### Layer 4 — Storage
- SQLite or DuckDB → local store for processed run records
- Parquet files → fast columnar reads for dashboard queries
- DVC with local folder as remote → no cloud needed

### Layer 5 — Streamlit dashboard
- Cost vs accuracy scatter plot (each dot = one run, Pareto front highlighted)
- Optimal run card (best run + its hyperparams + why it won)
- SHAP bar chart (which hyperparams drove efficiency)
- Budget mode (slider: set cost cap → see best model within budget)
- Run comparison table (side-by-side diff of any two runs)
- CSV/PDF export of full experiment summary

---

## Tech stack

| Purpose              | Library / Tool              |
|----------------------|-----------------------------|
| Experiment tracking  | MLflow                      |
| Hyperparameter data  | Optuna                      |
| ML / Pareto analysis | Scikit-Learn, SHAP          |
| Data processing      | Pandas, PyArrow             |
| Local database       | DuckDB or SQLite            |
| Data versioning      | DVC (local remote)          |
| System metrics       | psutil, gputil              |
| Dashboard            | Streamlit, Plotly           |
| Config               | PyYAML                      |

---

## Data sources (all free, all local)

### Primary: MLflow runs (already on your machine)
```python
import mlflow
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=["0"])
# Each run has: run.info, run.data.params, run.data.metrics
```
If you have no past runs yet, generate 20–30 demo runs with a sweep script
(vary lr, batch_size, n_estimators etc. on Iris / MNIST / any sklearn dataset).

### Compute cost: calculated, not fetched
```yaml
# params.yaml — you write this
instance_costs:
  local_cpu: 0.05       # $/hr estimate
  local_gpu: 0.20
  colab_free: 0.0
  aws_t3_medium: 0.052
  aws_p3_xlarge: 3.06
```
Cost formula: `cost = duration_hours × instance_rate`
Duration comes from `run.info.start_time` and `run.info.end_time` (already in MLflow).

### Optuna data (if using Optuna)
```python
import optuna
study = optuna.load_study(study_name="my_study", storage="sqlite:///optuna.db")
df = study.trials_dataframe()
```

### System metrics (optional, adds depth)
```python
import psutil, GPUtil
gpus = GPUtil.getGPUs()
mlflow.log_metric("gpu_mem_mb", gpus[0].memoryUsed)
mlflow.log_metric("cpu_pct", psutil.cpu_percent())
```

### Demo data generation script (run this if starting from scratch)
```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for n_estimators in [10, 50, 100, 200]:
    for max_depth in [2, 4, 6, None]:
        with mlflow.start_run():
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            start = time.time()
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            clf.fit(X_train, y_train)
            duration = time.time() - start
            acc = clf.score(X_test, y_test)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("train_duration_sec", duration)
```
This gives you 16 runs instantly. Run it a few times with different params for richer data.

---

## Build order (step by step)

1. **Generate demo data** — run the sweep script above, confirm runs appear in MLflow UI
2. **mlflow_connector.py** — pull all runs into a raw Pandas dataframe
3. **cost_calculator.py** — add estimated cost column using params.yaml rates
4. **feature_builder.py** — clean, normalize, save as Parquet
5. **pareto.py** — implement Pareto-front algorithm (most important file)
6. **importance.py** — Random Forest on hyperparams → SHAP values
7. **budget_optimizer.py** — filter runs by cost cap, return best accuracy
8. **app.py** — Streamlit dashboard wiring all analysis together
9. **DVC pipeline** — define dvc.yaml stages to make the pipeline reproducible
10. **Polish** — README, requirements.txt, demo GIF for portfolio

---

## Key ML concepts the interviewer will ask about

- **Pareto front**: A set of solutions where you cannot improve one objective (accuracy)
  without worsening another (cost). Your engine finds these runs automatically.
- **SHAP values**: Explain *why* a hyperparameter mattered — not just that it did.
  Uses TreeExplainer from the SHAP library on a Random Forest trained on run data.
- **DVC**: Treats your data pipeline like Git — each stage is reproducible and versioned.
  Local remote means no cloud spend while still being professional.
- **Cost estimation**: Honest approximation via duration × rate. Not exact, but
  directionally correct and transparent — that's the right engineering trade-off.

---

## What to tell the AI assistant in the new chat

Paste everything above. Then add:

> "Start with step 1 of the build order. Write me the demo data generation script
> first, then mlflow_connector.py. Keep everything local and free.
> Use the folder structure exactly as defined above."
