# ML Experiment Cost Tracker — Project Brief

## What we are building

A **local, fully free** ML portfolio project called the **ML Experiment Cost Tracker**.

It connects to MLflow and Optuna, visualizing the cost vs. accuracy tradeoffs across all your
experiments. The core ML insight is a **Pareto-front analysis** — surfacing the runs
where no other run is both cheaper AND more accurate. It also uses SHAP to explain
which hyperparameters drove efficiency.

This runs 100% locally. No paid APIs, no cloud storage, no external data purchases.
It is designed to be a professional portfolio piece for a Data Science / ML Engineer role.

---

## Project constraints (non-negotiable)

- Free only — no paid services, no cloud billing APIs
- Runs entirely on localhost
- Data is self-generated (your own MLflow or Optuna experiment runs)
- Storage is local SQLite (`mlflow_tracking.db`, `optuna.db`) + Parquet files
- **Architecture**: FastAPI backend serving a modern React + Vite + TailwindCSS frontend.

---

## Full folder structure

```text
ml-cost-tracker/
├── data/
│   ├── raw/              # MLflow exports, any CSVs
│   ├── processed/        # Feature-engineered Parquet files
│   └── .dvc/             # DVC tracking metadata
├── frontend/             # React + Vite + Tailwind CSS Frontend
│   ├── src/              # Frontend React components and logic
│   ├── package.json      # Node dependencies
│   └── vite.config.ts    # Vite configuration
├── src/
│   ├── api/
│   │   └── main.py                 # FastAPI backend server
│   ├── ingestion/
│   │   ├── mlflow_connector.py     # Pulls runs via MLflow Python API
│   │   └── optuna_connector.py     # Reads Optuna study DB
│   ├── processing/
│   │   ├── cost_calculator.py      # duration × rate = estimated cost
│   │   └── feature_builder.py      # Builds ML-ready dataframe
│   ├── analysis/
│   │   ├── pareto.py               # Pareto-front algorithm (THE core ML logic)
│   │   ├── importance.py           # Random Forest + SHAP on hyperparams
│   │   └── budget_optimizer.py     # Given cost cap → best accuracy achievable
│   └── dashboard/
│       └── app.py                  # Legacy Streamlit entry point (Deprecated)
├── notebooks/
│   └── eda.ipynb                   # Exploratory analysis
├── dvc.yaml                        # DVC pipeline definition
├── params.yaml                     # Instance cost rates (you maintain this)
├── generate_demo_runs.py           # Script to generate MLflow runs
├── generate_optuna_runs.py         # Script to generate Optuna runs
└── requirements.txt
```

---

## Architecture layers (top to bottom)

### Layer 1 — Data ingestion
- MLflow connector → pulls run ID, start/end time, params, metrics, tags
- Optuna connector → reads hyperparameter trial history from `optuna.db`
- Cost config → `params.yaml` file with manual $/hr estimates

### Layer 2 — Data processing
- Cost calculator → `cost = (end_time - start_time) in hours × instance_rate`
- Feature builder → produces a clean Pandas dataframe saved as Parquet (`runs_features.parquet`)
- DVC versioning → tracks processed datasets as versioned artifacts (local remote)

### Layer 3 — ML analysis core (the brain of the project)
- **Pareto-front engine** → finds runs where no other run dominates on both cost and accuracy
- **Hyperparam importance** → trains a Random Forest on runs, applies SHAP to rank params
- **Budget optimizer** → given a max cost, returns the best model achievable within it

### Layer 4 — Storage & Backend
- SQLite → local store for MLflow (`mlflow_tracking.db`) and Optuna (`optuna.db`)
- Parquet files → fast columnar reads for dashboard queries
- **FastAPI** (`src/api/main.py`) → REST API serving the data, Pareto front, and SHAP calculations to the frontend.

### Layer 5 — React + Vite Dashboard
- Cost vs accuracy scatter plot (each dot = one run, Pareto front highlighted)
- Optimal run card (best run + its hyperparams + why it won)
- SHAP bar chart (which hyperparams drove efficiency)
- Budget mode (filter best model within budget)
- Optuna convergence visualization

---

## Tech stack

| Purpose              | Library / Tool                      |
|----------------------|-------------------------------------|
| Experiment tracking  | MLflow, Optuna                      |
| ML / Pareto analysis | Scikit-Learn, SHAP                  |
| Data processing      | Pandas, PyArrow                     |
| Local database       | SQLite                              |
| Data versioning      | DVC (local remote)                  |
| Backend API          | FastAPI, Uvicorn                    |
| Frontend Dashboard   | React, Vite, TailwindCSS, Recharts  |
| Config               | PyYAML                              |

---

## Running the Project

### 1. Generate Demo Data (Optional)
If you don't have existing runs, populate your local databases:
```bash
python generate_demo_runs.py
python generate_optuna_runs.py
```

### 2. Process Features
```bash
python src/processing/feature_builder.py
```

### 3. Start the FastAPI Backend
```bash
python src/api/main.py
```
*(The API will run at `http://127.0.0.1:8000`)*

### 4. Start the React Frontend
Open a new terminal and run:
```bash
cd frontend
npm install
npm run dev
```
*(The dashboard will be available at `http://localhost:5173`)*
