# ML Experiment Cost Tracker

A local, completely free ML portfolio project that connects to MLflow and Optuna, tracking your machine learning experiments to analyze the tradeoff between model accuracy and compute cost.

It features a built-in Pareto-front engine to find optimal runs, SHAP analysis to determine hyperparameter importance, and a dashboard for a comprehensive overview of experiment ROI. 

## Features

- **Local Strategy:** Runs fully on your local machine using SQLite tracking databases (for MLflow and Optuna) and Streamlit out-of-the-box.
- **Cost Estimation Mapping:** Uses a `params.yaml` rate map to estimate compute costs dynamically per experiment.
- **Data Engineering:** Features are engineered into columnar Apache Parquet files using a fully DVC-versioned pipeline.
- **Pareto Efficiency:** Scans the cost/accuracy space to find strictly optimal models, discarding expensive but underperforming runs.
- **ROI Dashboard:** Streamlit UI allows you to visualize convergence, explore hyperparameter importance (SHAP), restrict budgets, and review optimal selections. 

## Tech Stack

- **Experiment Tracking:** MLflow, Optuna
- **ML / Analysis:** Scikit-Learn, SHAP
- **Data Munging:** Pandas, PyArrow, DVC
- **Dashboard:** Streamlit, Plotly

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ml-cost-tracker
   ```
2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Generate Demo Data:**
   Generate artificial runs using MLflow tracking and Optuna TPE (Tree-structured Parzen Estimator) trials:
   ```bash
   python generate_demo_runs.py
   python generate_optuna_runs.py
   ```
4. **Run the DVC pipeline:**
   This processes the database tracking records, calculates cost constraints, applies feature extraction to the trials, computes the Pareto front, and exports artifacts to `data/processed`:
   ```bash
   dvc repro
   ```
5. **Launch the Dashboard:**
   We have two flavors of the dashboard available for you to analyze your runs. Please see their respective instruction manuals for architecture details and how to run them:
   - **[Streamlit Dashboard (Default)](dashboard_docs/streamlit/INSTRUCTIONS.md)**
   - **[React / FastAPI Dashboard](dashboard_docs/react/INSTRUCTIONS.md)**

## Development and Architecture
- `src/ingestion`: Connectors for MLFlow tracking database and Optuna registry.
- `src/processing`: Extraction and data preparation routines (`feature_builder.py`, `cost_calculator.py`).
- `src/analysis`: Mathematical and evaluation engines (SHAP importance modelling, Pareto optimization).
- `src/dashboard`: Presentation layer driven by Streamlit.

This project is intended strictly as a localhost portfolio piece to demonstrate rigorous feature engineering, software engineering skills, and machine learning insight.
