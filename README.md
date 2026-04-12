# ML Experiment Cost Tracker

A local, completely free ML portfolio project that connects to MLflow and Optuna, tracking your machine learning experiments to analyze the tradeoff between model accuracy and compute cost.

It features a built-in Pareto-front engine to find optimal runs, SHAP analysis to determine hyperparameter importance, and a dashboard for a comprehensive overview of experiment ROI. 

## Features

- **Local Strategy:** Runs fully on your local machine using SQLite tracking databases (for MLflow and Optuna).
- **Cost Estimation Mapping:** Uses a `params.yaml` rate map to estimate compute costs dynamically per experiment.
- **Data Engineering:** Features are engineered into columnar Apache Parquet files using a fully DVC-versioned pipeline.
- **Pareto Efficiency:** Scans the cost/accuracy space to find strictly optimal models, discarding expensive but underperforming runs.
- **Dual Dashboards:** Visualize convergence, explore hyperparameter importance (SHAP), restrict budgets, and review optimal selections via two distinct user interfaces (Streamlit or React). 

## Tech Stack

- **Experiment Tracking:** MLflow, Optuna
- **ML / Analysis:** Scikit-Learn, SHAP
- **Data Munging:** Pandas, PyArrow, DVC
- **Dashboards:** Streamlit, React, Vite, FastAPI

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
5. **Launch a Dashboard:**
   This project features two distinct dashboards for you to interact with your data. For full architecture details and start-up instructions, please see the dedicated documentation in the `dashboard_docs/` folder:
   
   * **[Streamlit Dashboard](dashboard_docs/streamlit/README.md)**: The classic Python-native UI built with Streamlit and Plotly.
   * **[React / FastAPI Dashboard](dashboard_docs/react/README.md)**: A modern decoupled web app featuring a Vite+React frontend and FastAPI backend.

   To quickly launch either of them, refer to their instruction manuals:
   - [👉 Run Streamlit Dashboard](dashboard_docs/streamlit/INSTRUCTIONS.md)
   - [👉 Run React Dashboard](dashboard_docs/react/INSTRUCTIONS.md)

## Development and Architecture
- `src/ingestion`: Connectors for MLFlow tracking database and Optuna registry.
- `src/processing`: Extraction and data preparation routines (`feature_builder.py`, `cost_calculator.py`).
- `src/analysis`: Mathematical and evaluation engines (SHAP importance modelling, Pareto optimization).
- `src/dashboard`: The unified alternative presentation layer driven by Streamlit.
- `src/api` & `frontend`: The decoupled presentation layer powered by a FastAPI backend and a React/Vite frontend.
- `dashboard_docs`: Documentation and instructions detailing both presentation layers.

This project is intended strictly as a localhost portfolio piece to demonstrate rigorous feature engineering, software engineering skills, and machine learning insight.
