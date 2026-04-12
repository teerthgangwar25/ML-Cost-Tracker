import sys
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import uvicorn

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.analysis.importance import calculate_shap_importance
from src.analysis.pareto import calculate_pareto_front
from src.processing.cost_calculator import load_cost_config, calculate_run_costs
from src.processing.feature_builder import build_features
from src.ingestion.optuna_connector import fetch_optuna_runs, get_study_summary

app = FastAPI(title="ML Cost Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURES_PATH = project_root / "data" / "processed" / "runs_features.parquet"
PARAMS_FILE = project_root / "params.yaml"
OPTUNA_DB = project_root / "optuna.db"
OPTUNA_URI = f"sqlite:///{OPTUNA_DB.as_posix()}"
STUDY_NAME = "rf_breast_cancer_tpe"

@app.get("/api/config")
def get_config():
    rates = load_cost_config(str(PARAMS_FILE))
    instance_types = list(rates.keys()) if rates else ["local_cpu"]
    return {
        "status": "success",
        "data": {
            "instance_types": instance_types
        }
    }

def load_optuna_data(instance_type):
    if not OPTUNA_DB.exists(): return pd.DataFrame()
    df_raw = fetch_optuna_runs(OPTUNA_URI, STUDY_NAME)
    if df_raw.empty: return pd.DataFrame()
    df_cost = calculate_run_costs(df_raw, instance_type, str(PARAMS_FILE))
    df_features = build_features(df_cost)
    return calculate_pareto_front(df_features)

@app.get("/api/dashboard-data")
def get_dashboard_data(source: str = "MLflow (Grid Search)", instance_type: str = "local_cpu"):
    try:
        df_mlflow = pd.DataFrame()
        df_optuna = pd.DataFrame()

        if source in ["MLflow (Grid Search)", "Both Combined"]:
            if FEATURES_PATH.exists():
                df_mlflow = pd.read_parquet(FEATURES_PATH)
                if 'source' not in df_mlflow.columns:
                    df_mlflow['source'] = 'MLflow (Grid)'
                df_mlflow = calculate_pareto_front(df_mlflow)

        if source in ["Optuna (TPE)", "Both Combined"] and OPTUNA_DB.exists():
            df_optuna = load_optuna_data(instance_type)

        if source == "Both Combined":
            if not df_mlflow.empty and not df_optuna.empty:
                df = pd.concat([df_mlflow, df_optuna], ignore_index=True)
                df = calculate_pareto_front(df)
            elif not df_mlflow.empty: df = df_mlflow
            else: df = df_optuna
        elif source == "Optuna (TPE)":
            df = df_optuna
        else:
            df = df_mlflow

        if df.empty:
            return {"status": "error", "message": "No data found for the selected source"}

        # Clean NaN to None for JSON
        df_clean = df.replace({float('nan'): None})
        runs = df_clean.to_dict(orient="records")

        # Calculate SHAP optionally
        param_cols = ['n_estimators', 'max_depth', 'min_samples_split']
        actual_cols = [c for c in param_cols if c in df.columns]
        
        _, _, importance_df = calculate_shap_importance(df, actual_cols, 'accuracy')
        shap_data = importance_df.to_dict(orient="records") if importance_df is not None else []

        summary = {}
        if OPTUNA_DB.exists() and source in ["Optuna (TPE)", "Both Combined"]:
            summary = get_study_summary(OPTUNA_URI, STUDY_NAME)

        return {
            "status": "success",
            "data": {
                "runs": runs,
                "shap_importance": shap_data,
                "optuna_summary": summary
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
