# src/ingestion/optuna_connector.py
"""
Reads a completed Optuna study from its SQLite database and returns a
normalized Pandas DataFrame that is schema-compatible with mlflow_connector.py.

This means the output can flow through the SAME cost_calculator → feature_builder
pipeline without any modifications — Optuna data and MLflow data are interchangeable
downstream.

Schema output (mirrors mlflow_connector output):
    - run_id                    : str  ("optuna_<trial_number>")
    - status                    : str  ("FINISHED")
    - start_time                : datetime
    - end_time                  : datetime
    - metrics.accuracy          : float
    - metrics.train_duration_sec: float
    - params.n_estimators       : int (as string, to match MLflow behavior)
    - params.max_depth          : int (as string)
    - params.min_samples_split  : int (as string)
    - params.min_samples_leaf   : int (as string)
    - source                    : str  ("Optuna (TPE)") — extra tag for dashboard
    - trial_number              : int  — for convergence plotting
"""
import optuna
import pandas as pd
import logging
from pathlib import Path
import sys

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_optuna_runs(storage_uri: str, study_name: str) -> pd.DataFrame:
    """
    Connects to an Optuna SQLite database, loads the specified study,
    and returns a normalized DataFrame compatible with the MLflow pipeline.

    Args:
        storage_uri : SQLite URI, e.g. "sqlite:///optuna.db" or absolute path
        study_name  : Name of the Optuna study to load

    Returns:
        pd.DataFrame: Normalized trial records (COMPLETE trials only)
                      Empty DataFrame if study not found or no complete trials.
    """
    # ─── Load Study ───────────────────────────────────────────────────────────
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_uri)
    except KeyError:
        logging.error(f"Study '{study_name}' not found in {storage_uri}. "
                      f"Run generate_optuna_runs.py first.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to load Optuna study: {e}")
        return pd.DataFrame()

    # ─── Get Trial DataFrame ──────────────────────────────────────────────────
    df_trials = study.trials_dataframe()

    if df_trials.empty:
        logging.warning("Optuna study exists but has no trials yet.")
        return pd.DataFrame()

    # Keep only successfully completed trials (ignore PRUNED, FAIL)
    df_complete = df_trials[df_trials['state'] == 'COMPLETE'].copy()

    if df_complete.empty:
        logging.warning("No COMPLETE trials found in study.")
        return pd.DataFrame()

    total   = len(df_trials)
    complete = len(df_complete)
    logging.info(f"Loaded {complete} COMPLETE trials out of {total} total from study '{study_name}'.")

    # ─── Normalize to MLflow-compatible Schema ────────────────────────────────
    normalized = pd.DataFrame()

    # Unique run identifier (prefix avoids collision with MLflow run IDs)
    normalized['run_id'] = "optuna_" + df_complete['number'].astype(str)

    # Status — map to MLflow convention so feature_builder filters correctly
    normalized['status'] = 'FINISHED'

    # Timestamps — Optuna stores these as datetime_start / datetime_complete
    normalized['start_time'] = pd.to_datetime(df_complete['datetime_start'])
    normalized['end_time']   = pd.to_datetime(df_complete['datetime_complete'])

    # Metrics — add 'metrics.' prefix so feature_builder.py strips it uniformly
    normalized['metrics.accuracy']            = df_complete['value']
    normalized['metrics.train_duration_sec']  = (
        df_complete['duration'].dt.total_seconds()
    )

    # Parameters — stored as 'params_<name>' in Optuna trials dataframe
    # Convert to string to match MLflow behavior (MLflow logs all params as strings)
    param_map = {
        'params_n_estimators':      'params.n_estimators',
        'params_max_depth':         'params.max_depth',
        'params_min_samples_split': 'params.min_samples_split',
        'params_min_samples_leaf':  'params.min_samples_leaf',
    }
    for optuna_col, mlflow_col in param_map.items():
        if optuna_col in df_complete.columns:
            # Convert to int first (Optuna stores as float), then to str like MLflow
            normalized[mlflow_col] = df_complete[optuna_col].astype(int).astype(str)

    # Extra metadata — preserved through feature_builder for the dashboard
    normalized['source']       = 'Optuna (TPE)'
    normalized['trial_number'] = df_complete['number'].values

    logging.info(f"Successfully normalized {len(normalized)} Optuna trials.")
    return normalized.reset_index(drop=True)


def get_study_summary(storage_uri: str, study_name: str) -> dict:
    """
    Returns a summary dict of the study: best value, best params, n_trials.
    Useful for the dashboard header card.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_uri)
        return {
            "study_name":  study_name,
            "n_trials":    len(study.trials),
            "best_value":  study.best_value,
            "best_params": study.best_params,
        }
    except Exception:
        return {}


# ─── Standalone Test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    db_path     = project_root / "optuna.db"
    storage_uri = f"sqlite:///{db_path.as_posix()}"
    study_name  = "rf_breast_cancer_tpe"

    print(f"Loading Optuna study from: {storage_uri}")
    df = fetch_optuna_runs(storage_uri, study_name)

    if not df.empty:
        print("\n✅  Optuna Connector test successful!")
        print(f"Shape: {df.shape}")
        print("\nColumn list:")
        for col in df.columns:
            print(f"  {col}")
        print("\nSnapshot (first 5 rows):")
        cols_to_show = [
            'run_id', 'metrics.accuracy', 'metrics.train_duration_sec',
            'params.n_estimators', 'params.max_depth', 'source', 'trial_number'
        ]
        print(df[cols_to_show].head().to_string(index=False))

        # Show study summary
        summary = get_study_summary(storage_uri, study_name)
        if summary:
            print(f"\n🏆 Study Best: accuracy={summary['best_value']:.4f}")
            print(f"   Best params: {summary['best_params']}")
