# src/processing/feature_builder.py
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_features(df_cost: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the MLflow dataframe, normalizes column names, 
    and handles data types for ML processing.
    """
    if df_cost.empty:
        logging.warning("Empty DataFrame provided to feature builder.")
        return pd.DataFrame()

    df = df_cost.copy()

    # 1. Filter out failed runs
    if 'status' in df.columns:
        df = df[df['status'] == 'FINISHED'].copy()

    # 2. Extract and rename key columns
    # We strip the 'metrics.' and 'params.' prefixes for cleaner code downstream
    rename_map = {}
    for col in df.columns:
        if col.startswith('metrics.'):
            rename_map[col] = col.replace('metrics.', '')
        elif col.startswith('params.'):
            rename_map[col] = col.replace('params.', '')
            
    df = df.rename(columns=rename_map)

    # 3. Define the core columns we care about for the dashboard/analysis
    # 'source' and 'trial_number' are optional — only present for Optuna data
    # 'min_samples_leaf' is an extra param logged by generate_optuna_runs.py
    core_cols = [
        'run_id', 'start_time', 'duration_hours', 'instance_type',
        'estimated_cost_usd', 'accuracy', 'n_estimators',
        'max_depth', 'min_samples_split', 'min_samples_leaf',
        'source', 'trial_number'
    ]
    
    # Only keep columns that actually exist in the dataframe
    final_cols = [c for c in core_cols if c in df.columns]
    df_clean = df[final_cols].copy()

    # 4. Fix Data Types
    # MLflow logs all params as strings. We must convert them to numeric for the ML engine.
    if 'n_estimators' in df_clean.columns:
        df_clean['n_estimators'] = pd.to_numeric(df_clean['n_estimators'], errors='coerce')
        
    if 'min_samples_split' in df_clean.columns:
        df_clean['min_samples_split'] = pd.to_numeric(df_clean['min_samples_split'], errors='coerce')

    if 'max_depth' in df_clean.columns:
        # Handle the "None" string representing no max depth in Random Forest
        df_clean['max_depth'] = df_clean['max_depth'].replace('None', np.nan)
        df_clean['max_depth'] = pd.to_numeric(df_clean['max_depth'], errors='coerce')
        # Fill NaN with a high number (e.g., 99) so the ML explainer can process it
        df_clean['max_depth'] = df_clean['max_depth'].fillna(99).astype(int)

    if 'min_samples_leaf' in df_clean.columns:
        df_clean['min_samples_leaf'] = pd.to_numeric(df_clean['min_samples_leaf'], errors='coerce')

    # Fill missing source (MLflow grid search data won't have this column)
    if 'source' not in df_clean.columns:
        df_clean['source'] = 'MLflow (Grid)'
    else:
        df_clean['source'] = df_clean['source'].fillna('MLflow (Grid)')

    logging.info(f"Feature engineering complete. Final shape: {df_clean.shape}")
    return df_clean


if __name__ == "__main__":
    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))
    
    from src.ingestion.mlflow_connector import fetch_mlflow_runs
    from src.processing.cost_calculator import calculate_run_costs
    
    # 1. Fetch
    tracking_uri = f"sqlite:///{(project_root / 'mlflow_tracking.db').as_posix()}"
    df_raw = fetch_mlflow_runs(tracking_uri, ["Cost_Tracker_Demo"])
    
    # 2. Cost Calculation
    config_file = str(project_root / "params.yaml")
    df_cost = calculate_run_costs(df_raw, "local_cpu", config_file)
    
    # 3. Build Features
    print("\nBuilding ML-ready features...")
    df_features = build_features(df_cost)
    
    if not df_features.empty:
        # 4. Save to Parquet
        output_dir = project_root / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        output_path = output_dir / "runs_features.parquet"
        
        # PyArrow is required for this, which we installed earlier
        df_features.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"\nSuccess! Cleaned data saved to: {output_path}")
        print("\nSnapshot of ML-ready data:")
        print(df_features.head())