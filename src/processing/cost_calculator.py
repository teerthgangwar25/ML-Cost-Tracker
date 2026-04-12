# src/processing/cost_calculator.py
import pandas as pd
import yaml
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_cost_config(config_path: str) -> dict:
    """Loads instance hourly rates from the YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config.get("instance_costs", {})
    except FileNotFoundError:
        logging.error(f"Config file not found at {config_path}")
        return {}

def calculate_run_costs(df: pd.DataFrame, instance_type: str, config_path: str) -> pd.DataFrame:
    """
    Takes the raw MLflow dataframe, calculates duration in hours, 
    and multiplies by the instance rate to get estimated cost.
    """
    if df.empty:
        return df
        
    rates = load_cost_config(config_path)
    if instance_type not in rates:
        logging.warning(f"Instance type '{instance_type}' not found. Defaulting rate to $0.0/hr")
        rate_per_hour = 0.0
    else:
        rate_per_hour = rates[instance_type]

    df_cost = df.copy()
    
    # MLflow automatically tracks 'start_time' and 'end_time' natively as datetime objects
    if 'start_time' in df_cost.columns and 'end_time' in df_cost.columns:
        # Calculate duration in seconds, then convert to hours
        duration_sec = (df_cost['end_time'] - df_cost['start_time']).dt.total_seconds()
        duration_hours = duration_sec / 3600.0
    elif 'metrics.train_duration_sec' in df_cost.columns:
        # Fallback to the explicit metric we logged in the demo script
        duration_hours = df_cost['metrics.train_duration_sec'] / 3600.0
    else:
        logging.warning("No time columns found to calculate duration. Setting cost to 0.")
        duration_hours = 0.0
        
    # Append the new calculated columns
    df_cost['duration_hours'] = duration_hours
    df_cost['estimated_cost_usd'] = duration_hours * rate_per_hour
    df_cost['instance_type'] = instance_type
    
    return df_cost

if __name__ == "__main__":
    # Add root directory to path so we can import the ingestion module
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))
    
    from src.ingestion.mlflow_connector import fetch_mlflow_runs
    
    # 1. Fetch the raw data
    db_path = project_root / "mlflow_tracking.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    
    print("Fetching raw data from MLflow...")
    df_raw = fetch_mlflow_runs(tracking_uri, ["Cost_Tracker_Demo"])
    
    # 2. Calculate costs
    if not df_raw.empty:
        config_file = project_root / "params.yaml"
        print(f"\nCalculating costs using the 'aws_p3_2xlarge' rate to simulate cloud spend...")
        
        df_processed = calculate_run_costs(
            df=df_raw, 
            instance_type="aws_p3_2xlarge", 
            config_path=str(config_file)
        )
        
        print("\nCost Calculation Successful! Here is a snapshot of the processing:")
        cols_to_show = ['run_id', 'duration_hours', 'instance_type', 'estimated_cost_usd']
        print(df_processed[cols_to_show].head())