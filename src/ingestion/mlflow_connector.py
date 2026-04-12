# src/ingestion/mlflow_connector.py
import mlflow
import pandas as pd
import logging
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_mlflow_runs(tracking_uri: str, experiment_names: list) -> pd.DataFrame:
    """
    Connects to the local MLflow tracking server and pulls all runs 
    for the specified experiments into a Pandas DataFrame.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    
    experiment_ids = []
    for name in experiment_names:
        exp = client.get_experiment_by_name(name)
        if exp:
            experiment_ids.append(exp.experiment_id)
        else:
            logging.warning(f"Experiment '{name}' not found.")
            
    if not experiment_ids:
        logging.error("No valid experiment IDs found. Exiting.")
        return pd.DataFrame()

    logging.info(f"Fetching runs for experiment IDs: {experiment_ids}")
    
    # search_runs returns a list of Run objects. 
    # MLflow conveniently has a built-in search_runs that returns a DataFrame natively!
    runs_df = mlflow.search_runs(experiment_ids=experiment_ids)
    
    if runs_df.empty:
        logging.warning("No runs found for the specified experiments.")
        return runs_df
        
    logging.info(f"Successfully fetched {len(runs_df)} runs.")
    return runs_df

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    TRACKING_URI = f"sqlite:///{(project_root / 'mlflow_tracking.db').as_posix()}" 
    EXPERIMENTS = ["Cost_Tracker_Demo"]
    
    df = fetch_mlflow_runs(TRACKING_URI, EXPERIMENTS)
    
    if not df.empty:
        print("\nConnector test successful! Here is a snapshot of the raw data:")
        cols_to_show = [col for col in df.columns if 'metrics' in col or 'params' in col or 'tags' in col]
        print(df[['run_id', 'status'] + cols_to_show[:3]].head())
        