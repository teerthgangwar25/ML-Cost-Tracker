# src/analysis/budget_optimizer.py
import pandas as pd
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_best_model_under_budget(df: pd.DataFrame, max_budget_usd: float, cost_col: str = 'estimated_cost_usd', acc_col: str = 'accuracy') -> pd.Series:
    """
    Filters the experiment history for runs that cost less than or equal to 
    the max_budget_usd, and returns the single run with the highest accuracy.
    """
    if df.empty or cost_col not in df.columns or acc_col not in df.columns:
        logging.warning("Invalid dataframe or missing columns.")
        return None

    # 1. Filter out runs that are too expensive
    affordable_runs = df[df[cost_col] <= max_budget_usd]

    if affordable_runs.empty:
        logging.warning(f"No models found under the budget of ${max_budget_usd}")
        return None

    # 2. Sort the affordable runs by accuracy (highest first)
    best_runs = affordable_runs.sort_values(by=acc_col, ascending=False)

    # 3. Return the top row (the best model we can afford)
    best_model = best_runs.iloc[0]
    
    return best_model

if __name__ == "__main__":
    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    # Load the clean features
    data_path = project_root / "data" / "processed" / "runs_features.parquet"
    
    if not data_path.exists():
        logging.error("Data file not found. Run feature_builder.py first.")
        sys.exit(1)

    print("Loading processed features...")
    df = pd.read_parquet(data_path)

    # Let's set a strict hypothetical budget!
    # Looking at your Pareto front, let's cap it at a very tiny number to see what it picks.
    BUDGET_CAP = 0.0000010  # Less than 1 micro-dollar!

    print(f"\nSearching for the best model under a strict budget of ${BUDGET_CAP:f}...")
    
    best_run = get_best_model_under_budget(df, max_budget_usd=BUDGET_CAP)

    if best_run is not None:
        print("\n💰 BUDGET OPTIMIZER WINNER:")
        print(f"Run ID:            {best_run['run_id']}")
        print(f"Cost:              ${best_run['estimated_cost_usd']:.8f}")
        print(f"Accuracy Achieved: {best_run['accuracy']:.4f}")
        print("\nHyperparameters used:")
        print(f" - n_estimators:      {best_run['n_estimators']}")
        print(f" - max_depth:         {best_run['max_depth']}")
        print(f" - min_samples_split: {best_run['min_samples_split']}")