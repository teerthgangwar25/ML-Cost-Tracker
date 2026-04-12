# src/analysis/pareto.py
import pandas as pd
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_pareto_front(df: pd.DataFrame, cost_col: str = 'estimated_cost_usd', acc_col: str = 'accuracy') -> pd.DataFrame:
    """
    Identifies the Pareto front of ML experiments.
    A run is on the Pareto front if no other run is strictly better 
    (i.e., both cheaper AND more accurate).
    """
    if df.empty or cost_col not in df.columns or acc_col not in df.columns:
        logging.warning("Invalid dataframe or missing columns for Pareto analysis.")
        return df

    # 1. Sort by Cost (Ascending) and then Accuracy (Descending)
    # If two runs cost the exact same, the more accurate one is processed first.
    df_sorted = df.sort_values(by=[cost_col, acc_col], ascending=[True, False]).reset_index(drop=True)

    pareto_front = []
    max_accuracy_so_far = -1.0

    # 2. Iterate and find dominant runs
    for index, row in df_sorted.iterrows():
        # Since we sorted by cost ascending, every new row is more expensive than the last.
        # Therefore, to justify the higher cost, its accuracy MUST be higher than anything 
        # we've seen so far. If it is, it's on the Pareto front!
        if row[acc_col] > max_accuracy_so_far:
            pareto_front.append(True)
            max_accuracy_so_far = row[acc_col]
        else:
            # It costs more but offers worse or equal accuracy. It is dominated.
            pareto_front.append(False)

    # 3. Add the flag to the dataframe
    df_sorted['is_pareto_optimal'] = pareto_front

    logging.info(f"Identified {sum(pareto_front)} Pareto optimal runs out of {len(df)} total runs.")
    return df_sorted

if __name__ == "__main__":
    # Setup paths
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))

    # 1. Load the clean features we built in Step 4
    data_path = project_root / "data" / "processed" / "runs_features.parquet"
    
    if not data_path.exists():
        logging.error(f"Data file not found at {data_path}. Run feature_builder.py first.")
        sys.exit(1)

    print("Loading processed features...")
    df = pd.read_parquet(data_path)

    # 2. Calculate the Pareto front
    print("\nCalculating Pareto front (Cost vs Accuracy)...")
    df_pareto = calculate_pareto_front(df)

    # 3. Display the absolute best runs for the money!
    print("\n👑 THE PARETO FRONT (Best runs for the money):")
    pareto_winners = df_pareto[df_pareto['is_pareto_optimal'] == True]
    
    cols_to_show = ['estimated_cost_usd', 'accuracy', 'n_estimators', 'max_depth', 'min_samples_split']
    print(pareto_winners[cols_to_show].to_string(index=False))

    # 4. Save the result back to the processed folder for the dashboard
    output_path = project_root / "data" / "processed" / "runs_pareto.parquet"
    df_pareto.to_parquet(output_path, engine='pyarrow', index=False)
    print(f"\nPareto data saved to: {output_path}")