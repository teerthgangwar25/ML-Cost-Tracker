# src/analysis/importance.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_shap_importance(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Trains a meta-model on the experiment history to predict the target metric (e.g., accuracy).
    Uses SHAP to explain which hyperparameters had the biggest impact.
    """
    # Ensure we have clean data
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    
    if len(df_clean) < 5:
        logging.warning("Not enough runs to train a reliable SHAP explainer.")
        return None, None, None

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    # Train a meta-model (a Random Forest predicting the accuracy of our actual models)
    meta_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    meta_model.fit(X, y)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(meta_model)
    
    # We add check_additivity=False to avoid a known bug in some versions of shap/sklearn
    shap_values = explainer.shap_values(X, check_additivity=False)

    # Calculate mean absolute SHAP values for global importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'Hyperparameter': feature_cols,
        'Mean_Absolute_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False)

    return meta_model, explainer, importance_df

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

    # Define features and target
    features = ['n_estimators', 'max_depth', 'min_samples_split']
    target = 'accuracy'

    print("\nTraining Meta-Model and calculating SHAP values...")
    model, explainer, importance_df = calculate_shap_importance(df, features, target)

    if importance_df is not None:
        print(f"\n🧠 HYPERPARAMETER IMPORTANCE (Predicting {target}):")
        print("Higher SHAP value = stronger impact on the model's performance.\n")
        print(importance_df.to_string(index=False))
        
        print("\nNext step: We will visualize these SHAP values dynamically in the Streamlit dashboard!")