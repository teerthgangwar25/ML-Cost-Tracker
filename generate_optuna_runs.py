# generate_optuna_runs.py
"""
Runs an Optuna TPE (Tree-structured Parzen Estimator) hyperparameter search
on the Breast Cancer dataset.

Unlike generate_demo_runs.py (which does a fixed grid search), Optuna
intelligently learns from previous trials and focuses on the most promising
regions of the hyperparameter space.

Each trial is ALSO mirrored to MLflow so both systems share the same data.

Usage:
    python generate_optuna_runs.py
"""
import optuna
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import time
import logging
import warnings

# Suppress verbose output for cleaner logs
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ─── Config ───────────────────────────────────────────────────────────────────
MLFLOW_URI  = "sqlite:///mlflow_tracking.db"
OPTUNA_DB   = "sqlite:///optuna.db"
STUDY_NAME  = "rf_breast_cancer_tpe"
N_TRIALS    = 50   # Optuna will run 50 intelligent trials

# Load data once at module level (efficiency)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# ──────────────────────────────────────────────────────────────────────────────


def objective(trial: optuna.trial.Trial) -> float:
    """
    Optuna objective function.

    Optuna calls this function repeatedly, each time suggesting different
    hyperparameter values based on what it learned from previous trials.
    The TPE sampler focuses more trials near the hyperparameter combinations
    that previously gave the best accuracy.

    Returns:
        float: Validation accuracy (Optuna maximizes this)
    """
    # --- Optuna suggests hyperparameters (TPE guided, not random grid) ---
    n_estimators      = trial.suggest_int("n_estimators",      10,  300)
    max_depth         = trial.suggest_int("max_depth",          2,   20)
    min_samples_split = trial.suggest_int("min_samples_split",  2,   20)
    min_samples_leaf  = trial.suggest_int("min_samples_leaf",   1,   10)

    # --- Mirror each trial to MLflow for unified tracking ---
    with mlflow.start_run():
        mlflow.log_param("n_estimators",      n_estimators)
        mlflow.log_param("max_depth",         max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf",  min_samples_leaf)

        # Tag so we can distinguish Optuna trials from grid search in the dashboard
        mlflow.set_tag("source",     "optuna_tpe")
        mlflow.set_tag("study_name", STUDY_NAME)
        mlflow.set_tag("trial_number", str(trial.number))

        start = time.time()
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        duration = time.time() - start

        acc = clf.score(X_test, y_test)
        mlflow.log_metric("accuracy",          acc)
        mlflow.log_metric("train_duration_sec", duration)

    return acc


def run_optuna_study():
    """Creates (or resumes) the Optuna study and runs N_TRIALS trials."""

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Cost_Tracker_Optuna")

    logging.info(f"Starting Optuna study: '{STUDY_NAME}' — {N_TRIALS} trials")
    logging.info(f"Optuna DB:   {OPTUNA_DB}")
    logging.info(f"MLflow URI:  {MLFLOW_URI}")

    # load_if_exists=True means you can run this script multiple times
    # to add more trials without losing previous results
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=OPTUNA_DB,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print(f"\nRunning {N_TRIALS} TPE trials (this may take 1-2 minutes)...\n")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ✅  OPTUNA STUDY COMPLETE")
    print("="*60)
    print(f"  Total trials run  : {len(study.trials)}")
    print(f"  Best accuracy     : {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"      {k}: {v}")
    print("="*60)
    print(f"\n  Data saved to   : {OPTUNA_DB}")
    print(f"  MLflow experiment: Cost_Tracker_Optuna")
    print("\n  Run the dashboard:")
    print("  streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    run_optuna_study()
