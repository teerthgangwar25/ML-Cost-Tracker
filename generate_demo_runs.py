# generate_demo_runs.py
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import time
import os

def generate_runs():
    # Keep it local and clean using SQLite
    mlflow.set_tracking_uri("sqlite:///mlflow_tracking.db")
    mlflow.set_experiment("Cost_Tracker_Demo")

    print("Generating demo MLflow runs...")
    
    # Using breast cancer dataset for variety
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Grid search simulation
    for n_estimators in [10, 50, 100, 200, 300]:
        for max_depth in [2, 4, 6, 10, None]:
            for min_samples_split in [2, 5, 10]:
                with mlflow.start_run():
                    # Log parameters
                    mlflow.log_param("n_estimators", n_estimators)
                    mlflow.log_param("max_depth", max_depth)
                    mlflow.log_param("min_samples_split", min_samples_split)
                    
                    start_time = time.time()
                    
                    # Train model
                    clf = RandomForestClassifier(
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42
                    )
                    clf.fit(X_train, y_train)
                    
                    # Calculate duration and metrics
                    duration = time.time() - start_time
                    acc = clf.score(X_test, y_test)
                    
                    # Log metrics
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("train_duration_sec", duration)
                    
    print("Successfully generated demo runs! You can view them by running: mlflow ui --backend-store-uri sqlite:///mlflow_tracking.db")

if __name__ == "__main__":
    generate_runs()