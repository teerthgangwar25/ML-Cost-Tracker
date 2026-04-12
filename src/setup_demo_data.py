import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time
import os

def generate_data():
    print("Loading Iris dataset...")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    mlflow.set_experiment("ml_cost_tracker_demo")

    n_estimators_list = [10, 50, 100, 200]
    max_depth_list = [2, 4, 6, None]
    
    total_runs = len(n_estimators_list) * len(max_depth_list)
    print(f"Starting {total_runs} demo runs in MLflow...")
    
    run_count = 0
    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            run_count += 1
            print(f"Executing run {run_count}/{total_runs} (estimators={n_estimators}, depth={max_depth})...")
            with mlflow.start_run():
                mlflow.log_param("n_estimators", n_estimators)
                # MLFlow requires string or numbers. None may be converted to "None"
                mlflow.log_param("max_depth", max_depth if max_depth is not None else -1)
                
                start = time.time()
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                clf.fit(X_train, y_train)
                duration = time.time() - start
                
                acc = clf.score(X_test, y_test)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("train_duration_sec", duration)

    print("Successfully generated all demo runs!")

if __name__ == "__main__":
    generate_data()
