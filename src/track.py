import mlflow
import mlflow.sklearn

from src.train import train


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="logreg-balanced"):
        mlflow.log_params({
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "split": "time 60/20/20",
            "min_precision": 0.8,
        })
        model, metrics = train()
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")
        print(metrics)


if __name__ == "__main__":
    main()