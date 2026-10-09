import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MODEL_NAME = "fraud-detector"
METRIC = "pr_auc"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    runs = mlflow.search_runs(
        experiment_names=["fraud-detection"],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{METRIC} DESC"],
        max_results=1,
    )
    best = runs.iloc[0]
    run_id = best["run_id"]
    print("best run:", best["tags.mlflow.runName"], "|", METRIC, "=", best[f"metrics.{METRIC}"])

    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)

    client = MlflowClient()
    client.set_model_version_tag(
        MODEL_NAME, version.version, "threshold", str(best["metrics.threshold"])
    )
    client.set_registered_model_alias(MODEL_NAME, "champion", version.version)
    print("registered version", version.version, "as champion")


if __name__ == "__main__":
    main()