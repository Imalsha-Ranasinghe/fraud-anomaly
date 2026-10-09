import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.data import load_transactions
from src.train import FEATURES

MODEL_NAME = "fraud-detector"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
    print("champion = version", mv.version, "| threshold tag:", mv.tags.get("threshold"))

    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    X = load_transactions()[FEATURES].head(3)
    print(model.predict_proba(X))


if __name__ == "__main__":
    main()