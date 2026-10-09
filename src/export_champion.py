import json

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.train import FEATURES, META_PATH, MODEL_PATH

MODEL_NAME = "fraud-detector"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps({
        "version": f"registry-v{mv.version}",
        "threshold": float(mv.tags["threshold"]),
        "features": FEATURES,
    }))
    print("exported champion v", mv.version, type(model).__name__)


if __name__ == "__main__":
    main()