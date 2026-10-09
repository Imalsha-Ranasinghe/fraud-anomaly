import time

import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_transactions
from src.metrics import recall_at_precision, score
from src.train import FEATURES, pick_threshold

TRUSTED_TYPES = [
    "sklearn.tree._tree.Tree",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]

def candidates():
    return [
        (
            "logreg-balanced",
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
            ]),
            {"model": "LogisticRegression", "class_weight": "balanced", "max_iter": 1000},
        ),
        (
            "random-forest",
            RandomForestClassifier(
                n_estimators=100, max_depth=12, class_weight="balanced_subsample",
                n_jobs=-1, random_state=42,
            ),
            {"model": "RandomForest", "n_estimators": 100, "max_depth": 12,
             "class_weight": "balanced_subsample"},
        ),
        (
            "hist-gradient-boosting",
            HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
            {"model": "HistGradientBoosting", "class_weight": "balanced"},
        ),
    ]


def single_row_ms(model, X, n=200):
    start = time.perf_counter()
    for i in range(n):
        score(model, X.iloc[[i]])
    return (time.perf_counter() - start) / n * 1000


def run_one(name, model, params, splits):
    X_tr, y_tr, X_val, y_val, X_te, y_te = splits
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"split": "time 60/20/20", "min_precision": 0.8, **params})

        model.fit(X_tr, y_tr)
        if isinstance(model, RandomForestClassifier):
            model.set_params(n_jobs=1)

        threshold = pick_threshold(y_val, score(model, X_val))
        scores = score(model, X_te)
        preds = (scores >= threshold).astype(int)

        metrics = {
            "pr_auc": float(average_precision_score(y_te, scores)),
            "recall_at_p80": recall_at_precision(y_te, scores),
            "precision": float(precision_score(y_te, preds, zero_division=0)),
            "recall": float(recall_score(y_te, preds)),
            "threshold": float(threshold),
            "ms_per_row": single_row_ms(model, X_te),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=TRUSTED_TYPES)
        print(name, {k: round(v, 4) for k, v in metrics.items()})


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fraud-detection")

    df = load_transactions().sort_values("Time")
    n = len(df)
    train_df = df.iloc[: int(n * 0.6)]
    val_df = df.iloc[int(n * 0.6): int(n * 0.8)]
    test_df = df.iloc[int(n * 0.8):]
    splits = (
        train_df[FEATURES], train_df["Class"],
        val_df[FEATURES], val_df["Class"],
        test_df[FEATURES], test_df["Class"],
    )

    for name, model, params in candidates():
        run_one(name, model, params, splits)


if __name__ == "__main__":
    main()