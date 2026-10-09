import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import DATA_PATH, load_transactions

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
ROOT = DATA_PATH.parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
META_PATH = ROOT / "models" / "meta.json"
MODEL_VERSION = "1.0.0"

def split_by_time(df, test_frac=0.2):
    df = df.sort_values("Time")
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]

def pick_threshold(y, scores, min_precision=0.8):
    p, _, t = precision_recall_curve(y, scores)
    ok = np.where(p[:-1] >= min_precision)[0]
    return float(t[ok[0]]) if len(ok) else 0.0

def train(df=None):
    df = (load_transactions() if df is None else df).sort_values("Time")
    n = len(df)
    train_df = df.iloc[: int(n * 0.6)]
    val_df = df.iloc[int(n * 0.6): int(n * 0.8)]
    test_df = df.iloc[int(n * 0.8):]

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    model.fit(train_df[FEATURES], train_df["Class"])

    threshold = pick_threshold(val_df["Class"], model.decision_function(val_df[FEATURES]))

    y_test = test_df["Class"]
    scores = model.decision_function(test_df[FEATURES])
    preds = (scores >= threshold).astype(int)
    metrics = {
        "pr_auc": average_precision_score(y_test, scores),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds),
        "threshold": threshold,
    }
    return model, metrics

if __name__ == "__main__":
    model, metrics = train()
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps({
        "version": MODEL_VERSION,
        "threshold": metrics["threshold"],
        "features": FEATURES,
    }))
    print(metrics)