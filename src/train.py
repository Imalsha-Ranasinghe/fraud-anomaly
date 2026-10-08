import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from src.data import load_transactions, DATA_PATH

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
MODEL_PATH = DATA_PATH.parent.parent / "models" / "model.joblib"

def split_by_time(df, test_frac=0.2):
    df = df.sort_values("Time")
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]

def train(df=None):
    df = load_transactions() if df is None else df
    train_df, test_df = split_by_time(df)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    model.fit(train_df[FEATURES], train_df["Class"])

    y_test = test_df["Class"]
    scores = model.predict_proba(test_df[FEATURES])[:, 1]
    preds = (scores >= 0.5).astype(int)
    metrics = {
        "pr_auc": average_precision_score(y_test, scores),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds),
    }
    return model, metrics

if __name__ == "__main__":
    model, metrics = train()
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(metrics)