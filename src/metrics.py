import numpy as np
from sklearn.metrics import precision_recall_curve


def recall_at_precision(y, scores, min_p=0.8):
    p, r, _ = precision_recall_curve(y, scores)
    ok = p >= min_p
    return float(r[ok].max()) if ok.any() else 0.0


def score(model, X):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    return model.predict_proba(X)[:, 1]