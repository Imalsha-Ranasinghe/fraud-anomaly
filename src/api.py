import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import Field, create_model

from src.metrics import score
from src.train import FEATURES, META_PATH, MODEL_PATH

app = FastAPI(title="Fraud Detection API")

model = joblib.load(MODEL_PATH)
meta = json.loads(META_PATH.read_text())

fields = {f: (float, ...) for f in FEATURES}
fields["Amount"] = (float, Field(ge=0))
Transaction = create_model("Transaction", **fields)

@app.get("/health")
def health():
    return {"status": "ok", "model_version": meta["version"]}

@app.post("/predict")
def predict(tx: Transaction):
    row = pd.DataFrame([tx.model_dump()])[FEATURES]
    s = float(score(model, row)[0])
    return {
        "is_fraud": s >= meta["threshold"],
        "score": s,
        "threshold": meta["threshold"],
        "model_version": meta["version"],
    }