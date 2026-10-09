from fastapi.testclient import TestClient

from src.api import app
from src.train import FEATURES

client = TestClient(app)
VALID = {f: 0.0 for f in FEATURES}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_returns_decision():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert set(r.json()) == {"is_fraud", "score", "threshold", "model_version"}

def test_predict_rejects_missing_field():
    bad = {k: v for k, v in VALID.items() if k != "Amount"}
    assert client.post("/predict", json=bad).status_code == 422

def test_predict_rejects_negative_amount():
    assert client.post("/predict", json={**VALID, "Amount": -5}).status_code == 422