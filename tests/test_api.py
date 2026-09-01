"""API tests using a fast throwaway model artifact.

The real model file is git-ignored, so these tests train a tiny model, point
the app at it, and exercise the endpoints with the FastAPI test client.
"""

import joblib
import pytest
from fastapi.testclient import TestClient

from stroke_risk.data import load_raw, split_data
from stroke_risk.features import split_features_target
from stroke_risk.model import build_logreg

VALID_PATIENT = {
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Train a small, fast model and save it as the app's artifact.
    train = split_data(load_raw()).train.head(400)
    X, y = split_features_target(train)
    model = build_logreg().fit(X, y)
    artifact = {"model": model, "threshold": 0.5, "model_name": "LogReg-test"}

    path = tmp_path_factory.mktemp("models") / "stroke_model.joblib"
    joblib.dump(artifact, path)

    import app.main as api

    api.config.MODEL_PATH = path
    api.load_artifact.cache_clear()
    return TestClient(api.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid(client):
    resp = client.post("/predict", json=VALID_PATIENT)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["risk"], bool)
    assert body["threshold"] == 0.5


def test_predict_accepts_missing_bmi(client):
    patient = {**VALID_PATIENT, "bmi": None}
    resp = client.post("/predict", json=patient)
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["probability"] <= 1.0


def test_predict_rejects_invalid_category(client):
    patient = {**VALID_PATIENT, "gender": "Robot"}
    resp = client.post("/predict", json=patient)
    assert resp.status_code == 422


def test_predict_rejects_out_of_range_age(client):
    patient = {**VALID_PATIENT, "age": 999}
    resp = client.post("/predict", json=patient)
    assert resp.status_code == 422


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Stroke Risk" in resp.text
