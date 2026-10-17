import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.model import CLASS_NAMES, SIGNAL_LENGTH

SAMPLE_BEAT = json.loads((Path(__file__).resolve().parent.parent / "app" / "sample_beat.json").read_text())["beat"]


@pytest.fixture(scope="module")
def client():
    # The "with" block runs the app's startup, which loads the model.
    with TestClient(app) as c:
        yield c


def test_sample_beat_has_expected_length():
    assert len(SAMPLE_BEAT) == SIGNAL_LENGTH


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_beat(client):
    response = client.post("/predict", json={"beat": SAMPLE_BEAT})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in CLASS_NAMES
    assert CLASS_NAMES[body["class_index"]] == body["label"]


def test_predict_known_beat_gives_expected_class(client):
    # sample_beat.json is a real Ventricular beat.
    body = client.post("/predict", json={"beat": SAMPLE_BEAT}).json()
    assert body["label"] == "Ventricular"


def test_probabilities_cover_all_classes_and_sum_to_one(client):
    probs = client.post("/predict", json={"beat": SAMPLE_BEAT}).json()["probabilities"]
    assert list(probs) == CLASS_NAMES
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)


def test_predicted_label_has_highest_probability(client):
    body = client.post("/predict", json={"beat": SAMPLE_BEAT}).json()
    assert body["probabilities"][body["label"]] == max(body["probabilities"].values())


@pytest.mark.parametrize("length", [0, 10, SIGNAL_LENGTH - 1, SIGNAL_LENGTH + 1])
def test_wrong_length_is_rejected(client, length):
    response = client.post("/predict", json={"beat": [0.1] * length})
    assert response.status_code == 422


def test_non_numeric_values_are_rejected(client):
    response = client.post("/predict", json={"beat": ["a"] * SIGNAL_LENGTH})
    assert response.status_code == 422


def test_missing_field_is_rejected(client):
    assert client.post("/predict", json={}).status_code == 422


def test_non_finite_values_are_rejected(client):
    # JSON has no NaN, but Python's json parser accepts the bare word, so send the raw text.
    raw = '{"beat": [' + ",".join(["NaN"] * SIGNAL_LENGTH) + "]}"
    response = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_home_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ECG Heartbeat Classifier" in response.text


def test_demo_samples_are_valid_and_classified_correctly(client):
    samples = client.get("/static/samples.json").json()
    assert [s["class"] for s in samples] == CLASS_NAMES
    for sample in samples:
        assert len(sample["beat"]) == SIGNAL_LENGTH
        body = client.post("/predict", json={"beat": sample["beat"]}).json()
        assert body["label"] == sample["class"]
