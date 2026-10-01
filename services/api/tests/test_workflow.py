import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from visionloop import store, worker, ml
from visionloop.api import app, _model_cache
from visionloop.images import save_image


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA", tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    _model_cache.clear()
    with TestClient(app) as client:
        yield client


def photo(class_index=0, variation=0):
    image = Image.new("RGB", (64, 64), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    left = 3 + class_index * 10
    draw.rectangle((left, 5, left + 8, 55), fill=(20 + variation, 20, 20))
    draw.point((variation, 63), fill=(variation, 100, 180))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def seed():
    for ci, label in enumerate(store.LABELS):
        for n in range(10):
            split = "train" if n < 7 else "validation" if n < 9 else "test"
            save_image(photo(ci, n), f"{label}-{n}.png", "test-fixture", label, split)


def test_upload_validation_dedup_and_review(client):
    bad = client.post("/classify", files={"file": ("bad.png", b"not an image", "image/png")})
    assert bad.status_code == 422
    first = client.post("/classify", files={"file": ("piece.png", photo(), "image/png")}).json()
    assert first["prediction"] is None
    second = client.post("/classify", files={"file": ("renamed.png", photo(), "image/png")}).json()
    assert first["image"]["id"] == second["image"]["id"]
    image_id = first["image"]["id"]
    assert client.post(f"/images/{image_id}/review", json={"label": "unsupported"}).status_code == 422
    assert client.post(f"/images/{image_id}/review", json={"label": "Dress"}).status_code == 200
    assert client.get("/overview").json()["pending"] == 0
    assert client.get(f"/images/{image_id}/file").headers["content-type"] == "image/jpeg"
    normalized = client.get(f"/images/{image_id}/file").content
    reupload = client.post("/classify", files={"file": ("normalized.jpg", normalized, "image/jpeg")}).json()
    assert reupload["image"]["id"] == image_id


def test_training_activation_prediction_and_snapshot(client):
    seed()
    run = client.post("/runs")
    assert run.status_code == 202
    assert client.post("/runs").status_code == 409
    before = client.get("/runs").json()[0]
    snapshot = (store.DATA / before["snapshot"]).read_text()
    first = next(i for i in client.get("/images").json() if i["split"] == "train")
    client.post(f"/images/{first['id']}/review", json={"label": "Dress"})
    assert (store.DATA / before["snapshot"]).read_text() == snapshot
    assert worker.tick()
    completed = client.get("/runs").json()[0]
    assert completed["status"] == "completed", completed["error"]
    assert completed["metrics"]["samples"] == 10
    assert completed["metrics"]["training_samples"] == 35
    model = client.get("/models").json()[0]
    assert model["can_activate"]
    assert client.post(f"/models/{model['id']}/activate").status_code == 200
    result = client.post("/classify", files={"file": ("dress.png", photo(0, 12), "image/png")}).json()
    assert result["prediction"]["model_id"] == model["id"]
    assert len(result["prediction"]["scores"]) == 5
    assert sum(s["score"] for s in result["prediction"]["scores"]) == pytest.approx(1)


def test_missing_data_and_held_out_protection(client):
    assert client.post("/runs").status_code == 409
    seed()
    heldout = next(i for i in client.get("/images").json() if i["split"] == "test")
    assert client.post(f"/images/{heldout['id']}/review", json={"label": "Shirt"}).status_code == 409
    normalized = client.get(f"/images/{heldout['id']}/file").content
    reupload = client.post("/classify", files={"file": ("heldout.jpg", normalized, "image/jpeg")}).json()
    assert reupload["image"]["split"] == "test"


def test_failed_training_keeps_active_model(client, monkeypatch):
    seed()
    client.post("/runs")
    worker.tick()
    model = client.get("/models").json()[0]
    client.post(f"/models/{model['id']}/activate")
    client.post("/runs")
    def fail(*args):
        raise RuntimeError("Training fixture failed")
    monkeypatch.setattr(ml, "fit_model", fail)
    worker.tick()
    assert client.get("/runs").json()[0]["status"] == "failed"
    assert client.get("/overview").json()["active_model"]["id"] == model["id"]


def test_unimproved_candidate_cannot_activate(client):
    seed()
    client.post("/runs")
    worker.tick()
    first = client.get("/models").json()[0]
    client.post(f"/models/{first['id']}/activate")
    client.post("/runs")
    worker.tick()
    second = client.get("/models").json()[0]
    assert not second["eligible"]
    assert client.post(f"/models/{second['id']}/activate").status_code == 409
    assert client.get("/overview").json()["active_model"]["id"] == first["id"]
