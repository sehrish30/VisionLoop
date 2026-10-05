"""Offline contract tests; random weights exercise code, not model quality."""
import joblib
import numpy as np
import pytest

from visionloop import ml, store, worker
from visionloop.api import _model_cache
from test_workflow import client, seed


def test_trainer_validation_and_default(client):
    seed()
    assert client.post("/runs", json={"trainer": "unknown"}).status_code == 422
    assert client.post("/runs").status_code == 202
    assert client.get("/runs").json()[0]["trainer"] == "baseline"


def test_vit_roundtrip_and_comparison(client, monkeypatch):
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    torch.set_num_threads(2)
    seed()
    # Establish a real legacy baseline first.
    client.post("/runs")
    worker.tick()
    baseline = client.get("/models").json()[0]
    assert client.post(f"/models/{baseline['id']}/activate").status_code == 200

    original_create = timm.create_model
    def offline_create(name, **kwargs):
        kwargs["pretrained"] = False
        torch.manual_seed(42)
        return original_create(name, **kwargs)
    monkeypatch.setattr(timm, "create_model", offline_create)
    assert client.post("/runs", json={"trainer": "vit"}).status_code == 202
    worker.tick()
    run = client.get("/runs").json()[0]
    assert run["status"] == "completed", run["error"]
    assert run["trainer"] == "vit"
    assert run["metrics"]["backbone_frozen"] is True
    assert run["baseline_metrics"]["samples"] == run["metrics"]["samples"] == 10
    assert run["metrics"]["training_samples"] == 35
    assert client.get("/overview").json()["active_model"]["id"] == baseline["id"]

    candidate = client.get("/models").json()[0]
    model = joblib.load(store.DATA / candidate["path"])
    assert model._runtime is None
    image = next(i for i in client.get("/images").json() if i["split"] == "validation")
    path = store.DATA / image["path"]
    expected = ml.predict(model, path)
    assert len(expected) == 5
    assert sum(s["score"] for s in expected) == pytest.approx(1)
    assert all(not p.requires_grad for p in model._runtime[0].parameters())

    def no_download(name, **kwargs):
        assert kwargs["pretrained"] is False
        return original_create(name, **kwargs)
    monkeypatch.setattr(timm, "create_model", no_download)
    restored = joblib.load(store.DATA / candidate["path"])
    actual = ml.predict(restored, path)
    assert [s["label"] for s in actual] == [s["label"] for s in expected]
    np.testing.assert_allclose([s["score"] for s in actual], [s["score"] for s in expected])
    # Force only the eligibility flag to exercise activation/inference independently
    # of random fixture quality. Production gate behavior has separate tests.
    with store.connect() as db:
        db.execute("UPDATE models SET eligible=1 WHERE id=?", (candidate["id"],))
    _model_cache.clear()
    assert client.post(f"/models/{candidate['id']}/activate").status_code == 200
    content = path.read_bytes()
    result = client.post("/classify", files={"file": ("test.jpg", content, "image/jpeg")})
    assert result.json()["prediction"]["model_id"] == candidate["id"]
    assert client.post(f"/models/{candidate['id']}/test-report").status_code == 202
    worker.tick()
    report = client.get(f"/models/{candidate['id']}/test-report")
    assert report.status_code == 200
    assert report.json()["metrics"]["samples"] == 5
    assert client.post(f"/models/{baseline['id']}/activate").status_code == 200


def test_vit_failure_preserves_active_model(client, monkeypatch):
    from visionloop import vit
    seed()
    client.post("/runs")
    worker.tick()
    baseline = client.get("/models").json()[0]
    client.post(f"/models/{baseline['id']}/activate")
    def unavailable():
        raise RuntimeError("Vision Transformer dependencies are missing.")
    monkeypatch.setattr(vit, "dependencies", unavailable)
    client.post("/runs", json={"trainer": "vit"})
    worker.tick()
    run = client.get("/runs").json()[0]
    assert run["status"] == "failed"
    assert "dependencies are missing" in run["error"]
    assert client.get("/overview").json()["active_model"]["id"] == baseline["id"]
