import joblib
import numpy as np
import pytest

from visionloop import store, ml, worker
from test_workflow import client, seed


def test_fine_tuning_weights_artifact_and_workflow(client, monkeypatch):
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    torch.set_num_threads(2)
    seed()
    client.post("/runs")
    worker.tick()
    baseline = client.get("/models").json()[0]
    client.post(f"/models/{baseline['id']}/activate")
    original_create = timm.create_model
    initial = {}
    def offline_model(name, **kwargs):
        requested_pretrained = kwargs["pretrained"]
        kwargs["pretrained"] = False
        model = original_create(name, **kwargs)
        if requested_pretrained:
            initial.update({k: v.detach().numpy().copy() for k, v in model.state_dict().items()})
        return model
    monkeypatch.setattr(timm, "create_model", offline_model)
    recorded_steps = []
    real_progress = worker.progress
    def progress(run_id, step, value):
        recorded_steps.append(step)
        real_progress(run_id, step, value)
    monkeypatch.setattr(worker, "progress", progress)
    original_fit = ml.fit_model
    with store.connect() as db:
        train_paths = {store.DATA / r[0] for r in db.execute("SELECT path FROM images WHERE split='train'")}
    def fit(paths, labels, trainer, **kwargs):
        assert set(paths) == train_paths
        return original_fit(paths, labels, trainer, **kwargs)
    monkeypatch.setattr(ml, "fit_model", fit)
    assert client.post("/runs", json={"trainer": "vit_finetune"}).status_code == 202
    queued = client.get("/runs").json()[0]
    assert queued["training_config"]["epochs"] == 3
    assert worker.tick()
    run = client.get("/runs").json()[0]
    assert run["status"] == "completed", run["error"]
    assert run["metrics"]["backbone_frozen"] is False
    assert len(run["metrics"]["training_losses"]) == 3
    assert all(np.isfinite(run["metrics"]["training_losses"]))
    assert any("epoch 3/3" in step for step in recorded_steps)
    candidate = client.get("/models").json()[0]
    assert client.get("/overview").json()["active_model"]["id"] == baseline["id"]
    model = joblib.load(store.DATA / candidate["path"])
    assert model._runtime is None
    # Verify every earlier weight is unchanged, and the final block/head learned.
    changed = {k for k in initial if not np.array_equal(initial[k], model.weights[k])}
    assert any(k.startswith("blocks.11.") for k in changed)
    assert "head.weight" in changed
    assert all(k.startswith(("blocks.11.", "norm.", "head.")) for k in changed)

    def no_download(name, **kwargs):
        assert kwargs["pretrained"] is False
        return original_create(name, **kwargs)
    monkeypatch.setattr(timm, "create_model", no_download)
    image = next(i for i in client.get("/images").json() if i["split"] == "validation")
    scores = ml.predict(model, store.DATA / image["path"])
    assert len(scores) == 5
    assert sum(s["score"] for s in scores) == pytest.approx(1)
    assert set(model.classes_) == set(store.LABELS)
    reloaded = joblib.load(store.DATA / candidate["path"])
    again = ml.predict(reloaded, store.DATA / image["path"])
    np.testing.assert_allclose([s["score"] for s in scores], [s["score"] for s in again])
    # Exercise serving independently of random-fixture quality; gate tests are separate.
    with store.connect() as db:
        db.execute("UPDATE models SET eligible=1 WHERE id=?", (candidate["id"],))
    assert client.post(f"/models/{candidate['id']}/activate").status_code == 200
    result = client.post("/classify", files={"file": ("sample.jpg", (store.DATA / image["path"]).read_bytes(), "image/jpeg")})
    assert result.json()["prediction"]["model_id"] == candidate["id"]
    assert client.post(f"/models/{candidate['id']}/test-report").status_code == 202
    worker.tick()
    assert client.get(f"/models/{candidate['id']}/test-report").json()["metrics"]["samples"] == 5
    assert client.post(f"/models/{baseline['id']}/activate").status_code == 200


def test_failed_fine_tuning_preserves_active_model(client, monkeypatch):
    from visionloop import fine_tune
    seed()
    client.post("/runs")
    worker.tick()
    model = client.get("/models").json()[0]
    client.post(f"/models/{model['id']}/activate")
    def failure(*args, **kwargs):
        raise RuntimeError("Fixture optimizer failure")
    monkeypatch.setattr(fine_tune, "fit", failure)
    client.post("/runs", json={"trainer": "vit_finetune"})
    worker.tick()
    assert client.get("/runs").json()[0]["status"] == "failed"
    assert client.get("/overview").json()["active_model"]["id"] == model["id"]
