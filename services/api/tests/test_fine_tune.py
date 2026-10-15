import joblib
import numpy as np
import pytest

from visionloop import store, ml, worker
from test_workflow import client, seed


def test_fine_tuning_weights_artifact_and_workflow(client, monkeypatch):
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    from visionloop import fine_tune
    torch.set_num_threads(2)
    # A shorter preset keeps the test fast; the recipe is otherwise identical.
    monkeypatch.setattr(fine_tune, "DEFAULT_CONFIG", dict(fine_tune.DEFAULT_CONFIG, max_epochs=3, patience=3))
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
    assert queued["training_config"]["max_epochs"] == 3
    assert queued["training_config"]["head_init"] == "linear_probe"
    assert worker.tick()
    run = client.get("/runs").json()[0]
    assert run["status"] == "completed", run["error"]
    assert run["metrics"]["backbone_frozen"] is False
    assert len(run["metrics"]["training_losses"]) == 3
    assert all(np.isfinite(run["metrics"]["training_losses"]))
    # One stopping-set loss for the linear-probe start plus one per epoch.
    assert len(run["metrics"]["holdout_losses"]) == 4
    assert run["metrics"]["holdout_samples"] == 5
    assert 0 <= run["metrics"]["best_epoch"] <= 3
    assert any("epoch 3 of up to 3" in step for step in recorded_steps)
    candidate = client.get("/models").json()[0]
    validation = client.get(f"/models/{candidate['id']}/validation").json()
    assert validation["samples"] == 10 and len(validation["items"]) == 10
    assert client.get("/overview").json()["active_model"]["id"] == baseline["id"]
    model = joblib.load(store.DATA / candidate["path"])
    assert model._runtime is None
    # Verify every earlier weight is unchanged, and the final block/head learned.
    changed = {k for k in initial if not np.array_equal(initial[k], model.weights[k])}
    assert "head.weight" in changed  # replaced by the linear probe at minimum
    assert all(k.startswith(("blocks.11.", "norm.", "head.")) for k in changed)
    # The final block only moves if a fine-tuned epoch beat the starting point.
    assert any(k.startswith("blocks.11.") for k in changed) == (run["metrics"]["best_epoch"] > 0)

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


def test_stratified_holdout_is_balanced_and_repeatable():
    from visionloop import fine_tune
    labels = ["Dress"] * 20 + ["Shirt"] * 7 + ["Blouse"] * 3
    held = fine_tune.stratified_holdout(labels, 0.15, seed=42)
    assert held == fine_tune.stratified_holdout(labels, 0.15, seed=42)
    counts = {label: sum(labels[i] == label for i in held) for label in set(labels)}
    assert counts == {"Dress": 3, "Shirt": 1, "Blouse": 1}
    # Never leaves a class with fewer than two training images.
    assert fine_tune.stratified_holdout(["Dress"] * 2, 0.5, seed=1) == []


def test_linear_probe_head_matches_logistic_regression():
    from visionloop import fine_tune
    rng = np.random.default_rng(0)
    labels = [label for label in store.LABELS for _ in range(12)]
    features = rng.normal(size=(len(labels), 16)) + np.repeat(np.eye(5, 16) * 3, 12, axis=0)
    weight, bias, probe = fine_tune.linear_probe_weights(features, labels)
    logits = features @ weight.T + bias
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    expected = probe.predict_proba(features)[:, [list(probe.classes_).index(l) for l in store.LABELS]]
    np.testing.assert_allclose(probabilities, expected, atol=1e-8)


def test_fine_tuning_keeps_starting_point_when_epochs_get_worse(client, monkeypatch):
    torch = pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    from visionloop import fine_tune
    torch.set_num_threads(2)
    seed()
    original_create = timm.create_model
    start = {}
    def offline_model(name, **kwargs):
        kwargs["pretrained"] = False
        torch.manual_seed(0)
        model = original_create(name, **kwargs)
        start.update({k: v.detach().numpy().copy() for k, v in model.state_dict().items()})
        return model
    monkeypatch.setattr(timm, "create_model", offline_model)
    # Absurd learning rates make every epoch worse, so early stopping must fall back.
    config = dict(fine_tune.DEFAULT_CONFIG, max_epochs=4, patience=2, block_learning_rate=5.0, head_learning_rate=50.0)
    monkeypatch.setattr(fine_tune, "DEFAULT_CONFIG", config)
    with store.connect() as db:
        rows = db.execute("SELECT path,label FROM images WHERE split='train'").fetchall()
    model = fine_tune.fit([store.DATA / r["path"] for r in rows], [r["label"] for r in rows], config)
    assert model.best_epoch == 0
    assert len(model.training_losses) == 2  # stopped after `patience` epochs without improvement
    assert min(model.holdout_losses) == model.holdout_losses[0]
    assert all(np.array_equal(start[k], model.weights[k]) for k in start if k.startswith("blocks."))
