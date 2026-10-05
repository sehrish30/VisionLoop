import json

from visionloop import store, worker, ml
from visionloop.images import save_image
from test_workflow import client, seed, photo


def trained_model(client, activate=True):
    seed()
    client.post("/runs")
    worker.tick()
    model = client.get("/models").json()[0]
    if activate:
        assert client.post(f"/models/{model['id']}/activate").status_code == 200
    return model


def test_report_uses_only_original_test_snapshot_and_is_cached(client, monkeypatch):
    model = trained_model(client)
    before = client.get("/models").json()[0]
    # Later imports must not silently expand this model's original test set.
    save_image(photo(0, 40), "later.png", "fixture", "Dress", "test")
    endpoint = f"/models/{model['id']}/test-report"
    created = client.post(endpoint)
    assert created.status_code == 202
    assert client.post(endpoint).json()["id"] == created.json()["id"]
    assert client.get(endpoint).status_code == 409
    with store.connect() as db:
        report = db.execute("SELECT * FROM test_reports").fetchone()
        snapshot = json.loads(report["snapshot"])
    assert len(snapshot) == 5
    assert all(r["split"] == "test" for r in snapshot)
    original_evaluate = ml.evaluate
    def evaluate(model, paths, labels, **kwargs):
        assert set(paths) == {store.DATA / r["path"] for r in snapshot}
        return original_evaluate(model, paths, labels, **kwargs)
    def no_training(*args, **kwargs):
        raise AssertionError("Final reports must not train models")
    monkeypatch.setattr(ml, "fit_model", no_training)
    monkeypatch.setattr(ml, "evaluate", evaluate)
    assert worker.tick()
    after = client.get("/models").json()[0]
    assert after["metrics"] == before["metrics"]
    assert after["eligible"] == before["eligible"]
    assert after["active"]
    result = after["test_report"]
    assert result["status"] == "completed", result["error"]
    assert result["metrics"]["samples"] == 5
    assert sum(r["samples"] for r in result["metrics"]["per_class"]) == 5
    assert len(result["metrics"]["confusion_matrix"]) == 5
    exported = client.get(endpoint)
    assert exported.status_code == 200
    assert "attachment" in exported.headers["content-disposition"]
    assert exported.json()["snapshot"] == snapshot
    assert exported.json()["model_id"] == model["id"]
    assert client.post(endpoint).json()["status"] == "completed"
    assert not worker.tick()


def test_report_requires_active_model_and_all_classes(client):
    assert client.post("/models/missing/test-report").status_code == 404
    model = trained_model(client, activate=False)
    endpoint = f"/models/{model['id']}/test-report"
    assert client.get(endpoint).status_code == 404
    assert client.post(endpoint).status_code == 409
    client.post(f"/models/{model['id']}/activate")
    # Exercise legacy snapshots lacking coverage without changing image assignments.
    with store.connect() as db:
        snapshot = db.execute("SELECT snapshot FROM runs WHERE id=?", (model["run_id"],)).fetchone()[0]
    path = store.DATA / snapshot
    rows = json.loads(path.read_text())
    path.write_text(json.dumps([r for r in rows if not (r["split"] == "test" and r["label"] == "Dress")]))
    assert client.post(endpoint).status_code == 409


def test_failed_report_retry_keeps_snapshot_and_active_model(client, monkeypatch):
    model = trained_model(client)
    endpoint = f"/models/{model['id']}/test-report"
    report_id = client.post(endpoint).json()["id"]
    original_evaluate = ml.evaluate
    def fail(*args, **kwargs):
        raise RuntimeError("Fixture evaluation failure")
    monkeypatch.setattr(ml, "evaluate", fail)
    worker.tick()
    assert client.get("/models").json()[0]["test_report"]["status"] == "failed"
    with store.connect() as db:
        snapshot = db.execute("SELECT snapshot FROM test_reports").fetchone()[0]
    save_image(photo(0, 40), "later.png", "fixture", "Dress", "test")
    assert client.post(endpoint).json()["id"] == report_id
    monkeypatch.setattr(ml, "evaluate", original_evaluate)
    worker.tick()
    with store.connect() as db:
        assert db.execute("SELECT snapshot FROM test_reports").fetchone()[0] == snapshot
    assert client.get(endpoint).json()["metrics"]["samples"] == 5
    assert client.get("/overview").json()["active_model"]["id"] == model["id"]


def test_interrupted_report_is_retryable(client):
    model = trained_model(client)
    endpoint = f"/models/{model['id']}/test-report"
    client.post(endpoint)
    with store.connect() as db:
        db.execute("UPDATE test_reports SET status='running',heartbeat='2000-01-01T00:00:00+00:00'")
    assert not worker.tick()
    report = client.get("/models").json()[0]["test_report"]
    assert report["status"] == "failed"
    assert "Worker stopped" in report["error"]
    assert client.post(endpoint).status_code == 202
    assert worker.tick()
    assert client.get(endpoint).status_code == 200


def test_training_and_test_jobs_do_not_run_together(client):
    model = trained_model(client)
    client.post(f"/models/{model['id']}/test-report")
    client.post("/runs")
    # Simulate a healthy worker holding the report claim.
    with store.connect() as db:
        db.execute("UPDATE test_reports SET status='running',heartbeat=?", (store.now(),))
    assert not worker.tick()
    assert client.get("/runs").json()[0]["status"] == "queued"


def test_modified_test_split_fails_report(client):
    model = trained_model(client)
    client.post(f"/models/{model['id']}/test-report")
    with store.connect() as db:
        db.execute("UPDATE images SET split='train' WHERE split='test'")
    worker.tick()
    report = client.get("/models").json()[0]["test_report"]
    assert report["status"] == "failed"
    assert "no longer matches" in report["error"]
