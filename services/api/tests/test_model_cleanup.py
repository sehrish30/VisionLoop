from visionloop import store, worker
from test_workflow import client, seed


def train(client, eligible=None):
    client.post("/runs")
    assert worker.tick()
    model = client.get("/models").json()[0]
    if eligible is not None:
        with store.connect() as db:
            db.execute("UPDATE models SET eligible=? WHERE id=?", (int(eligible), model["id"]))
    return model


def ids(client):
    return {m["id"] for m in client.get("/models").json()}


def test_delete_candidate_keeps_run_and_protects_active(client):
    seed()
    active = train(client)
    assert client.post(f"/models/{active['id']}/activate").status_code == 200
    candidate = train(client, eligible=False)
    assert client.get(f"/models/{candidate['id']}/validation").status_code == 200
    artifact = store.DATA / candidate["path"]
    cache = store.DATA / "validation" / f"{candidate['id']}.json"
    assert artifact.exists() and cache.exists()

    assert client.delete(f"/models/{active['id']}").status_code == 409
    assert client.delete("/models/missing").status_code == 404
    # A queued run may still need its baseline model.
    client.post("/runs")
    assert client.delete(f"/models/{candidate['id']}").status_code == 409
    with store.connect() as db:
        db.execute("UPDATE runs SET status='failed' WHERE status='queued'")

    assert client.delete(f"/models/{candidate['id']}").json() == {"deleted": [candidate["id"]]}
    assert ids(client) == {active["id"]}
    assert not artifact.exists() and not cache.exists()
    run = next(r for r in client.get("/runs").json() if r["id"] == candidate["run_id"])
    assert run["status"] == "completed" and run["model_id"] is None and run["metrics"]["samples"] == 10
    assert client.get("/overview").json()["active_model"]["id"] == active["id"]
    # The active model still answers predictions.
    sample = (store.DATA / client.get("/images").json()[0]["path"]).read_bytes()
    result = client.post("/classify", files={"file": ("sample.jpg", sample, "image/jpeg")}).json()
    assert result["prediction"]["model_id"] == active["id"]


def test_earlier_active_model_and_its_report_can_be_deleted(client):
    seed()
    first = train(client)
    client.post(f"/models/{first['id']}/activate")
    client.post(f"/models/{first['id']}/test-report")
    assert worker.tick()
    second = train(client, eligible=True)
    assert client.post(f"/models/{second['id']}/activate").status_code == 200
    assert client.delete(f"/models/{first['id']}").status_code == 200
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM test_reports").fetchone()[0] == 0
        # Activation history is an audit log and is not rewritten.
        assert db.execute("SELECT COUNT(*) FROM activations WHERE model_id=?", (first["id"],)).fetchone()[0] == 1
    assert client.post(f"/models/{first['id']}/activate").status_code == 404


def test_cleanup_removes_only_failed_candidates(client):
    seed()
    active = train(client)
    client.post(f"/models/{active['id']}/activate")
    failed = [train(client, eligible=False)["id"] for _ in range(2)]
    passed = train(client, eligible=True)["id"]
    deleted = client.post("/models/cleanup").json()["deleted"]
    assert sorted(deleted) == sorted(failed)
    assert ids(client) == {active["id"], passed}
    assert client.post("/models/cleanup").json() == {"deleted": []}
