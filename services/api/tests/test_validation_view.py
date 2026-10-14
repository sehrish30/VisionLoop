from visionloop import ml, store, worker
from visionloop.api import _model_cache
from test_workflow import client, seed


def train(client):
    client.post("/runs")
    assert worker.tick()
    return client.get("/models").json()[0]


def test_validation_view_matches_metrics_and_is_cached(client, monkeypatch):
    seed()
    model = train(client)
    response = client.get(f"/models/{model['id']}/validation")
    assert response.status_code == 200
    result = response.json()
    images = {i["id"]: i for i in client.get("/images").json()}
    assert result["samples"] == model["metrics"]["samples"] == 10
    assert all(images[item["image_id"]]["split"] == "validation" for item in result["items"])
    # Same images and labels the run was scored on, so totals agree with its accuracy.
    assert result["correct"] == round(model["metrics"]["accuracy"] * result["samples"])
    assert all(item["correct"] == (item["label"] == item["predicted"]) for item in result["items"])
    assert all(0 <= item["score"] <= 1 for item in result["items"])
    assert (store.DATA / "validation" / f"{model['id']}.json").exists()

    def no_recompute(*args, **kwargs):
        raise AssertionError("Cached validation results should be reused")
    monkeypatch.setattr(ml, "predict_labels", no_recompute)
    assert client.get(f"/models/{model['id']}/validation").json() == result


def test_validation_view_keeps_active_model_loaded(client):
    seed()
    first = train(client)
    assert client.post(f"/models/{first['id']}/activate").status_code == 200
    second = train(client)
    assert client.get(f"/models/{second['id']}/validation").status_code == 200
    assert list(_model_cache) == [first["id"]]
    assert client.get("/overview").json()["active_model"]["id"] == first["id"]


def test_validation_view_unknown_model(client):
    assert client.get("/models/missing/validation").status_code == 404
