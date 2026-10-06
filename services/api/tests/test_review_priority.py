import json
import uuid

from visionloop import store
from visionloop.images import save_image
from test_workflow import client, photo, seed


def prediction(image_id, score, model="saved-model", label="Dress"):
    with store.connect() as db:
        db.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?)", (
            uuid.uuid4().hex, image_id, model, label,
            json.dumps([{"label": label, "score": score}]), "2026-10-06T00:00:00Z",
        ))


def test_priority_scores_latest_prediction_and_review(client):
    items = [save_image(photo(0, n + 20), f"{n}.png") for n in range(5)]
    prediction(items[0]["id"], .9)
    prediction(items[1]["id"], .64)
    prediction(items[2]["id"], .65)
    prediction(items[3]["id"], .8)
    # Same timestamp: score, label and model must all come from the latest row.
    prediction(items[3]["id"], .3, "new-model", "Shirt")
    rows = client.get("/images?sort=review_priority").json()
    assert [r["id"] for r in rows] == [items[n]["id"] for n in (3, 1, 2, 0, 4)]
    assert rows[0]["predicted_label"] == "Shirt"
    assert rows[0]["model_id"] == "new-model"
    assert rows[0]["prediction_score"] == .3
    assert [r["low_score"] for r in rows] == [True, True, False, False, False]
    assert rows[-1]["prediction_score"] is None
    overview = client.get("/overview").json()
    assert overview["review_threshold"] == .65
    assert overview["low_score_pending"] == 2
    client.post(f'/images/{items[3]["id"]}/review', json={"label": "Blouse"})
    assert client.get("/overview").json()["low_score_pending"] == 1
    assert client.get("/overview").json()["pending"] == 4
    rows = client.get("/images?sort=review_priority").json()
    assert rows[-1]["id"] == items[3]["id"]
    assert rows[-1]["reviewed_label"] == "Blouse"
    assert rows[-1]["predicted_label"] == "Shirt"


def test_priority_does_not_label_or_change_snapshots(client):
    seed()
    upload = save_image(photo(0, 30), "uncertain.png")
    prediction(upload["id"], .4)
    heldout = next(r for r in client.get("/images").json() if r["split"] == "test")
    prediction(heldout["id"], .2)
    assert client.get("/overview").json()["low_score_pending"] == 1
    assert client.post("/runs").status_code == 202
    run = client.get("/runs").json()[0]
    path = store.DATA / run["snapshot"]
    original = path.read_text()
    assert upload["id"] not in {r["id"] for r in json.loads(original)}
    client.post(f'/images/{upload["id"]}/review', json={"label": "Sweater"})
    assert path.read_text() == original
    assert client.get("/overview").json()["low_score_pending"] == 0
    with store.connect() as db:
        db.execute("UPDATE runs SET status='failed' WHERE id=?", (run["id"],))
    next_run = client.post("/runs").json()
    with store.connect() as db:
        snapshot = db.execute("SELECT snapshot FROM runs WHERE id=?", (next_run["id"],)).fetchone()[0]
    rows = json.loads((store.DATA / snapshot).read_text())
    assert next(r for r in rows if r["id"] == upload["id"])["label"] == "Sweater"
    assert next(r for r in rows if r["id"] == heldout["id"])["split"] == "test"
