import json
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from . import store, ml
from .images import MAX_BYTES, save_image


@asynccontextmanager
async def lifespan(app):
    store.init()
    yield


app = FastAPI(title="VisionLoop local API", lifespan=lifespan)
_model_cache = {}


def load_model(model_id, db):
    if model_id not in _model_cache:
        row = db.execute("SELECT path FROM models WHERE id=?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Model not found.")
        # Only load our own locally generated artifacts, never uploaded pickle files.
        loaded = joblib.load(store.DATA / row["path"])
        if set(loaded.classes_) != set(store.LABELS):
            raise HTTPException(409, "Model labels do not match the workspace.")
        _model_cache.clear()
        _model_cache[model_id] = loaded
    return _model_cache[model_id]


def image_rows(db):
    return [dict(r) for r in db.execute('''
        SELECT i.*, COALESCE((SELECT r.label FROM reviews r WHERE r.image_id=i.id
        ORDER BY r.created_at DESC, r.rowid DESC LIMIT 1), i.label) AS reviewed_label,
        (SELECT p.label FROM predictions p WHERE p.image_id=i.id ORDER BY p.created_at DESC LIMIT 1) AS predicted_label,
        (SELECT p.model_id FROM predictions p WHERE p.image_id=i.id ORDER BY p.created_at DESC LIMIT 1) AS model_id
        FROM images i ORDER BY i.created_at DESC
    ''').fetchall()]


@app.get("/health")
def health():
    return {"status": "ok", "mode": "local", "aws_enabled": False}


@app.get("/overview")
def overview():
    with store.connect() as db:
        images = image_rows(db)
        current = store.active_id(db)
        model = db.execute("SELECT * FROM models WHERE id=?", (current,)).fetchone()
        return {
            "labels": store.LABELS, "dataset": store.DATASET,
            "images": len(images), "reviewed": sum(bool(i["reviewed_label"]) for i in images),
            "pending": sum(not i["reviewed_label"] for i in images),
            "class_counts": dict(Counter(i["reviewed_label"] for i in images if i["reviewed_label"])),
            "split_counts": dict(Counter(i["split"] for i in images if i["reviewed_label"])),
            "runs": db.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
            "active_model": {"id": current, "metrics": json.loads(model["metrics"])} if model else None,
            "worker_heartbeat": (db.execute("SELECT value FROM settings WHERE key='worker_heartbeat'").fetchone() or [None])[0],
        }


@app.get("/images")
def images():
    with store.connect() as db:
        return image_rows(db)


@app.get("/images/{image_id}/file")
def image_file(image_id: str):
    with store.connect() as db:
        row = db.execute("SELECT path FROM images WHERE id=?", (image_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Image not found.")
    return FileResponse(store.DATA / row["path"], media_type="image/jpeg")


def classify_content(content, filename):
    try:
        image = save_image(content, filename)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with store.connect() as db:
        current = store.active_id(db)
        prediction = None
        if current:
            scores = ml.predict(load_model(current, db), store.DATA / image["path"])
            prediction = {"label": scores[0]["label"], "scores": scores, "model_id": current}
            db.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?)", (uuid.uuid4().hex, image["id"], current, scores[0]["label"], json.dumps(scores), store.now()))
    return {"image": image, "prediction": prediction}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    content = await file.read(MAX_BYTES + 1)
    await file.close()
    return await run_in_threadpool(classify_content, content, file.filename or "Upload")


class ReviewInput(BaseModel):
    label: str


@app.post("/images/{image_id}/review")
def review(image_id: str, body: ReviewInput):
    if body.label not in store.LABELS:
        raise HTTPException(422, "Choose one of the five supported garment types.")
    with store.connect() as db:
        image = db.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        if not image:
            raise HTTPException(404, "Image not found.")
        if image["split"] != "train":
            raise HTTPException(409, "Evaluation labels are held out and cannot be edited here.")
        db.execute("INSERT INTO reviews VALUES (?,?,?,?)", (uuid.uuid4().hex, image_id, body.label, store.now()))
    return {"saved": True, "label": body.label}


@app.get("/runs")
def runs():
    with store.connect() as db:
        result = [dict(r) for r in db.execute("SELECT * FROM runs ORDER BY created_at DESC")]
    for r in result:
        for name in ("metrics", "baseline_metrics"):
            r[name] = json.loads(r[name]) if r[name] else None
    return result


class TrainingInput(BaseModel):
    trainer: Literal["baseline", "vit"] = "baseline"


@app.post("/runs", status_code=202)
def start_run(body: TrainingInput | None = None):
    trainer = body.trainer if body else "baseline"
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT id FROM runs WHERE status IN ('queued','running')").fetchone():
            raise HTTPException(409, "A training run is already queued or running.")
        rows = image_rows(db)
        counts = Counter((i["split"], i["reviewed_label"]) for i in rows if i["reviewed_label"])
        if any(counts[("train", label)] < 3 or counts[("validation", label)] < 2 for label in store.LABELS):
            raise HTTPException(409, "Import a dataset sample first. Each class needs at least 3 training and 2 validation images.")
        run_id = uuid.uuid4().hex[:12]
        snapshot = f"snapshots/{run_id}.json"
        payload = [{"id": r["id"], "path": r["path"], "hash": r["hash"], "label": r["reviewed_label"], "split": r["split"]} for r in rows if r["reviewed_label"]]
        (store.DATA / snapshot).write_text(json.dumps(payload, indent=2))
        db.execute("INSERT INTO runs (id,status,step,created_at,snapshot,baseline_id,trainer) VALUES (?,?,?,?,?,?,?)", (run_id, "queued", "Queued", store.now(), snapshot, store.active_id(db), trainer))
    return {"id": run_id, "status": "queued"}


@app.get("/models")
def models():
    with store.connect() as db:
        current = store.active_id(db)
        result = [dict(r) for r in db.execute("SELECT m.*,r.baseline_id FROM models m JOIN runs r ON m.run_id=r.id ORDER BY m.created_at DESC")]
        activated = {r[0] for r in db.execute("SELECT model_id FROM activations")}
    for m in result:
        m["metrics"] = json.loads(m["metrics"])
        m["active"] = m["id"] == current
        m["previously_active"] = m["id"] in activated
        m["can_activate"] = not m["active"] and (m["id"] in activated or (bool(m["eligible"]) and current == m["baseline_id"]))
    return result


@app.post("/models/{model_id}/activate")
def activate(model_id: str):
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT m.*,r.baseline_id FROM models m JOIN runs r ON r.id=m.run_id WHERE m.id=?", (model_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Model not found.")
        old = store.active_id(db)
        rollback = db.execute("SELECT 1 FROM activations WHERE model_id=?", (model_id,)).fetchone()
        if not rollback and (not row["eligible"] or old != row["baseline_id"]):
            raise HTTPException(409, "This candidate did not pass the gate against the current model. Start a new run.")
        model = load_model(model_id, db)
        sample = db.execute("SELECT path FROM images WHERE split='validation' LIMIT 1").fetchone()
        if not sample:
            raise HTTPException(409, "A validation image is required to check this model.")
        ml.predict(model, store.DATA / sample["path"])
        db.execute("INSERT OR REPLACE INTO settings VALUES ('active_model',?)", (model_id,))
        db.execute("INSERT INTO activations (model_id,previous_id,created_at) VALUES (?,?,?)", (model_id, old, store.now()))
    return {"active_model": model_id}
