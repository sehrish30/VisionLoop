import json
import logging
import os
import threading
import time
from datetime import datetime, timezone, timedelta
import joblib

from . import store, ml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def progress(run_id, step, value):
    with store.connect() as db:
        db.execute("UPDATE runs SET step=?,progress=?,heartbeat=? WHERE id=?", (step, value, store.now(), run_id))


def heartbeat(stop, run_id, test_report=False):
    while not stop.wait(5):
        with store.connect() as db:
            if test_report:
                db.execute("UPDATE test_reports SET heartbeat=? WHERE id=?", (store.now(), run_id))
            else:
                db.execute("UPDATE runs SET heartbeat=? WHERE id=?", (store.now(), run_id))
            db.execute("INSERT OR REPLACE INTO settings VALUES ('worker_heartbeat',?)", (store.now(),))


def execute(run):
    run_id = run["id"]
    stop = threading.Event()
    thread = threading.Thread(target=heartbeat, args=(stop, run_id), daemon=True)
    thread.start()
    try:
        progress(run_id, "Prepare dataset", 15)
        rows = json.loads((store.DATA / run["snapshot"]).read_text())
        train = [r for r in rows if r["split"] == "train"]
        validation = [r for r in rows if r["split"] == "validation"]
        # The reserved test partition is never used to train or select models.
        paths = lambda subset: [store.DATA / r["path"] for r in subset]
        labels = lambda subset: [r["label"] for r in subset]
        trainer = run.get("trainer", "baseline")
        step = "Load pretrained transformer and train classifier" if trainer == "vit" else "Train classifier"
        progress(run_id, step, 35)
        model = ml.fit_model(paths(train), labels(train), trainer)
        progress(run_id, "Evaluate candidate", 65)
        metrics = ml.evaluate(model, paths(validation), labels(validation))
        metrics["training_samples"] = len(train)
        baseline_metrics = None
        if run["baseline_id"]:
            with store.connect() as db:
                baseline = db.execute("SELECT path FROM models WHERE id=?", (run["baseline_id"],)).fetchone()
            baseline_metrics = ml.evaluate(joblib.load(store.DATA / baseline["path"]), paths(validation), labels(validation))
        progress(run_id, "Compare models", 80)
        threshold = baseline_metrics["macro_f1"] + 0.01 if baseline_metrics else 0.20
        eligible = metrics["macro_f1"] >= threshold
        metrics["gate_threshold"] = threshold
        metrics["gate_passed"] = eligible
        metrics["algorithm"] = ml.ALGORITHMS[trainer]
        metrics["trainer"] = trainer
        if trainer == "vit":
            metrics["backbone"] = model.backbone_name
            metrics["backbone_frozen"] = True
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("VisionLoop")
            with mlflow.start_run(run_name=run_id):
                mlflow.log_params({"classifier": "logistic-regression", "trainer": trainer, "dataset_snapshot": run_id, "C": 0.1})
                mlflow.log_metrics({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
                mlflow.log_artifact(str(store.DATA / run["snapshot"]))
        progress(run_id, "Save model", 95)
        model_id = "vl-" + run_id
        path = f"models/{model_id}.joblib"
        joblib.dump(model, store.DATA / path)
        with store.connect() as db:
            db.execute("INSERT INTO models VALUES (?,?,?,?,?,?)", (model_id, run_id, path, json.dumps(metrics), int(eligible), store.now()))
            db.execute("UPDATE runs SET status='completed',step='Complete',progress=100,metrics=?,baseline_metrics=?,finished_at=? WHERE id=?", (json.dumps(metrics), json.dumps(baseline_metrics) if baseline_metrics else None, store.now(), run_id))
    except Exception as exc:
        logging.exception("Training run %s failed", run_id)
        with store.connect() as db:
            db.execute("UPDATE runs SET status='failed',step='Failed',error=?,finished_at=? WHERE id=?", (str(exc), store.now(), run_id))
    finally:
        stop.set()
        thread.join(timeout=6)


def execute_test_report(report):
    stop = threading.Event()
    thread = threading.Thread(target=heartbeat, args=(stop, report["id"], True), daemon=True)
    thread.start()
    try:
        rows = json.loads(report["snapshot"])
        with store.connect() as db:
            model = db.execute("SELECT path FROM models WHERE id=?", (report["model_id"],)).fetchone()
            for row in rows:
                image = db.execute("SELECT split,hash,label FROM images WHERE id=?", (row["id"],)).fetchone()
                if not image or image["split"] != "test" or image["hash"] != row["hash"] or image["label"] != row["label"]:
                    raise ValueError("Reserved test data no longer matches the saved snapshot.")
        loaded = joblib.load(store.DATA / model["path"])
        metrics = ml.evaluate(loaded, [store.DATA / r["path"] for r in rows],
                              [r["label"] for r in rows], include_per_class=True)
        with store.connect() as db:
            db.execute("UPDATE test_reports SET status='completed',metrics=?,finished_at=? WHERE id=?",
                       (json.dumps(metrics), store.now(), report["id"]))
    except Exception as exc:
        logging.exception("Test report %s failed", report["id"])
        with store.connect() as db:
            db.execute("UPDATE test_reports SET status='failed',error=?,finished_at=? WHERE id=?",
                       (str(exc), store.now(), report["id"]))
    finally:
        stop.set()
        thread.join(timeout=6)


def tick():
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("INSERT OR REPLACE INTO settings VALUES ('worker_heartbeat',?)", (store.now(),))
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
        db.execute("UPDATE runs SET status='failed',step='Interrupted',error='Worker stopped before completion. Start a new run.',finished_at=? WHERE status='running' AND heartbeat < ?", (store.now(), cutoff))
        db.execute("UPDATE test_reports SET status='failed',error='Worker stopped before completion. Retry the report.',finished_at=? WHERE status='running' AND heartbeat < ?", (store.now(), cutoff))
        if db.execute("SELECT 1 FROM runs WHERE status='running'").fetchone() or db.execute("SELECT 1 FROM test_reports WHERE status='running'").fetchone():
            return False
        row = db.execute("SELECT * FROM runs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
        is_report = row is None
        if is_report:
            row = db.execute("SELECT * FROM test_reports WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
        if not row:
            return False
        run = dict(row)
        if is_report:
            db.execute("UPDATE test_reports SET status='running',heartbeat=? WHERE id=?", (store.now(), run["id"]))
        else:
            db.execute("UPDATE runs SET status='running',heartbeat=? WHERE id=?", (store.now(), run["id"]))
    if is_report:
        execute_test_report(run)
    else:
        execute(run)
    return True


def main():
    store.init()
    logging.info("Local worker ready. AWS execution is not available in this worker.")
    try:
        while True:
            tick()
            time.sleep(2)
    except KeyboardInterrupt:
        logging.info("Worker stopped.")


if __name__ == "__main__":
    main()
