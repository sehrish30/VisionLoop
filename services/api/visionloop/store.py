import os
import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("VISIONLOOP_DATA_DIR", ROOT / "data")).resolve()
LABELS = ["Dress", "T-shirt", "Shirt", "Sweater", "Blouse"]
DATASET = "fnauman/fashion-second-hand-front-only-rgb"


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect():
    DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATA / "app.db", timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init():
    for name in ("images", "models", "snapshots"):
        (DATA / name).mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript('''
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY, path TEXT NOT NULL, hash TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL, source TEXT NOT NULL, label TEXT,
            split TEXT NOT NULL DEFAULT 'train', created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES images(id),
            model_id TEXT NOT NULL, label TEXT NOT NULL, scores TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES images(id),
            label TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY, status TEXT NOT NULL, step TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0, error TEXT, metrics TEXT,
            baseline_metrics TEXT, baseline_id TEXT, created_at TEXT NOT NULL,
            finished_at TEXT, heartbeat TEXT, snapshot TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
            path TEXT NOT NULL, metrics TEXT NOT NULL, eligible INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS test_reports (
            id TEXT PRIMARY KEY, model_id TEXT UNIQUE NOT NULL REFERENCES models(id),
            status TEXT NOT NULL, snapshot TEXT NOT NULL, metrics TEXT, error TEXT,
            created_at TEXT NOT NULL, finished_at TEXT, heartbeat TEXT
        );
        CREATE TABLE IF NOT EXISTS activations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, model_id TEXT NOT NULL,
            previous_id TEXT, created_at TEXT NOT NULL
        );
        ''')
        run_columns = {r[1] for r in db.execute("PRAGMA table_info(runs)")}
        if "trainer" not in run_columns:
            db.execute("ALTER TABLE runs ADD COLUMN trainer TEXT NOT NULL DEFAULT 'baseline'")
        if "training_config" not in run_columns:
            db.execute("ALTER TABLE runs ADD COLUMN training_config TEXT NOT NULL DEFAULT '{}'")
        # Also recognize a re-upload of an image already normalized by this app.
        columns = {r[1] for r in db.execute("PRAGMA table_info(images)")}
        if "stored_hash" not in columns:
            db.execute("ALTER TABLE images ADD COLUMN stored_hash TEXT")
        for row in db.execute("SELECT id,path FROM images WHERE stored_hash IS NULL").fetchall():
            path = DATA / row["path"]
            if path.exists():
                db.execute("UPDATE images SET stored_hash=? WHERE id=?", (hashlib.sha256(path.read_bytes()).hexdigest(), row["id"]))


def active_id(db):
    row = db.execute("SELECT value FROM settings WHERE key='active_model'").fetchone()
    return row[0] if row else None
