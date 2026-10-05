"""Download a bounded learning sample through Hugging Face's dataset viewer.

This downloads the viewer's image renditions, not the entire original dataset.
All selected rows come from the upstream train split. Our fixed local split
reserves validation/test examples before any training; upstream test is untouched.
"""
import argparse
import json
import hashlib
import random
import uuid
import time
from collections import Counter
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from . import store
from .images import MAX_BYTES, save_image


def download(url, limit=MAX_BYTES):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "VisionLoop-learning-demo/0.1"}), timeout=40) as response:
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise ValueError("Remote file exceeds the download size limit.")
                return data
        except HTTPError as exc:
            if attempt == 3 or (exc.code < 500 and exc.code != 429):
                raise
            retry_after = exc.headers.get("Retry-After", "")
            delay = min(60, max(2 ** attempt, int(retry_after))) if retry_after.isdigit() else (30 if exc.code == 429 else 2 ** attempt)
            print(f"Remote service returned {exc.code}; retrying in {delay}s", flush=True)
            time.sleep(delay)
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def fetch_page(offset, length):
    query = urlencode({"dataset": store.DATASET, "config": "default", "split": "train",
                       "offset": offset, "length": length})
    return json.loads(download("https://datasets-server.huggingface.co/rows?" + query,
                               limit=4 * 1024 * 1024))


def candidates(max_rows, seed):
    """Scan seeded page locations across the split, then order rows by a stable hash."""
    first = fetch_page(0, min(50, max_rows))
    total = first["num_rows_total"]
    rows = list(first.get("rows", []))
    offsets = list(range(50, total, 50))
    random.Random(seed).shuffle(offsets)
    scanned = len(rows)
    for offset in offsets:
        if scanned >= max_rows:
            break
        time.sleep(2)  # Pace metadata requests to reduce rate-limit pressure.
        page = fetch_page(offset, min(50, max_rows - scanned, total - offset))
        batch = page.get("rows", [])
        rows.extend(batch)
        scanned += len(batch)
        print(f"Scanned {scanned}/{min(max_rows, total)} rows", flush=True)
    unique = {item["row_idx"]: item for item in rows}
    ordered = sorted(unique.values(), key=lambda item: hashlib.sha256(
        f"{seed}:{item['row_idx']}".encode()).digest())
    return ordered, total


def import_sample(per_class=30, max_rows=1000, seed=42):
    store.init()
    prefix = f"huggingface:{store.DATASET}:train:"
    with store.connect() as db:
        existing = [dict(r) for r in db.execute("SELECT * FROM images WHERE source LIKE ?", (prefix + "%",))]
    counts = Counter(r["label"] for r in existing)
    sources = {r["source"] for r in existing}
    report = {"dataset": store.DATASET, "created_at": store.now(), "seed": seed,
              "per_class": per_class, "max_rows": max_rows, "scanned": 0,
              "added": [], "skipped_duplicates": 0, "skipped_invalid": 0}
    if all(counts[label] >= per_class for label in store.LABELS):
        rows, total = [], None
    else:
        rows, total = candidates(max_rows, seed)
    report.update(scanned=len(rows), upstream_rows=total)
    for item in rows:
        if all(counts[label] >= per_class for label in store.LABELS):
            break
        row = item["row"]
        label = row.get("type")
        if label not in store.LABELS or counts[label] >= per_class:
            continue
        source = prefix + str(item["row_idx"])
        if source in sources:
            continue
        # Keep the existing 70/20/10 cycle. Never reassign an imported image.
        index = counts[label] % 10
        split = "train" if index < 7 else "validation" if index < 9 else "test"
        try:
            image = save_image(download(row["image"]["src"]), f"{label}-{item['row_idx']}.jpg", source, label, split)
        except (ValueError, KeyError) as exc:
            report["skipped_invalid"] += 1
            print(f"Skipping row {item['row_idx']}: {exc}", flush=True)
            continue
        if image["source"] != source:
            report["skipped_duplicates"] += 1
            continue
        counts[label] += 1
        sources.add(source)
        report["added"].append({"id": image["id"], "row": item["row_idx"], "label": label, "split": image["split"]})
        print(f"Imported {label}: {counts[label]}/{per_class} ({split})", flush=True)
    report["counts"] = {label: counts[label] for label in store.LABELS}
    report["shortfalls"] = {label: per_class - counts[label] for label in store.LABELS if counts[label] < per_class}
    folder = store.DATA / "imports"
    folder.mkdir(exist_ok=True)
    report_path = folder / f"{uuid.uuid4().hex[:12]}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Added {len(report['added'])} images. Counts: {report['counts']}", flush=True)
    print(f"Import report: {report_path}", flush=True)
    if report["shortfalls"]:
        print(f"Below target: {report['shortfalls']}. Increase --max-rows to sample more rows.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=30, help="Target total imported images per class, not number to add")
    parser.add_argument("--max-rows", type=int, default=1000, help="Maximum metadata rows to scan across seeded page locations")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible sampling seed; existing splits never change")
    args = parser.parse_args()
    if not 10 <= args.per_class <= 500 or not 50 <= args.max_rows <= 10000:
        parser.error("Use 10–500 images per class and 50–10000 scanned rows.")
    try:
        import_sample(args.per_class, args.max_rows, args.seed)
    except (URLError, TimeoutError) as exc:
        parser.exit(1, f"Import stopped: {exc}. Saved images are retained; retry this command later.\n")
    print("Dataset: https://huggingface.co/datasets/" + store.DATASET)
    print("Listed license: CC BY 4.0. Local renditions are resized/re-encoded. See ATTRIBUTION.md.")


if __name__ == "__main__":
    main()
