"""Download a bounded learning sample through Hugging Face's dataset viewer.

This downloads the viewer's image renditions, not the entire original dataset.
All selected rows come from the upstream train split. Our fixed local split
reserves validation/test examples before any training; upstream test is untouched.
"""
import argparse
import json
import time
from collections import Counter
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--max-rows", type=int, default=1000)
    args = parser.parse_args()
    if not 10 <= args.per_class <= 500 or not 50 <= args.max_rows <= 10000:
        parser.error("Use 10–500 images per class and 50–10000 scanned rows.")
    store.init()
    with store.connect() as db:
        counts = Counter(r[0] for r in db.execute("SELECT label FROM images WHERE source LIKE 'huggingface:%'"))
    added = 0
    for offset in range(0, args.max_rows, 50):
        if all(counts[label] >= args.per_class for label in store.LABELS):
            break
        query = urlencode({"dataset": store.DATASET, "config": "default", "split": "train", "offset": offset, "length": min(50, args.max_rows - offset)})
        page = json.loads(download("https://datasets-server.huggingface.co/rows?" + query, limit=4 * 1024 * 1024))
        if not page.get("rows"):
            break
        for item in page["rows"]:
            row = item["row"]
            label = row.get("type")
            if label not in store.LABELS or counts[label] >= args.per_class:
                continue
            source = f"huggingface:{store.DATASET}:train:{item['row_idx']}"
            with store.connect() as db:
                if db.execute("SELECT 1 FROM images WHERE source=?", (source,)).fetchone():
                    continue
            index = counts[label] % 10
            split = "train" if index < 7 else "validation" if index < 9 else "test"
            try:
                image = save_image(download(row["image"]["src"]), f"{label}-{item['row_idx']}.jpg", source, label, split)
            except (ValueError, KeyError) as exc:
                print(f"Skipping row {item['row_idx']}: {exc}", flush=True)
                continue
            if image["source"] == source:
                counts[label] += 1
                added += 1
                print(f"Imported {label}: {counts[label]}/{args.per_class} ({split})", flush=True)
    print(f"Added {added} images. Counts: {dict(counts)}", flush=True)
    if any(counts[label] < args.per_class for label in store.LABELS):
        print("Some classes are below the target. Increase --max-rows to sample more rows.")
    print("Dataset: https://huggingface.co/datasets/" + store.DATASET)
    print("Listed license: CC BY 4.0. Local renditions are resized/re-encoded. See ATTRIBUTION.md.")


if __name__ == "__main__":
    main()
