from collections import Counter

from visionloop import import_dataset as importer, store
from test_workflow import client, photo, seed


def fake_source(monkeypatch, total=200):
    monkeypatch.setattr(importer.time, "sleep", lambda _: None)
    calls = []
    def page(offset, length):
        calls.append((offset, length))
        return {"num_rows_total": total, "rows": [
            {"row_idx": i, "row": {"type": store.LABELS[i % 5], "image": {"src": str(i)}}}
            for i in range(offset, min(offset + length, total))
        ]}
    monkeypatch.setattr(importer, "fetch_page", page)
    monkeypatch.setattr(importer, "download", lambda url: photo(int(url) % 5, int(url) // 5))
    return calls


def test_seeded_scan_is_bounded_and_repeatable(monkeypatch):
    calls = fake_source(monkeypatch, total=525)
    first, _ = importer.candidates(125, 42)
    assert sum(length for _, length in calls) == 125
    assert len(first) == 125
    assert len({r["row_idx"] for r in first}) == 125
    again, _ = importer.candidates(125, 42)
    different, _ = importer.candidates(125, 7)
    assert first == again
    assert first != different
    assert any(offset >= 150 for offset, _ in calls)


def test_import_balances_splits_and_repeat_is_noop(client, monkeypatch):
    fake_source(monkeypatch)
    report = importer.import_sample(10, 200, 42)
    assert len(report["added"]) == 50
    before = client.get("/images").json()
    splits = Counter((r["label"], r["split"]) for r in before)
    for label in store.LABELS:
        assert [splits[(label, s)] for s in ("train", "validation", "test")] == [7, 2, 1]
    assert not report["shortfalls"]
    assert list((store.DATA / "imports").glob("*.json"))
    def unexpected(*args):
        raise AssertionError("Completed quotas should not contact the network")
    monkeypatch.setattr(importer, "fetch_page", unexpected)
    assert not importer.import_sample(10, 200, 7)["added"]
    assert client.get("/images").json() == before


def test_expansion_preserves_existing_splits_and_snapshot(client, monkeypatch):
    fake_source(monkeypatch)
    importer.import_sample(10, 200, 42)
    before = {r["id"]: r["split"] for r in client.get("/images").json()}
    client.post("/runs")
    run = client.get("/runs").json()[0]
    snapshot = (store.DATA / run["snapshot"]).read_bytes()
    report = importer.import_sample(20, 200, 7)
    assert len(report["added"]) == 50
    for r in client.get("/images").json():
        if r["id"] in before:
            assert r["split"] == before[r["id"]]
    assert (store.DATA / run["snapshot"]).read_bytes() == snapshot
    coverage = client.get("/overview").json()["class_splits"]
    assert all(v == {"train": 14, "validation": 4, "test": 2} for v in coverage.values())


def test_duplicate_images_report_shortfall_without_reassignment(client, monkeypatch):
    fake_source(monkeypatch)
    monkeypatch.setattr(importer, "download", lambda url: photo())
    report = importer.import_sample(10, 100, 42)
    assert len(report["added"]) == 1
    assert report["skipped_duplicates"] == 99
    assert report["shortfalls"]
    assert len(client.get("/images").json()) == 1


def test_coverage_uses_latest_review_but_excludes_unlabeled(client):
    seed()
    rows = client.get("/images").json()
    image = next(r for r in rows if r["label"] == "Dress" and r["split"] == "train")
    client.post(f"/images/{image['id']}/review", json={"label": "Shirt"})
    coverage = client.get("/overview").json()["class_splits"]
    assert coverage["Dress"]["train"] == 6
    assert coverage["Shirt"]["train"] == 8
    assert coverage["Dress"]["validation"] == 2


def test_rate_limit_respects_retry_after(monkeypatch):
    import io
    from urllib.error import HTTPError
    delays = []
    calls = []
    def response(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise HTTPError("https://example.org", 429, "Too many requests", {"Retry-After": "12"}, None)
        return io.BytesIO(b"ok")
    monkeypatch.setattr(importer, "urlopen", response)
    monkeypatch.setattr(importer.time, "sleep", delays.append)
    assert importer.download("https://example.org") == b"ok"
    assert delays == [12]
