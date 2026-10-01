# VisionLoop

A local clothing-classification studio: upload → review → train → evaluate → activate.

**Next.js + React + TypeScript** frontend, **FastAPI + Python** API and worker,
SQLite metadata, and local image/model files. No AWS account or paid API is needed.
The initial model is a real CPU baseline using image features and logistic regression;
it is **not a Vision Transformer** and should not be treated as production-quality.

## Start locally

Requirements: Node 20.9+ and Python 3.11+. From this folder:

```sh
npm install
python3 -m venv services/api/.venv
services/api/.venv/bin/python -m pip install -e 'services/api[dev]'
npm run dev
```

Open http://localhost:3000. API documentation: http://127.0.0.1:8000/docs.
`npm run dev` starts the web app, API, and one training worker. Ctrl+C stops all three.
The servers bind to localhost. This single-user learning app has no login system;
do not expose it publicly without authentication and request limits.

## First real model

In another terminal, from the project root:

```sh
npm run data:import -- --per-class 30
```

This downloads a bounded sample (up to 150 images) from the Hugging Face viewer.
It scans up to 1,000 upstream training rows by default, stopping when quotas are met.
If a class is short, increase `--max-rows`. It does not download the full dataset.
Viewer images may be smaller than originals. Network access is needed for import.
Re-running skips already imported rows; duplicate decoded images are deduplicated.

1. Open **Training runs** and click **Start training**.
2. Inspect the actual metrics in **Model library**.
3. Activate a candidate that passes the demo quality gate.
4. Upload a photo in **Classify**. Review or correct the label.
5. Start another run to incorporate reviewed training images.

Each class cycles through 7 training, 2 validation, and 1 reserved test image.
The upstream test split is untouched. The local reserved test set is never used
for selection. Initial sequence-based sampling is for a learning demo, not a
representative benchmark. Inspect class coverage and near-duplicate/product
groups before serious evaluation. Exact pixel duplicates are prevented, but
near-duplicate or same-item photographs are not automatically detected.

The initial gate is macro F1 ≥ 0.20. Later models must exceed the active model's
F1 by 0.01, evaluated on the same snapshot. Passing makes a model eligible;
activation is manual. Previously activated versions may be restored. No accuracy
is promised. Model scores are not calibrated confidence.

## Optional MLflow tracking

Local runs and metrics already appear in the app. To also log to MLflow:

```sh
services/api/.venv/bin/python -m pip install -e 'services/api[mlflow-ui]'
npm run mlflow
```

Restart the app in another terminal with:

```sh
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 npm run dev
```

Open http://127.0.0.1:5001. MLflow has its own database. Tracking is opt-in; if an
explicitly configured tracker fails, the run fails visibly rather than silently
losing its experiment record. The prior active model stays unchanged.

## Structure

```text
apps/web/                 TypeScript UI and server-side API forwarding
services/api/visionloop/  Python API, worker, dataset importer, shared ML functions
services/api/tests/       Local integration tests using synthetic fixtures
cloud/sagemaker/          Non-executed SageMaker reference implementation
docs/architecture.md      Boundaries, records, flows, and next steps
data/                     Generated images, SQLite DB, snapshots, model artifacts
```

`data/` is ignored by Git. Image uploads are re-encoded without EXIF metadata,
limited to 8 MB and 20 megapixels. Only the application's own model artifacts
are loaded; uploaded pickle/joblib models are not accepted.

## Checks

```sh
npm run typecheck
npm run build
npm run test:api
```

Tests use isolated temporary storage and do not download data or contact AWS.

## SageMaker learning path

See `cloud/sagemaker/README.md`. Cloud code is **not validated on AWS** and is
not imported by the app. There is no AWS switch in the UI. The next modeling
milestone is replacing the shared CPU feature/model functions with a pretrained
Vision Transformer, keeping the same upload/review/training contracts.

Dataset credit and changes: [ATTRIBUTION.md](ATTRIBUTION.md).
