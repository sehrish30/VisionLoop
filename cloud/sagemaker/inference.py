"""Reference inference hooks for a future SageMaker scikit-learn container.

Requires packaging the visionloop Python module and matching dependency versions
with model.joblib. This module does not create models, endpoints, or AWS clients.
Not validated on AWS.
"""
import json
import tempfile
from pathlib import Path
import joblib
from visionloop.ml import predict


def model_fn(model_dir):
    return joblib.load(Path(model_dir) / "model.joblib")


def input_fn(request_body, content_type):
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("Send a JPEG, PNG, or WebP image.")
    if len(request_body) > 8 * 1024 * 1024:
        raise ValueError("Image exceeds 8 MB.")
    return request_body


def predict_fn(input_data, model):
    with tempfile.NamedTemporaryFile(suffix=".image") as image:
        image.write(input_data)
        image.flush()
        return predict(model, image.name)


def output_fn(prediction, accept):
    if accept != "application/json":
        raise ValueError("Only application/json responses are supported.")
    return json.dumps({"scores": prediction})
