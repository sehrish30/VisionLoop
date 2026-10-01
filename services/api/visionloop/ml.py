"""A deliberately small CPU baseline; scores are not calibrated confidence.

Replace features/model construction with a pretrained ViT to follow the course.
Training and inference always share this exact image transformation.
"""
import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from .store import LABELS


def features(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.pad(image, (32, 32), color=(255, 255, 255))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    gray = pixels.mean(axis=2)
    horizontal = np.abs(np.diff(gray, axis=1))
    vertical = np.abs(np.diff(gray, axis=0))
    histograms = [np.histogram(pixels[:, :, c], bins=16, range=(0, 1), density=True)[0] for c in range(3)]
    return np.concatenate([gray.flatten(), horizontal.mean(axis=0), vertical.mean(axis=1), *histograms])


def fit_model(paths, labels):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1, random_state=42))
    model.fit(np.stack([features(p) for p in paths]), labels)
    return model


def evaluate(model, paths, labels):
    predicted = model.predict(np.stack([features(p) for p in paths]))
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, labels=LABELS, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=LABELS).tolist(),
        "labels": LABELS, "samples": len(labels),
    }


def predict(model, path):
    probabilities = model.predict_proba(features(path).reshape(1, -1))[0]
    return sorted([{"label": str(label), "score": float(score)} for label, score in zip(model.classes_, probabilities)], key=lambda x: x["score"], reverse=True)
