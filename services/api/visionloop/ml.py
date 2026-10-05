"""A deliberately small CPU baseline; scores are not calibrated confidence.

Supports both the original baseline and optional frozen pretrained ViT features.
Training and inference share the transformation saved with each model.
"""
import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

from .store import LABELS

ALGORITHMS = {
    "baseline": "CPU baseline · image features + logistic regression",
    "vit": "Vision Transformer · frozen DeiT-Tiny + logistic regression",
}


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


def fit_model(paths, labels, trainer="baseline"):
    if trainer == "vit":
        from .vit import fit
        return fit(paths, labels)
    if trainer != "baseline":
        raise ValueError(f"Unknown trainer: {trainer}")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1, random_state=42))
    model.fit(np.stack([features(p) for p in paths]), labels)
    return model


def model_inputs(model, paths):
    from .vit import VisionTransformerModel
    if isinstance(model, VisionTransformerModel):
        return model.head, model.features(paths)
    return model, np.stack([features(p) for p in paths])


def evaluate(model, paths, labels, include_per_class=False):
    classifier, inputs = model_inputs(model, paths)
    predicted = classifier.predict(inputs)
    metrics = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, labels=LABELS, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=LABELS).tolist(),
        "labels": LABELS, "samples": len(labels),
    }
    if include_per_class:
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predicted, labels=LABELS, zero_division=0)
        metrics["per_class"] = [{"label": label, "precision": float(precision[i]),
                                 "recall": float(recall[i]), "f1": float(f1[i]),
                                 "samples": int(support[i])} for i, label in enumerate(LABELS)]
    return metrics


def predict(model, path):
    classifier, inputs = model_inputs(model, [path])
    probabilities = classifier.predict_proba(inputs)[0]
    return sorted([{"label": str(label), "score": float(score)} for label, score in zip(model.classes_, probabilities)], key=lambda x: x["score"], reverse=True)
