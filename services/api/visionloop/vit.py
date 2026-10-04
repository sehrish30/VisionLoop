"""Frozen pretrained DeiT features with a freshly fitted clothing classifier.

Artifacts include the backbone weights and preprocessing configuration, so
prediction and activation never download weights. Optional imports stay lazy.
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BACKBONE = "deit_tiny_patch16_224.fb_in1k"


def dependencies():
    try:
        import torch
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "Vision Transformer dependencies are missing. From the project folder run: "
            "services/api/.venv/bin/python -m pip install -e 'services/api[vit]'"
        ) from exc
    return torch, timm


def encode(backbone, transform, paths):
    torch, _ = dependencies()
    batches = []
    backbone.eval()
    with torch.inference_mode():
        for start in range(0, len(paths), 8):
            images = []
            for path in paths[start:start + 8]:
                with Image.open(path) as image:
                    images.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
            batches.append(backbone(torch.stack(images)).cpu().numpy())
    return np.concatenate(batches)


@dataclass
class VisionTransformerModel:
    head: object
    weights: dict
    data_config: dict
    backbone_name: str = BACKBONE
    _runtime: object = field(default=None, repr=False)

    @property
    def classes_(self):
        return self.head.classes_

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_runtime"] = None
        return state

    def features(self, paths):
        if self._runtime is None:
            torch, timm = dependencies()
            backbone = timm.create_model(self.backbone_name, pretrained=False, num_classes=0)
            backbone.load_state_dict({k: torch.from_numpy(v) for k, v in self.weights.items()})
            backbone.requires_grad_(False).eval()
            transform = timm.data.create_transform(**self.data_config, is_training=False)
            self._runtime = (backbone, transform)
        return encode(*self._runtime, paths)


def fit(paths, labels):
    from . import store
    _, timm = dependencies()
    backbone = timm.create_model(BACKBONE, pretrained=True, num_classes=0, cache_dir=str(store.DATA / "pretrained"))
    backbone.requires_grad_(False).eval()
    config = timm.data.resolve_model_data_config(backbone)
    transform = timm.data.create_transform(**config, is_training=False)
    head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1, random_state=42))
    head.fit(encode(backbone, transform, paths), labels)
    return VisionTransformerModel(
        head=head,
        weights={k: v.cpu().numpy().copy() for k, v in backbone.state_dict().items()},
        data_config=config,
        _runtime=(backbone, transform),
    )
