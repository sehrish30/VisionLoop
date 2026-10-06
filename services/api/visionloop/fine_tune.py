"""Small local fine-tuning experiment: last DeiT block, final norm, and head."""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

from . import store
from .vit import BACKBONE, dependencies, encode

DEFAULT_CONFIG = {"epochs": 3, "learning_rate": 0.0001, "batch_size": 8,
                  "seed": 42, "trainable_scope": "last_block_norm_and_head"}


@dataclass
class FineTunedModel:
    weights: dict
    data_config: dict
    labels: list
    training_config: dict
    training_losses: list
    backbone_name: str = BACKBONE
    _runtime: object = field(default=None, repr=False)

    @property
    def classes_(self):
        return np.asarray(self.labels)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_runtime"] = None
        return state

    def predict_proba(self, paths):
        torch, timm = dependencies()
        if self._runtime is None:
            model = timm.create_model(self.backbone_name, pretrained=False, num_classes=len(self.labels))
            model.load_state_dict({k: torch.from_numpy(v) for k, v in self.weights.items()})
            model.requires_grad_(False).eval()
            transform = timm.data.create_transform(**self.data_config, is_training=False)
            self._runtime = (model, transform)
        logits = encode(*self._runtime, paths)
        return torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    def predict(self, paths):
        return self.classes_[self.predict_proba(paths).argmax(axis=1)]


def fit(paths, labels, config=None, on_progress=None):
    config = dict(DEFAULT_CONFIG if config is None else config)
    if config != DEFAULT_CONFIG:
        raise ValueError("Unsupported fine-tuning settings. Start a new run with the current preset.")
    torch, timm = dependencies()
    # This preset is CPU-only and seeded; it always starts from pretrained weights,
    # never from the active model or any held-out examples.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        model = timm.create_model(BACKBONE, pretrained=True, num_classes=len(store.LABELS),
                                  cache_dir=str(store.DATA / "pretrained"))
        model.requires_grad_(False).eval()
        model.blocks[-1].requires_grad_(True).train()
        model.norm.requires_grad_(True).train()
        model.head.requires_grad_(True).train()
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        targets = torch.tensor([store.LABELS.index(label) for label in labels], dtype=torch.long)
        losses = []
        for epoch in range(config["epochs"]):
            order = torch.randperm(len(paths)).tolist()
            loss_sum = 0.0
            for start in range(0, len(order), config["batch_size"]):
                indices = order[start:start + config["batch_size"]]
                images = []
                for index in indices:
                    with Image.open(paths[index]) as image:
                        images.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(torch.stack(images)), targets[indices])
                if not torch.isfinite(loss):
                    raise ValueError("Fine-tuning produced a non-finite loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(indices)
            losses.append(loss_sum / len(paths))
            if on_progress:
                on_progress(epoch + 1, config["epochs"], losses[-1])
        model.requires_grad_(False).eval()
        return FineTunedModel(
            weights={k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()},
            data_config=data_config, labels=list(store.LABELS), training_config=config,
            training_losses=losses, _runtime=(model, transform))
