"""Local fine-tuning experiment: last DeiT block, final norm, and head.

Recipe ("linear probe, then fine-tune"):
1. Hold out a small, class-balanced stopping set from the training images.
2. Fit a logistic regression on frozen features and copy it into the head, so
   fine-tuning starts from a working classifier instead of a random one.
3. Train the last block and norm at a low learning rate and the head at a higher
   one, with light augmentation (horizontal flips, mild crops).
4. Keep the epoch with the lowest stopping-set loss; stop after `patience` epochs
   without improvement. Epoch 0 (the linear-probe start) is also a candidate.

Validation images are never used here. They stay reserved for comparing models.
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import store
from .vit import BACKBONE, dependencies, encode

DEFAULT_CONFIG = {
    "version": 2, "max_epochs": 12, "patience": 3, "batch_size": 8, "seed": 42,
    "block_learning_rate": 0.0001, "head_learning_rate": 0.001, "weight_decay": 0.01,
    "trainable_scope": "last_block_norm_and_head", "head_init": "linear_probe",
    "augmentation": "hflip_and_crop_75_100", "stopping_holdout": 0.15,
}


@dataclass
class FineTunedModel:
    weights: dict
    data_config: dict
    labels: list
    training_config: dict
    training_losses: list
    backbone_name: str = BACKBONE
    # Defaults keep models saved by the first fine-tuning version loadable.
    holdout_losses: list = None
    best_epoch: int = None
    holdout_samples: int = 0
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


def stratified_holdout(labels, fraction, seed):
    """Indices to hold out: about `fraction` of each class, at least one, keeping two for training."""
    rng = np.random.default_rng(seed)
    held = []
    for label in sorted(set(labels)):
        members = [i for i, value in enumerate(labels) if value == label]
        count = min(max(1, round(fraction * len(members))), len(members) - 2)
        if count > 0:
            held += [int(i) for i in rng.permutation(members)[:count]]
    return sorted(held)


def linear_probe_weights(features, labels):
    """Fit the frozen-feature classifier and express it as one linear layer in LABELS order.

    StandardScaler + LogisticRegression is z = (x - mean) / scale, logits = W z + b,
    which equals (W / scale) x + (b - W @ (mean / scale)).
    """
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.1, random_state=42))
    probe.fit(features, labels)
    scaler, classifier = probe.named_steps["standardscaler"], probe.named_steps["logisticregression"]
    weight = classifier.coef_ / scaler.scale_
    bias = classifier.intercept_ - weight @ scaler.mean_
    order = [list(classifier.classes_).index(label) for label in store.LABELS]
    return weight[order], bias[order], probe


def _load(path, transform):
    with Image.open(path) as image:
        return transform(ImageOps.exif_transpose(image).convert("RGB"))


def fit(paths, labels, config=None, on_progress=None):
    config = dict(DEFAULT_CONFIG if config is None else config)
    if config != DEFAULT_CONFIG:
        raise ValueError("Unsupported fine-tuning settings. Start a new run with the current preset.")
    torch, timm = dependencies()
    from torchvision import transforms

    # CPU-only and seeded; always starts from pretrained weights, never from the active model.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        model = timm.create_model(BACKBONE, pretrained=True, num_classes=len(store.LABELS),
                                  cache_dir=str(store.DATA / "pretrained"))
        model.requires_grad_(False).eval()
        data_config = timm.data.resolve_model_data_config(model)
        evaluate_transform = timm.data.create_transform(**data_config, is_training=False)
        interpolation = transforms.InterpolationMode(data_config.get("interpolation", "bicubic"))
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(data_config["input_size"][1:], scale=(0.75, 1.0), ratio=(0.9, 1.1),
                                         interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_config["mean"], data_config["std"]),
        ])

        held = set(stratified_holdout(labels, config["stopping_holdout"], config["seed"]))
        fit_paths = [p for i, p in enumerate(paths) if i not in held]
        fit_labels = [l for i, l in enumerate(labels) if i not in held]
        holdout_images = torch.stack([_load(paths[i], evaluate_transform) for i in sorted(held)])
        holdout_targets = torch.tensor([store.LABELS.index(labels[i]) for i in sorted(held)])
        targets = torch.tensor([store.LABELS.index(label) for label in fit_labels], dtype=torch.long)

        # 1. Linear probe on frozen features, copied into the head.
        with torch.inference_mode():
            features = np.concatenate([
                model.forward_head(model.forward_features(torch.stack([_load(p, evaluate_transform) for p in fit_paths[i:i + 16]])), pre_logits=True).numpy()
                for i in range(0, len(fit_paths), 16)])
        weight, bias, _ = linear_probe_weights(features, fit_labels)
        with torch.no_grad():
            model.head.weight.copy_(torch.from_numpy(weight).float())
            model.head.bias.copy_(torch.from_numpy(bias).float())

        criterion = torch.nn.CrossEntropyLoss()

        def holdout_loss():
            model.eval()
            with torch.inference_mode():
                return float(criterion(model(holdout_images), holdout_targets))

        # 2. Unfreeze the last block, final norm and head, with separate learning rates.
        trainable = [model.blocks[-1], model.norm, model.head]
        for module in trainable:
            module.requires_grad_(True)
        head_parameters = list(model.head.parameters())
        block_parameters = [p for m in trainable[:2] for p in m.parameters()]
        optimizer = torch.optim.AdamW([
            {"params": block_parameters, "lr": config["block_learning_rate"]},
            {"params": head_parameters, "lr": config["head_learning_rate"]},
        ], weight_decay=config["weight_decay"])

        best = holdout_loss()
        holdout_losses, training_losses = [best], []
        best_epoch, best_state, without_improvement = 0, {k: v.clone() for k, v in model.state_dict().items()}, 0
        for epoch in range(1, config["max_epochs"] + 1):
            for module in trainable:
                module.train()
            order = torch.randperm(len(fit_paths)).tolist()
            loss_sum = 0.0
            for start in range(0, len(order), config["batch_size"]):
                indices = order[start:start + config["batch_size"]]
                images = torch.stack([_load(fit_paths[i], train_transform) for i in indices])
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), targets[indices])
                if not torch.isfinite(loss):
                    raise ValueError("Fine-tuning produced a non-finite loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(block_parameters + head_parameters, 1.0)
                optimizer.step()
                loss_sum += loss.item() * len(indices)
            training_losses.append(loss_sum / len(fit_paths))
            holdout_losses.append(holdout_loss())
            if holdout_losses[-1] < best - 1e-4:
                best, best_epoch, without_improvement = holdout_losses[-1], epoch, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                without_improvement += 1
            if on_progress:
                on_progress(epoch, config["max_epochs"], training_losses[-1], holdout_losses[-1])
            if without_improvement >= config["patience"]:
                break

        # 3. Keep the best epoch (0 means the linear-probe start was best).
        model.load_state_dict(best_state)
        model.requires_grad_(False).eval()
        return FineTunedModel(
            weights={k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()},
            data_config=data_config, labels=list(store.LABELS), training_config=config,
            training_losses=training_losses, holdout_losses=holdout_losses, best_epoch=best_epoch,
            holdout_samples=len(held), _runtime=(model, evaluate_transform))
