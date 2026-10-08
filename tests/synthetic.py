"""Synthetic, perfectly separable text data with hand-made embeddings.

A randomly initialised encoder puts every text in nearly the same place, so
accuracy tests replace `_get_embeddings` with this. Each class sits around its
own axis; `noise` controls how spread out a class is. With these geometries a
good classifier should be close to perfect, so any shortfall is the
classifier's, not the data's.
"""

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_classifier import AdaptiveClassifier

DIM = 32


def point(axis: int, noise: float, seed: int) -> torch.Tensor:
    v = torch.zeros(DIM)
    v[axis % DIM] = 1.0
    if noise:
        g = torch.Generator().manual_seed(seed)
        v = v + noise * torch.randn(DIM, generator=g)
    return F.normalize(v, dim=0)


class Dataset:
    """Texts mapped to embeddings; install with `patch(monkeypatch)`."""

    def __init__(self):
        self.store = {}

    def add(self, name: str, axis: int, noise: float, seed: int) -> str:
        self.store[name] = point(axis, noise, seed)
        return name

    def classes(self, n_per: int, n_classes: int = 3, noise: float = 0.1,
                seed: int = 0, prefix: str = "tr"):
        """Return (texts, labels): `n_per` examples around each class axis."""
        texts, labels = [], []
        for c in range(n_classes):
            for i in range(n_per):
                texts.append(self.add(f"{prefix}{c}_{i}", c, noise, seed * 100003 + c * 1009 + i))
                labels.append(f"C{c}")
        return texts, labels

    def patch(self, monkeypatch):
        store = self.store
        monkeypatch.setattr(AdaptiveClassifier, "_get_embeddings",
                            lambda self, texts: [store[t] for t in texts])
        return self


def accuracy(clf: AdaptiveClassifier, texts, labels) -> float:
    return float(np.mean([clf.predict(t)[0][0] == label for t, label in zip(texts, labels)]))
