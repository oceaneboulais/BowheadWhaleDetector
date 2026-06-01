"""Scorer adapters — wrap each pipeline behind the common ``Scorer`` interface.

Currently: the custom CNN. BirdNET / Perch probes and the AE+kNN baseline get
their own adapters here as those benchmarks are built, so ``compare_pipelines``
can score them all identically.
"""

from __future__ import annotations

import numpy as np
import torch

from bowhead.data.dataset import per_sample_minmax
from bowhead.models.custom_cnn import EncoderClassifier


class CNNScorer:
    """Wrap a trained ``EncoderClassifier`` as a Scorer (images -> P(call))."""

    def __init__(
        self,
        model: EncoderClassifier,
        name: str = "custom_cnn",
        device: str = "cpu",
        batch_size: int = 256,
        normalize: bool = True,
    ) -> None:
        self.model = model.to(device).eval()
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

    @torch.no_grad()
    def score(self, images: np.ndarray) -> np.ndarray:
        if images.ndim == 3:
            images = images[:, None, :, :]
        probs = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start:start + self.batch_size]
            if self.normalize:
                batch = np.stack(
                    [np.stack([per_sample_minmax(ch) for ch in img]) for img in batch]
                )
            else:
                batch = batch.astype(np.float32)
            x = torch.from_numpy(batch).to(self.device)
            probs.append(self.model.predict_proba(x).cpu().numpy())
        return np.concatenate(probs)
