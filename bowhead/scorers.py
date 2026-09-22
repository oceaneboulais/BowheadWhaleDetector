"""Scorer adapters — wrap each pipeline behind the common ``Scorer`` interface.

Currently: the custom CNN (``CNNScorer``) and the AE+kNN baseline
(``build_ae_knn_scorer``, backed by ``bowhead.eval.ae_knn_baseline.AEKNNScorer``).
BirdNET / Perch probes get their own adapters here as those benchmarks are built,
so ``compare_pipelines`` can score them all identically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bowhead.data.dataset import per_sample_minmax
from bowhead.eval.ae_knn_baseline import AEKNNScorer, load_training_embeddings
from bowhead.models.custom_cnn import EncoderClassifier
from bowhead.models.encoder import load_ae_encoder


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


def build_ae_knn_scorer(
    train_embeddings_path: str | Path,
    ae_checkpoint_path: str | Path,
    k: int = 40,
    label_source: str = "cleaned",
    min_duration: float | None = None,
    device: str = "cpu",
) -> AEKNNScorer:
    """One-liner factory: loads the training pool + AE encoder and returns an
    ``AEKNNScorer`` ready to plug into ``compare_pipelines``/``evaluate_scorer``
    alongside ``CNNScorer`` on the SAME raw-image test set.

    Note: this requires raw spectrogram images (not precomputed latents) for
    whatever set you score, since ``compare_pipelines`` resamples/scores by
    image array. If you already have precomputed latents (e.g. from a MATLAB
    export), call ``AEKNNScorer.score_embeddings`` directly instead — see
    ``bowhead/benchmark/run_ae_knn_curves.py`` for that path.
    """
    train = load_training_embeddings(train_embeddings_path)
    encoder = load_ae_encoder(ae_checkpoint_path, device=device)
    return AEKNNScorer(train, k=k, label_source=label_source,
                       min_duration=min_duration, encoder=encoder)

