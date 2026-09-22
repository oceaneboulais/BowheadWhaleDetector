"""AE + k-nearest-neighbor baseline — "the existing baseline" from the README's
benchmark table, and the pipeline Aaron's advisor email describes ("an
autoencoder followed by a nearest neighbor clustering algorithm").

Score = fraction of a query's k nearest neighbors, in the AE's 32-D latent
space, that are labeled calls in a reference *training* pool. This module
loads that reference pool (exported from MATLAB) and wraps the kNN vote as a
``Scorer`` (see ``bowhead/eval/evaluate.py``), so it plugs into the same
comparison harness as the CNN scorers.

The training pool retains BOTH the "cleaned" (reviewed) label and the
"original" (pre-review) label for every point, so the reviewed-vs-original
dataset comparison described in the advisor's email (label_source=
"cleaned" vs "original") can be reproduced directly from one file, without
re-exporting two separate MATLAB datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors


@dataclass
class TrainingEmbeddings:
    """The kNN reference pool: one row per training spectrogram."""

    latent: np.ndarray            # (N, 32) AE latent vectors
    iscall: np.ndarray            # (N,) int, cleaned/reviewed call label
    iscall_original: np.ndarray   # (N,) int, label before dataset review
    type: np.ndarray              # (N,) float, cleaned call sub-type (may be NaN/negative)
    type_org: np.ndarray          # (N,) int, pre-review call sub-type (0-7)
    ischanged: np.ndarray         # (N,) bool, True if review changed this label
    duration: np.ndarray          # (N,) float seconds, event duration (``duration1``)
    filenames: np.ndarray         # (N,) object, original .mat filename per row

    def __len__(self) -> int:
        return len(self.latent)

    def label_for(self, source: str) -> np.ndarray:
        if source == "cleaned":
            return self.iscall
        if source == "original":
            return self.iscall_original
        raise ValueError(f"label source must be 'cleaned' or 'original', got {source!r}")


def load_training_embeddings(path: str | Path) -> TrainingEmbeddings:
    """Load a MATLAB-exported latent-embeddings training pool (e.g.
    ``latent_embeddings_3d_train_MATLAB.mat``).

    Expects top-level ``latent_embeddings`` (N,32), ``original_filenames``
    (N,), and a ``features`` struct with ``iscall``, ``type``, ``type_org``,
    ``ischanged``, ``duration1`` fields (one value per row).
    """
    d = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    feat = d["features"]

    # Re-derive both labels from type/type_org (as master_evaluate_autoencoder_performance.m
    # does) instead of trusting the precomputed `iscall` field, which can go stale
    # relative to `type` (e.g. type==12 rows stored as iscall=1 but excluded by
    # MATLAB's live `type>0 & type<12` formula).
    type_ = np.asarray(feat.type, dtype=float)
    type_org = np.asarray(feat.type_org).astype(int)
    return TrainingEmbeddings(
        latent=np.asarray(d["latent_embeddings"], dtype=np.float32),
        iscall=((type_ > 0) & (type_ < 12)).astype(int),
        iscall_original=((type_org > 0) & (type_org < 12)).astype(int),
        type=type_,
        type_org=type_org,
        ischanged=np.asarray(feat.ischanged).astype(bool),
        duration=np.asarray(feat.duration1, dtype=float),
        filenames=np.asarray(d["original_filenames"]),
    )


class AEKNNScorer:
    """AE latent-space kNN vote: score = fraction of k nearest training
    neighbors labeled a call. Matches the ``Scorer`` protocol
    (``bowhead/eval/evaluate.py``) by embedding images through an AE encoder
    before voting; call ``score_embeddings`` directly if latents are already
    computed.
    """

    def __init__(
        self,
        train: TrainingEmbeddings,
        k: int = 40,
        label_source: str = "cleaned",
        min_duration: float | None = None,
        encoder=None,
        name: str | None = None,
    ) -> None:
        self.train = train
        self.k = k
        self.label_source = label_source
        self.encoder = encoder
        self.name = name or f"ae_knn_k{k}_{label_source}"

        labels = train.label_for(label_source)
        keep = np.ones(len(train), dtype=bool)
        if min_duration is not None:
            keep &= train.duration >= min_duration
        self._labels = labels[keep]
        self._nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(train.latent[keep])

    def score_embeddings(self, latents: np.ndarray) -> np.ndarray:
        """Vote fraction for already-computed 32-D latent vectors (N, 32)."""
        _, idx = self._nn.kneighbors(np.asarray(latents, dtype=np.float32))
        return self._labels[idx].mean(axis=1)

    def score(self, images: np.ndarray) -> np.ndarray:
        """Vote fraction for raw spectrogram images (N, 1, H, W); requires an
        ``encoder`` to have been passed at construction time."""
        if self.encoder is None:
            raise ValueError("AEKNNScorer.score(images) requires an `encoder`; "
                              "pass one at construction, or call score_embeddings "
                              "directly with precomputed latents.")
        import torch
        from bowhead.data.dataset import per_sample_minmax

        if images.ndim == 3:
            images = images[:, None, :, :]
        batch = np.stack(
            [np.stack([per_sample_minmax(ch) for ch in img]) for img in images]
        )
        device = next(self.encoder.parameters()).device
        with torch.no_grad():
            latents = self.encoder(torch.from_numpy(batch).to(device)).cpu().numpy()
        return self.score_embeddings(latents)
