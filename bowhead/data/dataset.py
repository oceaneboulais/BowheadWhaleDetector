"""Spectrogram dataset + the data interface contract.

⚠️ WRITTEN AGAINST AN ASSUMED SCHEMA — adapt once we see the real AE data.

Assumed on-disk layout (one ``.npz`` for now; easy to swap for HDF5/per-file):
    images : uint8, shape (N, H, W) or (N, 1, H, W)   -- dB-SNR spectrograms
    plus a metadata table (structured array / parallel arrays) with, per image,
    the columns in REQUIRED_METADATA_COLUMNS.

Preprocessing matches the autoencoder: per-sample min-max normalization to
[0, 1], identical for all four pipelines so the comparison is fair.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Columns the metadata MUST provide. `label` drives the binary task; the rest
# enable leakage-free grouped splitting and later multiclass / airgun analysis.
REQUIRED_METADATA_COLUMNS = (
    "unique_call",  # unique-call id  -> grouping key (strictest)
    "date",         # recording date -> grouping key (coarse: date x site)
    "site",         # DASAR site (3 or 5)
    "dasar",        # instrument A/D/G
    "label",        # 1 = whale call, 0 = other transient
    "call_type",    # str/int call-type label (for later multiclass); may be ""
    "is_airgun",    # bool flag (airguns are ~40% of transients)
)


def per_sample_minmax(img: np.ndarray) -> np.ndarray:
    """Min-max normalize a single image to [0, 1] (matches the AE preprocessing)."""
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


class SpectrogramDataset(Dataset):
    """Wraps in-memory spectrogram images + labels for the PyTorch pipelines.

    Parameters
    ----------
    images : np.ndarray
        (N, H, W) or (N, 1, H, W). Any dtype; normalized per-sample on access.
    labels : np.ndarray
        (N,) binary int labels.
    indices : np.ndarray, optional
        Restrict the dataset to a subset (e.g. a split's train indices).
    normalize : bool
        Apply per-sample min-max (default True, matching the AE).
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray | None = None,
        normalize: bool = True,
    ) -> None:
        if images.ndim == 3:
            images = images[:, None, :, :]  # add channel dim
        elif images.ndim != 4:
            raise ValueError(f"images must be 3- or 4-D, got shape {images.shape}")
        self.images = images
        self.labels = np.asarray(labels).astype(np.int64)
        self.indices = (
            np.arange(len(images)) if indices is None else np.asarray(indices)
        )
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        j = self.indices[i]
        img = self.images[j]
        if self.normalize:
            img = np.stack([per_sample_minmax(ch) for ch in img])
        else:
            img = img.astype(np.float32)
        return torch.from_numpy(img), int(self.labels[j])

    # ------------------------------------------------------------------ #
    # Loader — ADAPT to the real on-disk format once provided.
    # ------------------------------------------------------------------ #
    @classmethod
    def load_npz(cls, path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
        """Load images, labels, and a metadata dict from a single ``.npz``.

        Returns ``(images, labels, metadata)`` where ``metadata`` maps each name
        in REQUIRED_METADATA_COLUMNS to an array of length N. Raises if any
        required column is missing so problems surface early.
        """
        data = np.load(path, allow_pickle=True)
        images = data["images"]
        metadata = {}
        missing = []
        for col in REQUIRED_METADATA_COLUMNS:
            if col in data:
                metadata[col] = data[col]
            else:
                missing.append(col)
        if missing:
            raise KeyError(
                f"{path}: missing required metadata columns {missing}. "
                f"Present keys: {list(data.keys())}. "
                f"See REQUIRED_METADATA_COLUMNS / README data contract."
            )
        return images, metadata["label"].astype(np.int64), metadata
