"""Multi-backbone embedding fusion.

Two fusion strategies, both from the embedding-fusion literature:

concatenation
    [e_1 | e_2 | ... | e_k]  — preserves all information, dimension grows.
    Works best when backbones are complementary (e.g. local-texture ResNet
    + global-patch ViT + frequency-aware waveform model).

mean-pooling
    (e_1 + e_2 + ... + e_k) / k  — after L2-normalising each embedding.
    Lower-dimensional; good when backbones are roughly commensurate.

Both produce a single (N, D_fused) array that feeds directly into any probe.
"""

from __future__ import annotations

import numpy as np


def fuse_concat(embeddings: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate embeddings from multiple backbones along the feature axis.

    Parameters
    ----------
    embeddings : dict backbone_name -> (N, D_i) array

    Returns
    -------
    (N, sum(D_i)) float32 array
    """
    arrays = list(embeddings.values())
    if len({a.shape[0] for a in arrays}) != 1:
        raise ValueError(
            "All embeddings must have the same number of samples. "
            f"Got shapes: { {k: v.shape for k, v in embeddings.items()} }"
        )
    return np.concatenate([a.astype(np.float32) for a in arrays], axis=1)


def fuse_mean(embeddings: dict[str, np.ndarray]) -> np.ndarray:
    """L2-normalise each backbone's embeddings then average.

    Parameters
    ----------
    embeddings : dict backbone_name -> (N, D) array
        All arrays must have the same shape (N, D).

    Returns
    -------
    (N, D) float32 array
    """
    arrays = list(embeddings.values())
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(
            "fuse_mean requires all backbones to have the same embedding dimension. "
            f"Got shapes: { {k: v.shape for k, v in embeddings.items()} }. "
            "Use fuse_concat instead for backbones with different dimensions."
        )
    normed = []
    for a in arrays:
        a = a.astype(np.float32)
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed.append(a / norms)
    return np.mean(normed, axis=0)


def fuse(
    embeddings: dict[str, np.ndarray],
    strategy: str = "concat",
) -> np.ndarray:
    """Dispatch to fuse_concat or fuse_mean.

    Parameters
    ----------
    embeddings : dict backbone_name -> (N, D_i)
    strategy : "concat" | "mean"
    """
    if strategy == "concat":
        return fuse_concat(embeddings)
    if strategy == "mean":
        return fuse_mean(embeddings)
    raise ValueError(f"Unknown fusion strategy {strategy!r}. Use 'concat' or 'mean'.")


def fuse_sensors(
    sensor_embeddings: dict[str, np.ndarray],
    strategy: str = "concat",
) -> np.ndarray:
    """Fuse per-sensor (DASAR A/D/G) embeddings for the same detection event.

    Parameters
    ----------
    sensor_embeddings : dict sensor_id -> (N, D) array
        Embeddings from sensors A, D, G for the N detection events.
        Missing sensors should be handled before calling (e.g. zero-fill).
    strategy : "concat" | "mean"

    Returns
    -------
    (N, D_fused) float32 array
    """
    return fuse(sensor_embeddings, strategy=strategy)
