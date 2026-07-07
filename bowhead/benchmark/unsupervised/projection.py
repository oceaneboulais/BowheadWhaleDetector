"""Embedding projection utilities for 2-D visualisation (UMAP / PCA).

Used to visualise the latent space produced by any backbone or the supervised
CNN encoder, and to overlay call-type or cluster labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def pca_project(
    embeddings: np.ndarray,
    n_components: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Reduce ``embeddings`` to ``n_components`` dimensions via PCA.

    Parameters
    ----------
    embeddings   : (N, D) float array
    n_components : target dimensionality (2 for scatter plots)

    Returns
    -------
    (N, n_components) float32 array
    """
    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(embeddings.astype(np.float64)).astype(np.float32)


def umap_project(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Reduce ``embeddings`` via UMAP (requires ``umap-learn``).

    Parameters
    ----------
    embeddings   : (N, D) float array
    n_components : target dimensionality (2 for scatter plots, 3 for 3-D)
    n_neighbors  : UMAP local neighbourhood size
    min_dist     : minimum distance between embedded points

    Returns
    -------
    (N, n_components) float32 array

    Raises
    ------
    ImportError
        If ``umap-learn`` is not installed.
    """
    try:
        import umap as _umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is not installed. Run: pip install umap-learn"
        ) from exc
    reducer = _umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    return reducer.fit_transform(embeddings.astype(np.float64)).astype(np.float32)
