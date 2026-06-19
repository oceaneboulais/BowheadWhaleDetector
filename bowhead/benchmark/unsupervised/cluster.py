"""Unsupervised subdivision of the latent embedding space.

Uses encoder embeddings (from a trained ConvEncoder or any backbone) to cluster
and then characterise sub-groups within the call and non-call populations,
potentially revealing call-type structure, DASAR artefacts, or SNR stratification
without using any human labels.

Supported algorithms
--------------------
k_means        — k-Means (scikit-learn, deterministic with fixed seed)
hdbscan        — HDBSCAN (density-based, discovers the number of clusters)
gmm            — Gaussian Mixture Model (soft assignment, BIC-selected k)

Typical usage
-------------
from bowhead.benchmark.unsupervised.cluster import fit_clusters, ClusterResult
result = fit_clusters(embeddings, method="hdbscan", min_cluster_size=30)
print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterResult:
    """Output of a single clustering run."""
    method: str
    labels: np.ndarray         # (N,) int; -1 = noise (HDBSCAN only)
    n_clusters: int
    noise_fraction: float      # fraction of points assigned label -1
    embeddings_2d: np.ndarray | None = None  # optional UMAP/PCA projection (N,2)

    # Cluster–label contingency: rows=clusters, cols=[non-call, call]
    contingency: np.ndarray | None = None    # (K, 2) int

    def summary(self) -> str:
        lines = [
            f"method={self.method}  n_clusters={self.n_clusters}  "
            f"noise_fraction={self.noise_fraction:.3f}"
        ]
        if self.contingency is not None:
            lines.append("  cluster  non-call   call  call_frac")
            for ci in range(self.n_clusters):
                nc, c = self.contingency[ci]
                frac = c / (nc + c) if (nc + c) > 0 else float("nan")
                lines.append(f"  {ci:>6d}  {nc:>8,}  {c:>5,}  {frac:.3f}")
        return "\n".join(lines)


def _scale(embeddings: np.ndarray) -> np.ndarray:
    """Standard-scale embeddings in-place (returns a copy)."""
    return StandardScaler().fit_transform(embeddings.astype(np.float64))


def fit_clusters(
    embeddings: np.ndarray,
    method: str = "hdbscan",
    n_clusters: int = 8,
    min_cluster_size: int = 30,
    seed: int = 0,
    labels_gt: np.ndarray | None = None,
) -> ClusterResult:
    """Fit a clustering model on ``embeddings`` and return a :class:`ClusterResult`.

    Parameters
    ----------
    embeddings      : (N, D) float32/float64
    method          : "k_means" | "hdbscan" | "gmm"
    n_clusters      : number of clusters for k_means / gmm (ignored by hdbscan)
    min_cluster_size: HDBSCAN parameter (min points in a cluster)
    seed            : random seed for k_means / gmm
    labels_gt       : optional (N,) ground-truth binary labels for contingency
    """
    X = _scale(embeddings)

    if method == "k_means":
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
        cluster_labels = km.fit_predict(X)
        n_clus = n_clusters
        noise_frac = 0.0

    elif method == "hdbscan":
        try:
            import hdbscan as _hdbscan  # optional; not in default requirements
        except ImportError as exc:
            raise ImportError(
                "hdbscan is not installed. Run: pip install hdbscan"
            ) from exc
        clusterer = _hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(X)
        n_clus = int(cluster_labels.max()) + 1  # -1 = noise
        noise_frac = float((cluster_labels == -1).mean())

    elif method == "gmm":
        gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
        cluster_labels = gmm.fit_predict(X)
        n_clus = n_clusters
        noise_frac = 0.0

    else:
        raise ValueError(f"Unknown method {method!r}. Choose: k_means, hdbscan, gmm")

    # Build contingency table if ground-truth labels supplied.
    contingency: np.ndarray | None = None
    if labels_gt is not None:
        k = max(n_clus, 1)
        contingency = np.zeros((k, 2), dtype=np.int64)
        valid_mask = cluster_labels >= 0
        for ci in range(k):
            mask = valid_mask & (cluster_labels == ci)
            if not mask.any():
                continue
            gt = labels_gt[mask]
            contingency[ci, 0] = int((gt == 0).sum())
            contingency[ci, 1] = int((gt == 1).sum())

    return ClusterResult(
        method=method,
        labels=cluster_labels,
        n_clusters=n_clus,
        noise_fraction=noise_frac,
        contingency=contingency,
    )


def bic_select_gmm(
    embeddings: np.ndarray,
    k_range: tuple[int, ...] = (2, 4, 6, 8, 12, 16),
    seed: int = 0,
) -> tuple[int, list[float]]:
    """Select the number of GMM components by BIC.

    Returns
    -------
    best_k : int
        Number of components with lowest BIC.
    bic_scores : list[float]
        BIC for each k in ``k_range``.
    """
    X = _scale(embeddings)
    bics: list[float] = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=seed)
        gmm.fit(X)
        bics.append(float(gmm.bic(X)))
    best_k = k_range[int(np.argmin(bics))]
    return best_k, bics
