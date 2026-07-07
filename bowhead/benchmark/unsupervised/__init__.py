"""Unsupervised subdivision sub-package.

Provides clustering and 2-D projection utilities for analysing the latent
embedding spaces produced by the benchmark backbones.
"""

from bowhead.benchmark.unsupervised.cluster import (
    ClusterResult,
    fit_clusters,
    bic_select_gmm,
)
from bowhead.benchmark.unsupervised.projection import pca_project, umap_project

__all__ = [
    "ClusterResult",
    "fit_clusters",
    "bic_select_gmm",
    "pca_project",
    "umap_project",
]
