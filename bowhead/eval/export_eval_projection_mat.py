"""Export a pure projection of the eval dataset to MATLAB .mat.

This script does not use any trained CNN encoder. It loads the eval NPZ,
preprocesses each spectrogram, and projects the raw features with PCA, UMAP, or
PaCMAP. The resulting .mat file contains only projection coordinates and metadata.

Example:
    PYTHONPATH=. .venv312/bin/python bowhead/eval/export_eval_projection_mat.py \
        --data data/spectrograms_eval_v2.npz \
        --out data/eval_v2_review_2026-07-27/matlab_latents/eval_raw_pca.mat \
        --method pca \
        --n-components 2
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np


def per_sample_minmax(img: np.ndarray) -> np.ndarray:
    """Min-max normalize a single image to [0, 1]."""
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def _load_data(data_path: str):
    data = np.load(data_path, allow_pickle=True)
    images = data["images"]
    labels = data["label"].astype(int)
    call_types = data["call_type"].astype(str)
    sites = data["site"].astype(str)
    dasars = data["dasar"].astype(str)
    return images, labels, call_types, sites, dasars


def _flatten_images(images: np.ndarray) -> np.ndarray:
    num_samples, height, width = images.shape
    flat = np.empty((num_samples, height * width), dtype=np.float32)
    for i, im in enumerate(images):
        flat[i] = per_sample_minmax(im).ravel()
    return flat


def _project_pca(features: np.ndarray, n_components: int, seed: int = 42) -> np.ndarray:
    features = features.astype(np.float32)
    features = features - features.mean(axis=0, keepdims=True)
    try:
        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        return svd.fit_transform(features).astype(np.float32)
    except ImportError:
        X = features.astype(np.float64)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        return (U[:, :n_components] * S[:n_components]).astype(np.float32)


def _project_raw(features: np.ndarray, method: str, n_components: int, seed: int = 42) -> np.ndarray:
    method = method.lower()
    if method == "pca":
        return _project_pca(features, n_components, seed=seed)

    features = features.astype(np.float32)
    if method in {"umap", "pacmap"} and features.shape[1] > 50:
        features = _project_pca(features, 50, seed=seed)

    if method == "umap":
        try:
            import umap as _umap
        except ImportError as exc:
            raise ImportError("umap-learn is not installed. Run: pip install umap-learn") from exc
        reducer = _umap.UMAP(
            n_components=n_components,
            random_state=seed,
            n_neighbors=15,
            min_dist=0.1,
        )
        return reducer.fit_transform(features).astype(np.float32)
    if method == "pacmap":
        try:
            import pacmap as _pacmap
        except ImportError as exc:
            raise ImportError("pacmap is not installed. Run: pip install pacmap") from exc
        reducer = _pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=15,
            MN_ratio=0.5,
            FP_ratio=2.0,
            apply_pca=False,
            random_state=seed,
        )
        return reducer.fit_transform(features).astype(np.float32)
    raise ValueError("method must be 'pca', 'umap', or 'pacmap'")


def _to_matlab_cell_str(values: np.ndarray | list[str]) -> np.ndarray:
    return np.array([str(v) for v in values], dtype=object)


def _save_matlab(out_path: Path, projection: np.ndarray, labels: np.ndarray, call_types: np.ndarray, sites: np.ndarray, dasars: np.ndarray, sample_ids: np.ndarray | None = None) -> None:
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("projection", data=projection.astype(np.float32), compression="gzip")
        f.create_dataset("labels", data=labels.reshape(-1, 1).astype(np.uint8), compression="gzip")
        f.create_dataset("call_type", data=np.array(call_types.astype(str), dtype=object), dtype=str_dt)
        f.create_dataset("site", data=np.array(sites.astype(str), dtype=object), dtype=str_dt)
        f.create_dataset("dasar", data=np.array(dasars.astype(str), dtype=object), dtype=str_dt)
        if sample_ids is not None:
            f.create_dataset("sample_id", data=np.array(sample_ids.astype(str), dtype=object), dtype=str_dt)


def parse_args():
    parser = argparse.ArgumentParser(description="Export a pure projection of eval data to MATLAB .mat.")
    parser.add_argument("--data", required=True, help="Eval NPZ file path")
    parser.add_argument("--out", required=True, help="Output .mat file path")
    parser.add_argument("--method", default="pca", choices=["pca", "umap", "pacmap"], help="Projection method")
    parser.add_argument("--n-components", type=int, default=2, help="Projection dimensionality")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for projection")
    return parser.parse_args()


def main():
    args = parse_args()
    images, labels, call_types, sites, dasars = _load_data(args.data)
    print(f"Loaded {len(images)} eval images from {args.data}")

    if args.method in {"umap", "pacmap"} and args.n_components not in {2, 3}:
        raise ValueError("UMAP and PaCMAP exports currently support only 2 or 3 components")

    features = _flatten_images(images)
    print(f"Flattened images to feature matrix {features.shape}")
    projection = _project_raw(features, args.method, args.n_components, seed=args.seed)
    print(f"Computed {args.method} projection with shape {projection.shape}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_ids = np.array([f"{sites[i]}_{dasars[i]}_{i}" for i in range(len(images))], dtype=object)
    _save_matlab(out_path, projection, labels, call_types, sites, dasars, sample_ids=sample_ids)
    print(f"Wrote pure eval projection to {out_path}")


if __name__ == "__main__":
    main()
