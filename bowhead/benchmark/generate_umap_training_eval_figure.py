"""Regenerate ``Placeholder_UMAP_figure_training.png`` from the final,
already-UMAP-projected MATLAB exports (no AE re-encoding/re-projection
needed, since both exports already carry a precomputed ``umap_embeddings_3d``
column):

    paper/FinalDatabases_/latent_embeddings_3d_train_MATLAB_Angel_Final_26Aug2026.mat
    paper/FinalDatabases_/latent_embeddings_3d_eval_8to1_MATLAB_combined.mat

Run:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.benchmark.generate_umap_training_eval_figure
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bowhead.benchmark.generate_umap_paper_figures import plot_training_eval_grid

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_MAT = REPO_ROOT / "paper" / "FinalDatabases_" / "latent_embeddings_3d_train_MATLAB_Angel_Final_26Aug2026.mat"
DEFAULT_EVAL_MAT = REPO_ROOT / "paper" / "FinalDatabases_" / "latent_embeddings_3d_eval_8to1_MATLAB_combined.mat"
DEFAULT_OUT = REPO_ROOT / "paper" / "Figures" / "Placeholder_UMAP_figure_training.png"


def load_train(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    feat = d["features"]
    type_ = np.asarray(feat.type, dtype=float)
    label = ((type_ > 0) & (type_ < 12)).astype(int)
    return np.asarray(d["umap_embeddings_3d"], dtype=np.float64), label


def load_eval(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    label = np.asarray(d["iscall"]).astype(int)
    return np.asarray(d["umap_embeddings_3d"], dtype=np.float64), label


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-mat", type=Path, default=DEFAULT_TRAIN_MAT)
    p.add_argument("--eval-mat", type=Path, default=DEFAULT_EVAL_MAT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    print(f"Loading training UMAP embeddings from {args.train_mat} ...")
    train_coords, train_label = load_train(args.train_mat)
    print(f"  {len(train_coords):,} rows ({int(train_label.sum()):,} CD / {int((train_label == 0).sum()):,} OT)")

    print(f"Loading evaluation UMAP embeddings from {args.eval_mat} ...")
    eval_coords, eval_label = load_eval(args.eval_mat)
    print(f"  {len(eval_coords):,} rows ({int(eval_label.sum()):,} CD / {int((eval_label == 0).sum()):,} OT)")

    coords = np.concatenate([train_coords, eval_coords], axis=0)
    label = np.concatenate([train_label, eval_label], axis=0)
    split = np.concatenate([
        np.full(len(train_coords), "train", dtype=object),
        np.full(len(eval_coords), "eval", dtype=object),
    ])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_training_eval_grid(coords, label, split, args.out)


if __name__ == "__main__":
    main()
