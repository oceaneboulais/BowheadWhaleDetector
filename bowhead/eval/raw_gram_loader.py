"""Load raw (unnormalized) single-channel SNR-gram arrays from eval .mat files.

Shared by classical/transfer relabeling-ablation scripts (Moan Detector,
BirdNET) that need the raw uint8-scale ``SNR_gram``, unlike the CNN scorer
(``bowhead.eval.score_eval_dataset``) which per-sample min-max normalizes
channels before feeding the network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_raw_grams(mat_paths: list[Path], gram: str = "SNR_gram") -> np.ndarray:
    """Read *gram* from each file in *mat_paths* -> array shape (N, 121, 104)."""
    n = len(mat_paths)
    out = np.empty((n, 121, 104), dtype=np.float32)
    for i, path in enumerate(mat_paths):
        try:
            m = loadmat(str(path))
            out[i] = m[gram].astype(np.float32)
        except Exception as e:
            print(f"  WARN: could not read {path.name}: {e}; using zeros")
            out[i] = 0.0
    return out
