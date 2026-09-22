"""
Robust loader for MATLAB latent-embedding .mat files.

Handles scipy (v5/v6) and h5py (v7.3 / HDF5) formats, with automatic
detection of embedding fields and metadata extraction.
"""

import os
import re
import numpy as np
from typing import Optional

# Priority order for recognising pre-computed embedding fields
_EMBEDDING_CANDIDATES = [
    "umap_embeddings_3d",
    "umap_embeddings_2d",
    "umap_3d",
    "umap_2d",
    "x_umap",
    "umap_embeddings",
    "x_tsne",
    "tsne_embeddings",
    "latent_embeddings",
    "latent",
    "z",
    "embeddings",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mat(path: str) -> dict:
    """Load a .mat file regardless of version.

    Returns a plain dict mapping field name → numpy array.
    Raises FileNotFoundError or ValueError on failure.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        from scipy.io import loadmat
        raw = loadmat(path, squeeze_me=True, struct_as_record=False)
        return {k: _squeeze(v) for k, v in raw.items() if not k.startswith("_")}
    except NotImplementedError:
        # MATLAB v7.3 files raise NotImplementedError in scipy
        return _load_hdf5(path)
    except Exception as exc:
        if "HDF" in str(exc) or "v7.3" in str(exc) or "Unable" in str(exc):
            return _load_hdf5(path)
        raise


def detect_embeddings(data: dict) -> tuple[str, np.ndarray]:
    """Return (field_name, 2-D float array) for the best embedding in *data*.

    Tries known field names first, then falls back to shape heuristics.
    """
    # Named candidates
    for name in _EMBEDDING_CANDIDATES:
        if name in data:
            arr = _as2d(data[name])
            if arr is not None:
                return name, arr

    # Shape heuristic: 2-D array with many rows and ≤512 columns
    for key, val in data.items():
        arr = _as2d(val)
        if arr is not None and arr.shape[0] > 10 and arr.shape[1] <= 512:
            return key, arr

    raise ValueError(
        "No embedding array found in file.\n"
        f"Available keys: {list(data.keys())}\n"
        "Use --field to specify the correct field name."
    )


def extract_labels(data: dict, n: int) -> Optional[np.ndarray]:
    """Return a (n,) float array of integer class labels, or None."""
    # Direct numeric label fields
    for field in ("type", "labels", "call_type", "cluster", "class"):
        if field in data:
            arr = np.asarray(data[field]).flatten()
            if len(arr) == n:
                try:
                    return arr.astype(float)
                except (ValueError, TypeError):
                    pass

    # Parse from filenames: e.g. "...Type3.mat"
    for field in ("original_filenames", "filenames", "file_names"):
        if field in data:
            fnames = data[field]
            try:
                types = []
                for fn in np.asarray(fnames).flat:
                    m = re.search(r"[Tt]ype(\d+)", str(fn))
                    types.append(int(m.group(1)) if m else 0)
                if len(types) == n:
                    return np.array(types, dtype=float)
            except Exception:
                pass

    return None


def extract_metadata(data: dict, n: int) -> dict:
    """Return all numeric (n,) arrays as potential colour channels."""
    meta = {}
    for key, val in data.items():
        try:
            arr = np.asarray(val, dtype=float).flatten()
            if len(arr) == n and np.isfinite(arr).any():
                meta[key] = arr
        except (ValueError, TypeError):
            pass
    return meta


def list_fields(data: dict) -> None:
    """Print a summary of all fields in *data*."""
    print(f"{'Field':<35} {'Shape':<20} {'Dtype'}")
    print("-" * 65)
    for k, v in data.items():
        try:
            arr = np.asarray(v)
            print(f"  {k:<33} {str(arr.shape):<20} {arr.dtype}")
        except Exception:
            print(f"  {k:<33} <non-array>")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _squeeze(v):
    """Collapse length-1 dims introduced by scipy loadmat."""
    if isinstance(v, np.ndarray):
        return v
    try:
        return np.asarray(v)
    except Exception:
        return v


def _as2d(val) -> Optional[np.ndarray]:
    """Return val as a (N, D) float array, or None if it doesn't qualify."""
    try:
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] >= 2:
            return arr
    except (ValueError, TypeError):
        pass
    return None


def _load_hdf5(path: str) -> dict:
    """Load a MATLAB v7.3 (HDF5) .mat file via h5py."""
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for MATLAB v7.3 (.mat) files.\n"
            "Install with:  pip install h5py"
        )

    out: dict = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            if key.startswith("#"):
                continue
            try:
                dset = f[key]
                raw = dset[()]
                if hasattr(raw, "dtype") and raw.dtype.kind == "O":
                    # Dereference MATLAB string cell arrays
                    strings = []
                    for ref in raw.flat:
                        try:
                            chars = f[ref][()]
                            strings.append("".join(chr(int(c)) for c in chars.flat))
                        except Exception:
                            strings.append("")
                    out[key] = np.array(strings)
                else:
                    # h5py stores arrays in column-major order — transpose to row-major
                    out[key] = np.array(raw).T
            except Exception:
                pass
    return out
