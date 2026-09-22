"""
Load a per-detection spectrogram (SNR_gram / NTV_gram / etc.) from the
per-file .mat databases referenced by the ``image_folder`` field of an
embeddings .mat file.
"""

import os
import numpy as np
from typing import Optional

# Preference order for which gram to display
_GRAM_CANDIDATES = ("SNR_gram", "NTV_gram", "KEtoPE_gram", "Polar_gram")


def load_detection_spectrogram(image_folder: str, detection_filename: str) -> dict:
    """Load one detection's spectrogram .mat file.

    ``detection_filename`` may be a bare stem/filename (joined onto
    ``image_folder``, the classic flat-directory layout) or an already-absolute
    path (used as-is) — the latter supports datasets nested under
    year/site/day/kind/dasar subfolders, e.g. the
    ``Spectrogram_Image_Database_Sites35_ADG_*_centered.dir`` layout, where
    ``original_filenames`` in the embeddings .mat stores full paths.

    Returns a dict with keys: gram (2-D array), gram_name, dT, dF (if present).
    Raises FileNotFoundError if the file/folder isn't accessible.
    """
    fname = detection_filename
    if not fname.endswith(".mat"):
        fname += ".mat"

    if os.path.isabs(fname):
        fpath = fname
    else:
        if not image_folder or not os.path.isdir(image_folder):
            raise FileNotFoundError(
                f"Spectrogram image folder not accessible: {image_folder!r}\n"
                "Mount the drive containing the per-detection .mat database, or "
                "pass --image-folder to override."
            )
        fpath = os.path.join(image_folder, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Detection spectrogram file not found: {fpath}")

    from scipy.io import loadmat
    raw = loadmat(fpath, squeeze_me=True, struct_as_record=False)

    gram_name = next((g for g in _GRAM_CANDIDATES if g in raw), None)
    if gram_name is None:
        avail = [k for k in raw if not k.startswith("_")]
        raise KeyError(f"No known gram field in {fpath}. Available: {avail}")

    gram = np.asarray(raw[gram_name], dtype=float)
    return {
        "gram": gram,
        "gram_name": gram_name,
        "dT": raw.get("dT"),
        "dF": raw.get("dF"),
        "path": fpath,
    }


def gram_to_png_bytes(gram: np.ndarray, cmap: str = "inferno") -> bytes:
    """Render a gram array to PNG bytes (for embedding in HTML / Flask responses)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO

    fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=110)
    ax.imshow(gram, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xlabel("time bin", fontsize=7)
    ax.set_ylabel("freq bin", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def save_detection_thumbnail(image_folder: str, detection_filename: str,
                              out_path: str, cmap: str = "inferno") -> None:
    """Load one detection's spectrogram and write it to ``out_path`` as a PNG.

    Thin wrapper around :func:`load_detection_spectrogram` +
    :func:`gram_to_png_bytes` used to pre-render per-point thumbnails for the
    static Plotly HTML viewer (see ``visualize_latent.py``'s
    ``--export-thumbnails``). Raises the same exceptions as
    ``load_detection_spectrogram`` on missing/unreadable files — callers
    should catch and skip.
    """
    result = load_detection_spectrogram(image_folder, detection_filename)
    png_bytes = gram_to_png_bytes(result["gram"], cmap=cmap)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(png_bytes)
