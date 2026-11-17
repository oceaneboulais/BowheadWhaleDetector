"""Sprite-sheet ("atlas") tiles of the training spectrograms for the interactive UMAP page.

Only rows whose image is VERIFIED to be the one the UMAP row was built from get a spectrogram:
``features.fpeak`` / ``features.tpeak`` in the training .mat are computed by MATLAB's
``extract_features_from_SNRgram`` straight from the pixels (Fmax = dF*argmax_row(rowmax),
Tmax = dT*argmax_col(colmax - min)), so an image that is not the original (different centering
or regeneration) fails this test even when the filename matches.  Source images are
``data/spectrograms_100k_v2.npz`` (121 x 104 uint8, SNR_gram = 5 x dB).

Tile layout (consumed by the page's JS):  COLS x ROWS cells per tile, each cell CELL_W x CELL_H px;
the cell holds the spectrogram flipped vertically (high frequency at the top, like the authors'
``axis xy`` plots) in its first IMG_H rows, remaining rows black.  Pixel = clip(SNR_gram / 100, 0, 1) * 255,
i.e. 0 dB..20 dB (SNR_gram/5) -- the display range of the authors' ``caxis([0 20])`` plots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_MAT = REPO_ROOT / "paper/FinalDatabases_/latent_embeddings_3d_train_MATLAB_Angel_Final_26Aug2026.mat"
SRC_NPZ = REPO_ROOT / "data/spectrograms_100k_v2.npz"
CACHE_DIR = REPO_ROOT / "runs/umap_click_tab"

DF, DT = 3.90625, 0.026  # bin sizes of the stored 121 x 104 grams (read from the source .mat files)
IMG_H, IMG_W = 121, 104
CELL_W, CELL_H = 104, 128  # multiple of 8 so JPEG/WebP blocks never straddle two spectrograms
COLS, ROWS = 8, 8
PER_TILE = COLS * ROWS
DB_FULL_SCALE = 20.0  # gray 255 <-> 20 dB
TOL = 1e-3


def peak_features(gram: np.ndarray) -> tuple[float, float]:
    """MATLAB extract_features_from_SNRgram: (Fmax, Tmax) with FF = dF*(1:nF), TT = dT*(1:nT)."""
    g = gram.astype(np.float64) / 5.0
    ff = DF * np.arange(1, g.shape[0] + 1)
    tt = DT * np.arange(1, g.shape[1] + 1)
    col = g.max(axis=0)
    return float(ff[int(np.argmax(g.max(axis=1)))]), float(tt[int(np.argmax(col - col.min()))])


def to_gray(gram: np.ndarray) -> np.ndarray:
    return np.clip(gram.astype(np.float32) / (DB_FULL_SCALE * 5.0) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def find_verified_rows() -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (row_to_src, slot_src_idx, stats): src image index per training row (-1 = none), verified."""
    m = sio.loadmat(str(TRAIN_MAT), simplify_cells=True)
    fn = np.asarray(m["original_filenames"]).tolist()
    fpk = np.asarray(m["features"]["fpeak"], dtype=float)
    tpk = np.asarray(m["features"]["tpeak"], dtype=float)

    npz = np.load(SRC_NPZ, allow_pickle=True)
    stems = npz["stem"].tolist()
    images = npz["images"]  # decompress once
    src_of = {}
    n_dup_src = 0
    for j, s in enumerate(stems):
        if s in src_of:
            n_dup_src += 1
        else:
            src_of[s] = j

    row_src = np.full(len(fn), -1, dtype=np.int64)
    cand = fail = 0
    for i, f in enumerate(fn):
        j = src_of.get(f[:-4])
        if j is None:
            continue
        cand += 1
        fp, tp = peak_features(images[j])
        if abs(fp - fpk[i]) < TOL and abs(tp - tpk[i]) < TOL:
            row_src[i] = j
        else:
            fail += 1
    stats = {"train_rows": len(fn), "candidate_rows": cand, "verified_rows": int((row_src >= 0).sum()),
             "failed_verification": fail, "duplicate_src_stems": n_dup_src}
    return row_src, images, stats


def build(out_dir: Path, quality: int = 80, rebuild: bool = False) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "row_slots.npy"
    meta_path = out_dir / "atlas.json"
    if cache.exists() and meta_path.exists() and not rebuild:
        meta = json.loads(meta_path.read_text())
        if meta.get("quality") == quality:
            return {"row_slots": np.load(cache), **meta}

    row_src, images, stats = find_verified_rows()
    # one slot per distinct source image, numbered in order of first training row
    slot_of_src: dict[int, int] = {}
    row_slots = np.full(len(row_src), -1, dtype=np.int32)
    for i in np.flatnonzero(row_src >= 0):
        j = int(row_src[i])
        row_slots[i] = slot_of_src.setdefault(j, len(slot_of_src))
    n_slots = len(slot_of_src)
    src_by_slot = np.empty(n_slots, dtype=np.int64)
    for j, s in slot_of_src.items():
        src_by_slot[s] = j

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("tile_*.webp"):
        old.unlink()
    n_tiles = -(-n_slots // PER_TILE)
    total = 0
    for t in range(n_tiles):
        sheet = np.zeros((ROWS * CELL_H, COLS * CELL_W), dtype=np.uint8)
        for k in range(PER_TILE):
            s = t * PER_TILE + k
            if s >= n_slots:
                break
            r, c = divmod(k, COLS)
            sheet[r * CELL_H: r * CELL_H + IMG_H, c * CELL_W: c * CELL_W + IMG_W] = to_gray(images[src_by_slot[s]])[::-1]
        path = out_dir / f"tile_{t:03d}.webp"
        Image.fromarray(sheet, mode="L").save(path, "WEBP", quality=quality, method=6)
        total += path.stat().st_size
    meta = {"quality": quality, "n_slots": n_slots, "n_tiles": n_tiles, "per_tile": PER_TILE, "cols": COLS,
            "rows": ROWS, "cell_w": CELL_W, "cell_h": CELL_H, "img_h": IMG_H, "img_w": IMG_W,
            "total_bytes": total, "stats": stats}
    meta_path.write_text(json.dumps(meta, indent=1))
    np.save(cache, row_slots)
    np.save(CACHE_DIR / "slot_src.npy", src_by_slot)
    return {"row_slots": row_slots, **meta}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    r = build(a.out_dir, a.quality, a.rebuild)
    print({k: v for k, v in r.items() if k != "row_slots"})
    print(f"{r['total_bytes'] / 1e6:.1f} MB over {r['n_tiles']} tiles "
          f"({r['total_bytes'] / r['n_slots'] / 1024:.2f} KB per spectrogram)")
