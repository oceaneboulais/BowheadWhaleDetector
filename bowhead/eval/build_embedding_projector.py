"""Populate TensorBoard with an Embedding Projector view of the 32-D latent space.

Loads the trained scratch and warm-start CNN checkpoints, encodes a sample of
spectrograms, writes:
  1. ``add_embedding`` → TensorBoard Embedding Projector (PCA/t-SNE/UMAP in browser)
     - label metadata: call/non-call, call-type, site, dasar
     - sprite sheet of 32×28 px spectrogram thumbnails
  2. Pre-computed UMAP 2-D coordinates written as ``add_image`` scatter PNGs so
     they appear in the Images tab even without an embedding projector plugin.
  3. Per-class mean spectrogram images in the Images tab.
  4. Optional MATLAB .mat latent exports suitable for import into MATLAB's
     Deep Learning GUI and workspace.

Run:
    PYTHONPATH=. /usr/local/bin/python3.8 -m bowhead.eval.build_embedding_projector \
        --data data/spectrograms_demo.npz \
        --scratch  runs/scratch_demo/best.pt \
        --warmstart runs/warmstart_demo/best.pt \
        --out runs \
        --matlab-out-dir runs/matlab_latents \
        --n 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from bowhead.models.custom_cnn import EncoderClassifier
from bowhead.data.dataset import per_sample_minmax


# ── colour palette for the scatter PNG (one colour per label) ──────────────
_LABEL_COLOURS = {
    "call":     np.array([78, 205, 196], dtype=np.uint8),   # teal
    "non-call": np.array([255, 107, 107], dtype=np.uint8),  # coral
}
_TYPE_COLOURS = {  # call-type 0-7; 0 = auto-detected transient
    "0": np.array([200, 200, 200], dtype=np.uint8),
    "1": np.array([68, 114, 196], dtype=np.uint8),
    "2": np.array([112, 173, 71], dtype=np.uint8),
    "3": np.array([255, 192, 0], dtype=np.uint8),
    "4": np.array([255, 107, 107], dtype=np.uint8),
    "5": np.array([147, 112, 219], dtype=np.uint8),
    "6": np.array([0, 176, 240], dtype=np.uint8),
    "7": np.array([255, 140, 0], dtype=np.uint8),
}

_THUMB_W, _THUMB_H = 32, 28   # sprite thumbnail size (px)


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_model(ckpt_path: str | None, device: str) -> EncoderClassifier:
    """Load a saved EncoderClassifier; random init if path is None."""
    model = EncoderClassifier(input_hw=(121, 104)).to(device)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=False)
        print(f"  loaded {ckpt_path}")
    else:
        print(f"  WARNING: {ckpt_path} not found, using random weights")
    model.eval()
    return model


@torch.no_grad()
def _extract_embeddings(
    model: EncoderClassifier,
    images: np.ndarray,   # (N, H, W) uint8
    batch: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Return (N, latent_dim) float32 embeddings."""
    out = []
    for start in range(0, len(images), batch):
        chunk = images[start : start + batch]
        imgs = np.stack([per_sample_minmax(im) for im in chunk])[:, None]  # (B,1,H,W)
        x = torch.from_numpy(imgs).to(device)
        z = model.encoder(x).cpu().numpy()
        out.append(z)
    return np.concatenate(out, axis=0)


def _make_sprite(images: np.ndarray) -> np.ndarray:
    """Build a square sprite sheet (N thumbnails, each _THUMB_W x _THUMB_H px).

    Returns an (H_total, W_total, 3) uint8 array compatible with add_embedding.
    """
    n = len(images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    sprite = np.zeros((rows * _THUMB_H, cols * _THUMB_W, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        y0, x0 = r * _THUMB_H, c * _THUMB_W
        # normalise and resize to thumbnail
        norm = ((img.astype(np.float32) - img.min()) /
                max(img.max() - img.min(), 1e-6) * 255).astype(np.uint8)
        thumb = np.array(Image.fromarray(norm).resize((_THUMB_W, _THUMB_H),
                                                       Image.LANCZOS))
        thumb_rgb = np.stack([thumb, thumb, thumb], axis=-1)
        sprite[y0:y0 + _THUMB_H, x0:x0 + _THUMB_W] = thumb_rgb
    return sprite


def _to_matlab_cell_str(values: np.ndarray | list[str]) -> np.ndarray:
    """Return an object array of strings for MATLAB cell array import."""
    return np.array([str(v) for v in values], dtype=object)


def _save_matlab_latents(
    out_path: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: np.ndarray,
    call_types: np.ndarray,
    sites: np.ndarray,
    dasars: np.ndarray,
    sample_ids: np.ndarray | None = None,
) -> None:
    """Save latent embeddings and metadata to a .mat file importable by MATLAB."""
    data = {
        "embeddings": embeddings.astype(np.float32),
        "labels": labels.reshape(-1, 1).astype(np.uint8),
        "label_names": _to_matlab_cell_str(label_names),
        "call_type": _to_matlab_cell_str(call_types),
        "site": _to_matlab_cell_str(sites),
        "dasar": _to_matlab_cell_str(dasars),
    }
    if sample_ids is not None:
        data["sample_id"] = _to_matlab_cell_str(sample_ids)
    savemat(str(out_path), data, do_compression=True)


def _umap_scatter_png(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    size: int = 800,
) -> np.ndarray:
    """Compute 2-D UMAP and render a scatter PNG; return (3, size, size) float32."""
    import umap as umap_lib
    print(f"  computing UMAP for {title} …")
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                             metric="euclidean", random_state=42, n_jobs=1)
    coords = reducer.fit_transform(embeddings)

    canvas = np.ones((size, size, 3), dtype=np.float32)
    cx = coords[:, 0]
    cy = coords[:, 1]
    # normalise to [8, size-8]
    def _norm(v):
        lo, hi = v.min(), v.max()
        return ((v - lo) / max(hi - lo, 1e-9) * (size - 16) + 8).astype(int)

    px = _norm(cx)
    py = size - 1 - _norm(cy)  # flip y so low coords are at bottom

    # draw dots (3 px radius)
    for i in range(len(px)):
        colour = (_LABEL_COLOURS["call"] / 255.0
                  if labels[i] == 1
                  else _LABEL_COLOURS["non-call"] / 255.0)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dx * dx + dy * dy <= 9:
                    yy, xx = py[i] + dy, px[i] + dx
                    if 0 <= yy < size and 0 <= xx < size:
                        canvas[yy, xx] = colour

    # (H, W, 3) -> (3, H, W)
    return canvas.transpose(2, 0, 1)


def _mean_spectrograms(images: np.ndarray, labels: np.ndarray,
                        call_types: np.ndarray) -> dict[str, np.ndarray]:
    """Return dict of tag -> (1, H, W) float32 mean spectrogram images."""
    out = {}
    for lbl, name in [(1, "call"), (0, "non_call")]:
        mask = labels == lbl
        if mask.sum():
            mean = images[mask].astype(np.float32).mean(axis=0)
            mean = (mean - mean.min()) / max(mean.max() - mean.min(), 1e-6)
            out[f"mean_spectrogram/{name}"] = mean[None]  # (1, H, W)
    for ct in np.unique(call_types):
        mask = call_types == ct
        if mask.sum() >= 5:
            mean = images[mask].astype(np.float32).mean(axis=0)
            mean = (mean - mean.min()) / max(mean.max() - mean.min(), 1e-6)
            tag = f"mean_spectrogram/type_{ct}" + ("_transient" if ct == "0" else "_call")
            out[tag] = mean[None]
    return out


# ── main ────────────────────────────────────────────────────────────────────

def build(
    data_path: str,
    scratch_ckpt: str | None,
    warmstart_ckpt: str | None,
    out_dir: str,
    matlab_out_dir: str | None,
    n: int,
    device: str,
) -> None:
    print(f"Loading data from {data_path} …")
    data = np.load(data_path, allow_pickle=True)
    images_all = data["images"]          # (N, H, W) uint8
    labels_all = data["label"].astype(int)
    call_types_all = data["call_type"].astype(str)
    sites_all = data["site"].astype(str)
    dasars_all = data["dasar"].astype(str)

    # stratified sample: equal calls / non-calls up to n
    rng = np.random.default_rng(42)
    pos = np.where(labels_all == 1)[0]
    neg = np.where(labels_all == 0)[0]
    half = min(n // 2, len(pos), len(neg))
    idx = np.concatenate([
        rng.choice(pos, half, replace=False),
        rng.choice(neg, half, replace=False),
    ])
    idx = rng.permutation(idx)

    images    = images_all[idx]
    labels    = labels_all[idx]
    call_types = call_types_all[idx]
    sites     = sites_all[idx]
    dasars    = dasars_all[idx]
    print(f"  sampled {len(idx)} images  (calls={int((labels==1).sum())}  non-calls={int((labels==0).sum())})")

    # metadata rows for the projector (one column header per field)
    meta_header = ["label_name", "call_type", "site", "dasar", "label_int"]
    meta_rows = [
        [
            "call" if labels[i] == 1 else "non-call",
            call_types[i],
            sites[i],
            dasars[i],
            str(labels[i]),
        ]
        for i in range(len(labels))
    ]

    # sprite sheet (same for both models — same images)
    print("  building sprite sheet …")
    sprite = _make_sprite(images)  # (H_sprite, W_sprite, 3)

    runs = []
    if scratch_ckpt:
        runs.append(("scratch_demo", scratch_ckpt))
    if warmstart_ckpt:
        runs.append(("warmstart_demo", warmstart_ckpt))

    matlab_out_dir_path = Path(matlab_out_dir) if matlab_out_dir else None
    if matlab_out_dir_path is not None:
        matlab_out_dir_path.mkdir(parents=True, exist_ok=True)

    for run_name, ckpt_path in runs:
        print(f"\n── {run_name} ──")
        model = _load_model(ckpt_path, device)
        embeddings = _extract_embeddings(model, images, device=device)
        print(f"  embeddings: {embeddings.shape}")

        if matlab_out_dir_path is not None:
            matlab_out_path = matlab_out_dir_path / f"{run_name}_latents.mat"
            sample_ids = np.array([f"{sites[i]}_{dasars[i]}_{i}" for i in range(len(images))], dtype=object)
            _save_matlab_latents(
                matlab_out_path,
                embeddings,
                labels,
                np.array(["call" if labels[i] == 1 else "non-call" for i in range(len(labels))], dtype=object),
                call_types,
                sites,
                dasars,
                sample_ids=sample_ids,
            )
            print(f"  wrote MATLAB latent export -> {matlab_out_path}")

        writer = SummaryWriter(log_dir=str(Path(out_dir) / run_name))

        # 1. Embedding Projector ─────────────────────────────────────────────
        # add_embedding expects (N, D) tensor; label_img must be square per-image
        # thumbnails — PyTorch's internal sprite builder breaks when H != W, so
        # we resize to 28×28 squares here.
        emb_tensor = torch.from_numpy(embeddings)

        _SQ = 28  # square thumbnail size for the projector sprite
        label_imgs_sq = np.stack([
            np.array(
                Image.fromarray(
                    (per_sample_minmax(im) * 255).astype(np.uint8)
                ).resize((_SQ, _SQ), Image.LANCZOS)
            ).astype(np.float32) / 255.0
            for im in images
        ])  # (N, SQ, SQ)
        label_imgs = torch.from_numpy(label_imgs_sq[:, None, :, :])  # (N, 1, SQ, SQ)

        writer.add_embedding(
            emb_tensor,
            metadata=meta_rows,
            metadata_header=meta_header,
            label_img=label_imgs,
            global_step=0,
            tag="latent_32D",
        )
        print("  wrote Embedding Projector data")

        # 2. UMAP scatter PNG ────────────────────────────────────────────────
        scatter = _umap_scatter_png(embeddings, labels, run_name)
        writer.add_image("UMAP_scatter/teal=call_coral=noncall",
                         torch.from_numpy(scatter), global_step=0)
        print("  wrote UMAP scatter image")

        # 3. Mean spectrograms ───────────────────────────────────────────────
        for tag, img in _mean_spectrograms(images, labels, call_types).items():
            writer.add_image(tag, torch.from_numpy(img), global_step=0)
        print("  wrote mean spectrogram images")

        # 4. Embedding norm histogram ────────────────────────────────────────
        writer.add_histogram("latent/L2_norm",
                             torch.from_numpy(np.linalg.norm(embeddings, axis=1)),
                             global_step=0)
        for cls_name, cls_lbl in [("call", 1), ("non_call", 0)]:
            mask = labels == cls_lbl
            if mask.sum():
                writer.add_histogram(
                    f"latent/L2_norm/{cls_name}",
                    torch.from_numpy(np.linalg.norm(embeddings[mask], axis=1)),
                    global_step=0,
                )

        # 5. Per-dimension mean bar (as scalars over dim index) ──────────────
        dim_means_call    = embeddings[labels == 1].mean(axis=0)
        dim_means_noncall = embeddings[labels == 0].mean(axis=0)
        for d in range(embeddings.shape[1]):
            writer.add_scalars(
                "latent/per_dim_mean",
                {"call": float(dim_means_call[d]),
                 "non_call": float(dim_means_noncall[d])},
                global_step=d,
            )

        writer.close()
        print(f"  done → {Path(out_dir) / run_name}")

    print("\nAll done. Refresh TensorBoard at http://127.0.0.1:6006")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/spectrograms_demo.npz")
    p.add_argument("--scratch",    default="runs/scratch_demo/best.pt")
    p.add_argument("--warmstart",  default="runs/warmstart_demo/best.pt")
    p.add_argument("--out",        default="runs")
    p.add_argument("--matlab-out-dir", default=None,
                   help="Optional directory to write MATLAB .mat latent exports for each run.")
    p.add_argument("--n",          type=int, default=2000,
                   help="total samples to embed (stratified call/non-call)")
    p.add_argument("--device",     default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    build(a.data, a.scratch, a.warmstart, a.out, a.matlab_out_dir, a.n, a.device)
