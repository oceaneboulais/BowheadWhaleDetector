"""Export latent embeddings for all images to MATLAB .mat files.

Loads the encoder from checkpoints, encodes every image in the provided NPZ,
and writes compressed .mat files suitable for MATLAB import.

Usage:
    PYTHONPATH=. .venv312/bin/python bowhead/eval/export_all_matlab_latents.py \
        --data data/eval_v2_2026-07-27/spectrograms_eval_v2.npz \
        --scratch runs/scratch_100k_matched/best.pt \
        --warmstart runs/warmstart_100k_matched/best.pt \
        --out-dir data/eval_v2_2026-07-27/matlab_latents \
        --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import h5py

from bowhead.models.custom_cnn import EncoderClassifier

# Avoid importing the bowhead.data package (which pulls in sklearn/scipy)
# by loading the dataset module directly. This keeps dependencies minimal
# when running the exporter in constrained environments.
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path as _P

_dataset_path = _P("bowhead/data/dataset.py").resolve()
spec = spec_from_file_location("bowhead.data.dataset", str(_dataset_path))
dataset_mod = module_from_spec(spec)
spec.loader.exec_module(dataset_mod)  # type: ignore
per_sample_minmax = dataset_mod.per_sample_minmax


def _load_model(ckpt_path: str | None, device: str) -> EncoderClassifier:
    model = EncoderClassifier(input_hw=(121, 104)).to(device)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state, strict=False)
        print(f"  loaded {ckpt_path}")
    else:
        print(f"  WARNING: checkpoint {ckpt_path} not found; using random weights")
    model.eval()
    return model


@torch.no_grad()
def _extract_embeddings(model: EncoderClassifier, images: np.ndarray, batch: int = 256, device: str = "cpu") -> np.ndarray:
    out = []
    for start in range(0, len(images), batch):
        chunk = images[start : start + batch]
        imgs = np.stack([per_sample_minmax(im) for im in chunk])[:, None]
        x = torch.from_numpy(imgs).to(device)
        z = model.encoder(x).cpu().numpy()
        out.append(z)
        print(f"    encoded {start + len(chunk):,}/{len(images):,}", end="\r")
    print()
    return np.concatenate(out, axis=0)


def _to_matlab_cell_str(values: np.ndarray | list[str]) -> np.ndarray:
    return np.array([str(v) for v in values], dtype=object)


def _save_matlab(out_path: Path, embeddings: np.ndarray, labels: np.ndarray, call_types: np.ndarray, sites: np.ndarray, dasars: np.ndarray, sample_ids: np.ndarray | None = None) -> None:
    # Write an HDF5-backed MAT v7.3 file so MATLAB can load large datasets
    # Use UTF-8 variable-length string dtype for metadata fields.
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("embeddings", data=embeddings.astype(np.float32), compression="gzip")
        f.create_dataset("labels", data=labels.reshape(-1, 1).astype(np.uint8), compression="gzip")
        f.create_dataset("call_type", data=np.array(call_types.astype(str), dtype=object), dtype=str_dt)
        f.create_dataset("site", data=np.array(sites.astype(str), dtype=object), dtype=str_dt)
        f.create_dataset("dasar", data=np.array(dasars.astype(str), dtype=object), dtype=str_dt)
        if sample_ids is not None:
            f.create_dataset("sample_id", data=np.array(sample_ids.astype(str), dtype=object), dtype=str_dt)


def _embeddings_to_image(embeddings: np.ndarray, height: int, width: int, method: str = "tile") -> np.ndarray:
    """Convert (N, D) embeddings to (N, height, width) image-like arrays.

    Methods:
      - "tile": repeat the vector values to fill the target length.
      - "pad": place vector at start and zero-pad the remainder.
      - "interp": 1D linear interpolation to stretch vector to target length.
    """
    n, d = embeddings.shape
    target = int(height) * int(width)
    out = np.zeros((n, height, width), dtype=np.float32)
    for i in range(n):
        v = embeddings[i].astype(np.float32)
        if method == "tile":
            reps = int(np.ceil(target / d))
            long = np.tile(v, reps)[:target]
        elif method == "pad":
            long = np.concatenate([v, np.zeros(max(0, target - d), dtype=np.float32)])[:target]
        elif method == "interp":
            if d == 1:
                long = np.full(target, v[0], dtype=np.float32)
            else:
                src_x = np.linspace(0, 1, d)
                dst_x = np.linspace(0, 1, target)
                long = np.interp(dst_x, src_x, v).astype(np.float32)
        else:
            raise ValueError(f"unknown method: {method}")
        out[i] = long.reshape(height, width)
    return out


def build_all(data_path: str, scratch_ckpt: str | None, warmstart_ckpt: str | None, out_dir: str, batch: int, device: str) -> None:
    print(f"Loading data from {data_path} …")
    data = np.load(data_path, allow_pickle=True)
    images = data["images"]
    labels = data["label"].astype(int)
    call_types = data["call_type"].astype(str)
    sites = data["site"].astype(str)
    dasars = data["dasar"].astype(str)

    print(f"  images: {images.shape}  count={len(images):,}")

    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)

    runs = [("scratch", scratch_ckpt), ("warmstart", warmstart_ckpt)]
    for run_name, ckpt in runs:
        if not ckpt:
            continue
        print(f"\nProcessing run: {run_name}")
        model = _load_model(ckpt, device)
        embeddings = _extract_embeddings(model, images, batch=batch, device=device)
        print(f"  embeddings shape: {embeddings.shape}")
        sample_ids = np.array([f"{sites[i]}_{dasars[i]}_{i}" for i in range(len(images))], dtype=object)
        out_path = out_path_dir / f"{run_name}_latents.mat"
        # Save original embeddings
        _save_matlab(out_path, embeddings, labels, call_types, sites, dasars, sample_ids=sample_ids)
        # Optionally also save image-shaped embeddings if requested via env var
        mat_shape = None
        try:
            import os
            ms = os.environ.get("MAT_SHAPE")
            mm = os.environ.get("MAT_METHOD", "tile")
            if ms:
                h, w = [int(x) for x in ms.split("x")]
                emb_img = _embeddings_to_image(embeddings, h, w, method=mm)
                # append to same HDF5 file as dataset 'embeddings_img'
                import h5py
                with h5py.File(str(out_path), "a") as f:
                    if "embeddings_img" in f:
                        del f["embeddings_img"]
                    f.create_dataset("embeddings_img", data=emb_img, compression="gzip")
                print(f"  wrote image-shaped embeddings -> {out_path}['embeddings_img'] ({h}x{w})")
        except Exception:
            pass
        print(f"  wrote {out_path}")


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--scratch", default=None)
    p.add_argument("--warmstart", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    build_all(a.data, a.scratch, a.warmstart, a.out_dir, a.batch, a.device)
