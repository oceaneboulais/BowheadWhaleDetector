"""Export 32-D autoencoder latent embeddings for the balanced eval dataset to MATLAB.

Loads ``data/spectrograms_eval_200k_balanced.npz``, encodes every image through
the trained 32-D convolutional autoencoder, and writes a ``latent_embeddings.mat``
file whose structure mirrors the one produced by the original MATLAB AE training
pipeline (Autoencoder_v15 / LD32_Hybrid) so it can be dropped straight into any
existing UMAP or t-SNE MATLAB workflow.

Checkpoint used: ``runs/ae_full/autoencoder_clean.pt``

Output fields in the .mat file
-------------------------------
latent_embeddings  : (200000, 32) float32  — 32-D encoder output
label              : (200000, 1)  int32    — 0 = non-call, 1 = bowhead call
call_type          : (200000, 1)  char     — MATLAB char array (Type 0–7)
site               : (200000, 1)  char     — '3' or '5'
dasar              : (200000, 1)  char     — DASAR letter (A, D, G)
date               : (200000, 1)  char     — YYYYMMDD string
dataset_label      : string               — provenance tag
original_filenames : (1, 200000)  cell    — stem of each source .mat file
                                            (reconstructed from metadata)

Run:
    source /Users/oboulais/.venv_py31018/bin/activate
    PYTHONPATH=. python -m bowhead.eval.export_balanced_eval_latents

    # Custom paths:
    PYTHONPATH=. python -m bowhead.eval.export_balanced_eval_latents \\
        --data  data/spectrograms_eval_200k_balanced.npz \\
        --ckpt  runs/ae_full/autoencoder_clean.pt \\
        --out   data/balanced_eval_latents.mat \\
        --batch 512 --device cpu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from torch import nn


# ---------------------------------------------------------------------------
# Autoencoder — must match architecture in bowhead/train/train_ae.py exactly
# ---------------------------------------------------------------------------

class _Autoencoder(nn.Module):
    def __init__(self, base_channels: int = 32, latent_dim: int = 32,
                 input_hw: tuple[int, int] = (121, 104)) -> None:
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c,   3, padding=1), nn.BatchNorm2d(c),   nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(c, c*2, 3, padding=1), nn.BatchNorm2d(c*2), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(c*2, c*4, 3, padding=1), nn.BatchNorm2d(c*4), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        with torch.no_grad():
            enc_out = self.encoder(torch.zeros(1, 1, *input_hw))
        self._flat_dim = int(enc_out.numel())
        self._H_e, self._W_e = enc_out.shape[-2:]
        self._enc_channels = c * 4
        self.to_latent = nn.Sequential(
            nn.Linear(self._flat_dim, latent_dim * 2), nn.ReLU(True),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.to_latent(h.flatten(1))


def _load_ae(ckpt_path: Path, device: str) -> _Autoencoder:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    # Support both full state_dict and checkpoint dicts
    if "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif "encoder" in ckpt:
        # Partial checkpoint — reconstruct manually
        state = {}
        for part in ("encoder", "to_latent"):
            for k, v in ckpt[part].items():
                state[f"{part}.{k}"] = v
    else:
        state = ckpt
    model = _Autoencoder()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARN missing keys: {missing[:5]}")
    model.to(device).eval()
    return model


def _per_sample_minmax(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo + 1e-12)


@torch.no_grad()
def _encode_all(model: _Autoencoder, images: np.ndarray,
                batch_size: int, device: str) -> np.ndarray:
    n = len(images)
    latents = np.empty((n, model.latent_dim), dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = images[start:end]
        imgs = np.stack([_per_sample_minmax(im) for im in chunk])[:, None]
        x = torch.from_numpy(imgs).to(device)
        latents[start:end] = model.encode(x).cpu().numpy()
        if (end % 20_000) == 0 or end == n:
            rate = end / (time.time() - t0)
            eta  = (n - end) / rate / 60
            print(f"  encoded {end:,}/{n:,}  ({rate:.0f}/s, ETA {eta:.1f} min)")
    return latents


def export(
    data_path: Path,
    ckpt_path: Path,
    out_path: Path,
    batch_size: int = 512,
    device: str = "cpu",
) -> None:
    # ------------------------------------------------------------------ load data
    print(f"Loading {data_path} ...")
    d = np.load(str(data_path), allow_pickle=True)
    images     = d["images"]         # (N, 121, 104) uint8
    labels     = d["label"]          # (N,) int64
    dates      = d["date"]           # (N,) U8
    sites      = d["site"]           # (N,) U1
    dasars     = d["dasar"]          # (N,) U1
    call_types = d["call_type"]      # (N,) U1
    n = len(labels)
    print(f"  {n:,} samples  calls={int(labels.sum()):,}  non-calls={int((labels==0).sum()):,}")

    # ------------------------------------------------------------------ load model
    print(f"\nLoading AE checkpoint from {ckpt_path} ...")
    model = _load_ae(ckpt_path, device)
    print(f"  latent_dim={model.latent_dim}  device={device}")

    # ------------------------------------------------------------------ encode
    print(f"\nEncoding {n:,} images (batch={batch_size}) ...")
    latents = _encode_all(model, images, batch_size, device)
    print(f"  latents shape: {latents.shape}  dtype={latents.dtype}")

    # ------------------------------------------------------------------ build .mat
    # Reconstruct pseudo-filenames from metadata (no stems stored in NPZ)
    # Format matches original: S{site}{yy}{dasar}?T{YYYYMMDD}T??????_Type{t}.mat
    # The time component is unknown so we use a placeholder (000000).
    pseudo_fnames = np.array(
        [f"S{si}{dt[2:4]}{da}0T{dt}T000000_Type{ct}.mat"
         for si, dt, da, ct in zip(sites, dates, dasars, call_types)],
        dtype=object,
    ).reshape(1, -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {out_path} ...")
    savemat(
        str(out_path),
        {
            "latent_embeddings":  latents,
            "label":              labels.reshape(-1, 1).astype(np.int32),
            "call_type":          np.array([[ct] for ct in call_types], dtype=object),
            "site":               np.array([[si] for si in sites],      dtype=object),
            "dasar":              np.array([[da] for da in dasars],      dtype=object),
            "date":               np.array([[dt] for dt in dates],       dtype=object),
            "dataset_label":      "BalancedEval_200K_50pct_calls",
            "original_filenames": pseudo_fnames,
            "n_calls":            int(labels.sum()),
            "n_noncalls":         int((labels == 0).sum()),
        },
        do_compression=True,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")

    # ------------------------------------------------------------------ summary
    unique_ds = sorted(set(f"{dt}_{si}" for dt, si in zip(dates, sites)))
    print(f"\n=== Export summary ===")
    print(f"  samples         : {n:,}")
    print(f"  calls (label=1) : {int(labels.sum()):,}  (50.0%)")
    print(f"  non-calls       : {int((labels==0).sum()):,}  (50.0%)")
    print(f"  latent dims     : {latents.shape[1]}")
    print(f"  sites           : { {s: int((sites==s).sum()) for s in ('3','5')} }")
    print(f"  years           : { {y: int(sum(1 for dt in dates if dt[:4]==y)) for y in ('2008','2010','2012','2014')} }")
    print(f"  date+site groups: {len(unique_ds)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export AE latent embeddings for balanced eval set to MATLAB .mat"
    )
    parser.add_argument("--data",  type=Path, default=Path("data/spectrograms_eval_200k_balanced.npz"))
    parser.add_argument("--ckpt",  type=Path, default=Path("runs/ae_full/autoencoder_clean.pt"))
    parser.add_argument("--out",   type=Path, default=Path("data/balanced_eval_latents.mat"))
    parser.add_argument("--batch", type=int,  default=512)
    parser.add_argument("--device", default="cpu",
                        help="'cpu', 'cuda', or 'mps'  (default: cpu)")
    args = parser.parse_args()
    export(
        data_path=args.data,
        ckpt_path=args.ckpt,
        out_path=args.out,
        batch_size=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
