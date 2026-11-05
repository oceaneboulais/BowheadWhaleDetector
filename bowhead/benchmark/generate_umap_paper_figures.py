"""Generate the real UMAP figures referenced in ``paper/BowheadAI_v0.tex``,
replacing the placeholder/screenshot filenames with actual pipeline output:

    Figures/Placeholder_UMAP_figure_training.png
    Figures/Screenshot_UMAP_color_duration1_transient_detector.png
    Figures/Screenshot_UMAP_color_type.png
    Figures/Screenshot_UMAP_color_peak_frequency.png

Pipeline
--------
1. Load the 100,000-image training draw (``data/spectrograms_100k_matched.npz``)
   and, optionally, a random sample of the flat evaluation-directory ``.mat``
   files (``S<site><yy><dasar><idx>T<date>T<hms>_Type<n>.mat``).
2. Encode every sampled spectrogram through the trained autoencoder's encoder
   trunk (``bowhead.models.encoder.load_ae_encoder``) to obtain 32-D latent
   vectors, exactly as described in Sec. II.D of the paper.
3. Project the combined (train + eval) latents to 3-D with UMAP
   (``bowhead.benchmark.unsupervised.projection.umap_project``).
4. For call-detection (CD) spectrograms only, derive three per-image
   features directly from the SNR spectrogram pixels (no separate metadata
   file exists for these): peak frequency, active-signal duration, and
   frequency-modulation slope. These stand in for the transient detector's
   own duration/peak-frequency estimates (Sec.~\\ref{sec:TransientDetector}
   in the paper) since that metadata is not stored in the training npz.
5. Render the four figures with matplotlib's 3-D scatter and save them to
   ``paper/Figures/``.

Example
-------
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.benchmark.generate_umap_paper_figures \\
        --eval-dir "/path/to/Unsupervised_database_Evaluation_...dir" \\
        --train-sample-n 20000 --eval-sample-n 5000
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_NPZ = REPO_ROOT / "data" / "spectrograms_100k_matched.npz"
DEFAULT_AE_CKPT = REPO_ROOT / "runs" / "ae_100k_matched" / "autoencoder_clean.pt"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "Figures"

FREQ_LO_HZ, FREQ_HI_HZ = 25.0, 500.0
WINDOW_S = 3.0
_EVAL_FNAME_RE = re.compile(r"_Type(?P<type>\d+)\.mat$")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_training_sample(npz_path: Path, sample_n: int | None, seed: int) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    n = d["images"].shape[0]
    idx = np.arange(n)
    if sample_n is not None and sample_n < n:
        rng = np.random.default_rng(seed)
        # Stratify by label so CD/OT balance is preserved in the subsample.
        pos = idx[d["label"] == 1]
        neg = idx[d["label"] == 0]
        n_pos = min(len(pos), sample_n // 2)
        n_neg = min(len(neg), sample_n - n_pos)
        idx = np.concatenate([
            rng.choice(pos, size=n_pos, replace=False),
            rng.choice(neg, size=n_neg, replace=False),
        ])
        rng.shuffle(idx)
    return {
        "images": d["images"][idx],
        "label": d["label"][idx],
        "call_type": d["call_type"][idx],
        "split": np.full(len(idx), "train", dtype=object),
    }


def sample_eval_dir(eval_dir: Path, sample_n: int, seed: int) -> dict:
    """Random, label-stratified sample of a flat ``*_Type<n>.mat`` directory."""
    from scipy.io import loadmat

    print(f"Listing {eval_dir} ...")
    entries = [e.name for e in eval_dir.iterdir() if e.name.endswith(".mat") and not e.name.startswith("._")]
    print(f"  found {len(entries):,} .mat files")

    rng = random.Random(seed)
    call_names, noncall_names = [], []
    for name in entries:
        m = _EVAL_FNAME_RE.search(name)
        if not m:
            continue
        (call_names if m.group("type") != "0" else noncall_names).append(name)

    # Preserve the dataset's natural ~8:1 non-call:call ratio when subsampling.
    n_call = min(len(call_names), max(1, sample_n // 9))
    n_noncall = min(len(noncall_names), sample_n - n_call)
    chosen = rng.sample(call_names, n_call) + rng.sample(noncall_names, n_noncall)
    rng.shuffle(chosen)

    images, labels, call_types = [], [], []
    for name in chosen:
        m = _EVAL_FNAME_RE.search(name)
        ct = int(m.group("type"))
        try:
            mat = loadmat(str(eval_dir / name))
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN skip {name}: {exc}")
            continue
        gram = None
        for cand in ("SNR_gram", "NTV_gram", "KEtoPE_gram", "Polar_gram"):
            if cand in mat:
                gram = mat[cand]
                break
        if gram is None:
            continue
        images.append(np.asarray(gram, dtype=np.float32))
        labels.append(0 if ct == 0 else 1)
        call_types.append(str(ct))

    print(f"  loaded {len(images):,} eval spectrograms ({n_call} call, {n_noncall} non-call requested)")
    return {
        "images": np.stack(images) if images else np.zeros((0, 121, 104), dtype=np.float32),
        "label": np.array(labels, dtype=np.int64),
        "call_type": np.array(call_types, dtype="<U1"),
        "split": np.full(len(images), "eval", dtype=object),
    }


# --------------------------------------------------------------------------- #
# AE encoding
# --------------------------------------------------------------------------- #
def encode_images(images: np.ndarray, ae_checkpoint: Path, device: str, batch_size: int = 256) -> np.ndarray:
    import torch

    from bowhead.data.dataset import per_sample_minmax
    from bowhead.models.encoder import load_ae_encoder

    encoder = load_ae_encoder(ae_checkpoint, device=device)
    latents = np.zeros((images.shape[0], encoder.latent_dim), dtype=np.float32)
    for start in range(0, images.shape[0], batch_size):
        batch = images[start:start + batch_size]
        norm = np.stack([per_sample_minmax(img) for img in batch])
        x = torch.from_numpy(norm)[:, None, :, :].to(device)
        with torch.no_grad():
            z = encoder(x).cpu().numpy()
        latents[start:start + len(batch)] = z
    return latents


# --------------------------------------------------------------------------- #
# Per-image spectrogram feature extraction (proxy for the transient detector's
# own duration/peak-frequency/slope estimates, which are not stored per-image
# in the training npz).
# --------------------------------------------------------------------------- #
def spectrogram_features(img: np.ndarray) -> tuple[float, float, float]:
    """Return (peak_frequency_hz, duration_s, spectral_slope_hz_per_s)."""
    img = img.astype(np.float32)
    n_freq, n_time = img.shape
    freq_axis = FREQ_LO_HZ + np.arange(n_freq) / (n_freq - 1) * (FREQ_HI_HZ - FREQ_LO_HZ)
    dt = WINDOW_S / n_time

    col_max = img.max(axis=0)
    global_max = col_max.max()
    active = col_max > (global_max - 10.0)  # within 10 dB of the peak
    if not active.any():
        active = col_max >= np.percentile(col_max, 90)

    peak_bin = int(np.unravel_index(np.argmax(img), img.shape)[0])
    peak_freq_hz = float(freq_axis[peak_bin])

    active_idx = np.flatnonzero(active)
    duration_s = float((active_idx[-1] - active_idx[0] + 1) * dt) if active_idx.size else 0.0

    if active_idx.size >= 2:
        col_peak_bin = np.argmax(img[:, active_idx], axis=0)
        col_peak_hz = freq_axis[col_peak_bin]
        t = active_idx * dt
        slope, _ = np.polyfit(t, col_peak_hz, 1)
    else:
        slope = 0.0
    return peak_freq_hz, duration_s, float(slope)


def batch_spectrogram_features(images: np.ndarray) -> dict[str, np.ndarray]:
    peak = np.empty(len(images), dtype=np.float32)
    dur = np.empty(len(images), dtype=np.float32)
    slope = np.empty(len(images), dtype=np.float32)
    for i, img in enumerate(images):
        peak[i], dur[i], slope[i] = spectrogram_features(img)
    return {"peak_frequency_hz": peak, "duration_s": dur, "spectral_slope_hz_per_s": slope}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _new_3d_axes(fig, pos, elev: float, azim: float):
    if isinstance(pos, tuple):
        ax = fig.add_subplot(*pos, projection="3d")
    else:
        ax = fig.add_subplot(pos, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_zlabel("UMAP 3", fontsize=9)
    # Lighten the default dark-grey pane edges/gridlines so scatter points stand out.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
    return ax


def plot_training_eval_grid(coords: np.ndarray, label: np.ndarray, split: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    # Center on the dataset mean and scale by the per-axis standard deviation,
    # matching the paper's description of this figure.
    coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)

    azimuths = (20, 80, 140)
    has_eval = bool((split == "eval").any())
    ncols = 2 if has_eval else 1
    fig = plt.figure(figsize=(6 * ncols, 16))

    for row, azim in enumerate(azimuths):
        for col, which in enumerate(("train", "eval") if has_eval else ("train",)):
            mask = split == which
            ax = _new_3d_axes(fig, (3, ncols, row * ncols + col + 1), elev=20, azim=azim)
            # Okabe-Ito colorblind-safe palette, matching other paper figures.
            for lbl, color, name in ((1, "#D55E00", "Call detection"), (0, "#0072B2", "Other transient")):
                sel = mask & (label == lbl)
                ax.scatter(coords[sel, 0], coords[sel, 1], coords[sel, 2],
                           s=3.5, alpha=0.35, c=color, label=name, linewidths=0)
            if row == 0:
                title = "Training dataset" if which == "train" else "Evaluation dataset"
                ax.set_title(f"{title}\n" + r"$\mathit{n}$" + f" = {int(mask.sum()):,}",
                              fontsize=14, fontweight="bold")
            if row == 0 and col == 0:
                ax.legend(loc="upper left", markerscale=6, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_colored_scalar(coords: np.ndarray, values: np.ndarray, label_text: str, out_path: Path,
                         cmap: str = "viridis") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 6))
    ax = _new_3d_axes(fig, 111, elev=20, azim=60)
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=4, alpha=0.6, c=values, cmap=cmap, linewidths=0)
    fig.colorbar(sc, ax=ax, shrink=0.6, label=label_text)
    ax.set_title(f"Training CDs colored by {label_text}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_categorical_call_type(coords: np.ndarray, call_type: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 6))
    ax = _new_3d_axes(fig, 111, elev=20, azim=60)
    types = sorted(set(call_type.tolist()))
    cmap = plt.get_cmap("tab10")
    for i, ct in enumerate(types):
        sel = call_type == ct
        ax.scatter(coords[sel, 0], coords[sel, 1], coords[sel, 2],
                   s=4, alpha=0.6, color=cmap(i % 10), label=f"Type {ct}", linewidths=0)
    ax.legend(loc="upper left", fontsize=8, markerscale=2, title="Call sub-type")
    ax.set_title("Training CDs colored by manually-assigned call sub-type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_three_panel_features(coords: np.ndarray, feats: dict[str, np.ndarray], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    # Single perceptually-uniform colormap across panels, matching the training/eval grid's styling.
    panels = [
        ("duration_s", "Duration (s)", "viridis"),
        ("peak_frequency_hz", "Peak frequency (Hz)", "viridis"),
        ("spectral_slope_hz_per_s", "Spectral slope (Hz/s)", "viridis"),
    ]
    fig = plt.figure(figsize=(18, 6))
    for i, (key, label_text, cmap) in enumerate(panels):
        ax = _new_3d_axes(fig, (1, 3, i + 1), elev=20, azim=60)
        sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                         s=3.5, alpha=0.35, c=feats[key], cmap=cmap, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.6, label=label_text)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label(label_text, fontsize=10)
        ax.set_title(f"({chr(97 + i)}) {label_text}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-npz", type=Path, default=DEFAULT_TRAIN_NPZ)
    p.add_argument("--eval-dir", type=Path, default=None,
                   help="Flat directory of *_Type<n>.mat eval spectrograms (optional)")
    p.add_argument("--ae-checkpoint", type=Path, default=DEFAULT_AE_CKPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--train-sample-n", type=int, default=20_000)
    p.add_argument("--eval-sample-n", type=int, default=5_000)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training sample from {args.train_npz} ...")
    train = load_training_sample(args.train_npz, args.train_sample_n, args.seed)
    print(f"  {len(train['images']):,} training images "
          f"({int((train['label'] == 1).sum()):,} CD / {int((train['label'] == 0).sum()):,} OT)")

    if args.eval_dir is not None:
        eval_data = sample_eval_dir(args.eval_dir, args.eval_sample_n, args.seed)
    else:
        eval_data = {"images": np.zeros((0, 121, 104), dtype=np.float32),
                     "label": np.zeros(0, dtype=np.int64),
                     "call_type": np.zeros(0, dtype="<U1"),
                     "split": np.zeros(0, dtype=object)}

    images = np.concatenate([train["images"], eval_data["images"]], axis=0)
    label = np.concatenate([train["label"], eval_data["label"]], axis=0)
    call_type = np.concatenate([np.asarray(train["call_type"]).astype(str),
                                 np.asarray(eval_data["call_type"]).astype(str)], axis=0)
    split = np.concatenate([train["split"], eval_data["split"]], axis=0)

    print(f"Encoding {len(images):,} spectrograms through the AE encoder ({args.ae_checkpoint}) ...")
    latents = encode_images(images, args.ae_checkpoint, args.device)

    print(f"UMAP-projecting {len(latents):,} x {latents.shape[1]}-D latents to 3-D ...")
    from bowhead.benchmark.unsupervised.projection import umap_project

    coords = umap_project(latents, n_components=3, n_neighbors=args.umap_neighbors,
                           min_dist=args.umap_min_dist, seed=args.seed)

    plot_training_eval_grid(coords, label, split, args.out_dir / "Placeholder_UMAP_figure_training.png")

    train_mask = split == "train"
    cd_mask = train_mask & (label == 1)
    cd_coords = coords[cd_mask]
    cd_images = images[cd_mask]
    cd_call_type = call_type[cd_mask]

    print(f"Computing per-image spectrogram features for {len(cd_images):,} training CDs ...")
    feats = batch_spectrogram_features(cd_images)

    plot_colored_scalar(cd_coords, feats["duration_s"], "Duration (s)",
                         args.out_dir / "Screenshot_UMAP_color_duration1_transient_detector.png")
    plot_categorical_call_type(cd_coords, cd_call_type, args.out_dir / "Screenshot_UMAP_color_type.png")
    plot_three_panel_features(cd_coords, feats, args.out_dir / "Screenshot_UMAP_color_peak_frequency.png")

    print("Done.")


if __name__ == "__main__":
    main()
