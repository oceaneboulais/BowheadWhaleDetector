"""Re-render the exact spectrograms shown in Latent32_reconstructions[2].jpg
in the same paper-matching 2-row (input/reconstructed) format used by
``plot_reconstruction_examples.py`` (see
``reconstructions_Autoencoder_v13_..._part2.png``).

The original Latent32_reconstructions.jpg/2.jpg thumbnails are left
untouched; this script only locates the same underlying raw .mat
spectrograms (by stem, read off the jpg titles) and re-plots them in the
cleaner format.

Note: the checkpoint used by ``plot_reconstruction_examples.py``
(``runs/ae_100k_matched/autoencoder_clean.pt``) is a *different* trained
autoencoder from the one that produced the original Latent32 jpgs -- its
decoder uses nn.Upsample+Conv2d (which blurs closely-spaced harmonic bands
together), whereas the original jpgs' decoder used true
nn.ConvTranspose2d layers (matching the architecture described in the
paper's Methods section), which preserves fine multi-band structure much
better. This script therefore defines the ConvTranspose2d decoder and loads
the actual checkpoint that generated the original jpgs
(``Autoencoder_v13_..._Date20251228-124835.dir/autoencoder_clean.pth``).

Run:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.models.plot_latent32_reconstruction_examples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from torch import nn

from bowhead.data.dataset import per_sample_minmax

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})


class _Autoencoder(nn.Module):
    """True ConvTranspose2d-decoder architecture (matches the Methods section
    and the checkpoint that produced the original Latent32 reconstructions)."""

    def __init__(self, base_channels: int = 32, latent_dim: int = 32,
                 input_hw: tuple[int, int] = (121, 104)) -> None:
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(c, c * 2, 3, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_hw)
            enc_out = self.encoder(dummy)
        self._flat_dim = int(enc_out.numel())
        self._enc_spatial = enc_out.shape[-2:]

        self.to_latent = nn.Sequential(
            nn.Linear(self._flat_dim, latent_dim * 2), nn.ReLU(True),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2), nn.ReLU(True),
            nn.Linear(latent_dim * 2, self._flat_dim), nn.ReLU(True),
        )
        H_e, W_e = self._enc_spatial
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2), nn.BatchNorm2d(c * 2), nn.ReLU(True),
            nn.ConvTranspose2d(c * 2, c, 2, stride=2), nn.BatchNorm2d(c), nn.ReLU(True),
            nn.ConvTranspose2d(c, 1, 2, stride=2),
        )
        self._enc_channels = c * 4
        self._H_e, self._W_e = H_e, W_e
        self._out_hw = input_hw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.to_latent(self.encoder(x).flatten(1))
        h = self.from_latent(z).view(-1, self._enc_channels, self._H_e, self._W_e)
        out = self.decoder(h)
        H_out, W_out = self._out_hw
        if out.shape[-2:] != (H_out, W_out):
            out = out[:, :, :H_out, :W_out]
        return out

# Exact stems shown (titles truncated) in the two Latent32_reconstructions jpgs.
LATENT32_STEMS_1 = [
    "S510D0T20100829T170334_Type2",
    "S314D0T20140818T222025_Type1",
    "S314D0T20140818T154158_Type1",
]
LATENT32_STEMS_2 = [
    "S314D0T20140828T010058_Type1",
    "S514G0T20140828T214824_Type7",
    "S510A0T20100927T144407_Type2",
]

RAW_DATASET_DIR = (
    "/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets/"
    "Unsupervised_database_MostlyManual.dir"
)


def _load_snr_gram(stem: str, raw_dir: str) -> np.ndarray:
    path = Path(raw_dir) / f"{stem}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Could not locate {stem}.mat under {raw_dir}")
    mat = sio.loadmat(str(path), simplify_cells=True)
    return np.asarray(mat["SNR_gram"], dtype=np.uint8)


def _render(stems: list[str], raw_dir: str, out_path: Path, model: "_Autoencoder") -> None:
    raw_images = np.stack([_load_snr_gram(stem, raw_dir) for stem in stems])
    inputs = np.stack([per_sample_minmax(img) for img in raw_images])
    with torch.no_grad():
        x = torch.from_numpy(inputs).unsqueeze(1).float()
        recon = model(x).squeeze(1).numpy()

    n = len(stems)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.6))
    duration_s, freq_hz = 3.0, 500.0
    extent = (0.0, duration_s, 0.0, freq_hz)
    for col in range(n):
        axes[0, col].imshow(inputs[col], cmap="viridis", origin="lower", aspect="auto", extent=extent)
        axes[1, col].imshow(recon[col], cmap="viridis", origin="lower", aspect="auto", extent=extent)
        for row in range(2):
            ax = axes[row, col]
            ax.set_xticks([0, 1, 2, 3])
            ax.set_yticks([0, 100, 200, 300, 400, 500])
            ax.tick_params(labelsize=10)
            if row == 1:
                ax.set_xlabel("Time (s)", fontsize=12)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel("Frequency (Hz)", fontsize=12)
            else:
                ax.set_yticklabels([])

    fig.text(0.01, 0.76, "Input\nspectrograms", fontsize=16, fontweight="bold",
              ha="center", va="center", rotation=90)
    fig.text(0.01, 0.30, "Reconstructed", fontsize=16, fontweight="bold",
              ha="center", va="center", rotation=90)

    fig.tight_layout(rect=(0.05, 0, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


ORIGINAL_CHECKPOINT = (
    "/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results/"
    "Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/"
    "autoencoder_clean.pth"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=ORIGINAL_CHECKPOINT)
    parser.add_argument("--raw-dir", default=RAW_DATASET_DIR)
    parser.add_argument("--out1", default="paper/Figures/Latent32_reconstructions_matched_format.png")
    parser.add_argument("--out2", default="paper/Figures/Latent32_reconstructions2_matched_format.png")
    args = parser.parse_args()

    model = _Autoencoder()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    _render(LATENT32_STEMS_1, args.raw_dir, Path(args.out1), model)
    _render(LATENT32_STEMS_2, args.raw_dir, Path(args.out2), model)


if __name__ == "__main__":
    main()
