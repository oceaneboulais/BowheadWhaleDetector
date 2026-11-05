"""Regenerate the paper's AE input-vs-reconstruction example figures.

Loads the Autoencoder_v13 checkpoint (``runs/ae_100k_matched/autoencoder_clean.pt``)
and the exact 10 raw spectrograms used in the originally-published figure (found by
their filenames on the mounted ``R3D_2024_1`` drive's master database), runs them
through the full encoder+decoder, and plots TWO separate 2-row x 5-column figures
(top: input spectrograms, bottom: reconstructions) with large, paper-matching row
labels — the first 5 stems in one figure, the last 5 in the other.

Run:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.models.plot_reconstruction_examples
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

# Exact stems from the originally-published figure (downloaded from Overleaf),
# resolved to real files on /Volumes/R3D_2024_1/BowheadDeepLearningMATLAB/BCB_Whale_Datasets.
ORIGINAL_STEMS = [
    "S308A0T20080828T030239_Type1",
    "S514G0T20140917T151757_Type1",
    "S508A0T20080929T225701_Type0",
    "S508A0T20080913T104136_Type1",
    "S508A0T20080929T215350_Type0",
    "S508A0T20080929T195338_Type0",
    "S312D0T20120825T104604_Type0",
    "S510G0T20100913T110806_Type4",
    "S514D0T20140828T223352_Type6",
    "S314A0T20140828T094541_Type2",
]

RAW_DATASET_DIRS = [
    "/Volumes/R3D_2024_1/BowheadDeepLearningMATLAB/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_ADG_Y08101214_centered_16Apr2026.dir",
    "/Volumes/R3D_2024_1/BowheadDeepLearningMATLAB/BCB_Whale_Datasets/Unsupervised_database_Auto_100K_ADG_Y08101214_centered_16Apr2026.dir",
    "/Volumes/R3D_2024_1/BowheadDeepLearningMATLAB/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214_centered.dir",
]


def _find_raw_mat(stem: str) -> Path:
    for d in RAW_DATASET_DIRS:
        candidate = Path(d) / f"{stem}.mat"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate {stem}.mat under any of {RAW_DATASET_DIRS}")


def _load_snr_gram(stem: str) -> np.ndarray:
    path = _find_raw_mat(stem)
    mat = sio.loadmat(str(path), simplify_cells=True)
    return np.asarray(mat["SNR_gram"], dtype=np.uint8)


class _Autoencoder(nn.Module):
    """Mirrors bowhead.train.train_ae._Autoencoder (same checkpoint keys)."""

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
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c * 4, c * 2, 3, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c * 2, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c, 1, 3, padding=1),
            nn.Sigmoid(),
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


def _render(stems: list[str], out_path: Path, model: "_Autoencoder") -> None:
    raw_images = np.stack([_load_snr_gram(stem) for stem in stems])
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/ae_100k_matched/autoencoder_clean.pt")
    parser.add_argument(
        "--out-prefix",
        default="paper/Figures/reconstructions_Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022",
    )
    args = parser.parse_args()

    model = _Autoencoder()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    first_half, second_half = ORIGINAL_STEMS[:5], ORIGINAL_STEMS[5:]
    _render(first_half, Path(f"{args.out_prefix}_part1.png"), model)
    _render(second_half, Path(f"{args.out_prefix}_part2.png"), model)


if __name__ == "__main__":
    main()
