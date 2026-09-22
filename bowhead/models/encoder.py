"""Convolutional encoder trunk — mirrors the trained autoencoder's encoder.

Layout and module names are kept BYTE-FOR-BYTE compatible with
``Autoencoder_v02_LD32_20251118.py`` (Thode, BowheadDeepLearningMATLAB) so the
autoencoder's ``state_dict`` warm-starts the supervised classifier with a direct
key match. The autoencoder defines two sequential blocks:

    self.encoder   = nn.Sequential(  # conv trunk, flat-indexed
        Conv2d(1, C, 3, pad=1), BatchNorm2d(C), ReLU, MaxPool2d(2,2),          # 0-3
        Conv2d(C, 2C, 3, pad=1), BatchNorm2d(2C), ReLU, MaxPool2d(2,2),        # 4-7
        Conv2d(2C, 4C, 3, pad=1), BatchNorm2d(4C), ReLU, MaxPool2d(2,2))       # 8-11
    self.to_latent = nn.Sequential(
        Linear(flat, latent_dim*2), ReLU, Linear(latent_dim*2, latent_dim))   # 0,1,2

with C = base_channels (default 32), input (1, 121, 104) -> conv output
(4C, 15, 13), flat = 4C*15*13. Min-max [0,1] input normalization, MSE loss,
Adam(lr=1e-3). The decoder is NOT needed for classification and is omitted here.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ConvEncoder(nn.Module):
    """The autoencoder's encoder path (conv trunk + ``to_latent`` FC bottleneck).

    Matches the AE's ``self.encoder`` / ``self.to_latent`` attribute names and
    flat Sequential indexing so AE checkpoints load by direct key match. The
    flattened feature size is inferred from a dummy pass, so it adapts if the
    spectrogram dims differ from 121x104 (the draft also mentions 128x119).
    """

    def __init__(
        self,
        in_channels: int = 1,
        input_hw: tuple[int, int] = (121, 104),
        base_channels: int = 32,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self._flat_dim = self._infer_flat_dim(in_channels, input_hw)

        self.to_latent = nn.Sequential(
            nn.Linear(self._flat_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.latent_dim = latent_dim

    def _infer_flat_dim(self, in_channels: int, input_hw: tuple[int, int]) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_hw)
            feat = self.encoder(dummy)
        return int(feat.numel())

    @property
    def flat_dim(self) -> int:
        return self._flat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        return self.to_latent(h)


def load_ae_encoder(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    in_channels: int = 1,
    input_hw: tuple[int, int] = (121, 104),
    base_channels: int = 32,
    latent_dim: int = 32,
) -> ConvEncoder:
    """Load the encoder half of a full ``ImprovedAutoencoder`` checkpoint.

    The trained checkpoints (e.g. the ``Autoencoder_v13_..._Date20260416-180022``
    "AutoManual Combined 100K" run used for all current UMAP work) are saved as a
    flat ``state_dict`` with ``encoder.*`` / ``to_latent.*`` / ``from_latent.*`` /
    ``decoder.*`` keys — no wrapper dict, no key-prefix remapping needed. Only the
    ``encoder.*`` and ``to_latent.*`` keys are loaded (``strict=True`` against just
    those keys), since the decoder/from_latent halves are not needed to produce
    latent embeddings.
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected a state_dict-like mapping, got {type(ckpt)}")
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt.get("model_state", ckpt)))

    model = ConvEncoder(
        in_channels=in_channels,
        input_hw=input_hw,
        base_channels=base_channels,
        latent_dim=latent_dim,
    )
    encoder_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in encoder_keys}
    missing = encoder_keys - filtered.keys()
    if missing:
        raise KeyError(f"Checkpoint {checkpoint_path} is missing encoder keys: {sorted(missing)}")
    model.load_state_dict(filtered, strict=True)
    model.to(device)
    model.eval()
    return model
