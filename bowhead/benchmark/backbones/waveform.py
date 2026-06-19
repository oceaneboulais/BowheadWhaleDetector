"""Waveform-based frozen-embedding backbones (cluster-only stubs).

These backbones map raw audio waveforms → (N, D) embeddings. They require:
  - Raw audio at the model's native sample rate (24–48 kHz)
  - TensorFlow ≥ 2.15 + tensorflow_hub ≥ 0.16
  - GPU cluster access (too slow for CPU inference at scale)

They are intentionally NOT imported at module load time; all heavy deps are
inside ``_load()``. This file imports cleanly on any machine.

Waveform models
---------------
birdnet   BirdNET 2.3, 48 kHz, 3-s windows, 1024-D embeddings.
          TF-Hub: "https://tfhub.dev/google/bird-vocalization-classifier/4"

perch     Perch 2.0, 32 kHz, 5-s windows, 1280-D embeddings.
          TF-Hub: verify handle on cluster; see Burns et al. 2025 (arXiv:2512.03219)

gmwm      Google Multispecies Whale Model, 24 kHz, 5-s windows, 1280-D embeddings.
          TF-Hub / Kaggle: Harvey et al. 2024

All three also support the two preprocessing arms (see bowhead/transfer/preprocess.py):
  native   — bandpass + resample to model SR, no pitch change
  shifted  — 10× speed-up trick (NARW recipe, bioRxiv 2025.07.11.664307)
             shifts 25–500 Hz → 250–5000 Hz, landing in the bird-sensitive band.

Interface mirrors ImageBackbone:
    .name    str
    .dim     int
    .embed(waveforms: np.ndarray) -> np.ndarray   # (N, raw_len) → (N, D) float32
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Re-use the preprocessing and hub-embedder infra from bowhead/transfer/
from bowhead.transfer.embedders import EmbedderSpec, HubEmbedder, EMBED_DIMS

# Cluster TF-Hub handles — update on the cluster before running.
# Verify exact handle strings against the TF-Hub / Kaggle model pages.
_HANDLES: dict[str, str] = {
    "birdnet": "https://tfhub.dev/google/bird-vocalization-classifier/4",
    "perch":   "https://tfhub.dev/google/multispecies_bird/1",   # ← verify on cluster
    "gmwm":    "https://www.kaggle.com/models/google/multispecies-whale/tensorFlow2/default/2",
}

WAVEFORM_BACKBONE_REGISTRY: dict[str, str] = _HANDLES


@dataclass
class WaveformBackboneSpec:
    """Configuration for one waveform backbone + preprocessing arm."""
    name: str                      # e.g. "birdnet_native", "perch_shifted"
    model_key: str                 # "birdnet" | "perch" | "gmwm"
    arm: str = "native"            # "native" | "shifted"
    native_sr: int = 1000          # DASAR sample rate


class WaveformBackbone:
    """Thin wrapper around HubEmbedder with the shared backbone interface."""

    def __init__(self, spec: WaveformBackboneSpec) -> None:
        hub_spec = EmbedderSpec(
            name=spec.model_key,
            handle=_HANDLES[spec.model_key],
            arm=spec.arm,
            native_sr=spec.native_sr,
        )
        self._embedder = HubEmbedder(hub_spec)
        self.name = spec.name or f"{spec.model_key}_{spec.arm}"
        self.dim  = EMBED_DIMS[spec.model_key]

    def embed(self, waveforms: np.ndarray) -> np.ndarray:
        """(N, raw_len) at native_sr → (N, D) embeddings.

        Raises ImportError with a helpful message if TensorFlow is not installed.
        """
        try:
            return self._embedder.embed(waveforms)
        except ImportError as e:
            raise ImportError(
                f"Waveform backbone '{self.name}' requires TensorFlow + tensorflow_hub. "
                "Install them on the GPU cluster, or use image backbones locally."
            ) from e


def load_waveform_backbone(
    model_key: str,
    arm: str = "native",
    native_sr: int = 1000,
) -> WaveformBackbone:
    """Instantiate a waveform backbone by model key and preprocessing arm.

    Parameters
    ----------
    model_key : str
        One of ``"birdnet"``, ``"perch"``, ``"gmwm"``.
    arm : str
        ``"native"`` (no frequency shift) or ``"shifted"`` (10× speed-up trick).
    native_sr : int
        DASAR recording sample rate (default 1000 Hz).
    """
    if model_key not in _HANDLES:
        raise KeyError(
            f"Unknown waveform model {model_key!r}. "
            f"Available: {list(_HANDLES)}"
        )
    spec = WaveformBackboneSpec(
        name=f"{model_key}_{arm}",
        model_key=model_key,
        arm=arm,
        native_sr=native_sr,
    )
    return WaveformBackbone(spec)
