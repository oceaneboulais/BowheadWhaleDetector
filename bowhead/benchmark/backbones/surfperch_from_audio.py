"""SurfPerch (Williams et al. 2024, arXiv:2404.16436) embeddings computed from
REAL raw DASAR audio -- as opposed to a Griffin-Lim pseudo-waveform reconstructed
from a stored spectrogram (see ``bowhead.benchmark.backbones.birdnet_from_gram``,
which this module intentionally does NOT subclass, since there is no spectrogram
reconstruction step here at all).

Model: Kaggle slug ``google/surfperch/TensorFlow2/1`` (confirmed against
google-research/perch-hoplite's ``model_configs.py``): 32 kHz input, 5.0 s
window (160,000 samples), 1280-D ``embedding`` output, loaded via ``kagglehub``
+ ``tf.saved_model.load`` and called through the SavedModel's ``infer_tf``
convenience method (returns a dict keyed by output head, including
``"embedding"``).

Two preprocessing arms share the same real-audio input but differ in how the
native ~1 kHz DASAR signal is mapped onto SurfPerch's 32 kHz window (see
``bowhead/transfer/preprocess.py``):

  native  -- plain resample, no time/frequency change. Preferred default: the
             most directly relevant benchmark (Burns et al. 2025, arXiv:2512.03219,
             "Perch 2.0 transfers 'whale' to underwater tasks") evaluates
             SurfPerch itself on baleen whale species -- including species
             calling at or below bowhead's 25-500 Hz range -- using audio
             resampled straight to the model's native rate, with no
             frequency-shifting, and finds strong transfer performance.
  shifted -- the NARW "10x speed-up" recipe (bioRxiv 2025.07.11.664307),
             reported as a secondary ablation since that trick was validated
             only for BirdNET (bird-only training) on a different, harder task
             (individual identification), not for SurfPerch on call detection.
"""

from __future__ import annotations

import numpy as np

from bowhead.transfer.preprocess import FrequencyShiftConfig, native_arm, shifted_arm, preprocess_clip

SURFPERCH_SLUG = "google/surfperch/TensorFlow2/1"
TARGET_SR = 32_000
WINDOW_S = 5.0
EMBED_DIM = 1280
NATIVE_SR = 1000  # DASAR nominal sample rate


class SurfPerchFromAudioBackbone:
    """Real-audio SurfPerch embedder.

    Parameters
    ----------
    arm : "native" | "shifted"
        Preprocessing arm (see module docstring). Native is the primary,
        literature-supported default.
    speedup : float
        Speed-up factor for the "shifted" arm (ignored for "native").
    """

    name = "surfperch"
    embed_dim = EMBED_DIM

    def __init__(self, arm: str = "native", speedup: float = 10.0) -> None:
        if arm not in ("native", "shifted"):
            raise ValueError(f"arm must be 'native' or 'shifted', got {arm!r}")
        self.arm = arm
        self.speedup = speedup
        self.cfg: FrequencyShiftConfig = (
            native_arm(NATIVE_SR, TARGET_SR, WINDOW_S) if arm == "native"
            else shifted_arm(NATIVE_SR, TARGET_SR, WINDOW_S, speedup=speedup)
        )
        self._model = None

    @property
    def native_window_s(self) -> float:
        """Real (native-rate) seconds of audio to extract per clip for this arm:
        5 s for native (1:1), or 5*speedup s for shifted (so the sped-up clip
        fully fills the 5 s model window instead of mostly silence/padding)."""
        return WINDOW_S if self.arm == "native" else WINDOW_S * self.speedup

    def _load_model(self):
        if self._model is None:
            import kagglehub
            import tensorflow as tf

            path = kagglehub.model_download(SURFPERCH_SLUG)
            print(f"Loaded SurfPerch ({self.arm} arm) from {path}")
            self._model = tf.saved_model.load(path)
        return self._model

    def embed(self, waveforms: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """waveforms: (N, native_window_s * NATIVE_SR) float64 at NATIVE_SR ->
        (N, 1280) float32 embeddings."""
        model = self._load_model()
        clips = np.stack([preprocess_clip(w, self.cfg) for w in waveforms]).astype(np.float32)
        out = []
        for start in range(0, len(clips), batch_size):
            batch = clips[start:start + batch_size]
            result = model.infer_tf(batch)
            out.append(np.asarray(result["embedding"]))
        return np.concatenate(out, axis=0)
