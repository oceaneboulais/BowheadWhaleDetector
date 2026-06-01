"""Frozen-embedding extractors for the transfer benchmarks.

Each embedder maps a batch of fixed-length waveforms -> (N, D) embeddings. Heavy
backends (tensorflow, tensorflow_hub) are imported lazily inside ``_load`` so
this module imports on machines without them; instantiate/run only on the GPU
cluster where the deps + model weights are present.

Model handles below are CONFIGURABLE and must be confirmed on the cluster — the
exact TF-Hub / Kaggle handles and embedding dims should be verified against the
versions actually installed (see ``EMBED_DIMS`` for the documented sizes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bowhead.transfer.preprocess import (
    FrequencyShiftConfig,
    preprocess_clip,
    native_arm,
    shifted_arm,
)

# Documented embedding sizes / model sample rates (verify on cluster).
EMBED_DIMS = {"birdnet": 1024, "perch": 1536, "gmwm": 1280}
MODEL_SR = {"birdnet": 48000, "perch": 32000, "gmwm": 24000}
MODEL_WINDOW_S = {"birdnet": 3.0, "perch": 5.0, "gmwm": 5.0}


@dataclass
class EmbedderSpec:
    name: str                       # "birdnet" | "perch" | "gmwm"
    handle: str                     # TF-Hub / Kaggle / local path (set on cluster)
    arm: str = "native"             # "native" or "shifted"
    native_sr: int = 1000           # bowhead DASAR rate
    batch_size: int = 64

    def preprocess_cfg(self) -> FrequencyShiftConfig:
        target_sr = MODEL_SR[self.name]
        window = MODEL_WINDOW_S[self.name]
        if self.arm == "shifted":
            return shifted_arm(self.native_sr, target_sr, window)
        return native_arm(self.native_sr, target_sr, window)


class HubEmbedder:
    """Generic TF-Hub embedder (BirdNET / Perch / GMWM share this interface)."""

    def __init__(self, spec: EmbedderSpec) -> None:
        self.spec = spec
        self.cfg = spec.preprocess_cfg()
        self._model = None  # lazy
        self.name = f"{spec.name}_{spec.arm}"

    def _load(self):
        if self._model is None:
            import tensorflow_hub as hub  # lazy; cluster-only
            self._model = hub.load(self.spec.handle)
        return self._model

    def embed(self, waveforms: np.ndarray) -> np.ndarray:
        """waveforms: (N, raw_len) at native_sr -> embeddings (N, D)."""
        import tensorflow as tf  # lazy; cluster-only

        model = self._load()
        clips = np.stack([preprocess_clip(w, self.cfg) for w in waveforms])
        out = []
        for start in range(0, len(clips), self.spec.batch_size):
            batch = tf.convert_to_tensor(
                clips[start:start + self.spec.batch_size], dtype=tf.float32
            )
            # TF-Hub bioacoustic models expose embeddings via an "embedding"
            # output (Perch/BirdNET) — ADAPT the call signature per model on the
            # cluster (some return a dict, some a tuple of (logits, embedding)).
            result = model.infer_tf(batch) if hasattr(model, "infer_tf") else model(batch)
            emb = result["embedding"] if isinstance(result, dict) else result
            out.append(np.asarray(emb))
        return np.concatenate(out, axis=0)
