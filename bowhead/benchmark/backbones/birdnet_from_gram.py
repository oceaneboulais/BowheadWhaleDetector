"""BirdNET / Perch 2.0 frozen embeddings reconstructed from ``.mat`` SNR-grams.

The public TF-Hub model (``google/bird-vocalization-classifier/4``, what this
repo calls "birdnet") exposes exactly ONE signature:

    serving_default: inputs float32[None, 160000] -> {embedding (1280-D), logits}

i.e. it accepts ONLY raw 5-second / 32 kHz waveform — there is no spectrogram
input path to bypass the audio front-end (verified directly against the loaded
SavedModel's ``signatures['serving_default'].structured_input_signature``).

Since this repo's data is stored as pre-cut dB-SNR spectrogram crops (not raw
audio), this module reconstructs a pseudo-waveform per crop via inverse-STFT
(Griffin-Lim) and runs the REAL pretrained model on that reconstructed audio.
This is the same "out-of-domain transfer" spirit as Burns et al. 2025 ("Perch
2.0 transfers to whale"), with one caveat clearly surfaced wherever these
results are reported: phase is approximated by Griffin-Lim (SNR-grams store
magnitude/SNR only, not phase), so the reconstructed audio is not the original
recording, just a magnitude-consistent synthesis of it.

STFT calibration constants match the MATLAB pipeline that produced the grams
(see ``bowhead/eval/moan_detector.py`` docstring for the same derivation):
Fs=1000 Hz, Nfft=256, ovlap=0.75, frequency axis starts at 25 Hz,
``image_scale_factor=5`` (uint8 -> dB-SNR is ``value / 5``).
"""

from __future__ import annotations

import numpy as np

_BIRDNET_URL = "https://tfhub.dev/google/bird-vocalization-classifier/4"
_TARGET_SR = 32_000
_TARGET_SAMPLES = 5 * _TARGET_SR  # 160,000


class BirdNetFromGramBackbone:
    """Reconstructs audio from SNR-grams and embeds it with the real BirdNET model."""

    name = "birdnet"
    embed_dim = 1280

    def __init__(
        self,
        fs: float = 1000.0,
        nfft: int = 256,
        ovlap: float = 0.75,
        freq_offset_hz: float = 25.0,
        image_scale_factor: float = 5.0,
        griffinlim_iters: int = 32,
    ) -> None:
        import tensorflow_hub as hub  # lazy import: heavy optional dependency

        self.fs = fs
        self.nfft = nfft
        self.hop = int(round((1.0 - ovlap) * nfft))
        self.freq_offset_hz = freq_offset_hz
        self.image_scale_factor = image_scale_factor
        self.griffinlim_iters = griffinlim_iters
        self.n_full_bins = nfft // 2 + 1  # 129 for nfft=256
        self.df = fs / nfft
        self._bin_offset = int(round(freq_offset_hz / self.df))

        print(f"Loading BirdNET from {_BIRDNET_URL} (first call may download) ...")
        self._model = hub.load(_BIRDNET_URL)
        self._infer = self._model.signatures["serving_default"]

    def _reconstruct_waveform(self, gram_uint8: np.ndarray) -> np.ndarray:
        import librosa

        gram_db = gram_uint8.astype(np.float32) / self.image_scale_factor
        n_bins, n_cols = gram_db.shape
        full_mag = np.full((self.n_full_bins, n_cols), 1e-3, dtype=np.float32)
        lo = self._bin_offset
        hi = min(lo + n_bins, self.n_full_bins)
        full_mag[lo:hi, :] = 10.0 ** (gram_db[: hi - lo, :] / 20.0)

        y = librosa.griffinlim(
            full_mag,
            n_iter=self.griffinlim_iters,
            hop_length=self.hop,
            win_length=self.nfft,
            n_fft=self.nfft,
            window="hann",
        )
        y = librosa.resample(y, orig_sr=self.fs, target_sr=_TARGET_SR)

        peak = np.abs(y).max()
        if peak > 1e-8:
            y = 0.9 * y / peak

        if len(y) >= _TARGET_SAMPLES:
            start = (len(y) - _TARGET_SAMPLES) // 2
            y = y[start:start + _TARGET_SAMPLES]
        else:
            pad = _TARGET_SAMPLES - len(y)
            y = np.pad(y, (pad // 2, pad - pad // 2))
        return y.astype(np.float32)

    def embed(self, images: np.ndarray) -> np.ndarray:
        """images: (N, H, W) raw uint8 SNR-grams -> (N, 1280) embeddings."""
        if images.ndim == 4:
            images = images[:, 0, :, :]
        waveforms = np.stack([self._reconstruct_waveform(g) for g in images])
        out = self._infer(inputs=waveforms.astype(np.float32))
        return out["output_1"].numpy()
