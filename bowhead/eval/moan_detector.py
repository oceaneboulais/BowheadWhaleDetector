"""Classical multi-band energy detector — the "whale moan detector" baseline.

Ports the core detection statistic of ``matlab/matlab/MultipleBandEnergyDetector.m``
(Thode lab; a generalized multi-band SNR detector in the spirit of Baumgartner &
Mussoline 2011, "A generalized baleen whale call detection and classification
system") from a streaming/continuous-recording detector to a scorer that runs
directly on the pre-cut SNR-gram crops this repo already stores.

Why this adaptation is faithful rather than a re-invention:
  - The MATLAB detector sums equalized (dB re background) power across a bank
    of overlapping sub-bands spanning the bowhead call band, then requires the
    summed SNR to stay above a threshold for a minimum duration.
  - Our ``SNR_gram`` crops are ALREADY background-equalized dB-SNR images (that
    is what "SNR" means here), so the "equalization" stage is already baked in
    upstream in MATLAB — we only need to replicate the sub-band summation +
    duration-gated peak-tracking on the given crop.

Calibration constants (frequency/time axes, dB quantization) are taken directly
from the MATLAB pipeline that produced ``data/*.npz``:
  - ``Fs = 1000`` Hz, ``Nfft = 256`` -> frequency bin width ``dF = Fs/Nfft``.
  - Frequency axis starts at 25 Hz (``master_plot_call_type_samples.m``:
    ``FF = 25 + dF*(0:120)``).
  - ``ovlap = 0.75`` -> time bin width ``dT = (1-ovlap)*Nfft/Fs``.
  - Images are stored as ``uint8(image_scale_factor * SNR_dB)`` with
    ``image_scale_factor = 5`` (``master_create_datasets_v2.m``), so
    ``SNR_dB = uint8_value / 5``.
  - Detection band + sub-band bandwidth match the bowhead example in the
    MATLAB docstring: ``flo_det=25, fhi_det=350, bandwidth=37`` (50% hop).
"""

from __future__ import annotations

import numpy as np


class MultiBandEnergyDetector:
    """Rule-based Scorer: no training, matches the ``Scorer`` protocol.

    ``score(images)`` expects raw (unnormalized) SNR-gram arrays, i.e. the same
    convention as ``CNNScorer`` — uint8 or float, NOT pre-normalized to [0, 1].
    """

    name = "moan_detector"

    def __init__(
        self,
        fs: float = 1000.0,
        nfft: int = 256,
        ovlap: float = 0.75,
        freq_offset_hz: float = 25.0,
        flo_det: float = 25.0,
        fhi_det: float = 350.0,
        bandwidth_hz: float = 37.0,
        image_scale_factor: float = 5.0,
        min_time_s: float = 0.3,
    ) -> None:
        self.df = fs / nfft
        self.dt = (1.0 - ovlap) * nfft / fs
        self.freq_offset_hz = freq_offset_hz
        self.image_scale_factor = image_scale_factor
        self.min_cols = max(1, round(min_time_s / self.dt))

        flo = np.arange(flo_det, fhi_det, 0.5 * bandwidth_hz)
        fhi = flo + bandwidth_hz
        keep = fhi <= fhi_det
        flo, fhi = flo[keep], fhi[keep]
        self._lo_idx = np.clip(
            np.round((flo - freq_offset_hz) / self.df).astype(int), 0, None
        )
        self._hi_idx = np.round((fhi - freq_offset_hz) / self.df).astype(int)
        if len(self._lo_idx) == 0:
            raise ValueError("No sub-bands fit in [flo_det, fhi_det); check params.")

    def _score_one(self, gram_db: np.ndarray) -> float:
        n_freq = gram_db.shape[0]
        best = -np.inf
        for lo, hi in zip(self._lo_idx, self._hi_idx):
            hi = min(int(hi), n_freq - 1)
            if hi <= lo:
                continue
            # dB-sum across the sub-band -> equivalent SEL/dB-RMS per column.
            band_db = 10.0 * np.log10(
                self.df * np.sum(10.0 ** (gram_db[lo:hi + 1] / 10.0), axis=0) + 1e-12
            )
            if len(band_db) >= self.min_cols:
                kernel = np.ones(self.min_cols, dtype=np.float32) / self.min_cols
                sustained = np.convolve(band_db, kernel, mode="valid")
                cand = float(sustained.max())
            else:
                cand = float(band_db.max())
            best = max(best, cand)
        return best

    def score(self, images: np.ndarray) -> np.ndarray:
        if images.ndim == 4:
            images = images[:, 0, :, :]
        gram_db = images.astype(np.float32) / self.image_scale_factor
        return np.array([self._score_one(g) for g in gram_db], dtype=np.float32)
