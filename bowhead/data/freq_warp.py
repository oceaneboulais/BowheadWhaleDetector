"""Log-frequency (constant-Q-style) axis warp for the linear-frequency SNR
spectrograms used throughout this project.

Motivation (see paper/ml_paper/ml_manuscript.tex, "Future work"): on the
project's native linear 25-500 Hz frequency axis, the pixel spacing between a
fundamental F0 and its harmonics changes with F0 itself, so a
translation-invariant convolutional filter cannot learn one reusable
"harmonic stack" pattern that generalizes across calls of different pitch.
Resampling each spectrogram column onto a log-spaced frequency grid spanning
the same band makes that spacing constant in pixels regardless of F0
(Brown, 1991, "Calculation of a constant Q spectral transform",
J. Acoust. Soc. Am. 89(1):425-434), at the cost of no longer being a literal
constant-Q *transform* (no new FFT/filterbank is computed here -- this warps
the already-computed 121-bin linear-frequency spectrogram to approximate one).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def log_freq_warp(
    img: np.ndarray,
    freq_lo: float = 25.0,
    freq_hi: float = 500.0,
    n_freq_bins: int | None = None,
) -> np.ndarray:
    """Resample a linear-frequency spectrogram's frequency axis (axis 0) onto
    a log-spaced grid spanning ``[freq_lo, freq_hi]`` Hz.

    Parameters
    ----------
    img : np.ndarray
        (n_freq, n_time) spectrogram, frequency axis assumed linearly spaced
        from ``freq_lo`` to ``freq_hi`` Hz (matches this project's 121x104
        SNR spectrograms: 25-500 Hz).
    freq_lo, freq_hi : float
        Band edges of the input's linear frequency axis, in Hz.
    n_freq_bins : int, optional
        Number of output frequency bins (default: same as input, so the
        warped image is a drop-in replacement with unchanged shape).

    Returns
    -------
    np.ndarray
        (n_freq_bins, n_time) float32 spectrogram on a log-frequency axis.
    """
    n_freq, n_time = img.shape
    n_out = n_freq if n_freq_bins is None else n_freq_bins

    linear_freqs = np.linspace(freq_lo, freq_hi, n_freq)
    log_freqs = np.logspace(np.log10(freq_lo), np.log10(freq_hi), n_out)

    interp = interp1d(
        linear_freqs, img.astype(np.float32), axis=0,
        kind="linear", bounds_error=False, fill_value=0.0,
    )
    return interp(log_freqs).astype(np.float32)


def log_freq_warp_batch(
    images: np.ndarray,
    freq_lo: float = 25.0,
    freq_hi: float = 500.0,
    n_freq_bins: int | None = None,
) -> np.ndarray:
    """Apply :func:`log_freq_warp` to a batch of images, shape (N, H, W) or
    (N, C, H, W)."""
    if images.ndim == 3:
        return np.stack([
            log_freq_warp(im, freq_lo, freq_hi, n_freq_bins) for im in images
        ])
    if images.ndim == 4:
        return np.stack([
            np.stack([log_freq_warp(ch, freq_lo, freq_hi, n_freq_bins) for ch in im])
            for im in images
        ])
    raise ValueError(f"images must be 3- or 4-D, got shape {images.shape}")
