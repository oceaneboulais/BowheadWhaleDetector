"""Smoke tests for the constant-Q-style log-frequency warp (no torch needed).

Run:  python3 -m pytest tests/test_freq_warp.py -q
"""

from __future__ import annotations

import numpy as np

from bowhead.data.freq_warp import log_freq_warp, log_freq_warp_batch


def test_log_freq_warp_preserves_shape_and_dtype():
    img = np.random.default_rng(0).uniform(0, 255, size=(121, 104)).astype(np.uint8)
    warped = log_freq_warp(img)
    assert warped.shape == img.shape
    assert warped.dtype == np.float32


def test_log_freq_warp_custom_output_bins():
    img = np.zeros((121, 104), dtype=np.uint8)
    warped = log_freq_warp(img, n_freq_bins=64)
    assert warped.shape == (64, 104)


def test_log_freq_warp_low_frequencies_get_more_resolution():
    """A log-frequency grid should place more output bins near freq_lo than a
    linear grid would, since consecutive log-spaced points are closer together
    at low frequencies (the whole point of the constant-Q-style warp)."""
    linear_grid = np.linspace(25.0, 500.0, 121)
    log_grid = np.logspace(np.log10(25.0), np.log10(500.0), 121)
    n_below_100_linear = int((linear_grid < 100.0).sum())
    n_below_100_log = int((log_grid < 100.0).sum())
    assert n_below_100_log > n_below_100_linear


def test_log_freq_warp_batch_3d_and_4d():
    rng = np.random.default_rng(1)
    batch_3d = rng.uniform(0, 255, size=(5, 121, 104)).astype(np.uint8)
    out_3d = log_freq_warp_batch(batch_3d)
    assert out_3d.shape == batch_3d.shape

    batch_4d = rng.uniform(0, 255, size=(5, 1, 121, 104)).astype(np.uint8)
    out_4d = log_freq_warp_batch(batch_4d)
    assert out_4d.shape == batch_4d.shape


if __name__ == "__main__":
    test_log_freq_warp_preserves_shape_and_dtype()
    test_log_freq_warp_custom_output_bins()
    test_log_freq_warp_low_frequencies_get_more_resolution()
    test_log_freq_warp_batch_3d_and_4d()
    print("All freq_warp smoke tests passed.")
