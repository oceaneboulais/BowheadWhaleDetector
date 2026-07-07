"""Training-time augmentations for spectrogram inputs.

All transforms operate on float32 tensors of shape (B, C, H, W) that have
already been per-sample min-max normalised to [0, 1].

SpecAugment
-----------
Randomly masks contiguous blocks of frequency bins (rows) and time bins
(columns) by zeroing them out.  Hyper-parameters follow the "LD" policy from
Park et al. 2019 scaled to our 121×104 SNR-gram size:
    F = max frequency-mask width  (default 15 bins  ≈ 12% of 121)
    T = max time-mask width       (default 12 bins  ≈ 12% of 104)
    n_freq_masks / n_time_masks   (default 1 each)

Mixup
-----
Convex combination of two training examples and their labels (Zhang et al.
2018).  Returns mixed images and soft labels as floats for use with the
Focal-loss / BCE variant that accepts non-integer targets.
    alpha — Beta distribution parameter (default 0.2)
"""

from __future__ import annotations

import torch
import numpy as np


# ── SpecAugment ──────────────────────────────────────────────────────────────

def spec_augment(
    x: torch.Tensor,
    F: int = 15,
    T: int = 12,
    n_freq_masks: int = 1,
    n_time_masks: int = 1,
) -> torch.Tensor:
    """Apply SpecAugment in-place on a batch tensor (B, C, H, W).

    Masks are sampled independently per sample in the batch.
    """
    x = x.clone()
    B, C, H, W = x.shape
    for b in range(B):
        for _ in range(n_freq_masks):
            f = int(torch.randint(0, F + 1, (1,)).item())
            f0 = int(torch.randint(0, max(H - f, 1), (1,)).item())
            x[b, :, f0: f0 + f, :] = 0.0
        for _ in range(n_time_masks):
            t = int(torch.randint(0, T + 1, (1,)).item())
            t0 = int(torch.randint(0, max(W - t, 1), (1,)).item())
            x[b, :, :, t0: t0 + t] = 0.0
    return x


# ── Mixup ────────────────────────────────────────────────────────────────────

def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Mixup to a batch.

    Parameters
    ----------
    x : (B, C, H, W) float tensor
    y : (B,) int64 label tensor
    alpha : Beta distribution parameter (0 → no mixup)

    Returns
    -------
    x_mix : (B, C, H, W) mixed images
    y_mix : (B,) float soft labels (values in [0, 1])
    """
    if alpha <= 0.0:
        return x, y.float()

    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1 - lam)          # keep the dominant sample dominant

    B = x.size(0)
    idx = torch.randperm(B, device=x.device)

    x_mix = lam * x + (1 - lam) * x[idx]
    y_float = y.float()
    y_mix = lam * y_float + (1 - lam) * y_float[idx]
    return x_mix, y_mix
