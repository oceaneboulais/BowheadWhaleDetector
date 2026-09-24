"""Port of matlab/matlab/evaluate_overlap_between_manual_automated.m.

For every manual detection, finds the automated detection with which it has
maximum fractional time-overlap, and records the manual index against that
automated detection's "slot" (up to 3 manual detections can share one
automated detection). Manual detections whose best fractional overlap is
negative (no true overlap) are left unmatched.

NOTE (faithful bug-for-bug port): the `ovlap` threshold argument is accepted
for signature compatibility but, exactly as in the original MATLAB function,
it is never actually applied -- any non-negative fractional overlap counts as
a candidate match. Do not "fix" this without also updating the MATLAB source,
or results will silently diverge from the original pipeline.
"""
from __future__ import annotations

import numpy as np


def evaluate_overlap_between_manual_automated(
    t1_s: np.ndarray, t1_e: np.ndarray, t2_s: np.ndarray, t2_e: np.ndarray, ovlap: float
) -> tuple[np.ndarray, np.ndarray]:
    """t1 = manual detections, t2 = automated detections (all times in seconds).

    Returns (score, manual_index), each shape (len(t2_s), 3), matching the
    MATLAB function's outputs (unused `ovlap` kept only for parity).
    """
    del ovlap  # intentionally unused -- see module docstring
    t1_s = np.asarray(t1_s, dtype=float).ravel()
    t1_e = np.asarray(t1_e, dtype=float).ravel()
    t2_s = np.asarray(t2_s, dtype=float).ravel()
    t2_e = np.asarray(t2_e, dtype=float).ravel()

    n2 = len(t2_s)
    score = np.full((n2, 3), np.nan)
    manual_index = np.full((n2, 3), np.nan)
    duration2 = t2_e - t2_s

    for i in range(len(t1_s)):
        tmin = np.maximum(t1_s[i], t2_s)
        tmax = np.minimum(t1_e[i], t2_e)
        duration1_i = t1_e[i] - t1_s[i]
        min_duration = np.minimum(duration1_i, duration2)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac_ovlap = (tmax - tmin) / min_duration

        if not np.any(np.isfinite(frac_ovlap)):
            continue
        imax = int(np.nanargmax(frac_ovlap))
        fmax = frac_ovlap[imax]
        if not (fmax >= 0):
            continue

        if np.isnan(manual_index[imax, 0]):
            score[imax, 0] = fmax
            manual_index[imax, 0] = i
        elif np.isnan(manual_index[imax, 1]):
            score[imax, 1] = fmax
            manual_index[imax, 1] = i
        elif np.isnan(manual_index[imax, 2]):
            score[imax, 2] = fmax
            manual_index[imax, 2] = i
        # else: "Too many manual detections match this automated detection" --
        # matches MATLAB, which silently drops the 4th+ match.

    return score, manual_index
