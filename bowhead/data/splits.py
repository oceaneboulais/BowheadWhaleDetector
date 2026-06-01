"""Leakage-free grouped train/val/test splitting.

THE central rigor invariant of this study. One *unique call* (UC) produces
multiple near-identical *call detections* (CDs) across DASARs A/D/G, and calls
within a day are temporally autocorrelated. A naive random per-image split puts
near-duplicate views of the same call on both sides of the split, inflating
every metric. We therefore split by a **group key** so a given call (or
date x site) appears entirely in one partition.

Use ``group_col="unique_call"`` when a UC id is available (strictest), or
``group_col="date_site"`` for a coarser, even more conservative split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # StratifiedGroupKFold available in scikit-learn >= 1.0
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


@dataclass
class GroupedSplit:
    """Index arrays for one train/val/test partition (indices into the dataset)."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def summary(self, labels: np.ndarray, groups: np.ndarray) -> str:
        lines = []
        for name, idx in (("train", self.train), ("val", self.val), ("test", self.test)):
            pos = float(labels[idx].mean()) if len(idx) else float("nan")
            n_groups = len(np.unique(groups[idx]))
            lines.append(
                f"  {name:5s}: {len(idx):>7d} images | {n_groups:>6d} groups | "
                f"call fraction {pos:.3f}"
            )
        # Verify the core invariant: no group shared across partitions.
        overlap = (
            set(groups[self.train]) & set(groups[self.val])
            | set(groups[self.train]) & set(groups[self.test])
            | set(groups[self.val]) & set(groups[self.test])
        )
        lines.append(f"  group overlap across partitions: {len(overlap)} (must be 0)")
        return "\n".join(lines)


def grouped_split(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> GroupedSplit:
    """Stratified, group-aware train/val/test split.

    Parameters
    ----------
    labels : array of int, shape (N,)
        Binary call (1) / non-call (0) labels — used only to *stratify* so each
        partition keeps a similar call fraction; the split is still by group.
    groups : array, shape (N,)
        Group id per image (e.g. unique-call id, or "date_site" string). All
        images sharing a group land in the same partition.
    val_frac, test_frac : float
        Approximate fraction of *images* held out (group boundaries make these
        approximate, not exact).
    seed : int
        Reproducibility.

    Returns
    -------
    GroupedSplit
    """
    if StratifiedGroupKFold is None:  # pragma: no cover
        raise ImportError("scikit-learn >= 1.0 required for StratifiedGroupKFold")

    labels = np.asarray(labels)
    groups = np.asarray(groups)

    # First carve off the test set, then split the remainder into train/val.
    # _holdout returns indices into its input arrays, so compose carefully.
    test_idx, rest_idx = _holdout(labels, groups, test_frac, seed)

    # val_frac is relative to the whole dataset; rescale to the remainder.
    rel_val = val_frac / (1.0 - test_frac)
    val_rel, train_rel = _holdout(labels[rest_idx], groups[rest_idx], rel_val, seed + 1)

    return GroupedSplit(
        train=rest_idx[train_rel],
        val=rest_idx[val_rel],
        test=test_idx,
    )


def _holdout(
    labels: np.ndarray, groups: np.ndarray, frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split into (holdout, remainder) index arrays, grouped + stratified.

    Implemented via StratifiedGroupKFold with ``n_splits = round(1/frac)``; the
    first fold becomes the holdout. Returns indices into the *input* arrays.
    """
    n_splits = max(2, round(1.0 / frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    remainder_idx, holdout_idx = next(iter(sgkf.split(np.zeros(len(labels)), labels, groups)))
    return holdout_idx, remainder_idx


def make_date_site_group(date: np.ndarray, site: np.ndarray) -> np.ndarray:
    """Convenience: build a coarse ``date_site`` group key from two columns."""
    return np.array([f"{d}_{s}" for d, s in zip(date, site)])
