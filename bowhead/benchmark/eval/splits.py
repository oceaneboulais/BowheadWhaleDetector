"""Temporal-drift and site-shift evaluation splits.

Constructs index arrays for:

  temporal_drift   — train on early years, test on later years.
                     Tests whether a model generalises across time (decade-scale
                     climate + equipment changes, shipping traffic growth).

  site_shift       — train on one DASAR site, test on the other.
                     Tests generalisation across spatial locations with different
                     background noise and call-to-noise ratios.

Both are strict splits: every sample in the test set comes from a time/place
the model has never seen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvalSplit:
    """Index arrays partitioning the full dataset for one evaluation axis."""
    name: str
    train: np.ndarray    # indices into the full dataset
    test: np.ndarray

    def summary(self, labels: np.ndarray) -> str:
        rows = []
        for tag, idx in (("train", self.train), ("test", self.test)):
            if len(idx) == 0:
                rows.append(f"  {self.name} {tag:5s}: EMPTY")
                continue
            pos = float(labels[idx].mean())
            rows.append(
                f"  {self.name} {tag:5s}: {len(idx):>8,} samples  "
                f"call fraction {pos:.4f}"
            )
        return "\n".join(rows)


def temporal_drift_split(
    labels: np.ndarray,
    date: np.ndarray,
    train_years: tuple[str, ...] = ("2008", "2010"),
    test_years: tuple[str, ...] = ("2012", "2014"),
) -> EvalSplit:
    """Split by recording year.

    Parameters
    ----------
    labels : (N,) int
    date   : (N,) YYYYMMDD strings (first four chars = year)
    train_years, test_years : year strings e.g. ("2008", "2010")
    """
    years = np.array([d[:4] for d in date])
    train_mask = np.isin(years, list(train_years))
    test_mask  = np.isin(years, list(test_years))

    if not train_mask.any():
        raise ValueError(
            f"No samples found for train_years={train_years}. "
            f"Available years: {sorted(set(years.tolist()))}"
        )
    if not test_mask.any():
        raise ValueError(
            f"No samples found for test_years={test_years}. "
            f"Available years: {sorted(set(years.tolist()))}"
        )

    return EvalSplit(
        name="temporal",
        train=np.where(train_mask)[0],
        test=np.where(test_mask)[0],
    )


def site_shift_split(
    labels: np.ndarray,
    site: np.ndarray,
    train_site: str = "3",
    test_site: str = "5",
) -> EvalSplit:
    """Split by DASAR site.

    Parameters
    ----------
    labels     : (N,) int
    site       : (N,) site-id strings ('3' or '5')
    train_site : site on which to train
    test_site  : held-out site for evaluation
    """
    train_mask = np.asarray(site) == train_site
    test_mask  = np.asarray(site) == test_site

    if not train_mask.any():
        raise ValueError(
            f"No samples for train_site={train_site!r}. "
            f"Available: {sorted(set(np.asarray(site).tolist()))}"
        )
    if not test_mask.any():
        raise ValueError(
            f"No samples for test_site={test_site!r}. "
            f"Available: {sorted(set(np.asarray(site).tolist()))}"
        )

    return EvalSplit(
        name="site",
        train=np.where(train_mask)[0],
        test=np.where(test_mask)[0],
    )
