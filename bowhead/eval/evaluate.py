"""Unified evaluation across all four pipelines.

A ``Scorer`` is anything that maps spectrogram images -> call-probability in
[0, 1]. The custom CNN, the AE+kNN baseline, and the BirdNET/Perch linear probes
all implement this one interface, so they are scored identically and plotted on
the same axes.
"""

from __future__ import annotations

from typing import Protocol, Iterable

import numpy as np

from bowhead.eval.metrics import (
    DetectionMetrics,
    score_predictions,
    resample_to_prevalence,
)


class Scorer(Protocol):
    """Maps a batch of images (N, 1, H, W) to call-probabilities (N,) in [0,1]."""

    name: str

    def score(self, images: np.ndarray) -> np.ndarray: ...


def evaluate_scorer(
    scorer: Scorer,
    images: np.ndarray,
    y_true: np.ndarray,
    *,
    target_prevalence: float | None = 1.0 / 9.0,
    seed: int = 0,
) -> DetectionMetrics:
    """Score one pipeline on a (held-out) set, optionally at realistic prevalence.

    If ``target_prevalence`` is given, negatives are subsampled to that ratio
    BEFORE scoring so average-precision reflects deployment. Pass ``None`` to
    score on the set as-is.
    """
    y_true = np.asarray(y_true).astype(int)
    if target_prevalence is not None:
        idx = resample_to_prevalence(y_true, target_prevalence, seed=seed)
        images, y_true = images[idx], y_true[idx]
    y_score = np.asarray(scorer.score(images), dtype=float)
    return score_predictions(y_true, y_score)


def compare_pipelines(
    scorers: Iterable[Scorer],
    images: np.ndarray,
    y_true: np.ndarray,
    *,
    target_prevalence: float | None = 1.0 / 9.0,
    seed: int = 0,
) -> dict[str, DetectionMetrics]:
    """Evaluate several pipelines on the SAME resampled test set.

    Resampling is done once (same seed) so every pipeline sees an identical set
    of images — the fair-comparison guarantee.
    """
    y_true = np.asarray(y_true).astype(int)
    if target_prevalence is not None:
        idx = resample_to_prevalence(y_true, target_prevalence, seed=seed)
        images, y_true = images[idx], y_true[idx]

    results: dict[str, DetectionMetrics] = {}
    for scorer in scorers:
        y_score = np.asarray(scorer.score(images), dtype=float)
        results[scorer.name] = score_predictions(y_true, y_score)
    return results


def metrics_table(results: dict[str, DetectionMetrics]) -> str:
    """Render a compact comparison table (ROC-AUC, AP, precision@0.7 recall)."""
    rows = ["pipeline                     ROC-AUC      AP   P@R0.70",
            "-" * 56]
    for name, m in sorted(results.items(), key=lambda kv: -kv[1].roc_auc):
        rows.append(
            f"{name:26s}  {m.roc_auc:6.3f}  {m.average_precision:6.3f}   "
            f"{m.precision_at_recall(0.70):6.3f}"
        )
    return "\n".join(rows)
