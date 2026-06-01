"""Detection metrics — precision/recall, ROC-AUC, and prevalence handling.

Two design choices that keep the four pipelines comparable AND honest:

1. Every pipeline reduces to a continuous *call-probability* score in [0, 1];
   we sweep the threshold to trace full PR and ROC curves rather than reporting
   a single operating point. (The AE+kNN baseline's "score" is the fraction of
   the k nearest neighbors that are calls.)
2. Precision is prevalence-dependent. Training is balanced ~1:1, but the real
   stream is ~1:8 call:transient. We report average precision on a test set
   *resampled to realistic prevalence* so precision numbers reflect deployment.
   ROC-AUC is prevalence-invariant and reported as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class DetectionMetrics:
    roc_auc: float
    average_precision: float          # at the evaluated prevalence
    prevalence: float                 # positive fraction of the scored set
    n: int
    # Full curves (for plotting / picking operating points)
    pr_precision: np.ndarray
    pr_recall: np.ndarray
    pr_thresholds: np.ndarray
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray
    roc_thresholds: np.ndarray

    def scalar_dict(self) -> dict[str, float]:
        """Just the scalar metrics (drops the curve arrays)."""
        d = asdict(self)
        for k in ("pr_precision", "pr_recall", "pr_thresholds",
                  "roc_fpr", "roc_tpr", "roc_thresholds"):
            d.pop(k)
        return d

    def precision_at_recall(self, target_recall: float) -> float:
        """Highest precision achievable at >= target_recall (for reporting)."""
        mask = self.pr_recall >= target_recall
        return float(self.pr_precision[mask].max()) if mask.any() else float("nan")


def score_predictions(
    y_true: np.ndarray, y_score: np.ndarray
) -> DetectionMetrics:
    """Compute detection metrics from labels and call-probability scores."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Need both classes present to score detection metrics.")

    prec, rec, pr_thr = precision_recall_curve(y_true, y_score)
    fpr, tpr, roc_thr = roc_curve(y_true, y_score)
    return DetectionMetrics(
        roc_auc=float(roc_auc_score(y_true, y_score)),
        average_precision=float(average_precision_score(y_true, y_score)),
        prevalence=float(y_true.mean()),
        n=int(len(y_true)),
        pr_precision=prec,
        pr_recall=rec,
        pr_thresholds=pr_thr,
        roc_fpr=fpr,
        roc_tpr=tpr,
        roc_thresholds=roc_thr,
    )


def resample_to_prevalence(
    y_true: np.ndarray,
    target_prevalence: float = 1.0 / 9.0,  # ~1 call : 8 transients
    seed: int = 0,
) -> np.ndarray:
    """Return indices subsampling the majority class to hit a target prevalence.

    Keeps all positives, downsamples negatives (or vice versa) so that
    positives / total == target_prevalence. Use the returned indices to slice
    BOTH y_true and y_score before calling ``score_predictions`` for a
    deployment-realistic precision estimate.
    """
    y_true = np.asarray(y_true).astype(int)
    rng = np.random.default_rng(seed)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return np.arange(len(y_true))

    # n_pos / (n_pos + n_neg_kept) = target  ->  n_neg_kept = n_pos*(1-t)/t
    n_neg_target = int(round(len(pos) * (1 - target_prevalence) / target_prevalence))
    if n_neg_target <= len(neg):
        neg_keep = rng.choice(neg, size=n_neg_target, replace=False)
        idx = np.concatenate([pos, neg_keep])
    else:
        # Not enough negatives; downsample positives instead to hit target.
        n_pos_target = int(round(len(neg) * target_prevalence / (1 - target_prevalence)))
        pos_keep = rng.choice(pos, size=min(n_pos_target, len(pos)), replace=False)
        idx = np.concatenate([pos_keep, neg])
    rng.shuffle(idx)
    return idx
