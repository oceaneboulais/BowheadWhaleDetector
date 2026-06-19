"""Binary call-detection probe.

Trains a logistic-regression linear probe on frozen embeddings for the binary
call / non-call task. Mirrors the Burns et al. 2025 (Perch 2.0 whale study)
protocol: few-shot sweeps (k ∈ {4, 8, 16, 32}, 5 repeats, ROC-AUC) plus a
full-train upper-bound.

All labels: 1 = bowhead call (Types 1–7), 0 = non-call transient (Type 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from bowhead.eval.metrics import DetectionMetrics, score_predictions, resample_to_prevalence


# ── Probe ─────────────────────────────────────────────────────────────────────

class DetectionProbe:
    """StandardScaler → balanced LogisticRegression binary probe."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 0) -> None:
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C, max_iter=max_iter,
                class_weight="balanced",
                random_state=seed,
            ),
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DetectionProbe":
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Call-probability P(call) for each sample, shape (N,)."""
        return self.clf.predict_proba(X)[:, 1]

    def score(self, X: np.ndarray, y: np.ndarray) -> DetectionMetrics:
        """Fit + score in one call; returns full DetectionMetrics."""
        probs = self.predict_proba(X)
        return score_predictions(np.asarray(y).astype(int), probs)


# ── Few-shot evaluation ───────────────────────────────────────────────────────

@dataclass
class DetectionFewShotResult:
    backbone: str
    k_values: tuple[int, ...]
    roc_auc_mean:  dict[int, float] = field(default_factory=dict)
    roc_auc_std:   dict[int, float] = field(default_factory=dict)
    ap_mean:       dict[int, float] = field(default_factory=dict)
    ap_std:        dict[int, float] = field(default_factory=dict)

    def table(self) -> str:
        rows = [f"few-shot detection  [{self.backbone}]",
                "  k    ROC-AUC (±std)   AP (±std)"]
        for k in self.k_values:
            rows.append(
                f"  {k:<4d} "
                f"{self.roc_auc_mean[k]:.3f} ±{self.roc_auc_std[k]:.3f}   "
                f"{self.ap_mean[k]:.3f} ±{self.ap_std[k]:.3f}"
            )
        return "\n".join(rows)


def few_shot_detection(
    emb_train: np.ndarray,
    y_train: np.ndarray,
    emb_test: np.ndarray,
    y_test: np.ndarray,
    backbone_name: str,
    *,
    k_values: tuple[int, ...] = (4, 8, 16, 32),
    repeats: int = 5,
    C: float = 1.0,
    seed: int = 0,
) -> DetectionFewShotResult:
    """Few-shot linear detection probing.

    For each k, draw k positive + k negative examples from the train set,
    fit a DetectionProbe, and evaluate ROC-AUC + AP on the fixed test set.
    Repeats independently ``repeats`` times.
    """
    y_train = np.asarray(y_train).astype(int)
    y_test  = np.asarray(y_test).astype(int)
    pos = np.where(y_train == 1)[0]
    neg = np.where(y_train == 0)[0]
    res = DetectionFewShotResult(backbone=backbone_name, k_values=tuple(k_values))

    for k in k_values:
        aucs, aps = [], []
        for r in range(repeats):
            rng = np.random.default_rng(seed + r)
            if len(pos) < k or len(neg) < k:
                raise ValueError(
                    f"Not enough examples for k={k} "
                    f"(pos={len(pos)}, neg={len(neg)})."
                )
            idx = np.concatenate([
                rng.choice(pos, k, replace=False),
                rng.choice(neg, k, replace=False),
            ])
            probe = DetectionProbe(C=C, seed=seed + r).fit(emb_train[idx], y_train[idx])
            probs = probe.predict_proba(emb_test)
            aucs.append(float(roc_auc_score(y_test, probs)))
            aps.append(float(average_precision_score(y_test, probs)))
        res.roc_auc_mean[k] = float(np.mean(aucs))
        res.roc_auc_std[k]  = float(np.std(aucs))
        res.ap_mean[k]      = float(np.mean(aps))
        res.ap_std[k]       = float(np.std(aps))

    return res


def full_train_detection(
    emb_train: np.ndarray,
    y_train: np.ndarray,
    emb_test: np.ndarray,
    y_test: np.ndarray,
    backbone_name: str,
    *,
    C: float = 1.0,
    target_prevalence: float | None = 1.0 / 9.0,
    seed: int = 0,
) -> DetectionMetrics:
    """Full-train detection probe — upper-bound result.

    Optionally resamples the test set to realistic prevalence (1:8 ratio)
    before computing metrics.
    """
    probe = DetectionProbe(C=C, seed=seed).fit(emb_train, y_train)
    y_test = np.asarray(y_test).astype(int)
    probs  = probe.predict_proba(emb_test)
    if target_prevalence is not None:
        idx = resample_to_prevalence(y_test, target_prevalence, seed=seed)
        y_test, probs = y_test[idx], probs[idx]
    return score_predictions(y_test, probs)
