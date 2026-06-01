"""Linear probing + few-shot evaluation over frozen embeddings.

Mirrors Burns et al. 2025: train a logistic-regression probe on k embeddings per
class, sweep k in {4,8,16,32}, repeat 5x with different random draws, report
ROC-AUC mean/std. Also exposes a full-train ``LinearProbe`` and a ``ProbeScorer``
so a fitted probe plugs into the shared ``bowhead.eval`` comparison harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


class LinearProbe:
    """Standardize -> logistic regression on frozen embeddings."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 0) -> None:
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced",
                               random_state=seed),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]


@dataclass
class FewShotResult:
    k_values: tuple[int, ...]
    roc_auc_mean: dict[int, float] = field(default_factory=dict)
    roc_auc_std: dict[int, float] = field(default_factory=dict)

    def table(self, label: str = "") -> str:
        rows = [f"few-shot ROC-AUC  {label}".rstrip(), "  k    mean    std"]
        for k in self.k_values:
            rows.append(f"  {k:<4d} {self.roc_auc_mean[k]:.3f}  ±{self.roc_auc_std[k]:.3f}")
        return "\n".join(rows)


def few_shot_eval(
    emb_train: np.ndarray,
    y_train: np.ndarray,
    emb_test: np.ndarray,
    y_test: np.ndarray,
    *,
    k_values: tuple[int, ...] = (4, 8, 16, 32),
    repeats: int = 5,
    seed: int = 0,
) -> FewShotResult:
    """Few-shot linear probing: for each k, draw k examples/class, fit, score.

    The test set is fixed; only the (small) training draw varies across repeats.
    """
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)
    pos = np.where(y_train == 1)[0]
    neg = np.where(y_train == 0)[0]
    res = FewShotResult(k_values=tuple(k_values))

    for k in k_values:
        aucs = []
        for r in range(repeats):
            rng = np.random.default_rng(seed + r)
            if len(pos) < k or len(neg) < k:
                raise ValueError(f"Not enough examples for k={k} "
                                 f"(pos={len(pos)}, neg={len(neg)}).")
            idx = np.concatenate([
                rng.choice(pos, k, replace=False),
                rng.choice(neg, k, replace=False),
            ])
            probe = LinearProbe(seed=seed + r).fit(emb_train[idx], y_train[idx])
            aucs.append(roc_auc_score(y_test, probe.predict_proba(emb_test)))
        res.roc_auc_mean[k] = float(np.mean(aucs))
        res.roc_auc_std[k] = float(np.std(aucs))
    return res


class ProbeScorer:
    """Wrap (embedder + fitted probe) as a Scorer over raw waveforms.

    ``embedder`` is any object with ``.embed(waveforms) -> (N, D)`` and a
    ``.name``; ``probe`` is a fitted ``LinearProbe``. Plugs into
    ``bowhead.eval.compare_pipelines`` alongside the CNN and AE+kNN scorers.

    NOTE: unlike the CNN/AE scorers (which take spectrogram images), this scores
    raw WAVEFORMS — keep the test waveforms aligned with the test images by index.
    """

    def __init__(self, embedder, probe: LinearProbe, name: str | None = None) -> None:
        self.embedder = embedder
        self.probe = probe
        self.name = name or getattr(embedder, "name", "probe")

    def score(self, waveforms: np.ndarray) -> np.ndarray:
        return self.probe.predict_proba(self.embedder.embed(waveforms))
