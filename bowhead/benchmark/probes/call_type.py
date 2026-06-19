"""Multiclass call-type classification probe (Types 1–7).

Trains a multi-class logistic-regression probe on frozen embeddings to
distinguish the seven morphological bowhead call types. Only positives
(label=1, Types 1–7) are used; Type 0 (non-call) is excluded.

Task: given a spectrogram known to contain a bowhead call, identify which
of the seven morphological types it is.

Metrics
-------
* One-vs-rest ROC-AUC (macro-averaged)
* Top-1 accuracy
* Per-type precision / recall / F1
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
)


@dataclass
class CallTypeResult:
    backbone: str
    call_types: tuple[int, ...]       # e.g. (1, 2, 3, 4, 5, 6, 7)
    roc_auc_macro: float              # one-vs-rest macro average
    accuracy: float
    per_type_roc_auc: dict[int, float] = field(default_factory=dict)
    classification_report: str = ""   # sklearn text report

    def summary(self) -> str:
        lines = [
            f"call-type classification  [{self.backbone}]",
            f"  macro ROC-AUC : {self.roc_auc_macro:.3f}",
            f"  top-1 accuracy: {self.accuracy:.3f}",
            "  per-type ROC-AUC:",
        ]
        for t, auc in sorted(self.per_type_roc_auc.items()):
            lines.append(f"    Type {t}: {auc:.3f}")
        return "\n".join(lines)


class CallTypeProbe:
    """StandardScaler → balanced multiclass LogisticRegression."""

    def __init__(
        self,
        call_types: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
        C: float = 1.0,
        max_iter: int = 2000,
        seed: int = 0,
    ) -> None:
        self.call_types = call_types
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                max_iter=max_iter,
                class_weight="balanced",
                multi_class="multinomial",
                solver="lbfgs",
                random_state=seed,
            ),
        )

    def fit(
        self,
        X: np.ndarray,
        call_type_labels: np.ndarray,
    ) -> "CallTypeProbe":
        """Fit on embeddings of positive samples only.

        Parameters
        ----------
        X : (N, D) embeddings of the positive (call) samples
        call_type_labels : (N,) int array — values in self.call_types
        """
        self.clf.fit(X, call_type_labels)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(N, n_types) class probabilities, ordered by self.call_types."""
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)


def evaluate_call_type(
    emb_train: np.ndarray,
    y_type_train: np.ndarray,
    emb_test: np.ndarray,
    y_type_test: np.ndarray,
    backbone_name: str,
    *,
    call_types: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    C: float = 1.0,
    seed: int = 0,
) -> CallTypeResult:
    """Fit and evaluate a call-type classification probe.

    Parameters
    ----------
    emb_train / emb_test : (N, D) embeddings of POSITIVE samples only
    y_type_train / y_type_test : (N,) int call-type labels (1–7)
    """
    probe = CallTypeProbe(call_types=call_types, C=C, seed=seed)
    probe.fit(emb_train, y_type_train)

    y_pred  = probe.predict(emb_test)
    y_proba = probe.predict_proba(emb_test)
    acc     = float(accuracy_score(y_type_test, y_pred))

    # One-vs-rest ROC-AUC per type, then macro-average
    classes = probe.clf[-1].classes_
    per_type: dict[int, float] = {}
    for i, c in enumerate(classes):
        try:
            per_type[int(c)] = float(roc_auc_score(
                (y_type_test == c).astype(int), y_proba[:, i]
            ))
        except ValueError:
            per_type[int(c)] = float("nan")

    macro_auc = float(np.nanmean(list(per_type.values())))
    report = classification_report(
        y_type_test, y_pred,
        labels=list(classes),
        target_names=[f"Type{c}" for c in classes],
        zero_division=0,
    )
    return CallTypeResult(
        backbone=backbone_name,
        call_types=tuple(int(c) for c in classes),
        roc_auc_macro=macro_auc,
        accuracy=acc,
        per_type_roc_auc=per_type,
        classification_report=report,
    )
