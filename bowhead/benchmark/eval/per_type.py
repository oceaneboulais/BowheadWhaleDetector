"""Per-call-type precision-recall and ROC curves.

Breaks down detection performance by call type (Types 1-7) so that the
benchmark can reveal which vocalisations are hardest to detect under each
backbone / probe combination.

Each call type is evaluated in a one-vs-rest fashion:
  positive = current call type
  negative = all non-calls (Type 0)
The evaluation is restricted to the test split so there is no data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass
class CallTypeCurves:
    """PR and ROC curves for a single call type."""
    call_type: int
    n_positive: int
    n_negative: int
    ap: float
    roc_auc: float
    precision: np.ndarray
    recall: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray


@dataclass
class PerTypeResult:
    """Collection of per-type curves for one backbone."""
    curves: list[CallTypeCurves] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["call_type  n_pos  n_neg    AP   ROC-AUC"]
        for c in sorted(self.curves, key=lambda x: x.call_type):
            lines.append(
                f"  Type {c.call_type}   {c.n_positive:>5,}  "
                f"{c.n_negative:>5,}  {c.ap:.4f}  {c.roc_auc:.4f}"
            )
        return "\n".join(lines)


def per_type_curves(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    train_call_types: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_call_types: np.ndarray,
    call_types: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    seed: int = 0,
) -> PerTypeResult:
    """Fit one logistic probe per call type and return per-type PR + ROC curves.

    Parameters
    ----------
    train_embeddings : (N_train, D) float
    train_labels     : (N_train,) int  0=non-call 1=call
    train_call_types : (N_train,) int  0=non-call 1-7=call type
    test_embeddings  : (N_test, D) float
    test_labels      : (N_test,) int
    test_call_types  : (N_test,) int
    call_types       : which call types to evaluate
    seed             : random seed
    """
    result = PerTypeResult()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(train_embeddings)
    X_test_s  = scaler.transform(test_embeddings)

    for ct in call_types:
        # Train: this call type vs all non-calls
        train_mask = (train_labels == 0) | (train_call_types == ct)
        y_train = np.where(train_call_types[train_mask] == ct, 1, 0)
        X_tr = X_train_s[train_mask]

        # Test: same filter
        test_mask = (test_labels == 0) | (test_call_types == ct)
        y_test = np.where(test_call_types[test_mask] == ct, 1, 0)
        X_te  = X_test_s[test_mask]

        if y_train.sum() == 0 or y_test.sum() == 0:
            continue   # no positive examples for this type

        clf = LogisticRegression(
            max_iter=1000, random_state=seed, class_weight="balanced"
        )
        clf.fit(X_tr, y_train)
        scores = clf.predict_proba(X_te)[:, 1]

        prec, rec, _  = precision_recall_curve(y_test, scores)
        fpr, tpr, _   = roc_curve(y_test, scores)
        ap  = float(average_precision_score(y_test, scores))
        auc = float(roc_auc_score(y_test, scores))

        result.curves.append(CallTypeCurves(
            call_type=ct,
            n_positive=int(y_test.sum()),
            n_negative=int((y_test == 0).sum()),
            ap=ap, roc_auc=auc,
            precision=prec, recall=rec,
            fpr=fpr, tpr=tpr,
        ))

    return result
