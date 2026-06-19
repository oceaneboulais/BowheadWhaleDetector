"""Open-set rejection evaluation.

Tests whether a detector trained on a *known* subset of call types can:
  (a) still detect held-out call types (open-set recall), and
  (b) assign lower confidence to entirely unseen call classes.

This is relevant for bowhead whale monitoring because:
  - Types 1-7 are not equally common or equally easy to detect.
  - A practical detector must handle call types it has not been explicitly
    trained on (new vocalisations, geographic variants).

Protocol
--------
1. Split the call types into a *known* set (used during probe training) and
   a *novel* set (withheld from training; treated as positive detections at
   eval time).
2. Train a logistic probe on known-type calls + all non-calls.
3. Evaluate on novel-type calls + non-calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class OpenSetResult:
    """Results of one open-set rejection experiment."""
    known_types: tuple[int, ...]
    novel_types: tuple[int, ...]
    ap: float           # AP on novel calls + all non-calls
    roc_auc: float
    n_train: int
    n_test_novel: int
    n_test_noncall: int


def open_set_rejection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    call_types: np.ndarray,
    known_types: tuple[int, ...] = (1, 2, 3, 4),
    novel_types: tuple[int, ...] = (5, 6, 7),
    seed: int = 0,
) -> OpenSetResult:
    """Evaluate open-set rejection.

    Parameters
    ----------
    embeddings  : (N, D) float
    labels      : (N,) int  0=non-call 1=call
    call_types  : (N,) int  0=non-call 1-7=call type
    known_types : call types available during probe training
    novel_types : call types withheld from training
    seed        : random seed
    """
    # Training set: known-type calls + all non-calls
    train_mask = (labels == 0) | np.isin(call_types, list(known_types))
    X_train = embeddings[train_mask]
    y_train = labels[train_mask]

    # Test set: novel-type calls + all non-calls
    test_mask  = (labels == 0) | np.isin(call_types, list(novel_types))
    X_test = embeddings[test_mask]
    y_test = labels[test_mask]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(
            "Both classes must be present in train and test sets."
        )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    scores = clf.predict_proba(X_test_s)[:, 1]

    ap  = float(average_precision_score(y_test, scores))
    roc = float(roc_auc_score(y_test, scores))

    n_novel    = int(np.isin(call_types[test_mask], list(novel_types)).sum())
    n_noncall  = int((y_test == 0).sum())

    return OpenSetResult(
        known_types=known_types,
        novel_types=novel_types,
        ap=ap, roc_auc=roc,
        n_train=int(train_mask.sum()),
        n_test_novel=n_novel,
        n_test_noncall=n_noncall,
    )
