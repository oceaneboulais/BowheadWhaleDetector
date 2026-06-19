"""Evaluation splits and metrics for the frozen-embedding benchmark.

Provides:
  splits      — temporal-drift and site-shift index arrays
  open_set    — open-set rejection evaluation (known vs unknown call types)
  per_type    — per-call-type PR and ROC curves
"""

from bowhead.benchmark.eval.splits import (
    EvalSplit,
    temporal_drift_split,
    site_shift_split,
)
from bowhead.benchmark.eval.open_set import open_set_rejection
from bowhead.benchmark.eval.per_type import per_type_curves

__all__ = [
    "EvalSplit",
    "temporal_drift_split",
    "site_shift_split",
    "open_set_rejection",
    "per_type_curves",
]
