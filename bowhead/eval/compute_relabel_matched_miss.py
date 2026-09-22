"""Recompute the scratch-CNN relabeling ablation at a MATCHED operating point
(miss fraction = 10%, i.e. recall = 0.90) for both label sets, instead of the
previous unmatched ROC-AUC/AP/precision-at-recall-0.70 comparison.

Reads the SAME frozen scratch-CNN predictions already saved in
runs/scratch_cnn_relabel_100k_matched.json (bowhead.eval.run_scratch_relabel_curves)
-- no rescoring, just re-reading the saved precision/recall curves at a fixed
recall level for each label set -- so the two rows are directly comparable
(same recall, i.e. same fraction of true calls found) instead of comparing at
whatever recall each label set's PR curve happens to reach 0.70 precision.

Usage:
    PYTHONPATH=. python -m bowhead.eval.compute_relabel_matched_miss \\
        --in-json runs/scratch_cnn_relabel_100k_matched.json \\
        --out-json runs/scratch_cnn_relabel_matched_miss10.json \\
        --target-miss 0.10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN_JSON = _REPO_ROOT / "runs" / "scratch_cnn_relabel_100k_matched.json"
DEFAULT_OUT_JSON = _REPO_ROOT / "runs" / "scratch_cnn_relabel_matched_miss10.json"

LABEL_NAMES = {"original": "Original (pre-review)", "reviewed": "Reviewed (post-review)"}


def _precision_at_recall(rec: np.ndarray, prec: np.ndarray, target_recall: float) -> float:
    """Highest precision achievable at >= target_recall (same convention as
    bowhead.eval.metrics.DetectionMetrics.precision_at_recall, used elsewhere
    in this project for the "Precision @ Recall=0.70" columns)."""
    mask = rec >= target_recall
    return float(prec[mask].max()) if mask.any() else float("nan")


def run(in_json: Path, out_json: Path, target_miss: float = 0.10) -> dict:
    results = json.loads(Path(in_json).read_text())
    target_recall = 1.0 - target_miss

    out: dict = {"target_miss_fraction": target_miss, "target_recall": target_recall}
    print(f"Matched operating point: miss fraction = {target_miss:.1%} "
          f"(recall = {target_recall:.2f})\n")
    print(f"  {'Label set':24s} {'prevalence':>10s} {'n':>8s} {'precision':>10s} {'FDR':>7s}")
    for key in ("original", "reviewed"):
        m = results[key]
        rec = np.asarray(m["pr_recall"])
        prec = np.asarray(m["pr_precision"])
        p = _precision_at_recall(rec, prec, target_recall)
        fdr = 1.0 - p
        out[key] = {
            "label": LABEL_NAMES[key],
            "prevalence": m["prevalence"],
            "n": m["n"],
            "precision_at_target_recall": p,
            "fdr": fdr,
            "roc_auc": m["roc_auc"],
        }
        print(f"  {LABEL_NAMES[key]:24s} {m['prevalence']:10.4f} {m['n']:8,d} "
              f"{p:10.4f} {fdr:7.4f}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nWritten {out_json}")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-json", type=Path, default=DEFAULT_IN_JSON)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--target-miss", type=float, default=0.10)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    run(a.in_json, a.out_json, a.target_miss)
