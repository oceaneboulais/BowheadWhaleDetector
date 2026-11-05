"""Compute AP / ROC-AUC / Precision@R=0.90 for the AE+kNN detector, matching
the summary-metric format used by Table~\\ref{tab:cnn_family_summary} (the
CNN-family comparison table), so the AE+kNN table can be reported in the
same N / AP / ROC-AUC / Precision@R=0.90 / Protocol format instead of the
miss-fraction/FDR/FP:TP format used in the original miss-vs-FDR analysis.

Usage:
    PYTHONPATH=. python -m bowhead.eval.compute_ae_knn_summary_metrics \\
        --train ~/Github/latent_embedding_updates/latent_embeddings_3d_train_MATLAB.mat \\
        --eval  ~/Downloads/latent_embeddings_3d_eval_8to1_MATLAB_combined_102126.mat \\
        --k 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bowhead.benchmark.run_ae_knn_curves import CONDITIONS, _load_eval
from bowhead.eval.ae_knn_baseline import AEKNNScorer, load_training_embeddings
from bowhead.eval.metrics import score_predictions


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--eval", required=True)
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--target-recall", type=float, default=0.90)
    p.add_argument("--out-json", default="runs/ae_knn_summary_metrics.json")
    args = p.parse_args()

    train = load_training_embeddings(args.train)
    ev = _load_eval(args.eval)

    results = {}
    for name, label_source, min_dur in CONDITIONS:
        scorer = AEKNNScorer(train, k=args.k, label_source=label_source, min_duration=None)
        y_score = scorer.score_embeddings(ev["latent"])
        y_true = ev["iscall"] if label_source == "cleaned" else ev["iscall_original"]

        mask = np.ones(len(y_true), dtype=bool)
        if min_dur is not None:
            mask &= ev["duration"] >= min_dur

        metrics = score_predictions(y_true[mask], y_score[mask])
        precision_at_r = metrics.precision_at_recall(args.target_recall)
        results[name] = {
            "n": metrics.n,
            "ap": metrics.average_precision,
            "roc_auc": metrics.roc_auc,
            f"precision_at_recall_{args.target_recall:.2f}": precision_at_r,
            "prevalence": metrics.prevalence,
        }
        print(f"{name:38s}  N={metrics.n:7,d}  AP={metrics.average_precision:.3f}  "
              f"ROC-AUC={metrics.roc_auc:.3f}  P@R={args.target_recall:.2f}={precision_at_r:.3f}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWritten {out_path}")


if __name__ == "__main__":
    main()
