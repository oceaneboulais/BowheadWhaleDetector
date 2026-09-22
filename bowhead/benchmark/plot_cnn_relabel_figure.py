"""Render the paper Figure for the scratch-CNN relabeling ablation
(Sec. "Effect of dataset relabeling on CNN detection performance"): the SAME
frozen scratch-CNN checkpoint, scored once on the 199,825-spectrogram
independent evaluation set, graded against the pre-review (`type_org`) vs.
manually-reviewed (`iscall`) label sets.

Reads runs/scratch_cnn_relabel_100k_matched.json (produced by
bowhead.eval.run_scratch_relabel_curves) and writes a two-panel
precision-recall / FDR-vs-miss-fraction PNG to paper/Figures/, in the same
visual style as Figures/fdr_roc_external.png and Figures/ae_knn_miss_vs_fdr.png.

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.benchmark.plot_cnn_relabel_figure
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_JSON = _REPO_ROOT / "runs" / "scratch_cnn_relabel_100k_matched.json"
OUT_PNG = _REPO_ROOT / "paper" / "Figures" / "cnn_relabel_pr_fdr.png"

COLORS = {"reviewed": "#4e9af1", "original": "#f07040"}
LABELS = {
    "reviewed": "Reviewed labels (post-review)",
    "original": "Original labels (pre-review)",
}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads(RESULTS_JSON.read_text())
    n = results["reviewed"]["n"]
    target_miss = 0.10
    target_recall = 1.0 - target_miss

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for name in ("reviewed", "original"):
        m = results[name]
        prec = np.asarray(m["pr_precision"])
        rec = np.asarray(m["pr_recall"])
        color = COLORS[name]
        label = f"{LABELS[name]} (AP={m['average_precision']:.3f}, ROC-AUC={m['roc_auc']:.3f})"
        axes[0].plot(rec, prec, color=color, lw=2, label=label)
        axes[1].plot(1 - rec, 1 - prec, color=color, lw=2, label=label)

        # Mark the matched miss-fraction=10% operating point (same convention
        # as Table IV: highest precision at recall >= target_recall).
        mask = rec >= target_recall
        if mask.any():
            j = int(np.where(mask)[0][np.argmax(prec[mask])])
            axes[0].scatter([rec[j]], [prec[j]], marker="D", s=70, facecolor="none",
                             edgecolor=color, linewidth=1.8, zorder=4)
            axes[1].scatter([1 - rec[j]], [1 - prec[j]], marker="D", s=70, facecolor="none",
                             edgecolor=color, linewidth=1.8, zorder=4)

    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Precision-Recall")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")

    axes[1].set_xlabel("Miss fraction (1 \u2212 recall)")
    axes[1].set_ylabel("False discovery rate (1 \u2212 precision)")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("FDR vs. Miss Fraction")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Scratch CNN (cold start), before vs. after dataset relabeling "
                 f"(n={n:,}, same frozen predictions; \u25c7 = matched miss-fraction="
                 f"{target_miss:.0%} operating point)")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Saved figure -> {OUT_PNG}")


if __name__ == "__main__":
    main()
