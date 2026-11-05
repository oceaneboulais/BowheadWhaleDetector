"""Plot ROC and PR curves for the frozen scratch ("cold start") CNN scored on
the FULL 2.1M-call master database (``runs/scratch_cnn_master_2.1M_probs.npz``),
as produced by ``bowhead.eval.score_master_inventory``.

Labels here are the raw inventory labels (0=auto-detected transient,
1=manually-annotated call), NOT the manually reviewed post-review labels used
for the paper's 199,825-image eval set -- so these numbers are not directly
comparable to ``tab:cnn_family_summary`` / ``fig:cnn_family_summary_bars``.

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.benchmark.plot_scratch_cnn_master_2_1M
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

_REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = _REPO_ROOT / "paper" / "Figures"
RESULTS_JSON = _REPO_ROOT / "runs" / "scratch_cnn_master_2.1M.json"
PROBS_NPZ = _REPO_ROOT / "runs" / "scratch_cnn_master_2.1M_probs.npz"
# Same frozen scratch-CNN checkpoint, scored on the independent 199,825-image
# eval set (reviewed labels) -- overlaid for relative comparison.
EVAL_JSON = _REPO_ROOT / "runs" / "scratch_cnn_relabel_100k_matched_combined102126.json"
_COLOR = "#0072B2"  # Okabe-Ito colorblind-safe dark blue
_EVAL_COLOR = "#D55E00"  # Okabe-Ito colorblind-safe vermillion


def main() -> None:
    results = json.loads(RESULTS_JSON.read_text())
    d = np.load(PROBS_NPZ, allow_pickle=True)
    labels, probs = d["labels"], d["probs"]
    if "ok" in d:
        ok = d["ok"]
        labels, probs = labels[ok], probs[ok]

    precision = np.asarray(results["pr_precision"])
    recall = np.asarray(results["pr_recall"])
    fpr, tpr, _ = roc_curve(labels, probs)

    eval_results = json.loads(EVAL_JSON.read_text())["reviewed"]
    eval_precision = np.asarray(eval_results["pr_precision"])
    eval_recall = np.asarray(eval_results["pr_recall"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    miss_fraction = 1.0 - recall
    fdr = 1.0 - precision
    eval_miss_fraction = 1.0 - eval_recall
    eval_fdr = 1.0 - eval_precision

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(recall, precision, color=_COLOR, lw=2,
                 label=f"Master DB (AP={results['average_precision']:.3f})")
    axes[0].plot(eval_recall, eval_precision, color=_EVAL_COLOR, lw=2, ls="--",
                 label=f"Independent eval set, n={eval_results['n']:,} "
                       f"(AP={eval_results['average_precision']:.3f})")
    axes[0].axhline(results["prevalence"], color="gray", ls=":", lw=1,
                     label=f"Chance (prevalence={results['prevalence']:.3f})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Precision-Recall")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    axes[0].text(
        0.5, -0.30,
        "Precision $= \\dfrac{TP}{TP + FP}$: fraction of detections that are true calls\n"
        "Recall $= \\dfrac{TP}{TP + FN}$: fraction of true calls that are detected",
        transform=axes[0].transAxes, fontsize=8, va="top", ha="center", clip_on=False,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    axes[1].plot(fpr, tpr, color=_COLOR, lw=2,
                 label=f"Master DB (ROC-AUC={results['roc_auc']:.3f})")
    # The eval-set run only persisted the scalar ROC-AUC, not the raw
    # fpr/tpr arrays, so it can't be overlaid as a curve here.
    axes[1].plot([], [], color=_EVAL_COLOR, lw=2, ls="--",
                 label=f"Independent eval set (ROC-AUC={eval_results['roc_auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], color="gray", ls=":", lw=1, label="Chance")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Receiver Operating Characteristic (ROC)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right", frameon=False, fontsize=9)
    axes[1].text(
        0.5, -0.30,
        "TPR (recall) $= \\dfrac{TP}{TP + FN}$: fraction of true calls detected\n"
        "FPR $= \\dfrac{FP}{FP + TN}$: fraction of non-calls wrongly flagged",
        transform=axes[1].transAxes, fontsize=8, va="top", ha="center", clip_on=False,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    axes[2].plot(miss_fraction, fdr, color=_COLOR, lw=2,
                 label=f"Master DB (AP={results['average_precision']:.3f})")
    axes[2].plot(eval_miss_fraction, eval_fdr, color=_EVAL_COLOR, lw=2, ls="--",
                 label=f"Independent eval set, n={eval_results['n']:,} "
                       f"(AP={eval_results['average_precision']:.3f})")
    axes[2].set_xlabel("Miss fraction (1 \u2212 recall)")
    axes[2].set_ylabel("False discovery rate (1 \u2212 precision)")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("FDR vs. Miss Fraction")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", frameon=False, fontsize=9)
    axes[2].text(
        0.5, -0.30,
        "Miss fraction $= \\dfrac{FN}{TP + FN}$: fraction of true calls missed\n"
        "FDR $= \\dfrac{FP}{TP + FP}$: fraction of detections that are false alarms",
        transform=axes[2].transAxes, fontsize=8, va="top", ha="center", clip_on=False,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9),
    )

    fig.suptitle(
        f"Supervised CNN scored on the full master database "
        f"(raw inventory labels, n={results['n']:,}, prevalence={results['prevalence']:.3f})\n"
        f"{results['_meta'].get('n_excluded_unreadable', 0):,} unreadable files excluded"
    )
    fig.subplots_adjust(left=0.06, right=0.98, wspace=0.3, top=0.80, bottom=0.40)

    out_png = FIG_DIR / "scratch_cnn_master_2.1M_roc_pr.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
