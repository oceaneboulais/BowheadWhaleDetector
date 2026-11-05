"""Render a paper figure comparing cold-start (scratch) vs. warm-start CNN
detection performance, both scored on the SAME, most recent protocol: trained
on `data/spectrograms_100k_matched.npz` and evaluated on the full
199,825-spectrogram independent eval set using the most recently provided
(2026-10-21) reviewed-label export (`..._combined_102126.mat`).

Reads runs/scratch_cnn_relabel_100k_matched_combined102126.json and
runs/warmstart_cnn_relabel_100k_matched_combined102126.json (both produced by
bowhead.eval.run_scratch_relabel_curves against the same eval-mat/eval-dir),
using only each file's "reviewed" label condition, and writes a two-panel
precision-recall / FDR-vs-miss-fraction PNG to paper/Figures/, in the same
visual style as Figures/cnn_relabel_pr_fdr.png.

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.benchmark.plot_cnn_coldstart_vs_warmstart_combined102126
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_JSON = _REPO_ROOT / "runs" / "scratch_cnn_relabel_100k_matched_combined102126.json"
WARMSTART_JSON = _REPO_ROOT / "runs" / "warmstart_cnn_relabel_100k_matched_combined102126.json"
OUT_PNG = _REPO_ROOT / "paper" / "Figures" / "cnn_coldstart_vs_warmstart_combined102126_pr_fdr.png"

COLORS = {"scratch": "#D55E00", "warmstart": "#0072B2"}
LABELS = {
    "scratch": "Scratch CNN (cold start)",
    "warmstart": "Warm-start CNN (AE-init)",
}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    results = {
        "scratch": json.loads(SCRATCH_JSON.read_text())["reviewed"],
        "warmstart": json.loads(WARMSTART_JSON.read_text())["reviewed"],
    }
    n = results["scratch"]["n"]
    assert n == results["warmstart"]["n"], "scratch/warmstart eval set sizes must match"
    assert results["scratch"]["prevalence"] == results["warmstart"]["prevalence"], (
        "scratch/warmstart eval prevalence must match (same eval-mat/label set)"
    )
    target_miss = 0.10
    target_recall = 1.0 - target_miss

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for name in ("scratch", "warmstart"):
        m = results[name]
        prec = np.asarray(m["pr_precision"])
        rec = np.asarray(m["pr_recall"])
        color = COLORS[name]
        label = LABELS[name]
        axes[0].plot(rec, prec, color=color, lw=2, label=label)
        axes[1].plot(1 - rec, 1 - prec, color=color, lw=2, label=label)

        # Mark the matched miss-fraction=10% operating point (same convention
        # as the scratch-CNN relabeling figure).
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
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)

    axes[1].set_xlabel("Miss fraction (1 \u2212 recall)")
    axes[1].set_ylabel("False discovery rate (1 \u2212 precision)")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("FDR vs. Miss Fraction")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)

    fig.suptitle(
        f"Cold-start vs. warm-start CNN (reviewed labels, 10/21/2026 update)\n"
        f"n={n:,}; \u25c7 = matched miss-fraction={target_miss:.0%} operating point",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"Saved figure -> {OUT_PNG}")


if __name__ == "__main__":
    main()
