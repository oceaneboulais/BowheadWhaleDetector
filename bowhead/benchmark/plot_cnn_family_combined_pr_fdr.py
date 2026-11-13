"""Render ONE combined precision-recall / FDR-vs-miss-fraction figure overlaying
every CNN-family detector for which a reproducible, saved per-threshold curve
exists, each annotated with AP (average precision).

All rows are scored against the manually reviewed (post-review) labels from
the Oct-21 MATLAB export (``latent_embeddings_3d_eval_8to1_MATLAB_combined_102126.mat``):
  - Supervised CNN (scratch/cold-start) -- runs/scratch_cnn_relabel_100k_matched_combined102126.json
    ("reviewed" condition), full 199,825-image independent eval set. The
    warm-start (AE-init) variant is deliberately dropped from all plots going
    forward: its performance does not meaningfully differ from the
    scratch-initialized model, so only one custom CNN is retained for
    clarity.
  - ResNet-18 (ImageNet), EfficientNet-B0 (ImageNet) -- both frozen backbones,
    runs/pr_curves_image_backbones_combined102126.npz ("reviewed" condition),
    probe trained on the matched 100k training draw and scored once on the
    SAME full 199,825-image eval set as the Supervised CNN (see
    bowhead/benchmark/save_image_backbone_curves_relabel.py).
  - BirdNET/Perch 2.0 -- runs/birdnet_relabel_100k_matched_full199825.json
    ("reviewed" condition); frozen linear probe over Griffin-Lim-reconstructed
    audio embeddings, fit on the full 100k_matched training draw and scored on
    the SAME full 199,825-image eval set as the other three rows -- fully
    protocol-matched (see bowhead.eval.run_birdnet_relabel_curves_full).

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.benchmark.plot_cnn_family_combined_pr_fdr
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc

_REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = _REPO_ROOT / "paper" / "Figures"

# Okabe-Ito colorblind-safe palette (https://jfly.uni-koeln.de/color/).
_COLORS = {
    "Supervised CNN": "#0072B2",
    "ResNet-18 (ImageNet)": "#E69F00",
    "EfficientNet-B0 (ImageNet)": "#009E73",
    "BirdNET/Perch 2.0": "#CC79A7",
}
_LINESTYLES = {
    "Supervised CNN": "-",
    "ResNet-18 (ImageNet)": "--",
    "EfficientNet-B0 (ImageNet)": "-.",
    "BirdNET/Perch 2.0": ":",
}


def _pr_auc(precision: np.ndarray, recall: np.ndarray) -> float:
    order = np.argsort(recall)
    return float(auc(recall[order], precision[order]))


def _load_supervised_cnn() -> dict:
    path = _REPO_ROOT / "runs" / "scratch_cnn_relabel_100k_matched_combined102126.json"
    d = json.loads(path.read_text())
    m = d["reviewed"]
    precision = np.asarray(m["pr_precision"])
    recall = np.asarray(m["pr_recall"])
    return {"Supervised CNN": dict(precision=precision, recall=recall,
                                    ap=float(m["average_precision"]),
                                    pr_auc=_pr_auc(precision, recall),
                                    n=int(m["n"]))}


def _load_image_backbones() -> dict:
    npz_path = _REPO_ROOT / "runs" / "pr_curves_image_backbones_combined102126.npz"
    if not npz_path.exists():
        return {}
    d = np.load(npz_path, allow_pickle=True)
    out = {}
    for key, name in (
        ("resnet18", "ResNet-18 (ImageNet)"),
        ("efficientnet_b0", "EfficientNet-B0 (ImageNet)"),
    ):
        if f"{key}_reviewed_precision" not in d:
            continue
        precision = d[f"{key}_reviewed_precision"]
        recall = d[f"{key}_reviewed_recall"]
        out[name] = dict(precision=precision, recall=recall,
                          ap=float(d[f"{key}_reviewed_ap"]),
                          pr_auc=_pr_auc(precision, recall),
                          n=int(d[f"{key}_reviewed_n"]))
    return out


def _load_birdnet() -> dict:
    path = _REPO_ROOT / "runs" / "birdnet_relabel_100k_matched_full199825.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    m = d["reviewed"]
    precision = np.asarray(m["pr_precision"])
    recall = np.asarray(m["pr_recall"])
    return {"BirdNET/Perch 2.0": dict(precision=precision, recall=recall,
                                       ap=float(m["average_precision"]),
                                       pr_auc=_pr_auc(precision, recall),
                                       n=int(m["n"]))}


def main() -> None:
    models = {**_load_supervised_cnn(), **_load_image_backbones(), **_load_birdnet()}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for name, m in models.items():
        color = _COLORS[name]
        ls = _LINESTYLES[name]
        axes[0].plot(m["recall"], m["precision"], color=color, ls=ls, lw=2, label=name)
        axes[1].plot(1 - m["recall"], 1 - m["precision"], color=color, ls=ls, lw=2, label=name)

    tick_locs = np.arange(0.1, 1.01, 0.1)

    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xticks(tick_locs)
    axes[0].set_yticks(tick_locs)
    axes[0].set_title("Precision-Recall")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Miss fraction (1 \u2212 recall)")
    axes[1].set_ylabel("False discovery rate (1 \u2212 precision)")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xticks(tick_locs)
    axes[1].set_yticks(tick_locs)
    axes[1].set_title("FDR vs. Miss Fraction")
    axes[1].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.18),
               frameon=False, fontsize=9)
    fig.suptitle("CNN-family detectors, manually reviewed (post-review) labels\n"
                 "(all four models: full eval set, n=199,825, protocol-matched)")
    fig.tight_layout()

    out_png = FIG_DIR / "cnn_family_combined_pr_fdr.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_png}")
    for name, m in models.items():
        print(f"  {name}: AP={m['ap']:.4f}  PR-AUC={m['pr_auc']:.4f}  n={m['n']:,}")

    _plot_summary_bars(models)


def _plot_summary_bars(models: dict) -> None:
    """AP-only bar chart (ROC-AUC dropped project-wide in favor of AP/PR-AUC)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    supervised_cnn = "Supervised CNN"
    names = list(models.keys())
    aps = [models[n]["ap"] for n in names]
    # Okabe-Ito colorblind-safe palette: dark blue highlights the supervised
    # CNN against sky-blue for the frozen-backbone comparison models.
    colors = ["#0072B2" if n == supervised_cnn else "#56B4E9" for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(names))
    ax.barh(y, aps, color=colors)
    ax.set_yticks(y, names)
    ax.set_xlabel("Average Precision")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    for yi, v in zip(y, aps):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)

    fig.suptitle("Supervised CNN vs. other CNN-family detectors (reviewed labels)")
    fig.tight_layout()

    out_png = FIG_DIR / "cnn_family_summary_bars.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
