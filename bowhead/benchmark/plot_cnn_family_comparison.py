"""Render SEPARATE per-model figures (one PNG each) comparing the cold-start
(scratch) CNN against the other CNN-family detectors evaluated in this
project, plus a bar-chart summary across all of them.

All metrics/curves below use the REVIEWED (post-review, `iscall`) label set
of Sec. DatasetEditing, so every model in this comparison is now scored
against the same manually-reviewed labels (protocol/N still differs by
model -- see notes below and the Protocol column of the paper table).

Data sources:
  - Scratch CNN (cold start): runs/scratch_cnn_relabel_100k_matched.json
    ["reviewed"] -- full 199,825-spectrogram independent eval set.
  - Warm-start CNN: runs/warmstart_cnn_relabel_100k_matched_combined102126.json
    ["reviewed"] -- full 199,825-spectrogram independent eval set.
  - BirdNET/Perch 2.0: runs/birdnet_relabel_100k_matched_full199825.json
    ["reviewed"] -- full 199,825-spectrogram independent eval set (Griffin-Lim
    reconstruction + frozen linear probe fit on a 4,000-image train subsample).
  - ResNet-18, EfficientNet-B0: runs/resnet_efficientnet_relabel_100k_matched_combined102126.json
    -- 8,000-image (train) / 4,000-image (test) random subsample of the same
    eval set, frozen embeddings + linear probe (bowhead.benchmark.run_image_pr_curves).

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.benchmark.plot_cnn_family_comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = _REPO_ROOT / "paper" / "Figures"

SCRATCH_JSON = _REPO_ROOT / "runs" / "scratch_cnn_relabel_100k_matched.json"
WARMSTART_JSON = _REPO_ROOT / "runs" / "warmstart_cnn_relabel_100k_matched_combined102126.json"
BIRDNET_JSON = _REPO_ROOT / "runs" / "birdnet_relabel_100k_matched_full199825.json"
BACKBONES_JSON = _REPO_ROOT / "runs" / "resnet_efficientnet_relabel_100k_matched_combined102126.json"

# name -> scalar AP/ROC-AUC/n/note -- kept only as a sanity-check cross-reference
# against the JSON files below; the JSON files are the source of truth for curves.
MODELS = {
    "Scratch CNN (cold start)": dict(
        ap=0.7221, roc_auc=0.9655, n=199_825,
        note="Full independent eval set, reviewed labels."),
    "Warm-start CNN (AE-init)": dict(
        ap=0.7095, roc_auc=0.9621, n=199_825,
        note="Full independent eval set, reviewed labels."),
    "BirdNET/Perch 2.0": dict(
        ap=0.3781, roc_auc=0.8068, n=199_825,
        note="Full independent eval set, reviewed labels."),
    "ResNet-18 (ImageNet)": dict(
        ap=0.6459, roc_auc=0.9232, n=8_000,
        note="8,000-image subsample, reviewed labels."),
    "EfficientNet-B0 (ImageNet)": dict(
        ap=0.6532, roc_auc=0.9299, n=8_000,
        note="8,000-image subsample, reviewed labels."),
}

COLD_START = "Scratch CNN (cold start)"


def _plot_single_model(name: str, prec: np.ndarray, rec: np.ndarray,
                        ap: float, roc_auc: float, n: int, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(rec, prec, color="#4e9af1", lw=2)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Precision-Recall")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(1 - rec, 1 - prec, color="#4e9af1", lw=2)
    axes[1].set_xlabel("Miss fraction (1 \u2212 recall)")
    axes[1].set_ylabel("False discovery rate (1 \u2212 precision)")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("FDR vs. Miss Fraction")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{name}  (AP={ap:.3f}, ROC-AUC={roc_auc:.3f}, n={n:,}, original labels)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out_png}")


def _plot_summary_bars(out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    names = list(MODELS)
    aps = [MODELS[n]["ap"] for n in names]
    colors = ["#2e8b57" if n == COLD_START else "#4e9af1" for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(names))
    ax.barh(y, aps, color=colors)
    ax.set_yticks(y, names)
    ax.set_xlabel("Average Precision")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    for yi, v in zip(y, aps):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)

    fig.suptitle("Cold-start CNN vs. other CNN-family detectors (reviewed labels)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out_png}")


def _derive_roc_from_pr(prec: np.ndarray, rec: np.ndarray, prevalence: float,
                         n: int) -> tuple[np.ndarray, np.ndarray]:
    """Recover an (FPR, TPR) curve pointwise from a (precision, recall) curve.

    These JSON result files store only the PR curve, not the raw ROC curve
    arrays. But FPR is fully recoverable from (precision, recall, prevalence,
    n) alone: TP = recall*P, FP = TP*(1/precision - 1), FPR = FP/N, where P
    and N are the known positive/negative counts. TPR is just recall.
    """
    n_pos = max(1, round(prevalence * n))
    n_neg = max(1, n - n_pos)
    tp = rec * n_pos
    with np.errstate(divide="ignore", invalid="ignore"):
        fp = np.where(prec > 0, tp * (1.0 / prec - 1.0), 0.0)
    fpr = np.clip(fp / n_neg, 0.0, 1.0)
    return fpr, rec


def _plot_combined(curves: list[tuple[str, np.ndarray, np.ndarray, float, float, float, int, str]],
                    out_png: Path, title: str, subtitle: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for name, prec, rec, ap, roc_auc, prevalence, n, color in curves:
        fpr, tpr = _derive_roc_from_pr(prec, rec, prevalence, n)
        label = f"{name} (AP={ap:.3f}, AUC={roc_auc:.3f}, n={n:,})"
        axes[0].plot(rec, prec, color=color, lw=2, label=label)
        axes[1].plot(fpr, tpr, color=color, lw=2)
        axes[2].plot(1 - rec, 1 - prec, color=color, lw=2)

    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Precision-Recall")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot([0, 1], [0, 1], color="gray", lw=1, ls="--")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("ROC")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Miss fraction (1 \u2212 recall)")
    axes[2].set_ylabel("False discovery rate (1 \u2212 precision)")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1.02)
    axes[2].set_title("FDR vs. Miss Fraction")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{title}\n{subtitle}")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=1, bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_png}")


def main() -> None:
    scratch = json.loads(SCRATCH_JSON.read_text())["reviewed"]
    _plot_single_model(
        COLD_START,
        np.asarray(scratch["pr_precision"]), np.asarray(scratch["pr_recall"]),
        scratch["average_precision"], scratch["roc_auc"], scratch["n"],
        FIG_DIR / "cnn_coldstart_pr_fdr.png",
    )

    warmstart = json.loads(WARMSTART_JSON.read_text())["reviewed"]
    _plot_single_model(
        "Warm-start CNN (AE-init)",
        np.asarray(warmstart["pr_precision"]), np.asarray(warmstart["pr_recall"]),
        warmstart["average_precision"], warmstart["roc_auc"], warmstart["n"],
        FIG_DIR / "warmstart_cnn_pr_fdr.png",
    )

    birdnet = json.loads(BIRDNET_JSON.read_text())["reviewed"]
    _plot_single_model(
        "BirdNET/Perch 2.0",
        np.asarray(birdnet["pr_precision"]), np.asarray(birdnet["pr_recall"]),
        birdnet["average_precision"], birdnet["roc_auc"], birdnet["n"],
        FIG_DIR / "birdnet_perch_pr_fdr.png",
    )

    backbones = json.loads(BACKBONES_JSON.read_text())
    for key, name, out_name in (
        ("resnet18", "ResNet-18 (ImageNet)", "resnet18_pr_fdr.png"),
        ("efficientnet_b0", "EfficientNet-B0 (ImageNet)", "efficientnet_b0_pr_fdr.png"),
    ):
        m = backbones[key]
        _plot_single_model(
            name,
            np.asarray(m["pr_precision"]), np.asarray(m["pr_recall"]),
            m["average_precision"], m["roc_auc"], m["n"],
            FIG_DIR / out_name,
        )

    _plot_summary_bars(FIG_DIR / "cnn_family_summary_bars.png")

    resnet18 = backbones["resnet18"]
    efficientnet_b0 = backbones["efficientnet_b0"]
    _plot_combined(
        [
            ("Scratch CNN (cold start)", np.asarray(scratch["pr_precision"]),
             np.asarray(scratch["pr_recall"]), scratch["average_precision"],
             scratch["roc_auc"], scratch["prevalence"], scratch["n"], "#c0392b"),
            ("Warm-start CNN (AE-init)", np.asarray(warmstart["pr_precision"]),
             np.asarray(warmstart["pr_recall"]), warmstart["average_precision"],
             warmstart["roc_auc"], warmstart["prevalence"], warmstart["n"], "#4e9af1"),
            ("BirdNET/Perch 2.0", np.asarray(birdnet["pr_precision"]),
             np.asarray(birdnet["pr_recall"]), birdnet["average_precision"],
             birdnet["roc_auc"], birdnet["prevalence"], birdnet["n"], "#2ecc71"),
            ("ResNet-18 (ImageNet)", np.asarray(resnet18["pr_precision"]),
             np.asarray(resnet18["pr_recall"]), resnet18["average_precision"],
             resnet18["roc_auc"], resnet18["prevalence"], resnet18["n"], "#8e44ad"),
            ("EfficientNet-B0 (ImageNet)", np.asarray(efficientnet_b0["pr_precision"]),
             np.asarray(efficientnet_b0["pr_recall"]), efficientnet_b0["average_precision"],
             efficientnet_b0["roc_auc"], efficientnet_b0["prevalence"], efficientnet_b0["n"], "#e67e22"),
        ],
        FIG_DIR / "cnn_family_combined_pr_fdr.png",
        title="CNN-family detectors, reviewed labels",
        subtitle=(f"(scratch/warm-start/BirdNET: full eval set, n={scratch['n']:,}; "
                   f"ResNet-18/EfficientNet-B0: subsample, n={resnet18['n']:,})"),
    )


if __name__ == "__main__":
    main()
