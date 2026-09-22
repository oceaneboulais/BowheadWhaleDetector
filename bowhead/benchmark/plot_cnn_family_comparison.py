"""Render SEPARATE per-model figures (one PNG each) comparing the cold-start
(scratch) CNN against the other CNN-family detectors evaluated in this
project, plus a bar-chart summary across all of them.

All metrics/curves below use the ORIGINAL (pre-review, `type_org`) label set,
for apples-to-apples comparison, since not every model has been re-scored
against the manually-reviewed (`iscall`) labels.

Data sources (see docstring notes on each for exact n / protocol -- these are
NOT all evaluated identically, and that is called out explicitly rather than
implied by a shared plot):
  - Scratch CNN (cold start): runs/scratch_cnn_relabel_100k_matched.json
    ["original"] -- full 199,825-spectrogram independent eval set, full PR
    curve available.
  - BirdNET/Perch 2.0: runs/birdnet_relabel_100k_matched.json ["original"] --
    a fixed 4,000-image subsample of the same eval set (Griffin-Lim
    reconstruction + frozen linear probe), full PR curve available.
  - Warm-start CNN: runs/pr_curves_100k_matched.npz (regenerated 2026-09-04
    against the R3D_2024_1 external drive after the previous copy of this
    file was found to hold corrupted/near-random warmstart data) --
    AP=0.7869, ROC-AUC=0.9563, full 198,118-point curve, matches the
    already-published paper Table II value (0.787/0.956).
  - ResNet-18, EfficientNet-B0: only SCALAR summary metrics are currently
    reproducible -- their curve arrays only exist baked into the giant
    `docs/benchmark_pr_curves.html` Frozen-Embedding Benchmark panel (no
    separately-saved array). Re-deriving true curves for these would require
    re-running `run_image_pr_curves.py` against the raw evaluation directory
    -- flagged rather than silently approximated.

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.benchmark.plot_cnn_family_comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = _REPO_ROOT / "paper" / "Figures"

# name -> (AP, ROC-AUC, n, has_curve, note)
MODELS = {
    "Scratch CNN (cold start)": dict(
        ap=0.8090, roc_auc=0.9637, n=199_825, has_curve=True,
        note="Full independent eval set, original labels."),
    "Warm-start CNN (AE-init)": dict(
        ap=0.7869, roc_auc=0.9563, n=199_825, has_curve=True,
        note="Full independent eval set, original labels; re-scored 2026-09-04."),
    "BirdNET/Perch 2.0": dict(
        ap=0.4827, roc_auc=0.8025, n=4_000, has_curve=True,
        note="4,000-image subsample, Griffin-Lim reconstruction + frozen "
             "linear probe."),
    "ResNet-18 (ImageNet)": dict(
        ap=0.6637, roc_auc=0.8886, n=199_825, has_curve=False,
        note="Grouped 15% held-out split of eval dir, frozen + linear probe; "
             "curve only baked into docs/benchmark_pr_curves.html, not saved separately."),
    "EfficientNet-B0 (ImageNet)": dict(
        ap=0.7134, roc_auc=0.9134, n=199_825, has_curve=False,
        note="Same protocol as ResNet-18."),
}

COLD_START = "Scratch CNN (cold start)"


def _plot_single_model(name: str, prec: np.ndarray, rec: np.ndarray,
                        ap: float, roc_auc: float, n: int, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    names = list(MODELS.keys())
    aps = [MODELS[n]["ap"] for n in names]
    rocs = [MODELS[n]["roc_auc"] for n in names]
    colors = ["#c0392b" if n == COLD_START else "#4e9af1" for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    y = np.arange(len(names))
    axes[0].barh(y, aps, color=colors)
    axes[0].set_yticks(y, names)
    axes[0].set_xlabel("Average Precision")
    axes[0].set_xlim(0, 1)
    axes[0].invert_yaxis()
    axes[0].set_title("Average Precision")
    for yi, v in zip(y, aps):
        axes[0].text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)

    axes[1].barh(y, rocs, color=colors)
    axes[1].set_yticks(y, [])
    axes[1].set_xlabel("ROC-AUC")
    axes[1].set_xlim(0, 1)
    axes[1].invert_yaxis()
    axes[1].set_title("ROC-AUC")
    for yi, v in zip(y, rocs):
        axes[1].text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)

    fig.suptitle("Cold-start CNN vs. other CNN-family detectors (original labels)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out_png}")


def main() -> None:
    scratch = json.loads((_REPO_ROOT / "runs" / "scratch_cnn_relabel_100k_matched.json").read_text())
    _plot_single_model(
        COLD_START,
        np.asarray(scratch["original"]["pr_precision"]),
        np.asarray(scratch["original"]["pr_recall"]),
        scratch["original"]["average_precision"], scratch["original"]["roc_auc"],
        scratch["original"]["n"],
        FIG_DIR / "cnn_coldstart_pr_fdr.png",
    )

    birdnet = json.loads((_REPO_ROOT / "runs" / "birdnet_relabel_100k_matched.json").read_text())
    _plot_single_model(
        "BirdNET/Perch 2.0",
        np.asarray(birdnet["original"]["pr_precision"]),
        np.asarray(birdnet["original"]["pr_recall"]),
        birdnet["original"]["average_precision"], birdnet["original"]["roc_auc"],
        birdnet["original"]["n"],
        FIG_DIR / "birdnet_perch_pr_fdr.png",
    )

    warmstart = np.load(_REPO_ROOT / "runs" / "pr_curves_100k_matched.npz")
    _plot_single_model(
        "Warm-start CNN (AE-init)",
        warmstart["warmstart_100k_matched_precision"],
        warmstart["warmstart_100k_matched_recall"],
        float(warmstart["warmstart_100k_matched_ap"]), float(warmstart["warmstart_100k_matched_roc_auc"]),
        int(warmstart["n"]),
        FIG_DIR / "warmstart_cnn_pr_fdr.png",
    )

    _plot_summary_bars(FIG_DIR / "cnn_family_summary_bars.png")

    print("\nNOTE: ResNet-18 and EfficientNet-B0 plotted only as scalar bars -- "
          "no reproducible full PR-curve source is currently available for them "
          "in this repo (see module docstring).")


if __name__ == "__main__":
    main()
