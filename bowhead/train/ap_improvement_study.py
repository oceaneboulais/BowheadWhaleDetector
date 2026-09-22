"""AP-improvement ablation study for the cold-start (scratch) CNN.

Trains the scratch CNN on the matched 100K dataset under five conditions —
baseline (reproduces the published AP=0.809 run) plus four individually-toggled
"Tier-1" AP boosters, then a sixth "combined" run with all four enabled at
once — and reports/plots the held-out-test AP, ROC-AUC, and P@R0.70 for each,
using the exact same leakage-free grouped split and realistic-prevalence
resampling as the published pipeline (``bowhead.train.train_cnn``).

Also reports a *calibration* comparison (Brier score, before/after prior
correction) since raw prediction probabilities are needed for that and are not
returned by ``evaluate_scorer``. NOTE: prior correction is a monotonic
per-sample score shift, so it cannot move AP/ROC-AUC/PR — see
``bowhead.eval.metrics.prior_corrected_probs`` docstring.

This uses the internal held-out test split (~9.3K images resampled to ~11%
prevalence) as a fast, offline proxy for the true 199,825-image independent
eval set (which requires the external R3D_2024_1 drive to be mounted). Run
``bowhead.eval.score_eval_dataset`` against that drive afterwards to confirm
the winning configuration on the full eval set before updating the paper.

Usage (from repo root):
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.train.ap_improvement_study
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from bowhead.config import TrainConfig, best_device
from bowhead.data.dataset import SpectrogramDataset
from bowhead.data.splits import grouped_split, make_date_site_group
from bowhead.eval.metrics import (
    brier_score,
    prior_corrected_probs,
    resample_to_prevalence,
    score_predictions,
)
from bowhead.models.custom_cnn import EncoderClassifier
from bowhead.scorers import CNNScorer
from bowhead.train.train_cnn import _build_groups, train_custom_cnn

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = "data/spectrograms_100k_matched.npz"
OUT_DIR = _REPO_ROOT / "runs" / "ap_improvement_study"
FIG_PATH = _REPO_ROOT / "results" / "ap_improvement_study_comparison.png"

BASE_CFG = TrainConfig(data_path=DATA_PATH, seed=42, device="auto")

# name -> TrainConfig field overrides
VARIANTS: dict[str, dict] = {
    "baseline": {},
    "dropout": {"dropout": 0.3},
    "spec_augment": {"use_spec_augment": True},
    "weight_decay": {"weight_decay": 1e-4},
    "lr_schedule_cosine": {"lr_schedule": "cosine"},
    "combined_all": {
        "dropout": 0.3,
        "use_spec_augment": True,
        "weight_decay": 1e-4,
        "lr_schedule": "cosine",
    },
}


def _calibration_pass(tag: str, cfg: TrainConfig) -> dict:
    """Reload a trained checkpoint and reproduce its held-out test split to
    compute raw probabilities, then compare Brier score before/after prior
    correction. Uses the same seed/group_col/val_frac/test_frac as training,
    so the split is bit-for-bit identical to the one used inside
    ``train_custom_cnn``.
    """
    device = best_device() if cfg.device == "auto" else cfg.device
    images, labels, metadata = SpectrogramDataset.load_npz(cfg.data_path)
    groups = _build_groups(metadata, cfg.group_col)
    split = grouped_split(labels, groups, val_frac=cfg.val_frac,
                           test_frac=cfg.test_frac, seed=cfg.seed)
    train_prior = float(labels[split.train].mean())

    idx = resample_to_prevalence(labels[split.test], cfg.eval_prevalence, seed=cfg.seed)
    test_idx = split.test[idx]
    test_images, test_labels = images[test_idx], labels[test_idx]

    ckpt = torch.load(str(OUT_DIR / tag / "best.pt"), map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("cfg", {})
    model = EncoderClassifier(
        num_classes=ckpt_cfg.get("num_classes", 2),
        in_channels=ckpt_cfg.get("in_channels", 1),
        input_hw=tuple(ckpt_cfg.get("input_hw", (121, 104))),
        latent_dim=ckpt_cfg.get("latent_dim", 32),
        # must match the checkpoint's architecture (Dropout module shifts the
        # classifier's layer indices); model.eval() below disables it anyway
        dropout=ckpt_cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["state_dict"])
    scorer = CNNScorer(model, name=tag, device=device)
    probs = scorer.score(test_images)

    corrected = prior_corrected_probs(probs, train_prior=train_prior, target_prior=cfg.eval_prevalence)
    raw_metrics = score_predictions(test_labels, probs)
    corrected_metrics = score_predictions(test_labels, corrected)
    return {
        "train_prior": train_prior,
        "target_prior": cfg.eval_prevalence,
        "brier_raw": brier_score(test_labels, probs),
        "brier_prior_corrected": brier_score(test_labels, corrected),
        # sanity check: rank-based metrics must be identical (monotonic shift)
        "ap_raw": raw_metrics.average_precision,
        "ap_prior_corrected": corrected_metrics.average_precision,
    }


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    results_path = OUT_DIR / "results.json"

    for name, overrides in VARIANTS.items():
        cfg = replace(BASE_CFG, tag=name, out_dir=str(OUT_DIR), **overrides)
        print(f"\n{'=' * 70}\nVariant: {name}  ({overrides or 'no overrides (baseline)'})\n{'=' * 70}")
        summary = train_custom_cnn(cfg)
        calibration = _calibration_pass(name, cfg)
        results[name] = {"train_summary": summary, "calibration": calibration}
        results_path.write_text(json.dumps(results, indent=2))
        print(f"[{name}] test AP={summary['test']['average_precision']:.4f}  "
              f"ROC-AUC={summary['test']['roc_auc']:.4f}  "
              f"(brier raw={calibration['brier_raw']:.4f} -> "
              f"corrected={calibration['brier_prior_corrected']:.4f})")

    _plot(results)
    _print_table(results)
    return results


def _print_table(results: dict) -> None:
    print(f"\n{'variant':22s}{'AP':>8s}{'ROC-AUC':>10s}{'P@R0.70':>10s}{'Brier(raw)':>12s}{'Brier(corr)':>12s}")
    print("-" * 74)
    for name, r in results.items():
        t = r["train_summary"]["test"]
        p70 = r["train_summary"]["test_precision_at_recall_0.70"]
        c = r["calibration"]
        print(f"{name:22s}{t['average_precision']:8.4f}{t['roc_auc']:10.4f}{p70:10.4f}"
              f"{c['brier_raw']:12.4f}{c['brier_prior_corrected']:12.4f}")


def _plot(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    ap = [results[n]["train_summary"]["test"]["average_precision"] for n in names]
    roc = [results[n]["train_summary"]["test"]["roc_auc"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(names))
    colors = ["#888888"] + ["#4e9af1"] * (len(names) - 2) + ["#2ca02c"]

    axes[0].bar(x, ap, color=colors)
    axes[0].axhline(0.8090, color="red", ls="--", lw=1, label="published AP=0.809")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_ylabel("Average Precision (held-out test, ~11% prevalence)")
    axes[0].set_title("AP-improvement ablation study")
    axes[0].legend()

    axes[1].bar(x, roc, color=colors)
    axes[1].axhline(0.9637, color="red", ls="--", lw=1, label="published ROC-AUC=0.964")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_ylabel("ROC-AUC (held-out test)")
    axes[1].set_title("ROC-AUC by variant")
    axes[1].legend()

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nSaved comparison plot -> {FIG_PATH}")


if __name__ == "__main__":
    run()
