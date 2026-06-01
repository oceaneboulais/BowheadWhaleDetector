"""Build a rich TensorBoard dashboard from a trained run's held-out test set.

The training loop (``bowhead.train.train_cnn``) only logs bare scalars — train
loss, val AUC, a handful of test scalars. This script reloads a trained
checkpoint, rebuilds the *exact* leakage-free test split it was evaluated on, and
emits a comprehensive TensorBoard summary so every result in the project is
viewable on one dashboard:

    * SUMMARY  — markdown panel: headline metrics + dataset / split facts (TEXT tab)
    * SCALARS  — all detection metrics as scalars
    * PR-CURVE — native TensorBoard precision-recall curve (PR CURVES tab)
    * FIGURES  — PR curve, ROC curve, confusion matrix, score histogram (IMAGES tab)
    * EXAMPLES — TP / FP / FN / TN spectrogram grids (IMAGES tab)
    * HPARAMS  — the run's hyperparameters vs its metrics (HPARAMS tab)

Run:
    PYTHONPATH=. python -m bowhead.eval.build_dashboard \
        --ckpt runs/scratch_demo/best.pt --tag scratch_demo_dashboard

The dashboard run lands in ``runs/<tag>/`` so it shows up alongside the training
run under ``tensorboard --logdir runs`` (and in the Hugging Face Space).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")  # headless — write figures, never open a window
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

from bowhead.data.dataset import SpectrogramDataset, per_sample_minmax
from bowhead.data.splits import grouped_split, make_date_site_group
from bowhead.eval.metrics import score_predictions, resample_to_prevalence
from bowhead.models.custom_cnn import EncoderClassifier


# --------------------------------------------------------------------------- #
# Checkpoint + data
# --------------------------------------------------------------------------- #
def _load_checkpoint(ckpt_path: str) -> tuple[EncoderClassifier, dict]:
    """Rebuild the EncoderClassifier and return (model, cfg) from a best.pt."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    model = EncoderClassifier(
        num_classes=cfg.get("num_classes", 2),
        in_channels=cfg.get("in_channels", 1),
        input_hw=tuple(cfg.get("input_hw", (121, 104))),
        latent_dim=cfg.get("latent_dim", 32),
        dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


def _build_groups(metadata: dict, group_col: str) -> np.ndarray:
    if group_col == "date_site":
        return make_date_site_group(metadata["date"], metadata["site"])
    if group_col in metadata:
        return np.asarray(metadata[group_col])
    raise KeyError(f"group_col {group_col!r} not in metadata keys {list(metadata)}")


@torch.no_grad()
def _score(model: EncoderClassifier, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """P(call) for a stack of images, with the same per-sample min-max as training."""
    if images.ndim == 3:
        images = images[:, None, :, :]
    probs = []
    for s in range(0, len(images), batch_size):
        batch = images[s:s + batch_size]
        batch = np.stack([np.stack([per_sample_minmax(ch) for ch in img]) for img in batch])
        x = torch.from_numpy(batch.astype(np.float32))
        probs.append(model.predict_proba(x).cpu().numpy())
    return np.concatenate(probs)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _fig_pr(m) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(m.pr_recall, m.pr_precision, lw=2, color="#1f77b4")
    ax.axhline(m.prevalence, ls="--", lw=1, color="grey",
               label=f"chance (prevalence {m.prevalence:.3f})")
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02),
           title=f"Precision–Recall (AP = {m.average_precision:.3f})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _fig_roc(m) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(m.roc_fpr, m.roc_tpr, lw=2, color="#d62728")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="grey", label="chance")
    ax.set(xlabel="false positive rate", ylabel="true positive rate",
           xlim=(0, 1), ylim=(0, 1.02), title=f"ROC (AUC = {m.roc_auc:.3f})")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def _operating_threshold(m, target_recall: float = 0.70) -> float:
    """Threshold giving the highest precision at >= target_recall."""
    mask = m.pr_recall[:-1] >= target_recall  # last point has no threshold
    if not mask.any():
        return 0.5
    # among thresholds meeting the recall floor, take the most precise
    idx = np.where(mask)[0]
    best = idx[np.argmax(m.pr_precision[:-1][idx])]
    return float(m.pr_thresholds[best])


def _fig_confusion(y_true, y_score, thr: float, target_recall: float) -> plt.Figure:
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["non-call", "call"], yticklabels=["non-call", "call"],
           xlabel="predicted", ylabel="true",
           title=f"Confusion @ recall≥{target_recall:.2f}\n(threshold {thr:.3f})")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _fig_score_hist(y_true, y_score) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_score[y_true == 0], bins=40, range=(0, 1), alpha=0.6,
            label="non-call", color="#7f7f7f", density=True)
    ax.hist(y_score[y_true == 1], bins=40, range=(0, 1), alpha=0.6,
            label="call", color="#2ca02c", density=True)
    ax.set(xlabel="P(call)", ylabel="density", title="Score separation by class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _fig_examples(images, y_true, y_score, thr: float, title: str,
                  pred_pos: bool, correct: bool, n: int = 8) -> plt.Figure | None:
    """Grid of example spectrograms for one cell of the confusion matrix.

    pred_pos/correct select TP (T,T), FP (T,F-ish)... we compute the mask directly.
    """
    y_pred = (y_score >= thr).astype(int)
    want_pred = 1 if pred_pos else 0
    want_true = want_pred if correct else 1 - want_pred
    mask = (y_pred == want_pred) & (y_true == want_true)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    # most-confident-first so the grid is representative of the cell
    order = idx[np.argsort(-np.abs(y_score[idx] - 0.5))][:n]
    cols = min(4, len(order))
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.0 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, j in enumerate(order):
        ax = axes[k // cols][k % cols]
        img = images[j]
        img = img[0] if img.ndim == 3 else img
        ax.imshow(per_sample_minmax(img), aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"P={y_score[j]:.2f}", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_dashboard(
    ckpt_path: str,
    tag: str,
    data_path: str | None = None,
    out_dir: str = "runs",
    target_recall: float = 0.70,
) -> dict:
    model, cfg = _load_checkpoint(ckpt_path)
    data_path = data_path or cfg.get("data_path", "data/spectrograms.npz")
    group_col = cfg.get("group_col", "date_site")
    seed = int(cfg.get("seed", 0))
    prevalence = float(cfg.get("eval_prevalence", 1.0 / 9.0))

    images, labels, metadata = SpectrogramDataset.load_npz(data_path)
    groups = _build_groups(metadata, group_col)
    split = grouped_split(
        labels, groups,
        val_frac=float(cfg.get("val_frac", 0.15)),
        test_frac=float(cfg.get("test_frac", 0.15)),
        seed=seed,
    )

    # Held-out test set, resampled to deployment prevalence (same recipe as eval).
    test_idx = split.test
    y_test = labels[test_idx].astype(int)
    keep = resample_to_prevalence(y_test, prevalence, seed=seed)
    test_images = images[test_idx][keep]
    y_true = y_test[keep]

    y_score = _score(model, test_images)
    m = score_predictions(y_true, y_score)
    thr = _operating_threshold(m, target_recall)
    p_at_r = m.precision_at_recall(target_recall)

    out = Path(out_dir) / tag
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out))

    # --- SCALARS ---------------------------------------------------------- #
    scalars = {
        "test/roc_auc": m.roc_auc,
        "test/average_precision": m.average_precision,
        f"test/precision_at_recall_{target_recall:.2f}": p_at_r,
        "test/prevalence": m.prevalence,
        "test/n_examples": float(m.n),
    }
    for k, v in scalars.items():
        writer.add_scalar(k, v, 0)

    # --- PR CURVES tab (native) ------------------------------------------ #
    writer.add_pr_curve("test/pr_curve", y_true, y_score, global_step=0)

    # --- FIGURES ---------------------------------------------------------- #
    writer.add_figure("curves/precision_recall", _fig_pr(m), 0)
    writer.add_figure("curves/roc", _fig_roc(m), 0)
    writer.add_figure("curves/confusion_matrix",
                      _fig_confusion(y_true, y_score, thr, target_recall), 0)
    writer.add_figure("curves/score_histogram", _fig_score_hist(y_true, y_score), 0)

    # --- EXAMPLE DETECTIONS ---------------------------------------------- #
    grids = {
        "examples/true_positive": (True, True, "True positives (call → call)"),
        "examples/false_positive": (True, False, "False positives (non-call → call)"),
        "examples/false_negative": (False, False, "False negatives (call → non-call)"),
        "examples/true_negative": (False, True, "True negatives (non-call → non-call)"),
    }
    for name, (pred_pos, correct, title) in grids.items():
        fig = _fig_examples(test_images, y_true, y_score, thr, title,
                            pred_pos=pred_pos, correct=correct)
        if fig is not None:
            writer.add_figure(name, fig, 0)

    # --- SUMMARY (markdown TEXT tab) ------------------------------------- #
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    summary_md = f"""
# Bowhead call detector — results dashboard

**Run:** `{tag}`  ·  **checkpoint:** `{ckpt_path}`  ·  **architecture:** AE-encoder trunk + softmax head ({cfg.get('latent_dim', 32)}-D latent)

## Headline metrics (held-out test, deployment prevalence ≈ {m.prevalence:.3f})
| metric | value |
|---|---|
| ROC-AUC | **{m.roc_auc:.3f}** |
| Average precision | **{m.average_precision:.3f}** |
| Precision @ recall ≥ {target_recall:.2f} | **{p_at_r:.3f}** |
| Operating threshold | {thr:.3f} |
| Test examples | {m.n} ({n_pos} call / {n_neg} non-call) |

## Split (leakage-free, grouped by `{group_col}`)
- group key: `{group_col}` — every unique call / date×site lands wholly in one partition
- val / test fraction: {cfg.get('val_frac', 0.15)} / {cfg.get('test_frac', 0.15)}  ·  seed: {seed}
- evaluation prevalence: {prevalence:.4f} (~1 call : {round((1 - prevalence) / prevalence)} transients)

## Data
- source: `{data_path}`
- {len(images):,} spectrograms · {int(labels.sum()):,} calls / {int((1 - labels).sum()):,} non-calls

*See the **PR CURVES** and **IMAGES** tabs for curves, confusion matrix, score
separation, and example TP/FP/FN/TN detections.*
""".strip()
    writer.add_text("summary", summary_md, 0)

    # --- HPARAMS ---------------------------------------------------------- #
    writer.add_hparams(
        {"tag": tag, "group_col": group_col, "latent_dim": int(cfg.get("latent_dim", 32)),
         "seed": seed, "eval_prevalence": prevalence,
         "warm_start": bool(cfg.get("warm_start_ckpt"))},
        {"hparam/test_roc_auc": m.roc_auc,
         "hparam/test_avg_precision": m.average_precision,
         f"hparam/test_precision_at_recall_{target_recall:.2f}": p_at_r},
    )
    writer.close()

    summary = {
        "tag": tag, "ckpt": ckpt_path, "data": data_path,
        "roc_auc": m.roc_auc, "average_precision": m.average_precision,
        f"precision_at_recall_{target_recall:.2f}": p_at_r,
        "operating_threshold": thr, "n": m.n, "prevalence": m.prevalence,
    }
    (out / "dashboard_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nDashboard written to {out}/ — view with:\n"
          f"  ./deploy/launch_tensorboard.sh   (or: tensorboard --logdir runs)")
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a rich TensorBoard results dashboard")
    p.add_argument("--ckpt", default="runs/scratch_demo/best.pt",
                   help="trained best.pt checkpoint")
    p.add_argument("--tag", default="scratch_demo_dashboard",
                   help="dashboard run name (subdir under runs/)")
    p.add_argument("--data", dest="data_path", default=None,
                   help="override data path (defaults to the ckpt's data_path)")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--target-recall", type=float, default=0.70)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    build_dashboard(a.ckpt, a.tag, data_path=a.data_path,
                    out_dir=a.out_dir, target_recall=a.target_recall)
