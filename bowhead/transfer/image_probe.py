"""Image-based transfer learning: frozen ImageNet CNN → linear probe on spectrograms.

Runs immediately on the existing spectrograms.npz — no raw audio required.

BirdNET / Perch 2.0 / GMWM are waveform models whose feature extraction is
baked in at 32–48 kHz; they cannot accept pre-computed SNR-gram images. Those
benchmarks require the underlying raw audio clips (see bowhead/transfer/embedders.py).

This module provides an equivalent comparison using ImageNet-pretrained image
CNNs (ResNet-18, EfficientNet-B0) as frozen feature extractors — the standard
spectrogram-image transfer approach used throughout bioacoustics.

Protocol mirrors Burns et al. 2025 (Perch 2.0 whale study):
  - frozen backbone → extract (N, D) embeddings
  - logistic-regression linear probe, class_weight='balanced'
  - few-shot sweeps: k ∈ {4, 8, 16, 32} examples/class, 5 repeats
  - full-train probe for the upper-bound comparison
  - ROC-AUC + PR curves written to TensorBoard

Run:
    PYTHONPATH=. /usr/local/bin/python3.8 -m bowhead.transfer.image_probe \
        --data data/spectrograms_demo.npz \
        --out runs \
        --models resnet18 efficientnet_b0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

from bowhead.data.dataset import per_sample_minmax, SpectrogramDataset
from bowhead.data.splits import grouped_split, make_date_site_group


# ── backbone registry ───────────────────────────────────────────────────────

def _resnet18(device: str) -> tuple[nn.Module, int]:
    m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    return m.to(device).eval(), 512


def _efficientnet_b0(device: str) -> tuple[nn.Module, int]:
    m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = nn.Identity()
    return m.to(device).eval(), 1280


BACKBONES: dict[str, callable] = {
    "resnet18":       _resnet18,
    "efficientnet_b0": _efficientnet_b0,
}

# Stubs for waveform-based models — cannot run without raw audio.
# Included here so the benchmark table clearly shows what is pending.
WAVEFORM_MODELS = {
    "birdnet":  "BirdNET 2.3  — needs raw audio at 48 kHz (TF-Hub; GPU cluster only)",
    "perch":    "Perch 2.0    — needs raw audio at 32 kHz (TF-Hub; GPU cluster only)",
    "gmwm":     "GMWM         — needs raw audio at 24 kHz (TF-Hub / Kaggle; GPU cluster only)",
}


# ── embedding extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    backbone: nn.Module,
    images: np.ndarray,   # (N, H, W) uint8
    batch_size: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """(N, H, W) uint8 → (N, D) float32 ImageNet embeddings.

    Spectrograms are 1-channel grayscale; broadcast to 3 channels and resize
    to 224×224 (ImageNet canonical size) via bilinear interpolation.
    """
    # Pre-normalise all images to [0, 1] float32
    norm = np.stack([per_sample_minmax(im) for im in images])  # (N, H, W)
    # -> (N, 3, 224, 224)
    t = torch.from_numpy(norm[:, None, :, :])           # (N, 1, H, W)
    t = t.expand(-1, 3, -1, -1)                         # broadcast to 3ch
    t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear",
                                        align_corners=False)
    # ImageNet normalisation
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    t = (t - mean) / std

    ds = TensorDataset(t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    for (batch,) in dl:
        out.append(backbone(batch.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


# ── linear probe ─────────────────────────────────────────────────────────────

def _fit_probe(X_tr: np.ndarray, y_tr: np.ndarray, seed: int = 0) -> object:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                           random_state=seed),
    )
    clf.fit(X_tr, y_tr)
    return clf


def full_probe_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 0,
) -> dict:
    clf = _fit_probe(embeddings[train_idx], labels[train_idx], seed)
    y_score = clf.predict_proba(embeddings[test_idx])[:, 1]
    y_true = labels[test_idx]
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "ap":      float(average_precision_score(y_true, y_score)),
        "pr_prec": prec, "pr_rec": rec,
        "roc_fpr": fpr,  "roc_tpr": tpr,
        "y_score": y_score, "y_true": y_true,
    }


def few_shot_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    k_values: tuple[int, ...] = (4, 8, 16, 32),
    repeats: int = 5,
    seed: int = 0,
) -> dict[int, dict]:
    pos = train_idx[labels[train_idx] == 1]
    neg = train_idx[labels[train_idx] == 0]
    results: dict[int, dict] = {}
    for k in k_values:
        aucs, aps = [], []
        for r in range(repeats):
            rng = np.random.default_rng(seed + r)
            if len(pos) < k or len(neg) < k:
                print(f"  WARN: not enough examples for k={k}, skipping")
                break
            idx = np.concatenate([
                rng.choice(pos, k, replace=False),
                rng.choice(neg, k, replace=False),
            ])
            clf = _fit_probe(embeddings[idx], labels[idx], seed + r)
            y_score = clf.predict_proba(embeddings[test_idx])[:, 1]
            y_true = labels[test_idx]
            aucs.append(float(roc_auc_score(y_true, y_score)))
            aps.append(float(average_precision_score(y_true, y_score)))
        if aucs:
            results[k] = {
                "roc_auc_mean": float(np.mean(aucs)),
                "roc_auc_std":  float(np.std(aucs)),
                "ap_mean":      float(np.mean(aps)),
                "ap_std":       float(np.std(aps)),
            }
    return results


# ── TensorBoard writing ───────────────────────────────────────────────────────

def _write_to_tensorboard(
    writer: SummaryWriter,
    model_name: str,
    full: dict,
    few_shot: dict[int, dict],
) -> None:
    # Scalars: full-train probe
    writer.add_scalar(f"transfer/{model_name}/full_probe/roc_auc", full["roc_auc"], 0)
    writer.add_scalar(f"transfer/{model_name}/full_probe/avg_precision", full["ap"], 0)

    # ROC curve as image
    _write_curve_image(writer, f"transfer/{model_name}/ROC_curve",
                       full["roc_fpr"], full["roc_tpr"],
                       xlabel="FPR", ylabel="TPR",
                       title=f"{model_name}  ROC  (AUC={full['roc_auc']:.3f})")

    # PR curve as image
    _write_curve_image(writer, f"transfer/{model_name}/PR_curve",
                       full["pr_rec"], full["pr_prec"],
                       xlabel="Recall", ylabel="Precision",
                       title=f"{model_name}  PR  (AP={full['ap']:.3f})")

    # Few-shot ROC-AUC vs k
    for k, res in few_shot.items():
        writer.add_scalar(f"transfer/{model_name}/few_shot/roc_auc_mean", res["roc_auc_mean"], k)
        writer.add_scalar(f"transfer/{model_name}/few_shot/ap_mean",      res["ap_mean"],      k)

    # Summary text in the Text tab
    lines = [f"### {model_name}  —  full-train linear probe",
             f"ROC-AUC: **{full['roc_auc']:.4f}**   AP: **{full['ap']:.4f}**",
             "", "#### Few-shot ROC-AUC (k examples/class, 5 repeats)",
             "| k | mean | ±std |",
             "|---|------|------|"]
    for k, res in sorted(few_shot.items()):
        lines.append(f"| {k} | {res['roc_auc_mean']:.3f} | ±{res['roc_auc_std']:.3f} |")
    lines += ["", "---",
              "**Note:** BirdNET / Perch 2.0 / GMWM pending — require raw audio waveforms (GPU cluster)."]
    writer.add_text(f"transfer/{model_name}/summary", "\n".join(lines), 0)

    print(f"  {model_name:20s}  ROC-AUC={full['roc_auc']:.4f}  AP={full['ap']:.4f}")
    for k, res in sorted(few_shot.items()):
        print(f"    k={k:<3d}  {res['roc_auc_mean']:.3f} ±{res['roc_auc_std']:.3f}")


def _write_curve_image(
    writer: SummaryWriter,
    tag: str,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, linewidth=1.8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # render to (3, H, W) tensor
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    writer.add_image(tag, torch.from_numpy(buf).permute(2, 0, 1), 0)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def run(
    data_path: str,
    model_names: list[str],
    out_dir: str,
    device: str,
    seed: int,
) -> None:
    print(f"Loading {data_path} …")
    data = np.load(data_path, allow_pickle=True)
    images = data["images"]          # (N, H, W) uint8
    labels = data["label"].astype(int)
    dates  = data["date"].astype(str)
    sites  = data["site"].astype(str)

    # Grouped split — same leakage-free protocol as the CNN runs
    groups = make_date_site_group(dates, sites)
    split = grouped_split(labels, groups, val_frac=0.15, test_frac=0.15, seed=seed)
    print(f"  train={len(split.train)}  val={len(split.val)}  test={len(split.test)}")

    # Print waveform model stubs so they appear in the run output
    print("\nWaveform-based models (pending raw audio on GPU cluster):")
    for name, note in WAVEFORM_MODELS.items():
        print(f"  {note}")

    print()
    # One SummaryWriter per run tag (same as CNN runs, so all appear together)
    writers = {
        "scratch_demo":   SummaryWriter(log_dir=str(Path(out_dir) / "scratch_demo")),
        "warmstart_demo": SummaryWriter(log_dir=str(Path(out_dir) / "warmstart_demo")),
    }

    for model_name in model_names:
        if model_name not in BACKBONES:
            print(f"Unknown model {model_name!r}, skipping. Available: {list(BACKBONES)}")
            continue

        print(f"\n── {model_name} ──")
        backbone, embed_dim = BACKBONES[model_name](device)
        print(f"  extracting embeddings (dim={embed_dim}) …")
        embeddings = extract_embeddings(backbone, images, device=device)
        print(f"  embeddings shape: {embeddings.shape}")

        full   = full_probe_metrics(embeddings, labels, split.train, split.test, seed)
        fs     = few_shot_metrics(embeddings, labels, split.train, split.test, seed=seed)

        # Write to both run writers (same backbone, same data — identical results,
        # but makes the transfer baselines visible in each run's Scalars tab)
        for tag, writer in writers.items():
            _write_to_tensorboard(writer, model_name, full, fs)

    for w in writers.values():
        w.close()

    print("\nAll done. Refresh TensorBoard at http://127.0.0.1:6006")
    print("  → Scalars tab: transfer/<model>/full_probe/roc_auc")
    print("  → Images tab:  transfer/<model>/ROC_curve  +  PR_curve")
    print("  → Text tab:    transfer/<model>/summary")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="data/spectrograms_demo.npz")
    p.add_argument("--out",    default="runs")
    p.add_argument("--models", nargs="+", default=list(BACKBONES.keys()))
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed",   type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(a.data, a.models, a.out, a.device, a.seed)
