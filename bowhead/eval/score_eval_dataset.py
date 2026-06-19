"""Score both CNN checkpoints on a mixed evaluation .dir and rebuild the HTML.

A "mixed" directory holds both call (Type 1-7) and non-call (Type 0) .mat files
in the same folder; labels are inferred from the _TypeN suffix rather than the
source directory.

Usage (from repo root):
    python -m bowhead.eval.score_eval_dataset \\
        --eval-dir /path/to/Evaluation_200K.dir \\
        --scratch   runs/scratch/best.pt \\
        --warmstart runs/warmstart/best.pt \\
        --npz-out   runs/pr_curves_full_dataset.npz \\
        --html-out  docs/detection_curves.html
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from bowhead.data.dataset import per_sample_minmax
from bowhead.models.custom_cnn import EncoderClassifier
from bowhead.eval.metrics import score_predictions
from bowhead.config import best_device
from bowhead.eval.build_curves_html import build_html

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FNAME_RE = re.compile(
    r"S(?P<site>\d)(?P<yy>\d{2})(?P<dasar>[A-G])\dT"
    r"(?P<date>\d{8})T(?P<hms>\d{6})_Type(?P<type>\d+)"
)
GRAM = "SNR_gram"
BATCH_SIZE = 1024


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(ckpt_path: Path, device: str) -> EncoderClassifier:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {})
    model = EncoderClassifier(
        num_classes=cfg.get("num_classes", 2),
        in_channels=cfg.get("in_channels", 1),
        input_hw=tuple(cfg.get("input_hw", (121, 104))),
        latent_dim=cfg.get("latent_dim", 32),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


# ── Scoring ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def _score_model(
    model: EncoderClassifier,
    mat_paths: list[Path],
    labels: np.ndarray,
    device: str,
) -> np.ndarray:
    """Stream .mat files in batches; return call-probability array shape (N,)."""
    n = len(mat_paths)
    probs = np.empty(n, dtype=np.float32)
    t0 = time.time()

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_imgs = []
        for path in mat_paths[start:end]:
            try:
                m = loadmat(str(path))
                img = m[GRAM].astype(np.float32)
            except Exception as e:
                print(f"  WARN: could not read {path.name}: {e}; using zeros")
                img = np.zeros((121, 104), dtype=np.float32)
            batch_imgs.append(per_sample_minmax(img))

        x = torch.from_numpy(
            np.stack(batch_imgs)[:, None, :, :]   # (B, 1, H, W)
        ).to(device)
        probs[start:end] = model.predict_proba(x).cpu().numpy()

        if (end % 10000) < BATCH_SIZE or end == n:
            rate = end / (time.time() - t0)
            eta = (n - end) / rate / 60 if rate > 0 else 0
            print(f"  {end:>7,}/{n:,}  ({rate:.0f}/s  ETA {eta:.1f} min)")

    return probs


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    eval_dir: Path,
    scratch_ckpt: Path,
    warmstart_ckpt: Path,
    npz_out: Path,
    html_out: Path,
    device: str | None = None,
) -> None:
    if device is None:
        device = best_device()
        print(f"Device: {device}")
    # 1. Scan directory; infer labels from filename type
    print(f"\nScanning {eval_dir} ...")
    mat_paths: list[Path] = []
    raw_labels: list[int] = []
    skipped = 0

    for fp in sorted(eval_dir.glob("*.mat")):
        m = _FNAME_RE.match(fp.stem)
        if m is None:
            skipped += 1
            continue
        call_type = int(m.group("type"))
        label = 0 if call_type == 0 else 1
        mat_paths.append(fp)
        raw_labels.append(label)

    if skipped:
        print(f"  Skipped {skipped} files not matching filename pattern")

    labels = np.array(raw_labels, dtype=np.int64)
    n = len(mat_paths)
    prevalence = float(labels.mean())
    print(f"  {n:,} files | {int(labels.sum()):,} calls | "
          f"{int((labels == 0).sum()):,} non-calls | prevalence={prevalence:.4f}")

    if len(np.unique(labels)) < 2:
        raise RuntimeError("Both classes must be present to compute metrics.")

    # 2. Score scratch model
    print(f"\nLoading scratch model from {scratch_ckpt} ...")
    scratch_model = _load_model(scratch_ckpt, device)
    print("Scoring scratch ...")
    scratch_probs = _score_model(scratch_model, mat_paths, labels, device)
    del scratch_model

    # 3. Score warmstart model
    print(f"\nLoading warmstart model from {warmstart_ckpt} ...")
    warmstart_model = _load_model(warmstart_ckpt, device)
    print("Scoring warmstart ...")
    warmstart_probs = _score_model(warmstart_model, mat_paths, labels, device)
    del warmstart_model

    # 4. Compute PR curves (full dataset, no prevalence resampling)
    print("\nComputing metrics ...")
    scratch_metrics   = score_predictions(labels, scratch_probs)
    warmstart_metrics = score_predictions(labels, warmstart_probs)

    print(f"  scratch   AP={scratch_metrics.average_precision:.4f}  "
          f"ROC-AUC={scratch_metrics.roc_auc:.4f}")
    print(f"  warmstart AP={warmstart_metrics.average_precision:.4f}  "
          f"ROC-AUC={warmstart_metrics.roc_auc:.4f}")

    # 5. Save npz
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(npz_out),
        scratch_precision=scratch_metrics.pr_precision,
        scratch_recall=scratch_metrics.pr_recall,
        warmstart_precision=warmstart_metrics.pr_precision,
        warmstart_recall=warmstart_metrics.pr_recall,
        scratch_ap=scratch_metrics.average_precision,
        warmstart_ap=warmstart_metrics.average_precision,
        scratch_roc_auc=scratch_metrics.roc_auc,
        warmstart_roc_auc=warmstart_metrics.roc_auc,
        n=n,
        prevalence=prevalence,
    )
    print(f"\nSaved PR curves → {npz_out}")

    # 6. Rebuild HTML
    build_html(npz_out, html_out, eval_dir=eval_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--eval-dir", type=Path, required=True,
        help="Mixed evaluation .dir folder (Type0=non-call, Type1-7=call)")
    p.add_argument(
        "--scratch", type=Path,
        default=_REPO_ROOT / "runs" / "scratch" / "best.pt")
    p.add_argument(
        "--warmstart", type=Path,
        default=_REPO_ROOT / "runs" / "warmstart" / "best.pt")
    p.add_argument(
        "--npz-out", type=Path,
        default=_REPO_ROOT / "runs" / "pr_curves_full_dataset.npz")
    p.add_argument(
        "--html-out", type=Path,
        default=_REPO_ROOT / "docs" / "detection_curves.html")
    p.add_argument("--device", default=None,
                   help="Torch device: cpu | mps | cuda (default: auto-detect mps > cuda > cpu)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        eval_dir=args.eval_dir,
        scratch_ckpt=args.scratch,
        warmstart_ckpt=args.warmstart,
        npz_out=args.npz_out,
        html_out=args.html_out,
        device=args.device,
    )
