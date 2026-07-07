"""Score one or more CNN checkpoints on a mixed evaluation .dir and rebuild the HTML.

A "mixed" directory holds both call (Type 1-7) and non-call (Type 0) .mat files
in the same folder; labels are inferred from the _TypeN suffix rather than the
source directory.

Usage (from repo root):
    # Legacy two-model call (still works):
    python -m bowhead.eval.score_eval_dataset \\
        --eval-dir /path/to/Evaluation_200K.dir \\
        --scratch   runs/scratch/best.pt \\
        --warmstart runs/warmstart/best.pt

    # Multi-model call (repeatable --model NAME PATH):
    python -m bowhead.eval.score_eval_dataset \\
        --eval-dir /path/to/Evaluation_200K.dir \\
        --model scratch_1ch    runs/scratch_100k/best.pt \\
        --model warmstart_1ch  runs/warmstart_100k/best.pt \\
        --model scratch_2ch    runs/scratch_100k_dual/best.pt \\
        --model warmstart_2ch  runs/warmstart_100k_dual/best.pt
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
GRAMS_1CH = ("SNR_gram",)
GRAMS_2CH = ("SNR_gram", "NTV_gram")
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
    grams: tuple[str, ...] = GRAMS_1CH,
) -> np.ndarray:
    """Stream .mat files in batches; return call-probability array shape (N,)."""
    n = len(mat_paths)
    n_ch = len(grams)
    probs = np.empty(n, dtype=np.float32)
    t0 = time.time()

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_imgs = []
        for path in mat_paths[start:end]:
            try:
                m = loadmat(str(path))
                channels = [per_sample_minmax(m[g].astype(np.float32)) for g in grams]
            except Exception as e:
                print(f"  WARN: could not read {path.name}: {e}; using zeros")
                channels = [np.zeros((121, 104), dtype=np.float32)] * n_ch
            batch_imgs.append(np.stack(channels, axis=0))  # (C, H, W)

        x = torch.from_numpy(np.stack(batch_imgs)).to(device)  # (B, C, H, W)
        probs[start:end] = model.predict_proba(x).cpu().numpy()

        if (end % 10000) < BATCH_SIZE or end == n:
            rate = end / (time.time() - t0)
            eta = (n - end) / rate / 60 if rate > 0 else 0
            print(f"  {end:>7,}/{n:,}  ({rate:.0f}/s  ETA {eta:.1f} min)")

    return probs


# ── Main ──────────────────────────────────────────────────────────────────────

def run(
    eval_dir: Path,
    models: list[tuple[str, Path]],   # [(name, ckpt_path), ...]
    npz_out: Path,
    html_out: Path,
    device: str | None = None,
) -> None:
    """Score every model in *models* and rebuild the HTML.

    The legacy two-model signature (scratch/warmstart) is emulated by the
    CLI wrappers below so old call sites keep working.
    """
    if device is None or device == "auto":
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

    # 2. Score every model
    all_metrics: dict = {}   # name -> EvalMetrics
    for name, ckpt_path in models:
        print(f"\nLoading {name} from {ckpt_path} ...")
        model = _load_model(ckpt_path, device)
        # Infer channel count from checkpoint; pick gram tuple accordingly
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        n_ch = ckpt.get("cfg", {}).get("in_channels", 1)
        grams = GRAMS_2CH if n_ch == 2 else GRAMS_1CH
        print(f"  in_channels={n_ch}, grams={grams}")
        print(f"Scoring {name} ...")
        probs = _score_model(model, mat_paths, labels, device, grams=grams)
        del model
        metrics = score_predictions(labels, probs)
        all_metrics[name] = metrics
        print(f"  {name}: AP={metrics.average_precision:.4f}  "
              f"ROC-AUC={metrics.roc_auc:.4f}")

    # 3. Save npz — one array pair per model, plus shared fields
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    save_dict: dict = {"n": n, "prevalence": prevalence,
                       "model_names": np.array(list(all_metrics.keys()))}
    for name, m in all_metrics.items():
        key = name.replace(" ", "_")
        save_dict[f"{key}_precision"] = m.pr_precision
        save_dict[f"{key}_recall"]    = m.pr_recall
        save_dict[f"{key}_ap"]        = m.average_precision
        save_dict[f"{key}_roc_auc"]   = m.roc_auc
    np.savez_compressed(str(npz_out), **save_dict)
    print(f"\nSaved PR curves → {npz_out}")

    # 4. Rebuild HTML
    build_html(npz_out, html_out, eval_dir=eval_dir)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--eval-dir", type=Path, required=True,
        help="Mixed evaluation .dir folder (Type0=non-call, Type1-7=call)")
    # Legacy flags (backward-compatible)
    p.add_argument("--scratch",   type=Path, default=None,
                   help="Scratch checkpoint (legacy shorthand for --model scratch <path>)")
    p.add_argument("--warmstart", type=Path, default=None,
                   help="Warm-start checkpoint (legacy shorthand)")
    # New multi-model flag
    p.add_argument("--model", nargs=2, action="append", metavar=("NAME", "CKPT"),
                   default=[],
                   help="Repeatable: --model <name> <checkpoint_path>")
    p.add_argument(
        "--npz-out", type=Path,
        default=_REPO_ROOT / "runs" / "pr_curves_full_dataset.npz")
    p.add_argument(
        "--html-out", type=Path,
        default=_REPO_ROOT / "docs" / "detection_curves.html")
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Build ordered model list: legacy flags first, then --model entries
    models: list[tuple[str, Path]] = []
    if args.scratch:
        models.append(("scratch", Path(args.scratch)))
    if args.warmstart:
        models.append(("warmstart", Path(args.warmstart)))
    for name, ckpt in args.model:
        models.append((name, Path(ckpt)))

    if not models:
        # Fall back to repo defaults
        models = [
            ("scratch",   _REPO_ROOT / "runs" / "scratch"   / "best.pt"),
            ("warmstart", _REPO_ROOT / "runs" / "warmstart" / "best.pt"),
        ]

    run(
        eval_dir=Path(args.eval_dir),
        models=models,
        npz_out=args.npz_out,
        html_out=args.html_out,
        device=args.device,
    )
