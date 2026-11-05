"""Run the frozen-embedding + linear-probe pipeline for image backbones
(ResNet-18, EfficientNet-B0) on the full 199,825-image independent
evaluation directory, and save the resulting per-threshold PR curves to a
reusable .npz (mirrors bowhead.eval.score_eval_dataset's output format).

Unlike bowhead.benchmark.run_image_pr_curves (which only builds an HTML
panel), this script persists the raw precision/recall arrays so they can be
reused in other figures (e.g. plot_cnn_family_combined_pr_fdr.py).

Protocol: grouped date x site split (val_frac=0.10, test_frac=0.15, seed=0),
same as run_image_pr_curves.py, so results are directly comparable to the
existing scalar ResNet-18/EfficientNet-B0 rows in the paper.

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.benchmark.save_image_backbone_curves \\
        --eval-dir /path/to/Evaluation_200K.dir \\
        --npz-out runs/pr_curves_image_backbones.npz
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bowhead.config import best_device
from bowhead.data.splits import grouped_split
from bowhead.benchmark.probes.detection import full_train_detection

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FNAME_RE = re.compile(
    r"S(?P<site>\d)(?P<yy>\d{2})(?P<dasar>[A-G])\dT"
    r"(?P<date>\d{8})T(?P<hms>\d{6})_Type(?P<type>\d+)"
)
GRAM = "SNR_gram"


def _scan_dir(eval_dir: Path):
    paths, labels, dates, sites = [], [], [], []
    skipped = 0
    for fp in sorted(eval_dir.glob("*.mat")):
        m = _FNAME_RE.match(fp.stem)
        if m is None:
            skipped += 1
            continue
        paths.append(fp)
        labels.append(0 if int(m.group("type")) == 0 else 1)
        dates.append(m.group("date"))
        sites.append(m.group("site"))
    if skipped:
        print(f"  Skipped {skipped} files (no pattern match)")
    return paths, np.array(labels, np.int64), np.array(dates), np.array(sites)


def _load_images(paths: list[Path], indices: np.ndarray) -> np.ndarray:
    imgs = []
    t0 = time.time()
    for j, i in enumerate(indices):
        try:
            arr = loadmat(str(paths[i]))[GRAM].astype(np.float32)
        except Exception:
            arr = np.zeros((121, 104), dtype=np.float32)
        imgs.append(arr)
        if (j + 1) % 20_000 == 0:
            rate = (j + 1) / (time.time() - t0)
            eta = (len(indices) - j - 1) / max(rate, 1) / 60
            print(f"    loaded {j+1:,}/{len(indices):,}  ({rate:.0f}/s  ETA {eta:.1f} min)")
    return np.stack(imgs)


def _extract(backbone, images: np.ndarray, batch_size: int = 64) -> np.ndarray:
    all_embs = []
    t0 = time.time()
    n = len(images)
    for start in range(0, n, batch_size):
        all_embs.append(backbone.embed(images[start:start + batch_size], batch_size=batch_size))
        done = min(start + batch_size, n)
        if (start // batch_size + 1) % 20 == 0 or done == n:
            rate = done / (time.time() - t0)
            eta = (n - done) / max(rate, 1) / 60
            print(f"    {done:>7,}/{n:,}  ({rate:.0f}/s  ETA {eta:.1f} min)")
    return np.concatenate(all_embs, axis=0)


def run(eval_dir: Path, npz_out: Path, backbones: list[str], device: str | None,
        matched_train_npz: Path | None = None, batch_size: int = 64) -> None:
    if device is None or device == "auto":
        device = best_device()
    print(f"Device: {device}")

    print(f"\nScanning {eval_dir.name} ...")
    paths, labels, dates, sites = _scan_dir(eval_dir)
    n = len(paths)
    prev = float(labels.mean())
    print(f"  {n:,} files | {int(labels.sum()):,} calls | "
          f"{int((labels == 0).sum()):,} non-calls | prevalence={prev:.4f}")

    if matched_train_npz is not None:
        # Protocol-matched with the custom CNNs: probe trained on the SAME
        # 100,000-image training draw used for scratch/warm-start CNN
        # training (data/spectrograms_100k_matched.npz), then scored on
        # EVERY image in the independent eval directory -- zero leakage,
        # since the two datasets are entirely disjoint, and N matches the
        # custom CNN rows exactly (199,825).
        print(f"\nLoading matched training set {matched_train_npz.name} ...")
        d = np.load(str(matched_train_npz), allow_pickle=True)
        train_imgs = d["images"].astype(np.float32)
        train_labels = d["label"].astype(np.int64)
        print(f"  train: {train_imgs.shape}")
        print(f"\nLoading all {n:,} eval images ...")
        test_imgs = _load_images(paths, np.arange(n))
        test_labels = labels
        print(f"  test:  {test_imgs.shape}")
    else:
        groups = np.array([f"{d}_{s}" for d, s in zip(dates, sites)])
        split = grouped_split(labels, groups, val_frac=0.10, test_frac=0.15, seed=0)
        print(split.summary(labels, groups))

        print("\nLoading train images ...")
        train_imgs = _load_images(paths, split.train)
        print(f"  train: {train_imgs.shape}")
        print("Loading test images ...")
        test_imgs = _load_images(paths, split.test)
        print(f"  test:  {test_imgs.shape}")

        train_labels = labels[split.train]
        test_labels = labels[split.test]

    from bowhead.benchmark.backbones.image import load_image_backbone

    # When scoring on the FULL eval directory (matched_train_npz mode), don't
    # resample to a target prevalence -- the eval dir's natural prevalence
    # (~0.1109) already matches the realistic ~1:9 rate used everywhere else
    # in this paper, and skipping the resample keeps N exactly 199,825,
    # identical to the custom CNN rows (Table~\ref{tab:benchmark_results}).
    target_prevalence = None if matched_train_npz is not None else 1.0 / 9.0

    save_dict: dict = {"n": n, "prevalence": prev,
                        "test_n": len(test_labels), "test_prevalence": float(test_labels.mean())}
    for bb_name in backbones:
        print(f"\n[{bb_name}] Loading backbone ...")
        bb = load_image_backbone(bb_name, device=device)

        print(f"  Extracting train embeddings ({len(train_imgs):,} images) ...")
        emb_train = _extract(bb, train_imgs, batch_size=batch_size)
        print(f"  Extracting test embeddings ({len(test_imgs):,} images) ...")
        emb_test = _extract(bb, test_imgs, batch_size=batch_size)

        print("  Fitting full-train detection probe ...")
        m = full_train_detection(
            emb_train, train_labels, emb_test, test_labels,
            backbone_name=bb_name, target_prevalence=target_prevalence, seed=0,
        )
        print(f"  {bb_name}: AP={m.average_precision:.4f}  ROC-AUC={m.roc_auc:.4f}  "
              f"n_test={m.n:,}")
        save_dict[f"{bb_name}_precision"] = m.pr_precision
        save_dict[f"{bb_name}_recall"] = m.pr_recall
        save_dict[f"{bb_name}_ap"] = m.average_precision
        save_dict[f"{bb_name}_roc_auc"] = m.roc_auc
        save_dict[f"{bb_name}_n"] = m.n

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(npz_out), **save_dict)
    print(f"\nSaved -> {npz_out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--npz-out", type=Path,
                   default=_REPO_ROOT / "runs" / "pr_curves_image_backbones.npz")
    p.add_argument("--backbones", nargs="+", default=["resnet18", "efficientnet_b0"])
    p.add_argument("--device", default=None)
    p.add_argument("--matched-train-npz", type=Path, default=None,
                   help="If set, train the probe on this npz (e.g. the same "
                        "100k set used for the custom CNNs) instead of a "
                        "split of --eval-dir, and score on EVERY image in "
                        "--eval-dir (N matches the custom CNN rows exactly).")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Embedding extraction batch size (lower this if MPS/host "
                        "memory pressure causes swap thrashing, e.g. with "
                        "EfficientNet-B0).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(eval_dir=args.eval_dir, npz_out=args.npz_out,
        backbones=args.backbones, device=args.device,
        matched_train_npz=args.matched_train_npz, batch_size=args.batch_size)
