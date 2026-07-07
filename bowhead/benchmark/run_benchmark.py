"""Unified benchmark entry point.

Runs all evaluation axes for every registered backbone and produces a
``runs/benchmark/<timestamp>/`` results tree plus a final HTML report.

Usage (from repo root)
----------------------
    python -m bowhead.benchmark.run_benchmark \\
        --eval-dir /path/to/Evaluation_200K.dir \\
        --out-dir  runs/benchmark

The script is intentionally modular: each backbone–probe pair is self-contained
so partial runs can be resumed by re-running with ``--backbones resnet18``.

Output layout
-------------
runs/benchmark/<timestamp>/
    resnet18/
        detection_full.npz        # PR + ROC arrays, full-train probe
        detection_fewshot.npz     # AP per k and repeat
        call_type.npz             # macro-F1 + per-class AP
    efficientnet_b0/  ...
    summary_detection.csv
    summary_call_type.csv
    report.html
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Lazy imports so the CLI starts fast ──────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_mat_images(
    eval_dir: Path,
    max_samples: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan a mixed .dir, load SNR_grams into RAM.

    Returns
    -------
    images  : (N, 121, 104) uint8
    labels  : (N,) int  0=non-call 1=call
    dates   : (N,) str  YYYYMMDD
    sites   : (N,) str
    dasars  : (N,) str
    """
    import re
    from scipy.io import loadmat

    _RE = re.compile(
        r"S(?P<site>\d)(?P<yy>\d{2})(?P<dasar>[A-G])\dT"
        r"(?P<date>\d{8})T(?P<hms>\d{6})_Type(?P<type>\d+)"
    )
    GRAM = "SNR_gram"

    paths, labels_list, dates_list, sites_list, dasars_list = [], [], [], [], []
    for fp in sorted(eval_dir.glob("*.mat")):
        m = _RE.match(fp.stem)
        if m is None:
            continue
        g = m.groupdict()
        paths.append(fp)
        labels_list.append(0 if int(g["type"]) == 0 else 1)
        dates_list.append(g["date"])
        sites_list.append(g["site"])
        dasars_list.append(g["dasar"])

    n = len(paths)
    if max_samples and n > max_samples:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(n, size=max_samples, replace=False).tolist())
        paths = [paths[i] for i in idx]
        labels_list = [labels_list[i] for i in idx]
        dates_list = [dates_list[i] for i in idx]
        sites_list = [sites_list[i] for i in idx]
        dasars_list = [dasars_list[i] for i in idx]
        n = max_samples

    images = np.empty((n, 121, 104), dtype=np.uint8)
    t0 = time.time()
    for i, fp in enumerate(paths):
        try:
            images[i] = loadmat(str(fp))[GRAM].astype(np.uint8)
        except Exception:
            images[i] = 0
        if (i + 1) % 20000 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (n - i - 1) / rate / 60
            print(f"  loaded {i + 1:,}/{n:,}  ({rate:.0f}/s  ETA {eta:.1f} min)")

    return (
        images,
        np.array(labels_list, dtype=np.int64),
        np.array(dates_list),
        np.array(sites_list),
        np.array(dasars_list),
    )


def _run_backbone(
    backbone_name: str,
    images: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    test_idx: np.ndarray,
    train_idx: np.ndarray,
    device: str,
    few_shot_k: tuple[int, ...],
    few_shot_repeats: int,
    seed: int,
) -> dict:
    """Run all probes for one backbone; return a summary dict."""
    from bowhead.benchmark.backbones.image import load_image_backbone
    from bowhead.benchmark.probes.detection import full_train_detection, few_shot_detection

    backbone_dir = out_dir / backbone_name
    backbone_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{backbone_name}] Loading backbone ...")
    backbone = load_image_backbone(backbone_name, device=device)

    # ── full-train detection ─────────────────────────────────────────────
    print(f"  Extracting train embeddings ({len(train_idx):,}) ...")
    train_emb = backbone.embed(images[train_idx])
    print(f"  Extracting test  embeddings ({len(test_idx):,}) ...")
    test_emb  = backbone.embed(images[test_idx])

    print("  Fitting full-train detection probe ...")
    det = full_train_detection(
        train_emb, labels[train_idx],
        test_emb,  labels[test_idx],
        target_prevalence=1.0 / 9.0,
        seed=seed,
    )
    np.savez_compressed(
        str(backbone_dir / "detection_full.npz"),
        precision=det.precision, recall=det.recall,
        ap=det.ap, roc_auc=det.roc_auc,
        train_n=len(train_idx), test_n=len(test_idx),
    )
    print(f"  full-train  AP={det.ap:.4f}  ROC-AUC={det.roc_auc:.4f}")

    # ── few-shot detection ───────────────────────────────────────────────
    fewshot_rows: list[dict] = []
    for k in few_shot_k:
        ap_list, roc_list = [], []
        for rep in range(few_shot_repeats):
            fs = few_shot_detection(
                train_emb, labels[train_idx],
                test_emb,  labels[test_idx],
                k=k, seed=seed + rep * 1000,
            )
            ap_list.append(fs.ap)
            roc_list.append(fs.roc_auc)
        mean_ap  = float(np.mean(ap_list))
        mean_roc = float(np.mean(roc_list))
        std_ap   = float(np.std(ap_list))
        print(f"  few-shot k={k:2d}  AP={mean_ap:.4f}±{std_ap:.4f}  "
              f"ROC-AUC={mean_roc:.4f}")
        fewshot_rows.append({"k": k, "ap_mean": mean_ap, "ap_std": std_ap,
                              "roc_mean": mean_roc})

    np.savez_compressed(
        str(backbone_dir / "detection_fewshot.npz"),
        k=np.array([r["k"] for r in fewshot_rows]),
        ap_mean=np.array([r["ap_mean"] for r in fewshot_rows]),
        ap_std=np.array([r["ap_std"] for r in fewshot_rows]),
        roc_mean=np.array([r["roc_mean"] for r in fewshot_rows]),
    )

    return {
        "backbone": backbone_name,
        "full_ap": det.ap,
        "full_roc_auc": det.roc_auc,
        "fewshot": fewshot_rows,
    }


def run_benchmark(
    eval_dir: Path,
    out_dir: Path,
    backbones: list[str],
    few_shot_k: tuple[int, ...] = (4, 8, 16, 32),
    few_shot_repeats: int = 5,
    device: str = "auto",
    seed: int = 0,
    test_frac: float = 0.15,
) -> None:
    from bowhead.config import best_device
    from bowhead.data.splits import grouped_split, make_date_site_group

    if device == "auto":
        device = best_device()
    print(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    # ── load all images ──────────────────────────────────────────────────
    print(f"\nLoading images from {eval_dir.name} ...")
    images, labels, dates, sites, dasars = _load_mat_images(eval_dir)
    n = len(images)
    prevalence = float(labels.mean())
    print(f"  {n:,} images | prevalence={prevalence:.4f}")

    # ── grouped split ────────────────────────────────────────────────────
    groups = make_date_site_group(dates, sites)
    split  = grouped_split(labels, groups, val_frac=0.0, test_frac=test_frac, seed=seed)
    train_idx = np.concatenate([split.train, split.val])   # val not needed here
    test_idx  = split.test
    print(f"  train: {len(train_idx):,}  test: {len(test_idx):,}")

    # ── per-backbone evaluation ──────────────────────────────────────────
    summaries: list[dict] = []
    for bb in backbones:
        try:
            result = _run_backbone(
                bb, images, labels, run_dir,
                test_idx, train_idx, device,
                few_shot_k, few_shot_repeats, seed,
            )
            summaries.append(result)
        except Exception as exc:
            print(f"  ERROR in {bb}: {exc}")
            continue

    # ── CSV summary ──────────────────────────────────────────────────────
    if summaries:
        csv_path = run_dir / "summary_detection.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["backbone", "full_ap", "full_roc_auc"])
            writer.writeheader()
            for s in summaries:
                writer.writerow({
                    "backbone": s["backbone"],
                    "full_ap": f"{s['full_ap']:.4f}",
                    "full_roc_auc": f"{s['full_roc_auc']:.4f}",
                })
        print(f"\nWritten {csv_path}")

    # ── JSON metadata ────────────────────────────────────────────────────
    meta = {
        "timestamp": timestamp,
        "eval_dir": str(eval_dir),
        "n": n, "prevalence": prevalence,
        "train_n": len(train_idx), "test_n": len(test_idx),
        "backbones": backbones,
        "few_shot_k": list(few_shot_k),
        "few_shot_repeats": few_shot_repeats,
        "device": device, "seed": seed,
        "summaries": summaries,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # ── HTML report ──────────────────────────────────────────────────────
    from bowhead.benchmark.report import build_report
    report_path = run_dir / "report.html"
    build_report(meta, run_dir, report_path)
    print(f"Written {report_path}")
    print("\nBenchmark complete.")


def _parse_args() -> argparse.Namespace:
    from bowhead.benchmark.config import IMAGE_BACKBONES
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True, type=Path)
    p.add_argument("--out-dir",  type=Path, default=_REPO_ROOT / "runs" / "benchmark")
    p.add_argument("--backbones", nargs="+", default=list(IMAGE_BACKBONES),
                   help="Image backbones to evaluate (default: all registered)")
    p.add_argument("--few-shot-k", nargs="+", type=int, default=[4, 8, 16, 32])
    p.add_argument("--few-shot-repeats", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        eval_dir=Path(args.eval_dir),
        out_dir=Path(args.out_dir),
        backbones=args.backbones,
        few_shot_k=tuple(args.few_shot_k),
        few_shot_repeats=args.few_shot_repeats,
        device=args.device,
        test_frac=args.test_frac,
        seed=args.seed,
    )
