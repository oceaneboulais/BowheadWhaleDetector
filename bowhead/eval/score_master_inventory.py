"""Score the frozen scratch ("cold start") CNN on the FULL 2.1M-call master
database (``runs/train_eval_v2_master_inventory_cache.json``), not just the
199,825-image independent evaluation subset used throughout the paper.

Reuses the SAME checkpoint as the paper's benchmark table
(``runs/scratch_100k_matched_retrain/best.pt``) and the same single-channel
SNR-gram scoring path as ``bowhead.eval.score_eval_dataset``. Labels and file
paths come directly from the cached inventory (each row already carries
``path`` and ``label``; 0=auto-detected transient, 1=manually-annotated call),
so there is no flat-directory/filename-parsing step -- this script bypasses
that entirely and reads the inventory's nested nested-folder paths as-is.

Requires the external drive the inventory's paths point at
(``/Volumes/R3D_2024_1/...``) to be mounted.

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.eval.score_master_inventory \\
        --inventory-json runs/train_eval_v2_master_inventory_cache.json \\
        --ckpt runs/scratch_100k_matched_retrain/best.pt \\
        --out-json runs/scratch_cnn_master_2.1M.json \\
        --out-probs-npz runs/scratch_cnn_master_2.1M_probs.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bowhead.config import best_device
from bowhead.eval.metrics import score_predictions
from bowhead.eval.score_eval_dataset import GRAMS_1CH, _load_model, _score_model

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = _REPO_ROOT / "runs" / "train_eval_v2_master_inventory_cache.json"
DEFAULT_CKPT = _REPO_ROOT / "runs" / "scratch_100k_matched_retrain" / "best.pt"
DEFAULT_OUT_JSON = _REPO_ROOT / "runs" / "scratch_cnn_master_2.1M.json"
DEFAULT_OUT_PROBS = _REPO_ROOT / "runs" / "scratch_cnn_master_2.1M_probs.npz"


def _subgroup_report(labels: np.ndarray, probs: np.ndarray, keys: np.ndarray, name: str) -> None:
    for val in sorted(set(keys.tolist())):
        mask = keys == val
        yl, yp = labels[mask], probs[mask]
        if len(np.unique(yl)) < 2:
            print(f"    {name}={val}: n={mask.sum():,} (single class, skipped)")
            continue
        m = score_predictions(yl, yp)
        print(f"    {name}={val}: n={mask.sum():,}  prevalence={yl.mean():.4f}  "
              f"ROC-AUC={m.roc_auc:.4f}  AP={m.average_precision:.4f}")


def run(
    inventory_json: Path,
    ckpt: Path,
    out_json: Path,
    out_probs_npz: Path,
    limit: int | None = None,
    device: str | None = None,
) -> None:
    device = device or best_device()
    print(f"Device: {device}")

    print(f"Loading inventory: {inventory_json}")
    rows = json.loads(Path(inventory_json).read_text())
    if limit is not None:
        rows = rows[:limit]
    n = len(rows)
    print(f"  {n:,} rows")

    mat_paths = [Path(r["path"]) for r in rows]
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    years = np.array([r["year"] for r in rows])
    sites = np.array([r["site"] for r in rows])
    dasars = np.array([r["dasar"] for r in rows])

    missing = [p for p in mat_paths[:1000] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Sample check found missing files (first: {missing[0]}). Is the "
            f"drive holding the master database mounted?"
        )

    prevalence = float(labels.mean())
    print(f"  {int(labels.sum()):,} calls | {int((labels == 0).sum()):,} non-calls | "
          f"prevalence={prevalence:.4f}")

    print(f"Loading scratch CNN checkpoint: {ckpt}")
    model = _load_model(Path(ckpt), device)

    print(f"Scoring {n:,} spectrograms (single forward pass) ...")
    probs, ok = _score_model(model, mat_paths, labels, device, grams=GRAMS_1CH, return_mask=True)

    n_bad = int((~ok).sum())
    if n_bad:
        print(f"\n{n_bad:,}/{n:,} files failed to load ({n_bad / n:.2%}); "
              f"excluding from metrics/curves (placeholder scores are meaningless)")
        print("Unreadable files by year:")
        for yr in sorted(set(years[~ok].tolist())):
            print(f"    year={yr}: {int((years[~ok] == yr).sum()):,}")
        print("Unreadable files by site:")
        for s in sorted(set(sites[~ok].tolist())):
            print(f"    site={s}: {int((sites[~ok] == s).sum()):,}")
        print("Unreadable files by dasar:")
        for dr in sorted(set(dasars[~ok].tolist())):
            print(f"    dasar={dr}: {int((dasars[~ok] == dr).sum()):,}")

    labels_ok, probs_ok, years_ok, sites_ok, dasars_ok = (
        labels[ok], probs[ok], years[ok], sites[ok], dasars[ok],
    )

    metrics = score_predictions(labels_ok, probs_ok)
    print(f"\n[overall] AP={metrics.average_precision:.4f}  ROC-AUC={metrics.roc_auc:.4f}  "
          f"prevalence={metrics.prevalence:.4f}  n={metrics.n:,}")

    print("\nBy year:")
    _subgroup_report(labels_ok, probs_ok, years_ok, "year")
    print("By site:")
    _subgroup_report(labels_ok, probs_ok, sites_ok, "site")
    print("By dasar:")
    _subgroup_report(labels_ok, probs_ok, dasars_ok, "dasar")

    results = dict(
        roc_auc=metrics.roc_auc,
        average_precision=metrics.average_precision,
        prevalence=metrics.prevalence,
        n=metrics.n,
        precision_at_recall_0_70=metrics.precision_at_recall(0.70),
        pr_precision=metrics.pr_precision.tolist(),
        pr_recall=metrics.pr_recall.tolist(),
        _meta=dict(
            ckpt=str(ckpt),
            inventory_json=str(inventory_json),
            n=n,
            n_excluded_unreadable=n_bad,
        ),
    )
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWritten {out_json}")

    out_probs_npz = Path(out_probs_npz)
    out_probs_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_probs_npz),
        probs=probs, labels=labels, year=years, site=sites, dasar=dasars, ok=ok,
    )
    print(f"Written {out_probs_npz}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inventory-json", type=Path, default=DEFAULT_INVENTORY)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-probs-npz", type=Path, default=DEFAULT_OUT_PROBS)
    p.add_argument("--limit", type=int, default=None,
                   help="Score only the first N rows (for a quick smoke test).")
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.inventory_json, args.ckpt, args.out_json, args.out_probs_npz,
        limit=args.limit, device=args.device)
