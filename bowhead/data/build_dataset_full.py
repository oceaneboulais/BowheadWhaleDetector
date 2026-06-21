"""Build a deduplicated NPZ dataset from ALL available training directories.

Scans 4 training .dir directories (Evaluation_200K is excluded — kept as
held-out eval only), deduplicates by filename stem (priority order below),
and writes the combined .npz.

Priority order (first occurrence wins for dedup):
  1. Manual_100K_ADG_16Apr2026       → label 1 (manually verified calls)
  2. Auto_100K_ADG_16Apr2026         → label 0 (auto, no airguns)
  3. Manual_100K_Y08101214_centered  → label 1 (older manual calls)
  4. AutoWithAirguns_100K            → label 0 (auto, may include airguns)

Dedup criterion: filename stem (without .mat extension) is globally unique.
Files with the same stem in lower-priority dirs are silently dropped.

Expected unique counts (from prior inventory run):
  Manual_100K_ADG      → 98,933
  Auto_100K_ADG        → 100,723
  Manual_100K_older    → 58,594
  AutoWithAirguns_100K → 94,622
  TOTAL unique         → 352,872 training files

(If --include-eval flag is set, Evaluation_200K files are also included in the
AE NPZ — valid because the AE is trained unsupervised with no labels used —
but this is NOT recommended since that dataset is the CNN test/eval partition.)

Run:
    python -m bowhead.data.build_dataset_full \\
        --base-dir /Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets \\
        --out data/spectrograms_all_training.npz

    # Include Evaluation_200K in AE pretraining (unsupervised is OK):
    python -m bowhead.data.build_dataset_full \\
        --base-dir ... --out data/spectrograms_all.npz --include-eval
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# Filename pattern
_FNAME_RE = re.compile(
    r"S(?P<site>\d)(?P<yy>\d{2})(?P<dasar>[A-G])\dT"
    r"(?P<date>\d{8})T(?P<hms>\d{6})_Type(?P<type>\d+)"
)

# Priority order: (dir_glob_fragment, label)
_PRIORITY_DIRS = [
    ("Manual_100K_ADG_Y08101214_centered_16Apr2026", 1),
    ("Auto_100K_ADG_Y08101214_centered_16Apr2026",   0),
    ("Manual_100K_Y08101214_centered",               1),   # older, no ADG suffix
    ("AutoWithAirguns_100K_Y08101214_centered",      0),
]
_EVAL_DIR_FRAG = "Evaluation_200K_8Auto1Manual_ADG_Y08101214_centered_06May2026"


def _find_dir(base: Path, fragment: str) -> Path | None:
    """Return the first .dir entry whose name contains ``fragment``."""
    for d in sorted(base.iterdir()):
        if fragment in d.name and d.is_dir():
            return d
    return None


def _parse_name(stem: str) -> dict | None:
    m = _FNAME_RE.match(stem)
    if not m:
        return None
    g = m.groupdict()
    return {
        "site":      g["site"],
        "dasar":     g["dasar"],
        "date":      g["date"],
        "call_type": g["type"],
        "datetime":  g["date"] + g["hms"],
    }


def _scan_dir(directory: Path, label: int) -> list[dict]:
    rows: list[dict] = []
    skipped = 0
    for fp in sorted(directory.glob("*.mat")):
        meta = _parse_name(fp.stem)
        if meta is None:
            skipped += 1
            continue
        meta["path"] = fp
        meta["label"] = label
        meta["stem"]  = fp.stem
        rows.append(meta)
    if skipped:
        print(f"    skipped {skipped} unmatched filename(s)")
    return rows


def build_full(
    base_dir: Path,
    out_path: Path,
    include_eval: bool = False,
    gram: str = "SNR_gram",
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ find dirs
    priority_dirs = []
    for frag, lbl in _PRIORITY_DIRS:
        d = _find_dir(base_dir, frag)
        if d is None:
            print(f"  WARNING: directory fragment '{frag}' not found in {base_dir}")
        else:
            priority_dirs.append((d, lbl))
            print(f"  Found ({lbl}): {d.name}")

    if include_eval:
        ev_dir = _find_dir(base_dir, _EVAL_DIR_FRAG)
        if ev_dir is None:
            print(f"  WARNING: eval dir fragment '{_EVAL_DIR_FRAG}' not found")
        else:
            # Eval dir has both call and non-call files; use Type parse for label:
            # Type 0 → non-call (label=0), Type 1-7 → call (label=1)
            priority_dirs.append((ev_dir, -1))   # -1 = use type-based label
            print(f"  Found (eval/mixed): {ev_dir.name}")

    # ------------------------------------------------------------------ deduplicate
    print("\nScanning and deduplicating ...")
    seen_stems: set[str] = set()
    all_rows: list[dict] = []

    for dir_path, lbl in priority_dirs:
        print(f"  Scanning {dir_path.name} ...")
        rows = _scan_dir(dir_path, lbl)
        added = 0
        for row in rows:
            if row["stem"] in seen_stems:
                continue
            seen_stems.add(row["stem"])
            if lbl == -1:   # eval dir: derive label from call_type
                row["label"] = 0 if row["call_type"] == "0" else 1
            all_rows.append(row)
            added += 1
        print(f"    {len(rows):,} files → {added:,} unique (total so far: {len(all_rows):,})")

    n = len(all_rows)
    print(f"\nTotal unique files: {n:,}")
    if n == 0:
        raise RuntimeError("No files found. Check base-dir path.")

    # ------------------------------------------------------------------ load images
    first_mat = loadmat(all_rows[0]["path"])
    if gram not in first_mat:
        avail = [k for k in first_mat if not k.startswith("__")]
        raise KeyError(f"Gram '{gram}' not in {all_rows[0]['path'].name}. Available: {avail}")
    h, w = first_mat[gram].shape
    print(f"Image shape per sample: ({h}, {w})  gram={gram}")

    images = np.empty((n, h, w), dtype=np.uint8)
    keep_mask = np.ones(n, dtype=bool)

    t0 = time.time()
    for i, row in enumerate(all_rows):
        try:
            m = loadmat(row["path"])
            im = m[gram]
            if im.shape != (h, w):
                raise ValueError(f"Shape mismatch: {im.shape} vs ({h},{w})")
            images[i] = im.astype(np.uint8)
        except Exception as e:      # noqa: BLE001
            print(f"  WARN dropping {row['path'].name}: {e}")
            keep_mask[i] = False

        if (i + 1) % 20_000 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta  = (n - i - 1) / rate / 60
            print(f"  loaded {i+1:,}/{n:,}  ({rate:.0f}/s, ETA {eta:.1f} min)")

    n_dropped = int((~keep_mask).sum())
    if n_dropped:
        images    = images[keep_mask]
        all_rows  = [r for r, ok in zip(all_rows, keep_mask) if ok]
        print(f"Dropped {n_dropped} unreadable files; {len(all_rows):,} remain")

    n = len(all_rows)

    labels     = np.array([r["label"]    for r in all_rows], dtype=np.int64)
    dates      = np.array([r["date"]     for r in all_rows])
    sites      = np.array([r["site"]     for r in all_rows])
    dasars     = np.array([r["dasar"]    for r in all_rows])
    call_types = np.array([r["call_type"] for r in all_rows])
    date_site  = np.array([f"{r['date']}_{r['site']}" for r in all_rows])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {out_path} ...")
    np.savez_compressed(
        out_path,
        images=images,
        label=labels,
        date=dates,
        site=sites,
        dasar=dasars,
        call_type=call_types,
        unique_call=date_site,
        is_airgun=np.zeros(n, dtype=bool),
    )

    # ------------------------------------------------------------------ summary
    print("\n=== Dataset summary ===")
    print(f"  file    : {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  images  : {images.shape}  {images.dtype}")
    print(f"  calls   : {int(labels.sum()):,}  ({100*labels.mean():.1f}%)")
    print(f"  non-calls: {int((labels==0).sum()):,}")
    print(f"  sites   : {dict(zip(*np.unique(sites, return_counts=True)))}")
    print(f"  years   : {dict(zip(*np.unique([d[:4] for d in dates], return_counts=True)))}")
    print(f"  date_site groups: {len(np.unique(date_site)):,}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build full deduplicated training NPZ")
    p.add_argument("--base-dir",    type=Path,
                   default=Path("/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets"),
                   help="Directory containing the .dir archives")
    p.add_argument("--out",         type=Path, default=Path("data/spectrograms_all_training.npz"))
    p.add_argument("--gram",        default="SNR_gram",
                   help="Which gram to extract (default: SNR_gram)")
    p.add_argument("--include-eval", action="store_true",
                   help="Also include Evaluation_200K files (valid for unsupervised AE, "
                        "but kept as separate eval for CNN)")
    p.add_argument("--seed",        type=int, default=0)
    args = p.parse_args()

    build_full(
        base_dir    = args.base_dir,
        out_path    = args.out,
        include_eval= args.include_eval,
        gram        = args.gram,
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()
