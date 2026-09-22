"""Copy the exact original .mat files for the balanced eval selection into one flat dir.

Re-runs the identical deterministic selection used to build
``spectrograms_eval_200k_balanced.npz`` (seed=0, call_fraction=0.5, n=200,000),
then copies each source .mat directly — preserving every original field
(SNR_gram, NTV_gram, KEtoPE_gram, Polar_gram, bearing, dF, dT, tabs_tstartt,
features) without modification.

Output: one flat directory, 200,000 .mat files, no subdirectories.

Run:
    source /Users/oboulais/.venv_py31018/bin/activate
    PYTHONPATH=. python bowhead/data/copy_balanced_eval_flat.py \\
        --db-dir /Users/oboulais/Public/Bowhead_DL_Project/Spectrogram_Image_Database_Sites35_ADG_Y08101214_centered.dir \\
        --out    /Users/oboulais/Public/Bowhead_DL_Project/Eval_200K_Balanced_50pct.dir
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bowhead.data.build_eval_v3 import (
    _load_cnn_training_stems,
    _load_ae_training_stems,
    enumerate_candidates,
)


def copy_flat(db_dir: Path, out_dir: Path,
              n_samples: int = 200_000,
              call_fraction: float = 0.5,
              seed: int = 0) -> None:

    print(f"Loading index ...")
    idx = loadmat(str(db_dir / "Database_index.mat"))["index"]

    print("Loading training stems for exclusion ...")
    cnn_stems = _load_cnn_training_stems()
    ae_stems  = _load_ae_training_stems()
    all_stems = cnn_stems | ae_stems
    print(f"  Total training stems excluded: {len(all_stems):,}")

    print("Enumerating eval candidates ...")
    candidates = enumerate_candidates(db_dir, idx, extra_exclude_stems=all_stems)

    # Reproduce identical stratified sample
    rng = np.random.default_rng(seed)
    n_calls_want    = round(n_samples * call_fraction)
    n_noncalls_want = n_samples - n_calls_want
    call_pool    = [r for r in candidates if r["label"] == 1]
    noncall_pool = [r for r in candidates if r["label"] == 0]
    call_idx    = rng.choice(len(call_pool),    size=n_calls_want,    replace=False)
    noncall_idx = rng.choice(len(noncall_pool), size=n_noncalls_want, replace=False)
    selected = [call_pool[i] for i in call_idx] + [noncall_pool[i] for i in noncall_idx]
    print(f"Selected {len(selected):,} files  "
          f"(calls={n_calls_want:,}  non-calls={n_noncalls_want:,})")

    # Verify uniqueness before touching disk
    names = [r["path"].name for r in selected]
    dupes = len(names) - len(set(names))
    if dupes:
        raise RuntimeError(f"{dupes} duplicate filenames in selection — cannot create flat dir.")

    # Wipe and recreate flat output directory
    if out_dir.exists():
        print(f"Removing existing {out_dir} ...")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    print(f"Copying to {out_dir} ...")

    t0 = time.time()
    missing = 0
    for i, rec in enumerate(selected):
        src = rec["path"]
        dst = out_dir / src.name
        if not src.exists():
            print(f"  WARN missing source: {src}")
            missing += 1
            continue
        shutil.copy2(src, dst)   # copy2 preserves metadata timestamps

        if (i + 1) % 10_000 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta  = (len(selected) - i - 1) / rate / 60
            print(f"  {i+1:,}/{len(selected):,}  ({rate:.0f} files/s  ETA {eta:.1f} min)")

    elapsed = time.time() - t0
    written = len(selected) - missing
    print(f"\nDone.  {written:,} files copied to {out_dir}  ({elapsed/60:.1f} min)")
    if missing:
        print(f"WARNING: {missing} source files were missing on disk.")

    # Spot-check: verify one copied file has all expected original fields
    print("\nSpot-checking a copied file ...")
    sample_dst = out_dir / selected[0]["path"].name
    m = loadmat(str(sample_dst))
    fields = [k for k in m if not k.startswith("__")]
    expected = {"SNR_gram", "NTV_gram", "KEtoPE_gram", "Polar_gram",
                "bearing", "dF", "dT", "tabs_tstartt", "features"}
    present  = set(fields)
    missing_fields = expected - present
    if missing_fields:
        print(f"  WARNING: missing fields in spot-check file: {missing_fields}")
    else:
        print(f"  OK — all original fields present: {sorted(present)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db-dir", type=Path, required=True)
    p.add_argument("--out",    type=Path, required=True)
    p.add_argument("--seed",   type=int, default=0)
    args = p.parse_args()
    copy_flat(args.db_dir, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
