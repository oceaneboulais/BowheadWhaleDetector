"""Explode a balanced-eval NPZ into individual .mat files.

Each row of the NPZ becomes one .mat file containing ``SNR_gram`` (uint8,
121×104) plus the metadata fields preserved from the original database.
Files are written into a directory tree that mirrors the source layout:
    {out_dir}/{year}/Site{site}/Day_{date}T000000/{folder}/D{dasar_idx}.dir/

The original per-file time component (HHMMSS) is not stored in the NPZ, so
filenames use a zero-padded sequential index within each (date, site, dasar,
label) bucket as a surrogate timestamp.

Run:
    source /Users/oboulais/.venv_py31018/bin/activate
    PYTHONPATH=. python -m bowhead.data.npz_to_mat_files \\
        --npz  data/spectrograms_eval_200k_balanced.npz \\
        --out  /Users/oboulais/Public/Bowhead_DL_Project/Eval_200K_Balanced_50pct.dir
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import savemat

_DASAR_TO_IDX = {"A": 1, "D": 2, "G": 3}
_FOLDER = {0: "Event_sounds.dir", 1: "Manually_selected_bowhead_calls.dir"}


def export_mat_files(npz_path: Path, out_dir: Path) -> None:
    print(f"Loading {npz_path} ...")
    d = np.load(str(npz_path), allow_pickle=True)
    images     = d["images"]      # (N, 121, 104) uint8
    labels     = d["label"]
    dates      = d["date"]
    sites      = d["site"]
    dasars     = d["dasar"]
    call_types = d["call_type"]
    n = len(labels)
    print(f"  {n:,} samples")

    # Counter per (date, site, dasar, label) bucket to generate unique time idx
    bucket_counter: dict[tuple, int] = defaultdict(int)

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = 0

    for i in range(n):
        date8     = str(dates[i])        # YYYYMMDD
        site      = str(sites[i])        # '3' or '5'
        dasar     = str(dasars[i])       # 'A','D','G'
        call_type = str(call_types[i])   # '0'..'7'
        label     = int(labels[i])
        year      = date8[:4]
        yy        = date8[2:4]

        bucket = (date8, site, dasar, label)
        idx    = bucket_counter[bucket]
        bucket_counter[bucket] += 1

        dasar_idx = _DASAR_TO_IDX.get(dasar, 1)
        folder    = _FOLDER[label]
        time_str  = f"{idx:06d}"

        fname = f"S{site}{yy}{dasar}0T{date8}T{time_str}_Type{call_type}.mat"
        dest  = (out_dir / year / f"Site{site}"
                 / f"Day_{date8}T000000" / folder / f"D{dasar_idx}.dir" / fname)
        dest.parent.mkdir(parents=True, exist_ok=True)

        savemat(str(dest), {"SNR_gram": images[i]}, do_compression=True)
        written += 1

        if written % 10_000 == 0:
            rate = written / (time.time() - t0)
            eta  = (n - written) / rate / 60
            print(f"  {written:,}/{n:,}  ({rate:.0f} files/s, ETA {eta:.1f} min)")

    elapsed = time.time() - t0
    print(f"\nDone. {written:,} .mat files written to {out_dir}  ({elapsed/60:.1f} min)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explode eval NPZ into individual .mat files")
    parser.add_argument("--npz", type=Path, default=Path("data/spectrograms_eval_200k_balanced.npz"))
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory (will be created)")
    args = parser.parse_args()
    export_mat_files(args.npz, args.out)


if __name__ == "__main__":
    main()
