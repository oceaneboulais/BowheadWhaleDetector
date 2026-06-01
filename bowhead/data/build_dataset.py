"""Build the pipeline ``.npz`` from the author's two ``.mat`` databases.

Converts the per-file MATLAB spectrogram databases into the single ``.npz`` that
``SpectrogramDataset.load_npz`` (and therefore ``train_cnn``) consumes, emitting
exactly the columns in ``REQUIRED_METADATA_COLUMNS``.

Real on-disk format (verified 2026-05-30 on R3D_2024_1):
    Each ``.mat`` holds 121x104 uint8 grams ``SNR_gram`` / ``NTV_gram`` /
    ``KEtoPE_gram`` / ``Polar_gram`` plus ``bearing``, ``dF``, ``dT``,
    ``tabs_tstartt`` (MATLAB datenum) and a ``features`` struct.
    Filename: ``S{site}{yy}{dasar}0T{YYYYMMDD}T{HHMMSS}_Type{n}.mat``.

Labels come from the *source directory*, not the file:
    * Manual database -> Type 1-7 = manually-verified bowhead calls -> label 1
    * Auto   database -> Type 0    = auto-detected transients        -> label 0

Caveats baked into the metadata (documented, not faked):
    * No unique-call ID exists on disk, so ``unique_call`` is set to the coarse
      ``"{date}_{site}"`` key — the same value as the ``date_site`` grouping.
      This keeps grouped splitting leakage-safe (all DASAR views of a call on a
      given site-day stay together); it does NOT pretend to a finer UC identity.
      Default ``group_col`` should stay ``date_site``.
    * No per-file airgun flag exists, so ``is_airgun`` is all False. (The draft
      notes ~40% of Auto transients are airguns, but they are not individually
      tagged in these files.)

Run:
    PYTHONPATH=. /usr/local/bin/python3.8 -m bowhead.data.build_dataset \
        --manual-dir /Volumes/.../Unsupervised_database_Manual_100K_..._centered_16Apr2026.dir \
        --auto-dir   /Volumes/.../Unsupervised_database_Auto_100K_..._centered_16Apr2026.dir \
        --out data/spectrograms.npz
    # quick iteration: --max-per-class 5000
    # two-channel SNR+NTV (to warm-start a BOTH-mode AE): --gram SNR_gram --gram NTV_gram
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# S{site}{yy}{dasar}{0}T{YYYYMMDD}T{HHMMSS}_Type{n}
_FNAME_RE = re.compile(
    r"S(?P<site>\d)(?P<yy>\d{2})(?P<dasar>[A-G])\dT"
    r"(?P<date>\d{8})T(?P<hms>\d{6})_Type(?P<type>\d+)"
)


def _parse_name(stem: str) -> dict | None:
    m = _FNAME_RE.match(stem)
    if not m:
        return None
    g = m.groupdict()
    return {
        "site": g["site"],
        "dasar": g["dasar"],
        "date": g["date"],            # YYYYMMDD
        "call_type": g["type"],       # "0".."7"
        "datetime": g["date"] + g["hms"],
    }


def _scan(directory: Path, label: int) -> list[dict]:
    """Parse every well-named ``.mat`` in ``directory`` into a metadata row."""
    rows: list[dict] = []
    skipped = 0
    for fp in sorted(directory.glob("*.mat")):
        meta = _parse_name(fp.stem)
        if meta is None:
            skipped += 1
            continue
        meta["path"] = fp
        meta["label"] = label
        rows.append(meta)
    if skipped:
        print(f"  {directory.name}: skipped {skipped} file(s) not matching the name pattern")
    return rows


def build(
    manual_dir: Path,
    auto_dir: Path,
    out_path: Path,
    grams: list[str],
    max_per_class: int | None = None,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)

    print(f"Scanning databases ...")
    manual = _scan(manual_dir, label=1)
    auto = _scan(auto_dir, label=0)
    print(f"  manual (calls)     : {len(manual):,} files")
    print(f"  auto   (non-calls) : {len(auto):,} files")

    if max_per_class is not None:
        for rows, nm in ((manual, "manual"), (auto, "auto")):
            if len(rows) > max_per_class:
                keep = rng.choice(len(rows), size=max_per_class, replace=False)
                rows[:] = [rows[i] for i in sorted(keep)]
                print(f"  subsampled {nm} -> {len(rows):,} (max_per_class={max_per_class})")

    rows = manual + auto
    n = len(rows)
    if n == 0:
        raise RuntimeError("No usable .mat files found in either directory.")

    # Peek at the first file to fix image shape, then preallocate.
    first = loadmat(rows[0]["path"])
    for gram in grams:
        if gram not in first:
            raise KeyError(f"{rows[0]['path'].name} lacks gram {gram!r}. "
                           f"Available: {[k for k in first if not k.startswith('__')]}")
    h, w = first[grams[0]].shape
    c = len(grams)
    print(f"Image shape: {c}x{h}x{w} ({n:,} samples, grams={grams})")

    images = np.empty((n, c, h, w), dtype=np.uint8)
    keep_mask = np.ones(n, dtype=bool)

    t0 = time.time()
    for i, row in enumerate(rows):
        try:
            m = loadmat(row["path"])
            for ci, gram in enumerate(grams):
                im = m[gram]
                if im.shape != (h, w):
                    raise ValueError(f"{gram} shape {im.shape} != {(h, w)}")
                images[i, ci] = im.astype(np.uint8)
        except Exception as e:  # noqa: BLE001 - record and drop the bad row
            print(f"  WARN dropping {row['path'].name}: {e}")
            keep_mask[i] = False
        if (i + 1) % 10000 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (n - i - 1) / rate / 60
            print(f"  loaded {i + 1:,}/{n:,}  ({rate:.0f}/s, ETA {eta:.1f} min)")

    if not keep_mask.all():
        images = images[keep_mask]
        rows = [r for r, ok in zip(rows, keep_mask) if ok]
        print(f"Dropped {int((~keep_mask).sum())} unreadable file(s); {len(rows):,} remain")

    # If single-channel, store as (N, H, W) so it round-trips to the AE's 1-channel form.
    if c == 1:
        images = images[:, 0, :, :]

    site = np.array([r["site"] for r in rows])
    date = np.array([r["date"] for r in rows])
    date_site = np.array([f"{r['date']}_{r['site']}" for r in rows])

    meta = {
        "images": images,
        "label": np.array([r["label"] for r in rows], dtype=np.int64),
        "date": date,
        "site": site,
        "dasar": np.array([r["dasar"] for r in rows]),
        "call_type": np.array([r["call_type"] for r in rows]),
        # No true UC id on disk -> coarse, leakage-safe proxy (== date_site).
        "unique_call": date_site,
        # No per-file airgun tag on disk.
        "is_airgun": np.zeros(len(rows), dtype=bool),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out_path} ...")
    np.savez_compressed(out_path, **meta)

    # Quick sanity summary.
    lab = meta["label"]
    print("\nBuilt dataset:")
    print(f"  file        : {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  images      : {images.shape} {images.dtype}")
    print(f"  calls / non : {int(lab.sum()):,} / {int((lab == 0).sum()):,}")
    print(f"  date_site groups : {len(np.unique(date_site)):,}")
    print(f"  sites       : {dict(zip(*np.unique(site, return_counts=True)))}")
    print(f"  call_types  : {dict(zip(*np.unique(meta['call_type'], return_counts=True)))}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pipeline .npz from the two .mat databases")
    p.add_argument("--manual-dir", required=True, type=Path,
                   help="Manual_100K database (.dir) -> label 1 (calls)")
    p.add_argument("--auto-dir", required=True, type=Path,
                   help="Auto_100K database (.dir) -> label 0 (non-call transients)")
    p.add_argument("--out", type=Path, default=Path("data/spectrograms.npz"))
    p.add_argument("--gram", dest="grams", action="append",
                   help="Gram key to load (repeatable). Default: SNR_gram. "
                        "Use '--gram SNR_gram --gram NTV_gram' for a 2-channel build.")
    p.add_argument("--max-per-class", type=int, default=None,
                   help="Subsample this many files per class (for quick iteration)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if not a.grams:
        a.grams = ["SNR_gram"]
    return a


if __name__ == "__main__":
    args = _parse_args()
    build(args.manual_dir, args.auto_dir, args.out, args.grams,
          max_per_class=args.max_per_class, seed=args.seed)
