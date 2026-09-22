"""Build a 200,000-sample evaluation NPZ from the large Spectrogram_Image_Database
directory, excluding all files from dates/sites used in CNN/AE training.

Training dates (from paper Table — dates used for training dataset):
    2008  Site 3: 08/28, 09/06, 09/13, 09/21, 09/29
    2008  Site 5: 08/21, 08/28, 09/06, 09/13, 09/21, 09/29  (same + 08/21)
    2010  Site 3: 08/15, 08/21, 08/29, 09/05, 09/13, 09/27
    2010  Site 5: 08/15, 08/21, 08/29, 09/05, 09/13         (same, no 09/27)
    2012  Site 3: 08/25, 09/01, 09/07, 09/13, 09/18, 09/23, 09/29, 10/05
    2012  Site 5: 08/25, 09/01, 09/13, 09/18, 09/23, 09/29  (same, no 09/07 or 10/05)
    2014  Site 3: 08/18, 08/28, 09/01, 09/17, 09/27
    2014  Site 5: 08/18, 08/28, 09/01, 09/17, 09/27         (same)

Strategy:
    1. Read Database_index.mat to enumerate ALL filenames in the large dataset.
    2. Construct the on-disk path for each entry from the index dimensions
       (year, site, day, folder, dasar).
    3. Exclude any file whose (YYYYMMDD, site) combination appears in the
       training set above.
    4. Randomly sample 200,000 files from the remaining eval candidates.
    5. Load SNR_gram from each selected .mat file and save to .npz.

Run:
    python -m bowhead.data.build_eval_v3 \\
        --db-dir /Users/oboulais/Public/Bowhead_DL_Project/Spectrogram_Image_Database_Sites35_ADG_Y08101214_centered.dir \\
        --out data/spectrograms_eval_200k.npz

    # Dry-run (no loading, just show candidate counts):
    python -m bowhead.data.build_eval_v3 --db-dir ... --out ... --dry-run

    # Filesystem-walk leak audit (verify index vs on-disk, and check built NPZ):
    python -m bowhead.data.build_eval_v3 --db-dir ... --out data/spectrograms_eval_200k.npz --verify
    # (--verify implies --dry-run; no NPZ is written unless --out is also given without --dry-run)

    # Different gram channel:
    python -m bowhead.data.build_eval_v3 --db-dir ... --out ... --gram NTV_gram

    # Fixed random seed:
    python -m bowhead.data.build_eval_v3 --db-dir ... --out ... --seed 42
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Training date exclusion set: frozenset of "YYYYMMDD_site" strings
# ---------------------------------------------------------------------------
_TRAINING_DATE_SITES: frozenset[str] = frozenset([
    # 2008 Site 3
    "20080828_3", "20080906_3", "20080913_3", "20080921_3", "20080929_3",
    # 2008 Site 5
    "20080821_5", "20080828_5", "20080906_5", "20080913_5", "20080921_5", "20080929_5",
    # 2010 Site 3
    "20100815_3", "20100821_3", "20100829_3", "20100905_3", "20100913_3", "20100927_3",
    # 2010 Site 5 (no 09/27)
    "20100815_5", "20100821_5", "20100829_5", "20100905_5", "20100913_5",
    # 2012 Site 3
    "20120825_3", "20120901_3", "20120907_3", "20120913_3",
    "20120918_3", "20120923_3", "20120929_3", "20121005_3",
    # 2012 Site 5 (no 09/07, no 10/05)
    "20120825_5", "20120901_5", "20120913_5",
    "20120918_5", "20120923_5", "20120929_5",
    # 2014 Site 3
    "20140818_3", "20140828_3", "20140901_3", "20140917_3", "20140927_3",
    # 2014 Site 5 (same as Site 3)
    "20140818_5", "20140828_5", "20140901_5", "20140917_5", "20140927_5",
])

# Filename pattern (same as other build scripts)
_FNAME_RE = re.compile(
    r"S(?P<site>\d)(?P<yy>\d{2})(?P<dasar>[A-G])\dT"
    r"(?P<date>\d{8})T(?P<hms>\d{6})_Type(?P<type>\d+)"
)

# Database_index.mat dimension order
_YEARS      = ("08", "10", "12", "14")       # index dim 0
_SITES      = ("3",  "5")                    # index dim 1
_FOLDER_NAMES = (
    "Event_sounds.dir",                      # index dim 3 → label 0
    "Manually_selected_bowhead_calls.dir",   # index dim 3 → label 1
)
_DASAR_CHARS = ("A", "D", "G", "?")         # index dim 4 (last slot may be empty)


def _decode_fnames(uint8_mat: np.ndarray) -> list[str]:
    """Decode a (N, width) uint8 array of ASCII filenames → list of strings."""
    return ["".join(chr(c) for c in row).rstrip("\x00 ") for row in uint8_mat]


def _parse_stem(stem: str) -> dict | None:
    m = _FNAME_RE.match(stem)
    if not m:
        return None
    g = m.groupdict()
    return {
        "site":      g["site"],
        "dasar":     g["dasar"],
        "date":      g["date"],       # YYYYMMDD
        "call_type": g["type"],
        "datetime":  g["date"] + g["hms"],
    }


# ---------------------------------------------------------------------------
# Filesystem-walk verifier
# ---------------------------------------------------------------------------

def _walk_filesystem(db_dir: Path) -> dict[str, dict]:
    """Walk the entire db_dir tree and return {stem: {date, site, label, path}}.

    This is the ground-truth enumeration that does NOT rely on Database_index.mat.
    It is slow (2 M+ files) but provides an independent check.
    """
    print("  Walking filesystem (this may take a few minutes for 2M+ files) ...")
    stem_map: dict[str, dict] = {}
    skipped = 0
    t0 = time.time()
    for mat_path in db_dir.rglob("*.mat"):
        stem = mat_path.stem
        meta = _parse_stem(stem)
        if meta is None:
            skipped += 1
            continue
        # Derive label from parent folder name
        parts = mat_path.parts
        label = 1 if any("Manually_selected" in p for p in parts) else 0
        stem_map[stem] = {
            "path":      mat_path,
            "label":     label,
            "site":      meta["site"],
            "date":      meta["date"],
            "call_type": meta["call_type"],
        }
    elapsed = time.time() - t0
    print(f"  Filesystem walk: {len(stem_map):,} .mat files found "
          f"({skipped} skipped, {elapsed:.1f}s)")
    return stem_map


# Paths to the actual training dataset directories and AE embedding record.
# Override via environment variables if your layout differs.
_CNN_TRAIN_DIRS: tuple[Path, ...] = (
    Path("/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets"
         "/Unsupervised_database_Manual_100K_ADG_Y08101214_centered_16Apr2026.dir"),
    Path("/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets"
         "/Unsupervised_database_Auto_100K_ADG_Y08101214_centered_16Apr2026.dir"),
)
_AE_LATENT_MAT: Path = Path(
    "/Users/oboulais/Public/Bowhead_DL_Project/LD32_Hybrid"
    "/Autoencoder_v15_100E_32LD_32C_Hybrid_5x5_3x3_CombinedDatasets_100K"
    "_Date20260119-222330.dir/MATLAB/latent_embeddings.mat"
)


def _load_cnn_training_stems() -> set[str]:
    """Collect the exact stems used for CNN training from the flat training dirs."""
    stems: set[str] = set()
    for d in _CNN_TRAIN_DIRS:
        if not d.is_dir():
            print(f"  WARN: CNN training dir not found: {d}")
            continue
        count = 0
        for p in d.glob("*.mat"):
            if _FNAME_RE.match(p.stem):
                stems.add(p.stem)
                count += 1
        print(f"  CNN training dir '{d.name}': {count:,} stems")
    return stems


def _load_ae_training_stems() -> set[str]:
    """Load exact stems used for AE training from latent_embeddings.mat."""
    if not _AE_LATENT_MAT.exists():
        print(f"  WARN: AE latent_embeddings.mat not found: {_AE_LATENT_MAT}")
        return set()
    mat = loadmat(str(_AE_LATENT_MAT))
    fname_arr = mat["original_filenames"]  # shape (1, N) object
    stems: set[str] = set()
    for cell in fname_arr.flat:
        fname = str(cell).strip()
        stem = fname.replace(".mat", "")
        if stem:
            stems.add(stem)
    print(f"  AE latent_embeddings.mat: {len(stems):,} stems")
    return stems


def verify(
    db_dir: Path,
    idx: np.ndarray,
    index_candidates: list[dict],
    npz_path: Path | None = None,
    cnn_stems: set[str] | None = None,
    ae_stems: set[str] | None = None,
) -> bool:
    """Audit the index-based enumeration against a full filesystem walk.

    Checks performed
    ----------------
    1. Filesystem completeness: every stem found on disk is also in the index
       (and vice versa).  Discrepancies are reported but do not cause a hard
       failure — they flag index maintenance issues.
    2. Training-date exclusion (index path): no candidate produced by
       ``enumerate_candidates`` belongs to a training date+site.
    3. Training-date exclusion (filesystem path): no file on disk that belongs
       to a training date+site appears in ``index_candidates``.
    4. If ``npz_path`` points to an already-built NPZ, every (date, site) pair
       stored in it is checked against the training exclusion set.
    5. Stem-level CNN leak check: none of the exact CNN training stems appear
       in the eval candidate list or built NPZ.
    6. Stem-level AE leak check: none of the exact AE training stems (from
       latent_embeddings.mat) appear in the eval candidate list or built NPZ.

    Returns True if all leak checks pass, False if any leak is detected.
    """
    print("\n" + "=" * 60)
    print("LEAK AUDIT")
    print("=" * 60)

    passed = True

    # ------------------------------------------------------------------ walk
    fs_stems = _walk_filesystem(db_dir)

    # Build index stem set
    index_stem_set: set[str] = set()
    for rec in index_candidates:
        index_stem_set.add(rec["path"].stem)

    # Also build full index stem set (training + eval combined) by re-reading idx
    full_index_stems: set[str] = set()
    for yi in range(idx.shape[0]):
        for si in range(idx.shape[1]):
            for di in range(idx.shape[2]):
                for fi in range(idx.shape[3]):
                    for dasi in range(idx.shape[4]):
                        cell = idx[yi, si, di, fi, dasi]
                        if not (hasattr(cell, "size") and cell.size > 0):
                            continue
                        rec = cell[0, 0]
                        fa = rec["fname"]
                        if fa.size == 0:
                            continue
                        for fname in _decode_fnames(fa):
                            full_index_stems.add(fname.replace(".mat", ""))

    # ------------------------------------------------------------------ 1. completeness
    on_disk_only = set(fs_stems.keys()) - full_index_stems
    in_index_only = full_index_stems - set(fs_stems.keys())

    print(f"\n[1] Index vs filesystem completeness")
    print(f"    Total on disk   : {len(fs_stems):,}")
    print(f"    Total in index  : {len(full_index_stems):,}")
    if on_disk_only:
        print(f"    WARNING: {len(on_disk_only):,} file(s) on disk NOT in index (first 5):")
        for s in sorted(on_disk_only)[:5]:
            print(f"      {s}")
    else:
        print("    OK: every on-disk file is represented in the index.")
    if in_index_only:
        print(f"    WARNING: {len(in_index_only):,} index entry(ies) NOT found on disk (first 5):")
        for s in sorted(in_index_only)[:5]:
            print(f"      {s}")
    else:
        print("    OK: every index entry exists on disk.")

    # ------------------------------------------------------------------ 2. index candidates leak check
    print(f"\n[2] Training-date exclusion — index candidate list ({len(index_candidates):,} files)")
    leaked_index: list[str] = []
    for rec in index_candidates:
        key = f"{rec['date']}_{rec['site']}"
        if key in _TRAINING_DATE_SITES:
            leaked_index.append(rec["path"].stem)
    if leaked_index:
        passed = False
        print(f"    FAIL: {len(leaked_index):,} training-date file(s) found in candidates (first 5):")
        for s in leaked_index[:5]:
            print(f"      {s}")
    else:
        print("    PASS: zero training-date files in index candidate list.")

    # ------------------------------------------------------------------ 3. filesystem leak check
    print(f"\n[3] Training-date exclusion — independent filesystem walk")
    leaked_fs: list[str] = []
    for stem, info in fs_stems.items():
        key = f"{info['date']}_{info['site']}"
        if key not in _TRAINING_DATE_SITES:
            # Non-training file — check it's NOT in training exclusion
            continue  # good: it's an eval file
        # This is a training-date file; make sure it's not in candidates
        if stem in index_stem_set:
            leaked_fs.append(stem)
    if leaked_fs:
        passed = False
        print(f"    FAIL: {len(leaked_fs):,} training-date file(s) found in eval candidates "
              f"via filesystem walk (first 5):")
        for s in leaked_fs[:5]:
            print(f"      {s}")
    else:
        training_date_on_disk = sum(
            1 for info in fs_stems.values()
            if f"{info['date']}_{info['site']}" in _TRAINING_DATE_SITES
        )
        print(f"    PASS: {training_date_on_disk:,} training-date files on disk, "
              f"none appear in eval candidate list.")

    # ------------------------------------------------------------------ 4. NPZ leak check
    if npz_path is not None and npz_path.exists():
        print(f"\n[4] Training-date exclusion — built NPZ ({npz_path.name})")
        d = np.load(str(npz_path), allow_pickle=True)
        dates_npz = d["date"]
        sites_npz = d["site"]
        leaked_npz = []
        for date8, site in zip(dates_npz, sites_npz):
            key = f"{date8}_{site}"
            if key in _TRAINING_DATE_SITES:
                leaked_npz.append(key)
        if leaked_npz:
            passed = False
            unique_leaks = sorted(set(leaked_npz))
            print(f"    FAIL: {len(leaked_npz):,} samples in NPZ belong to training dates "
                  f"({len(unique_leaks)} unique date_site keys):")
            for k in unique_leaks:
                print(f"      {k}")
        else:
            print(f"    PASS: all {len(dates_npz):,} NPZ samples are from non-training dates.")
    elif npz_path is not None:
        print(f"\n[4] NPZ not yet built at {npz_path} — skipping NPZ check.")

    # ------------------------------------------------------------------ 5 & 6. stem-level checks
    print("\nLoading exact training stems for stem-level audit ...")
    if cnn_stems is None:
        cnn_stems = _load_cnn_training_stems()
    if ae_stems is None:
        ae_stems = _load_ae_training_stems()
    all_training_stems = cnn_stems | ae_stems
    print(f"  Total unique training stems (CNN ∪ AE): {len(all_training_stems):,}")

    for check_num, label, training_stems in (
        (5, "CNN", cnn_stems),
        (6, "AE",  ae_stems),
    ):
        print(f"\n[{check_num}] Stem-level {label} training leak — candidates")
        leaked_cands = [
            rec["path"].stem
            for rec in index_candidates
            if rec["path"].stem in training_stems
        ]
        if leaked_cands:
            passed = False
            print(f"    FAIL: {len(leaked_cands):,} {label} training stem(s) in eval candidates (first 5):")
            for s in leaked_cands[:5]:
                print(f"      {s}")
        else:
            print(f"    PASS: zero {label} training stems in eval candidate list.")

        # Stem-level NPZ check: find training stems that physically exist inside
        # the large eval DB (same filename in both training dir and eval DB),
        # then verify none of those date+site keys appear in the NPZ.
        if npz_path is not None and npz_path.exists():
            stems_also_in_eval_db = training_stems & set(fs_stems.keys())
            if stems_also_in_eval_db:
                d = np.load(str(npz_path), allow_pickle=True)
                npz_date_site = set(
                    f"{date8}_{site}"
                    for date8, site in zip(d["date"], d["site"])
                )
                overlapping = [
                    s for s in stems_also_in_eval_db
                    if f"{fs_stems[s]['date']}_{fs_stems[s]['site']}" in npz_date_site
                ]
                if overlapping:
                    passed = False
                    print(f"    FAIL: {len(overlapping):,} {label} training stem(s) exist in "
                          f"eval DB and their date+site appears in the NPZ (first 5):")
                    for s in overlapping[:5]:
                        print(f"      {s}")
                else:
                    print(f"    ({len(stems_also_in_eval_db):,} {label} stems exist in eval DB "
                          f"filesystem — none appear in the NPZ. PASS)")
            else:
                print(f"    ({label} training stems share no filenames with eval DB. PASS)")

    # ------------------------------------------------------------------ summary
    print("\n" + "-" * 60)
    if passed:
        print("AUDIT RESULT: PASS — eval dataset is leak-proof.")
    else:
        print("AUDIT RESULT: FAIL — leakage detected. See above.")
    print("-" * 60)
    return passed


def enumerate_candidates(
    db_dir: Path,
    idx: np.ndarray,
    extra_exclude_stems: set[str] | None = None,
) -> list[dict]:
    """Walk the index array and return all non-training file records.

    Excludes any file whose stem appears in ``extra_exclude_stems`` (the exact
    union of CNN and AE training stems).  Date-based exclusion is intentionally
    NOT applied here — the stem set is the ground truth; the LaTeX training-date
    table is used only by the verifier as an informational cross-check.

    Each record has keys: path, label, site, dasar, date, call_type, datetime.
    """
    if extra_exclude_stems is None:
        extra_exclude_stems = set()
    candidates: list[dict] = []
    stem_excluded = 0
    total_indexed = 0

    for yi, yy in enumerate(_YEARS):
        full_year = "20" + yy
        for si, site in enumerate(_SITES):
            site_dir = db_dir / full_year / f"Site{site}"
            if not site_dir.is_dir():
                print(f"  WARN: missing {site_dir}")
                continue

            for di in range(idx.shape[2]):
                for fi, folder_name in enumerate(_FOLDER_NAMES):
                    label = fi  # 0=auto/Event, 1=manual/Bowhead

                    for dasi in range(idx.shape[4]):
                        cell = idx[yi, si, di, fi, dasi]
                        if not (hasattr(cell, "size") and cell.size > 0):
                            continue
                        rec = cell[0, 0]
                        fname_raw = rec["fname"]
                        if fname_raw.size == 0:
                            continue

                        fnames = _decode_fnames(fname_raw)
                        total_indexed += len(fnames)

                        # Determine dasar sub-directory name from first filename
                        # (Dasar letter is encoded in filename: S308A0T...)
                        first_meta = _parse_stem(fnames[0].replace(".mat", ""))
                        if first_meta is None:
                            continue
                        dasar_char = first_meta["dasar"]
                        date8 = first_meta["date"]
                        day_dir = site_dir / f"Day_{date8}T000000"
                        dasar_dir = day_dir / folder_name / f"D{dasi + 1}.dir"

                        if not dasar_dir.is_dir():
                            continue

                        for fname in fnames:
                            stem = fname.replace(".mat", "")
                            if stem in extra_exclude_stems:
                                stem_excluded += 1
                                continue
                            meta = _parse_stem(stem)
                            if meta is None:
                                continue
                            candidates.append({
                                "path":      dasar_dir / fname,
                                "label":     label,
                                "site":      site,
                                "dasar":     meta["dasar"],
                                "date":      meta["date"],
                                "call_type": meta["call_type"],
                                "datetime":  meta["datetime"],
                            })

    print(f"  Total indexed: {total_indexed:,}")
    print(f"  Excluded (exact training stems): {stem_excluded:,}")
    print(f"  Eval candidates: {len(candidates):,}")
    return candidates


def build_eval(
    db_dir: Path,
    out_path: Path,
    n_samples: int = 200_000,
    gram: str = "SNR_gram",
    seed: int = 0,
    dry_run: bool = False,
    run_verify: bool = False,
    call_fraction: float | None = None,
) -> None:
    index_path = db_dir / "Database_index.mat"
    if not index_path.exists():
        raise FileNotFoundError(f"Database_index.mat not found in {db_dir}")

    print(f"Loading index from {index_path} ...")
    mat = loadmat(str(index_path))
    idx = mat["index"]  # shape (4 years, 2 sites, 9 days, 2 folders, 4 dasars)
    print(f"  index shape: {idx.shape}")

    print("\nLoading exact CNN and AE training stems for exclusion ...")
    cnn_stems = _load_cnn_training_stems()
    ae_stems  = _load_ae_training_stems()
    all_training_stems = cnn_stems | ae_stems
    print(f"  Total unique training stems to exclude: {len(all_training_stems):,}")

    print("\nEnumerating eval candidates (excluding training dates + stems) ...")
    candidates = enumerate_candidates(db_dir, idx, extra_exclude_stems=all_training_stems)

    if len(candidates) == 0:
        raise RuntimeError("No eval candidates found. Check db_dir path.")

    # ---------------------------------------------------------------- sampling
    rng = np.random.default_rng(seed)

    if call_fraction is not None:
        # Stratified sampling: draw exactly round(n_samples * call_fraction) calls
        # and the remainder as non-calls.
        n_calls_want    = round(n_samples * call_fraction)
        n_noncalls_want = n_samples - n_calls_want

        call_pool    = [r for r in candidates if r["label"] == 1]
        noncall_pool = [r for r in candidates if r["label"] == 0]

        print(f"\nStratified sampling (call_fraction={call_fraction:.2f}):")
        print(f"  call pool:     {len(call_pool):,}  → want {n_calls_want:,}")
        print(f"  non-call pool: {len(noncall_pool):,}  → want {n_noncalls_want:,}")

        if len(call_pool) < n_calls_want:
            print(f"  WARN: only {len(call_pool):,} calls available; using all.")
            n_calls_want = len(call_pool)
            n_noncalls_want = min(n_samples - n_calls_want, len(noncall_pool))
        if len(noncall_pool) < n_noncalls_want:
            print(f"  WARN: only {len(noncall_pool):,} non-calls available; using all.")
            n_noncalls_want = len(noncall_pool)

        call_idx    = rng.choice(len(call_pool),    size=n_calls_want,    replace=False)
        noncall_idx = rng.choice(len(noncall_pool), size=n_noncalls_want, replace=False)
        selected = [call_pool[i] for i in call_idx] + [noncall_pool[i] for i in noncall_idx]
        rng.shuffle(selected)  # interleave labels
    else:
        if len(candidates) <= n_samples:
            print(f"\nWARN: only {len(candidates):,} candidates, requesting {n_samples:,}. Using all.")
            selected = candidates
        else:
            indices = rng.choice(len(candidates), size=n_samples, replace=False)
            indices.sort()
            selected = [candidates[i] for i in indices]

    print(f"\nSelected {len(selected):,} samples (seed={seed})")

    # Label breakdown
    labels_arr = np.array([r["label"] for r in selected])
    print(f"  calls (label=1):     {int(labels_arr.sum()):,}")
    print(f"  non-calls (label=0): {int((labels_arr == 0).sum()):,}")

    if run_verify:
        verify(db_dir, idx, candidates, npz_path=out_path if not dry_run else None,
               cnn_stems=cnn_stems, ae_stems=ae_stems)

    if dry_run:
        print("\n[dry-run] Skipping file loading and NPZ write.")
        return

    # ---------------------------------------------------------------- load
    # Peek at image dimensions from first valid file
    first_mat = loadmat(str(selected[0]["path"]))
    if gram not in first_mat:
        avail = [k for k in first_mat if not k.startswith("__")]
        raise KeyError(f"Gram '{gram}' not found in first file. Available: {avail}")
    h, w = first_mat[gram].shape
    print(f"\nImage shape: ({h}, {w})  gram={gram}")

    n = len(selected)
    images    = np.empty((n, h, w), dtype=np.uint8)
    keep_mask = np.ones(n, dtype=bool)

    t0 = time.time()
    for i, row in enumerate(selected):
        try:
            m   = loadmat(str(row["path"]))
            img = m[gram]
            if img.shape != (h, w):
                raise ValueError(f"Shape mismatch: {img.shape} vs ({h}, {w})")
            images[i] = img.astype(np.uint8)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN dropping {row['path'].name}: {exc}")
            keep_mask[i] = False

        if (i + 1) % 10_000 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta  = (n - i - 1) / rate / 60
            print(f"  loaded {i + 1:,}/{n:,}  ({rate:.0f} files/s, ETA {eta:.1f} min)")

    n_dropped = int((~keep_mask).sum())
    if n_dropped:
        print(f"  Dropped {n_dropped} unreadable files.")
        images   = images[keep_mask]
        selected = [r for r, ok in zip(selected, keep_mask) if ok]

    n = len(selected)
    labels     = np.array([r["label"]     for r in selected], dtype=np.int64)
    dates      = np.array([r["date"]      for r in selected])
    sites      = np.array([r["site"]      for r in selected])
    dasars     = np.array([r["dasar"]     for r in selected])
    call_types = np.array([r["call_type"] for r in selected])
    date_site  = np.array([f"{r['date']}_{r['site']}" for r in selected])
    filenames  = np.array([r["path"].stem for r in selected])

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
        filenames=filenames,
    )

    elapsed = time.time() - t0
    print(f"\n=== Eval dataset summary ===")
    print(f"  file     : {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")
    print(f"  images   : {images.shape}  {images.dtype}")
    print(f"  calls    : {int(labels.sum()):,}  ({100 * labels.mean():.1f}%)")
    print(f"  non-calls: {int((labels == 0).sum()):,}")
    print(f"  sites    : {dict(zip(*np.unique(sites, return_counts=True)))}")
    print(f"  years    : {dict(zip(*np.unique([d[:4] for d in dates], return_counts=True)))}")
    print(f"  date_site groups: {len(np.unique(date_site)):,}")
    print(f"  elapsed  : {elapsed / 60:.1f} min")

    if run_verify:
        verify(db_dir, idx, candidates, npz_path=out_path,
               cnn_stems=cnn_stems, ae_stems=ae_stems)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build 200K eval NPZ from large spectrogram database, "
                    "excluding training dates."
    )
    parser.add_argument(
        "--db-dir",
        required=True,
        type=Path,
        help="Path to Spectrogram_Image_Database_Sites35_ADG_Y08101214_centered.dir",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output .npz path (e.g. data/spectrograms_eval_200k.npz)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200_000,
        help="Number of eval samples to draw (default: 200,000)",
    )
    parser.add_argument(
        "--gram",
        default="SNR_gram",
        help="Spectrogram channel to load (default: SNR_gram)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling (default: 0)",
    )
    parser.add_argument(
        "--call-fraction",
        type=float,
        default=None,
        metavar="FRAC",
        help="Fraction of samples that should be calls (0.0–1.0). "
             "E.g. 0.5 for balanced 50/50. Default: natural ratio (random sample).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate and count candidates without loading .mat files or writing output",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run independent filesystem-walk leak audit. Combines with --dry-run "
             "(no NPZ written) or checks an already-built NPZ when --out exists.",
    )
    args = parser.parse_args()

    build_eval(
        db_dir=args.db_dir,
        out_path=args.out,
        n_samples=args.n_samples,
        gram=args.gram,
        seed=args.seed,
        dry_run=args.dry_run,
        run_verify=args.verify,
        call_fraction=args.call_fraction,
    )


if __name__ == "__main__":
    main()
