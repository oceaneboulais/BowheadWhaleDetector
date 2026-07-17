"""Build a leakage-free evaluation dataset (Evaluation_v2), guaranteed to be
group-disjoint from the existing 100K training draw, by sampling directly
from the full ~1.2M-file master call database.

WHY THIS SCRIPT EXISTS
-----------------------
The original "independent" evaluation set (Evaluation_200K, scored by
``bowhead/eval/score_eval_dataset.py``) was found to overlap the 100K
training draw: 9,218 of 22,181 annotated eval samples (41.6%) turned out to
also be present in training. Both sets had been drawn independently from the
same master database without ever checking group- or file-level disjointness
against each other. This script fixes that by construction:

  1. Load the existing training NPZ (``data/spectrograms_100k_matched.npz``)
     and compute the set of ``date_site`` groups it uses. ``date_site`` is
     the *coarsest* safe grouping key available in this codebase (no true
     per-call UC/event ID is stored anywhere on disk -- see
     ``bowhead/data/splits.py: make_date_site_group``), so it is the
     conservative unit of exclusion: if a date+site combination contributed
     even one file to training, every remaining file that shares that
     date+site is excluded from eligibility for this new eval set, even if
     it comes from a different DASAR channel or a different call.
  2. Optionally (recommended, more precise): also exclude by *exact
     filename stem* if you can point ``--train-source-dirs`` at the actual
     Manual_100K / Auto_100K directories used to build the training NPZ.
     This is defense-in-depth on top of (1), not a replacement for it.
  3. Scan the full master database recursively for every ``.mat`` file,
     parse its metadata from the filename, and keep only files in
     eligible (non-excluded) groups.
  4. Sample exactly ``--n-manual`` manually-annotated calls (Type 1-7) and
     ``--n-auto`` unlabeled/auto transients (Type 0) from the eligible pool,
     stratified proportionally by date_site group -- but WITHOUT replacement
     and WITHOUT ever padding a sparse group back up to some quota. If a
     group has fewer eligible calls than its proportional share, it
     contributes all of what it has and the shortfall is redistributed to
     other groups. This operationalizes the policy agreed with your
     advisor: zero replication is non-negotiable; perfect proportional
     coverage yields to that constraint when the two conflict, and any
     resulting under-representation is written to the manifest rather than
     silently patched over by re-using calls.
  5. Computationally verify -- not just assume -- that the resulting eval
     set shares zero date_site groups (and zero filenames, if step 2 ran)
     with the training set, before writing any output.

USAGE
-----
Fast, filenames-only dry run first (no images loaded, tells you whether
50K manual + 150K auto is even feasible under group-exclusivity):

    python -m bowhead.data.build_eval_v2 \\
        --base-dir /Volumes/<your-drive>/BCB_Whale_Datasets \\
        --train-npz data/spectrograms_100k_matched.npz \\
        --out data/spectrograms_eval_v2.npz \\
        --manifest runs/eval_v2_manifest.json \\
        --dry-run

Full run (loads images, writes the NPZ):

    python -m bowhead.data.build_eval_v2 \\
        --base-dir /Volumes/<your-drive>/BCB_Whale_Datasets \\
        --train-npz data/spectrograms_100k_matched.npz \\
        --train-source-dirs \\
            /Volumes/<your-drive>/.../Manual_100K_ADG_Y08101214_centered_16Apr2026.dir \\
            /Volumes/<your-drive>/.../Auto_100K_ADG_Y08101214_centered_16Apr2026.dir \\
        --out data/spectrograms_eval_v2.npz \\
        --manifest runs/eval_v2_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_FNAME_RE = re.compile(
    r"S(?P<site>\d)(?P<yy>\d{2})(?P<dasar>[A-G])\dT"
    r"(?P<date>\d{8})T(?P<hms>\d{6})_Type(?P<type>\d+)"
)


def _parse_name(stem: str) -> dict | None:
    m = _FNAME_RE.match(stem)
    if not m:
        return None
    g = m.groupdict()
    date_site = f"{g['date']}_{g['site']}"
    return {
        "site": g["site"],
        "dasar": g["dasar"],
        "date": g["date"],
        "call_type": g["type"],
        "date_site": date_site,
        "label": 0 if g["type"] == "0" else 1,
    }


def scan_master_inventory(base_dir: Path, cache_path: Path | None = None) -> list[dict]:
    """Recursively scan every .mat file under base_dir and parse its metadata.

    This is a filenames-only pass (no ``loadmat`` calls), so it is fast even
    over ~1.2M files on a slow external drive. Results can be cached to JSON
    via ``cache_path`` so repeated dry-runs / sampling-policy experiments
    don't have to re-walk the whole drive.
    """
    if cache_path and cache_path.exists():
        print(f"Loading cached inventory from {cache_path} ...")
        rows = json.loads(cache_path.read_text())
        print(f"  {len(rows):,} files (cached)")
        return rows

    print(f"Scanning master database at {base_dir} (recursive, filenames only) ...")
    rows: list[dict] = []
    skipped = 0
    t0 = time.time()
    for i, fp in enumerate(base_dir.rglob("*.mat")):
        meta = _parse_name(fp.stem)
        if meta is None:
            skipped += 1
            continue
        meta["stem"] = fp.stem
        meta["path"] = str(fp)
        rows.append(meta)
        if (i + 1) % 100_000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  scanned {i + 1:,} files ({rate:.0f}/s, {len(rows):,} parsed so far)")

    print(f"Scan complete: {len(rows):,} parsed, {skipped:,} skipped (unmatched filename pattern)")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(rows))
        print(f"  cached inventory -> {cache_path}")
    return rows


def load_exclusion_sets(
    train_npz_path: Path,
    train_source_dirs: list[Path] | None = None,
) -> tuple[set[str], set[str] | None]:
    """Return (excluded_date_site_groups, excluded_stems_or_None).

    ``excluded_date_site_groups`` is always computed from the training NPZ's
    stored ``date``/``site`` fields -- this is the primary, guaranteed-available
    exclusion mechanism. ``excluded_stems`` is only available if the caller
    points at the actual source .dir directories used to build that NPZ
    (the NPZ itself does not retain per-file stems), and is an optional,
    stricter defense-in-depth check on top of the group exclusion.
    """
    d = np.load(train_npz_path, allow_pickle=True)
    excluded_groups = set(f"{dd}_{ss}" for dd, ss in zip(d["date"], d["site"]))

    excluded_stems: set[str] | None = None
    if train_source_dirs:
        excluded_stems = set()
        for td in train_source_dirs:
            td = Path(td)
            n_before = len(excluded_stems)
            for fp in td.glob("*.mat"):
                excluded_stems.add(fp.stem)
            print(f"  {td.name}: {len(excluded_stems) - n_before:,} filenames added to exact-exclusion set")

    return excluded_groups, excluded_stems


def _stratified_sample(
    pool_rows: list[dict],
    n_target: int,
    label_name: str,
    rng: np.random.Generator,
) -> tuple[list[dict], dict[str, int]]:
    """Sample up to n_target rows from pool_rows, proportional to each
    date_site group's availability, WITHOUT replacement and WITHOUT ever
    exceeding a group's true size (i.e. never replicating a call).

    If the pool can't supply n_target, all available rows are used and a
    warning is printed -- this is the "no replication wins over proportional
    coverage" policy, applied automatically and reported rather than patched.
    """
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in pool_rows:
        by_group[r["date_site"]].append(r)
    group_sizes = {g: len(v) for g, v in by_group.items()}
    total_available = sum(group_sizes.values())

    if total_available <= n_target:
        if total_available < n_target:
            print(
                f"  WARNING: requested {n_target:,} {label_name} samples but only "
                f"{total_available:,} are available in the group-exclusive eligible pool. "
                f"Using all {total_available:,} instead of replicating any call."
            )
        picked = [r for rows in by_group.values() for r in rows]
        rng.shuffle(picked)
        return picked, group_sizes

    groups = sorted(group_sizes)
    raw = {g: n_target * group_sizes[g] / total_available for g in groups}
    alloc = {g: min(group_sizes[g], int(np.floor(raw[g]))) for g in groups}
    shortfall = n_target - sum(alloc.values())

    # Largest-remainder method: hand out leftover quota to groups with the
    # biggest fractional remainder first, never exceeding group capacity.
    order = sorted(groups, key=lambda g: (raw[g] - np.floor(raw[g])), reverse=True)
    idx = 0
    guard = 0
    while shortfall > 0 and guard < len(order) * 10:
        g = order[idx % len(order)]
        if alloc[g] < group_sizes[g]:
            alloc[g] += 1
            shortfall -= 1
        idx += 1
        guard += 1

    picked: list[dict] = []
    for g in groups:
        k = alloc[g]
        if k <= 0:
            continue
        chosen_idx = rng.choice(len(by_group[g]), size=k, replace=False)
        picked.extend(by_group[g][i] for i in chosen_idx)
    rng.shuffle(picked)
    return picked, group_sizes


def build_eval_v2(
    base_dir: Path,
    train_npz: Path,
    out_path: Path,
    manifest_path: Path,
    n_manual: int = 50_000,
    n_auto: int = 150_000,
    gram: str = "SNR_gram",
    seed: int = 0,
    train_source_dirs: list[Path] | None = None,
    inventory_cache: Path | None = None,
    dry_run: bool = False,
) -> None:
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- scan
    rows = scan_master_inventory(base_dir, cache_path=inventory_cache)
    if not rows:
        raise RuntimeError(f"No .mat files found under {base_dir}. Check --base-dir.")

    # ---------------------------------------------------------- exclusion
    print(f"\nLoading exclusion sets from training NPZ {train_npz} ...")
    excluded_groups, excluded_stems = load_exclusion_sets(train_npz, train_source_dirs)
    print(f"  {len(excluded_groups):,} date_site groups excluded (used in training)")
    if excluded_stems is not None:
        print(f"  {len(excluded_stems):,} exact training filenames excluded (defense-in-depth)")
    else:
        print(
            "  NOTE: no --train-source-dirs given, so exact-filename exclusion is not active; "
            "relying solely on date_site group exclusion (the same conservative key already "
            "used by bowhead/data/splits.py: make_date_site_group)."
        )

    n_excl_group = 0
    n_excl_stem = 0
    eligible: list[dict] = []
    for r in rows:
        if r["date_site"] in excluded_groups:
            n_excl_group += 1
            continue
        if excluded_stems is not None and r["stem"] in excluded_stems:
            n_excl_stem += 1
            continue
        eligible.append(r)

    print(
        f"\nEligible pool after group-exclusive filtering: {len(eligible):,} / {len(rows):,} "
        f"(excluded {n_excl_group:,} by date_site group, {n_excl_stem:,} by exact filename)"
    )

    manual_rows = [r for r in eligible if r["label"] == 1]
    auto_rows = [r for r in eligible if r["label"] == 0]
    print(f"  eligible manual (Type 1-7): {len(manual_rows):,}")
    print(f"  eligible auto   (Type 0)  : {len(auto_rows):,}")

    # -------------------------------------------------------------- sample
    manual_sample, manual_group_sizes = _stratified_sample(manual_rows, n_manual, "manual", rng)
    auto_sample, auto_group_sizes = _stratified_sample(auto_rows, n_auto, "auto", rng)
    sampled_rows = manual_sample + auto_sample
    n = len(sampled_rows)
    print(
        f"\nSampled eval_v2 set: {n:,} files "
        f"({len(manual_sample):,} manual + {len(auto_sample):,} auto)"
    )

    # ------------------------------------------------------- verify (hard)
    sampled_groups = set(r["date_site"] for r in sampled_rows)
    group_overlap = sampled_groups & excluded_groups
    assert not group_overlap, (
        f"LEAKAGE DETECTED: {len(group_overlap)} date_site groups overlap training set: "
        f"{sorted(group_overlap)[:10]}..."
    )
    if excluded_stems is not None:
        sampled_stems = set(r["stem"] for r in sampled_rows)
        stem_overlap = sampled_stems & excluded_stems
        assert not stem_overlap, (
            f"LEAKAGE DETECTED: {len(stem_overlap)} exact filenames overlap training set."
        )
        print("Verified: zero date_site group overlap AND zero exact-filename overlap with training.")
    else:
        print("Verified: zero date_site group overlap with training (filename-level check not run).")

    # ------------------------------------------------------------ manifest
    manifest = {
        "base_dir": str(base_dir),
        "train_npz": str(train_npz),
        "train_source_dirs": [str(p) for p in train_source_dirs] if train_source_dirs else None,
        "seed": seed,
        "n_master_files_scanned": len(rows),
        "n_excluded_by_group": n_excl_group,
        "n_excluded_by_stem": n_excl_stem,
        "n_eligible_manual": len(manual_rows),
        "n_eligible_auto": len(auto_rows),
        "n_requested_manual": n_manual,
        "n_requested_auto": n_auto,
        "n_sampled_manual": len(manual_sample),
        "n_sampled_auto": len(auto_sample),
        "n_final_total": n,
        "n_excluded_training_groups": len(excluded_groups),
        "n_sampled_date_site_groups": len(sampled_groups),
        "shortfall_manual": max(0, n_manual - len(manual_sample)),
        "shortfall_auto": max(0, n_auto - len(auto_sample)),
        "verification": (
            "zero date_site group overlap with training (assert passed)"
            + (" + zero exact-filename overlap (assert passed)" if excluded_stems is not None else "")
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written -> {manifest_path}")

    if dry_run:
        print("\n--dry-run set: stopping before image loading / NPZ write.")
        return

    # ------------------------------------------------------------ load imgs
    first = loadmat(sampled_rows[0]["path"])
    if gram not in first:
        avail = [k for k in first if not k.startswith("__")]
        raise KeyError(f"Gram '{gram}' not in {sampled_rows[0]['path']}. Available: {avail}")
    h, w = first[gram].shape
    print(f"\nLoading {n:,} images ({h}x{w}, gram={gram}) ...")

    images = np.empty((n, h, w), dtype=np.uint8)
    keep_mask = np.ones(n, dtype=bool)
    t0 = time.time()
    for i, r in enumerate(sampled_rows):
        try:
            m = loadmat(r["path"])
            im = m[gram]
            if im.shape != (h, w):
                raise ValueError(f"shape mismatch {im.shape} vs {(h, w)}")
            images[i] = im.astype(np.uint8)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN dropping {r['path']}: {e}")
            keep_mask[i] = False
        if (i + 1) % 20_000 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (n - i - 1) / rate / 60
            print(f"  loaded {i + 1:,}/{n:,} ({rate:.0f}/s, ETA {eta:.1f} min)")

    if not keep_mask.all():
        images = images[keep_mask]
        sampled_rows = [r for r, ok in zip(sampled_rows, keep_mask) if ok]
        n = len(sampled_rows)
        print(f"Dropped {int((~keep_mask).sum())} unreadable files; {n:,} remain")

    # -------------------------------------------------------------- write
    labels = np.array([r["label"] for r in sampled_rows], dtype=np.int64)
    date = np.array([r["date"] for r in sampled_rows])
    site = np.array([r["site"] for r in sampled_rows])
    dasar = np.array([r["dasar"] for r in sampled_rows])
    call_type = np.array([r["call_type"] for r in sampled_rows])
    date_site = np.array([r["date_site"] for r in sampled_rows])
    stems = np.array([r["stem"] for r in sampled_rows])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        images=images,
        label=labels,
        date=date,
        site=site,
        dasar=dasar,
        call_type=call_type,
        unique_call=date_site,
        is_airgun=np.zeros(n, dtype=bool),
        stem=stems,  # retained here (unlike the original training NPZ) for future audit/dedup
    )
    print(f"\nWritten {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")

    print("\n=== eval_v2 summary ===")
    print(f"  calls (manual, Type1-7): {int(labels.sum()):,}")
    print(f"  non-calls (auto, Type0): {int((labels == 0).sum()):,}")
    print(f"  date_site groups used  : {len(np.unique(date_site)):,}")
    print(f"  years  : {dict(zip(*np.unique([dd[:4] for dd in date], return_counts=True)))}")
    print(f"  sites  : {dict(zip(*np.unique(site, return_counts=True)))}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a group-exclusive, leakage-free evaluation dataset "
        "from the master call database, guaranteed disjoint from the 100K training draw."
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets"),
        help="Root of the master ~1.2M-file database (recursively scanned for .mat files). "
        "Update this to the external hard-drive mount point.",
    )
    p.add_argument(
        "--train-npz",
        type=Path,
        default=Path("data/spectrograms_100k_matched.npz"),
        help="Existing training NPZ; its date/site groups define the exclusion set.",
    )
    p.add_argument(
        "--train-source-dirs",
        type=Path,
        nargs="*",
        default=None,
        help="Optional: original Manual_100K/Auto_100K .dir directories used to build "
        "--train-npz, for exact-filename exclusion (defense-in-depth on top of group exclusion).",
    )
    p.add_argument("--out", type=Path, default=Path("data/spectrograms_eval_v2.npz"))
    p.add_argument("--manifest", type=Path, default=Path("runs/eval_v2_manifest.json"))
    p.add_argument("--n-manual", type=int, default=50_000)
    p.add_argument("--n-auto", type=int, default=150_000)
    p.add_argument("--gram", default="SNR_gram")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--inventory-cache",
        type=Path,
        default=Path("runs/eval_v2_master_inventory_cache.json"),
        help="Cache the filenames-only master-drive scan here so repeated dry-runs "
        "don't re-walk the whole drive. Delete this file to force a fresh scan.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan, exclude, sample, and verify -- but stop before loading any images "
        "or writing the NPZ. Use this first to confirm feasibility and inspect the manifest.",
    )
    args = p.parse_args()

    build_eval_v2(
        base_dir=args.base_dir,
        train_npz=args.train_npz,
        out_path=args.out,
        manifest_path=args.manifest,
        n_manual=args.n_manual,
        n_auto=args.n_auto,
        gram=args.gram,
        seed=args.seed,
        train_source_dirs=args.train_source_dirs,
        inventory_cache=args.inventory_cache,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
