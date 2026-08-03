"""Build a genuinely leakage-free train/eval pair by jointly re-splitting the
master call database at the date_site GROUP level, then sampling both a new
training set and a new evaluation set from their own disjoint group pools.

WHY THIS SCRIPT (AND NOT build_eval_v2.py) IS THE RIGHT TOOL NOW
------------------------------------------------------------------
build_eval_v2.py assumed there would be date_site groups on the master drive
that the existing 100K training draw never touched, so a new eval set could
be carved out "around" the fixed training set. In practice, the master
database (``/Volumes/R3D_2024_1/Bowhead_Whale_2026/<year>/Site{3,5}/Day_*``)
contains exactly 51 non-empty date_site groups, and the existing training
set (``data/spectrograms_100k_matched.npz``) already draws from ALL 51 of
them. There are two additional day-folders on the drive
(``20080821_3`` under 2008/Site3 and ``20121005_5`` under 2012/Site5) but
both are empty placeholders with zero .mat files -- confirmed by direct
inspection, not inferred.

So there is no way to build an independent eval set while leaving the
existing training set's group membership untouched: the constraint that
matters for leakage safety is at the GROUP level (same day + site =
correlated background noise / recording conditions / possible duplicate
detections of the same physical call), not the individual file level, and
every available group has already contributed *some* training samples.

The fix is to stop treating the training set as fixed and instead:
  1. Partition all 51 groups into a TRAIN-POOL and an EVAL-POOL up front,
     stratified by year (so all four field seasons remain represented in
     both new datasets) and balanced by available manual-annotation volume
     per pool (greedy largest-first bin balancing), so neither pool is
     starved of manually-labeled calls.
  2. Draw the new 100K training set (50K manual + 50K auto, matching the
     existing set's composition) exclusively from TRAIN-POOL groups.
  3. Draw the new 200K evaluation set (50K manual + 150K auto, per your
     request) exclusively from EVAL-POOL groups.
  4. Assert zero group overlap between the two pools before writing
     anything (this is true by construction, but is verified, not assumed).

This necessarily changes which specific files/groups end up in "training"
relative to the current data/spectrograms_100k_matched.npz -- there is no
way to avoid that given what's physically on the drive (see module-level
analysis above). The new training set is comparable in size/composition/
year-coverage to the old one, just drawn from a different (deliberately
independent-of-eval) subset of groups.

USAGE
-----
Dry run (fast, filenames only, no images loaded -- inspect the manifest and
per-pool group assignment before committing to a slow full load):

    python -m bowhead.data.build_train_eval_v2 \\
        --base-dir "/Volumes/R3D_2024_1/Bowhead_Whale_2026" \\
        --out-train data/spectrograms_100k_v2.npz \\
        --out-eval  data/spectrograms_eval_v2.npz \\
        --manifest  runs/train_eval_v2_manifest.json \\
        --dry-run

Full run (loads images, writes both NPZ files):

    python -m bowhead.data.build_train_eval_v2 \\
        --base-dir "/Volumes/R3D_2024_1/Bowhead_Whale_2026" \\
        --out-train data/spectrograms_100k_v2.npz \\
        --out-eval  data/spectrograms_eval_v2.npz \\
        --manifest  runs/train_eval_v2_manifest.json
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
    r"S(?P<site>\d)\d{2}(?P<dasar>[A-G])\dT"
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
        "year": g["date"][:4],
        "label": 0 if g["type"] == "0" else 1,
    }


def scan_master_inventory(base_dir: Path, cache_path: Path | None = None) -> list[dict]:
    """Recursively scan Event_sounds.dir (auto, label 0) and
    Manually_selected_bowhead_calls.dir (manual, label 1) under every
    <year>/Site{3,5}/Day_* folder. Filenames only -- fast even at ~2M files.
    """
    if cache_path and cache_path.exists():
        print(f"Loading cached inventory from {cache_path} ...")
        rows = json.loads(cache_path.read_text())
        print(f"  {len(rows):,} files (cached)")
        return rows

    print(f"Scanning master database at {base_dir} ...")
    rows: list[dict] = []
    skipped = 0
    t0 = time.time()
    for i, fp in enumerate(base_dir.rglob("*.mat")):
        parts = fp.parts
        is_manual = "Manually_selected_bowhead_calls.dir" in parts
        is_event = "Event_sounds.dir" in parts
        if not (is_manual or is_event):
            continue
        meta = _parse_name(fp.stem)
        if meta is None:
            skipped += 1
            continue
        meta["stem"] = fp.stem
        meta["path"] = str(fp)
        rows.append(meta)
        if (i + 1) % 200_000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  scanned {i + 1:,} files ({rate:.0f}/s, {len(rows):,} parsed so far)")

    print(f"Scan complete: {len(rows):,} parsed (manual+auto), {skipped:,} skipped")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(rows))
        print(f"  cached inventory -> {cache_path}")
    return rows


def partition_groups(
    rows: list[dict],
    seed: int = 0,
) -> tuple[set[str], set[str], dict]:
    """Greedy, year-stratified, manual-volume-balanced 2-way partition of
    date_site groups into a train-pool and an eval-pool.

    Within each year, groups are sorted by their manual-annotation count
    (largest first) and alternately assigned to whichever pool currently has
    less accumulated manual volume -- a simple longest-processing-time-first
    bin-balancing heuristic. This keeps both pools represented across all
    four field seasons and prevents either pool from being starved of
    manually-labeled calls, without ever splitting an individual group.
    """
    manual_counts: dict[str, int] = defaultdict(int)
    auto_counts: dict[str, int] = defaultdict(int)
    year_of: dict[str, str] = {}
    for r in rows:
        g = r["date_site"]
        year_of[g] = r["year"]
        if r["label"] == 1:
            manual_counts[g] += 1
        else:
            auto_counts[g] += 1

    by_year: dict[str, list[str]] = defaultdict(list)
    for g, y in year_of.items():
        by_year[y].append(g)

    train_groups: set[str] = set()
    eval_groups: set[str] = set()
    per_year_report = {}
    for year in sorted(by_year):
        yg = sorted(by_year[year], key=lambda g: manual_counts[g], reverse=True)
        t_load = e_load = 0
        for g in yg:
            if t_load <= e_load:
                train_groups.add(g)
                t_load += manual_counts[g]
            else:
                eval_groups.add(g)
                e_load += manual_counts[g]
        per_year_report[year] = {
            "n_train_groups": sum(1 for g in yg if g in train_groups),
            "n_eval_groups": sum(1 for g in yg if g in eval_groups),
            "train_manual": t_load,
            "eval_manual": e_load,
        }

    report = {
        "per_year": per_year_report,
        "train_groups": sorted(train_groups),
        "eval_groups": sorted(eval_groups),
        "train_manual_total": sum(manual_counts[g] for g in train_groups),
        "eval_manual_total": sum(manual_counts[g] for g in eval_groups),
        "train_auto_total": sum(auto_counts[g] for g in train_groups),
        "eval_auto_total": sum(auto_counts[g] for g in eval_groups),
    }
    return train_groups, eval_groups, report


def _stratified_sample(
    pool_rows: list[dict],
    n_target: int,
    label_name: str,
    rng: np.random.Generator,
) -> list[dict]:
    """Sample up to n_target rows from pool_rows proportional to each
    date_site group's availability, without replacement and without ever
    padding a sparse group -- shortfalls are logged, not patched.
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
                f"{total_available:,} available. Using all {total_available:,}, no replication."
            )
        picked = [r for rows in by_group.values() for r in rows]
        rng.shuffle(picked)
        return picked

    groups = sorted(group_sizes)
    raw = {g: n_target * group_sizes[g] / total_available for g in groups}
    alloc = {g: min(group_sizes[g], int(np.floor(raw[g]))) for g in groups}
    shortfall = n_target - sum(alloc.values())

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
    return picked


def _load_images(rows: list[dict], gram: str, tag: str) -> tuple[np.ndarray, list[dict]]:
    n = len(rows)
    first = loadmat(rows[0]["path"])
    if gram not in first:
        avail = [k for k in first if not k.startswith("__")]
        raise KeyError(f"Gram '{gram}' not in {rows[0]['path']}. Available: {avail}")
    h, w = first[gram].shape
    print(f"\n[{tag}] Loading {n:,} images ({h}x{w}, gram={gram}) ...")

    images = np.empty((n, h, w), dtype=np.uint8)
    keep_mask = np.ones(n, dtype=bool)
    t0 = time.time()
    for i, r in enumerate(rows):
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
            print(f"  [{tag}] loaded {i + 1:,}/{n:,} ({rate:.0f}/s, ETA {eta:.1f} min)")

    if not keep_mask.all():
        images = images[keep_mask]
        rows = [r for r, ok in zip(rows, keep_mask) if ok]
        print(f"[{tag}] Dropped {int((~keep_mask).sum())} unreadable files; {len(rows):,} remain")
    return images, rows


def _write_npz(rows: list[dict], images: np.ndarray, out_path: Path) -> None:
    n = len(rows)
    labels = np.array([r["label"] for r in rows], dtype=np.int64)
    date = np.array([r["date"] for r in rows])
    site = np.array([r["site"] for r in rows])
    dasar = np.array([r["dasar"] for r in rows])
    call_type = np.array([r["call_type"] for r in rows])
    date_site = np.array([r["date_site"] for r in rows])
    stems = np.array([r["stem"] for r in rows])

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
        stem=stems,
    )
    print(f"Written {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")


def build_train_eval_v2(
    base_dir: Path,
    out_train: Path,
    out_eval: Path,
    manifest_path: Path,
    n_train_manual: int = 50_000,
    n_train_auto: int = 50_000,
    n_eval_manual: int = 50_000,
    n_eval_auto: int = 150_000,
    gram: str = "SNR_gram",
    seed: int = 0,
    inventory_cache: Path | None = None,
    dry_run: bool = False,
) -> None:
    rng = np.random.default_rng(seed)

    rows = scan_master_inventory(base_dir, cache_path=inventory_cache)
    if not rows:
        raise RuntimeError(f"No manual/auto .mat files found under {base_dir}.")

    print("\nPartitioning date_site groups into train-pool / eval-pool ...")
    train_groups, eval_groups, report = partition_groups(rows, seed=seed)

    overlap = train_groups & eval_groups
    assert not overlap, f"Internal error: group partition produced overlap: {overlap}"
    print(f"  {len(train_groups)} groups -> TRAIN-POOL, {len(eval_groups)} groups -> EVAL-POOL")
    for year, r in sorted(report["per_year"].items()):
        print(
            f"    {year}: train_groups={r['n_train_groups']} eval_groups={r['n_eval_groups']} "
            f"train_manual={r['train_manual']:,} eval_manual={r['eval_manual']:,}"
        )
    print(
        f"  TOTAL train-pool: manual={report['train_manual_total']:,} auto={report['train_auto_total']:,}"
    )
    print(
        f"  TOTAL eval-pool : manual={report['eval_manual_total']:,} auto={report['eval_auto_total']:,}"
    )

    train_pool_rows = [r for r in rows if r["date_site"] in train_groups]
    eval_pool_rows = [r for r in rows if r["date_site"] in eval_groups]

    train_manual_pool = [r for r in train_pool_rows if r["label"] == 1]
    train_auto_pool = [r for r in train_pool_rows if r["label"] == 0]
    eval_manual_pool = [r for r in eval_pool_rows if r["label"] == 1]
    eval_auto_pool = [r for r in eval_pool_rows if r["label"] == 0]

    print("\nSampling new training set from TRAIN-POOL ...")
    train_manual = _stratified_sample(train_manual_pool, n_train_manual, "train-manual", rng)
    train_auto = _stratified_sample(train_auto_pool, n_train_auto, "train-auto", rng)
    train_rows = train_manual + train_auto
    rng.shuffle(train_rows)

    print("\nSampling new evaluation set from EVAL-POOL ...")
    eval_manual = _stratified_sample(eval_manual_pool, n_eval_manual, "eval-manual", rng)
    eval_auto = _stratified_sample(eval_auto_pool, n_eval_auto, "eval-auto", rng)
    eval_rows = eval_manual + eval_auto
    rng.shuffle(eval_rows)

    # ------------------------------------------------------- hard verification
    train_final_groups = set(r["date_site"] for r in train_rows)
    eval_final_groups = set(r["date_site"] for r in eval_rows)
    group_overlap = train_final_groups & eval_final_groups
    assert not group_overlap, f"LEAKAGE DETECTED: shared date_site groups {group_overlap}"

    train_stems = set(r["stem"] for r in train_rows)
    eval_stems = set(r["stem"] for r in eval_rows)
    stem_overlap = train_stems & eval_stems
    assert not stem_overlap, f"LEAKAGE DETECTED: shared exact filenames {sorted(stem_overlap)[:10]}"
    print(
        f"\nVerified: zero date_site group overlap AND zero exact-filename overlap "
        f"between new train ({len(train_rows):,} samples, {len(train_final_groups)} groups) "
        f"and new eval ({len(eval_rows):,} samples, {len(eval_final_groups)} groups)."
    )

    manifest = {
        "base_dir": str(base_dir),
        "seed": seed,
        "group_partition": {
            "per_year": report["per_year"],
            "train_groups": report["train_groups"],
            "eval_groups": report["eval_groups"],
        },
        "train": {
            "requested_manual": n_train_manual,
            "requested_auto": n_train_auto,
            "sampled_manual": len(train_manual),
            "sampled_auto": len(train_auto),
            "total": len(train_rows),
            "n_groups": len(train_final_groups),
            "shortfall_manual": max(0, n_train_manual - len(train_manual)),
            "shortfall_auto": max(0, n_train_auto - len(train_auto)),
        },
        "eval": {
            "requested_manual": n_eval_manual,
            "requested_auto": n_eval_auto,
            "sampled_manual": len(eval_manual),
            "sampled_auto": len(eval_auto),
            "total": len(eval_rows),
            "n_groups": len(eval_final_groups),
            "shortfall_manual": max(0, n_eval_manual - len(eval_manual)),
            "shortfall_auto": max(0, n_eval_auto - len(eval_auto)),
        },
        "verification": "zero date_site group overlap AND zero exact-filename overlap (asserts passed)",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written -> {manifest_path}")

    if dry_run:
        print("\n--dry-run set: stopping before image loading / NPZ write.")
        return

    train_images, train_rows = _load_images(train_rows, gram, "train")
    eval_images, eval_rows = _load_images(eval_rows, gram, "eval")

    _write_npz(train_rows, train_images, out_train)
    _write_npz(eval_rows, eval_images, out_eval)

    for tag, rows_, images_ in (("TRAIN", train_rows, train_images), ("EVAL", eval_rows, eval_images)):
        labels = np.array([r["label"] for r in rows_])
        date = np.array([r["date"] for r in rows_])
        print(f"\n=== {tag} summary ===")
        print(f"  n images : {images_.shape}")
        print(f"  calls (manual): {int(labels.sum()):,}  non-calls (auto): {int((labels == 0).sum()):,}")
        print(f"  years: {dict(zip(*np.unique([d[:4] for d in date], return_counts=True)))}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Jointly rebuild a group-disjoint training set and evaluation set "
        "from the master call database, guaranteeing zero group/filename overlap by construction."
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/Volumes/R3D_2024_1/Bowhead_Whale_2026"),
        help="Root of the master database (year/Site/Day_*/{Event_sounds,Manually_selected_bowhead_calls}.dir)",
    )
    p.add_argument("--out-train", type=Path, default=Path("data/spectrograms_100k_v2.npz"))
    p.add_argument("--out-eval", type=Path, default=Path("data/spectrograms_eval_v2.npz"))
    p.add_argument("--manifest", type=Path, default=Path("runs/train_eval_v2_manifest.json"))
    p.add_argument("--n-train-manual", type=int, default=50_000)
    p.add_argument("--n-train-auto", type=int, default=50_000)
    p.add_argument("--n-eval-manual", type=int, default=50_000)
    p.add_argument("--n-eval-auto", type=int, default=150_000)
    p.add_argument("--gram", default="SNR_gram")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--inventory-cache",
        type=Path,
        default=Path("runs/train_eval_v2_master_inventory_cache.json"),
        help="Cache the filenames-only master-drive scan here. Delete to force a fresh scan.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan, partition, sample, and verify -- but stop before loading images / writing NPZs.",
    )
    args = p.parse_args()

    build_train_eval_v2(
        base_dir=args.base_dir,
        out_train=args.out_train,
        out_eval=args.out_eval,
        manifest_path=args.manifest,
        n_train_manual=args.n_train_manual,
        n_train_auto=args.n_train_auto,
        n_eval_manual=args.n_eval_manual,
        n_eval_auto=args.n_eval_auto,
        gram=args.gram,
        seed=args.seed,
        inventory_cache=args.inventory_cache,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
