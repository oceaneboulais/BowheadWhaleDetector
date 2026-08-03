"""Repair pass for data/spectrograms_eval_v2.npz.

The original build_train_eval_v2.py run dropped 61,525 of the 200,000
sampled eval files (34,567 manual short of 50,000; 103,908 auto short of
150,000) due to intermittent external-drive I/O errors (`Device not
configured` / `No such file or directory`) during the ~2h eval-loading pass.
Training set loading finished before the drive became unstable and had zero
drops.

This script:
  1. Retries loading exactly the 61,525 originally-sampled-but-dropped files
     (recorded in runs/eval_v2_missing_rows.json) now that the drive is
     stable again.
  2. For any files that still fail (permanently unreadable, not just
     transient), draws replacement rows from the same EVAL-POOL groups
     (never from TRAIN-POOL, so the group/stem leakage guarantees from the
     original build still hold) without replacement/duplication.
  3. Merges the repaired rows into the existing 138,475-row eval npz and
     rewrites data/spectrograms_eval_v2.npz with the full (up to) 200,000
     samples.

Usage:
    python -m bowhead.data.repair_eval_v2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bowhead.data.build_train_eval_v2 import scan_master_inventory, partition_groups


def _load_rows(rows: list[dict], gram: str, tag: str) -> tuple[np.ndarray, list[dict]]:
    n = len(rows)
    if n == 0:
        return np.empty((0, 0, 0), dtype=np.uint8), []
    first = None
    h = w = None
    for r in rows:
        try:
            first = loadmat(r["path"])
            h, w = first[gram].shape
            break
        except Exception:
            continue
    if first is None:
        raise RuntimeError("Could not read even one file to determine image shape.")

    images = np.empty((n, h, w), dtype=np.uint8)
    keep_mask = np.ones(n, dtype=bool)
    t0 = time.time()
    n_fail = 0
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
            n_fail += 1
        if (i + 1) % 5_000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  [{tag}] {i + 1:,}/{n:,} ({rate:.0f}/s, {n_fail} failed so far)")

    if not keep_mask.all():
        images = images[keep_mask]
        rows = [r for r, ok in zip(rows, keep_mask) if ok]
    return images, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", type=Path,
                     default=Path("/Volumes/R3D_2024_1/Bowhead_Whale_2026"))
    ap.add_argument("--missing-rows", type=Path,
                     default=Path("runs/eval_v2_missing_rows.json"))
    ap.add_argument("--eval-npz", type=Path, default=Path("data/spectrograms_eval_v2.npz"))
    ap.add_argument("--inventory-cache", type=Path,
                     default=Path("runs/train_eval_v2_master_inventory_cache.json"))
    ap.add_argument("--gram", default="SNR_gram")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval-manual-target", type=int, default=50_000)
    ap.add_argument("--n-eval-auto-target", type=int, default=150_000)
    args = ap.parse_args()

    missing_rows = json.load(open(args.missing_rows))
    print(f"Attempting to repair {len(missing_rows):,} originally-dropped files ...")
    repaired_images, repaired_rows = _load_rows(missing_rows, args.gram, "repair")
    n_still_missing = len(missing_rows) - len(repaired_rows)
    print(f"Repair pass: recovered {len(repaired_rows):,}/{len(missing_rows):,} "
          f"({n_still_missing} still unreadable)")

    # Load existing (138,475-row) eval npz
    d = np.load(args.eval_npz, allow_pickle=True)
    existing_stems = set(d["stem"])
    existing_n = len(d["label"])
    print(f"Existing eval npz: {existing_n:,} rows")

    manual_have = int((d["label"] == 1).sum())
    auto_have = int((d["label"] == 0).sum())
    manual_repaired = sum(1 for r in repaired_rows if r["label"] == 1)
    auto_repaired = sum(1 for r in repaired_rows if r["label"] == 0)
    manual_total = manual_have + manual_repaired
    auto_total = auto_have + auto_repaired
    print(f"After repair: manual={manual_total:,}/{args.n_eval_manual_target:,}  "
          f"auto={auto_total:,}/{args.n_eval_auto_target:,}")

    manual_shortfall = max(0, args.n_eval_manual_target - manual_total)
    auto_shortfall = max(0, args.n_eval_auto_target - auto_total)

    backfill_rows: list[dict] = []
    if manual_shortfall > 0 or auto_shortfall > 0:
        print(f"\nBackfilling remaining shortfall: manual={manual_shortfall:,} "
              f"auto={auto_shortfall:,} from EVAL-POOL groups (excluding all "
              f"used/attempted stems, no replication) ...")
        rows = scan_master_inventory(args.base_dir, cache_path=args.inventory_cache)
        _, eval_groups, _ = partition_groups(rows, seed=args.seed)
        used_stems = existing_stems | set(r["stem"] for r in repaired_rows) | \
            set(r["stem"] for r in missing_rows)
        eval_pool_rows = [r for r in rows if r["date_site"] in eval_groups
                          and r["stem"] not in used_stems]
        rng = np.random.default_rng(args.seed + 1)

        def sample_more(pool: list[dict], k: int, tag: str) -> list[dict]:
            if k <= 0 or not pool:
                return []
            k = min(k, len(pool))
            idx = rng.choice(len(pool), size=k, replace=False)
            print(f"  drawing {k:,} replacement {tag} rows from {len(pool):,} available")
            return [pool[i] for i in idx]

        manual_pool = [r for r in eval_pool_rows if r["label"] == 1]
        auto_pool = [r for r in eval_pool_rows if r["label"] == 0]
        backfill_candidates = (
            sample_more(manual_pool, manual_shortfall, "manual")
            + sample_more(auto_pool, auto_shortfall, "auto")
        )
        backfill_images, backfill_rows = _load_rows(backfill_candidates, args.gram, "backfill")
    else:
        backfill_images = np.empty((0,) + repaired_images.shape[1:], dtype=np.uint8) \
            if len(repaired_images) else np.empty((0, 0, 0), dtype=np.uint8)

    # ---- merge everything and write out ----
    all_images = [d["images"]]
    all_rows_extra = []
    if len(repaired_rows):
        all_images.append(repaired_images)
        all_rows_extra.extend(repaired_rows)
    if len(backfill_rows):
        all_images.append(backfill_images)
        all_rows_extra.extend(backfill_rows)

    final_images = np.concatenate(all_images, axis=0) if len(all_images) > 1 else d["images"]

    final_label = np.concatenate([d["label"], [r["label"] for r in all_rows_extra]])
    final_date = np.concatenate([d["date"], [r["date"] for r in all_rows_extra]])
    final_site = np.concatenate([d["site"], [r["site"] for r in all_rows_extra]])
    final_dasar = np.concatenate([d["dasar"], [r["dasar"] for r in all_rows_extra]])
    final_call_type = np.concatenate([d["call_type"], [r["call_type"] for r in all_rows_extra]])
    final_unique_call = np.concatenate([d["unique_call"], [r["date_site"] for r in all_rows_extra]])
    final_is_airgun = np.concatenate([d["is_airgun"], np.zeros(len(all_rows_extra), dtype=bool)])
    final_stem = np.concatenate([d["stem"], [r["stem"] for r in all_rows_extra]])

    # final leakage check against training set
    d_train = np.load("data/spectrograms_100k_v2.npz", allow_pickle=True)
    train_stems = set(d_train["stem"])
    train_groups = set(d_train["unique_call"])
    eval_stems = set(final_stem)
    eval_groups_final = set(final_unique_call)
    assert not (train_stems & eval_stems), "LEAKAGE: stem overlap after repair!"
    assert not (train_groups & eval_groups_final), "LEAKAGE: group overlap after repair!"
    print("\nVerified: zero stem/group overlap between train and repaired eval set.")

    out_path = args.eval_npz
    np.savez_compressed(
        out_path,
        images=final_images,
        label=final_label,
        date=final_date,
        site=final_site,
        dasar=final_dasar,
        call_type=final_call_type,
        unique_call=final_unique_call,
        is_airgun=final_is_airgun,
        stem=final_stem,
    )
    print(f"\nWritten {out_path} ({out_path.stat().st_size / 1e9:.2f} GB), "
          f"{len(final_label):,} total rows "
          f"(manual={int((final_label == 1).sum()):,}, auto={int((final_label == 0).sum()):,})")


if __name__ == "__main__":
    main()
