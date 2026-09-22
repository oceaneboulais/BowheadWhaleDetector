"""Stratified single-pass sample of a nested spectrogram-.mat database, encoded
through the April-2026 production autoencoder and projected with UMAP for the
interactive viewer in ``matlab/Pytorch_scripts/display_latent_matlab_spaces``.

Target dataset layout (verified on disk, ``_centered.dir`` convention):

    <data-dir>/
      Database_index.mat
      Trash.dir/
      <year>/                          e.g. 2008, 2010, 2012, 2014
        Site{3,5}/
          Day_<YYYYMMDD>T000000/
            Event_sounds.dir/                       -> label 0 (auto transient)
            Manually_selected_bowhead_calls.dir/    -> label 1 (verified call)
              [D1.dir/ D2.dir/ D3.dir/ ...]          -> optional extra nesting
                S{site}{yy}{dasar}{idx}T{date}T{hms}_Type{n}.mat

Why NOT "scan everything into a list, then sample" (the approach used by
``bowhead/data/build_train_eval_v2.py``'s ``scan_master_inventory``):
That script's own logs show a single ``2008/Site3`` subtree already holds
~600K files (with macOS ``._*`` AppleDouble siblings roughly doubling that);
four years x two sites is plausibly in the multi-million range. Materializing
every path into memory (or a JSON cache) before choosing a sample is wasteful
and forces a slow up-front pass with nothing to show until it finishes.

Instead this script does a SINGLE pass over the tree with reservoir sampling
(Algorithm R, Vitter 1985, "Random sampling with a reservoir", ACM TOMS)
maintained independently per (year, site, dasar, label) stratum. Algorithm R
guarantees each item seen in a stratum has equal probability of ending up in
that stratum's k-slot reservoir, using O(k) memory regardless of how many
items n are ultimately streamed past (no need to know n in advance, no need
to store more than k items per stratum at any time). Stratifying by
year x site x dasar x label (rather than one global reservoir) follows
standard stratified-sampling practice (Cochran, 1977, "Sampling Techniques")
so that strata with vastly different population sizes (e.g. a rare DASAR/year
combination vs. the dominant one) are still represented in the final sample,
instead of being drowned out the way naive global-random sampling would do.

Cheap stratification during the hot loop: the (year, site) come from
directory names already being walked, the label comes from which of the two
known subfolder names we're under, and the DASAR letter is read directly out
of a fixed filename offset (``name[4]``) rather than running the full
filename regex on every single file. The expensive regex parse (extracting
date/time/call-type) is deferred to the tiny sampled subset only.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from bowhead.data.dataset import per_sample_minmax
from bowhead.models.encoder import load_ae_encoder

_FNAME_RE = re.compile(
    r"S(?P<site>\d)(?P<yy>\d{2})(?P<dasar>[A-G])\d?T"
    r"(?P<date>\d{8})T(?P<hms>\d{6})_Type(?P<type>\d+)"
)
_GRAM_CANDIDATES = ("SNR_gram", "NTV_gram", "KEtoPE_gram", "Polar_gram")
_KIND_DIRS = (("Event_sounds.dir", 0), ("Manually_selected_bowhead_calls.dir", 1))


# --------------------------------------------------------------------------- #
# Reservoir sampling (Algorithm R, Vitter 1985)
# --------------------------------------------------------------------------- #
class ReservoirSampler:
    """Unbiased uniform sample of up to ``k`` items from a stream of unknown
    length, O(k) memory, single pass, no replacement."""

    def __init__(self, k: int, rng: random.Random) -> None:
        self.k = k
        self.rng = rng
        self.items: list = []
        self.n_seen = 0

    def offer(self, item) -> None:
        self.n_seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self.rng.randint(0, self.n_seen - 1)
            if j < self.k:
                self.items[j] = item


# --------------------------------------------------------------------------- #
# Streaming directory walk (cheap stratification, no per-file regex)
# --------------------------------------------------------------------------- #
def _iter_mat_files(kind_path: Path):
    """Recursively yield every real ``.mat`` file under ``kind_path``.

    Tolerant of extra nesting (``D1.dir``/``D2.dir``/... subfolders were found
    on disk but are not documented anywhere) since ``os.walk`` doesn't care
    how deep the tree goes. Skips macOS AppleDouble sidecar files (``._*``).
    """
    for root, _dirs, files in os.walk(kind_path):
        for fn in files:
            if fn.endswith(".mat") and not fn.startswith("._"):
                yield os.path.join(root, fn)


def stratified_reservoir_scan(
    data_dir: Path,
    per_stratum: int,
    seed: int = 0,
    years: list[str] | None = None,
    sites: list[str] | None = None,
    progress_every: int = 200_000,
) -> tuple[list[dict], int, dict[tuple, int]]:
    """Single pass over ``data_dir``; returns (sampled_rows, n_total_seen,
    per_stratum_population)."""
    rng = random.Random(seed)
    reservoirs: dict[tuple, ReservoirSampler] = {}
    n_total = 0
    t0 = time.time()

    year_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.isdigit())
    if years:
        year_dirs = [p for p in year_dirs if p.name in years]
    if not year_dirs:
        raise FileNotFoundError(f"No year subfolders found under {data_dir} (years filter={years})")

    for year_dir in year_dirs:
        site_dirs = sorted(p for p in year_dir.iterdir() if p.is_dir() and p.name.startswith("Site"))
        if sites:
            site_dirs = [p for p in site_dirs if p.name in sites]
        for site_dir in site_dirs:
            with os.scandir(site_dir) as it:
                day_dirs = sorted(e.path for e in it if e.is_dir() and e.name.startswith("Day_"))
            for day_path in day_dirs:
                for kind_name, label in _KIND_DIRS:
                    kind_path = Path(day_path) / kind_name
                    if not kind_path.is_dir():
                        continue
                    for fpath in _iter_mat_files(kind_path):
                        n_total += 1
                        name = os.path.basename(fpath)
                        dasar = name[4] if len(name) > 6 and name[6] == "T" else "?"
                        key = (year_dir.name, site_dir.name, dasar, label)
                        sampler = reservoirs.get(key)
                        if sampler is None:
                            sampler = ReservoirSampler(per_stratum, rng)
                            reservoirs[key] = sampler
                        sampler.offer(fpath)
                        if n_total % progress_every == 0:
                            rate = n_total / (time.time() - t0)
                            print(
                                f"  scanned {n_total:,} files across {len(reservoirs)} strata "
                                f"({rate:,.0f} files/s) ... currently in {year_dir.name}/{site_dir.name}"
                            )

    rows: list[dict] = []
    population = {key: sampler.n_seen for key, sampler in reservoirs.items()}
    for (year, site, dasar, label), sampler in reservoirs.items():
        for fpath in sampler.items:
            m = _FNAME_RE.match(os.path.basename(fpath))
            row = {
                "path": fpath,
                "year": year,
                "site": site,
                "dasar": dasar,
                "label": label,
            }
            if m:
                g = m.groupdict()
                row["date"] = g["date"]
                row["hms"] = g["hms"]
                row["call_type"] = int(g["type"])
            else:
                row["date"] = ""
                row["hms"] = ""
                row["call_type"] = -1
            rows.append(row)

    elapsed = time.time() - t0
    print(
        f"\nScan complete: {n_total:,} files seen in {elapsed / 60:.1f} min, "
        f"{len(reservoirs)} strata, {len(rows):,} sampled (target {per_stratum}/stratum)."
    )
    return rows, n_total, population


# --------------------------------------------------------------------------- #
# AE encoding
# --------------------------------------------------------------------------- #
def _load_gram(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    m = loadmat(path)
    gram = None
    for cand in _GRAM_CANDIDATES:
        if cand in m:
            gram = m[cand]
            break
    if gram is None:
        avail = [k for k in m if not k.startswith("__")]
        raise KeyError(f"No known gram field in {path}. Available: {avail}")
    gram = np.asarray(gram, dtype=np.float32)
    if gram.shape != target_hw:
        from scipy.ndimage import zoom

        zoom_factors = (target_hw[0] / gram.shape[0], target_hw[1] / gram.shape[1])
        gram = zoom(gram, zoom_factors, order=1)
    return gram


def encode_rows(
    rows: list[dict],
    checkpoint_path: str,
    device: str = "cpu",
    batch_size: int = 256,
    input_hw: tuple[int, int] = (121, 104),
    base_channels: int = 32,
    latent_dim: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Load each sampled row's spectrogram, run it through the AE encoder.

    Returns (latents (N, latent_dim) float32, ok_mask (N,) bool) — rows whose
    .mat file couldn't be read are zero-filled and flagged False in ok_mask.
    """
    import torch
    from tqdm import tqdm

    encoder = load_ae_encoder(
        checkpoint_path,
        device=device,
        input_hw=input_hw,
        base_channels=base_channels,
        latent_dim=latent_dim,
    )

    n = len(rows)
    latents = np.zeros((n, encoder.latent_dim), dtype=np.float32)
    ok_mask = np.zeros(n, dtype=bool)

    batch_imgs: list[np.ndarray] = []
    batch_idx: list[int] = []

    def flush() -> None:
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs))[:, None, :, :].to(device)
        with torch.no_grad():
            z = encoder(x).cpu().numpy()
        for bi, zi in zip(batch_idx, z):
            latents[bi] = zi
            ok_mask[bi] = True
        batch_imgs.clear()
        batch_idx.clear()

    for i, row in enumerate(tqdm(rows, desc="AE encode")):
        try:
            img = _load_gram(row["path"], input_hw)
            img = per_sample_minmax(img)
        except Exception as e:  # noqa: BLE001 - record and skip
            print(f"  WARN skip {row['path']}: {e}")
            continue
        batch_imgs.append(img)
        batch_idx.append(i)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    return latents, ok_mask


# --------------------------------------------------------------------------- #
# Manifest export (schema consumed by
# matlab/Pytorch_scripts/display_latent_matlab_spaces/visualize_latent.py)
# --------------------------------------------------------------------------- #
def save_manifest(
    out_path: Path,
    rows: list[dict],
    latents: np.ndarray,
    umap3d: np.ndarray,
    data_dir: Path,
) -> None:
    payload = {
        "latent_embeddings": latents.astype(np.float32),
        "umap_embeddings_3d": umap3d.astype(np.float32),
        "original_filenames": np.array([r["path"] for r in rows], dtype=object),
        "type": np.array([r["call_type"] for r in rows], dtype=np.float64),
        "call_type": np.array([r["call_type"] for r in rows], dtype=np.float64),
        "label": np.array([r["label"] for r in rows], dtype=np.float64),
        "site": np.array([r["site"] for r in rows], dtype=object),
        "dasar": np.array([r["dasar"] for r in rows], dtype=object),
        "year": np.array([r["year"] for r in rows], dtype=object),
        "date": np.array([r["date"] for r in rows], dtype=object),
        "image_folder": str(data_dir),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(str(out_path), payload, do_compression=True)
    print(f"Wrote manifest: {out_path}  ({len(rows):,} points)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Stratified single-pass reservoir sample of a nested spectrogram "
            ".mat database, encoded with the production AE and UMAP-projected."
        )
    )
    p.add_argument("--data-dir", required=True, type=Path,
                   help="e.g. /Volumes/Bowhead_Int/Spectrogram_Image_Database_Sites35_ADG_Y09101214_centered.dir")
    p.add_argument("--ae-checkpoint", required=True, type=Path,
                   help="Path to autoencoder_clean.pth (or .pt) — encoder-only keys are loaded")
    p.add_argument("--out", required=True, type=Path, help="Output .mat manifest path")
    p.add_argument("--per-stratum", type=int, default=200,
                   help="Reservoir size per (year, site, dasar, label) stratum")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--years", nargs="+", default=None, help="Restrict to these year folders")
    p.add_argument("--sites", nargs="+", default=None, help="Restrict to these site folders (e.g. Site3 Site5)")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--input-h", type=int, default=121)
    p.add_argument("--input-w", type=int, default=104)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--n-components", type=int, default=3)
    p.add_argument("--progress-every", type=int, default=200_000)
    args = p.parse_args()

    print(f"Scanning {args.data_dir} (single pass, stratified reservoir, seed={args.seed}) ...")
    rows, n_total, population = stratified_reservoir_scan(
        args.data_dir,
        per_stratum=args.per_stratum,
        seed=args.seed,
        years=args.years,
        sites=args.sites,
        progress_every=args.progress_every,
    )

    label_counts = Counter(r["label"] for r in rows)
    print(f"Sampled label balance: manual(1)={label_counts.get(1, 0):,}  auto(0)={label_counts.get(0, 0):,}")
    print(f"Strata population sizes (top 5 largest): "
          f"{sorted(population.items(), key=lambda kv: -kv[1])[:5]}")

    latents, ok_mask = encode_rows(
        rows,
        str(args.ae_checkpoint),
        device=args.device,
        batch_size=args.batch_size,
        input_hw=(args.input_h, args.input_w),
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
    )
    if not ok_mask.all():
        n_bad = int((~ok_mask).sum())
        print(f"Dropping {n_bad} unreadable file(s); {int(ok_mask.sum()):,} remain")
        rows = [r for r, ok in zip(rows, ok_mask) if ok]
        latents = latents[ok_mask]

    print(f"Projecting {latents.shape[0]:,} x {latents.shape[1]}-D latents via UMAP ...")
    from bowhead.benchmark.unsupervised.projection import umap_project

    umap_coords = umap_project(
        latents,
        n_components=args.n_components,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        seed=args.seed,
    )

    save_manifest(args.out, rows, latents, umap_coords, args.data_dir)


if __name__ == "__main__":
    main()
