"""SurfPerch, from REAL raw DASAR audio (not Griffin-Lim reconstruction), scored
under the same "score once, grade twice" protocol as the other CNN-family rows
(custom CNN, ResNet-18, EfficientNet-B0, and the earlier Griffin-Lim-based
"Perch v1"/BirdNET row): a frozen linear probe fit once on embeddings of the
100k matched training draw, scored once on the full 199,825-image independent
evaluation set, and re-graded against both the original (pre-review) and
manually reviewed (post-review) label sets.

Ground truth and raw audio
---------------------------
Unlike the other rows, which load ``data/spectrograms_100k_matched.npz``
(which has no per-sample timestamp, so it cannot be used to locate real audio),
this script loads the training set from the MATLAB latent-embeddings export
(``paper/FinalDatabases_/latent_embeddings_3d_train_MATLAB_Angel_Final_26Aug2026.mat``),
which has the same 100,000 events (confirmed: identical CD count, 47,691, via
the ``(0 < type < 12)`` convention already used by
``bowhead.benchmark.build_umap_project_page``) AND the per-sample filenames
needed to locate real audio. Each sample's audio is located and verified via
``bowhead.eval.raw_audio_loader`` against the continuous day-long WAV archive
on ``/Volumes/Bowhead_WAV``, then embedded with the real SurfPerch model
(``bowhead.benchmark.backbones.surfperch_from_audio``) -- no spectrogram
reconstruction involved.

The eval set uses the same MATLAB export (and the same ``_load_eval_labels``
helper) as every other row in this family, so the "reviewed"/"original" label
grading is identical across all five CNN-family rows.

Checkpointing
-------------
Embeddings are computed in batches and written to an on-disk cache
incrementally (``--cache-dir``), keyed by split/arm, so an interrupted run can
be resumed without recomputing already-embedded samples.

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.eval.run_surfperch_relabel_curves \\
        --arm native \\
        --out-json runs/surfperch_relabel_100k_matched_full199825.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bowhead.benchmark.backbones.surfperch_from_audio import SurfPerchFromAudioBackbone
from bowhead.benchmark.probes.detection import DetectionProbe
from bowhead.eval.metrics import score_predictions
from bowhead.eval.raw_audio_loader import parse_event_filename, extract_verified_clip
from bowhead.eval.run_scratch_relabel_curves import _load_eval_labels

_REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_MAT = _REPO_ROOT / "paper/FinalDatabases_/latent_embeddings_3d_train_MATLAB_Angel_Final_26Aug2026.mat"
EVAL_MAT = _REPO_ROOT / "paper/FinalDatabases_/latent_embeddings_3d_eval_8to1_MATLAB_combined.mat"
DEFAULT_CACHE_DIR = _REPO_ROOT / "runs" / "surfperch_embedding_cache"


def _load_train() -> dict:
    d = loadmat(str(TRAIN_MAT), squeeze_me=True, struct_as_record=False)
    f = d["features"]
    t = np.asarray(f.type, dtype=float)
    cd = (t > 0) & (t < 12)
    return dict(
        filenames=np.asarray(d["original_filenames"]),
        label=cd.astype(np.int64),
        tpeak=np.asarray(f.tpeak, dtype=float),
        fpeak=np.asarray(f.fpeak, dtype=float),
        fmin=np.asarray(f.fmin, dtype=float),
        fmax=np.asarray(f.fmax, dtype=float),
    )


def _load_eval() -> dict:
    meta = _load_eval_labels(EVAL_MAT)
    d = loadmat(str(EVAL_MAT), squeeze_me=True, struct_as_record=False)
    f = d["features"]
    meta["tpeak"] = np.asarray(f.tpeak, dtype=float)
    meta["fpeak"] = np.asarray(f.fpeak, dtype=float)
    meta["fmin"] = np.asarray(f.fmin, dtype=float)
    meta["fmax"] = np.asarray(f.fmax, dtype=float)
    return meta


def _embed_split(
    name: str,
    filenames: np.ndarray,
    tpeak: np.ndarray,
    fpeak: np.ndarray,
    fmin: np.ndarray,
    fmax: np.ndarray,
    backbone: SurfPerchFromAudioBackbone,
    cache_path: Path,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (embeddings (N, D) float32, ok_mask (N,) bool, stats dict).

    ``ok_mask[i]`` is False where the audio could not be located at all (e.g.
    missing day-file); those rows are embedded as all-NaN and should be
    excluded by the caller. Resumable via an on-disk .npz cache.
    """
    n = len(filenames)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        emb = cached["emb"].copy()
        ok = cached["ok"].copy()
        verified = cached["verified"].copy()
        done = int(cached["n_done"])
        print(f"  [{name}] resuming from cache: {done:,}/{n:,} already embedded")
    else:
        emb = np.full((n, backbone.embed_dim), np.nan, dtype=np.float32)
        ok = np.zeros(n, dtype=bool)
        verified = np.zeros(n, dtype=bool)
        done = 0

    window_s = backbone.native_window_s
    t0 = time.time()
    batch_audio: list[np.ndarray] = []
    batch_idx: list[int] = []

    def _flush():
        nonlocal batch_audio, batch_idx
        if not batch_audio:
            return
        vecs = backbone.embed(np.stack(batch_audio), batch_size=len(batch_audio))
        for j, v in zip(batch_idx, vecs):
            emb[j] = v
        batch_audio, batch_idx = [], []

    for i in range(done, n):
        try:
            ev = parse_event_filename(str(filenames[i]))
            clip = extract_verified_clip(
                ev, tpeak=float(tpeak[i]), fpeak=float(fpeak[i]),
                fmin=float(fmin[i]), fmax=float(fmax[i]), window_s=window_s,
            )
        except Exception:
            clip = None
        if clip is not None:
            ok[i] = True
            verified[i] = clip.verified
            batch_audio.append(clip.audio)
            batch_idx.append(i)
        if len(batch_audio) >= batch_size:
            _flush()
        if (i + 1) % 2000 == 0 or (i + 1) == n:
            _flush()
            elapsed = time.time() - t0
            rate = (i + 1 - done) / max(elapsed, 1e-6)
            eta_min = (n - i - 1) / max(rate, 1e-6) / 60
            print(f"  [{name}] {i + 1:,}/{n:,}  ok={ok[:i + 1].sum():,}  "
                  f"verified={verified[:i + 1].sum():,}  ({rate:.1f}/s, ETA {eta_min:.1f} min)",
                  flush=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, emb=emb, ok=ok, verified=verified, n_done=i + 1)
    _flush()
    np.savez(cache_path, emb=emb, ok=ok, verified=verified, n_done=n)

    stats = dict(n=n, n_ok=int(ok.sum()), n_verified=int(verified.sum()))
    return emb, ok, stats


def run(arm: str, out_json: Path, cache_dir: Path, batch_size: int) -> None:
    backbone = SurfPerchFromAudioBackbone(arm=arm)

    print(f"Loading training ground truth: {TRAIN_MAT.name}")
    tr = _load_train()
    print(f"  train: {len(tr['filenames']):,} samples, {int(tr['label'].sum()):,} CD")

    print(f"Loading eval ground truth: {EVAL_MAT.name}")
    ev = _load_eval()
    n_flip = int((ev["iscall_reviewed"] != ev["iscall_original"]).sum())
    print(f"  eval: {len(ev['filenames']):,} samples, {n_flip:,} labels changed by review")

    emb_train, ok_train, train_stats = _embed_split(
        "train", tr["filenames"], tr["tpeak"], tr["fpeak"], tr["fmin"], tr["fmax"],
        backbone, cache_dir / f"train_{arm}.npz", batch_size,
    )
    emb_eval, ok_eval, eval_stats = _embed_split(
        "eval", ev["filenames"], ev["tpeak"], ev["fpeak"], ev["fmin"], ev["fmax"],
        backbone, cache_dir / f"eval_{arm}.npz", batch_size,
    )

    X_train, y_train = emb_train[ok_train], tr["label"][ok_train]
    print(f"Fitting detection probe on {ok_train.sum():,}/{len(ok_train):,} located train samples ...")
    probe = DetectionProbe(seed=0).fit(X_train, y_train)

    X_eval = emb_eval[ok_eval]
    probs_located = probe.predict_proba(X_eval)
    probs = np.full(len(ev["filenames"]), np.nan)
    probs[ok_eval] = probs_located

    results: dict = {}
    for cond, y_true_full in (("original", ev["iscall_original"]), ("reviewed", ev["iscall_reviewed"])):
        mask = ok_eval
        m = score_predictions(y_true_full[mask], probs[mask])
        results[cond] = dict(
            roc_auc=m.roc_auc, average_precision=m.average_precision,
            prevalence=m.prevalence, n=m.n,
            precision_at_recall_0_70=m.precision_at_recall(0.70),
            pr_precision=m.pr_precision.tolist(), pr_recall=m.pr_recall.tolist(),
        )
        print(f"  [{cond}] AP={m.average_precision:.4f}  ROC-AUC={m.roc_auc:.4f}  "
              f"prevalence={m.prevalence:.4f}  n={m.n:,}")

    results["_meta"] = dict(
        model=f"surfperch_{arm}", arm=arm,
        train_data=str(TRAIN_MAT), eval_mat=str(EVAL_MAT),
        n_train=train_stats["n"], n_train_located=train_stats["n_ok"], n_train_verified=train_stats["n_verified"],
        n_eval=eval_stats["n"], n_eval_located=eval_stats["n_ok"], n_eval_verified=eval_stats["n_verified"],
        n_labels_changed_by_review=n_flip,
        caveat=(
            "Embeddings computed from REAL raw DASAR audio (not a Griffin-Lim "
            "reconstruction) read from the continuous day-long WAV archive on "
            "/Volumes/Bowhead_WAV, located per-sample from the event filename's "
            "embedded timestamp and locally re-verified against the stored "
            "peak-frequency feature (not every sample's local search finds an "
            "independently-confirming match -- see n_*_verified -- unverified "
            "samples still use the filename-derived position, just without "
            "that extra confirmation). Frozen linear probe fit on all located "
            f"train samples ({train_stats['n_ok']:,}/{train_stats['n']:,}), scored on all "
            f"located eval samples ({eval_stats['n_ok']:,}/{eval_stats['n']:,})."
        ),
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWritten {out_json}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["native", "shifted"], default="native")
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out_json = args.out_json or (_REPO_ROOT / "runs" / f"surfperch_{args.arm}_relabel_100k_matched_full199825.json")
    run(args.arm, out_json, args.cache_dir, args.batch_size)
