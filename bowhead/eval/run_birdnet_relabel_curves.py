"""Before-vs-after-relabeling ablation for BirdNET-from-gram (subsample).

Fits the frozen linear probe on a train subsample of
``data/spectrograms_100k_matched.npz`` (same protocol as
``bowhead.eval.build_extra_baselines``), then embeds + scores a FIXED
subsample of the 199,825-sample eval set ONCE with BirdNET/Perch 2.0
(Griffin-Lim reconstructed audio), and re-grades those same predictions
against both the pre-review (``type_org``) and reviewed (``iscall``) label
sets -- isolating the effect of relabeling from the detector itself.
Griffin-Lim reconstruction is CPU-bound, so only a subsample of the eval set
is scored (same tradeoff as the existing BirdNET baseline panel).

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.eval.run_birdnet_relabel_curves \\
        --eval-mat /path/to/latent_embeddings_3d_eval_8to1_MATLAB_Raquel.mat \\
        --eval-dir /path/to/Evaluation_200K.dir \\
        --n-subsample 4000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bowhead.data.dataset import SpectrogramDataset
from bowhead.data.splits import grouped_split, make_date_site_group
from bowhead.eval.metrics import score_predictions
from bowhead.eval.raw_gram_loader import load_raw_grams
from bowhead.eval.run_scratch_relabel_curves import _load_eval_labels

_REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA_PATH = _REPO_ROOT / "data" / "spectrograms_100k_matched.npz"
DEFAULT_OUT_JSON = _REPO_ROOT / "runs" / "birdnet_relabel_100k_matched.json"
SEED = 0
TRAIN_N = 4000


def _embed_batched(backbone, imgs: np.ndarray, batch_size: int = 256, tag: str = "") -> np.ndarray:
    embs = []
    t0 = time.time()
    for start in range(0, len(imgs), batch_size):
        embs.append(backbone.embed(imgs[start:start + batch_size]))
        done = min(start + batch_size, len(imgs))
        rate = done / max(time.time() - t0, 1e-6)
        eta = (len(imgs) - done) / rate / 60 if rate > 0 else 0
        print(f"    [{tag}] {done:,}/{len(imgs):,}  ({rate:.1f}/s  ETA {eta:.1f} min)")
    return np.concatenate(embs, axis=0)


def _fit_probe():
    from bowhead.benchmark.backbones.birdnet_from_gram import BirdNetFromGramBackbone
    from bowhead.benchmark.probes.detection import DetectionProbe

    print(f"Loading {TRAIN_DATA_PATH.name} for probe training split ...")
    images, labels, metadata = SpectrogramDataset.load_npz(str(TRAIN_DATA_PATH))
    groups = make_date_site_group(metadata["date"], metadata["site"])
    split = grouped_split(labels, groups, val_frac=0.15, test_frac=0.15, seed=SEED)
    train_images, train_labels = images[split.train], labels[split.train]

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train_labels), size=min(TRAIN_N, len(train_labels)), replace=False)
    train_images, train_labels = train_images[idx], train_labels[idx]

    print("Loading BirdNET (TF-Hub) ...")
    backbone = BirdNetFromGramBackbone()
    print(f"Reconstructing audio + embedding {len(train_images):,} train images ...")
    emb_train = _embed_batched(backbone, train_images, tag="train-probe")
    probe = DetectionProbe(seed=SEED).fit(emb_train, train_labels)
    return backbone, probe


def run(eval_mat: Path, eval_dir: Path, out_json: Path, n_subsample: int) -> None:
    print(f"Loading eval labels: {eval_mat}")
    meta = _load_eval_labels(eval_mat)
    filenames = meta["filenames"]
    n_flip_full = int((meta["iscall_reviewed"] != meta["iscall_original"]).sum())
    print(f"  {len(filenames):,} rows | {n_flip_full:,} labels changed by review "
          f"({n_flip_full / len(filenames):.2%})")

    rng = np.random.default_rng(SEED + 1)
    sub_idx = rng.choice(len(filenames), size=min(n_subsample, len(filenames)), replace=False)
    sub_filenames = filenames[sub_idx]
    sub_reviewed = meta["iscall_reviewed"][sub_idx]
    sub_original = meta["iscall_original"][sub_idx]
    n_flip_sub = int((sub_reviewed != sub_original).sum())

    mat_paths = [Path(eval_dir) / str(f) for f in sub_filenames]
    missing = [p for p in mat_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing):,}/{len(mat_paths):,} eval spectrogram files not found "
            f"under {eval_dir} (first missing: {missing[0]})."
        )

    backbone, probe = _fit_probe()

    print(f"Reconstructing audio + embedding {len(mat_paths):,} eval subsample images ...")
    test_images = load_raw_grams(mat_paths)
    emb_test = _embed_batched(backbone, test_images, tag="eval-subsample")
    probs = probe.predict_proba(emb_test)

    results: dict = {}
    for name, y_true in (("reviewed", sub_reviewed), ("original", sub_original)):
        m = score_predictions(y_true, probs)
        results[name] = dict(
            roc_auc=m.roc_auc, average_precision=m.average_precision,
            prevalence=m.prevalence, n=m.n,
            precision_at_recall_0_70=m.precision_at_recall(0.70),
            pr_precision=m.pr_precision.tolist(), pr_recall=m.pr_recall.tolist(),
        )
        print(f"  [{name}] AP={m.average_precision:.4f}  ROC-AUC={m.roc_auc:.4f}  "
              f"prevalence={m.prevalence:.4f}  n={m.n:,}")

    results["_meta"] = dict(
        model="birdnet", eval_mat=str(eval_mat), eval_dir=str(eval_dir),
        train_data=str(TRAIN_DATA_PATH), n=len(sub_filenames), n_train=TRAIN_N,
        n_full_eval=len(filenames),
        n_labels_changed_by_review=n_flip_sub,
        n_labels_changed_by_review_full_eval=n_flip_full,
        caveat=(
            "Audio reconstructed from SNR-gram magnitude via Griffin-Lim (phase is "
            "synthesized, not the true recorded phase); embeddings from the real "
            "pretrained BirdNET/Perch 2.0 TF-Hub model; frozen linear probe fit on a "
            f"{TRAIN_N}-image subsample of the 100k_matched train split, scored on a "
            f"{len(sub_filenames):,}-image subsample of the 199,825-image eval set "
            "(Griffin-Lim reconstruction is CPU-bound)."
        ),
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWritten {out_json}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-mat", required=True, type=Path)
    p.add_argument("--eval-dir", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--n-subsample", type=int, default=4000)
    args = p.parse_args()
    run(args.eval_mat, args.eval_dir, args.out_json, args.n_subsample)


if __name__ == "__main__":
    main()
