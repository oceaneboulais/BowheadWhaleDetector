"""Full-protocol BirdNET/Perch 2.0 relabel ablation (no subsampling).

Unlike ``bowhead.eval.run_birdnet_relabel_curves`` (which restricts both the
train and eval sets to small subsamples for tractability, since Griffin-Lim
audio reconstruction is CPU-bound), this script scores BirdNET/Perch 2.0
under the SAME protocol used for the ResNet-18/EfficientNet-B0 relabel runs
(``bowhead.benchmark.save_image_backbone_curves_relabel``): the probe is
fit on the FULL ``data/spectrograms_100k_matched.npz`` training draw (all
images, no subsampling), and scored ONCE on the FULL 199,825-image
independent evaluation set, with predictions re-graded against both the
original (pre-review, ``type_org``) and manually reviewed (post-review,
``iscall``) label sets from the Oct-21 MATLAB export.

This is substantially slower than the subsampled script (Griffin-Lim
reconstruction + TF-Hub inference on ~300k images total instead of 8k), but
makes BirdNET/Perch 2.0 directly protocol-matched to the other three
CNN-family rows (Supervised CNN, ResNet-18, EfficientNet-B0).

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.eval.run_birdnet_relabel_curves_full \\
        --eval-mat /path/to/latent_embeddings_3d_eval_8to1_MATLAB_combined_102126.mat \\
        --eval-dir /path/to/Evaluation_200K.dir \\
        --matched-train-npz data/spectrograms_100k_matched.npz \\
        --out-json runs/birdnet_relabel_100k_matched_full199825.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bowhead.eval.metrics import score_predictions
from bowhead.eval.raw_gram_loader import load_raw_grams
from bowhead.eval.run_scratch_relabel_curves import _load_eval_labels

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_NPZ = _REPO_ROOT / "data" / "spectrograms_100k_matched.npz"
DEFAULT_OUT_JSON = _REPO_ROOT / "runs" / "birdnet_relabel_100k_matched_full199825.json"


def _embed_batched(backbone, imgs: np.ndarray, batch_size: int = 256, tag: str = "") -> np.ndarray:
    embs = []
    t0 = time.time()
    for start in range(0, len(imgs), batch_size):
        embs.append(backbone.embed(imgs[start:start + batch_size]))
        done = min(start + batch_size, len(imgs))
        rate = done / max(time.time() - t0, 1e-6)
        eta_min = (len(imgs) - done) / rate / 60 if rate > 0 else 0
        print(f"    [{tag}] {done:,}/{len(imgs):,}  ({rate:.2f}/s  ETA {eta_min:.1f} min)", flush=True)
    return np.concatenate(embs, axis=0)


def run(eval_mat: Path, eval_dir: Path, matched_train_npz: Path, out_json: Path) -> None:
    from bowhead.benchmark.backbones.birdnet_from_gram import BirdNetFromGramBackbone
    from bowhead.benchmark.probes.detection import DetectionProbe

    print(f"Loading eval labels: {eval_mat}")
    meta = _load_eval_labels(eval_mat)
    filenames = meta["filenames"]
    n_flip = int((meta["iscall_reviewed"] != meta["iscall_original"]).sum())
    print(f"  {len(filenames):,} rows | {n_flip:,} labels changed by review "
          f"({n_flip / len(filenames):.2%})")

    mat_paths = [Path(eval_dir) / str(f) for f in filenames]
    missing = [p for p in mat_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing):,}/{len(mat_paths):,} eval spectrogram files not found "
            f"under {eval_dir} (first missing: {missing[0]})."
        )

    print(f"Loading full matched training set {matched_train_npz.name} ...")
    d = np.load(str(matched_train_npz), allow_pickle=True)
    train_images = d["images"].astype(np.float32)
    train_labels = d["label"].astype(np.int64)
    print(f"  train: {train_images.shape}")

    print("Loading BirdNET/Perch 2.0 (TF-Hub) ...")
    backbone = BirdNetFromGramBackbone()

    print(f"Reconstructing audio + embedding {len(train_images):,} train images ...")
    emb_train = _embed_batched(backbone, train_images, tag="train-full")
    probe = DetectionProbe(seed=0).fit(emb_train, train_labels)

    print(f"Loading raw SNR-grams for all {len(mat_paths):,} eval images ...")
    test_images = load_raw_grams(mat_paths)

    print(f"Reconstructing audio + embedding {len(test_images):,} eval images ...")
    emb_test = _embed_batched(backbone, test_images, tag="eval-full")
    probs = probe.predict_proba(emb_test)

    results: dict = {}
    for name, y_true in (("original", meta["iscall_original"]), ("reviewed", meta["iscall_reviewed"])):
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
        train_data=str(matched_train_npz), n=len(filenames), n_train=len(train_images),
        n_full_eval=len(filenames),
        n_labels_changed_by_review=n_flip,
        caveat=(
            "Audio reconstructed from SNR-gram magnitude via Griffin-Lim (phase is "
            "synthesized, not the true recorded phase); embeddings from the real "
            "pretrained BirdNET/Perch 2.0 TF-Hub model; frozen linear probe fit on the "
            f"full {len(train_images):,}-image 100k_matched train draw (same protocol as "
            "the ResNet-18/EfficientNet-B0 relabel runs), scored on the full "
            f"{len(filenames):,}-image eval set -- fully protocol-matched to the other "
            "three CNN-family rows."
        ),
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWritten {out_json}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-mat", required=True, type=Path)
    p.add_argument("--eval-dir", required=True, type=Path)
    p.add_argument("--matched-train-npz", type=Path, default=DEFAULT_TRAIN_NPZ)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = p.parse_args()
    run(args.eval_mat, args.eval_dir, args.matched_train_npz, args.out_json)


if __name__ == "__main__":
    main()


