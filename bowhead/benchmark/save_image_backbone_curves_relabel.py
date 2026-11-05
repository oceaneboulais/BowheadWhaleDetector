"""ResNet-18 / EfficientNet-B0 linear probes, scored ONCE on the full
199,825-image independent evaluation set and re-graded against BOTH the
original (pre-review, `type_org`) and manually reviewed (post-review,
`iscall`) label sets pulled from the same Oct-21 MATLAB export used
elsewhere in this paper (``latent_embeddings_3d_eval_8to1_MATLAB_combined_102126.mat``).

Mirrors bowhead.eval.run_scratch_relabel_curves's "score once, grade twice"
protocol so results are directly comparable to the scratch/warm-start CNN
relabel rows: the probe is trained on the SAME 100k matched training draw
(data/spectrograms_100k_matched.npz, original labels, unaffected by the
eval-set review), then its probabilities on the test set don't change
between conditions -- only the labels used to grade them do.

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.benchmark.save_image_backbone_curves_relabel \\
        --eval-mat /path/to/latent_embeddings_3d_eval_8to1_MATLAB_combined_102126.mat \\
        --eval-dir /path/to/Evaluation_200K.dir \\
        --matched-train-npz data/spectrograms_100k_matched.npz \\
        --npz-out runs/pr_curves_image_backbones_combined102126.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bowhead.config import best_device
from bowhead.eval.metrics import score_predictions
from bowhead.benchmark.probes.detection import DetectionProbe
from bowhead.benchmark.save_image_backbone_curves import _load_images, _extract

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ_OUT = _REPO_ROOT / "runs" / "pr_curves_image_backbones_combined102126.npz"


def _load_eval_labels(mat_path: Path) -> dict:
    d = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    feat = d["features"]
    type_org = np.asarray(feat.type_org).astype(int)
    return dict(
        filenames=np.asarray(d["original_filenames"]),
        iscall_reviewed=np.asarray(feat.iscall).astype(int),
        iscall_original=(type_org > 0).astype(int),
    )


def run(eval_mat: Path, eval_dir: Path, matched_train_npz: Path, npz_out: Path,
        backbones: list[str], device: str | None, batch_size: int = 32) -> None:
    device = device or best_device()
    print(f"Device: {device}")

    print(f"Loading eval labels: {eval_mat}")
    meta = _load_eval_labels(eval_mat)
    filenames = meta["filenames"]
    n_flip = int((meta["iscall_reviewed"] != meta["iscall_original"]).sum())
    print(f"  {len(filenames):,} rows | {n_flip:,} labels changed by review "
          f"({n_flip / len(filenames):.2%})")

    paths = [Path(eval_dir) / str(f) for f in filenames]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing):,}/{len(paths):,} eval spectrogram files not found "
            f"under {eval_dir} (first missing: {missing[0]})."
        )

    print(f"Loading matched training set {matched_train_npz.name} ...")
    d = np.load(str(matched_train_npz), allow_pickle=True)
    train_imgs = d["images"].astype(np.float32)
    train_labels = d["label"].astype(np.int64)
    print(f"  train: {train_imgs.shape}")

    print(f"Loading all {len(paths):,} eval images ...")
    test_imgs = _load_images(paths, np.arange(len(paths)))
    print(f"  test:  {test_imgs.shape}")

    from bowhead.benchmark.backbones.image import load_image_backbone

    save_dict: dict = {"n": len(paths), "n_labels_changed_by_review": n_flip}
    for bb_name in backbones:
        print(f"\n[{bb_name}] Loading backbone ...")
        bb = load_image_backbone(bb_name, device=device)

        print(f"  Extracting train embeddings ({len(train_imgs):,} images) ...")
        emb_train = _extract(bb, train_imgs, batch_size=batch_size)
        print(f"  Extracting test embeddings ({len(test_imgs):,} images) ...")
        emb_test = _extract(bb, test_imgs, batch_size=batch_size)

        print("  Fitting full-train detection probe (original training labels) ...")
        probe = DetectionProbe(seed=0).fit(emb_train, train_labels)
        probs = probe.predict_proba(emb_test)

        for cond, y_true in (
            ("original", meta["iscall_original"]),
            ("reviewed", meta["iscall_reviewed"]),
        ):
            m = score_predictions(y_true, probs)
            print(f"  {bb_name} [{cond:8s}] AP={m.average_precision:.4f}  "
                  f"ROC-AUC={m.roc_auc:.4f}  prevalence={m.prevalence:.4f}  n={m.n:,}")
            save_dict[f"{bb_name}_{cond}_precision"] = m.pr_precision
            save_dict[f"{bb_name}_{cond}_recall"] = m.pr_recall
            save_dict[f"{bb_name}_{cond}_ap"] = m.average_precision
            save_dict[f"{bb_name}_{cond}_roc_auc"] = m.roc_auc
            save_dict[f"{bb_name}_{cond}_prevalence"] = m.prevalence
            save_dict[f"{bb_name}_{cond}_n"] = m.n

    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(npz_out), **save_dict)
    print(f"\nSaved -> {npz_out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-mat", type=Path, required=True)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--matched-train-npz", type=Path,
                   default=_REPO_ROOT / "data" / "spectrograms_100k_matched.npz")
    p.add_argument("--npz-out", type=Path, default=DEFAULT_NPZ_OUT)
    p.add_argument("--backbones", nargs="+", default=["resnet18", "efficientnet_b0"])
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(eval_mat=args.eval_mat, eval_dir=args.eval_dir,
        matched_train_npz=args.matched_train_npz, npz_out=args.npz_out,
        backbones=args.backbones, device=args.device, batch_size=args.batch_size)
