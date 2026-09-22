"""Before-vs-after relabeling comparison for the scratch ("cold start") CNN.

Analogous to ``bowhead/benchmark/run_ae_knn_curves.py``'s reviewed-vs-original
comparison, but for the CNN trained directly on raw SNR-gram images instead of
AE latents + kNN vote.

Reuses the SAME scratch CNN checkpoint trained on
``data/spectrograms_100k_matched.npz`` (``runs/scratch_100k_matched_retrain/best.pt``)
and scores it ONCE on the raw SNR-gram images referenced by the MATLAB-exported
eval-set metadata file (``latent_embeddings_3d_eval_8to1_MATLAB_Raquel.mat``),
which carries BOTH the pre-review label (``type_org``) and the manually
reviewed label (``iscall``) for each of the 199,825 evaluation spectrograms.
Because the model's predicted probabilities don't change between the two
conditions, any difference in the resulting PR/ROC curves isolates the effect
of the dataset relabeling itself, not the model.

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.eval.run_scratch_relabel_curves \\
        --eval-mat /path/to/latent_embeddings_3d_eval_8to1_MATLAB_Raquel.mat \\
        --eval-dir /path/to/Evaluation_200K.dir \\
        --ckpt runs/scratch_100k_matched_retrain/best.pt \\
        --out-json runs/scratch_cnn_relabel_100k_matched.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bowhead.config import best_device
from bowhead.eval.metrics import score_predictions
from bowhead.eval.score_eval_dataset import GRAMS_1CH, _load_model, _score_model

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = _REPO_ROOT / "runs" / "scratch_100k_matched_retrain" / "best.pt"
DEFAULT_OUT_JSON = _REPO_ROOT / "runs" / "scratch_cnn_relabel_100k_matched.json"


def _load_eval_labels(mat_path: Path) -> dict:
    """Load per-file identifiers + both label sets from the MATLAB eval export.

    Expects the same ``features`` struct (``iscall``, ``type_org``) and
    ``original_filenames`` fields used by ``bowhead.eval.ae_knn_baseline``.
    """
    d = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    feat = d["features"]
    type_org = np.asarray(feat.type_org).astype(int)
    return dict(
        filenames=np.asarray(d["original_filenames"]),
        iscall_reviewed=np.asarray(feat.iscall).astype(int),
        iscall_original=(type_org > 0).astype(int),
    )


def run(
    eval_mat: Path,
    eval_dir: Path,
    ckpt: Path,
    out_json: Path,
    device: str | None = None,
) -> None:
    device = device or best_device()
    print(f"Device: {device}")

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
            f"under {eval_dir} (first missing: {missing[0]}). Is the drive holding "
            f"the Evaluation_200K.dir mounted?"
        )

    print(f"Loading scratch CNN checkpoint: {ckpt}")
    model = _load_model(Path(ckpt), device)

    print(f"Scoring {len(mat_paths):,} eval spectrograms (single forward pass) ...")
    probs = _score_model(model, mat_paths, meta["iscall_reviewed"], device, grams=GRAMS_1CH)

    results: dict = {}
    for name, y_true in (
        ("reviewed", meta["iscall_reviewed"]),
        ("original", meta["iscall_original"]),
    ):
        m = score_predictions(y_true, probs)
        results[name] = dict(
            roc_auc=m.roc_auc,
            average_precision=m.average_precision,
            prevalence=m.prevalence,
            n=m.n,
            precision_at_recall_0_70=m.precision_at_recall(0.70),
            pr_precision=m.pr_precision.tolist(),
            pr_recall=m.pr_recall.tolist(),
        )
        print(f"  [{name:8s}] AP={m.average_precision:.4f}  ROC-AUC={m.roc_auc:.4f}  "
              f"prevalence={m.prevalence:.4f}  n={m.n:,}")

    results["_meta"] = dict(
        ckpt=str(ckpt),
        eval_mat=str(eval_mat),
        eval_dir=str(eval_dir),
        train_data="data/spectrograms_100k_matched.npz",
        n=len(filenames),
        n_labels_changed_by_review=n_flip,
    )

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWritten {out_json}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-mat", type=Path, required=True,
                   help="MATLAB-exported eval metadata .mat "
                        "(latent_embeddings_3d_eval_8to1_MATLAB_Raquel.mat)")
    p.add_argument("--eval-dir", type=Path, required=True,
                   help="Directory holding the raw eval .mat spectrogram files "
                        "(Evaluation_200K.dir)")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help="Scratch CNN checkpoint (default: %(default)s)")
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        eval_mat=args.eval_mat,
        eval_dir=args.eval_dir,
        ckpt=args.ckpt,
        out_json=args.out_json,
        device=args.device,
    )
