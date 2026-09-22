"""Before-vs-after-relabeling ablation for the classical Moan Detector.

Analogous to ``bowhead.eval.run_scratch_relabel_curves``, but for the
rule-based ``bowhead.eval.moan_detector.MultiBandEnergyDetector`` (no
training, no frozen checkpoint): the SAME detector scores the SAME
199,825-sample eval set ONCE, then its precision-recall / FDR-vs-miss curves
are computed twice -- once against the pre-review label set (``type_org``)
and once against the manually reviewed label set (``iscall``) -- isolating
the effect of the dataset relabeling from the detector itself.

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.eval.run_moan_relabel_curves \\
        --eval-mat /path/to/latent_embeddings_3d_eval_8to1_MATLAB_Raquel.mat \\
        --eval-dir /path/to/Evaluation_200K.dir
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bowhead.eval.metrics import score_predictions
from bowhead.eval.moan_detector import MultiBandEnergyDetector
from bowhead.eval.raw_gram_loader import load_raw_grams
from bowhead.eval.run_scratch_relabel_curves import _load_eval_labels

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_JSON = _REPO_ROOT / "runs" / "moan_detector_relabel_100k_matched.json"
BATCH_SIZE = 20_000


def _score_streaming(mat_paths: list[Path], detector: MultiBandEnergyDetector):
    import numpy as np

    n = len(mat_paths)
    probs = np.empty(n, dtype=float)
    t0 = time.time()
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = load_raw_grams(mat_paths[start:end])
        probs[start:end] = detector.score(batch)
        rate = end / (time.time() - t0)
        eta = (n - end) / rate / 60 if rate > 0 else 0
        print(f"  {end:>7,}/{n:,}  ({rate:.0f}/s  ETA {eta:.1f} min)")
    return probs


def run(eval_mat: Path, eval_dir: Path, out_json: Path) -> None:
    print(f"Loading eval labels: {eval_mat}")
    meta = _load_eval_labels(eval_mat)
    filenames = meta["filenames"]
    n_flip = int((meta["iscall_reviewed"] != meta["iscall_original"]).sum())
    print(f"  {len(filenames):,} rows | {n_flip:,} labels changed by review ({n_flip / len(filenames):.2%})")

    mat_paths = [Path(eval_dir) / str(f) for f in filenames]
    missing = [p for p in mat_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing):,}/{len(mat_paths):,} eval spectrogram files not found "
            f"under {eval_dir} (first missing: {missing[0]})."
        )

    detector = MultiBandEnergyDetector()
    print(f"Scoring {len(mat_paths):,} eval spectrograms with MultiBandEnergyDetector ...")
    probs = _score_streaming(mat_paths, detector)

    results: dict = {}
    for name, y_true in (("reviewed", meta["iscall_reviewed"]), ("original", meta["iscall_original"])):
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
        model="moan_detector", eval_mat=str(eval_mat), eval_dir=str(eval_dir),
        n=len(filenames), n_labels_changed_by_review=n_flip,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWritten {out_json}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-mat", required=True, type=Path)
    p.add_argument("--eval-dir", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = p.parse_args()
    run(args.eval_mat, args.eval_dir, args.out_json)


if __name__ == "__main__":
    main()
