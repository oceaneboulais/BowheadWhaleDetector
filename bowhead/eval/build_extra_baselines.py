"""Evaluate the two NEW classical/transfer baselines on the 100k_matched split.

Uses the SAME leakage-safe grouped date_site split (val_frac=0.15,
test_frac=0.15, seed=0) and the SAME realistic-prevalence resampling
(1 call : 8 non-calls, seed=0) as ``bowhead.train.train_cnn`` so the numbers
here are directly comparable to the scratch/warm-start CNN retrain results in
runs/scratch_100k_matched_retrain/summary.json and
runs/warmstart_100k_matched_retrain/summary.json.

Baselines:
  1. moan_detector  — classical multi-band SNR energy detector
     (bowhead.eval.moan_detector.MultiBandEnergyDetector), a faithful
     port of the Baumgartner & Mussoline (2011) - style detector used in
     matlab/matlab/MultipleBandEnergyDetector.m. No training: scored
     directly on the held-out test split.
  2. birdnet — real pretrained BirdNET/Perch 2.0 (TF-Hub) embeddings,
     extracted from audio RECONSTRUCTED from each SNR-gram via Griffin-Lim
     (bowhead.benchmark.backbones.birdnet_from_gram.BirdNetFromGramBackbone),
     then a frozen linear probe (bowhead.benchmark.probes.detection.DetectionProbe)
     fit on a subsample of the train split and evaluated on a subsample of the
     held-out test split (Griffin-Lim reconstruction is CPU-bound, so the full
     71k/14k train/test split is subsampled for tractability).

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.eval.build_extra_baselines
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from bowhead.data.dataset import SpectrogramDataset
from bowhead.data.splits import grouped_split, make_date_site_group
from bowhead.eval.evaluate import evaluate_scorer
from bowhead.eval.metrics import DetectionMetrics, resample_to_prevalence, score_predictions
from bowhead.eval.moan_detector import MultiBandEnergyDetector

_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = _REPO_ROOT / "data" / "spectrograms_100k_matched.npz"
OUT_JSON = _REPO_ROOT / "runs" / "extra_baselines_100k_matched.json"

TARGET_PREVALENCE = 1.0 / 9.0
SEED = 0
BIRDNET_TRAIN_N = 4000   # subsample sizes for BirdNET (Griffin-Lim is slow)
BIRDNET_TEST_N = 4000


def _metrics_to_dict(m: DetectionMetrics) -> dict:
    d = m.scalar_dict()
    d["precision_at_recall"] = m.precision_at_recall(0.70)
    d["pr_precision"] = m.pr_precision.tolist()
    d["pr_recall"] = m.pr_recall.tolist()
    return d


def _subsample(images, labels, n, seed):
    if len(labels) <= n:
        return images, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels), size=n, replace=False)
    return images[idx], labels[idx]


def main() -> None:
    print(f"Loading {DATA_PATH.name} ...")
    images, labels, metadata = SpectrogramDataset.load_npz(str(DATA_PATH))
    groups = make_date_site_group(metadata["date"], metadata["site"])

    split = grouped_split(labels, groups, val_frac=0.15, test_frac=0.15, seed=SEED)
    print(split.summary(labels, groups))

    test_images, test_labels = images[split.test], labels[split.test]
    train_images, train_labels = images[split.train], labels[split.train]

    results: dict[str, dict] = {}

    # ── 1. Classical moan detector — no training, score full test split ────
    print("\n[moan_detector] Scoring held-out test split ...")
    detector = MultiBandEnergyDetector()
    m = evaluate_scorer(
        detector, test_images, test_labels,
        target_prevalence=TARGET_PREVALENCE, seed=SEED,
    )
    print(f"  ROC-AUC={m.roc_auc:.4f}  AP={m.average_precision:.4f}  "
          f"P@R0.70={m.precision_at_recall(0.70):.4f}  n={m.n}")
    results["moan_detector"] = _metrics_to_dict(m)

    # ── 2. BirdNET-from-gram — subsample for tractability, fit linear probe ─
    print("\n[birdnet] Loading model + reconstructing audio (Griffin-Lim) ...")
    from bowhead.benchmark.backbones.birdnet_from_gram import BirdNetFromGramBackbone

    bb_train_imgs, bb_train_labels = _subsample(
        train_images, train_labels, BIRDNET_TRAIN_N, seed=SEED)
    bb_test_imgs, bb_test_labels = _subsample(
        test_images, test_labels, BIRDNET_TEST_N, seed=SEED + 1)

    backbone = BirdNetFromGramBackbone()

    def _embed_batched(imgs, batch_size=256, tag=""):
        embs = []
        t0 = time.time()
        for start in range(0, len(imgs), batch_size):
            embs.append(backbone.embed(imgs[start:start + batch_size]))
            done = min(start + batch_size, len(imgs))
            if (start // batch_size + 1) % 5 == 0 or done == len(imgs):
                rate = done / max(time.time() - t0, 1e-6)
                print(f"    [{tag}] {done:,}/{len(imgs):,}  ({rate:.1f}/s)")
        return np.concatenate(embs, axis=0)

    print(f"  Extracting BirdNET train embeddings ({len(bb_train_imgs):,} images) ...")
    emb_train = _embed_batched(bb_train_imgs, tag="train")
    print(f"  Extracting BirdNET test embeddings ({len(bb_test_imgs):,} images) ...")
    emb_test = _embed_batched(bb_test_imgs, tag="test")

    from bowhead.benchmark.probes.detection import DetectionProbe

    idx = resample_to_prevalence(bb_test_labels, target_prevalence=TARGET_PREVALENCE, seed=SEED)
    emb_test_r, y_test_r = emb_test[idx], bb_test_labels[idx]

    probe = DetectionProbe(seed=SEED).fit(emb_train, bb_train_labels)
    y_score = probe.predict_proba(emb_test_r)
    m_bn = score_predictions(y_test_r, y_score)
    print(f"  ROC-AUC={m_bn.roc_auc:.4f}  AP={m_bn.average_precision:.4f}  "
          f"P@R0.70={m_bn.precision_at_recall(0.70):.4f}  n={m_bn.n}")
    results["birdnet"] = _metrics_to_dict(m_bn)
    results["birdnet"]["_meta"] = {
        "n_train_probe": int(len(bb_train_imgs)),
        "n_test_subsample": int(len(bb_test_imgs)),
        "caveat": (
            "Audio reconstructed from SNR-gram magnitude via Griffin-Lim "
            "(phase is synthesized, not the true recorded phase); embeddings "
            "from the real pretrained BirdNET/Perch 2.0 TF-Hub model; "
            "frozen linear probe on a subsample (Griffin-Lim reconstruction "
            "is CPU-bound)."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nWritten {OUT_JSON}")


if __name__ == "__main__":
    main()
