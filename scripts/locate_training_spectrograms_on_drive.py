"""Which directory on the attached drive holds the images behind the training UMAP rows?

Samples random training rows; for every candidate directory records (a) the file exists and
(b) its SNR_gram reproduces the stored features.fpeak / features.tpeak (MATLAB
extract_features_from_SNRgram), i.e. it is the SAME image, not just the same filename.
Per-file stat() only -- no directory listing (listing these 100k-file dirs on the drive stalls).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from check_spectrogram_source_by_peak_features import MAT, fpeak_tpeak  # noqa: E402

BASE = Path("/Volumes/R3D_2024_1/BowheadDeepLearningMATLAB/BCB_Whale_Datasets")
DIRS = [
    "Unsupervised_database_Auto_100K_ADG_Y08101214_centered_16Apr2026.dir",
    "Unsupervised_database_Manual_100K_ADG_Y08101214_centered_16Apr2026.dir",
    "Unsupervised_database_Manual_100K_Y08101214_centered.dir",
    "Unsupervised_database_AutoWithAirguns_100K_Y08101214_centered.dir",
    "Unsupervised_database_Evaluation_200K_8Auto1Manual_ADG_Y08101214_centered_06May2026.dir",
]
N = int(os.environ.get("N", "300"))


def main() -> None:
    m = sio.loadmat(str(MAT), simplify_cells=True)
    fn = np.asarray(m["original_filenames"]).tolist()
    fpk = np.asarray(m["features"]["fpeak"], dtype=float)
    tpk = np.asarray(m["features"]["tpeak"], dtype=float)
    typ = np.array([int(f.split("_Type")[1].split(".")[0]) for f in fn])
    rng = np.random.default_rng(7)
    rows = rng.choice(len(fn), N, replace=False)
    print(f"sampled {N} rows: {int((typ[rows] == 0).sum())} filename-Type0, {int((typ[rows] > 0).sum())} Type1-7")
    print(f"{'directory':60s} {'exists':>7s} {'fpeak ok':>9s} {'tpeak ok':>9s}  (of exists)")
    hit_any = np.zeros(N, bool)
    for d in DIRS:
        ex = fo = to = 0
        for k, i in enumerate(rows):
            p = BASE / d / fn[i]
            if not p.exists():
                continue
            ex += 1
            hit_any[k] = True
            try:
                r = sio.loadmat(str(p))
                f, t = fpeak_tpeak(r["SNR_gram"], float(np.squeeze(r["dF"])), float(np.squeeze(r["dT"])))
            except Exception:
                continue
            fo += abs(f - fpk[i]) < 0.01
            to += abs(t - tpk[i]) < 0.01
        print(f"{d[:60]:60s} {ex:7d} {fo:9d} {to:9d}", flush=True)
    print(f"rows found in at least one directory: {int(hit_any.sum())} / {N}")
    miss = [fn[i] for k, i in enumerate(rows) if not hit_any[k]][:8]
    print("examples not found anywhere:", miss)


if __name__ == "__main__":
    main()
