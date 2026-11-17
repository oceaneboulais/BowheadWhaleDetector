"""Encoder-free test: is a candidate spectrogram the same image the training row was built from?

``features.fpeak`` / ``features.tpeak`` in the training file come from MATLAB's
``extract_features_from_SNRgram``: Fmax = FF(argmax over rows of the row-wise max), with
FF = dF*(1:nF).  A matching spectrogram must reproduce them exactly (to float precision).
Same-filename-but-different-image (re-centered / regenerated) spectrograms will not.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parents[1]
MAT = REPO / "paper/FinalDatabases_/latent_embeddings_3d_train_MATLAB_Angel_Final_26Aug2026.mat"
THODE = Path("/Users/oceaneboulais/Github/ThodeLab/BCB_Whale_Datasets")
N_SAMPLE = 400


def fpeak_tpeak(gram: np.ndarray, dF: float, dT: float) -> tuple[float, float]:
    g = gram.astype(np.float64) / 5.0
    FF = dF * np.arange(1, g.shape[0] + 1)
    TT = dT * np.arange(1, g.shape[1] + 1)
    fmax = FF[int(np.argmax(g.max(axis=1)))]
    col = g.max(axis=0)
    col = col - col.min()
    return float(fmax), float(TT[int(np.argmax(col))])


def main() -> None:
    m = sio.loadmat(str(MAT), simplify_cells=True)
    fn = np.asarray(m["original_filenames"]).tolist()
    fpk = np.asarray(m["features"]["fpeak"], dtype=float)
    tpk = np.asarray(m["features"]["tpeak"], dtype=float)
    first = {}
    for i, f in enumerate(fn):
        first.setdefault(f, i)
    rng = np.random.default_rng(1)

    def score(pairs, loader):
        okf = okt = n = 0
        for i, key in pairs:
            try:
                gram, dF, dT = loader(key)
            except Exception:
                continue
            f, t = fpeak_tpeak(gram, dF, dT)
            n += 1
            okf += abs(f - fpk[i]) < 0.01
            okt += abs(t - tpk[i]) < 0.01
        return n, okf, okt

    print(f"{'source':58s} {'avail':>7s} {'tested':>7s} {'fpeak match':>12s} {'tpeak match':>12s}")
    npz = np.load(REPO / "data/spectrograms_100k_v2.npz", allow_pickle=True)
    stems = npz["stem"].tolist()
    images = npz["images"]  # decompress once; indexing npz["images"] per sample re-reads the whole array
    pairs = [(first[s + ".mat"], j) for j, s in enumerate(stems) if s + ".mat" in first]
    pick = [pairs[k] for k in rng.choice(len(pairs), min(N_SAMPLE, len(pairs)), replace=False)]
    n, a, b = score(pick, lambda j: (images[j], 3.90625, 0.026))
    print(f"{'repo npz 100k_v2 (assumed dF=3.90625,dT=0.026)':58s} {len(pairs):7d} {n:7d} {a / max(n, 1):12.1%} {b / max(n, 1):12.1%}")

    for d in sorted(os.listdir(THODE)):
        p = THODE / d
        if not p.is_dir():
            continue
        rows = [(first[f], f) for f in os.listdir(p) if f in first]
        if len(rows) < 500:
            continue
        pick = [rows[k] for k in rng.choice(len(rows), min(N_SAMPLE, len(rows)), replace=False)]

        def load(f, p=p):
            r = sio.loadmat(str(p / f))
            return r["SNR_gram"], float(np.squeeze(r["dF"])), float(np.squeeze(r["dT"]))

        n, a, b = score(pick, load)
        print(f"{d[:58]:58s} {len(rows):7d} {n:7d} {a / max(n, 1):12.1%} {b / max(n, 1):12.1%}")


if __name__ == "__main__":
    main()
