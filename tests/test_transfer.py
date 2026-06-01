"""Transfer-scaffold tests that need no TF/librosa (preprocess + probe logic).

Run:  PYTHONPATH=. /usr/local/bin/python3.8 tests/test_transfer.py
"""

from __future__ import annotations

import numpy as np

from bowhead.transfer.preprocess import native_arm, shifted_arm, preprocess_clip
from bowhead.transfer.probe import few_shot_eval


def _peak_hz(wav: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(wav))
    freqs = np.fft.rfftfreq(len(wav), 1 / sr)
    return float(freqs[np.argmax(spec)])


def test_speedup_shifts_frequency():
    native_sr, target_sr, clip_s, f0 = 1000, 48000, 3.0, 200.0
    t = np.arange(int(clip_s * native_sr)) / native_sr
    tone = np.sin(2 * np.pi * f0 * t)

    native = preprocess_clip(tone, native_arm(native_sr, target_sr, clip_s))
    shifted = preprocess_clip(tone, shifted_arm(native_sr, target_sr, clip_s, speedup=10.0))

    assert len(native) == int(clip_s * target_sr)
    assert len(shifted) == int(clip_s * target_sr)

    # native arm preserves the tone; shifted arm multiplies it by ~10x.
    assert abs(_peak_hz(native, target_sr) - f0) < 15, _peak_hz(native, target_sr)
    assert abs(_peak_hz(shifted, target_sr) - f0 * 10) < 60, _peak_hz(shifted, target_sr)


def test_few_shot_eval_separable():
    rng = np.random.default_rng(0)
    d = 16

    def blobs(n):
        pos = rng.normal(+1.0, 1.0, size=(n, d))
        neg = rng.normal(-1.0, 1.0, size=(n, d))
        X = np.vstack([pos, neg])
        y = np.r_[np.ones(n), np.zeros(n)].astype(int)
        return X, y

    Xtr, ytr = blobs(100)
    Xte, yte = blobs(200)
    res = few_shot_eval(Xtr, ytr, Xte, yte, k_values=(4, 8, 16, 32), repeats=5, seed=0)

    # separable blobs -> high AUC, and more shots shouldn't hurt much
    assert res.roc_auc_mean[32] > 0.9, res.roc_auc_mean
    assert res.roc_auc_mean[4] > 0.75, res.roc_auc_mean
    print(res.table("synthetic"))


if __name__ == "__main__":
    test_speedup_shifts_frequency()
    test_few_shot_eval_separable()
    print("\nTRANSFER SCAFFOLD TESTS PASSED")
