"""Waveform preprocessing for bird/whale foundation models.

Bowhead calls live at 25-500 Hz (1 kHz sample rate); BirdNET/Perch are most
sensitive in the mid-frequency band of birdsong (~1-8 kHz). We support two arms,
reported side by side:

  * NATIVE  -- bandpass + resample to the model's sample rate, no pitch change.
  * SHIFTED -- the documented "speed-up" trick (NARW upcalls, bioRxiv
    2025.07.11.664307): speed the signal up by ~10x so 25-500 Hz maps to
    ~250-5000 Hz, landing in the bird-sensitive band. That paper found speed-up
    (which scales time AND frequency) outperforms pure pitch-shift.

Speed-up by factor f is implemented by reinterpreting the clip as if recorded at
``sr * f`` and then resampling to the target rate: this compresses duration by f
and multiplies all frequencies by f -- exactly the "play it faster" effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, resample, sosfiltfilt


@dataclass
class FrequencyShiftConfig:
    native_sr: int = 1000           # bowhead DASAR sample rate
    speedup: float = 1.0            # 1.0 = native arm; ~10.0 = shifted arm
    bandpass: tuple[float, float] | None = (25.0, 500.0)  # Hz, on native signal
    target_sr: int = 48000          # BirdNET=48k, Perch=32k (set per model)
    clip_seconds: float = 3.0       # model window: BirdNET=3s, Perch=5s
    mean_normalize: bool = True     # divide by mean abs amplitude (NARW recipe)


def _bandpass(wav: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    nyq = sr / 2.0
    hi = min(hi, nyq * 0.999)
    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, wav)


def _resample_to(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav
    n_out = int(round(len(wav) * sr_out / sr_in))
    return resample(wav, n_out)


def _fit_to_length(wav: np.ndarray, n: int) -> np.ndarray:
    """Center the signal in a length-``n`` window (zero-pad or center-crop)."""
    if len(wav) == n:
        return wav
    if len(wav) < n:
        pad = n - len(wav)
        left = pad // 2
        return np.pad(wav, (left, pad - left))
    start = (len(wav) - n) // 2
    return wav[start:start + n]


def preprocess_clip(wav: np.ndarray, cfg: FrequencyShiftConfig) -> np.ndarray:
    """Turn a raw mono clip into a fixed-length waveform at ``cfg.target_sr``.

    Steps: (1) optional bandpass on the native signal, (2) optional mean-abs
    normalize, (3) speed-up by ``cfg.speedup`` (frequency x speedup), (4) resample
    to ``target_sr``, (5) center to ``clip_seconds`` samples. Returns float32.
    """
    wav = np.asarray(wav, dtype=np.float64).ravel()

    if cfg.bandpass is not None:
        wav = _bandpass(wav, cfg.native_sr, *cfg.bandpass)

    if cfg.mean_normalize:
        m = float(np.mean(np.abs(wav)))
        if m > 1e-12:
            wav = wav / m

    # Speed-up: treat samples as if captured at sr*speedup, then resample.
    effective_sr = int(round(cfg.native_sr * cfg.speedup))
    wav = _resample_to(wav, effective_sr, cfg.target_sr)

    n = int(round(cfg.clip_seconds * cfg.target_sr))
    wav = _fit_to_length(wav, n)
    return wav.astype(np.float32)


def native_arm(native_sr: int, target_sr: int, clip_seconds: float) -> FrequencyShiftConfig:
    return FrequencyShiftConfig(
        native_sr=native_sr, speedup=1.0, target_sr=target_sr, clip_seconds=clip_seconds
    )


def shifted_arm(
    native_sr: int, target_sr: int, clip_seconds: float, speedup: float = 10.0
) -> FrequencyShiftConfig:
    return FrequencyShiftConfig(
        native_sr=native_sr, speedup=speedup, target_sr=target_sr, clip_seconds=clip_seconds
    )
