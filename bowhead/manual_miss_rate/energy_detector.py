"""Port of matlab/matlab/MultipleBandEnergyDetector.m.

Only the fields consumed downstream by master_compute_manual_miss_rate.m
(``detect.tstart`` / ``detect.tend``, in seconds relative to the start of the
input chunk) are computed. The original MATLAB function also derives
``fmin``/``fmax``/``dB_RMS``/absolute timestamps per detection, but none of
those feed into evaluate_overlap_between_manual_automated.m, so they are
intentionally omitted here -- this does not change the detection timing
logic in any way, only what is reported about each detection.

The detector is a bank of overlapping-frequency-band CFAR sub-detectors, each
tracked through a 4-state machine (OFF / POSSIBLE_ON / ON / POSSIBLE_OFF).
An overall detection starts when the first sub-detector goes ON and ends when
the last active sub-detector has been below threshold for longer than
`TolTime`. This state machine is inherently sequential, so the hot loop is
JIT-compiled with numba when available (falling back transparently, if
unavailable, to plain interpreted Python -- correct either way, just much
slower without numba).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised in environments without numba
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def _decorator(func):
            return func
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _decorator


# Sub-detector states (mirrors the MATLAB STATE struct)
_OFF = -2
_POSSIBLE_ON = 1
_ON = 2
_POSSIBLE_OFF = -1


@dataclass
class EnergyDetectorParams:
    flo_det: float = 25.0
    fhi_det: float = 500.0
    bandwidth: float = 37.0
    Nfft: int = 256
    Fs: float = 1000.0
    ovlap: float = 0.75
    threshold: float = 5.0          # dB above adaptive background
    eq_time: float = 23.8           # seconds
    burn_in_time: float = 0.25      # minutes
    TolTime: float = 0.05           # seconds
    MinTime: float = 0.1            # seconds
    MaxTime: float = 10.0           # seconds


def _matlab_hanning(n: int) -> np.ndarray:
    """Matches MATLAB's `hanning(N)` exactly (N+1 denominator, non-zero endpoints).

    NOTE: this is deliberately NOT the same window as MATLAB's `hann(N)` (which
    matches numpy/scipy's default Hann window). The original detector uses
    `hanning`, so we must too, or the spectral leakage characteristics differ.
    """
    k = np.arange(1, n + 1)
    return 0.5 * (1 - np.cos(2 * np.pi * k / (n + 1)))


def _compute_psd(x: np.ndarray, params: EnergyDetectorParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (freqs, times, psd[freq, time]), matching MATLAB's 4-output spectrogram(x,hanning(Nfft),...).

    Implemented by hand (rather than via scipy.signal.spectrogram) so we have
    exact control over framing: no start/end zero-padding, hop = Nfft-noverlap,
    and per-segment time stamped at the window's center -- all matching
    MATLAB's default spectrogram() behavior. The absolute PSD scaling constant
    does not affect detection decisions (see module docstring / overlap.py),
    so only the window shape and framing need to match exactly.
    """
    Nfft = params.Nfft
    noverlap = round(params.ovlap * Nfft)
    hop = Nfft - noverlap
    n_seg = 1 + (len(x) - Nfft) // hop if len(x) >= Nfft else 0

    window = _matlab_hanning(Nfft)
    win_norm = params.Fs * np.sum(window ** 2)

    # Build all overlapping segments at once via stride tricks (no copies).
    starts = np.arange(n_seg) * hop
    segments = np.lib.stride_tricks.sliding_window_view(x, Nfft)[starts]  # (n_seg, Nfft)
    segments = segments * window  # broadcast

    spec = np.fft.rfft(segments, n=Nfft, axis=1)  # (n_seg, Nfft//2+1)
    psd = (np.abs(spec) ** 2) / win_norm
    # One-sided scaling: double all bins except DC (and Nyquist, if Nfft even).
    if Nfft % 2 == 0:
        psd[:, 1:-1] *= 2
    else:
        psd[:, 1:] *= 2

    freqs = np.fft.rfftfreq(Nfft, d=1.0 / params.Fs)
    times = (starts + Nfft / 2) / params.Fs
    return freqs, times, psd.T  # psd as (freq, time) to match scipy's convention


@njit(cache=True)
def _run_state_machine(
    detect_db: np.ndarray,   # (Ndet, Ncol) dB power per sub-band per time column
    Ndet: int,
    Ncol: int,
    Iburn: int,
    threshold: float,
    alpha: float,
    Imin_time: int,
    Itol_time: int,
    Imax_time: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    out_tstart = np.zeros(Ncol, dtype=np.int64)
    out_tend = np.zeros(Ncol, dtype=np.int64)
    count = 0

    status = np.full(Ndet, _OFF, dtype=np.int64)
    tstart = np.zeros(Ndet, dtype=np.int64)
    tend = np.zeros(Ndet, dtype=np.int64)
    band_eq = np.empty(Ndet, dtype=np.float64)
    # Burn-in equalization: median of each band's power over the first
    # Iburn columns, matching `eq=median(B(:,1:Iburn),2)` in the MATLAB source.
    for k in range(Ndet):
        band_eq[k] = np.median(detect_db[k, :Iburn]) if Iburn > 0 else detect_db[k, 0]

    active_detectors = 0
    tstart_total = 0
    tend_total = 0
    write_flag = False

    for col in range(Iburn, Ncol):
        global_reset = False
        for j in range(Ndet):
            if write_flag or global_reset:
                continue
            val = detect_db[j, col]
            criteria = band_eq[j] + threshold
            if val >= criteria:
                if status[j] == _OFF:
                    status[j] = _POSSIBLE_ON
                    tstart[j] = col
                    tend[j] = col
                elif status[j] == _POSSIBLE_ON:
                    if (col - tstart[j]) >= Imin_time:
                        status[j] = _ON
                        active_detectors += 1
                        if active_detectors == 1:
                            tstart_total = tstart[j]
                elif status[j] == _ON:
                    if (col - tstart_total) >= Imax_time:
                        active_detectors = 0
                        tend[j] = col
                        tend_total = tend[j]
                        write_flag = True
                        if not global_reset:
                            for k in range(Ndet):
                                status[k] = _OFF
                                band_eq[k] = 0.75 * detect_db[k, col] + 0.25 * band_eq[k]
                                tend[k] = col
                            global_reset = True
                elif status[j] == _POSSIBLE_OFF:
                    status[j] = _ON
                    tend[j] = col
            else:
                if status[j] == _OFF:
                    band_eq[j] = alpha * band_eq[j] + (1.0 - alpha) * val
                elif status[j] == _ON:
                    tend[j] = col
                    status[j] = _POSSIBLE_OFF
                elif status[j] == _POSSIBLE_ON:
                    status[j] = _OFF
                    tstart[j] = 0
                    tend[j] = 0
                elif status[j] == _POSSIBLE_OFF:
                    if tend[j] != 0 and (tend[j] + Itol_time) <= col:
                        status[j] = _OFF
                        active_detectors -= 1
                        if active_detectors == 0:
                            if (Imin_time + Itol_time) <= (tend[j] - tstart_total):
                                tend_total = tend[j]
                                write_flag = True
                                global_reset = True
                            else:
                                # Detection too short: discard silently (matches
                                # MATLAB's inline reset_detect(), which does NOT
                                # set global_reset, so remaining sub-detectors
                                # in this same column are still processed below).
                                active_detectors = 0
                                tend_total = 0
                                tstart_total = 0
                                write_flag = False
                                for k in range(Ndet):
                                    tstart[k] = 0
                                    tend[k] = 0
                                    status[k] = _OFF

        if write_flag:
            out_tstart[count] = tstart_total
            out_tend[count] = tend_total
            count += 1
            # reset_detect()
            active_detectors = 0
            tend_total = 0
            tstart_total = 0
            write_flag = False
            for k in range(Ndet):
                tstart[k] = 0
                tend[k] = 0
                status[k] = _OFF

    return out_tstart, out_tend, count


def multiple_band_energy_detector(x: np.ndarray, params: EnergyDetectorParams) -> dict:
    """Port of MultipleBandEnergyDetector.m, returning only tstart/tend (seconds).

    `x` is a 1-D real time series sampled at params.Fs.
    """
    flo = np.arange(params.flo_det, params.fhi_det + 1e-9, 0.5 * params.bandwidth)
    fhi = flo + params.bandwidth
    Iflo = np.ceil(flo * params.Nfft / params.Fs).astype(int)
    Ifhi_raw = fhi * params.Nfft / params.Fs

    good = Ifhi_raw <= (params.fhi_det * params.Nfft / params.Fs)
    Ifhi = np.ceil(Ifhi_raw[good]).astype(int)
    Iflo = Iflo[good]
    Ndet = len(Ifhi)
    if Ndet == 0:
        return {"tstart": np.array([]), "tend": np.array([])}

    freqs, times, psd = _compute_psd(x, params)
    dF = freqs[1] - freqs[0]

    span_lo, span_hi = Iflo[0], Ifhi[-1]
    B = psd[span_lo : span_hi + 1, :]  # MATLAB Ispan=Iflo(1):Ifhi(end), inclusive
    B_db = 10.0 * np.log10(np.clip(B, 1e-300, None))

    # Re-index band edges relative to the trimmed B (MATLAB: Ifhi-=Iflo(1)-1 etc.)
    Iflo_rel = Iflo - span_lo
    Ifhi_rel = Ifhi - span_lo

    Ncol = B_db.shape[1]
    detect_db = np.empty((Ndet, Ncol), dtype=np.float64)
    for j in range(Ndet):
        lo, hi = Iflo_rel[j], Ifhi_rel[j] + 1
        detect_db[j, :] = 10.0 * np.log10(dF * np.sum(10.0 ** (B_db[lo:hi, :] / 10.0), axis=0))

    dT = (1 - params.ovlap) * params.Nfft / params.Fs
    Imin_time = round(params.MinTime / dT)
    Itol_time = round(params.TolTime / dT)
    Imax_time = round(params.MaxTime / dT)

    _, Iburn = min(enumerate(times), key=lambda kv: abs(params.burn_in_time * 60 - kv[1]))

    dn = (1 - params.ovlap) * params.Nfft
    if np.isinf(params.eq_time):
        alpha = 1.0
    else:
        alpha = 0.01 ** (dn / (params.eq_time * params.Fs))

    out_tstart_idx, out_tend_idx, count = _run_state_machine(
        detect_db, Ndet, Ncol, Iburn, params.threshold, alpha, Imin_time, Itol_time, Imax_time
    )
    tstart_idx = out_tstart_idx[:count]
    tend_idx = out_tend_idx[:count]

    return {
        "tstart": times[tstart_idx] if count else np.array([]),
        "tend": times[tend_idx] if count else np.array([]),
    }
