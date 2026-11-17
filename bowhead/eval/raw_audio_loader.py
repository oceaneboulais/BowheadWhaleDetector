"""Locate and extract real DASAR audio for a dataset sample, from the continuous
day-long WAV archive on ``/Volumes/Bowhead_WAV`` -- as opposed to reconstructing
a pseudo-waveform from the stored spectrogram via Griffin-Lim
(``bowhead.benchmark.backbones.birdnet_from_gram``).

Each event filename (``S{site}{yy}{dasar}0T{YYYYMMDD}T{HHMMSS}_Type{n}.mat``)
encodes the wall-clock start time of the detector's crop for that event, on the
site/DASAR/day identified by the filename. The day-long WAV files are named the
same way (``S{site}{yy}{dasar}0T{YYYYMMDD}T{HHMMSS}.WAV``), with ``{HHMMSS}``
giving that WAV file's own recording start time (usually, but not always,
midnight -- the first day of a deployment can start mid-day).

Empirically (verified by cross-correlating a handful of known-correct manually
extracted 5-second call snippets against the day-long archive, correlation
>0.9999), the crop-start implied by the event filename is only off from the
true audio position by a small amount (around 1 second in the cases checked),
not the many-second clock-drift error one might expect from DASAR clock
drift -- but it is not exactly zero either. Rather than build a full clock-drift
model, each candidate position is locally verified and, if necessary, corrected
by recomputing the stored spectral features (peak frequency, inferred from the
dataset's own STFT calibration) from the real audio in a small window around the
naive predicted position and picking the best match -- the same "verify by
recomputing stored features" principle already used by
``bowhead.benchmark.umap_spectrogram_atlas``, generalized from a few known
candidate images to a continuous search over real audio.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

WAV_ROOT = Path("/Volumes/Bowhead_WAV")
NATIVE_SR = 1000  # DASAR nominal sample rate (Hz)

# Same STFT calibration used throughout this project to build/interpret the
# stored SNR-grams (see bowhead/benchmark/backbones/birdnet_from_gram.py).
NFFT = 256
OVLAP = 0.75
HOP = int(round((1.0 - OVLAP) * NFFT))

# The stored fpeak/fmin/fmax features are expressed in the SNR-gram's own row-
# index convention, which starts 25 Hz (quantized to whole bins) below a plain
# 0 Hz-referenced frequency axis computed directly from raw audio -- the same
# calibration documented in birdnet_from_gram.py and the interactive UMAP page
# ("the stored peak frequency ... reads about 21 Hz lower than its position on
# the axis"). Verified directly: a stored fpeak of 74.21875 Hz lines up with an
# actual spectral peak at 97.65625 Hz once this offset is added back.
DF = 3.90625  # Hz per gram row (bin size of the stored 121x104 grams)
FREQ_BAND_OFFSET_HZ = round(25.0 / DF) * DF  # == 23.4375 Hz

EVENT_RE = re.compile(r"^S(\d)(\d\d)([A-Z])(\d)T(\d{8})T(\d{6})_Type(\d+)\.mat$")
DAYFILE_RE = re.compile(r"^S(\d)(\d\d)([A-Z])(\d)T(\d{8})T(\d{6})\.WAV$")

# Local search margin (seconds) around the naive predicted crop-start time
# when verifying/correcting against the real audio. 5 s is generous relative
# to the ~1 s offsets observed empirically.
SEARCH_MARGIN_S = 5.0


@dataclass(frozen=True)
class ParsedEvent:
    site: str
    yy: str
    dasar: str
    date: str   # YYYYMMDD
    time: str   # HHMMSS

    @property
    def dasar_dir(self) -> Path:
        return (WAV_ROOT / f"Shell20{self.yy}_GSI_data" / f"S{self.site}{self.yy}gsif"
                 / f"S{self.site}{self.yy}{self.dasar}0_WAV")

    @property
    def crop_start_walltime(self) -> dt.datetime:
        return dt.datetime.strptime(self.date + self.time, "%Y%m%d%H%M%S")


def parse_event_filename(name: str) -> ParsedEvent:
    m = EVENT_RE.match(str(name))
    if not m:
        raise ValueError(f"Filename does not match the expected event pattern: {name!r}")
    return ParsedEvent(site=m.group(1), yy=m.group(2), dasar=m.group(3),
                        date=m.group(5), time=m.group(6))


@dataclass(frozen=True)
class DayFile:
    path: Path
    start: dt.datetime
    frames: int
    samplerate: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.samplerate

    @property
    def end(self) -> dt.datetime:
        return self.start + dt.timedelta(seconds=self.duration_s)


@lru_cache(maxsize=None)
def _day_files(dasar_dir: Path) -> tuple[DayFile, ...]:
    """All day-long WAV files for one (site, year, DASAR), sorted by start time."""
    out = []
    for p in sorted(dasar_dir.glob("*.WAV")):
        m = DAYFILE_RE.match(p.name)
        if not m:
            continue
        start = dt.datetime.strptime(m.group(5) + m.group(6), "%Y%m%d%H%M%S")
        info = sf.info(str(p))
        out.append(DayFile(path=p, start=start, frames=info.frames, samplerate=info.samplerate))
    return tuple(sorted(out, key=lambda d: d.start))


def _file_for_time(dasar_dir: Path, target: dt.datetime) -> DayFile | None:
    """The day-file whose [start, end) interval contains ``target``, if any."""
    for d in _day_files(dasar_dir):
        if d.start <= target < d.end:
            return d
    return None


@dataclass
class ExtractedClip:
    audio: np.ndarray       # float64, at NATIVE_SR
    native_sr: int
    verified: bool          # True if the recomputed peak matched stored fpeak within tolerance
    offset_correction_s: float  # applied shift (true - naive), seconds


def _read_window(dasar_dir: Path, center: dt.datetime, half_width_s: float) -> tuple[np.ndarray, dt.datetime] | None:
    """Read audio spanning [center - half_width_s, center + half_width_s), crossing
    a day-file boundary if necessary by reading from (and concatenating) the
    neighboring file. Returns (audio, window_start_walltime) or None if the
    requested span isn't covered by any file on disk."""
    want_start = center - dt.timedelta(seconds=half_width_s)
    want_end = center + dt.timedelta(seconds=half_width_s)

    chunks: list[np.ndarray] = []
    cur = want_start
    cursor_start: dt.datetime | None = None
    while cur < want_end:
        f = _file_for_time(dasar_dir, cur)
        if f is None:
            # Try the very next file (handles reading right up to a file's end
            # or slightly before a file's declared start due to rounding).
            candidates = [d for d in _day_files(dasar_dir) if d.start > cur]
            if not candidates:
                return None
            f = min(candidates, key=lambda d: d.start)
            if f.start >= want_end:
                return None
            cur = f.start
        if cursor_start is None:
            cursor_start = cur
        read_start_s = (cur - f.start).total_seconds()
        read_end_s = min((want_end - f.start).total_seconds(), f.duration_s)
        n_frames = int(round((read_end_s - read_start_s) * f.samplerate))
        if n_frames <= 0:
            cur = f.end
            continue
        with sf.SoundFile(str(f.path)) as sfh:
            sfh.seek(int(round(read_start_s * f.samplerate)))
            chunk = sfh.read(n_frames, dtype="float64")
        chunks.append(chunk)
        cur = f.end
    if not chunks or cursor_start is None:
        return None
    return np.concatenate(chunks), cursor_start


def extract_verified_clip(
    event: ParsedEvent,
    tpeak: float,
    fpeak: float,
    fmin: float,
    fmax: float,
    window_s: float,
    freq_tol_hz: float = 15.0,
    align_to_crop: float | None = None,
) -> ExtractedClip | None:
    """Extract ``window_s`` seconds of real audio centered on the event's true
    (verified) peak time -- or, with ``align_to_crop`` (the stored crop's length in
    seconds), centered on the middle of that crop (filename time plus the same
    verified/naive correction), so a window longer than the crop leaves equal
    margin either side and the audio lines up with the spectrogram's time axis.

    The naive predicted peak time is ``event.crop_start_walltime + tpeak``. A
    small local search (+/- SEARCH_MARGIN_S) recomputes the dominant frequency
    in the stored [fmin, fmax] band around that time and re-centers on the best
    match; if no candidate matches ``fpeak`` within ``freq_tol_hz``, the naive
    position is used as-is and the clip is marked unverified (rather than
    dropping the sample).
    """
    naive_peak = event.crop_start_walltime + dt.timedelta(seconds=float(tpeak))
    # Stored features use the gram's row-index frequency convention; convert to
    # a plain Hz axis to compare against a directly-computed spectrogram.
    fpeak_hz = fpeak + FREQ_BAND_OFFSET_HZ
    fmin_hz = fmin + FREQ_BAND_OFFSET_HZ
    fmax_hz = fmax + FREQ_BAND_OFFSET_HZ

    search_half = SEARCH_MARGIN_S + window_s / 2.0
    got = _read_window(event.dasar_dir, naive_peak, search_half)
    if got is None:
        return None
    audio, window_start = got

    f_, t_, Sxx = signal.spectrogram(audio, fs=NATIVE_SR, nperseg=NFFT, noverlap=NFFT - HOP,
                                      mode="magnitude")
    fmask = (f_ >= max(0.0, fmin_hz - freq_tol_hz)) & (f_ <= fmax_hz + freq_tol_hz)
    if not fmask.any():
        fmask = (f_ >= max(0.0, fpeak_hz - freq_tol_hz)) & (f_ <= fpeak_hz + freq_tol_hz)
    sub = Sxx[fmask]
    if sub.size == 0:
        verified, true_peak = False, naive_peak
    else:
        colmax = sub.max(axis=0)
        best = int(np.argmax(colmax))
        best_freq = f_[fmask][int(np.argmax(sub[:, best]))]
        best_t = (window_start - naive_peak).total_seconds() + t_[best]
        candidate_peak = naive_peak + dt.timedelta(seconds=best_t)
        verified = abs(best_freq - fpeak_hz) <= freq_tol_hz
        true_peak = candidate_peak if verified else naive_peak

    center = true_peak
    if align_to_crop:
        center = event.crop_start_walltime + (true_peak - naive_peak) + dt.timedelta(seconds=align_to_crop / 2.0)
    clip = _read_window(event.dasar_dir, center, window_s / 2.0)
    if clip is None:
        return None
    clip_audio, clip_start = clip
    n_want = int(round(window_s * NATIVE_SR))
    if len(clip_audio) < n_want:
        pad = n_want - len(clip_audio)
        clip_audio = np.pad(clip_audio, (pad // 2, pad - pad // 2))
    elif len(clip_audio) > n_want:
        start = (len(clip_audio) - n_want) // 2
        clip_audio = clip_audio[start:start + n_want]

    return ExtractedClip(
        audio=clip_audio.astype(np.float64),
        native_sr=NATIVE_SR,
        verified=verified,
        offset_correction_s=(true_peak - naive_peak).total_seconds(),
    )
