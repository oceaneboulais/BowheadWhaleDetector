"""
Locate and extract audio clips from 24-hour DASAR .WAV recordings, given a
detection filename stem (e.g. ``S308A0T20080828T000239_Type0``).

Directory layout on the raw acoustic drives (verified against
/Volumes/PortableSSD/BowheadWhaleDL):

    {base_dir}/Shell20{yy}_GSI_[dD]ata/S{site}{yy}gsif/S{site}{yy}{dasar}0_WAV/
        S{site}{yy}{dasar}0T{YYYYMMDD}T{HHMMSS}.WAV

Each .WAV file is a ~24-hour, mono, 16-bit PCM recording at 1000 Hz sampled
starting at the timestamp in its own filename (usually, but not always,
midnight -- the first file of a deployment can start mid-day).
"""

import re
import glob
import os
import wave
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

# Matches the same convention as bowhead/data/build_dataset.py
_DETECTION_RE = re.compile(
    r"S(?P<site>\d)(?P<yy>\d{2})(?P<dasar>[A-Z])\dT"
    r"(?P<date>\d{8})T(?P<hms>\d{6})"
)
_WAV_FNAME_RE = re.compile(
    r"S\d\d\d[A-Z]\dT(?P<date>\d{8})T(?P<hms>\d{6})\.WAV$", re.IGNORECASE
)

DEFAULT_SAMPLE_RATE = 1000  # Hz, native DASAR rate


@dataclass
class DetectionInfo:
    site: str
    yy: str
    dasar: str
    date: str          # YYYYMMDD
    hms: str           # HHMMSS
    dt: datetime


def parse_detection_filename(name: str) -> Optional[DetectionInfo]:
    """Parse a detection filename/stem into site/dasar/date/time components."""
    m = _DETECTION_RE.search(os.path.basename(name))
    if not m:
        return None
    g = m.groupdict()
    dt = datetime.strptime(g["date"] + g["hms"], "%Y%m%d%H%M%S")
    return DetectionInfo(site=g["site"], yy=g["yy"], dasar=g["dasar"],
                         date=g["date"], hms=g["hms"], dt=dt)


class WavLocator:
    """Finds and extracts clips from 24-hour DASAR .WAV files."""

    def __init__(self, base_dirs: list[str], sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.base_dirs = [b for b in base_dirs if b and os.path.isdir(b)]
        self.sample_rate = sample_rate
        self._wav_dir_cache: dict[tuple, Optional[str]] = {}

    def _find_dasar_wav_dir(self, info: DetectionInfo) -> Optional[str]:
        """Locate the *_WAV folder for a given site/year/DASAR."""
        key = (info.site, info.yy, info.dasar)
        if key in self._wav_dir_cache:
            return self._wav_dir_cache[key]

        found = None
        for base in self.base_dirs:
            pattern = os.path.join(
                base, f"Shell20{info.yy}_GSI_*[dD]ata*",
                f"S{info.site}{info.yy}gsif",
                f"S{info.site}{info.yy}{info.dasar}0_WAV",
            )
            matches = glob.glob(pattern)
            if matches:
                found = matches[0]
                break

        self._wav_dir_cache[key] = found
        return found

    def find_wav_file(self, info: DetectionInfo) -> Optional[tuple[str, datetime]]:
        """Return (path, file_start_datetime) for the .WAV covering *info*'s timestamp."""
        wav_dir = self._find_dasar_wav_dir(info)
        if wav_dir is None:
            return None

        candidates = []
        for fp in glob.glob(os.path.join(wav_dir, "*.[wW][aA][vV]")):
            m = _WAV_FNAME_RE.search(os.path.basename(fp))
            if not m:
                continue
            start_dt = datetime.strptime(m.group("date") + m.group("hms"), "%Y%m%d%H%M%S")
            candidates.append((start_dt, fp))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        # Pick the last file that starts at or before the detection time.
        best = None
        for start_dt, fp in candidates:
            if start_dt <= info.dt:
                best = (fp, start_dt)
            else:
                break
        return best

    def extract_clip(
        self,
        detection_name: str,
        pad_before: float = 2.0,
        pad_after: float = 3.0,
    ) -> dict:
        """Extract a short audio clip around a detection's timestamp.

        Returns a dict with keys: samples (int16 ndarray), sample_rate,
        wav_path, offset_sec (into the source file), clip_start_sec (relative
        to the detection time, i.e. -pad_before unless clamped).
        Raises FileNotFoundError / ValueError with a descriptive message on failure.
        """
        info = parse_detection_filename(detection_name)
        if info is None:
            raise ValueError(f"Could not parse detection filename: {detection_name!r}")

        found = self.find_wav_file(info)
        if found is None:
            raise FileNotFoundError(
                f"No 24-hour .WAV file found for site={info.site} dasar={info.dasar} "
                f"date={info.date}. Searched base_dirs={self.base_dirs}"
            )
        wav_path, file_start_dt = found

        detection_offset_sec = (info.dt - file_start_dt).total_seconds()

        with wave.open(wav_path, "rb") as w:
            sr = w.getframerate()
            n_frames = w.getnframes()
            sampwidth = w.getsampwidth()
            n_channels = w.getnchannels()

            duration_sec = n_frames / sr
            clip_start = max(0.0, detection_offset_sec - pad_before)
            clip_end = min(duration_sec, detection_offset_sec + pad_after)
            if clip_end <= clip_start:
                raise ValueError(
                    f"Computed empty clip range [{clip_start}, {clip_end}] for "
                    f"{detection_name} (offset={detection_offset_sec:.2f}s into "
                    f"{os.path.basename(wav_path)}, duration={duration_sec:.1f}s)"
                )

            start_frame = int(clip_start * sr)
            n_frames_to_read = int((clip_end - clip_start) * sr)

            w.setpos(start_frame)
            raw = w.readframes(n_frames_to_read)

        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
        samples = np.frombuffer(raw, dtype=dtype)
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels)

        return {
            "samples": samples,
            "sample_rate": sr,
            "wav_path": wav_path,
            "detection_offset_sec": detection_offset_sec,
            "clip_start_sec": clip_start,
            "clip_end_sec": clip_end,
        }

    def save_clip_wav(self, clip: dict, out_path: str) -> str:
        """Write a clip dict (from extract_clip) to a standalone .wav file."""
        from scipy.io import wavfile
        wavfile.write(out_path, clip["sample_rate"], clip["samples"])
        return out_path
