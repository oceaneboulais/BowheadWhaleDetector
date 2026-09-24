"""Port of matlab/matlab/readgsif_header.m and readgsi_omni_only.m.

GSI files store a 512-byte header (only the first ~137 bytes are populated;
the rest is padding) followed by interleaved big-endian uint16 samples for
`nc` channels (typically 3: omnidirectional pressure + 2 directional
components). We only need the omnidirectional (first) channel here, exactly
like readgsi_omni_only.m.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

_HEADER_DATA_OFFSET = 512


@dataclass
class GSIHeader:
    ctbc: float   # file start time, seconds since 1970-01-01 (UTC-ish "ctime")
    ctec: float   # file end time, same units
    tdrift: float  # clock drift, seconds/day
    Fs: float
    nc: int       # number of interleaved channels


def read_gsi_header(path: str) -> GSIHeader:
    """Port of readgsif_header.m. Big-endian; only the fields we actually use."""
    with open(path, "rb") as f:
        raw = f.read(_HEADER_DATA_OFFSET)
    if len(raw) < 137:
        raise ValueError(f"GSI file too short to contain a header: {path}")

    # bytes 0:10 dlabel (unused), 10:14 'contents' -> nc = count of bytes with code>34
    contents = raw[10:14]
    nc = sum(1 for b in contents if b > 34)

    # bytes 14:64 are 50 bytes of filler (unused)
    # bytes 64:136 are 9 big-endian float64 values
    doubles = struct.unpack(">9d", raw[64:136])
    ctbc, ctec, tdrift, Fs = doubles[0], doubles[1], doubles[2], doubles[3]
    return GSIHeader(ctbc=ctbc, ctec=ctec, tdrift=tdrift, Fs=Fs, nc=nc)


def read_gsi_omni_only(path: str) -> tuple[np.ndarray, GSIHeader]:
    """Port of readgsi_omni_only.m called with ctstart=0, tlen=Inf (read whole file).

    Reads only the first (omnidirectional) channel of the interleaved
    big-endian uint16 sample stream, matching the original MATLAB call's
    hard-coded 3-channel (2*2-byte skip) stride.
    """
    head = read_gsi_header(path)
    if head.nc != 3:
        # Original MATLAB reader hard-codes a 3-channel stride regardless of
        # head.nc; flag the mismatch loudly rather than silently mis-parsing.
        raise ValueError(f"{path}: expected 3-channel GSI file (nc={head.nc}); "
                          "the omni-only reader's fixed stride would be wrong.")

    with open(path, "rb") as f:
        f.seek(_HEADER_DATA_OFFSET)
        raw = f.read()

    n_u16 = len(raw) // 2
    n_u16 -= n_u16 % 3  # drop any trailing partial 3-channel frame
    samples = np.frombuffer(raw[: n_u16 * 2], dtype=">u2")
    omni = samples[0::3].astype(np.float64)
    return omni, head
