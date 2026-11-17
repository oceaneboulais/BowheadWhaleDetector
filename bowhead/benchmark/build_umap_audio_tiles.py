"""Cut one audio clip per spectrogram on the interactive UMAP page's "click a point" tab and pack
the clips into binary tiles that line up with the spectrogram tiles (tile = slot // 64).

Each clip is ``CLIP_S`` seconds of REAL DASAR audio (1 kHz, from the continuous day-long WAV archive).
The event's filename timestamp is only a rough guide to where its spectrogram sits in the recording (typically
~1 s off), so a wider window is read and slid until the energy-versus-time profile of its own spectrogram best
matches the stored spectrogram's profile; the clip is cut at that position, so the sound matches the time axis
drawn on the page.  Clips are peak-normalised and stored as little-endian int16.

Output (``--out``), meant to be published as a small static site:
    tile_000.bin ...   per_tile * n_samples int16 each (unused trailing slots zero-filled)
    audio.json         {sr, n_samples, per_tile, n_slots, flags_b64}; flags: 0 none, 1 approximate
                       position, 2 position independently verified against the stored peak frequency

Run:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.benchmark.build_umap_audio_tiles --out /path/to/audio_repo
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import spectrogram

from bowhead.benchmark import umap_spectrogram_atlas as atlas_mod
from bowhead.eval.raw_audio_loader import NATIVE_SR, extract_verified_clip, parse_event_filename

CLIP_S = 3.0
PEAK = 0.9
WIN_S = 6.0          # wider window searched; the crop's centre sits at its middle
HOP = 26             # samples; 26 ms = the stored spectrogram's column step (dT)
NFFT = 256
BIN0 = 6             # audio bin 6 (23.4 Hz) <-> stored spectrogram row 0 (see raw_audio_loader.FREQ_BAND_OFFSET_HZ)
CAL_S = 0.11         # measured bias of this profile match against the frequency-verified subset (see README)
MATCH_R = 0.6


def match_offset(audio: np.ndarray, gram_db: np.ndarray) -> tuple[int, float]:
    """Return (start sample of the best-matching crop within ``audio``, Pearson r of the match)."""
    f, t, S = spectrogram(audio, fs=NATIVE_SR, nperseg=NFFT, noverlap=NFFT - HOP, mode="magnitude")
    P = 20 * np.log10(S + 1e-9)
    P = np.clip(P - np.median(P, axis=1, keepdims=True), 0, None)[BIN0:BIN0 + gram_db.shape[0]]
    prof = P.sum(axis=0)
    g = np.clip(gram_db, 0, None).sum(axis=0)
    nc = len(g)
    win = np.lib.stride_tricks.sliding_window_view(prof, nc)
    w = win - win.mean(axis=1, keepdims=True)
    gc = g - g.mean()
    den = np.sqrt((w ** 2).sum(axis=1) * (gc ** 2).sum()) + 1e-12
    r = (w @ gc) / den
    lag = int(np.argmax(r))
    return lag * HOP + int(round(CAL_S * NATIVE_SR)), float(r[lag])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N slots (testing)")
    args = ap.parse_args()

    row_slots = np.load(atlas_mod.CACHE_DIR / "row_slots.npy")
    rows = np.flatnonzero(row_slots >= 0)
    slots, first = np.unique(row_slots[rows], return_index=True)
    row_of_slot = rows[first]
    n_slots = int(slots.max()) + 1
    assert len(slots) == n_slots, "slot numbering must be contiguous"

    m = sio.loadmat(str(atlas_mod.TRAIN_MAT), simplify_cells=True)
    fn = np.asarray(m["original_filenames"]).tolist()
    ft = m["features"]
    tpk, fpk, fmin, fmax = (np.asarray(ft[k], dtype=float) for k in ("tpeak", "fpeak", "fmin", "fmax"))

    n = n_slots if args.limit is None else min(args.limit, n_slots)
    ns = int(round(CLIP_S * NATIVE_SR))
    per_tile = atlas_mod.PER_TILE
    src_of_slot = np.load(atlas_mod.CACHE_DIR / "slot_src.npy")
    grams = np.load(atlas_mod.SRC_NPZ, allow_pickle=True)["images"]
    audio = np.zeros((n_slots, ns), dtype=np.int16)
    rs = np.full(n_slots, np.nan)
    freq_verified = np.zeros(n_slots, dtype=bool)
    starts = np.full(n_slots, np.nan)
    flags = np.zeros(n_slots, dtype=np.uint8)

    t0 = time.time()
    for s in range(n):
        i = int(row_of_slot[s])
        try:
            clip = extract_verified_clip(parse_event_filename(fn[i]), tpeak=tpk[i], fpeak=fpk[i], fmin=fmin[i],
                                         fmax=fmax[i], window_s=WIN_S, align_to_crop=CLIP_S)
        except Exception:
            clip = None
        if clip is not None and len(clip.audio) == int(WIN_S * NATIVE_SR):
            start, r = match_offset(clip.audio, grams[src_of_slot[s]].astype(float) / 5.0)
            start = int(np.clip(start, 0, len(clip.audio) - ns))
            x = clip.audio[start:start + ns]
            x = x - x.mean()
            pk = float(np.abs(x).max())
            if pk > 0:
                audio[s] = np.round(x / pk * PEAK * 32767).astype(np.int16)
                flags[s] = 2 if r >= MATCH_R else 1
                rs[s] = r
                starts[s] = start
                freq_verified[s] = clip.verified
        if (s + 1) % 1000 == 0 or s + 1 == n:
            rate = (s + 1) / (time.time() - t0)
            print(f"  {s + 1:,}/{n:,}  with audio={int((flags > 0).sum()):,}  verified={int((flags == 2).sum()):,}  "
                  f"({rate:.0f}/s)", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    n_tiles = -(-n_slots // per_tile)
    for t in range(n_tiles):
        block = np.zeros((per_tile, ns), dtype="<i2")
        chunk = audio[t * per_tile:(t + 1) * per_tile]
        block[:len(chunk)] = chunk
        (args.out / f"tile_{t:03d}.bin").write_bytes(block.tobytes())
    meta = {"sr": NATIVE_SR, "n_samples": ns, "per_tile": per_tile, "n_slots": n_slots, "n_tiles": n_tiles,
            "flags_b64": base64.b64encode(flags.tobytes()).decode("ascii")}
    (args.out / "audio.json").write_text(json.dumps(meta, separators=(",", ":")))
    total = sum(p.stat().st_size for p in args.out.glob("tile_*.bin"))
    ok = ~np.isnan(rs)
    print(f"match r: median {np.nanmedian(rs):.2f}, >= {MATCH_R}: {np.mean(rs[ok] >= MATCH_R):.1%} of {ok.sum():,} clips")
    fv = ok & freq_verified
    if fv.any():   # for frequency-verified events the filename-derived position is trustworthy, so the match should land on it
        print(f"calibration check on {fv.sum():,} frequency-verified clips: median match offset "
              f"{np.nanmedian(starts[fv] - 1500) / NATIVE_SR:+.3f} s (expect ~0), IQR "
              f"{np.nanpercentile(starts[fv] - 1500, 25) / NATIVE_SR:+.2f}..{np.nanpercentile(starts[fv] - 1500, 75) / NATIVE_SR:+.2f} s")
    print(f"wrote {n_tiles} tiles ({total / 1e6:.1f} MB) to {args.out}; "
          f"with audio {int((flags > 0).sum()):,}/{n_slots:,}, verified {int((flags == 2).sum()):,}")


if __name__ == "__main__":
    main()
