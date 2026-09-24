"""Port of matlab/matlab/read_tsv_archive.m (only the fields this pipeline needs).

Each line of a `*_manual_archive.txt` file is one call record: 25 tab-separated
global/localization fields, followed by one 12-field block per DASAR that
detected the call (`<DASAR name>\twgt\tbref\tbearing\tctime\tsigdb\tstndb\t
flo\tfhi\tduration\tsdm\tkappa`). We only need `wctype` (global field 11,
1-indexed) plus, per requested DASAR, `ctime` and `duration`.

Prefers loading an existing MATLAB-generated `*_manual_archive.mat` cache
(byte-identical to what MATLAB itself parsed) when present, and only falls
back to parsing the raw `.txt` when no cache exists.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

try:
    import scipy.io as _sio
except ImportError:  # pragma: no cover - scipy is a hard requirement in practice
    _sio = None


@dataclass
class ManualArchive:
    wctype: np.ndarray                 # shape (Ncalls,)
    ctime: dict[str, np.ndarray]        # DASAR letter -> shape (Ncalls,)
    duration: dict[str, np.ndarray]     # DASAR letter -> shape (Ncalls,)


def _to_float(s: str) -> float:
    s = s.strip()
    if s == "" or s.upper() == "NAN":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _mat_cache_path(manual_dir: str, year_yy: str, site: str, dasar_letters: str) -> str:
    return os.path.join(manual_dir, f"20{year_yy}", f"AllSite{site}_20{year_yy}_{dasar_letters}_manual_archive.mat")


def _txt_path(manual_dir: str, year_yy: str, site: str) -> str:
    return os.path.join(manual_dir, f"20{year_yy}", f"AllSite{site}_20{year_yy}_manual_archive.txt")


def _load_from_mat(path: str, dasar_letters: str) -> ManualArchive:
    d = _sio.loadmat(path, simplify_cells=True)
    wctype = np.asarray(d["localized"]["wctype"], dtype=float).ravel()
    ctime_cols = np.asarray(d["ind"]["ctime"], dtype=float)
    duration_cols = np.asarray(d["ind"]["duration"], dtype=float)

    # MATLAB's `ind` struct can have fewer rows than `localized` (its arrays
    # only grow when at least one DASAR column gets assigned for a given
    # call), so read_tsv_archive.m's caller clips to `size(ind.wgt,1)`.
    # Replicate that clip here rather than crash on the shape mismatch.
    n = min(len(wctype), ctime_cols.shape[0])
    wctype = wctype[:n]
    ctime_cols = ctime_cols[:n]
    duration_cols = duration_cols[:n]

    ctime, duration = {}, {}
    for i, letter in enumerate(dasar_letters):
        ctime[letter] = ctime_cols[:, i]
        duration[letter] = duration_cols[:, i]
    return ManualArchive(wctype=wctype, ctime=ctime, duration=duration)


def _parse_from_txt(path: str, dasar_names: dict[str, str]) -> ManualArchive:
    """dasar_names: letter -> full DASAR station name, e.g. {'A': 'S308A0'}."""
    wctype_list: list[float] = []
    ctime_lists: dict[str, list[float]] = {letter: [] for letter in dasar_names}
    duration_lists: dict[str, list[float]] = {letter: [] for letter in dasar_names}

    # For each requested DASAR, also accept the "...1" fallback name used by
    # read_tsv_archive.m when the primary station id isn't found verbatim.
    fallback_names = {letter: name[:-1] + "1" for letter, name in dasar_names.items()}

    with open(path, "r", errors="replace") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11:
                # Malformed/blank line; skip rather than crash the whole run.
                continue
            wctype_list.append(_to_float(fields[10]))

            for letter, name in dasar_names.items():
                idx = None
                target = {name, fallback_names[letter]}
                for j, fval in enumerate(fields):
                    if fval.strip() in target:
                        idx = j
                        break
                if idx is None or idx + 9 >= len(fields):
                    ctime_lists[letter].append(float("nan"))
                    duration_lists[letter].append(float("nan"))
                    continue
                ctime_lists[letter].append(_to_float(fields[idx + 4]))
                duration_lists[letter].append(_to_float(fields[idx + 9]))

    wctype = np.asarray(wctype_list, dtype=float)
    ctime = {letter: np.asarray(v, dtype=float) for letter, v in ctime_lists.items()}
    duration = {letter: np.asarray(v, dtype=float) for letter, v in duration_lists.items()}
    return ManualArchive(wctype=wctype, ctime=ctime, duration=duration)


def load_manual_archive(manual_dir: str, year_yy: str, site: str, dasar_letters: str) -> ManualArchive:
    """Load (Site, Year) manual archive, restricted to the given DASAR letters (e.g. 'ADG')."""
    mat_path = _mat_cache_path(manual_dir, year_yy, site, dasar_letters)
    if _sio is not None and os.path.exists(mat_path):
        return _load_from_mat(mat_path, dasar_letters)

    txt_path = _txt_path(manual_dir, year_yy, site)
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Manual archive not found: {txt_path} (and no cached {mat_path})")

    dasar_names = {letter: f"S{site}{year_yy}{letter}0" for letter in dasar_letters}
    return _parse_from_txt(txt_path, dasar_names)
