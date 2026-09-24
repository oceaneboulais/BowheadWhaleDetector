"""Python port of matlab/matlab/master_compute_manual_miss_rate.m.

Computes the aggregate statistic referenced by the TODO in
paper/BowheadAI_v0.tex: the percentage of manual whale-call annotations that
did NOT overlap any automated (CFAR) transient detection, across every
date/site/DASAR combination in Table I of that paper.

This reruns only the detection + overlap-matching portion of the original
MATLAB dataset-building pipeline (MultipleBandEnergyDetector.m followed by
evaluate_overlap_between_manual_automated.m) against the raw GSI acoustic
data -- it does not regenerate any spectrogram images -- and aggregates the
missed-annotation count over every date instead of reporting it per-chunk.

Usage:
    python -m bowhead.manual_miss_rate.compute_manual_miss_rate \
        --gsi-dir /Volumes/Shared-1/Data \
        --manual-dir /Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/Shell_Manual_Results \
        --out results/manual_miss_rate

Requires: numpy, scipy, and (strongly recommended for reasonable runtime) numba.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import os
from dataclasses import dataclass, field

import numpy as np

from bowhead.manual_miss_rate.energy_detector import EnergyDetectorParams, HAVE_NUMBA, multiple_band_energy_detector
from bowhead.manual_miss_rate.gsi_io import read_gsi_omni_only
from bowhead.manual_miss_rate.manual_archive import load_manual_archive
from bowhead.manual_miss_rate.overlap import evaluate_overlap_between_manual_automated

logger = logging.getLogger("manual_miss_rate")

# Table I of paper/BowheadAI_v0.tex: year/site/date combinations used for the
# 100,000-sample training dataset. Dates are (month, day) in the recording year.
TABLE_I = [
    {"year": "08", "site": "3", "dates": ["08/28", "09/06", "09/13", "09/21", "09/29"]},
    {"year": "08", "site": "5", "dates": ["08/21", "08/28", "09/06", "09/13", "09/21", "09/29"]},
    {"year": "10", "site": "3", "dates": ["08/15", "08/21", "08/29", "09/05", "09/13", "09/27"]},
    {"year": "10", "site": "5", "dates": ["08/15", "08/21", "08/29", "09/05", "09/13"]},
    {"year": "12", "site": "3", "dates": ["08/25", "09/01", "09/07", "09/13", "09/18", "09/23", "09/29", "10/05"]},
    {"year": "12", "site": "5", "dates": ["08/25", "09/01", "09/13", "09/18", "09/23", "09/29"]},
    {"year": "14", "site": "3", "dates": ["08/18", "08/28", "09/01", "09/17", "09/27"]},
    {"year": "14", "site": "5", "dates": ["08/18", "08/28", "09/01", "09/17", "09/27"]},
]

DASAR_LETTERS = "ADG"
CHUNK_SAMPLE_SEC = 6 * 60 * 60 - 1  # matches chunk_sample in the MATLAB script
SECONDS_PER_DAY = 86400
LOCAL_TZ_SHIFT_SEC = 8 * 3600  # datenum(...,-8,0,ctime): UTC ctime -> local (AKDT-ish)


@dataclass
class DateResult:
    year: str
    site: str
    dasar: str
    date: str
    n_matched: int
    n_missed: int


@dataclass
class RunResults:
    per_date: list = field(default_factory=list)

    @property
    def total_matched(self) -> int:
        return sum(r.n_matched for r in self.per_date)

    @property
    def total_missed(self) -> int:
        return sum(r.n_missed for r in self.per_date)

    @property
    def total(self) -> int:
        return self.total_matched + self.total_missed

    @property
    def pct_missed(self) -> float:
        return 100.0 * self.total_missed / self.total if self.total else float("nan")


def _day_epoch(year_yyyy: int, month: int, day: int) -> float:
    """Midnight UTC-calendar epoch seconds for a given calendar date."""
    import calendar
    return float(calendar.timegm((year_yyyy, month, day, 0, 0, 0, 0, 0, 0)))


def _find_dasar_raw_dir(gsi_dir: str, year_yy: str, site: str, letter: str) -> str | None:
    for suffix in ("0", "1"):  # mirrors the dir_want(end)='1' fallback in the MATLAB script
        candidate = os.path.join(
            gsi_dir, f"Shell20{year_yy}_GSI_Data", f"S{site}{year_yy}gsif", f"S{site}{year_yy}{letter}{suffix}"
        )
        if os.path.isdir(candidate):
            return candidate
    return None


def _find_raw_file(dasar_dir: str, target_yyyymmdd: str) -> str | None:
    needle = f"{target_yyyymmdd}T000000"
    for path in sorted(glob.glob(os.path.join(dasar_dir, "*gsi"))):
        name = os.path.basename(path)
        if name.startswith("."):
            continue
        if needle in name:
            return path
    return None


def process_date_dasar(
    manual_ctime: np.ndarray,
    manual_duration: np.ndarray,
    gsi_dir: str,
    year_yy: str,
    site: str,
    letter: str,
    date_str: str,
    params: EnergyDetectorParams,
) -> DateResult | None:
    month, day = (int(p) for p in date_str.split("/"))
    year_yyyy = 2000 + int(year_yy)
    target_epoch = _day_epoch(year_yyyy, month, day)

    local_epoch = manual_ctime - LOCAL_TZ_SHIFT_SEC
    with np.errstate(invalid="ignore"):
        day_epoch = np.floor(local_epoch / SECONDS_PER_DAY) * SECONDS_PER_DAY
        this_day_mask = day_epoch == target_epoch
    n_this_day = int(np.count_nonzero(this_day_mask))
    if n_this_day < 3:
        logger.info("%s/%s Site %s DASAR %s: %d manual detections (<3), skipping.",
                    year_yyyy, date_str, site, letter, n_this_day)
        return None

    dasar_dir = _find_dasar_raw_dir(gsi_dir, year_yy, site, letter)
    if dasar_dir is None:
        logger.warning("Site %s DASAR %s 20%s: raw data directory not found under %s", site, letter, year_yy, gsi_dir)
        return None

    target_yyyymmdd = f"{year_yyyy:04d}{month:02d}{day:02d}"
    raw_file = _find_raw_file(dasar_dir, target_yyyymmdd)
    if raw_file is None:
        logger.info("%s/%s Site %s DASAR %s: raw data file not found in %s, skipping.",
                     year_yyyy, date_str, site, letter, dasar_dir)
        return None

    logger.info("Processing 20%s-%s Site %s DASAR %s (%s)...", year_yy, date_str, site, letter, os.path.basename(raw_file))

    x, head = read_gsi_omni_only(raw_file)
    x = x - 2 ** 15

    drift_factor = 1 + head.tdrift / SECONDS_PER_DAY
    manual_tsec_day = (local_epoch[this_day_mask] - target_epoch) * drift_factor
    manual_duration_day = manual_duration[this_day_mask] * drift_factor
    manual_tend_day = manual_tsec_day + manual_duration_day

    fs = params.Fs
    n_chunks = max(1, int(len(x) // (CHUNK_SAMPLE_SEC * fs)))

    day_matched = 0
    day_missed = 0
    for i_chunk in range(n_chunks):
        i_ss = i_chunk * CHUNK_SAMPLE_SEC * int(fs)
        i_ee = min(len(x), i_ss + CHUNK_SAMPLE_SEC * int(fs))
        x_chunk = x[i_ss:i_ee]
        if len(x_chunk) < params.Nfft:
            continue

        detect = multiple_band_energy_detector(x_chunk, params)
        offset = i_chunk * CHUNK_SAMPLE_SEC
        det_tstart = detect["tstart"] + offset
        det_tend = detect["tend"] + offset

        chunk_tmin, chunk_tmax = offset, offset + CHUNK_SAMPLE_SEC
        chunk_mask = (manual_tsec_day >= chunk_tmin) & (manual_tsec_day < chunk_tmax)
        n_chunk_manual = int(np.count_nonzero(chunk_mask))
        if n_chunk_manual == 0 or len(det_tstart) == 0:
            continue

        _, manual_index = evaluate_overlap_between_manual_automated(
            manual_tsec_day[chunk_mask], manual_tend_day[chunk_mask], det_tstart, det_tend, 0.5
        )
        matched_ids = manual_index[~np.isnan(manual_index)]
        n_matched = len(np.unique(matched_ids))
        day_matched += n_matched
        day_missed += n_chunk_manual - n_matched

    logger.info("  -> %d matched, %d missed (of %d manual detections)", day_matched, day_missed, day_matched + day_missed)
    return DateResult(year=year_yy, site=site, dasar=letter, date=date_str, n_matched=day_matched, n_missed=day_missed)


def run(gsi_dir: str, manual_dir: str) -> RunResults:
    if not HAVE_NUMBA:
        logger.warning("numba not installed -- the CFAR detector will run in pure Python and may be very slow. "
                        "Recommended: pip install numba")

    params = EnergyDetectorParams()
    results = RunResults()

    for row in TABLE_I:
        year_yy, site, dates = row["year"], row["site"], row["dates"]
        try:
            archive = load_manual_archive(manual_dir, year_yy, site, DASAR_LETTERS)
        except Exception:
            logger.exception("Failed to load manual archive for Site %s 20%s; skipping this row.", site, year_yy)
            continue

        whale_mask = archive.wctype <= 7
        for letter in DASAR_LETTERS:
            try:
                ctime = np.where(whale_mask, archive.ctime[letter], np.nan)
                duration = np.where(whale_mask, archive.duration[letter], np.nan)
                valid = ~np.isnan(ctime)
                ctime, duration = ctime[valid], duration[valid]
            except Exception:
                logger.exception("Failed to filter manual archive for Site %s DASAR %s 20%s; skipping this DASAR.",
                                  site, letter, year_yy)
                continue

            for date_str in dates:
                try:
                    r = process_date_dasar(ctime, duration, gsi_dir, year_yy, site, letter, date_str, params)
                except Exception:
                    logger.exception("Error processing Site %s DASAR %s 20%s/%s; skipping.", site, letter, year_yy, date_str)
                    continue
                if r is not None:
                    results.per_date.append(r)

    return results


def save_results(results: RunResults, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "manual_miss_rate_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "site", "dasar", "date", "matched", "missed"])
        for r in results.per_date:
            writer.writerow([r.year, r.site, r.dasar, r.date, r.n_matched, r.n_missed])

    summary_path = os.path.join(out_dir, "manual_miss_rate_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "total_manual": results.total,
            "total_matched": results.total_matched,
            "total_missed": results.total_missed,
            "pct_missed": results.pct_missed,
        }, f, indent=2)
    logger.info("Wrote %s and %s", csv_path, summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gsi-dir", default="/Volumes/Shared-1/Data",
                         help="Root of the raw GSI acoustic data (contains Shell20YY_GSI_Data/...)")
    parser.add_argument("--manual-dir", default="/Volumes/Thode_AI_Working_Disk/Bowhead_DL_Project/Shell_Manual_Results",
                         help="Root of the Shell manual-analyst archive")
    parser.add_argument("--out", default="results/manual_miss_rate", help="Output directory for CSV/JSON results")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                         format="%(asctime)s %(levelname)s %(message)s")

    if not os.path.isdir(args.gsi_dir):
        raise SystemExit(f"Raw acoustic data volume not mounted/found: {args.gsi_dir}")
    if not os.path.isdir(args.manual_dir):
        raise SystemExit(f"Manual archive directory not found: {args.manual_dir}")

    results = run(args.gsi_dir, args.manual_dir)
    save_results(results, args.out)

    print("\n=====================================================")
    print(f"TOTAL across all Table I dates/sites/DASARs: {results.total} manual annotations")
    print(f"  {results.total_matched} matched by an automated (CFAR) detection")
    print(f"  {results.total_missed} missed by the automated detector ({results.pct_missed:.2f}%)")
    print("=====================================================")


if __name__ == "__main__":
    main()
