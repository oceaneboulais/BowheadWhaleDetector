"""Reproduce the AE+kNN detector's miss-rate vs false-discovery-rate curves
described in the advisor's email: sweep the k-nearest-neighbor "call vote
fraction" threshold, crossed with (a) the >=0.5s transient-duration filter
and (b) the reviewed/"cleaned" vs pre-review/"original" dataset labels.

Uses the AEKNNScorer built in bowhead/eval/ae_knn_baseline.py, scored against
the real 199,825-spectrogram independent evaluation set (already exported to
32-D AE latents by MATLAB, so no image loading/AE inference is needed here).

The advisor's "roughly 8 false detections to true detection in the original
dataset" figure is the NATURAL, unfiltered class imbalance of the raw
auto-detector output (i.e. what you'd get by accepting every candidate
event as a "call" with no classifier at all) -- confirmed two independent
ways: (1) the eval set's own pre-review prevalence (`type_org`) gives
FP:TP = (1-p)/p = 8.02, and (2) a full scan of the raw ~2.1M-file master
database (Event_sounds.dir vs Manually_selected_bowhead_calls.dir) gives an
auto:manual ratio of 7.86:1. Both are plotted as dashed reference lines.

Usage:
    PYTHONPATH=. python -m bowhead.benchmark.run_ae_knn_curves \\
        --train latent_embeddings_3d_train_MATLAB.mat \\
        --eval  /path/to/latent_embeddings_3d_eval_8to1_MATLAB_Raquel.mat \\
        --k 40 --out-png paper/Figures/ae_knn_miss_vs_fdr.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bowhead.eval.ae_knn_baseline import load_training_embeddings, AEKNNScorer

CONDITIONS = [
    # (label, label_source, min_duration)
    ("Reviewed dataset, all durations",        "cleaned",  None),
    ("Reviewed dataset, duration >= 0.5s",      "cleaned",  0.5),
    ("Original dataset, all durations",         "original", None),
    ("Original dataset, duration >= 0.5s",      "original", 0.5),
]
COLORS = {
    "Reviewed dataset, all durations":     "#4e9af1",
    "Reviewed dataset, duration >= 0.5s":  "#000000",
    "Original dataset, all durations":     "#f07040",
    "Original dataset, duration >= 0.5s":  "#c0392b",
}

# Master-database-wide auto:manual ratio (raw, unfiltered), from a full scan of
# the ~2.1M-file Spectrogram_Image_Database (see docstring). None to skip.
RAW_DB_FP_PER_TP = 1_877_497 / 238_751

# Per-season sample counts used by master_evaluate_autoencoder_performance.m to
# weight the old 2010 detector's per-season FDR curves (OldResults_for_Comparison.mat's
# `ygrid` cell array) into one combined curve. Order matches the MATLAB script's
# hardcoded `yearly_samples` vector, corresponding to field seasons 2008, 2010, 2012, 2014.
OLD_2010_YEARLY_SAMPLES = (38773, 151511, 93427, 148967)


def _load_eval(path: str) -> dict:
    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    feat = d["features"]
    # Re-derive from type/type_org (matches master_evaluate_autoencoder_performance.m's
    # live `type>0 & type<12` formula) rather than trusting the precomputed `iscall` field.
    type_ = np.asarray(feat.type, dtype=float)
    type_org = np.asarray(feat.type_org).astype(int)
    return dict(
        latent=np.asarray(d["latent_embeddings"], dtype=np.float32),
        iscall=((type_ > 0) & (type_ < 12)).astype(int),
        iscall_original=((type_org > 0) & (type_org < 12)).astype(int),
        duration=np.asarray(feat.duration1, dtype=float),
        filenames=np.asarray(d["original_filenames"]),
    )


def _miss_fdr_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Sweep unique score thresholds -> (thresholds, miss_fraction, fdr, fp_per_tp)."""
    order = np.argsort(-y_score, kind="stable")
    y_sorted = y_true[order]
    s_sorted = y_score[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    n_pos = int((y_true == 1).sum())

    # Keep only the last index at each distinct score value (standard threshold sweep).
    distinct = np.r_[np.diff(s_sorted) != 0, True]
    tp = tp_cum[distinct]
    fp = fp_cum[distinct]
    thr = s_sorted[distinct]

    recall = tp / max(n_pos, 1)
    miss = 1.0 - recall
    with np.errstate(divide="ignore", invalid="ignore"):
        fdr = np.where(tp + fp > 0, fp / (tp + fp), 0.0)
        fp_per_tp = np.where(tp > 0, fp / tp, np.inf)
    return thr, miss, fdr, fp_per_tp


def _report_at_threshold(name: str, thr: np.ndarray, miss: np.ndarray,
                          fdr: np.ndarray, fp_per_tp: np.ndarray, target: float) -> None:
    i = int(np.argmin(np.abs(thr - target)))
    print(f"  {name:38s} @ vote-threshold={thr[i]:.3f}  "
          f"miss={miss[i]:6.1%}  FDR={fdr[i]:6.3f}  FP:TP={fp_per_tp[i]:6.2f}")


def _report_at_miss(name: str, thr: np.ndarray, miss: np.ndarray,
                     fdr: np.ndarray, fp_per_tp: np.ndarray, target_miss: float) -> dict:
    """Find the point on the (discrete, k+1-valued) vote-threshold sweep whose
    miss fraction is closest to `target_miss`, so different conditions (label
    set x duration filter) can be compared at a MATCHED recall level rather
    than at a shared vote threshold (which yields different, incomparable
    miss fractions per condition)."""
    i = int(np.argmin(np.abs(miss - target_miss)))
    row = {
        "vote_threshold": float(thr[i]),
        "miss_fraction": float(miss[i]),
        "fdr": float(fdr[i]),
        "fp_per_tp": float(fp_per_tp[i]),
    }
    print(f"  {name:38s} @ vote-threshold={row['vote_threshold']:.3f}  "
          f"miss={row['miss_fraction']:6.1%}  FDR={row['fdr']:6.3f}  FP:TP={row['fp_per_tp']:6.2f}")
    return row


def _load_old_2010_curve(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the "Original 2010 network" dashed reference curve from
    master_evaluate_autoencoder_performance.m: at each `xgrid` (miss-fraction)
    point, combine the four per-season FDR values in `ygrid` into one curve by
    averaging them weighted by `OLD_2010_YEARLY_SAMPLES`, skipping any season
    that has no data (NaN) at that grid point -- exactly matching the MATLAB
    script's `Isgood=~isnan(numerator)` masking.
    """
    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    xgrid = np.asarray(d["xgrid"], dtype=float)
    ygrid = np.asarray(d["ygrid"])  # (4,) object array, one FDR vector per season
    yearly_samples = np.asarray(OLD_2010_YEARLY_SAMPLES, dtype=float)

    old_fdr = np.full(len(xgrid), np.nan)
    for i in range(len(xgrid)):
        y_i = np.array([np.asarray(ygrid[j])[i] for j in range(len(ygrid))])
        numerator = yearly_samples * y_i
        good = ~np.isnan(numerator)
        if good.any():
            old_fdr[i] = numerator[good].sum() / yearly_samples[good].sum()
    return xgrid, old_fdr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True, help="Training latent-embeddings .mat")
    p.add_argument("--eval", required=True, help="Evaluation latent-embeddings .mat")
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--verify-threshold", type=float, default=0.75,
                   help="Neighbor-vote-fraction operating point to print (default: 30/40=0.75)")
    p.add_argument("--out-png", default="paper/Figures/ae_knn_miss_vs_fdr.png")
    p.add_argument("--old-2010-mat", default="data/OldResults_for_Comparison.mat",
                   help="Old 2010 detector's xgrid/ygrid reference curve (Thode et al. "
                        "2012); pass '' to omit the overlay.")
    p.add_argument("--target-miss", type=float, default=0.10,
                   help="Matched miss-fraction operating point (default: 10%%) at which "
                        "all label-set/duration conditions are compared -- unlike a fixed "
                        "vote threshold, this puts every row at the same recall level.")
    p.add_argument("--matched-miss-out-json", default="runs/ae_knn_matched_miss10.json")
    args = p.parse_args()

    print(f"Loading training pool: {args.train}")
    train = load_training_embeddings(args.train)
    print(f"  {len(train):,} rows")

    print(f"Loading evaluation set: {args.eval}")
    ev = _load_eval(args.eval)
    print(f"  {len(ev['latent']):,} rows")

    # Natural (no-classifier) class-imbalance baselines: what you'd get by
    # accepting every candidate event as a "call", i.e. no filtering at all.
    n_call = int(ev["iscall"].sum())
    n_call_orig = int(ev["iscall_original"].sum())
    baseline_reviewed = (len(ev["iscall"]) - n_call) / max(n_call, 1)
    baseline_original = (len(ev["iscall_original"]) - n_call_orig) / max(n_call_orig, 1)
    print("\nNatural (unfiltered) FP:TP baselines -- accept every candidate as a call:")
    print(f"  Reviewed-dataset labels : FP:TP={baseline_reviewed:.2f}")
    print(f"  Original-dataset labels : FP:TP={baseline_original:.2f}  "
          f"(advisor's email: 'roughly 8')")
    print(f"  Full raw master DB (2.1M files, Event_sounds vs Manually_selected): "
          f"FP:TP={RAW_DB_FP_PER_TP:.2f}")

    print(f"\nVerification at neighbor-vote threshold={args.verify_threshold} (k={args.k}):")
    curves = {}
    matched_miss: dict = {
        "target_miss_fraction": args.target_miss,
        "no_filtering_baseline": {
            "reviewed": {"fdr": baseline_reviewed / (1.0 + baseline_reviewed),
                         "fp_per_tp": baseline_reviewed},
            "original": {"fdr": baseline_original / (1.0 + baseline_original),
                         "fp_per_tp": baseline_original},
        },
        "conditions": {},
    }
    for name, label_source, min_dur in CONDITIONS:
        scorer = AEKNNScorer(train, k=args.k, label_source=label_source, min_duration=min_dur)
        y_score = scorer.score_embeddings(ev["latent"])
        y_true = ev["iscall"] if label_source == "cleaned" else ev["iscall_original"]

        mask = np.ones(len(y_true), dtype=bool)
        if min_dur is not None:
            mask &= ev["duration"] >= min_dur

        thr, miss, fdr, fp_per_tp = _miss_fdr_curve(y_true[mask], y_score[mask])
        curves[name] = (thr, miss, fdr, fp_per_tp, mask.sum())
        _report_at_threshold(name, thr, miss, fdr, fp_per_tp, args.verify_threshold)
        matched_miss["conditions"][name] = {"n": int(mask.sum())}

    print(f"\nMatched miss-fraction={args.target_miss:.0%} comparison "
          f"(k={args.k}) -- raw-vs-reviewed effect at equal recall:")
    for name, label_source, min_dur in CONDITIONS:
        thr, miss, fdr, fp_per_tp, _n = curves[name]
        row = _report_at_miss(name, thr, miss, fdr, fp_per_tp, args.target_miss)
        matched_miss["conditions"][name].update(row)

    out_path = Path(args.matched_miss_out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(matched_miss, indent=2))
    print(f"\nWritten {out_path}")

    old_2010_curve = None
    if args.old_2010_mat:
        print(f"\nLoading old 2010 detector reference curve: {args.old_2010_mat}")
        old_2010_curve = _load_old_2010_curve(args.old_2010_mat)

    _plot(curves, args.out_png, args.k, old_2010_curve, args.target_miss)


def _plot(curves: dict, out_png: str, k: int,
          old_2010_curve: tuple[np.ndarray, np.ndarray] | None = None,
          mark_miss: float | None = None) -> None:
    """Two-panel figure matching Fig. 7's layout: a) recall vs. precision,
    b) miss fraction vs. false discovery rate (recall=1-miss, precision=1-fdr,
    so both panels reuse the same computed curves)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    fig, (ax_pr, ax_md) = plt.subplots(1, 2, figsize=(13, 5.5))

    sc = None
    for name, (thr, miss, fdr, _fp_per_tp, n) in curves.items():
        recall = 1.0 - miss
        precision = 1.0 - fdr
        ax_pr.plot(recall, precision, color=COLORS[name], lw=2, label=f"{name}  (n={n:,})")
        ax_pr.scatter(recall, precision, c=thr, cmap="viridis", s=8, zorder=3)

        ax_md.plot(miss, fdr, color=COLORS[name], lw=2)
        sc = ax_md.scatter(miss, fdr, c=thr, cmap="viridis", s=8, zorder=3)

        if mark_miss is not None:
            i = int(np.argmin(np.abs(miss - mark_miss)))
            ax_pr.scatter([recall[i]], [precision[i]], marker="D", s=55,
                          facecolor="none", edgecolor=COLORS[name], linewidth=1.8, zorder=4)
            ax_md.scatter([miss[i]], [fdr[i]], marker="D", s=55,
                          facecolor="none", edgecolor=COLORS[name], linewidth=1.8, zorder=4)

    if old_2010_curve is not None:
        old_xgrid, old_fdr = old_2010_curve
        ax_pr.plot(1.0 - old_xgrid, 1.0 - old_fdr, "--", color="black", lw=2, zorder=2,
                   label="Original 2010 network (Thode et al. 2012)")
        ax_md.plot(old_xgrid, old_fdr, "--", color="black", lw=2, zorder=2)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.set_title("a) Recall vs. precision")
    ax_pr.text(0.05, 0.95, "a)", fontweight="bold", fontsize=12, transform=ax_pr.transAxes)
    ax_pr.legend(fontsize=7, loc="lower left")
    ax_pr.grid(True, alpha=0.3)
    ax_pr.xaxis.set_minor_locator(AutoMinorLocator())
    ax_pr.yaxis.set_minor_locator(AutoMinorLocator())
    ax_pr.tick_params(which="minor", length=3)
    ax_pr.tick_params(which="major", length=6)

    ax_md.set_xlabel("Miss fraction (1 − recall)")
    ax_md.set_ylabel("False discovery rate (1 − precision)")
    ax_md.set_xlim(0, 1)
    ax_md.set_ylim(0, 1)
    ax_md.set_title("b) Miss fraction vs. false discovery rate")
    ax_md.text(0.05, 0.95, "b)", fontweight="bold", fontsize=12, transform=ax_md.transAxes)
    ax_md.grid(True, alpha=0.3)
    ax_md.xaxis.set_minor_locator(AutoMinorLocator())
    ax_md.yaxis.set_minor_locator(AutoMinorLocator())
    ax_md.tick_params(which="minor", length=3)
    ax_md.tick_params(which="major", length=6)

    fig.suptitle(f"AE + kNN (k={k}) detector performance")
    cbar = fig.colorbar(sc, ax=[ax_pr, ax_md], fraction=0.035, pad=0.02)
    cbar.set_label("Fraction of k neighbors labeled a call")

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure -> {out_path}")


if __name__ == "__main__":
    main()
