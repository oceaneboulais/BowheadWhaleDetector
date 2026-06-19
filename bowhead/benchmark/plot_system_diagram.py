"""Draw the full benchmark system diagram for the frozen-embedding benchmark.

Run:
    PYTHONPATH=. python -m bowhead.benchmark.plot_system_diagram \
        --out docs/benchmark_system_diagram.png [--dpi 180]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── palette ───────────────────────────────────────────────────────────────────
C_BG        = "#f8f9fa"
C_DATA      = "#cfd8dc"
C_IMG_BB    = "#a8d5e2"
C_WAV_BB    = "#7cb6cf"
C_FUSION    = "#b5c99a"
C_PROBE_DET = "#f4c89a"
C_PROBE_CLS = "#f4a56a"
C_PROBE_SEN = "#f0e07a"
C_EVAL      = "#c9b0d9"
C_UNSUP     = "#e07a5f"
C_BORDER    = "#37474f"
C_ARROW     = "#546e7a"

# ── figure dimensions used for font scaling ───────────────────────────────────
FIG_W, FIG_H = 22, 10          # inches
AX_XLim, AX_YLim = 110, 56    # axis units


def _pt_to_y(pt: float) -> float:
    """Convert font-size in points to approximate axis-y units for line spacing."""
    return pt * (1 / 72) * (AX_YLim / FIG_H) * 1.55


def _box(ax, x, y, w, h, color, lines, fontsize=8, bold_first=False,
         border=C_BORDER, lw=1.1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.25",
        linewidth=lw, edgecolor=border, facecolor=color, zorder=3))
    if isinstance(lines, str):
        lines = [lines]
    step = _pt_to_y(fontsize)
    n = len(lines)
    for i, line in enumerate(lines):
        yt = y + h / 2 + (n / 2 - i - 0.5) * step
        ax.text(x + w / 2, yt, line,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if (i == 0 and bold_first) else "normal",
                zorder=4)


def _arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                connectionstyle="arc3,rad=0.0"),
                zorder=2)


def _slabel(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=6.8,
            color="#666", style="italic", zorder=5)


def draw(out: Path, dpi: int = 180) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_facecolor(C_BG);  fig.patch.set_facecolor(C_BG)
    ax.set_xlim(0, AX_XLim); ax.set_ylim(0, AX_YLim)
    ax.axis("off")
    fig.suptitle("Bowhead Whale Frozen-Embedding Benchmark — System Diagram",
                 fontsize=13, fontweight="bold", y=0.97)

    # ── column layout ─────────────────────────────────────────────────────────
    XD   =  5.5          # data
    XI   = 18.0          # image backbones
    XW   = 31.5          # waveform backbones
    XF   = 45.0          # fusion
    XP   = 59.0          # probes
    XE   = 74.5          # eval axes
    XU   = 91.0          # unsupervised
    BW   = 10.5          # standard backbone box width
    FW   = 10.0          # fusion box width
    PW   = 11.5          # probe box width
    EW   = 12.5          # eval box width
    UW   = 11.5          # unsupervised box width

    # ── 1. Input data ──────────────────────────────────────────────────────────
    _box(ax, XD-5.5, 27.0, 11, 7.5, C_DATA,
         ["DASAR Dataset", "199,825 spectrograms",
          "121×104 px SNR-grams", "Sites 3&5 | A D G", "Years 2008–2014"],
         fontsize=7.5, bold_first=True)
    _box(ax, XD-5.5, 36.5, 11, 3.8, C_DATA,
         ["Raw Audio (cluster)", "1 kHz DASAR"],
         fontsize=7.5, bold_first=True, border="#90a4ae", lw=0.8)
    _slabel(ax, XD, 25.2, "INPUT")
    D_SPEC  = 30.75        # y-centre of spectrogram data box
    D_AUDIO = 38.4         # y-centre of audio data box

    # ── 2. Image backbones ─────────────────────────────────────────────────────
    img = [
        (["AST-style ViT", "vit_base_patch16_224", "ImageNet-21k  768-D"], 43.0),
        (["ResNet-18", "ImageNet-1k  512-D"],                               36.5),
        (["EfficientNet-B0", "ImageNet-1k  1280-D"],                        30.0),
    ]
    for lines, yc in img:
        h = len(lines) * _pt_to_y(7.5) + 1.5
        _box(ax, XI-BW/2, yc-h/2, BW, h, C_IMG_BB, lines, fontsize=7.5)
        _arrow(ax, XD+5.5, D_SPEC, XI-BW/2, yc)
    _slabel(ax, XI, 47.0, "IMAGE BACKBONES\n(local, spectrograms.npz)")

    # ── 3. Waveform backbones ──────────────────────────────────────────────────
    wav = [
        (["BirdNET 2.3", "TF-Hub  48 kHz  1024-D"],                        43.0),
        (["Perch 2.0", "TF-Hub  32 kHz  1280-D"],                          37.0),
        (["GMWM", "Google Multispecies Whale", "24 kHz  1280-D"],          30.5),
    ]
    for lines, yc in wav:
        h = len(lines) * _pt_to_y(7.5) + 1.5
        _box(ax, XW-BW/2, yc-h/2, BW, h, C_WAV_BB, lines,
             fontsize=7.5, border="#5585a0", lw=0.8)
        _arrow(ax, XD+5.5, D_AUDIO, XW-BW/2, yc, color="#90a4ae")
    ax.text(XW, 27.0,
            "native arm  ·  10× speed-up arm\n(25–500 Hz → 250–5000 Hz)",
            ha="center", va="center", fontsize=6.5, style="italic", color="#666")
    _slabel(ax, XW, 47.0, "WAVEFORM BACKBONES\n(cluster, raw audio)")

    # ── 4. Fusion ──────────────────────────────────────────────────────────────
    fus = [
        (["Single backbone", "(baseline)"],             43.5, C_FUSION),
        (["Concat fusion", "[BB₁ | BB₂ | …]"],          38.5, C_FUSION),
        (["Mean-pool fusion", "(L2-norm → avg)"],        33.5, C_FUSION),
        (["Multi-sensor fusion", "DASAR A⊕D⊕G / event"], 28.5, C_FUSION),
    ]
    for lines, yc, col in fus:
        h = len(lines) * _pt_to_y(7.5) + 1.4
        _box(ax, XF-FW/2, yc-h/2, FW, h, col, lines, fontsize=7.5)
    FUS_Y = 37.0    # representative y for fusion arrows
    for _, yc in img:
        _arrow(ax, XI+BW/2, yc, XF-FW/2, FUS_Y)
    for _, yc in wav:
        _arrow(ax, XW+BW/2, yc, XF-FW/2, FUS_Y, color="#90a4ae")
    _slabel(ax, XF, 47.0, "EMBEDDING FUSION")

    # ── 5. Probes ──────────────────────────────────────────────────────────────
    prb = [
        (["Binary Detection Probe", "Call vs. Non-call",
          "LogReg (balanced)", "Few-shot k∈{4,8,16,32}·5×"],          42.0, C_PROBE_DET),
        (["Call-Type Probe", "Types 1–7  (7-class)",
          "Multinomial LogReg", "Top-1 Acc + OvR ROC-AUC"],            34.5, C_PROBE_CLS),
        (["Open-Set Rejection", "Max-confidence score",
          "vs. non-bowhead signals", "AUROC"],                          27.0, C_PROBE_SEN),
    ]
    for lines, yc, col in prb:
        h = len(lines) * _pt_to_y(7.5) + 1.4
        _box(ax, XP-PW/2, yc-h/2, PW, h, col, lines, fontsize=7.5)
        _arrow(ax, XF+FW/2, FUS_Y, XP-PW/2, yc)
    _slabel(ax, XP, 47.0, "PROBE TASKS")

    # ── 6. Evaluation axes ─────────────────────────────────────────────────────
    evl = [
        (["Standard Few-Shot", "grouped split", "k ∈ {4, 8, 16, 32}"],     43.5),
        (["Temporal Drift", "train {2008,2010}", "→ test {2012,2014}"],     38.0),
        (["Site Shift", "train Site 3 → test Site 5"],                      33.0),
        (["Per-type PR Curves", "one curve per call type  1–7"],            28.0),
    ]
    for lines, eyc in evl:
        h = len(lines) * _pt_to_y(7.5) + 1.4
        _box(ax, XE-EW/2, eyc-h/2, EW, h, C_EVAL, lines, fontsize=7.5)
        for _, pyc, _ in prb:
            _arrow(ax, XP+PW/2, pyc, XE-EW/2, eyc, color="#9e86bb", lw=0.7)
    _slabel(ax, XE, 47.0, "EVALUATION AXES")

    # ── 7. Unsupervised subdivision ────────────────────────────────────────────
    _box(ax, XU-UW/2, 29.0, UW, 13.5, C_UNSUP,
         ["Unsupervised", "Call Subdivision", "",
          "Encoder-decoder", "representations",
          "(Types 1–3: complex)", "",
          "K-Means / HDBSCAN", "→ stable subtypes"],
         fontsize=7.5, bold_first=True)
    _arrow(ax, XF+FW/2, 33.5, XU-UW/2, 35.75, color="#c0564a", lw=1.2)
    _slabel(ax, XU, 47.0, "UNSUPERVISED\nSUBDIVISION")

    # ── section dividers ──────────────────────────────────────────────────────
    for xd in [XI-BW/2-1.0, XW-BW/2-1.0, XF-FW/2-1.0,
               XP-PW/2-1.0, XE-EW/2-1.0, XU-UW/2-1.0]:
        ax.axvline(xd, ymin=0.03, ymax=0.88, color="#ccc",
                   lw=0.6, linestyle="--", zorder=1)

    # ── legend ────────────────────────────────────────────────────────────────
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_DATA,      ec=C_BORDER, label="Input data"),
        mpatches.Patch(facecolor=C_IMG_BB,    ec=C_BORDER, label="Image backbone (local)"),
        mpatches.Patch(facecolor=C_WAV_BB,    ec="#5585a0", label="Waveform backbone (cluster)"),
        mpatches.Patch(facecolor=C_FUSION,    ec=C_BORDER, label="Embedding fusion"),
        mpatches.Patch(facecolor=C_PROBE_DET, ec=C_BORDER, label="Detection probe"),
        mpatches.Patch(facecolor=C_PROBE_CLS, ec=C_BORDER, label="Call-type probe"),
        mpatches.Patch(facecolor=C_PROBE_SEN, ec=C_BORDER, label="Open-set rejection"),
        mpatches.Patch(facecolor=C_EVAL,      ec=C_BORDER, label="Evaluation axes"),
        mpatches.Patch(facecolor=C_UNSUP,     ec=C_BORDER, label="Unsupervised subdivision"),
    ], loc="lower center", ncol=5, fontsize=7.5,
       framealpha=0.9, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.01))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"Written {out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[2]
                   / "docs" / "benchmark_system_diagram.png")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    draw(args.out, dpi=args.dpi)
