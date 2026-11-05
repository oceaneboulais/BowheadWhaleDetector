"""Draw the full autoencoder (encoder+latent+decoder) as two rows of isometric
feature-map boxes, split right after the last MaxPool / right before the FC
bottleneck -- row 1 is the convolutional encoder trunk, row 2 is the FC
bottleneck, latent feature vector, and the transposed-conv decoder trunk.

Mirrors the box-drawing style of ``plot_architecture_isometric.py`` (used for
the CNN classifier figure), but with larger boxes/fonts so the diagram reads
clearly at paper print size, and exact tensor shapes hard-coded to match the
Autoencoder_v13 checkpoint (32-121x104 input, 32/64/128 channel encoder,
32-D latent, symmetric transposed-conv decoder) described in the paper text.

Run:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.models.plot_autoencoder_architecture_isometric \\
        --out paper/Figures/autoencoder_arch_Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

# Real example input/reconstruction spectrograms (middle, harmonic call) cropped
# from the published Latent32 reconstruction grid -- used in place of synthetic
# noise thumbnails so the diagram shows an actual bowhead call representation.
INPUT_EXAMPLE_IMG = Path("paper/Figures/_ae_diagram_input_example.png")
OUTPUT_EXAMPLE_IMG = Path("paper/Figures/_ae_diagram_output_example.png")

# Isometric "skew" direction shared by every box (dx, dy per unit depth).
ISO_DX, ISO_DY = 0.55, 0.32

C_INPUT = "#f5f0dd"
C_CONV = ["#dbe9f6", "#aecbe8", "#7fa8d6"]
C_POOL = "#f6d9b8"
C_FC = "#ddc2e8"
C_LATENT = "#8b3a3a"
C_DECONV = ["#7fa8d6", "#aecbe8", "#dbe9f6"]
C_OUT = "#cdeccd"

FONT_LABEL = 13
FONT_DIM = 11.5
FONT_SECTION = 17
FONT_NOTE = 12.5


def _face_colors(base: str) -> tuple[str, str, str]:
    rgb = np.array(matplotlib.colors.to_rgb(base))
    front = rgb
    top = np.clip(rgb + (1 - rgb) * 0.35, 0, 1)
    right = np.clip(rgb * 0.72, 0, 1)
    return tuple(front), tuple(top), tuple(right)


def _brace(ax, x, y0, y1, width=0.35, lw=1.8, color="#546e7a"):
    """Draw a curly brace spanning [y0, y1] at x, tip pointing in +x."""
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    y0, y1 = sorted((y0, y1))
    ym = (y0 + y1) / 2
    h = (y1 - y0) / 2
    verts = [
        (x, y0),
        (x + width, y0), (x + width, ym - h * 0.5), (x + 2 * width, ym),
        (x + width, ym + h * 0.5), (x + width, y1), (x, y1),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 6
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="none", edgecolor=color, linewidth=lw))


def _box(ax, x0, y0, w, h, depth, color, label_below=""):
    """Draw one isometric box; return (right_mid_xy, next_cursor_x)."""
    dx, dy = ISO_DX * depth, ISO_DY * depth
    front, top, right = _face_colors(color)

    front_pts = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
    top_pts = [(x0, y0 + h), (x0 + w, y0 + h), (x0 + w + dx, y0 + h + dy), (x0 + dx, y0 + h + dy)]
    right_pts = [(x0 + w, y0), (x0 + w + dx, y0 + dy), (x0 + w + dx, y0 + h + dy), (x0 + w, y0 + h)]

    for pts, c in ((top_pts, top), (right_pts, right), (front_pts, front)):
        ax.add_patch(Polygon(pts, closed=True, facecolor=c, edgecolor="#2b2b2b", linewidth=1.1, zorder=3))

    cx = x0 + w / 2 + dx / 2
    if label_below:
        ax.text(cx, y0 - 0.7, label_below, ha="center", va="top", fontsize=FONT_LABEL, color="#263238")

    right_mid = (x0 + w + dx, y0 + dy + h / 2)
    return right_mid, x0 + w + dx


def _arrow(ax, p0, x1, y1=None):
    y0 = p0[1]
    y1 = y0 if y1 is None else y1
    ax.add_patch(FancyArrowPatch((p0[0], y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                  linewidth=1.6, color="#777777", zorder=2,
                                  connectionstyle="arc3,rad=0" if y0 == y1 else "angle3"))


def _hbar_scale(value: int, lo=6, hi=600, out_lo=1.4, out_hi=13.0) -> float:
    """Square-root compression so a 121-tall box isn't 16x a 1-tall box."""
    v = max(lo, min(hi, value))
    frac = (v ** 0.5 - lo ** 0.5) / (hi ** 0.5 - lo ** 0.5)
    return out_lo + frac * (out_hi - out_lo)


def draw(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(19, 11))
    ax.axis("off")

    row1_y0 = 23.0
    row2_y0 = 1.5

    # Colorblind-safe (Okabe-Ito) zone accents: sky blue for the encoder,
    # orange for the decoder -- distinct even under common color-vision
    # deficiencies.
    C_ENCODER_BG = "#d7ecfa"
    C_ENCODER_TITLE = "#1b4f72"
    C_DECODER_BG = "#fde6c8"
    C_DECODER_TITLE = "#8a5000"

    def _hbar_scale2(value: int) -> float:
        """Compressed height scale for decoder-row boxes (keeps their 'above'
        labels low, clear of the inter-row gap reserved for the connector)."""
        v = max(6, min(600, value))
        frac = (v ** 0.5 - 6 ** 0.5) / (600 ** 0.5 - 6 ** 0.5)
        return 1.4 + frac * 4.2

    # ---- Row 1: input -> convolutional encoder trunk -> flatten -> FC to latent.
    cursor = 1.0
    thumb = plt.imread(INPUT_EXAMPLE_IMG)
    thumb_h, thumb_w = _hbar_scale(121), 2.2
    ax.imshow(thumb, aspect="auto", cmap="viridis", zorder=3,
              extent=(cursor, cursor + thumb_w, row1_y0, row1_y0 + thumb_h))
    ax.text(cursor + thumb_w / 2, row1_y0 - 0.7, "Input\nexample", ha="center", va="top", fontsize=FONT_LABEL, zorder=3)
    cursor += thumb_w + 0.6
    _brace(ax, cursor, row1_y0, row1_y0 + thumb_h)
    cursor += 2.4

    hgt = _hbar_scale(121)
    p, cursor = _box(ax, cursor, row1_y0, 1.8, hgt, 1.0, C_INPUT,
                      label_below="Input\n1x121x104")
    cursor += 4.2

    conv_specs = [(32, 121, 104, C_CONV[0]), (64, 60, 52, C_CONV[1]), (128, 30, 26, C_CONV[2])]
    pool_specs = [(32, 60, 52), (64, 30, 26), (128, 15, 13)]
    for i, ((ch, cph, cpw, color), (_, ph, pw)) in enumerate(zip(conv_specs, pool_specs)):
        _arrow(ax, p, cursor)
        p, cursor = _box(ax, cursor, row1_y0, 2.4 + i * 0.9, _hbar_scale(cph), ch * 0.012, color,
                          label_below=f"Conv+BN+ReLU\n{ch}x{cph}x{cpw}")
        cursor += 4.0
        _arrow(ax, p, cursor)
        p, cursor = _box(ax, cursor, row1_y0, 1.8, _hbar_scale(ph), ch * 0.012, C_POOL,
                          label_below=f"MaxPool\n{ch}x{ph}x{pw}")
        cursor += 4.4

    _arrow(ax, p, cursor)
    p, cursor = _box(ax, cursor, row1_y0, 1.0, 1.6, 1.2, "#eceff1",
                      label_below="Flatten\n24,960")
    cursor += 4.2

    _arrow(ax, p, cursor)
    p, cursor = _box(ax, cursor, row1_y0, 2.2, _hbar_scale(6), 2.6, C_FC,
                      label_below="FC 24,960\n\u2192 64")
    cursor += 1.5

    row1_end = p
    row1_xmax = cursor

    # Generous padding so the rounded background clears every box top
    # (incl. the isometric top-face skew) and every two-line label below.
    enc_bg_x0, enc_bg_y0 = -1.1, row1_y0 - 3.4
    enc_bg_w = row1_xmax + 0.9 - enc_bg_x0
    enc_bg_h = (row1_y0 + hgt + 3.0) - enc_bg_y0
    ax.add_patch(FancyBboxPatch((enc_bg_x0, enc_bg_y0), enc_bg_w, enc_bg_h,
                                 boxstyle='round,pad=0,rounding_size=0.9',
                                 facecolor=C_ENCODER_BG, edgecolor="none", zorder=0))
    ax.text(1.0, row1_y0 + hgt + 1.0, "Encoder (convolutional trunk)",
            ha="left", va="bottom", fontsize=FONT_SECTION, fontweight="bold", color=C_ENCODER_TITLE)

    # ---- Row 2: FC from latent -> reshape -> transposed-conv decoder -> output.
    cursor = 1.0
    p, cursor = _box(ax, cursor, row2_y0, 2.2, _hbar_scale2(6), 2.6, C_FC,
                      label_below="FC 32\n\u2192 64")
    row2_start = p
    row2_start_x = 1.0
    cursor += 4.8

    _arrow(ax, p, cursor)
    p, cursor = _box(ax, cursor, row2_y0, 1.4, 1.8, 2.0, "#eceff1",
                      label_below="FC 64\n\u2192 24,960")
    cursor += 4.2

    _arrow(ax, p, cursor)
    p, cursor = _box(ax, cursor, row2_y0, 1.8, _hbar_scale2(15), 128 * 0.012, "#b7c9d8",
                      label_below="Reshape\n128x15x13")
    cursor += 4.8

    deconv_specs = [(64, 30, 26, C_DECONV[0]), (32, 60, 52, C_DECONV[1])]
    for i, (ch, ph, pw, color) in enumerate(deconv_specs):
        _arrow(ax, p, cursor)
        p, cursor = _box(ax, cursor, row2_y0, 2.4 - i * 0.6, _hbar_scale2(ph), ch * 0.012, color,
                          label_below=f"ConvT+BN+ReLU\n{ch}x{ph}x{pw}")
        cursor += 7.4

    _arrow(ax, p, cursor)
    p, cursor = _box(ax, cursor, row2_y0, 1.8, _hbar_scale2(121), 0.8, C_OUT,
                      label_below="ConvT (final)\n1x121x104")
    cursor += 4.8

    thumb2 = plt.imread(OUTPUT_EXAMPLE_IMG)
    thumb2_h, thumb2_w = _hbar_scale(121), 2.2
    _arrow(ax, p, cursor)
    _brace(ax, cursor + 0.3, row2_y0, row2_y0 + thumb2_h)
    cursor += 3.3
    ax.imshow(thumb2, aspect="auto", cmap="viridis", zorder=3,
              extent=(cursor, cursor + thumb2_w, row2_y0, row2_y0 + thumb2_h))
    ax.text(cursor + thumb2_w / 2, row2_y0 - 0.7, "Output\nexample", ha="center", va="top", fontsize=FONT_LABEL, zorder=3)
    row2_xmax = cursor + thumb2_w

    dec_bg_x0, dec_bg_y0 = -1.1, row2_y0 - 5.6
    dec_bg_w = row2_xmax + 0.9 - dec_bg_x0
    dec_bg_h = (row2_y0 + thumb2_h + 1.4) - dec_bg_y0
    ax.add_patch(FancyBboxPatch((dec_bg_x0, dec_bg_y0), dec_bg_w, dec_bg_h,
                                 boxstyle='round,pad=0,rounding_size=0.9',
                                 facecolor=C_DECODER_BG, edgecolor="none", zorder=0))
    ax.text(1.0, row2_y0 - 4.8, "Decoder (symmetric transposed-convolution trunk)",
            ha="left", va="bottom", fontsize=FONT_SECTION, fontweight="bold", color=C_DECODER_TITLE)

    # Feature vector sits in the transitional zone between the encoder and
    # decoder rows, directly on the connector's horizontal swerve segment.
    gap_y = row1_y0 - 7.0
    swerve_x = row1_xmax + 1.2
    lat_x = (swerve_x + row2_start_x) / 2
    ax.scatter([lat_x], [gap_y], s=1100, color=C_LATENT, zorder=4, edgecolor="#2b2b2b", linewidth=1.6)
    ax.text(lat_x, gap_y - 1.0, "Feature vector\n$z \\in \\mathbb{R}^{32}$", ha="center", va="top", fontsize=FONT_LABEL)

    # Connector from the end of row 1 (FC -> latent) down through the
    # transitional zone (carrying the feature vector) into the start of row 2
    # (FC from latent) -- routed around a safe lane so it never crosses text/boxes.
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    verts = [
        row1_end,
        (swerve_x, row1_end[1]),
        (swerve_x, gap_y),
        (row2_start_x, gap_y),
        (row2_start_x, row2_start[1] + 2.0),
    ]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
    connector = PathPatch(MplPath(verts, codes), facecolor="none", edgecolor="#777777",
                           linewidth=1.8, zorder=2)
    ax.add_patch(connector)
    ax.add_patch(FancyArrowPatch(verts[-1], (row2_start_x, row2_start[1] + 0.3),
                                  arrowstyle="-|>", mutation_scale=18, linewidth=1.8,
                                  color="#777777", zorder=2))

    n_params = 3_358_593
    fig.suptitle(f"Convolutional autoencoder architecture ({n_params:,} parameters)",
                 fontsize=FONT_SECTION + 2, y=1.015)
    fig.text(0.5, 0.93, "Block dimensions shown as $C \\times H \\times W$ (channels x height x width)",
             ha="center", fontsize=FONT_NOTE * 1.2, color="#37474f")

    ax.set_xlim(-1.4, max(row1_xmax, row2_xmax) + 1.5)
    ax.set_ylim(row2_y0 - 6.0, row1_y0 + hgt + 5.6)
    ax.set_aspect("auto")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the autoencoder architecture (two-row isometric style)")
    p.add_argument("--out", type=Path,
                   default=Path("paper/Figures/autoencoder_arch_Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.png"))
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    draw(a.out)
