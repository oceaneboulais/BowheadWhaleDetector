"""Draw the custom-CNN architecture as isometric feature-map boxes.

Mirrors the visual style of the existing autoencoder figure
(``Figures/autoencoder_arch_Autoencoder_v13_..._Date20260416-180022.png``):
each layer is drawn as a 3-D-looking box (front/top/right faces) sized
roughly to its C x H x W feature-map shape, connected by arrows, with a
real input spectrogram thumbnail on the left (same crop used in the
autoencoder figure) and a probability readout on the right. The shared
encoder trunk + FC bottleneck is split across two rows (row break right
after the first MaxPool), matching the autoencoder figure's layout, which
leaves room for larger, more readable fonts without cutting off boxes.
Shapes are introspected from ``EncoderClassifier`` (a forward trace), so
the diagram can never drift from the code.

Run:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.models.plot_architecture_isometric \
        --out paper/Figures/cnn_classifier_architecture.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

from bowhead.models.custom_cnn import EncoderClassifier

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

# Same real example input spectrogram (middle, harmonic call) used in the
# autoencoder figure, so both diagrams show an identical input example.
INPUT_EXAMPLE_IMG = Path("paper/Figures/_ae_diagram_input_example.png")


def trace_shapes(model: EncoderClassifier, hw: tuple[int, int]) -> dict:
    """Forward-trace shapes, keeping both pre-pool (conv) and post-pool shapes."""
    x = torch.zeros(1, 1, *hw)
    shapes = {"input": tuple(x.shape[1:])}
    h = x
    conv_pre, conv_post = [], []
    enc = model.encoder.encoder
    for layer in enc:
        h = layer(h)
        if isinstance(layer, torch.nn.ReLU):
            conv_pre.append(tuple(h.shape[1:]))  # (C, H, W) after conv+BN+ReLU, before pool
        if isinstance(layer, torch.nn.MaxPool2d):
            conv_post.append(tuple(h.shape[1:]))  # (C, H, W) after pool
    shapes["conv_pre"] = conv_pre
    shapes["conv"] = conv_post
    shapes["flat"] = int(torch.flatten(h, 1).shape[1])
    lat = model.encoder.to_latent
    shapes["fc_hidden"] = lat[0].out_features
    shapes["latent"] = lat[-1].out_features
    shapes["num_classes"] = model.classifier[-1].out_features
    return shapes

# Isometric "skew" direction shared by every box (dx, dy per unit depth).
ISO_DX, ISO_DY = 0.55, 0.32

# Color ramp, chosen to echo the autoencoder figure's pale palette:
# cream input -> blue conv ramp -> orange pool -> purple FC -> red latent.
C_INPUT = "#f5f0dd"
C_CONV = ["#dbe9f6", "#aecbe8", "#7fa8d6"]
C_POOL = "#f6d9b8"
C_FC = "#ddc2e8"
C_LATENT = "#8b3a3a"
C_HEAD = "#bcdba3"
C_OUT = "#cdeccd"

# Font sizes and the shared-trunk background color match the autoencoder
# figure exactly, so the two diagrams read as a matched pair.
FONT_LABEL = 13
FONT_SECTION = 17
FONT_NOTE = 12.5
C_ENCODER_BG = "#d7ecfa"


def _face_colors(base: str) -> tuple[str, str, str]:
    """Return (front, top, right) shades of a base hex color."""
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


def _rounded_polygon(ax, points, radius, **kwargs):
    """Single closed path with every corner (convex or concave) filleted by
    `radius`, so an L-shaped/stepped outline reads as one cohesive rounded
    shape instead of two overlapping rounded rectangles with a hard notch."""
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    n = len(points)
    verts, codes = [], []
    for i in range(n):
        prev_pt = np.array(points[(i - 1) % n])
        cur_pt = np.array(points[i])
        next_pt = np.array(points[(i + 1) % n])
        d_prev = prev_pt - cur_pt
        d_next = next_pt - cur_pt
        r = min(radius, np.linalg.norm(d_prev) / 2, np.linalg.norm(d_next) / 2)
        a = cur_pt + d_prev / np.linalg.norm(d_prev) * r
        b = cur_pt + d_next / np.linalg.norm(d_next) * r
        if i == 0:
            verts.append(tuple(a))
            codes.append(MplPath.MOVETO)
        else:
            verts.append(tuple(a))
            codes.append(MplPath.LINETO)
        verts.append(tuple(cur_pt))
        codes.append(MplPath.CURVE3)
        verts.append(tuple(b))
        codes.append(MplPath.CURVE3)
    verts.append(verts[0])
    codes.append(MplPath.CLOSEPOLY)
    ax.add_patch(PathPatch(MplPath(verts, codes), **kwargs))


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


def draw(model: EncoderClassifier, hw: tuple[int, int], out: Path) -> None:
    s = trace_shapes(model, hw)
    fig, ax = plt.subplots(figsize=(19, 11))
    ax.axis("off")

    row1_y0 = 15.0
    row2_y0 = 1.5

    # ---- Row 1: real input example -> Input box -> first two Conv+Pool
    # stages (split right after "MaxPool 64x30x26").
    cursor = 1.0
    c0, hh0, ww0 = s["input"]
    thumb = plt.imread(INPUT_EXAMPLE_IMG)
    thumb_h, thumb_w = _hbar_scale(hh0), 2.2
    ax.imshow(thumb, aspect="auto", cmap="viridis", zorder=3,
              extent=(cursor, cursor + thumb_w, row1_y0, row1_y0 + thumb_h))
    ax.text(cursor + thumb_w / 2, row1_y0 - 0.7, "Input\nexample", ha="center", va="top", fontsize=FONT_LABEL, zorder=3)
    cursor += thumb_w + 0.6
    _brace(ax, cursor, row1_y0, row1_y0 + thumb_h)
    cursor += 2.4

    warm_x0 = cursor
    hgt = _hbar_scale(hh0)
    p, cursor = _box(ax, cursor, row1_y0, 1.8, hgt, 1.0, C_INPUT, label_below=f"Input\n{c0}x{hh0}x{ww0}")
    cursor += 4.2

    for i in range(2):
        (ch, cph, cpw), (_, ph, pw) = s["conv_pre"][i], s["conv"][i]
        _arrow(ax, p, cursor)
        p, cursor = _box(ax, cursor, row1_y0, 2.4 + i * 0.9, _hbar_scale(cph), ch * 0.012, C_CONV[i],
                          label_below=f"Conv+BN+ReLU\n{ch}x{cph}x{cpw}")
        cursor += 4.0
        _arrow(ax, p, cursor)
        p, cursor = _box(ax, cursor, row1_y0, 1.8, _hbar_scale(ph), ch * 0.012, C_POOL,
                          label_below=f"MaxPool\n{ch}x{ph}x{pw}")
        cursor += 1.5 if i == 1 else 4.4

    row1_end = p
    row1_xmax = cursor

    # ---- Row 2: remaining Conv+Pool stage -> Flatten -> FC bottleneck (still
    # shared/warm-started), then the classifier-only latent dot + head + output.
    cursor = 1.0
    warm_x0_2 = cursor
    # Mid-height of the first row-2 box's left face, so the connector lands
    # perpendicular to it instead of sliding down along the edge.
    land_y = row2_y0 + _hbar_scale(s["conv_pre"][2][1]) / 2
    p = None
    for (ch, cph, cpw), (_, ph, pw) in zip(s["conv_pre"][2:], s["conv"][2:]):
        if p is not None:
            _arrow(ax, p, cursor)
        p, cursor = _box(ax, cursor, row2_y0, 2.4, _hbar_scale(cph), ch * 0.012, C_CONV[2],
                          label_below=f"Conv+BN+ReLU\n{ch}x{cph}x{cpw}")
        cursor += 4.0
        _arrow(ax, p, cursor)
        p, cursor = _box(ax, cursor, row2_y0, 1.8, _hbar_scale(ph), ch * 0.012, C_POOL,
                          label_below=f"MaxPool\n{ch}x{ph}x{pw}")
        cursor += 4.4

    _arrow(ax, p, cursor)
    p, cursor = _box(ax, cursor, row2_y0, 1.0, 1.6, 1.2, "#eceff1", label_below=f"Flatten\n{s['flat']:,}")
    cursor += 4.2

    _arrow(ax, p, cursor)
    p, cursor = _box(ax, cursor, row2_y0, 2.2, _hbar_scale(6), 2.6, C_FC,
                      label_below=f"FC {s['flat']:,}\n\u2192 {s['fc_hidden']}")
    cursor += 1.5

    warm_x1_2 = cursor

    # Single cohesive rounded-corner zone: an L-shaped/stepped outline (row 1
    # wider, row 2 narrower so it doesn't cover the classifier-only tail),
    # filleted at every corner -- including the step's notch -- so it reads
    # as one continuous rounded shape instead of two overlapping rectangles.
    enc_bg_x0 = min(warm_x0, warm_x0_2) - 0.8
    row1_bottom, row1_top = row1_y0 - 3.4, row1_y0 + hgt + 3.0
    row2_bottom, row2_top = row2_y0 - 3.4, row2_y0 + hgt + 3.0
    mid_y = (row1_bottom + row2_top) / 2
    row1_x1 = row1_xmax + 0.9
    row2_x1 = warm_x1_2 + 0.9
    enc_outline = [
        (enc_bg_x0, row1_top),
        (row1_x1, row1_top),
        (row1_x1, mid_y),
        (row2_x1, mid_y),
        (row2_x1, row2_bottom),
        (enc_bg_x0, row2_bottom),
    ]
    _rounded_polygon(ax, enc_outline, radius=0.9,
                      facecolor=C_ENCODER_BG, edgecolor="none", zorder=0)
    ax.text(enc_bg_x0 + 0.4, row1_top - 0.4, "Encoder trunk + FC bottleneck",
            ha="left", va="top", fontsize=FONT_SECTION, fontweight="bold", color="#1b4f72")
    cursor += 2.8

    # Latent / feature vector (red dot marker, same style as the AE figure)
    _arrow(ax, p, cursor)
    lat_x = cursor + 1.2
    lat_y = p[1]
    ax.scatter([lat_x], [lat_y], s=1100, color=C_LATENT, zorder=4, edgecolor="#2b2b2b", linewidth=1.6)
    ax.text(lat_x, lat_y - 1.4, f"Feature vector\n$z \\in \\mathbb{{R}}^{{{s['latent']}}}$",
            ha="center", va="top", fontsize=FONT_LABEL)
    p = (lat_x + 1.0, lat_y)
    cursor = lat_x + 3.6

    # Classification head (classifier-only, not part of the shared trunk)
    _arrow(ax, p, cursor)
    p, cursor = _box(ax, cursor, row2_y0, 1.4, _hbar_scale(6), 1.6, C_HEAD,
                      label_below=f"Linear+softmax\n{s['latent']}\u2192{s['num_classes']}")
    cursor += 4.2

    # Output readout
    _arrow(ax, p, cursor)
    _, cursor = _box(ax, cursor, row2_y0, 1.0, _hbar_scale(6), 0.6, C_OUT, label_below="P(call)\nin [0,1]")
    row2_xmax = cursor

    # Connector from the end of row 1 (first MaxPool) down into the start of
    # row 2 (second Conv stage), routed around a safe lane like the AE figure,
    # with a final horizontal leg so the arrowhead lands perpendicular to the
    # box's left face instead of sliding down along its corner.
    gap_y = row1_y0 - 5.0
    swerve_x = row1_xmax + 1.2
    lane_x = warm_x0_2 - 1.0
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    verts = [
        row1_end,
        (swerve_x, row1_end[1]),
        (swerve_x, gap_y),
        (lane_x, gap_y),
        (lane_x, land_y),
    ]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="none", edgecolor="#777777",
                            linewidth=1.8, zorder=2))
    ax.add_patch(FancyArrowPatch((lane_x, land_y), (warm_x0_2, land_y),
                                  arrowstyle="-|>", mutation_scale=18, linewidth=1.8,
                                  color="#777777", zorder=2))

    n_params = sum(p_.numel() for p_ in model.parameters())
    fig.suptitle(f"Supervised CNN classifier architecture ({n_params:,} parameters)",
                 fontsize=FONT_SECTION + 2, y=1.03)
    fig.text(0.5, 0.95, "Block dimensions shown as $C \\times H \\times W$ (channels x height x width)",
             ha="center", fontsize=FONT_NOTE, color="#37474f")

    ax.set_xlim(-1.4, max(row1_xmax, row2_xmax, warm_x1_2) + 1.5)
    ax.set_ylim(row2_y0 - 2.0, row1_y0 + hgt + 3.6)
    ax.set_aspect("auto")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    svg = out.with_suffix(".svg")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} and {svg}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the custom-CNN architecture (isometric style)")
    p.add_argument("--out", type=Path, default=Path("paper/Figures/cnn_classifier_architecture.png"))
    p.add_argument("--height", type=int, default=121)
    p.add_argument("--width", type=int, default=104)
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    model = EncoderClassifier(input_hw=(a.height, a.width))
    draw(model, (a.height, a.width), a.out)
