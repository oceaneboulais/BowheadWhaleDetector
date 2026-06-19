"""Generate docs/detection_curves.html from a saved PR-curves .npz file.

This script loads pre-computed precision-recall arrays (written by the scoring
pipeline) and produces a fully self-contained Plotly HTML file — no server, no
extra dependencies beyond numpy and plotly.

Typical usage (from repo root):
    python -m bowhead.eval.build_curves_html

Or with explicit paths:
    python -m bowhead.eval.build_curves_html \\
        --npz  runs/pr_curves_full_dataset.npz \\
        --out  docs/detection_curves.html
"""

from __future__ import annotations

import argparse
import base64
import io
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE   = "#4e9af1"   # scratch CNN
ORANGE = "#f07040"   # warm-start CNN
GREY   = "#aaaaaa"   # random baseline

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Per-type metadata (confirmed from evaluation dataset) ──────────────────────
_TYPE_COUNTS = {0: 177_666, 1: 6_543, 2: 4_700, 3: 3_438, 4: 2_143, 5: 1_638, 6: 704, 7: 2_993}
_TYPE_INFO = {
    0: ("Non-call transient",
        "Auto-detected by the Thode (2012) detector; not manually verified. "
        "Includes airgun pulses (~40% of transients), vessel noise, and other "
        "biological / abiotic sounds."),
    1: ("Bowhead call \u2013 Type 1",
        "Manually verified bowhead whale vocalization, morphological class 1 "
        "(most abundant \u2014 6,543 calls)."),
    2: ("Bowhead call \u2013 Type 2", "Manually verified, morphological class 2 (4,700 calls)."),
    3: ("Bowhead call \u2013 Type 3", "Manually verified, morphological class 3 (3,438 calls)."),
    4: ("Bowhead call \u2013 Type 4", "Manually verified, morphological class 4 (2,143 calls)."),
    5: ("Bowhead call \u2013 Type 5", "Manually verified, morphological class 5 (1,638 calls)."),
    6: ("Bowhead call \u2013 Type 6", "Manually verified, morphological class 6 (rarest \u2014 704 calls)."),
    7: ("Bowhead call \u2013 Type 7", "Manually verified, morphological class 7 (2,993 calls)."),
}


def _make_thumbnails(eval_dir: Path) -> dict:
    """Return {type_id: base64 data-URI} for one example SNR_gram per type."""
    from scipy.io import loadmat
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _RE = re.compile(r"_Type(\d+)$")
    examples: dict = {}
    for fp in sorted(eval_dir.glob("*.mat")):
        m = _RE.search(fp.stem)
        if not m:
            continue
        t = int(m.group(1))
        if t in examples:
            continue
        try:
            arr = loadmat(str(fp))["SNR_gram"].astype(np.float32)
            examples[t] = arr
        except Exception:
            pass
        if len(examples) == 8:
            break

    thumbs: dict = {}
    for t, arr in examples.items():
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        fig, ax = plt.subplots(figsize=(1.04, 1.21), dpi=80)
        ax.imshow(arr, aspect="auto", origin="lower", cmap="inferno",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
        thumbs[t] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return thumbs


def build_html(npz_path: Path, out_path: Path, eval_dir: Path | None = None) -> None:
    d    = np.load(npz_path)
    s_p  = d["scratch_precision"]
    s_r  = d["scratch_recall"]
    w_p  = d["warmstart_precision"]
    w_r  = d["warmstart_recall"]
    sap  = float(d["scratch_ap"])
    wap  = float(d["warmstart_ap"])
    n    = int(d["n"])
    prev = float(d["prevalence"])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["", ""],
        horizontal_spacing=0.12,
    )

    # ── Precision-Recall ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=s_r, y=s_p, mode="lines",
        name=f"Scratch CNN (AP={sap:.3f})",
        line=dict(color=BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=w_r, y=w_p, mode="lines",
        name=f"Warm-start CNN (AP={wap:.3f})",
        line=dict(color=ORANGE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[prev, prev], mode="lines",
        name="Random baseline (prevalence)",
        line=dict(color=GREY, dash="dash", width=1.5)), row=1, col=1)

    # ── FDR vs Miss Fraction ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=1 - s_r, y=1 - s_p, mode="lines",
        name=f"Scratch CNN (AP={sap:.3f})",
        line=dict(color=BLUE, width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=1 - w_r, y=1 - w_p, mode="lines",
        name=f"Warm-start CNN (AP={wap:.3f})",
        line=dict(color=ORANGE, width=2), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[1 - prev, 1 - prev], mode="lines",
        name="Random baseline",
        line=dict(color=GREY, dash="dash", width=1.5), showlegend=False),
        row=1, col=2)

    fig.update_layout(
        title=dict(
            text=(f"Custom CNN detector curves on FULL DATASET"
                  f" | n={n:,}  prevalence={prev:.3f}"),
            font=dict(size=15),
        ),
        xaxis =dict(title="Recall",                           range=[0, 1]),
        yaxis =dict(title="Precision",                        range=[0, 1.02]),
        xaxis2=dict(title="Miss Fraction (1 − Recall)",       range=[0, 1]),
        yaxis2=dict(title="False Discovery Rate (1 − Precision)", range=[0, 1.02]),
        legend=dict(
            x=0.01, y=0.15,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc", borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500, width=1100,
        annotations=[
            dict(text="Precision-Recall (all points)",
                 x=0.22, xanchor="center", xref="paper",
                 y=1.0, yanchor="bottom", yref="paper",
                 showarrow=False, font=dict(size=16)),
            dict(text="False Discovery Rate vs Miss Fraction (all points)",
                 x=0.78, xanchor="center", xref="paper",
                 y=1.0, yanchor="bottom", yref="paper",
                 showarrow=False, font=dict(size=16)),
        ],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)

    # ── Description panel ─────────────────────────────────────────────────────
    n_calls    = int(round(n * prev))
    n_noncalls = n - n_calls

    # Optional per-type spectrogram thumbnails
    thumbs: dict = {}
    if eval_dir is not None:
        try:
            thumbs = _make_thumbnails(eval_dir)
            print(f"  Generated {len(thumbs)} spectrogram thumbnails")
        except Exception as exc:
            print(f"  Note: could not generate thumbnails: {exc}")

    # ── helpers for table cells
    def _th(txt: str, extra: str = "") -> str:
        return (f'<th style="padding:4px 10px;background:#eef;'
                f'border:1px solid #ccc;{extra}">{txt}</th>')

    def _td(txt: str, extra: str = "") -> str:
        return f'<td style="padding:4px 10px;border:1px solid #ccc;{extra}">{txt}</td>'

    rows_html = ""
    for t in sorted(_TYPE_INFO):
        label, desc = _TYPE_INFO[t]
        count = _TYPE_COUNTS.get(t, "\u2014")
        badge_color = "#c44" if t == 0 else "#2a7"
        badge_text  = "non-call" if t == 0 else "call"
        badge = (f'<span style="background:{badge_color};color:#fff;'
                 f'padding:1px 6px;border-radius:3px;font-size:11px">{badge_text}</span>')
        img_cell = (f'<img src="{thumbs[t]}" '
                    f'style="height:60px;display:block;margin:auto;image-rendering:pixelated">'
                    if t in thumbs else "\u2014")
        rows_html += (
            "<tr>"
            + _td(f"<strong>Type {t}</strong>")
            + _td(f"{count:,}" if isinstance(count, int) else count,
                  "text-align:right")
            + _td(badge, "text-align:center")
            + _td(label)
            + _td(desc, "font-size:11px;color:#666;max-width:320px")
            + _td(img_cell, "text-align:center;padding:2px 6px")
            + "</tr>\n"
        )

    dataset_name = (
        "Unsupervised_database_Evaluation_200K_"
        "8Auto1Manual_ADG_Y08101214_centered_06May2026.dir"
    )
    dataset_section = (
        '<div style="font-family:sans-serif;font-size:13px;color:#444;'
        'max-width:1100px;margin:8px auto 10px auto;padding:12px 16px;'
        'background:#f9f9f9;border:1px solid #ddd;border-radius:4px;line-height:1.6">'
        "<strong>Evaluation dataset</strong><br>"
        f'<code style="font-size:11px;background:#efefef;padding:1px 5px;'
        f'border-radius:3px">{dataset_name}</code><br><br>'
        f"<strong>{n:,} spectrograms</strong> &mdash; 121&times;104&nbsp;px uint8 "
        "SNR-gram images, centered on each detection window.<br>"
        "Recorded at DASAR hydrophone arrays <strong>A, D, G</strong> &mdash; "
        "Beaufort Sea &mdash; years <strong>2008, 2010, 2012, 2014</strong> "
        "(sites 3 &amp; 5).<br>"
        f"Label breakdown: <strong>{n_calls:,} manually verified bowhead calls</strong> "
        f"(Types 1\u20137) + <strong>{n_noncalls:,} auto-detected transients</strong> "
        f"(Type 0) &rarr; <strong>prevalence&nbsp;=&nbsp;{prev:.4f}</strong> "
        f"(1 call per {1/prev - 1:.1f} non-calls on average).<br><br>"
        '<table style="border-collapse:collapse;width:100%;font-size:12px">'
        "<thead><tr>"
        + _th("Type") + _th("Count") + _th("Class")
        + _th("Label") + _th("Description") + _th("Example SNR-gram")
        + "</tr></thead><tbody>\n"
        + rows_html
        + "</tbody></table>"
        "<br><em style='font-size:11px;color:#888'>"
        "Type taxonomy follows the Thode et al. (2012) automated bowhead detector "
        "database. Example spectrograms show the raw SNR-gram (frequency \u00d7 time, "
        "inferno colormap, per-sample min-max normalised). "
        "One file was skipped (airgun_index.mat, not a spectrogram)."
        "</em>"
        "</div>"
    )

    how_to_read = (
        '<div style="font-family:sans-serif;font-size:13px;color:#444;'
        'max-width:1100px;margin:0 auto 10px auto;padding:12px 16px;'
        'background:#f9f9f9;border:1px solid #ddd;border-radius:4px;line-height:1.6">'
        "<strong>How to read these charts</strong><br>"
        "<strong>Left &mdash; Precision-Recall curve:</strong> "
        "As the detection threshold is swept from strict (high precision, low recall) "
        "to permissive (low precision, high recall), each model traces a curve. "
        "Higher curves &rarr; better detectors. "
        "Average Precision (AP, shown in the legend) is the area under the curve "
        "(1&nbsp;=&nbsp;perfect).<br><br>"
        "<strong>Right &mdash; False Discovery Rate vs Miss Fraction:</strong> "
        "The mirror view &mdash; FDR = 1&nbsp;&minus;&nbsp;Precision "
        "(fraction of flagged events that are <em>not</em> calls); "
        "Miss Fraction = 1&nbsp;&minus;&nbsp;Recall (fraction of real calls missed). "
        "Lower curves &rarr; better detectors.<br><br>"
        "<strong>Models &amp; baselines:</strong>&nbsp;"
        f'<span style="color:{BLUE}">&#9644; <strong>Scratch CNN (AP={sap:.3f})</strong></span>'
        " &mdash; custom convolutional classifier, random initialisation. "
        f'<span style="color:{ORANGE}">&#9644; <strong>Warm-start CNN (AP={wap:.3f})</strong></span>'
        " &mdash; same architecture, encoder trunk pre-initialised from the convolutional "
        "autoencoder and then fine-tuned end-to-end. "
        '<span style="color:#888">&#9135;&nbsp;&#9135; <strong>Random baseline</strong></span>'
        " &mdash; horizontal line at precision&nbsp;=&nbsp;prevalence "
        "(AP of a zero-skill classifier). Any useful detector must lie clearly "
        "above this line on the PR plot."
        "</div>"
    )

    scratch_note = (
        '<div style="font-family:sans-serif;font-size:13px;color:#444;'
        'max-width:1100px;margin:0 auto 24px auto;padding:12px 16px;'
        'background:#fffbf0;border:1px solid #e8d88a;border-radius:4px;line-height:1.6">'
        "<strong>&#9888;&nbsp;Why does the Scratch CNN curve have a flat shelf "
        "at the bottom-right?</strong><br><br>"
        "The scratch curve shows a characteristic <em>precision shelf</em>: from "
        "recall&nbsp;\u2248&nbsp;0.86 to recall&nbsp;=&nbsp;1.0, precision stays "
        "pinned near the class prior (0.11) rather than rising. "
        "Two factors cause this:<br><br>"
        "<strong>1.&nbsp;Under-training.</strong> "
        "The scratch model achieved its best validation ROC-AUC of 0.933 at "
        "<em>epoch&nbsp;2</em> of training on a small balanced demo dataset, "
        "then early-stopped 10 epochs later. "
        "After only 2 epochs from random initialisation the model has learned "
        "coarse discrimination, but its softmax outputs are compressed near&nbsp;0.5 "
        "for the majority of samples &mdash; it has not yet learned to confidently "
        "push non-call scores toward&nbsp;0.<br><br>"
        "<strong>2.&nbsp;Score compression near 0.5.</strong> "
        "<strong>9,928</strong> distinct threshold values all produce "
        "recall&nbsp;=&nbsp;1.0 (all 22,159 calls flagged) at "
        "precision&nbsp;\u2248&nbsp;prevalence. "
        "This means the model\u2019s lowest call score is barely higher than its "
        "typical non-call score. Below threshold&nbsp;\u2248&nbsp;0.5, both classes "
        "are flagged indiscriminately and precision collapses to the class prior "
        "while recall stays at&nbsp;1. "
        "This creates the dense vertical stack of curve points crowded into the "
        "recall&nbsp;0.86&ndash;1.0 band.<br><br>"
        "By contrast, the warm-start model (22 epochs, AE-pretrained encoder) has "
        "only <strong>1,391</strong> such degenerate threshold levels, "
        "resulting in a smoother, better-separated curve. "
        "Note that despite the shelf, the scratch model\u2019s overall "
        f"AP ({sap:.3f}) is <em>higher</em> than warm-start\u2019s ({wap:.3f}): "
        "at high confidence thresholds (recall&nbsp;&lt;&nbsp;0.86) the scratch "
        "model is very precise &mdash; when it is certain, it is usually correct."
        "</div>"
    )

    panel = dataset_section + how_to_read + scratch_note

    html = out_path.read_text()
    html = html.replace("</body>", panel + "\n</body>")
    out_path.write_text(html)

    print(f"Written {out_path}")
    print(f"scratch AP={sap:.4f}  warmstart AP={wap:.4f}  n={n:,}  prevalence={prev:.4f}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", type=Path,
                   default=_REPO_ROOT / "runs" / "pr_curves_full_dataset.npz",
                   help="Path to the PR-curves .npz file (default: runs/pr_curves_full_dataset.npz)")
    p.add_argument("--out", type=Path,
                   default=_REPO_ROOT / "docs" / "detection_curves.html",
                   help="Output HTML path (default: docs/detection_curves.html)")
    p.add_argument("--eval-dir", dest="eval_dir", type=Path, default=None,
                   help="Evaluation .dir folder; if given, embeds one example "
                        "spectrogram per call type in the HTML panel.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_html(args.npz, args.out, eval_dir=args.eval_dir)
