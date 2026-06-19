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
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Palette ───────────────────────────────────────────────────────────────────
BLUE   = "#4e9af1"   # scratch CNN
ORANGE = "#f07040"   # warm-start CNN
GREY   = "#aaaaaa"   # random baseline

_REPO_ROOT = Path(__file__).resolve().parents[2]


def build_html(npz_path: Path, out_path: Path) -> None:
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
    panel = (
        '<div style="font-family:sans-serif;font-size:13px;color:#444;'
        'max-width:1100px;margin:8px auto 24px auto;padding:12px 16px;'
        'background:#f9f9f9;border:1px solid #ddd;border-radius:4px;line-height:1.6">'
        "<strong>How to read these charts</strong><br>"
        "<strong>Left &mdash; Precision-Recall curve:</strong> "
        "As the detection threshold is swept from strict (high precision, low recall) "
        "to permissive (low precision, high recall), each model traces a curve. "
        "Higher curves &rarr; better detectors. Average Precision (AP, shown in the legend) "
        "summarises the area under the curve in a single number (1&nbsp;=&nbsp;perfect).<br><br>"
        "<strong>Right &mdash; False Discovery Rate vs Miss Fraction:</strong> "
        "The mirror view of the PR curve &mdash; FDR = 1&nbsp;&minus;&nbsp;Precision "
        "(fraction of flagged events that are not calls); "
        "Miss Fraction = 1&nbsp;&minus;&nbsp;Recall (fraction of real calls that are missed). "
        "Lower curves &rarr; better detectors.<br><br>"
        "<strong>Models &amp; baselines:</strong> "
        f'<span style="color:{BLUE}">&#9644; <strong>Scratch CNN</strong></span> &mdash; '
        "custom convolutional classifier trained from random initialisation. "
        f'<span style="color:{ORANGE}">&#9644; <strong>Warm-start CNN</strong></span> &mdash; '
        "same architecture but the encoder trunk is pre-trained with the convolutional autoencoder, "
        "then fine-tuned end-to-end. "
        '<span style="color:#888">&#9135; &#9135; <strong>Random baseline</strong></span> &mdash; '
        "a classifier that assigns every spectrogram the same constant score (no discrimination ability). "
        "On the PR curve this appears as a horizontal line at <em>precision&nbsp;=&nbsp;prevalence</em>; "
        "on the FDR curve it is a horizontal line at <em>FDR&nbsp;=&nbsp;1&nbsp;&minus;&nbsp;prevalence</em>. "
        "Any useful detector must lie clearly above this line on the PR plot (and below it on the FDR plot).<br><br>"
        f"<em>Dataset: {n:,} spectrograms ({n_calls:,} manually verified bowhead calls "
        f"+ {n_noncalls:,} auto-detected transients), "
        f"sites 3 &amp; 5, Beaufort Sea 2008&ndash;2014. "
        f"Prevalence&nbsp;=&nbsp;{prev:.3f} (call fraction in this balanced dataset).</em>"
        "</div>"
    )

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
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_html(args.npz, args.out)
