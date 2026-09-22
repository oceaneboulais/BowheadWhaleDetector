"""Add a "Cold-start CNN vs. other CNN-family detectors" summary section to
docs/benchmark_pr_curves.html: a bar-chart summary across all 5 CNN-family
models evaluated in this project, plus separate PR/FDR panels for the two
models with reproducible full curve data (scratch CNN, BirdNET/Perch 2.0).

All metrics use the ORIGINAL (pre-review) label set for apples-to-apples
comparison; see bowhead.benchmark.plot_cnn_family_comparison for the same
data/caveats used to build the paper's version of this section.

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.eval.add_cnn_family_panel
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bowhead.eval.relabel_panel_common import downsample_curve

_REPO_ROOT = Path(__file__).resolve().parents[2]
HTML_PATH = _REPO_ROOT / "docs" / "benchmark_pr_curves.html"
SCRATCH_JSON = _REPO_ROOT / "runs" / "scratch_cnn_relabel_100k_matched.json"
BIRDNET_JSON = _REPO_ROOT / "runs" / "birdnet_relabel_100k_matched.json"
WARMSTART_NPZ = _REPO_ROOT / "runs" / "pr_curves_100k_matched.npz"

# name -> (AP, ROC-AUC, n, protocol note)
MODELS = [
    ("Scratch CNN (cold start)", 0.8090, 0.9637, 199_825, "Full independent eval set"),
    ("Warm-start CNN (AE-init)", 0.7869, 0.9563, 199_825, "Full independent eval set (re-scored 2026-09-04)"),
    ("BirdNET/Perch 2.0", 0.4827, 0.8025, 4_000, "Subsample, Griffin-Lim + linear probe"),
    ("ResNet-18 (ImageNet)", 0.6637, 0.8886, 199_825, "Grouped 15% held-out split, linear probe"),
    ("EfficientNet-B0 (ImageNet)", 0.7134, 0.9134, 199_825, "Grouped 15% held-out split, linear probe"),
]
COLD_START = "Scratch CNN (cold start)"


def _bar_chart_html() -> str:
    names = [m[0] for m in MODELS]
    aps = [m[1] for m in MODELS]
    rocs = [m[2] for m in MODELS]
    colors = ["#c0392b" if n == COLD_START else "#4e9af1" for n in names]

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.15,
                         subplot_titles=["Average Precision", "ROC-AUC"])
    fig.add_trace(go.Bar(y=names, x=aps, orientation="h", marker_color=colors,
                          text=[f"{v:.3f}" for v in aps], textposition="outside"),
                  row=1, col=1)
    fig.add_trace(go.Bar(y=names, x=rocs, orientation="h", marker_color=colors,
                          text=[f"{v:.3f}" for v in rocs], textposition="outside"),
                  row=1, col=2)
    fig.update_layout(
        title=dict(text="Cold-start CNN vs. other CNN-family detectors (original labels)",
                   font=dict(size=14)),
        xaxis=dict(range=[0, 1.05]), xaxis2=dict(range=[0, 1.05]),
        yaxis=dict(autorange="reversed"), yaxis2=dict(autorange="reversed", showticklabels=False),
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        height=360, width=1000, margin=dict(t=60, l=220),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _pr_fdr_panel_html(name: str, prec: np.ndarray, rec: np.ndarray,
                        ap: float, roc_auc: float, n: int) -> str:
    prec, rec = downsample_curve(prec, rec)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12, subplot_titles=["", ""])
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", line=dict(color="#4e9af1", width=2),
                              name=f"{name} (AP={ap:.3f})"), row=1, col=1)
    fig.add_trace(go.Scatter(x=1 - rec, y=1 - prec, mode="lines",
                              line=dict(color="#4e9af1", width=2), showlegend=False), row=1, col=2)
    fig.update_layout(
        title=dict(text=f"{name} | n={n:,} (original labels)", font=dict(size=14)),
        xaxis=dict(title="Recall", range=[0, 1]), yaxis=dict(title="Precision", range=[0, 1.02]),
        xaxis2=dict(title="Miss Fraction (1-Recall)", range=[0, 1]),
        yaxis2=dict(title="False Discovery Rate (1-Precision)", range=[0, 1.02]),
        plot_bgcolor="white", paper_bgcolor="white", height=420, width=1000, margin=dict(t=60),
        annotations=[
            dict(text="Precision-Recall", x=0.22, xref="paper", y=1.0, yref="paper",
                 xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=14)),
            dict(text="FDR vs Miss Fraction", x=0.78, xref="paper", y=1.0, yref="paper",
                 xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=14)),
        ],
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _summary_table_html() -> str:
    def _th(t: str) -> str:
        return f'<th style="padding:4px 12px;background:#eef;border:1px solid #ccc">{t}</th>'

    rows = ""
    for name, ap, roc, n, note in MODELS:
        bold = "font-weight:bold;" if name == COLD_START else ""
        rows += (
            f'<tr style="{bold}"><td style="padding:4px 12px;border:1px solid #ddd">{name}</td>'
            f'<td style="padding:4px 12px;border:1px solid #ddd;text-align:right">{n:,}</td>'
            f'<td style="padding:4px 12px;border:1px solid #ddd;text-align:right">{ap:.4f}</td>'
            f'<td style="padding:4px 12px;border:1px solid #ddd;text-align:right">{roc:.4f}</td>'
            f'<td style="padding:4px 12px;border:1px solid #ddd">{note}</td></tr>\n'
        )
    return (
        '<table style="border-collapse:collapse;font-size:12px">'
        "<thead><tr>" + _th("Model") + _th("N") + _th("AP &uarr;") + _th("ROC-AUC &uarr;")
        + _th("Protocol") + "</tr></thead><tbody>\n" + rows + "</tbody></table>"
    )


def main() -> None:
    scratch = json.loads(SCRATCH_JSON.read_text())["original"]
    birdnet = json.loads(BIRDNET_JSON.read_text())["original"]
    warmstart = np.load(WARMSTART_NPZ)

    section = (
        '<h3 style="font-family:sans-serif;margin:22px auto 0 auto;max-width:1000px">'
        "Cold-start CNN vs. Other CNN-Family Detectors</h3>"
        '<div style="font-family:sans-serif;font-size:13px;color:#444;'
        'max-width:1000px;margin:6px auto 4px auto;padding:12px 18px;'
        'background:#f9f9f9;border:1px solid #ddd;border-radius:4px;line-height:1.6">'
        "All rows use the <b>original</b> (pre-review) label set, since not every "
        "model has been re-scored against the reviewed labels. Protocols differ by "
        "model (see Protocol column) &mdash; this is an orientation comparison, not a "
        "single controlled experiment.<br><br>"
        + _summary_table_html() + "</div>"
        + _bar_chart_html()
        + '<div style="font-family:sans-serif;font-size:12px;color:#666;'
          'max-width:1000px;margin:8px auto 8px auto;padding:8px 18px;'
          'background:#fff8e6;border:1px solid #f0d78c;border-radius:4px">'
          "<b>Note:</b> full PR/FDR curves below are shown for the three models "
          "whose per-threshold arrays were saved end-to-end (scratch CNN, warm-start "
          "CNN, BirdNET/Perch 2.0); ResNet-18 and EfficientNet-B0 currently only "
          "have reproducible scalar summary metrics in this repo.</div>"
        + _pr_fdr_panel_html(COLD_START, np.asarray(scratch["pr_precision"]),
                             np.asarray(scratch["pr_recall"]),
                             scratch["average_precision"], scratch["roc_auc"], scratch["n"])
        + _pr_fdr_panel_html("Warm-start CNN (AE-init)",
                             np.asarray(warmstart["warmstart_100k_matched_precision"]),
                             np.asarray(warmstart["warmstart_100k_matched_recall"]),
                             float(warmstart["warmstart_100k_matched_ap"]),
                             float(warmstart["warmstart_100k_matched_roc_auc"]),
                             int(warmstart["n"]))
        + _pr_fdr_panel_html("BirdNET/Perch 2.0", np.asarray(birdnet["pr_precision"]),
                             np.asarray(birdnet["pr_recall"]),
                             birdnet["average_precision"], birdnet["roc_auc"], birdnet["n"])
    )

    html = HTML_PATH.read_text()
    html = html.replace("</body>", section + "\n</body>")
    HTML_PATH.write_text(html)
    print(f"Inserted CNN-family comparison panel into {HTML_PATH}")


if __name__ == "__main__":
    main()
