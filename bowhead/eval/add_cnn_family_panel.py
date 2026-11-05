"""Add a "Cold-start CNN vs. other CNN-family detectors" summary section to
docs/benchmark_pr_curves.html: a bar-chart summary across all 5 CNN-family
models evaluated in this project, plus full PR/FDR panels for every model
(all five now have reproducible full curve data).

All metrics use the REVIEWED (post-review) label set of
bowhead.eval.run_scratch_relabel_curves / bowhead.eval.run_birdnet_relabel_curves /
bowhead.benchmark.run_image_pr_curves; see bowhead.benchmark.plot_cnn_family_comparison
for the same data/caveats used to build the paper's version of this section.

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
BIRDNET_JSON = _REPO_ROOT / "runs" / "birdnet_relabel_100k_matched_full199825.json"
WARMSTART_JSON = _REPO_ROOT / "runs" / "warmstart_cnn_relabel_100k_matched_combined102126.json"
BACKBONES_JSON = _REPO_ROOT / "runs" / "resnet_efficientnet_relabel_100k_matched_combined102126.json"
MARKER = '<div id="cnn-family-panel-marker" style="display:none"></div>'

# Static (name, protocol note) pairs; AP/ROC-AUC/N are read live from each
# model's JSON in main() so this list can never go stale relative to the data.
MODEL_PROTOCOLS = [
    ("Scratch CNN (cold start)", "Full independent eval set"),
    ("Warm-start CNN (AE-init)", "Full independent eval set"),
    ("BirdNET/Perch 2.0", "Full eval set, Griffin-Lim + linear probe"),
    ("ResNet-18 (ImageNet)", "Full independent eval set, linear probe"),
    ("EfficientNet-B0 (ImageNet)", "Full independent eval set, linear probe"),
]
COLD_START = "Scratch CNN (cold start)"


def _bar_chart_html(models: list) -> str:
    names = [m[0] for m in models]
    aps = [m[1] for m in models]
    rocs = [m[2] for m in models]
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
        title=dict(text="Cold-start CNN vs. other CNN-family detectors (reviewed labels)",
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
        title=dict(text=f"{name} | n={n:,} (reviewed labels)", font=dict(size=14)),
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


def _summary_table_html(models: list) -> str:
    def _th(t: str) -> str:
        return f'<th style="padding:4px 12px;background:#eef;border:1px solid #ccc">{t}</th>'

    rows = ""
    for name, ap, roc, n, note in models:
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
    scratch = json.loads(SCRATCH_JSON.read_text())["reviewed"]
    birdnet = json.loads(BIRDNET_JSON.read_text())["reviewed"]
    warmstart = json.loads(WARMSTART_JSON.read_text())["reviewed"]
    backbones = json.loads(BACKBONES_JSON.read_text())
    resnet18 = backbones["resnet18"]
    efficientnet_b0 = backbones["efficientnet_b0"]

    results_by_name = {
        "Scratch CNN (cold start)": scratch,
        "Warm-start CNN (AE-init)": warmstart,
        "BirdNET/Perch 2.0": birdnet,
        "ResNet-18 (ImageNet)": resnet18,
        "EfficientNet-B0 (ImageNet)": efficientnet_b0,
    }
    models = [
        (name, r["average_precision"], r["roc_auc"], r["n"], note)
        for (name, note), r in zip(MODEL_PROTOCOLS, (results_by_name[n] for n, _ in MODEL_PROTOCOLS))
    ]

    section = (
        MARKER
        + '<h3 style="font-family:sans-serif;margin:22px auto 0 auto;max-width:1000px">'
        "Cold-start CNN vs. Other CNN-Family Detectors</h3>"
        '<div style="font-family:sans-serif;font-size:13px;color:#444;'
        'max-width:1000px;margin:6px auto 4px auto;padding:12px 18px;'
        'background:#f9f9f9;border:1px solid #ddd;border-radius:4px;line-height:1.6">'
        "All rows use the <b>reviewed</b> (post-review) label set and are now "
        "protocol-matched: identical full 199,825-image independent eval set, each "
        "probe/model trained on the same 100k-image matched training draw.<br><br>"
        + _summary_table_html(models) + "</div>"
        + _bar_chart_html(models)
        + _pr_fdr_panel_html(COLD_START, np.asarray(scratch["pr_precision"]),
                             np.asarray(scratch["pr_recall"]),
                             scratch["average_precision"], scratch["roc_auc"], scratch["n"])
        + _pr_fdr_panel_html("Warm-start CNN (AE-init)",
                             np.asarray(warmstart["pr_precision"]),
                             np.asarray(warmstart["pr_recall"]),
                             warmstart["average_precision"], warmstart["roc_auc"], warmstart["n"])
        + _pr_fdr_panel_html("BirdNET/Perch 2.0", np.asarray(birdnet["pr_precision"]),
                             np.asarray(birdnet["pr_recall"]),
                             birdnet["average_precision"], birdnet["roc_auc"], birdnet["n"])
        + _pr_fdr_panel_html("ResNet-18 (ImageNet)", np.asarray(resnet18["pr_precision"]),
                             np.asarray(resnet18["pr_recall"]),
                             resnet18["average_precision"], resnet18["roc_auc"], resnet18["n"])
        + _pr_fdr_panel_html("EfficientNet-B0 (ImageNet)", np.asarray(efficientnet_b0["pr_precision"]),
                             np.asarray(efficientnet_b0["pr_recall"]),
                             efficientnet_b0["average_precision"], efficientnet_b0["roc_auc"],
                             efficientnet_b0["n"])
    )

    html = HTML_PATH.read_text()
    marker_idx = html.find(MARKER)
    if marker_idx != -1:
        html = html[:marker_idx]
    else:
        body_idx = html.find("</body>")
        html = html[:body_idx] if body_idx != -1 else html
    html = html.rstrip() + "\n" + section + "\n</body>\n</html>\n"
    HTML_PATH.write_text(html)
    print(f"Inserted CNN-family comparison panel into {HTML_PATH}")


if __name__ == "__main__":
    main()
