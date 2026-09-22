"""Shared Plotly panel + summary-card builder for "before vs. after relabeling"
ablation panels inserted into docs/benchmark_pr_curves.html.

Used by bowhead.eval.add_relabel_panel (scratch CNN) and
bowhead.eval.add_extra_relabel_panels (Moan Detector, BirdNET) so the three
panels share identical layout/styling.

Each *results* dict must have the shape produced by
bowhead.eval.run_scratch_relabel_curves.run() / its Moan-Detector / BirdNET
analogues:
    {"reviewed": {...DetectionMetrics fields...},
     "original": {...DetectionMetrics fields...},
     "_meta": {"n": int, "n_labels_changed_by_review": int, ...}}
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "reviewed": "#4e9af1",   # blue  — post-review ("cleaned") labels
    "original": "#f07040",   # orange — pre-review ("type_org") labels
}
LABELS = {
    "reviewed": "Reviewed labels (post-review, iscall)",
    "original": "Original labels (pre-review, type_org)",
}


def downsample_curve(prec: np.ndarray, rec: np.ndarray, max_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Thin a precision/recall curve for plotting only (AP/ROC-AUC are unaffected, since
    those are computed from the full arrays before this is called)."""
    n = len(prec)
    if n <= max_points:
        return prec, rec
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    return prec[idx], rec[idx]


def figure_html(results: dict, title_prefix: str) -> str:
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12, subplot_titles=["", ""])

    for name in ("reviewed", "original"):
        m = results[name]
        prec = np.asarray(m["pr_precision"])
        rec = np.asarray(m["pr_recall"])
        prec, rec = downsample_curve(prec, rec)
        color = COLORS[name]
        label = f"{LABELS[name]}  (AP={m['average_precision']:.3f})"
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=label,
                                  line=dict(color=color, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=1 - rec, y=1 - prec, mode="lines", name=label,
                                  showlegend=False, line=dict(color=color, width=2)),
                       row=1, col=2)

    n = results["reviewed"]["n"]
    fig.update_layout(
        title=dict(text=f"{title_prefix} | n={n:,}  (same frozen predictions)", font=dict(size=14)),
        xaxis=dict(title="Recall", range=[0, 1], showgrid=True,
                   gridcolor="#e5e7eb", gridwidth=1, griddash="solid"),
        yaxis=dict(title="Precision", range=[0, 1.02], showgrid=True,
                   gridcolor="#e5e7eb", gridwidth=1, griddash="solid"),
        xaxis2=dict(title="Miss Fraction (1-Recall)", range=[0, 1], showgrid=True,
                    gridcolor="#e5e7eb", gridwidth=1, griddash="solid"),
        yaxis2=dict(title="False Discovery Rate (1-Precision)", range=[0, 1.02],
                    showgrid=True, gridcolor="#e5e7eb", gridwidth=1, griddash="solid"),
        legend=dict(x=0.01, y=0.15, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ccc", borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white",
        height=460, width=1000,
        margin=dict(t=60),
        annotations=[
            dict(text="Precision-Recall", x=0.22, xref="paper", y=1.0, yref="paper",
                 xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=14)),
            dict(text="FDR vs Miss Fraction", x=0.78, xref="paper", y=1.0, yref="paper",
                 xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=14)),
        ],
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def summary_card(results: dict, model_desc_html: str) -> str:
    rev, orig = results["reviewed"], results["original"]
    meta = results["_meta"]
    n = meta["n"]
    n_flip = meta["n_labels_changed_by_review"]

    d_ap = rev["average_precision"] - orig["average_precision"]
    d_roc = rev["roc_auc"] - orig["roc_auc"]
    sign_ap = "+" if d_ap >= 0 else ""
    sign_roc = "+" if d_roc >= 0 else ""

    def _row(name: str) -> str:
        m = results[name]
        color = COLORS[name]
        return (
            "<tr>"
            f'<td style="padding:4px 12px;border:1px solid #ddd">'
            f'<span style="color:{color};font-size:16px">&#9644;</span>&nbsp;'
            f"<strong>{LABELS[name]}</strong></td>"
            f'<td style="padding:4px 12px;border:1px solid #ddd;text-align:right">{m["prevalence"]:.4f}</td>'
            f'<td style="padding:4px 12px;border:1px solid #ddd;text-align:right">{m["average_precision"]:.4f}</td>'
            f'<td style="padding:4px 12px;border:1px solid #ddd;text-align:right">{m["roc_auc"]:.4f}</td>'
            f'<td style="padding:4px 12px;border:1px solid #ddd;text-align:right">{m["precision_at_recall_0_70"]:.4f}</td>'
            "</tr>\n"
        )

    def _th(t: str) -> str:
        return f'<th style="padding:4px 12px;background:#eef;border:1px solid #ccc">{t}</th>'

    return (
        '<div style="font-family:sans-serif;font-size:13px;color:#444;'
        'max-width:1000px;margin:6px auto 4px auto;padding:12px 18px;'
        'background:#f9f9f9;border:1px solid #ddd;border-radius:4px;line-height:1.6">'
        f"<strong>Relabeling summary</strong> &mdash; {model_desc_html}, scored "
        f"<b>once</b> on n={n:,}, then re-graded against two label sets pulled "
        "from the same MATLAB export (<code style='font-size:11px;background:#efefef;"
        "padding:1px 5px;border-radius:3px'>latent_embeddings_3d_eval_8to1_MATLAB_Raquel.mat</code>): "
        "the <b>original</b> pre-review labels (<code>type_org</code>) vs. the "
        f"manually <b>reviewed</b> labels (<code>iscall</code>). "
        f"<b>{n_flip:,}</b> of {n:,} labels ({n_flip / n:.2%}) were changed by the review.<br><br>"
        '<table style="border-collapse:collapse;font-size:12px">'
        "<thead><tr>"
        + _th("Label set") + _th("Prevalence") + _th("AP &uarr;") + _th("ROC-AUC &uarr;") + _th("P@R0.70 &uarr;")
        + "</tr></thead><tbody>\n"
        + _row("reviewed")
        + _row("original")
        + "</tbody></table><br>"
        f"<b>&Delta;AP (reviewed &minus; original) = {sign_ap}{d_ap:.4f}</b> &nbsp; "
        f"<b>&Delta;ROC-AUC = {sign_roc}{d_roc:.4f}</b> &mdash; "
        "because the underlying probabilities are identical in both rows, this "
        "isolates how much of the apparent accuracy change comes purely from "
        "correcting mislabeled eval examples, versus any change in the detector "
        "itself (cf. the AE+kNN reviewed-vs-original comparison in the manuscript)."
        "</div>"
    )


def section_html(heading: str, title_prefix: str, model_desc_html: str, results: dict,
                  caveat_html: str | None = None) -> str:
    html = (
        f'<h3 style="font-family:sans-serif;margin:22px auto 0 auto;max-width:1000px">{heading}</h3>'
        + figure_html(results, title_prefix)
        + summary_card(results, model_desc_html)
    )
    if caveat_html:
        html += (
            '<div style="font-family:sans-serif;font-size:12px;color:#666;'
            'max-width:1000px;margin:0 auto 18px auto;padding:8px 18px;'
            'background:#fff8e6;border:1px solid #f0d78c;border-radius:4px">'
            f"<b>Caveat:</b> {caveat_html}</div>"
        )
    return html
