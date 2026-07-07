"""HTML report generator for a completed benchmark run.

Reads a ``meta.json`` + per-backbone ``.npz`` files produced by
``run_benchmark.py`` and assembles a self-contained Plotly HTML page with:

  - Summary table (full-train AP + ROC-AUC per backbone)
  - Few-shot learning curves (AP vs k for every backbone)
  - PR curves overlay (full-train, same axes as the CNN detector comparison)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ── Colour palette (consistent with run_image_pr_curves) ─────────────────────
_COLOURS: dict[str, str] = {
    "scratch_cnn":      "#4e9af1",
    "warmstart_cnn":    "#f07040",
    "resnet18":         "#2ca02c",
    "efficientnet_b0":  "#9467bd",
    "ast_imagenet":     "#8c564b",
}
_DEFAULT_COLOUR = "#17becf"


def _colour(name: str) -> str:
    return _COLOURS.get(name, _DEFAULT_COLOUR)


def build_report(
    meta: dict,
    run_dir: Path,
    out_path: Path,
) -> None:
    """Build a self-contained HTML report from ``meta`` + run-dir artefacts.

    Parameters
    ----------
    meta     : dict loaded from ``meta.json``
    run_dir  : directory containing per-backbone subdirectories
    out_path : output HTML file path
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    backbones = meta.get("backbones", [])
    prevalence = float(meta.get("prevalence", 0.1))
    n = int(meta.get("n", 0))

    # ── 1. PR-curve overlay ───────────────────────────────────────────────
    fig_pr = go.Figure()
    # Random baseline
    fig_pr.add_trace(go.Scatter(
        x=[0, 1], y=[prevalence, prevalence], mode="lines",
        name="Random baseline",
        line=dict(color="#aaaaaa", dash="dash", width=1.5)))

    for bb in backbones:
        npz_path = run_dir / bb / "detection_full.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        ap = float(d["ap"])
        fig_pr.add_trace(go.Scatter(
            x=d["recall"], y=d["precision"], mode="lines",
            name=f"{bb} (AP={ap:.3f})",
            line=dict(color=_colour(bb), width=2)))

    fig_pr.update_layout(
        title=f"Precision-Recall — frozen backbone probes | n={n:,}  "
              f"prevalence={prevalence:.3f}",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        plot_bgcolor="white", paper_bgcolor="white",
        height=480, width=860,
        legend=dict(bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ccc", borderwidth=1),
    )

    # ── 2. Few-shot learning curves ───────────────────────────────────────
    fig_fs = go.Figure()
    for bb in backbones:
        npz_path = run_dir / bb / "detection_fewshot.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        ks      = d["k"].tolist()
        ap_mean = d["ap_mean"].tolist()
        ap_std  = d["ap_std"].tolist()
        fig_fs.add_trace(go.Scatter(
            x=ks, y=ap_mean, mode="lines+markers",
            name=bb,
            line=dict(color=_colour(bb), width=2),
            error_y=dict(type="data", array=ap_std, visible=True)))

    fig_fs.update_layout(
        title="Few-shot detection AP vs labelled calls per class",
        xaxis_title="k (calls in train set)",
        yaxis_title="Average Precision (mean ± std)",
        plot_bgcolor="white", paper_bgcolor="white",
        height=420, width=720,
    )

    # ── 3. Summary table ──────────────────────────────────────────────────
    summaries = meta.get("summaries", [])
    tbl_bbs   = [s["backbone"]     for s in summaries]
    tbl_ap    = [f'{s["full_ap"]:.4f}'      for s in summaries]
    tbl_roc   = [f'{s["full_roc_auc"]:.4f}' for s in summaries]
    fig_tbl = go.Figure(go.Table(
        header=dict(values=["Backbone", "Full-train AP ↑", "ROC-AUC ↑"],
                    fill_color="#f0f0f0", align="left",
                    font=dict(size=13)),
        cells=dict(values=[tbl_bbs, tbl_ap, tbl_roc],
                   align="left", font=dict(size=12))))
    fig_tbl.update_layout(
        title="Summary: frozen backbone probes",
        margin=dict(t=50, b=20), height=max(200, 60 + 40 * len(tbl_bbs)))

    # ── 4. Combine into one HTML ──────────────────────────────────────────
    html_parts = [
        '<html><head><meta charset="utf-8"><title>Benchmark Report</title></head>',
        '<body style="font-family:sans-serif;max-width:1000px;margin:0 auto;padding:20px">',
        f'<h2>Benchmark Report — {meta.get("timestamp", "")}</h2>',
        f'<p style="color:#555">eval_dir: <code>{meta.get("eval_dir","")}</code><br>',
        f'n={n:,} &nbsp; prevalence={prevalence:.4f} &nbsp; '
        f'train={meta.get("train_n",0):,} &nbsp; test={meta.get("test_n",0):,}<br>',
        f'device={meta.get("device","")} &nbsp; seed={meta.get("seed",0)}</p>',
        fig_tbl.to_html(full_html=False, include_plotlyjs="cdn"),
        "<hr>",
        fig_pr.to_html(full_html=False, include_plotlyjs=False),
        "<hr>",
        fig_fs.to_html(full_html=False, include_plotlyjs=False),
        "</body></html>",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(html_parts))


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Rebuild HTML report from a completed benchmark run")
    p.add_argument("run_dir", type=Path, help="Directory containing meta.json")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    out = args.out or (run_dir / "report.html")
    build_report(meta, run_dir, out)
    print(f"Written {out}")
