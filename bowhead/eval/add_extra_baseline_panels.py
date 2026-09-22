"""Add two NEW, separate plot panels to docs/benchmark_pr_curves.html:
one for the classical multi-band "whale moan detector" and one for the
BirdNET-from-gram (Griffin-Lim reconstructed audio) baseline.

These are deliberately NOT merged into the existing "Frozen-Embedding
Benchmark" figure (different dataset: data/spectrograms_100k_matched.npz vs.
the 200K raw .mat eval set; different evaluation semantics for BirdNET, which
needs the audio-reconstruction caveat spelled out). Each gets its own
PR / FDR-vs-miss dual-panel figure plus its own summary card.

Usage:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.eval.add_extra_baseline_panels
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_JSON = _REPO_ROOT / "runs" / "extra_baselines_100k_matched.json"
HTML_PATH = _REPO_ROOT / "docs" / "benchmark_pr_curves.html"

DATASET_NAME = "data/spectrograms_100k_matched.npz"


def _pr_fdr_figure(name: str, color: str, prec, rec, prev: float, n: int, title: str) -> str:
    prec = np.asarray(prec)
    rec = np.asarray(rec)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12, subplot_titles=["", ""])

    label = f"{name}"
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=label,
                              line=dict(color=color, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=1 - rec, y=1 - prec, mode="lines", name=label,
                              showlegend=False, line=dict(color=color, width=2)),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=[0, 1], y=[prev, prev], mode="lines",
                              name="Random baseline",
                              line=dict(color="#aaa", dash="dash", width=1.5)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[1 - prev, 1 - prev], mode="lines",
                              name="Random baseline", showlegend=False,
                              line=dict(color="#aaa", dash="dash", width=1.5)),
                  row=1, col=2)

    fig.update_layout(
        title=dict(text=f"{title} | n={n:,}  prevalence={prev:.4f}", font=dict(size=14)),
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
        height=440, width=1000,
        margin=dict(t=60),
        annotations=[
            dict(text="Precision-Recall", x=0.22, xref="paper", y=1.0, yref="paper",
                 xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=14)),
            dict(text="FDR vs Miss Fraction", x=0.78, xref="paper", y=1.0, yref="paper",
                 xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=14)),
        ],
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _summary_card(name: str, color: str, roc_auc: float, ap: float, p_at_r70: float,
                   n: int, prev: float, caveat_html: str) -> str:
    return (
        '<div style="font-family:sans-serif;font-size:13px;color:#444;'
        'max-width:1000px;margin:6px auto 4px auto;padding:12px 18px;'
        'background:#f9f9f9;border:1px solid #ddd;border-radius:4px;line-height:1.6">'
        f'<span style="color:{color};font-size:16px">&#9644;</span>&nbsp;'
        f'<strong>{name}</strong> &mdash; evaluation dataset: '
        f'<code style="font-size:11px;background:#efefef;padding:1px 5px;border-radius:3px">'
        f'{DATASET_NAME}</code> (n={n:,}, prevalence {prev:.4f}, "date_site" grouped '
        f'held-out test split, seed=0 — SAME split used for the scratch/warm-start CNN retrain).<br>'
        f'AP=<b>{ap:.4f}</b> &nbsp; ROC-AUC=<b>{roc_auc:.4f}</b> &nbsp; '
        f'P@R0.70=<b>{p_at_r70:.4f}</b><br>'
        f'{caveat_html}'
        '</div>'
    )


def _precision_at_recall(m: dict, target_recall: float = 0.70) -> float:
    if "precision_at_recall" in m:
        return m["precision_at_recall"]
    rec = np.asarray(m["pr_recall"])
    prec = np.asarray(m["pr_precision"])
    mask = rec >= target_recall
    return float(prec[mask].max()) if mask.any() else float("nan")


def main() -> None:
    results = json.loads(RESULTS_JSON.read_text())

    sections = []

    # ── Panel 1: classical moan detector ───────────────────────────────────
    m = results["moan_detector"]
    fig_html = _pr_fdr_figure(
        "Moan Detector", "#d62728", m["pr_precision"], m["pr_recall"],
        m["prevalence"], m["n"],
        "Classical Whale Moan Detector — Detection PR Curves",
    )
    card_html = _summary_card(
        "Moan Detector (multi-band SNR energy detector)", "#d62728",
        m["roc_auc"], m["average_precision"], _precision_at_recall(m),
        m["n"], m["prevalence"],
        'Rule-based, no training: a per-image adaptation of the '
        '<a href="https://www.researchgate.net/" target="_blank" rel="noopener">'
        'Baumgartner &amp; Mussoline (2011)</a>-style generalized multi-band energy '
        'detector (ported from the repo\'s production '
        '<code style="font-size:11px;background:#efefef;padding:1px 5px;border-radius:3px">'
        'matlab/matlab/MultipleBandEnergyDetector.m</code>). Sums SNR-gram energy across '
        'overlapping 37 Hz sub-bands spanning 25&ndash;350 Hz, with a sustained-duration '
        'gate; no learned parameters, so it serves as the classical signal-processing '
        'baseline against which the learned models are compared.',
    )
    sections.append(f'<h3 style="font-family:sans-serif;margin:22px auto 0 auto;'
                     f'max-width:1000px">Classical Baseline — Whale Moan Detector</h3>'
                     + fig_html + card_html)

    # ── Panel 2: BirdNET-from-gram ──────────────────────────────────────────
    b = results["birdnet"]
    fig_html2 = _pr_fdr_figure(
        "BirdNET (from reconstructed audio)", "#17becf", b["pr_precision"], b["pr_recall"],
        b["prevalence"], b["n"],
        "BirdNET / Perch 2.0 (Griffin-Lim Reconstructed Audio) — Detection PR Curves",
    )
    meta = b.get("_meta", {})
    card_html2 = _summary_card(
        "BirdNET / Perch 2.0 (frozen linear probe on reconstructed-audio embeddings)",
        "#17becf",
        b["roc_auc"], b["average_precision"], _precision_at_recall(b),
        b["n"], b["prevalence"],
        '<b>Important caveat:</b> the real pretrained BirdNET/Perch 2.0 model '
        '(<a href="https://tfhub.dev/google/bird-vocalization-classifier/4" target="_blank" '
        'rel="noopener">tfhub.dev/google/bird-vocalization-classifier/4</a>) accepts '
        '<b>only raw 5s/32kHz audio</b> &mdash; it has no spectrogram-input path. Since this '
        'repo stores pre-computed dB-SNR spectrogram crops (not raw audio), a pseudo-waveform '
        'is reconstructed from each SNR-gram via inverse-STFT (Griffin-Lim, 32 iterations), '
        'resampled to 32kHz, and fed through the real model to get a 1280-D embedding. '
        'Phase is <i>synthesized</i> by Griffin-Lim, not the true recorded phase, so this '
        'measures how well BirdNET\'s learned features transfer to bowhead calls given only '
        'magnitude information &mdash; not a plug-and-play spectrogram benchmark. A frozen '
        f'linear probe (logistic regression) was fit on {meta.get("n_train_probe", "?"):,} '
        f'train embeddings and evaluated on {meta.get("n_test_subsample", "?"):,} held-out '
        'test embeddings (subsampled from the full split for tractability; Griffin-Lim '
        'reconstruction is CPU-bound).',
    )
    sections.append(f'<h3 style="font-family:sans-serif;margin:22px auto 0 auto;'
                     f'max-width:1000px">Transfer-Learning Baseline — BirdNET / Perch 2.0 '
                     f'(from Reconstructed Audio)</h3>' + fig_html2 + card_html2)

    html = HTML_PATH.read_text()
    insertion = "\n".join(sections)
    html = html.replace("</body>", insertion + "\n</body>")
    HTML_PATH.write_text(html)
    print(f"Inserted 2 new baseline panels into {HTML_PATH}")


if __name__ == "__main__":
    main()
