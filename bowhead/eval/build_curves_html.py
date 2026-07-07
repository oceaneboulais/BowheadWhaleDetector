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
# First N entries are used in order; add more if needed.
_PALETTE = [
    "#4e9af1",   # blue        scratch 1-ch
    "#f07040",   # orange      warmstart 1-ch
    "#2ca02c",   # green       scratch 2-ch
    "#9467bd",   # purple      warmstart 2-ch
    "#8c564b",   # brown
    "#e377c2",   # pink
    "#17becf",   # cyan
    "#bcbd22",   # olive
]
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
    d    = np.load(npz_path, allow_pickle=True)
    n    = int(d["n"])
    prev = float(d["prevalence"])

    # ── Detect whether this is the new multi-model format or the old 2-model one
    if "model_names" in d:
        model_names = list(d["model_names"])
    else:
        # Legacy format: scratch / warmstart keys
        model_names = []
        for prefix in ("scratch", "warmstart"):
            if f"{prefix}_precision" in d:
                model_names.append(prefix)

    # Build per-model curve data: {name: (precision, recall, ap)}
    curves: dict[str, tuple] = {}
    for name in model_names:
        key = name.replace(" ", "_")
        prec = d[f"{key}_precision"]
        rec  = d[f"{key}_recall"]
        ap   = float(d[f"{key}_ap"])
        curves[name] = (prec, rec, ap)

    # ── Friendly display labels
    _LABELS = {
        "scratch":        "Scratch CNN (1-ch)",
        "warmstart":      "Warm-start CNN (1-ch)",
        "scratch_1ch":    "Scratch CNN (1-ch)",
        "warmstart_1ch":  "Warm-start CNN (1-ch)",
        "scratch_2ch":    "Scratch CNN (2-ch SNR+NTV)",
        "warmstart_2ch":  "Warm-start CNN (2-ch SNR+NTV)",
    }

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["", ""],
        horizontal_spacing=0.12,
    )

    for i, (name, (prec, rec, ap)) in enumerate(curves.items()):
        color = _PALETTE[i % len(_PALETTE)]
        label = _LABELS.get(name, name)
        display = f"{label} (AP={ap:.3f})"
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines", name=display,
            line=dict(color=color, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=1 - rec, y=1 - prec, mode="lines", name=display,
            line=dict(color=color, width=2), showlegend=False), row=1, col=2)

    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[prev, prev], mode="lines",
        name="Random baseline (prevalence)",
        line=dict(color=GREY, dash="dash", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[1 - prev, 1 - prev], mode="lines",
        name="Random baseline",
        line=dict(color=GREY, dash="dash", width=1.5), showlegend=False), row=1, col=2)

    _grid = dict(showgrid=True, gridcolor="#e0e0e0", gridwidth=1,
                 zeroline=True, zerolinecolor="#cccccc", zerolinewidth=1)
    fig.update_layout(
        title=dict(
            text=(f"CNN detector curves on FULL DATASET"
                  f" | n={n:,}  prevalence={prev:.3f}"),
            font=dict(size=15),
        ),
        xaxis =dict(title="Recall",                               range=[0, 1], **_grid),
        yaxis =dict(title="Precision",                            range=[0, 1.02], **_grid),
        xaxis2=dict(title="Miss Fraction (1 − Recall)",           range=[0, 1], **_grid),
        yaxis2=dict(title="False Discovery Rate (1 − Precision)", range=[0, 1.02], **_grid),
        legend=dict(
            x=0.01, y=0.01 + 0.06 * len(curves),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc", borderwidth=1,
        ),
        plot_bgcolor="#fafafa",
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

    thumbs: dict = {}
    if eval_dir is not None:
        try:
            thumbs = _make_thumbnails(eval_dir)
            print(f"  Generated {len(thumbs)} spectrogram thumbnails")
        except Exception as exc:
            print(f"  Note: could not generate thumbnails: {exc}")

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
            + _td(f"{count:,}" if isinstance(count, int) else count, "text-align:right")
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
        "<br>"
        "<strong>Training dataset</strong><br>"
        "<em style='font-size:12px;color:#555'>"
        "The CNN models were trained on two separate databases "
        "(distinct from this evaluation set):"
        "</em><br>"
        '<table style="border-collapse:collapse;font-size:12px;margin-top:6px">'
        "<tr>"
        + _th("Role") + _th("Directory") + _th("Count") + _th("Label")
        + "</tr>"
        "<tr>"
        + _td("Calls (positive)", "font-weight:600")
        + _td('<code style="font-size:11px;background:#efefef;padding:1px 4px;border-radius:3px">'
              "Unsupervised_database_Manual_100K_ADG_Y08101214_centered_16Apr2026.dir"
              "</code>")
        + _td("98,933", "text-align:right")
        + _td('<span style="color:#2a7;font-weight:600">1</span>')
        + "</tr>"
        "<tr style='background:#fafafa'>"
        + _td("Non-call transients (negative)", "font-weight:600")
        + _td('<code style="font-size:11px;background:#efefef;padding:1px 4px;border-radius:3px">'
              "Unsupervised_database_Auto_100K_ADG_Y08101214_centered_16Apr2026.dir"
              "</code>")
        + _td("100,723", "text-align:right")
        + _td('<span style="color:#c33;font-weight:600">0</span>')
        + "</tr>"
        "</table>"
        "<br><em style='font-size:11px;color:#888'>"
        "Training split: grouped by date &times; DASAR site to prevent leakage "
        "(train 115,402 | val 32,861 | test 51,393 &mdash; zero group overlap). "
        "Warm-start CNN encoder initialised from "
        "<code style='font-size:11px'>Autoencoder_v13_100E_32LD_32C_AutoManual_"
        "Combined_100K_Date20260416-180022</code> (LD32, 100 epochs)."
        "</em>"
        "</div>"
    )

    # ── How to read
    model_legend_html = ""
    for i, (name, (_, _, ap)) in enumerate(curves.items()):
        color = _PALETTE[i % len(_PALETTE)]
        label = _LABELS.get(name, name)
        model_legend_html += (
            f'<span style="color:{color}">&#9644; <strong>{label} (AP={ap:.3f})</strong></span>'
            " &mdash; "
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
        + model_legend_html
        + '<span style="color:#888">&#9135;&nbsp;&#9135; <strong>Random baseline</strong></span>'
        " &mdash; horizontal line at precision&nbsp;=&nbsp;prevalence "
        "(AP of a zero-skill classifier). Any useful detector must lie clearly "
        "above this line on the PR plot."
        "</div>"
    )

    panel = dataset_section + how_to_read

    html = out_path.read_text()
    html = html.replace("</body>", panel + "\n</body>")
    out_path.write_text(html)

    ap_summary = "  ".join(f"{n}:AP={ap:.4f}" for n, (_, _, ap) in curves.items())
    print(f"Written {out_path}")
    print(f"  {ap_summary}  n={n:,}  prevalence={prev:.4f}")


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
