"""Run image-backbone detection probes on the evaluation dataset and
produce docs/benchmark_pr_curves.html comparing all backbones head-to-head.

Loads the 200K mixed evaluation .dir directly (no pre-built npz needed),
does a grouped date×site split, extracts frozen embeddings, trains
full-train detection probes, and plots PR + FDR-vs-miss curves for:
  - Scratch CNN  (existing, from runs/pr_curves_full_dataset.npz)
  - Warm-start CNN  (existing)
  - ResNet-18  (ImageNet, 512-D, local)
  - EfficientNet-B0  (ImageNet, 1280-D, local)
  - AST-style ViT  (ImageNet-21k, 768-D, local — requires timm)

By default, labels are derived from each file's \`_Type{N}\` suffix in its
filename (0 = non-call, else call) -- this matches the ORIGINAL (pre-review)
label convention. Pass \`--eval-mat\` (a MATLAB eval export with
\`original_filenames\`/\`features.iscall\`/\`features.type_org\`, same format as
\`bowhead.eval.run_scratch_relabel_curves\`) to instead score against the
manually reviewed (post-review) labels via \`--label-set reviewed\` (default
when \`--eval-mat\` is given), so these backbones can be compared on the same
label set as the other CNN-family detectors.

Usage:
    PYTHONPATH=. python -m bowhead.benchmark.run_image_pr_curves \\
        --eval-dir /path/to/Evaluation_200K.dir \\
        --eval-mat /path/to/latent_embeddings_3d_eval_8to1_MATLAB_combined_102126.mat \\
        --out-html docs/benchmark_pr_curves.html
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bowhead.config import best_device
from bowhead.data.dataset import per_sample_minmax
from bowhead.data.splits import grouped_split
from bowhead.eval.metrics import score_predictions
from bowhead.eval.relabel_panel_common import downsample_curve
from bowhead.benchmark.probes.detection import full_train_detection

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FNAME_RE = re.compile(
    r"S(?P<site>\d)(?P<yy>\d{2})(?P<dasar>[A-G])\dT"
    r"(?P<date>\d{8})T(?P<hms>\d{6})_Type(?P<type>\d+)"
)
GRAM        = "SNR_gram"
LOAD_BATCH  = 5_000     # images loaded into RAM at once during embedding

# ── palette (mirrors detection_curves.html) ─────────────────────────────────
COLORS = {
    "scratch_cnn":     "#4e9af1",   # blue
    "warmstart_cnn":   "#f07040",   # orange
    "resnet18":        "#2ca02c",   # green
    "efficientnet_b0": "#9467bd",   # purple
    "ast_imagenet":    "#8c564b",   # brown
}


# ── data loading ─────────────────────────────────────────────────────────────

def _scan_dir(eval_dir: Path):
    paths, labels, dates, sites = [], [], [], []
    skipped = 0
    for fp in sorted(eval_dir.glob("*.mat")):
        m = _FNAME_RE.match(fp.stem)
        if m is None:
            skipped += 1
            continue
        paths.append(fp)
        labels.append(0 if int(m.group("type")) == 0 else 1)
        dates.append(m.group("date"))
        sites.append(m.group("site"))
    if skipped:
        print(f"  Skipped {skipped} files (no pattern match)")
    return paths, np.array(labels, np.int64), np.array(dates), np.array(sites)


def _load_mat_labels(eval_mat: Path) -> dict[str, tuple[int, int]]:
    """Map filename -> (reviewed_label, original_label) from a MATLAB eval
    export (same \`original_filenames\`/\`features.iscall\`/\`features.type_org\`
    fields used by \`bowhead.eval.run_scratch_relabel_curves._load_eval_labels\`).
    """
    from scipy.io import loadmat as _loadmat
    d = _loadmat(str(eval_mat), squeeze_me=True, struct_as_record=False)
    feat = d["features"]
    type_org = np.asarray(feat.type_org).astype(int)
    reviewed = np.asarray(feat.iscall).astype(int)
    original = (type_org > 0).astype(int)
    filenames = np.asarray(d["original_filenames"])
    return {str(f): (int(r), int(o)) for f, r, o in zip(filenames, reviewed, original)}


def _apply_mat_labels(paths: list[Path], labels: np.ndarray, eval_mat: Path,
                       label_set: str) -> np.ndarray:
    """Override filename-derived labels with the MATLAB export's label set,
    for whichever files are present in the export (falls back to the
    filename-derived label for any file not found in the export)."""
    label_map = _load_mat_labels(eval_mat)
    idx = 0 if label_set == "reviewed" else 1
    out = labels.copy()
    n_matched = 0
    for i, p in enumerate(paths):
        pair = label_map.get(p.name)
        if pair is not None:
            out[i] = pair[idx]
            n_matched += 1
    print(f"  Matched {n_matched:,}/{len(paths):,} files to eval-mat "
          f"'{label_set}' labels ({len(paths) - n_matched:,} kept filename-derived fallback)")
    return out


def _load_images(paths: list[Path], indices: np.ndarray, tag: str = "") -> np.ndarray:
    """Load SNR_gram images for the given index subset, with progress logging
    (loading is a sequential per-file disk read, which can be slow on an
    external drive, so we print rate/ETA rather than loading silently)."""
    imgs = []
    t0 = time.time()
    n = len(indices)
    for j, i in enumerate(indices):
        try:
            arr = loadmat(str(paths[i]))[GRAM].astype(np.float32)
        except Exception:
            arr = np.zeros((121, 104), dtype=np.float32)
        imgs.append(arr)
        if (j + 1) % 500 == 0 or j + 1 == n:
            rate = (j + 1) / max(time.time() - t0, 1e-6)
            eta = (n - j - 1) / max(rate, 1e-6) / 60
            print(f"    [{tag}] {j + 1:,}/{n:,}  ({rate:.0f}/s  ETA {eta:.1f} min)")
    return np.stack(imgs)   # (N, H, W)


# ── embedding extraction ──────────────────────────────────────────────────────

def _extract(backbone, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Stream images through a backbone in batches; return (N, D) embeddings."""
    all_embs = []
    t0 = time.time()
    n = len(images)
    for start in range(0, n, batch_size):
        all_embs.append(backbone.embed(images[start:start + batch_size]))
        if (start // batch_size + 1) % 20 == 0:
            rate = (start + batch_size) / (time.time() - t0)
            eta  = (n - start - batch_size) / max(rate, 1) / 60
            print(f"    {min(start+batch_size, n):>7,}/{n:,}  "
                  f"({rate:.0f}/s  ETA {eta:.1f} min)")
    return np.concatenate(all_embs, axis=0)


# ── HTML generation ───────────────────────────────────────────────────────────

def _build_html(curves: dict, n: int, prev: float, out_path: Path) -> None:
    """curves: {name: DetectionMetrics}"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError("pip install plotly") from e

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=["", ""])

    for name, m in curves.items():
        col = COLORS.get(name, "#555555")
        label = f"{name.replace('_', ' ').title()}  (AP={m.average_precision:.3f})"
        prec, rec = downsample_curve(np.asarray(m.pr_precision), np.asarray(m.pr_recall))
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                 name=label,
                                 line=dict(color=col, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=1-rec, y=1-prec, mode="lines",
                                 name=label, showlegend=False,
                                 line=dict(color=col, width=2)), row=1, col=2)

    # random baseline
    fig.add_trace(go.Scatter(x=[0,1], y=[prev,prev], mode="lines",
                             name="Random baseline",
                             line=dict(color="#aaa", dash="dash", width=1.5)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[0,1], y=[1-prev,1-prev], mode="lines",
                             name="Random baseline", showlegend=False,
                             line=dict(color="#aaa", dash="dash", width=1.5)),
                  row=1, col=2)

    fig.update_layout(
        title=dict(
            text=(f"Frozen-Embedding Benchmark — Detection PR Curves"
                  f" | n={n:,}  prevalence={prev:.4f}"),
            font=dict(size=14)),
        xaxis =dict(title="Recall",                           range=[0,1],
                   showgrid=True, gridcolor="#e5e7eb", gridwidth=1, griddash="solid"),
        yaxis =dict(title="Precision",                        range=[0,1.02],
                   showgrid=True, gridcolor="#e5e7eb", gridwidth=1, griddash="solid"),
        xaxis2=dict(title="Miss Fraction (1−Recall)",         range=[0,1],
                   showgrid=True, gridcolor="#e5e7eb", gridwidth=1, griddash="solid"),
        yaxis2=dict(title="False Discovery Rate (1−Precision)", range=[0,1.02],
                   showgrid=True, gridcolor="#e5e7eb", gridwidth=1, griddash="solid"),
        legend=dict(x=0.01, y=0.15, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#ccc", borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white",
        height=520, width=1150,
        annotations=[
            dict(text="Precision-Recall", x=0.22, xref="paper",
                 y=1.0, yref="paper", xanchor="center", yanchor="bottom",
                 showarrow=False, font=dict(size=15)),
            dict(text="FDR vs Miss Fraction", x=0.78, xref="paper",
                 y=1.0, yref="paper", xanchor="center", yanchor="bottom",
                 showarrow=False, font=dict(size=15)),
        ],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_html = fig.to_html(full_html=False, include_plotlyjs=False)

    # summary table panel
    model_refs = {
        "scratch_cnn": "This study; CNN trained from scratch on the bowhead spectrogram dataset. Architecture: ConvEncoder-style CNN with 3 Conv-BN-ReLU-MaxPool blocks (32→64→128 channels), FC-64→FC-32 latent bottleneck, then linear 32→2 classifier head.",
        "warmstart_cnn": "This study; same CNN architecture warm-started from the AE encoder trunk. Architecture: ConvEncoder-style CNN with 3 Conv-BN-ReLU-MaxPool blocks (32→64→128 channels), FC-64→FC-32 latent bottleneck, then linear 32→2 classifier head.",
        "resnet18": "He et al. (2015). Deep Residual Learning for Image Recognition.",
        "efficientnet_b0": "Tan & Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.",
        "ast_imagenet": "Gong et al. (2021). AST: Audio Spectrogram Transformer.",
    }

    rows_html = ""
    for name, m in sorted(curves.items(), key=lambda kv: -kv[1].average_precision):
        col = COLORS.get(name, "#555")
        display_name = name.replace('_', ' ').title()
        ref = model_refs.get(name, "This study / local benchmark implementation.")
        rows_html += (
            f"<tr>"
            f'<td style="padding:4px 12px;border:1px solid #ddd">'
            f'<span style="color:{col};font-size:16px">&#9644;</span>&nbsp;'
            f"<strong>{display_name}</strong></td>"
            f'<td style="padding:4px 12px;border:1px solid #ddd;text-align:right">'
            f"{m.average_precision:.4f}</td>"
            f'<td style="padding:4px 12px;border:1px solid #ddd;text-align:right">'
            f"{m.roc_auc:.4f}</td>"
            f'<td style="padding:4px 12px;border:1px solid #ddd;text-align:right">'
            f"{m.precision_at_recall(0.70):.4f}</td>"
            f'<td style="padding:4px 12px;border:1px solid #ddd;max-width:360px;line-height:1.4">'
            f"{ref}</td>"
            f"</tr>\n"
        )

    def _th(t):
        return (f'<th style="padding:4px 12px;background:#eef;'
                f'border:1px solid #ccc">{t}</th>')

    metric_defs = (
        '<div style="margin-top:14px;padding-top:10px;border-top:1px solid #ddd">'
        '<strong>Metric definitions</strong><br>'
        '<ul style="margin:6px 0 0 18px;padding:0">'
        '<li><b>Recall</b> = TP / (TP + FN): fraction of true calls recovered.</li>'
        '<li><b>Miss fraction</b> = 1 − recall: fraction of true calls missed.</li>'
        '<li><b>Precision</b> = TP / (TP + FP): fraction of flagged detections that are true calls.</li>'
        '<li><b>False discovery rate (FDR)</b> = 1 − precision: fraction of detections that are false alarms.</li>'
        '<li><b>Average precision (AP)</b>: area under the precision–recall curve, summarizing ranking quality over all thresholds.</li>'
        '<li><b>ROC-AUC</b>: area under the receiver operating characteristic curve, summarizing TPR vs FPR tradeoff.</li>'
        '<li><b>Precision at recall 0.70 (P@R0.70)</b>: precision achieved when recall is constrained to be at least 0.70.</li>'
        '</ul>'
        '<br><strong>References</strong><br>'
        '<ul style="margin:6px 0 0 18px;padding:0">'
        '<li>Davis, J. and Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. ICML.</li>'
        '<li>Saito, T. and Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative Than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE.</li>'
        '<li>Fawcett, T. (2006). An Introduction to ROC Analysis. Pattern Recognition Letters.</li>'
        '</ul>'
        '</div>'
    )

    dataset_name = "Unsupervised_database_Evaluation_200K_8Auto1Manual_ADG_Y08101214_centered_07Aug2026.dir.zip"

    panel = (
        '<div style="font-family:sans-serif;font-size:13px;color:#444;'
        'max-width:1150px;margin:10px auto 24px auto;padding:14px 18px;'
        'background:#f9f9f9;border:1px solid #ddd;border-radius:4px;line-height:1.6">'
        "<strong>Benchmark results summary</strong> &mdash; "
        f"evaluation dataset: <code style='font-size:11px;background:#efefef;padding:1px 5px;border-radius:3px'>{dataset_name}</code> "
        f"({n:,} spectrograms, prevalence {prev:.4f}; 1 call per {1/prev-1:.1f} non-calls).<br>"
        "Detection probe: full-train logistic regression on frozen embeddings, "
        "grouped date×site train/test split (15% held-out test). "
        "Metrics at realistic prevalence (1:8 resampling).<br><br>"
        f'<table style="border-collapse:collapse;font-size:12px">'
        "<thead><tr>"
        + _th("Model") + _th("AP ↑") + _th("ROC-AUC ↑") + _th("P@R0.70 ↑") + _th("Reference(s)")
        + "</tr></thead><tbody>\n"
        + rows_html
        + "</tbody></table>"
        + metric_defs
        + "</div>"
    )

    # Inserted non-destructively, keyed off a stable marker, so re-running this
    # script never clobbers OTHER panels appended into the same dashboard file
    # (relabel ablations, CNN-family comparison, etc.) -- see
    # docs/benchmark_pr_curves.html gotchas in repo memory.
    marker = '<div id="frozen-embedding-benchmark-marker" style="display:none"></div>'
    section = marker + fig_html + panel

    if out_path.exists():
        html = out_path.read_text()
        marker_idx = html.find(marker)
        if marker_idx != -1:
            html = html[:marker_idx]
        else:
            body_idx = html.find("</body>")
            html = html[:body_idx] if body_idx != -1 else html
    else:
        import plotly.io as pio
        plotlyjs_bundle = f"<script>{pio.get_plotlyjs()}</script>"
        html = f"<html><head><meta charset='utf-8'>{plotlyjs_bundle}</head><body>\n"

    html = html.rstrip() + "\n" + section + "\n</body>\n</html>\n"
    out_path.write_text(html)
    print(f"Written {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(
    eval_dir: Path,
    out_html: Path,
    cnn_npz: Path | None = None,
    backbones: list[str] | None = None,
    device: str | None = None,
    eval_mat: Path | None = None,
    label_set: str = "reviewed",
    max_train: int | None = None,
    max_test: int | None = None,
    out_json: Path | None = None,
) -> None:
    device = device or best_device()
    backbones = backbones or ["resnet18", "efficientnet_b0"]
    print(f"\nDevice: {device}")

    # 1. Scan directory
    print(f"\nScanning {eval_dir.name} ...")
    paths, labels, dates, sites = _scan_dir(eval_dir)
    if eval_mat is not None:
        print(f"\nOverriding labels from {eval_mat.name} (label_set={label_set}) ...")
        labels = _apply_mat_labels(paths, labels, eval_mat, label_set)
    n = len(paths)
    prev = float(labels.mean())
    print(f"  {n:,} files | {int(labels.sum()):,} calls | "
          f"{int((labels==0).sum()):,} non-calls | prevalence={prev:.4f}")

    # 2. Grouped train/test split (date×site)
    groups = np.array([f"{d}_{s}" for d, s in zip(dates, sites)])
    split = grouped_split(labels, groups, val_frac=0.10, test_frac=0.15, seed=0)
    print(split.summary(labels, groups))

    train_idx, test_idx = split.train, split.test
    rng = np.random.default_rng(0)
    if max_train is not None and len(train_idx) > max_train:
        train_idx = rng.choice(train_idx, size=max_train, replace=False)
        print(f"  Subsampled train split to {max_train:,} (from {len(split.train):,})")
    if max_test is not None and len(test_idx) > max_test:
        test_idx = rng.choice(test_idx, size=max_test, replace=False)
        print(f"  Subsampled test split to {max_test:,} (from {len(split.test):,})")

    # 3. Load images for train + test subsets
    print("\nLoading train images ...")
    train_imgs = _load_images(paths, train_idx, tag="train")
    print(f"  train: {train_imgs.shape}")
    print("Loading test images ...")
    test_imgs  = _load_images(paths, test_idx, tag="test")
    print(f"  test:  {test_imgs.shape}")

    train_labels = labels[train_idx]
    test_labels  = labels[test_idx]

    curves: dict = {}

    # 4. Optional: load existing CNN curves
    if cnn_npz and cnn_npz.exists():
        print("\nLoading existing CNN PR curves ...")
        d = np.load(cnn_npz)
        for tag in ("scratch", "warmstart"):
            key = f"{tag}_cnn"
            from bowhead.eval.metrics import DetectionMetrics
            prec = d[f"{tag}_precision"]
            rec  = d[f"{tag}_recall"]
            ap   = float(d[f"{tag}_ap"])
            roc  = float(d.get(f"{tag}_roc_auc", 0.0))
            # Reconstruct minimal DetectionMetrics for display
            curves[key] = DetectionMetrics(
                roc_auc=roc, average_precision=ap,
                prevalence=float(d["prevalence"]), n=int(d["n"]),
                pr_precision=prec, pr_recall=rec,
                pr_thresholds=np.linspace(1, 0, len(prec)-1),
                roc_fpr=np.array([0, 1]),
                roc_tpr=np.array([0, 1]),
                roc_thresholds=np.array([1, 0]),
            )
            print(f"  {key}: AP={ap:.4f}  ROC-AUC={roc:.4f}")

    # 5. Image backbone probes
    from bowhead.benchmark.backbones.image import load_image_backbone
    for bb_name in backbones:
        print(f"\n[{bb_name}] Loading backbone ...")
        try:
            bb = load_image_backbone(bb_name, device=device)
        except ImportError as e:
            print(f"  SKIP {bb_name}: {e}")
            continue

        print(f"  Extracting train embeddings ({len(train_imgs):,} images) ...")
        emb_train = _extract(bb, train_imgs)
        print(f"  Extracting test embeddings ({len(test_imgs):,} images) ...")
        emb_test  = _extract(bb, test_imgs)

        print(f"  Fitting full-train detection probe ...")
        m = full_train_detection(
            emb_train, train_labels, emb_test, test_labels,
            backbone_name=bb_name,
            target_prevalence=1.0 / 9.0,
            seed=0,
        )
        curves[bb_name] = m
        print(f"  AP={m.average_precision:.4f}  ROC-AUC={m.roc_auc:.4f}")

    # 6. Build HTML
    print("\nBuilding HTML ...")
    _build_html(curves, n, prev, out_html)

    # 7. Optional: save per-model metrics/curves as JSON for downstream plotting
    if out_json is not None:
        import json
        payload = {
            name: dict(
                average_precision=m.average_precision, roc_auc=m.roc_auc,
                prevalence=m.prevalence, n=m.n,
                precision_at_recall_0_70=m.precision_at_recall(0.70),
                pr_precision=np.asarray(m.pr_precision).tolist(),
                pr_recall=np.asarray(m.pr_recall).tolist(),
            )
            for name, m in curves.items()
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"Written {out_json}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", type=Path, required=True)
    p.add_argument("--out-html", type=Path,
                   default=_REPO_ROOT / "docs" / "benchmark_pr_curves.html")
    p.add_argument("--cnn-npz", type=Path,
                   default=_REPO_ROOT / "runs" / "pr_curves_full_dataset.npz")
    p.add_argument("--backbones", nargs="+",
                   default=["resnet18", "efficientnet_b0"],
                   help="Image backbones to run (add 'ast_imagenet' if timm installed)")
    p.add_argument("--device", default=None)
    p.add_argument("--eval-mat", type=Path, default=None,
                   help="MATLAB eval export (original_filenames/features.iscall/"
                        "features.type_org) to override filename-derived labels")
    p.add_argument("--label-set", choices=["reviewed", "original"], default="reviewed",
                   help="Which label set to use from --eval-mat (default: reviewed)")
    p.add_argument("--max-train", type=int, default=None,
                   help="Cap the number of train-split images loaded (random subsample)")
    p.add_argument("--max-test", type=int, default=None,
                   help="Cap the number of test-split images loaded (random subsample)")
    p.add_argument("--out-json", type=Path, default=None,
                   help="Optional path to save per-model metrics/curves as JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        eval_dir=args.eval_dir,
        out_html=args.out_html,
        cnn_npz=args.cnn_npz,
        backbones=args.backbones,
        device=args.device,
        eval_mat=args.eval_mat,
        label_set=args.label_set,
        max_train=args.max_train,
        max_test=args.max_test,
        out_json=args.out_json,
    )
