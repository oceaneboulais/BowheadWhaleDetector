#!/usr/bin/env python3
"""
Visualise latent embeddings stored in a MATLAB .mat file.

Supports:
  • Pre-computed UMAP / tSNE / PaCMAP embeddings
  • Raw latent vectors (UMAP / tSNE / PCA reduction applied on the fly)
  • Interactive matplotlib GUI  (mirrors MATLAB Scatterplot_GUI.m)
  • Interactive Plotly HTML export (shareable, no server required)

USAGE EXAMPLES
--------------
# Auto-detect embeddings, launch interactive GUI
python visualize_latent.py path/to/umap_embeddings_3d_auto.mat

# Force a specific field and colour channel
python visualize_latent.py embeddings.mat --field latent_embeddings --reduce umap --dims 3

# Save Plotly HTML instead of GUI
python visualize_latent.py embeddings.mat --output viewer.html

# Both GUI and HTML
python visualize_latent.py embeddings.mat --output viewer.html --gui

# List all fields in the file (no plot)
python visualize_latent.py embeddings.mat --list-fields
"""

import argparse
import re
import sys
import os
import json
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Import project modules from this folder
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from loader import load_mat, detect_embeddings, extract_labels, extract_metadata, list_fields


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def _reduce(X: np.ndarray, method: str, n_components: int, verbose: bool = True) -> np.ndarray:
    method = method.lower()

    if method == "umap":
        try:
            import umap as umap_lib
        except ImportError:
            sys.exit("umap-learn is required:  pip install umap-learn")
        if verbose:
            print(f"  Running UMAP (n_components={n_components}, n={X.shape[0]})…")
        reducer = umap_lib.UMAP(n_components=n_components, n_neighbors=15,
                                min_dist=0.1, metric="euclidean",
                                random_state=42, verbose=verbose)
        return reducer.fit_transform(X)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        if verbose:
            print(f"  Running t-SNE (n_components={n_components}, n={X.shape[0]})…")
        return TSNE(n_components=n_components, perplexity=min(30, X.shape[0] - 1),
                    random_state=42, verbose=int(verbose)).fit_transform(X)

    elif method == "pca":
        from sklearn.decomposition import PCA
        if verbose:
            print(f"  Running PCA (n_components={n_components})…")
        return PCA(n_components=n_components, random_state=42).fit_transform(X)

    else:
        sys.exit(f"Unknown reduction method '{method}'. Choose: umap, tsne, pca")


# ---------------------------------------------------------------------------
# Overlap reduction ("declutter")
# ---------------------------------------------------------------------------

def _declutter(X: np.ndarray, min_sep_frac: float = 0.005,
               max_iters: int = 50, seed: int = 0) -> np.ndarray:
    """Nudge points apart so no two sit closer than ``min_sep_frac`` of the
    embedding's bounding-box diagonal (default 0.5%) — i.e. markers no longer
    overlap by more than that fraction of the plot's extent.

    Repeatedly finds close pairs with a KD-tree and pushes each pair apart
    along their connecting vector; cheap enough to redo every iteration for a
    few thousand points. Stops early once no violations remain. A tiny random
    jitter is mixed in so points stuck in a many-way tie (more than 2 points
    mutually within the threshold) don't just oscillate in place; very dense
    clusters may still have some residual violations after ``max_iters``
    passes — this is a best-effort spacing pass, not an exact packing solver.
    """
    from scipy.spatial import cKDTree

    n, d = X.shape
    if n < 2:
        return X

    rng = np.random.default_rng(seed)
    span = X.max(axis=0) - X.min(axis=0)
    diag = float(np.linalg.norm(span))
    if diag < 1e-12:
        return X
    min_dist = min_sep_frac * diag

    Y = X.astype(np.float64).copy()
    n_violations = 0
    for it in range(max_iters):
        pairs = cKDTree(Y).query_pairs(min_dist)
        n_violations = len(pairs)
        if not pairs:
            break
        shift = np.zeros_like(Y)
        for i, j in pairs:
            delta = Y[i] - Y[j]
            dist = np.linalg.norm(delta)
            if dist < 1e-9:
                delta = rng.normal(size=d)
                dist = np.linalg.norm(delta)
            push = 0.5 * (min_dist - dist) * (delta / dist)
            jitter = rng.normal(scale=0.05 * min_dist, size=d)
            shift[i] += push + jitter
            shift[j] += -push + jitter
        Y += shift

    print(f"  Decluttered {n:,} points (min separation {min_sep_frac:.2%} of extent); "
          f"{n_violations:,} overlapping pair(s) remain after {it + 1} pass(es)")
    return Y.astype(np.float32)


# ---------------------------------------------------------------------------
# Plotly HTML export
# ---------------------------------------------------------------------------

def _extract_year_month_site(data: dict, n: int) -> tuple:
    """Derive (years, months, sites) int arrays of length n from whatever
    metadata fields are present (``year``/``date``/``site`` — the fields
    written by ``bowhead/data/sample_spectrogram_dir.py``). Any field that's
    missing or the wrong length is returned as None so the filter bar in
    ``_save_html`` degrades gracefully (year/month come from the filename's
    embedded ``YYYYMMDD`` timestamp; site is parsed out of a ``"Site3"``-style
    string).
    """
    years = months = sites = None

    date_strs = None
    if "date" in data:
        date_strs = [str(d) for d in np.asarray(data["date"]).flat]

    if "year" in data:
        try:
            years = np.array([int(re.sub(r"\D", "", str(y))[:4]) for y in np.asarray(data["year"]).flat])
        except ValueError:
            years = None
    elif date_strs is not None:
        years = np.array([int(d[:4]) for d in date_strs])

    if date_strs is not None:
        months = np.array([int(d[4:6]) for d in date_strs])

    if "site" in data:
        raw = [re.sub(r"\D", "", str(s)) for s in np.asarray(data["site"]).flat]
        sites = np.array([int(r) if r else -1 for r in raw])

    if years is not None and len(years) != n:
        years = None
    if months is not None and len(months) != n:
        months = None
    if sites is not None and len(sites) != n:
        sites = None
    return years, months, sites


def _export_thumbnails(filenames: list, image_folder: str, out_dir: str,
                        cmap: str = "inferno") -> list:
    """Render one spectrogram PNG thumbnail per detection in ``filenames``.

    Writes ``<out_dir>/<i>.png`` for every index whose ``.mat`` file loads
    successfully; returns a list the same length as ``filenames`` holding the
    path (relative to ``out_dir``'s parent, i.e. what the HTML should use as
    an ``<img src>``) for each point that succeeded, or ``None`` for points
    whose spectrogram couldn't be loaded (missing file, unmounted drive,
    unreadable .mat, etc. — skipped rather than aborting the whole export).
    """
    from spectrogram_loader import save_detection_thumbnail
    from tqdm import tqdm

    os.makedirs(out_dir, exist_ok=True)
    dir_name = os.path.basename(os.path.normpath(out_dir))
    rel_paths: list = [None] * len(filenames)
    n_ok = 0
    for i, fname in enumerate(tqdm(filenames, desc="Rendering spectrogram thumbnails")):
        out_path = os.path.join(out_dir, f"{i}.png")
        try:
            save_detection_thumbnail(image_folder, str(fname), out_path, cmap=cmap)
            rel_paths[i] = f"{dir_name}/{i}.png"
            n_ok += 1
        except Exception as e:  # noqa: BLE001 - skip unreadable/missing files
            print(f"  WARN thumbnail skip [{i}]: {e}")
    print(f"  Rendered {n_ok:,} / {len(filenames):,} spectrogram thumbnails -> {out_dir}")
    return rel_paths


def _inject_spectrogram_viewer(html: str) -> str:
    """Add a click-to-view spectrogram modal to an exported Plotly HTML file.

    Reads the thumbnail path out of each point's last ``customdata`` entry
    (written by ``_save_html``) and shows it full-size in an overlay when
    the point is clicked. Points with no thumbnail (``null``) show a short
    "not available" message instead of a broken image. Client-side only, no
    server needed.
    """
    viewer = """
<div id="spectrogram-modal" style="display:none;position:fixed;inset:0;
     background:rgba(0,0,0,0.75);z-index:1000;align-items:center;justify-content:center">
  <div style="background:#1e1e2e;border-radius:6px;padding:14px;max-width:90vw;max-height:90vh;
       display:flex;flex-direction:column;align-items:center;gap:8px">
    <div style="display:flex;justify-content:space-between;width:100%;align-items:center">
      <span id="spectrogram-modal-title" style="font-family:sans-serif;font-size:13px;color:#ddd"></span>
      <button id="spectrogram-modal-close" style="background:#4a6ee0;color:#fff;border:none;
           border-radius:4px;padding:4px 10px;cursor:pointer;font-size:13px">Close</button>
    </div>
    <img id="spectrogram-modal-img" style="max-width:85vw;max-height:75vh;display:none" />
    <span id="spectrogram-modal-empty" style="font-family:sans-serif;font-size:13px;color:#888;display:none">
      No spectrogram available for this point (source file missing or unreadable when the
      viewer was generated).
    </span>
  </div>
</div>
<script>
(function() {
  function ready(fn) {
    if (document.readyState !== "loading") fn(); else document.addEventListener("DOMContentLoaded", fn);
  }
  ready(function() {
    var gd = document.getElementsByClassName("plotly-graph-div")[0];
    if (!gd) return;
    var modal = document.getElementById("spectrogram-modal");
    var img = document.getElementById("spectrogram-modal-img");
    var empty = document.getElementById("spectrogram-modal-empty");
    var titleEl = document.getElementById("spectrogram-modal-title");

    function closeModal() { modal.style.display = "none"; }
    document.getElementById("spectrogram-modal-close").addEventListener("click", closeModal);
    modal.addEventListener("click", function(ev) { if (ev.target === modal) closeModal(); });

    gd.on("plotly_click", function(evtData) {
      if (!evtData || !evtData.points || !evtData.points.length) return;
      var pt = evtData.points[0];
      var cd = pt.customdata;
      if (!cd) return;
      var thumb = cd[cd.length - 1];
      titleEl.textContent = "point idx=" + pt.pointIndex + (pt.data.name ? "  (" + pt.data.name + ")" : "");
      if (thumb) {
        img.src = thumb;
        img.style.display = "";
        empty.style.display = "none";
      } else {
        img.style.display = "none";
        empty.style.display = "";
      }
      modal.style.display = "flex";
    });
  });
})();
</script>
"""
    return html.replace("</body>", viewer + "</body>", 1)


def _inject_filter_bar(html: str, years: np.ndarray, months: np.ndarray, sites: np.ndarray) -> str:
    """Add Year / Month / DASAR-site dropdown filters plus an "Update Viewer"
    button to an exported Plotly HTML file. Client-side only (a small
    vanilla-JS snippet) — no server needed. Dropdown changes are staged and
    only take effect once the button is clicked; filters combine with AND
    logic by recomputing per-point marker opacity from each trace's
    ``customdata``.
    """
    def _options(values, label_fmt=str):
        uniq = sorted(int(v) for v in np.unique(values))
        opts = ['<option value="all">All</option>']
        opts += [f'<option value="{v}">{label_fmt(v)}</option>' for v in uniq]
        return "\n".join(opts)

    _MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    year_opts = _options(years) if years is not None else ""
    month_opts = _options(months, lambda v: _MONTH_NAMES[v]) if months is not None else ""
    site_opts = _options(sites) if sites is not None else ""

    bar = f"""
<div id="latent-filter-bar" style="font-family:sans-serif;font-size:13px;color:#ddd;
     background:#1e1e2e;padding:10px 16px;display:flex;gap:18px;align-items:center">
  <strong>Filter:</strong>
  <label>Year
    <select id="filter-year">{year_opts}</select>
  </label>
  <label>Month
    <select id="filter-month">{month_opts}</select>
  </label>
  <label>DASAR site
    <select id="filter-site">{site_opts}</select>
  </label>
  <button id="filter-apply-btn" style="background:#4a6ee0;color:#fff;border:none;
       border-radius:4px;padding:6px 14px;cursor:pointer;font-size:13px">Update Viewer</button>
  <span id="filter-count" style="color:#888"></span>
</div>
<script>
(function() {{
  function ready(fn) {{
    if (document.readyState !== "loading") fn(); else document.addEventListener("DOMContentLoaded", fn);
  }}
  ready(function() {{
    var gd = document.getElementsByClassName("plotly-graph-div")[0];
    if (!gd) return;
    function apply() {{
      var y = document.getElementById("filter-year").value;
      var m = document.getElementById("filter-month").value;
      var s = document.getElementById("filter-site").value;
      var shown = 0, total = 0;
      var opacities = gd.data.map(function(trace) {{
        if (!trace.customdata) return trace.marker.opacity;
        return trace.customdata.map(function(cd) {{
          total += 1;
          var ok = (y === "all" || String(cd[0]) === y) &&
                   (m === "all" || String(cd[1]) === m) &&
                   (s === "all" || String(cd[2]) === s);
          if (ok) shown += 1;
          return ok ? 0.6 : 0.02;
        }});
      }});
      Plotly.restyle(gd, {{"marker.opacity": opacities}});
      document.getElementById("filter-count").textContent =
        shown + " / " + total + " points shown";
    }}
    document.getElementById("filter-apply-btn").addEventListener("click", apply);
    apply();
  }});
}})();
</script>
"""
    return html.replace("<body>", "<body>" + bar, 1)


def _save_html(X: np.ndarray, labels: Optional[np.ndarray],
               color_fields: dict, output_path: str,
               title: str, filenames: Optional[list],
               years: Optional[np.ndarray] = None,
               months: Optional[np.ndarray] = None,
               sites: Optional[np.ndarray] = None,
               thumbnails: Optional[list] = None):
    try:
        import plotly.graph_objects as go
    except ImportError:
        sys.exit("plotly is required for HTML output:  pip install plotly")

    is_3d = X.shape[1] == 3
    n = X.shape[0]

    if labels is None:
        labels = np.zeros(n)

    unique_types = np.unique(labels[np.isfinite(labels)]).astype(int)
    have_filters = years is not None and months is not None and sites is not None
    have_thumbs = thumbnails is not None

    # Build hover text
    hover = []
    for i in range(n):
        parts = [f"idx={i}", f"type={labels[i]:.0f}"]
        for k, arr in color_fields.items():
            if k != "type / labels":
                parts.append(f"{k[:20]}={arr[i]:.4g}")
        if filenames is not None and i < len(filenames):
            parts.append(f"file={os.path.basename(str(filenames[i]))}")
        hover.append("<br>".join(parts))

    import plotly.express as px
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24

    traces = []
    for idx, t in enumerate(unique_types):
        mask = labels == t
        idxs = np.where(mask)[0]
        col = colors[idx % len(colors)]
        kwargs = dict(
            mode="markers",
            name=f"Type {t}",
            marker=dict(size=3, color=col, opacity=0.5),
            text=[hover[i] for i in idxs],
            hovertemplate="%{text}<extra></extra>",
        )
        if have_filters or have_thumbs:
            # Thumbnail path is always the LAST entry (see plotly_click handler
            # injected by _inject_spectrogram_viewer); year/month/site stay in
            # the first three slots so the existing filter bar keeps working.
            kwargs["customdata"] = [
                [
                    int(years[i]) if have_filters else -1,
                    int(months[i]) if have_filters else -1,
                    int(sites[i]) if have_filters else -1,
                    thumbnails[i] if have_thumbs else None,
                ]
                for i in idxs
            ]
        if is_3d:
            traces.append(go.Scatter3d(
                x=X[mask, 0], y=X[mask, 1], z=X[mask, 2], **kwargs))
        else:
            traces.append(go.Scatter(
                x=X[mask, 0], y=X[mask, 1], **kwargs))

    layout_kw = dict(
        title=title,
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    )

    if is_3d:
        fig = go.Figure(data=traces, layout=go.Layout(
            scene=dict(
                bgcolor="#1e1e2e",
                xaxis=dict(title="dim 1", backgroundcolor="rgba(0,0,0,0)",
                           showbackground=False, gridcolor="#333", zerolinecolor="#555"),
                yaxis=dict(title="dim 2", backgroundcolor="rgba(0,0,0,0)",
                           showbackground=False, gridcolor="#333", zerolinecolor="#555"),
                zaxis=dict(title="dim 3", backgroundcolor="rgba(0,0,0,0)",
                           showbackground=False, gridcolor="#333", zerolinecolor="#555"),
            ),
            **layout_kw,
        ))
    else:
        fig = go.Figure(data=traces, layout=go.Layout(
            xaxis=dict(title="dim 1", gridcolor="#333"),
            yaxis=dict(title="dim 2", gridcolor="#333"),
            **layout_kw,
        ))

    fig.write_html(output_path, include_plotlyjs="cdn")

    if have_filters or have_thumbs:
        html = open(output_path, encoding="utf-8").read()
        if have_filters:
            html = _inject_filter_bar(html, years, months, sites)
            print("  Added Year / Month / DASAR-site filter controls")
        if have_thumbs:
            html = _inject_spectrogram_viewer(html)
            n_with_thumb = sum(1 for t in thumbnails if t)
            print(f"  Added click-to-view spectrogram viewer ({n_with_thumb:,}/{n:,} points have a thumbnail)")
        open(output_path, "w", encoding="utf-8").write(html)

    print(f"  Saved interactive HTML → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("mat", help="Path to .mat file containing latent embeddings")
    p.add_argument("--field", default=None,
                   help="Force a specific mat field to use as embedding")
    p.add_argument("--dims", type=int, choices=[2, 3], default=3,
                   help="Embedding dimensionality to visualise (default: 3)")
    p.add_argument("--reduce", default=None, choices=["umap", "tsne", "pca"],
                   help="Reduction method when only raw latent vectors are available")
    p.add_argument("--color", default=None,
                   help="Field name to use as colour channel (default: auto-detect type)")
    p.add_argument("--output", default=None, metavar="PATH.html",
                   help="Save interactive Plotly HTML to this path")
    p.add_argument("--gui", action="store_true",
                   help="Launch matplotlib GUI (default when --output not given)")
    p.add_argument("--alpha", type=float, default=0.4,
                   help="Point alpha transparency (default: 0.4)")
    p.add_argument("--title", default=None,
                   help="Plot title (defaults to filename)")
    p.add_argument("--list-fields", action="store_true",
                   help="Print all fields in the .mat file and exit")
    p.add_argument("--sort-by-site", action="store_true",
                   help="Reorder points by DASAR site number before plotting "
                        "(groups same-site detections together in draw order)")
    p.add_argument("--no-declutter", action="store_true",
                   help="Disable overlap reduction (declutter is on by default)")
    p.add_argument("--min-separation", type=float, default=0.005, metavar="FRAC",
                   help="Minimum point separation as a fraction of the embedding's "
                        "bounding-box diagonal, i.e. max overlap allowed (default: 0.005 = 0.5%%)")
    p.add_argument("--save-reduced", default=None, metavar="PATH.mat",
                   help="Save the computed reduced embeddings to a .mat file")
    p.add_argument("--image-folder", default=None,
                   help="Directory of per-detection spectrogram .mat files "
                        "(defaults to the .mat file's own 'image_folder' field, if present)")
    p.add_argument("--wav-base-dir", action="append", default=None, metavar="DIR",
                   help="Root directory containing Shell20YY_GSI_data folders with 24-hour "
                        ".WAV recordings. Repeatable. Enables the 'Play audio clip' button.")
    p.add_argument("--pad-before", type=float, default=2.0,
                   help="Seconds of audio to include before the detection timestamp (default: 2.0)")
    p.add_argument("--pad-after", type=float, default=3.0,
                   help="Seconds of audio to include after the detection timestamp (default: 3.0)")
    p.add_argument("--no-thumbnails", action="store_true",
                   help="Disable per-point spectrogram thumbnails in the exported HTML "
                        "(enabled by default when --output is given and the image folder "
                        "is accessible)")
    p.add_argument("--thumbnail-dir", default=None, metavar="DIR",
                   help="Directory to write per-point spectrogram PNG thumbnails "
                        "(default: '<output-stem>_thumbnails' next to --output)")
    p.add_argument("--thumbnail-cmap", default="inferno",
                   help="Matplotlib colormap for spectrogram thumbnails (default: inferno)")
    return p.parse_args()


def main():
    args = _parse_args()

    mat_path = os.path.abspath(args.mat)
    title = args.title or os.path.basename(mat_path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"\nLoading: {mat_path}")
    data = load_mat(mat_path)
    print(f"  {len(data)} fields found.")

    if args.list_fields:
        list_fields(data)
        return

    # ------------------------------------------------------------------
    # Detect / select embedding field
    # ------------------------------------------------------------------
    if args.field:
        if args.field not in data:
            print(f"ERROR: field '{args.field}' not found. Available:")
            list_fields(data)
            sys.exit(1)
        from loader import _as2d
        X = _as2d(data[args.field])
        if X is None:
            sys.exit(f"Field '{args.field}' is not a 2-D numeric array.")
        field_name = args.field
    else:
        field_name, X = detect_embeddings(data)

    print(f"  Using embedding field: '{field_name}'  shape={X.shape}")
    n, d = X.shape

    # ------------------------------------------------------------------
    # Reduce if needed
    # ------------------------------------------------------------------
    if d > args.dims:
        if args.reduce is None:
            print(
                f"  Embedding is {d}-D; no --reduce method given.\n"
                f"  Add --reduce umap|tsne|pca to project to {args.dims}D."
            )
            # Try to find a pre-computed lower-dim field before failing
            alt_field = None
            target = f"umap_embeddings_{args.dims}d"
            if target in data:
                alt_field = target
            else:
                for name in (f"umap_{args.dims}d", "x_umap", "x_tsne"):
                    if name in data:
                        from loader import _as2d
                        candidate = _as2d(data[name])
                        if candidate is not None and candidate.shape[1] == args.dims:
                            alt_field = name
                            break
            if alt_field:
                from loader import _as2d
                X = _as2d(data[alt_field])
                field_name = alt_field
                d = X.shape[1]
                print(f"  Found pre-computed {d}D field '{alt_field}', using that.")
            else:
                sys.exit("Cannot proceed without a reduction method. Exiting.")

    if d > args.dims and args.reduce:
        X = _reduce(X, args.reduce, args.dims)
        d = args.dims
        if args.save_reduced:
            from scipy.io import savemat
            save_payload = {f"embeddings_{d}d": X}
            _labels_for_save = extract_labels(data, n)
            if _labels_for_save is not None:
                save_payload["type"] = _labels_for_save
            for _key, _val in extract_metadata(data, n).items():
                save_payload.setdefault(_key, _val)
            for _fname_key in ("original_filenames", "filenames", "file_names"):
                if _fname_key in data:
                    save_payload[_fname_key] = data[_fname_key]
                    break
            savemat(args.save_reduced, save_payload)
            print(f"  Saved reduced embeddings → {args.save_reduced}")

    # Slice to requested dims if embedding already has extra columns
    X = X[:, :args.dims]

    if not args.no_declutter:
        X = _declutter(X, min_sep_frac=args.min_separation)

    # ------------------------------------------------------------------
    # Labels & metadata
    # ------------------------------------------------------------------
    labels = extract_labels(data, n)
    if labels is not None:
        print(f"  Type labels detected: {np.unique(labels[np.isfinite(labels)]).astype(int).tolist()}")
    else:
        print("  No type labels found; colouring uniformly.")
        labels = np.zeros(n)

    meta = extract_metadata(data, n)
    # Keep only fields that differ from labels to avoid clutter
    color_fields = {k: v for k, v in meta.items() if k not in ("type",)}

    if args.color and args.color in meta:
        labels = meta[args.color]
        print(f"  Colouring by --color field: '{args.color}'")
    elif args.color:
        print(f"  WARNING: --color field '{args.color}' not found in metadata; using default labels.")

    # Filenames for click inspection
    filenames = None
    for fname_key in ("original_filenames", "filenames", "file_names"):
        if fname_key in data:
            filenames = list(np.asarray(data[fname_key]).flat)
            break

    years, months, sites = _extract_year_month_site(data, n)
    for field_arr, key in ((years, "year"), (months, "month"), (sites, "site")):
        if field_arr is not None:
            color_fields.setdefault(key, field_arr.astype(float))

    if args.sort_by_site:
        if sites is None:
            print("  WARNING: --sort-by-site requested but no 'site' field found; ignoring.")
        else:
            order = np.argsort(sites, kind="stable")
            X = X[order]
            labels = labels[order]
            sites = sites[order]
            if years is not None:
                years = years[order]
            if months is not None:
                months = months[order]
            color_fields = {k: v[order] for k, v in color_fields.items()}
            if filenames is not None:
                filenames = [filenames[i] for i in order]
            print(f"  Sorted {n:,} points by DASAR site number "
                  f"({', '.join(str(s) for s in np.unique(sites))})")

    # ------------------------------------------------------------------
    # Spectrogram image folder / audio (WAV) locator setup
    # ------------------------------------------------------------------
    image_folder = args.image_folder
    if image_folder is None and "image_folder" in data:
        candidate = str(data["image_folder"])
        if os.path.isdir(candidate):
            image_folder = candidate
        else:
            print(f"  NOTE: .mat file's image_folder is not accessible on this machine: {candidate}")
            print("        Pass --image-folder to point at an accessible copy, if any.")

    wav_locator = None
    if args.wav_base_dir:
        from wav_locator import WavLocator
        wav_locator = WavLocator(args.wav_base_dir)
        print(f"  Audio playback enabled; searching {args.wav_base_dir}")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    do_launch_gui = args.gui or (args.output is None)

    thumbnails = None
    if args.output and not args.no_thumbnails and filenames is not None:
        if not image_folder:
            print("  NOTE: skipping spectrogram thumbnails (no accessible image folder; "
                  "pass --image-folder or mount the drive, or use --no-thumbnails to silence this)")
        else:
            thumb_dir = args.thumbnail_dir or (os.path.splitext(args.output)[0] + "_thumbnails")
            print(f"\nExporting spectrogram thumbnails from {image_folder} ...")
            thumbnails = _export_thumbnails(filenames, image_folder, thumb_dir,
                                             cmap=args.thumbnail_cmap)

    if args.output:
        print(f"\nGenerating Plotly HTML…")
        _save_html(X, labels, color_fields, args.output, title, filenames,
                   years=years, months=months, sites=sites, thumbnails=thumbnails)

    if do_launch_gui:
        print(f"\nLaunching interactive GUI…")
        from scatter_gui import launch_gui as _launch
        _launch(X, labels=labels, title=title,
                filenames=filenames, color_fields=color_fields,
                image_folder=image_folder, wav_locator=wav_locator,
                pad_before=args.pad_before, pad_after=args.pad_after)


if __name__ == "__main__":
    main()
