"""Build an interactive UMAP / PaCMAP embedding viewer for the eval dataset.

This writes a self-contained HTML page with:
  - separate UMAP and PaCMAP tabs
  - point hover metadata for each eval sample
  - point-click detail panel with spectrogram thumbnail
  - raw dataset .mat path mapping from eval sample metadata
  - optional audio playback if a matching WAV/MP3/FLAC file is discovered

The current eval NPZ already contains stem/site/date/dasar/call_type metadata,
so the viewer can map each point back to the raw master database directory.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def per_sample_minmax(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def _serialize_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _make_thumbnail_data_urls(images: np.ndarray, size: int = 56) -> list[str]:
    thumbs: list[str] = []
    for im in images:
        if im.ndim == 3:
            im = im[0]
        img = per_sample_minmax(im)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img)
        pil = pil.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG", optimize=True)
        thumbs.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
    return thumbs


def _project_features(features: np.ndarray, method: str, seed: int = 42) -> np.ndarray:
    method = method.lower()
    if method not in {"umap", "pacmap"}:
        raise ValueError("method must be 'umap' or 'pacmap'")

    if features.shape[1] > 50:
        try:
            from sklearn.decomposition import TruncatedSVD
        except ImportError as exc:
            raise ImportError("sklearn is required to precompress high-dimensional features") from exc
        features = TruncatedSVD(n_components=50, random_state=seed).fit_transform(features)

    if method == "umap":
        try:
            import umap as _umap
        except ImportError as exc:
            raise ImportError("umap-learn is not installed. Install it with `pip install umap-learn`") from exc
        reducer = _umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(features).astype(np.float32)

    try:
        import pacmap as _pacmap
    except ImportError as exc:
        raise ImportError("pacmap is not installed. Install it with `pip install pacmap`") from exc
    reducer = _pacmap.PaCMAP(n_components=2, random_state=seed, n_neighbors=15, MN_ratio=0.5, FP_ratio=2.0, apply_pca=False)
    return reducer.fit_transform(features).astype(np.float32)


def _maybe_resolve_raw_paths(
    stems: np.ndarray,
    dates: np.ndarray,
    sites: np.ndarray,
    dasars: np.ndarray,
    call_types: np.ndarray,
    raw_base: Path | None,
    api_audio_base: str | None = None,
) -> tuple[list[str], list[str]]:
    """
    Resolve paths to raw .mat files and audio files.
    
    Args:
        stems, dates, sites, dasars, call_types: metadata arrays
        raw_base: Path to raw dataset directory (for local file:// URIs)
        api_audio_base: Optional base URL for Flask API endpoints (e.g., "http://localhost:8000/api/audio")
                       If provided, constructs URLs like "{api_audio_base}/{stem}.wav"
                       If None, looks for local files and uses file:// URIs
    
    Returns:
        (raw_paths, audio_paths): Lists of file URIs/URLs
    """
    raw_paths: list[str] = []
    audio_paths: list[str] = []
    
    if api_audio_base:
        # Use Flask API endpoints for audio
        for stem in stems:
            raw_paths.append("")  # No raw .mat access via API
            audio_url = f"{api_audio_base}/{stem}.wav"
            audio_paths.append(audio_url)
        return raw_paths, audio_paths
    
    if raw_base is None:
        return ["" for _ in stems], ["" for _ in stems]

    # Use local file:// URIs
    for stem, date, site, dasar, call_type in zip(stems, dates, sites, dasars, call_types):
        year = str(date)[:4]
        day_dir = f"Day_{date}T000000"
        site_dir = raw_base / year / f"Site{site}" / day_dir
        subdir = "Manually_selected_bowhead_calls.dir" if str(call_type) != "0" else "Event_sounds.dir"
        mat_path = site_dir / subdir / "D1.dir" / f"{stem}.mat"
        if mat_path.exists():
            raw_paths.append(mat_path.as_uri())
            audio_url = ""
            for ext in [".wav", ".WAV", ".mp3", ".flac"]:
                candidate = mat_path.with_suffix(ext)
                if candidate.exists():
                    audio_url = candidate.as_uri()
                    break
            audio_paths.append(audio_url)
        else:
            raw_paths.append("")
            audio_paths.append("")
    return raw_paths, audio_paths


def _build_html(
    out_path: Path,
    eval_npz: Path,
    raw_base_dir: Path | None = None,
    api_audio_base: str | None = None,
    thumbnail_size: int = 56,
    seed: int = 42,
) -> None:
    """
    Build interactive embedding viewer HTML.
    
    Args:
        out_path: Output HTML file path
        eval_npz: Input evaluation dataset NPZ file
        raw_base_dir: Optional path to raw dataset directory (for local file access)
        api_audio_base: Optional Flask API base URL for audio streaming (e.g., "http://localhost:8000/api/audio")
        thumbnail_size: Size of spectrogram thumbnails in pixels
        seed: Random seed for projections
    """
    data = np.load(eval_npz, allow_pickle=True)
    images = data["images"]
    labels = data["label"].astype(int)
    call_types = data["call_type"].astype(str)
    sites = data["site"].astype(str)
    dasars = data["dasar"].astype(str)
    dates = data["date"].astype(str)
    stems = data["stem"].astype(str)
    unique_call = data.get("unique_call", np.array([""] * len(labels), dtype=object)).astype(str)
    is_airgun = data.get("is_airgun", np.zeros(len(labels), dtype=bool)).astype(bool)

    n = len(images)
    print(f"Loaded {n} eval samples from {eval_npz}")

    raw_paths, audio_paths = _maybe_resolve_raw_paths(stems, dates, sites, dasars, call_types, raw_base_dir, api_audio_base)
    thumbnails = _make_thumbnail_data_urls(images, size=thumbnail_size)
    features = np.stack([per_sample_minmax(im).ravel().astype(np.float32) for im in images])

    print("Computing UMAP projection...")
    umap_xy = _project_features(features, "umap", seed=seed)
    print("Computing PaCMAP projection...")
    pacmap_xy = _project_features(features, "pacmap", seed=seed)

    labels_text = ["call" if lab == 1 else "non-call" for lab in labels]
    colors = ["#1f77b4" if lab == 1 else "#d62728" for lab in labels]
    sizes = [8 if lab == 1 else 6 for lab in labels]

    sample_records = []
    for i in range(n):
        sample_records.append({
            "index": int(i),
            "stem": stems[i],
            "date": dates[i],
            "site": sites[i],
            "dasar": dasars[i],
            "call_type": call_types[i],
            "label": labels_text[i],
            "unique_call": unique_call[i],
            "is_airgun": bool(is_airgun[i]),
            "raw_mat_uri": raw_paths[i],
            "audio_uri": audio_paths[i],
            "thumbnail": thumbnails[i],
        })

    def _dump(v):
        return json.dumps(v, default=_serialize_value)

    sample_records_json = json.dumps(sample_records)
    umap_x_json = json.dumps(umap_xy[:, 0].tolist())
    umap_y_json = json.dumps(umap_xy[:, 1].tolist())
    pacmap_x_json = json.dumps(pacmap_xy[:, 0].tolist())
    pacmap_y_json = json.dumps(pacmap_xy[:, 1].tolist())
    colors_json = json.dumps(colors)
    sizes_json = json.dumps(sizes)
    customdata_json = json.dumps(
        [[
            i,
            sample_records[i]["stem"],
            sample_records[i]["date"],
            sample_records[i]["site"],
            sample_records[i]["dasar"],
            sample_records[i]["call_type"],
            sample_records[i]["label"],
        ] for i in range(n)]
    )

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval embedding visualization</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body { margin: 0; font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; background: #f9fafb; color: #111; }
header { padding: 20px 24px; max-width: 1200px; margin: auto; }
header h1 { margin: 0 0 8px; font-size: 24px; }
.tab-bar { display: flex; gap: 10px; padding: 0 24px 16px; max-width: 1200px; margin: auto; }
.tab-bar button { padding: 10px 16px; border: 1px solid #cbd5e1; border-radius: 8px; background: white; cursor: pointer; font-size: 14px; }
.tab-bar button.active { background: #1d4ed8; color: white; border-color: #1d4ed8; }
.main-grid { display: grid; grid-template-columns: 1.5fr 0.85fr; gap: 18px; max-width: 1200px; margin: auto; padding: 0 24px 24px; }
.plot-card, .detail-card { background: white; border: 1px solid #e2e8f0; border-radius: 16px; box-shadow: 0 10px 25px rgba(15,23,42,0.06); }
.plot-card { min-height: 650px; padding: 14px; }
.detail-card { padding: 18px; display: flex; flex-direction: column; gap: 16px; }
.plot-container { width: 100%; height: 620px; }
.detail-card img { width: 100%; border-radius: 12px; object-fit: contain; background: #111; }
.detail-item { font-size: 14px; line-height: 1.6; word-break: break-word; }
.detail-item strong { color: #111; }
.detail-link a { color: #1d4ed8; text-decoration: none; }
@media (max-width: 1024px) { .main-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>Evaluation Embedding Viewer</h1>
  <p style="margin: 6px 0 0; color:#475569; max-width: 860px;">Click a point to inspect a sample, view the spectrogram thumbnail, and open the raw dataset file if available.</p>
</header>
<div class="tab-bar">
  <button id="btn-umap" class="active" onclick="switchTab('umap')">UMAP</button>
  <button id="btn-pacmap" onclick="switchTab('pacmap')">PaCMAP</button>
</div>
<div class="main-grid">
  <section class="plot-card">
    <div id="umap_plot" class="plot-container"></div>
    <div id="pacmap_plot" class="plot-container" style="display:none"></div>
  </section>
  <aside class="detail-card">
    <div class="detail-item"><strong>Selected sample</strong></div>
    <img id="detail-image" src="" alt="Spectrogram thumbnail" />
    <div id="detail-meta" class="detail-item">Click a point in the scatter above to inspect its metadata.</div>
    <div id="detail-path" class="detail-item detail-link"></div>
    <audio id="detail-audio" controls style="display:none; width: 100%;"></audio>
  </aside>
</div>
<script>
const sampleRecords = SAMPLE_RECORDS_JSON;

const umapData = [{
  type: 'scattergl',
  mode: 'markers',
  x: UMAP_X_JSON,
  y: UMAP_Y_JSON,
  marker: { color: COLORS_JSON, size: SIZES_JSON, opacity: 0.8 },
  customdata: CUSTOMDATA_JSON,
  hovertemplate: 'Site %{customdata[3]}<br>Date %{customdata[2]}<br>DASAR %{customdata[4]}<br>Type %{customdata[5]}<br>Label %{customdata[6]}<extra></extra>',
  name: 'Eval samples',
}];
const pacmapData = [{
  type: 'scattergl',
  mode: 'markers',
  x: PACMAP_X_JSON,
  y: PACMAP_Y_JSON,
  marker: { color: COLORS_JSON, size: SIZES_JSON, opacity: 0.8 },
  customdata: CUSTOMDATA_JSON,
  hovertemplate: 'Site %{customdata[3]}<br>Date %{customdata[2]}<br>DASAR %{customdata[4]}<br>Type %{customdata[5]}<br>Label %{customdata[6]}<extra></extra>',
  name: 'Eval samples',
}];

const layout = {
  margin: { t: 40, r: 20, l: 40, b: 40 },
  xaxis: { title: 'Component 1', zeroline: false },
  yaxis: { title: 'Component 2', zeroline: false },
  hovermode: 'closest',
  legend: { orientation: 'h', x: 0.02, y: 1.12 },
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'white',
  title: { text: 'Eval dataset embedding', x: 0.01, xanchor: 'left' },
};

const umapLayout = JSON.parse(JSON.stringify(layout));
umapLayout.title = { text: 'UMAP embedding', x: 0.01, xanchor: 'left' };
const pacmapLayout = JSON.parse(JSON.stringify(layout));
pacmapLayout.title = { text: 'PaCMAP embedding', x: 0.01, xanchor: 'left' };

Plotly.newPlot('umap_plot', umapData, umapLayout, {responsive:true});
Plotly.newPlot('pacmap_plot', pacmapData, pacmapLayout, {responsive:true});

function switchTab(tab) {
  document.getElementById('btn-umap').classList.toggle('active', tab === 'umap');
  document.getElementById('btn-pacmap').classList.toggle('active', tab === 'pacmap');
  document.getElementById('umap_plot').style.display = tab === 'umap' ? 'block' : 'none';
  document.getElementById('pacmap_plot').style.display = tab === 'pacmap' ? 'block' : 'none';
}

function showSampleDetails(index) {
  const rec = sampleRecords[index];
  document.getElementById('detail-image').src = rec.thumbnail;
  document.getElementById('detail-meta').innerHTML = `
    <strong>Stem:</strong> ${rec.stem}<br>
    <strong>Date:</strong> ${rec.date}<br>
    <strong>Site:</strong> ${rec.site}<br>
    <strong>DASAR:</strong> ${rec.dasar}<br>
    <strong>Type:</strong> ${rec.call_type}<br>
    <strong>Label:</strong> ${rec.label}<br>
    <strong>Unique call:</strong> ${rec.unique_call || '—'}<br>
    <strong>Airgun:</strong> ${rec.is_airgun ? 'yes' : 'no'}
  `;
  const pathEl = document.getElementById('detail-path');
  if (rec.raw_mat_uri) {
    pathEl.innerHTML = `<strong>Raw .mat:</strong> <a href="${rec.raw_mat_uri}" target="_blank">Open raw sample</a>`;
  } else {
    pathEl.innerHTML = `<strong>Raw .mat:</strong> unavailable`;
  }
  const audioEl = document.getElementById('detail-audio');
  if (rec.audio_uri) {
    audioEl.style.display = 'block';
    audioEl.src = rec.audio_uri;
  } else {
    audioEl.style.display = 'none';
    audioEl.src = '';
  }
}

function bindClicks(containerId) {
  const plot = document.getElementById(containerId);
  plot.on('plotly_click', function(event) {
    if (!event || !event.points || !event.points.length) {
      return;
    }
    const point = event.points[0];
    const idx = point.customdata[0];
    showSampleDetails(idx);
  });
}

bindClicks('umap_plot');
bindClicks('pacmap_plot');
</script>
</body>
</html>
"""

    html = html.replace('SAMPLE_RECORDS_JSON', sample_records_json)
    html = html.replace('UMAP_X_JSON', umap_x_json)
    html = html.replace('UMAP_Y_JSON', umap_y_json)
    html = html.replace('PACMAP_X_JSON', pacmap_x_json)
    html = html.replace('PACMAP_Y_JSON', pacmap_y_json)
    html = html.replace('COLORS_JSON', colors_json)
    html = html.replace('SIZES_JSON', sizes_json)
    html = html.replace('CUSTOMDATA_JSON', customdata_json)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote embedding viewer HTML -> {out_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build eval embedding visualization HTML")
    parser.add_argument("--data", default="data/spectrograms_eval_v2.npz", help="Eval NPZ file")
    parser.add_argument("--out", default="docs/eval_embedding_visualization.html", help="Output HTML file")
    parser.add_argument("--raw-base", default=None, help="Optional raw dataset base directory for mapping sample stems to .mat files (local file:// access)")
    parser.add_argument("--api-audio-base", default=None, help="Optional Flask API base URL for audio streaming (e.g., 'http://localhost:8000/api/audio' or 'https://bowhead-viewer.yourdomain.com/api/audio')")
    parser.add_argument("--thumbnail-size", type=int, default=56, help="Thumbnail size in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Projection random seed")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_base_dir = Path(args.raw_base) if args.raw_base else None
    _build_html(Path(args.out), Path(args.data), raw_base_dir, args.api_audio_base, thumbnail_size=args.thumbnail_size, seed=args.seed)


if __name__ == "__main__":
    main()
