"""Build the interactive UMAP project page: two tabs, one shared set of camera tools.

Tab 1 "Training vs. evaluation" -- the paper's ``fig:UMAP_displays`` as an interactive 3-D plot
(train | eval side by side, or every point in one scene).  Same inputs/conventions as
``generate_umap_training_eval_figure.py``: coordinates of the training and evaluation files are
standardized TOGETHER, CD = ``0 < features.type < 12`` (train) / ``iscall`` (eval), nothing subsampled.

Tab 2 "Click a point -> spectrogram" -- training dataset only.  Click a point and its spectrogram
is shown beside the plot.  Spectrograms exist only for rows whose image is VERIFIED to be the one
the row was built from (see ``umap_spectrogram_atlas.py``); every other point is drawn faint and
reports "no spectrogram available".

The page's own plotly.js bundle is left untouched; the block is wrapped in BEGIN/END comment markers
so re-running replaces it in place.  The browser draws everything from point data embedded once
(base64 float32) plus lazily fetched WebP tiles.

Run:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.benchmark.build_umap_project_page \\
        --page /Users/oceaneboulais/Github/portfolly/interactive_whale_calls_3d.html
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bowhead.benchmark import umap_spectrogram_atlas as atlas_mod

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_MAT = atlas_mod.TRAIN_MAT
EVAL_MAT = REPO_ROOT / "paper/FinalDatabases_/latent_embeddings_3d_eval_8to1_MATLAB_combined.mat"

CD_COLOR, OT_COLOR = "#D55E00", "#0072B2"
PAGE_TITLE = "Bowhead Whale UMAP"
ATLAS_DIR_NAME = "umap_train_spectrograms"
# The paper's three static rows are azimuths 20/80/140 deg at elevation 20 deg.
PAPER_VIEWS = (20, 80, 140)
PAPER_ELEV = 20
CAMERA_RADIUS = 2.2
SPEC_SCALE = 3  # css px per spectrogram pixel on the click tab
FNAME_RE = re.compile(r"^S(\d)(\d\d)([A-Z])(\d)T(\d{8})T(\d{6})_Type(\d+)\.mat$")


# --------------------------------------------------------------------------- data
def load() -> dict:
    tr = loadmat(str(TRAIN_MAT), squeeze_me=True, struct_as_record=False)
    ev = loadmat(str(EVAL_MAT), squeeze_me=True, struct_as_record=False)
    f = tr["features"]
    t = np.asarray(f.type, dtype=float)
    train_cd = (t > 0) & (t < 12)
    eval_cd = np.asarray(ev["iscall"]).astype(bool)
    train_xyz = np.asarray(tr["umap_embeddings_3d"], dtype=np.float64)
    eval_xyz = np.asarray(ev["umap_embeddings_3d"], dtype=np.float64)

    both = np.concatenate([train_xyz, eval_xyz])
    both = (both - both.mean(axis=0)) / both.std(axis=0)  # the paper's combined standardization
    n = len(train_xyz)
    return {
        "train": (both[:n].astype(np.float32), train_cd),
        "eval": (both[n:].astype(np.float32), eval_cd),
        "filenames": np.asarray(tr["original_filenames"]).tolist(),
        "features": {k: np.asarray(getattr(f, k), dtype=float)
                     for k in ("type", "type_org", "iscall", "ischanged", "duration1", "fpeak", "tpeak")},
    }


def _b64(arr: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(arr.astype(dtype)).tobytes()).decode("ascii")


def _b64_planar(xyz: np.ndarray) -> dict:
    """Float32 planar layout (all x, then all y, then all z) so the browser can take zero-copy views."""
    planar = np.ascontiguousarray(xyz.astype("<f4").T).reshape(-1)
    return {"n": int(len(xyz)), "b64": base64.b64encode(planar.tobytes()).decode("ascii")}


def _eye(azim_deg: float) -> dict:
    az, el = math.radians(azim_deg), math.radians(PAPER_ELEV)
    return {"x": CAMERA_RADIUS * math.cos(az) * math.cos(el),
            "y": CAMERA_RADIUS * math.sin(az) * math.cos(el),
            "z": CAMERA_RADIUS * math.sin(el)}


def _ranges(xyz: np.ndarray, pad: float = 0.25) -> list:
    return [[float(xyz[:, i].min() - pad), float(xyz[:, i].max() + pad)] for i in range(3)]


def pack_data(data: dict, atlas: dict) -> dict:
    tr_xyz, tr_cd = data["train"]
    ev_xyz, ev_cd = data["eval"]
    packed = {
        "ranges": _ranges(np.concatenate([tr_xyz, ev_xyz])),
        "rangesTrain": _ranges(tr_xyz),
        "views": [{"az": az, "eye": _eye(az)} for az in PAPER_VIEWS],
        "radius": CAMERA_RADIUS,
        "train": {"ot": _b64_planar(tr_xyz[~tr_cd]), "cd": _b64_planar(tr_xyz[tr_cd])},
        "eval": {"ot": _b64_planar(ev_xyz[~ev_cd]), "cd": _b64_planar(ev_xyz[ev_cd])},
    }

    # Per-point metadata for the click tab, in the SAME order as the packed train points:
    # all OT rows (file order) followed by all CD rows (file order).
    order = np.concatenate([np.flatnonzero(~tr_cd), np.flatnonzero(tr_cd)])
    fn = data["filenames"]
    parsed = [FNAME_RE.match(fn[i]).groups() for i in order]
    F = data["features"]
    ty = F["type"][order]
    slots = atlas["row_slots"][order]
    assert all(g[3] == "0" and g[1] == g[4][2:4] for g in parsed)
    dasar = {"A": 0, "D": 1, "G": 2}
    packed["click"] = {
        "dir": ATLAS_DIR_NAME,
        **{k: atlas[k] for k in ("n_slots", "per_tile", "cols", "rows", "cell_w", "cell_h", "img_h", "img_w")},
        "scale": SPEC_SCALE,
        "slot": _b64(slots, "<i4"),
        "site": _b64(np.array([int(g[0]) for g in parsed]), "u1"),
        "dasar": _b64(np.array([dasar[g[2]] for g in parsed]), "u1"),
        "date": _b64(np.array([int(g[4]) for g in parsed]), "<i4"),
        "time": _b64(np.array([int(g[5]) for g in parsed]), "<i4"),
        "ty0": _b64(F["type_org"][order], "u1"),
        "ty": _b64(np.where(np.isnan(ty), -128, ty), "i1"),
        "chg": _b64(F["ischanged"][order], "u1"),
        "dur": _b64(F["duration1"][order], "<f4"),
        "fp": _b64(F["fpeak"][order], "<f4"),
        "tp": _b64(F["tpeak"][order], "<f4"),
    }
    return packed


# --------------------------------------------------------------------------- html
def _viridis_table() -> str:
    import matplotlib
    c = matplotlib.colormaps["viridis"](np.linspace(0, 1, 256))
    return {ch: " ".join(f"{v:.4f}" for v in c[:, i]) for i, ch in enumerate("RGB")}


def _viridis_gradient() -> str:
    import matplotlib
    stops = matplotlib.colormaps["viridis"](np.linspace(0, 1, 12))
    return ", ".join(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)}) {i / 11 * 100:.0f}%"
                     for i, (r, g, b, _) in enumerate(stops))


def _zoom_group(prefix: str) -> str:
    return (
        f'<div class="grp"><b>Zoom</b>'
        f'<button id="{prefix}-zout" class="zb" title="Zoom out (key: -)">&minus;</button>'
        f'<input id="{prefix}-zoom" type="range" class="zoom-range" min="-1.32" max="3.32" step="0.01" value="0">'
        f'<button id="{prefix}-zin" class="zb" title="Zoom in (key: +)">+</button>'
        f'<span id="{prefix}-zoom-v" style="min-width:3.4em">&times;1.0</span>'
        f'<button id="{prefix}-zreset" title="Reset zoom (key: 0)">Reset</button></div>'
    )


def _spec_axes_html() -> str:
    S, W, H = SPEC_SCALE, atlas_mod.IMG_W * SPEC_SCALE, atlas_mod.IMG_H * SPEC_SCALE
    f0, f1 = 25.0, 25.0 + atlas_mod.DF * atlas_mod.IMG_H          # authors' display axis: 25 + dF*(0:nF)
    t1 = atlas_mod.DT * atlas_mod.IMG_W
    yt = "".join(f'<span class="tk" style="top:{H * (1 - (f - f0) / (f1 - f0)):.1f}px;right:100%">{f}&nbsp;</span>'
                 for f in (100, 200, 300, 400))
    xt = "".join(f'<span class="tk" style="left:{W * t / t1:.1f}px;top:100%;transform:translateX(-50%)">{t}</span>'
                 for t in (0, 1, 2))
    cb = "".join(f'<span class="tk" style="top:{H * (1 - v / 20):.1f}px;left:100%">&nbsp;{v}</span>' for v in (0, 5, 10, 15, 20))
    return yt, xt, cb


def build_html() -> str:
    yt, xt, cb = _spec_axes_html()
    tab = _viridis_table()
    W, H = atlas_mod.IMG_W * SPEC_SCALE, atlas_mod.IMG_H * SPEC_SCALE
    filt = "".join(f'<feFunc{ch} type="table" tableValues="{tab[ch]}"/>' for ch in "RGB")
    css = f"""
<style>
  .umap-ctl {{ font-family: sans-serif; font-size: 14px; display: flex; flex-wrap: wrap; gap: 8px 14px;
              align-items: center; justify-content: center; padding: 12px 8px 2px; }}
  .umap-ctl .grp {{ display: flex; gap: 6px; align-items: center; }}
  .umap-ctl button {{ font: inherit; padding: 5px 12px; border: 1px solid #888; border-radius: 4px;
                     background: #f4f4f4; color: #222; cursor: pointer; }}
  .umap-ctl button.on {{ background: #222; color: #fff; border-color: #222; }}
  .umap-ctl button.zb {{ font-size: 20px; line-height: 1; padding: 3px 14px; }}
  .umap-ctl input[type=range] {{ width: 110px; vertical-align: middle; }}
  .zoom-range {{ width: 220px; }}
  .umap-hint {{ font-family: sans-serif; font-size: 12px; color: #555; text-align: center; padding: 2px 8px 6px; }}
  #uclick-body {{ display: flex; flex-wrap: wrap; gap: 8px 16px; padding: 0 8px; align-items: flex-start; }}
  #uclick-plot {{ flex: 1 1 520px; min-width: 0; height: 760px; }}
  #uclick-side {{ flex: 0 0 {W + 150}px; font-family: sans-serif; font-size: 13px; color: #222; padding-top: 8px; }}
  #uclick-side h3 {{ margin: 0 0 8px; font-size: 15px; }}
  #uclick-wrap {{ position: relative; width: {W}px; height: {H}px; margin: 6px 0 36px 46px; background: #f2f2f2; }}
  #uclick-spec {{ width: {W}px; height: {H}px; image-rendering: pixelated; filter: url(#uclick-viridis); }}
  #uclick-wrap .tk {{ position: absolute; font-size: 11px; color: #444; white-space: nowrap; }}
  #uclick-cb {{ position: absolute; left: {W + 12}px; top: 0; width: 14px; height: {H}px; border: 1px solid #888;
               background: linear-gradient(to top, {_viridis_gradient()}); }}
  #uclick-empty {{ display: none; position: absolute; inset: 0; align-items: center; justify-content: center;
                  text-align: center; padding: 16px; color: #555; background: #f2f2f2; border: 1px dashed #aaa; }}
  #uclick-info td {{ padding: 2px 10px 2px 0; vertical-align: top; }}
  #uclick-info td:first-child {{ color: #666; white-space: nowrap; }}
  #uclick-note {{ margin-top: 10px; font-size: 11px; color: #666; line-height: 1.45; max-width: {W + 130}px; }}
  /* Narrow (phone-width) screens: let the fixed-pixel HTML panels shrink and
     stack instead of forcing horizontal overflow. The Plotly plots' own
     height/margins/fonts are sized responsively in JS instead of here,
     since Plotly sets an inline height on its own div that a CSS rule
     cannot reliably override. */
  @media (max-width: 700px) {{
    .umap-ctl {{ font-size: 13px; gap: 6px 10px; }}
    .umap-ctl .grp {{ flex-wrap: wrap; justify-content: center; row-gap: 4px; }}
    .umap-ctl input[type=range] {{ width: 80px; }}
    .zoom-range {{ width: 90px; }}
    #uclick-plot {{ flex: 1 1 100%; min-width: 0; }}
    /* The spectrogram's tick labels are positioned at fixed pixel offsets
       (computed from IMG_W/IMG_H), so rather than rescale that image and
       risk misaligning them, the whole side panel scrolls horizontally as
       one unit, stacked full-width below the 3-D plot. */
    #uclick-side {{ flex: 1 1 100%; width: 100%; max-width: 100%; padding-top: 4px; overflow-x: auto; }}
    #uclick-note {{ max-width: 100%; }}
  }}
</style>"""
    return css + f"""
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
  <filter id="uclick-viridis" color-interpolation-filters="sRGB"><feComponentTransfer>{filt}</feComponentTransfer></filter>
</defs></svg>
<div class="umap-ctl" id="umap-modes">
  <div class="grp"><b>Layout</b>
    <button data-layout="side">Train | Eval side by side</button>
    <button data-layout="combined">All points together</button>
    <button data-layout="click">Click a point &rarr; spectrogram</button></div>
</div>

<div id="umap-panel-overview">
  <div class="umap-ctl" id="umap-controls">
    <div class="grp"><b>View</b>
      <button data-view="0">1</button><button data-view="1">2</button><button data-view="2">3</button>
      <button id="umap-spin">&#9654; Spin</button></div>
    {_zoom_group("umap")}
    <div class="grp"><label>Point size <input id="umap-size" type="range" min="1" max="8" step="0.5" value="3">
      <span id="umap-size-v">3</span></label></div>
    <div class="grp"><label>Opacity <input id="umap-opacity" type="range" min="0.1" max="1" step="0.05" value="0.5">
      <span id="umap-opacity-v">0.5</span></label></div>
  </div>
  <div id="umap-hint" class="umap-hint"></div>
  <div id="umap-train-eval" class="plotly-graph-div" style="height:820px; width:100%;"></div>
</div>

<div id="umap-panel-click" style="display:none">
  <div class="umap-ctl">
    <div class="grp"><b>View</b><button id="uclick-spin">&#9654; Spin</button></div>
    {_zoom_group("uclick")}
  </div>
  <div id="uclick-hint" class="umap-hint"></div>
  <div id="uclick-body">
    <div id="uclick-plot" class="plotly-graph-div"></div>
    <div id="uclick-side">
      <h3 id="uclick-title">Click a point in the UMAP</h3>
      <div id="uclick-wrap">
        <div id="uclick-spec"></div>
        <div id="uclick-empty"><span id="uclick-empty-msg"></span></div>
        {yt}{xt}
        <span class="tk" style="left:50%;top:100%;margin-top:16px;transform:translateX(-50%)">Time (s)</span>
        <span class="tk" style="left:-46px;top:50%;transform:translate(-50%,-50%) rotate(-90deg);transform-origin:center;margin-left:-14px">Frequency (Hz)</span>
        <div id="uclick-cb"></div>{cb}
        <span class="tk" style="left:{W + 12}px;top:-16px">SNR (dB)</span>
      </div>
      <table id="uclick-info"><tbody></tbody></table>
      <div id="uclick-note">
        Spectrograms appear only for points whose image was <b>verified</b> to be the one the point was computed from:
        the peak frequency and peak time recomputed from its pixels match the stored values exactly.
        Colour scale 0&ndash;20&nbsp;dB SNR (as in the authors&rsquo; MATLAB figures); images are WebP-compressed (about 0.5&nbsp;dB RMS error).
        The stored peak frequency is computed as dF&middot;row&nbsp;index, without the 25&nbsp;Hz band offset used for this axis,
        so the same peak reads about 21&nbsp;Hz lower than its position on the axis.
      </div>
    </div>
  </div>
</div>
"""


# --------------------------------------------------------------------------- javascript
RENDER_JS = r"""
(function () {
  var D = __DATA__;
  var OT_C = "__OT__", CD_C = "__CD__";
  var CONFIG = {responsive: true, displaylogo: false, toImageButtonOptions:
    {format: "png", filename: "umap", height: 1080, width: 1920, scale: 2}};

  function decode(b64) {
    var s = atob(b64), u = new Uint8Array(s.length);
    for (var i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
    return u.buffer;
  }
  function planar(g) {
    var f = new Float32Array(decode(g.b64)), n = g.n;
    return {n: n, x: f.subarray(0, n), y: f.subarray(n, 2 * n), z: f.subarray(2 * n, 3 * n)};
  }
  function concat(a, b) {
    var o = {n: a.n + b.n};
    ["x", "y", "z"].forEach(function (k) {
      var arr = new Float32Array(o.n); arr.set(a[k], 0); arr.set(b[k], a.n); o[k] = arr;
    });
    return o;
  }
  function fmt(n) { return n.toLocaleString("en-US"); }
  function pad(n, w) { var s = String(n); while (s.length < w) s = "0" + s; return s; }
  var G = {tOT: planar(D.train.ot), tCD: planar(D.train.cd), eOT: planar(D.eval.ot), eCD: planar(D.eval.cd)};
  var current = "overview", A = null, B = null;

  /* ---- camera tools shared by both tabs: zoom (slider, buttons, wheel), spin, current eye ---- */
  function CameraTools(o) {
    var gd = o.gd, P = o.prefix, R0 = D.radius, ZMIN = 0.4, ZMAX = 10, SPIN_DEG_PER_S = 15;
    var zoomBusy = false, pendingZ = null, lastReqZ = 1, sliderDown = false;
    var spinning = false, inFlight = false, lastT = 0;
    function el(s) { return document.getElementById(P + "-" + s); }
    function currentEye() {
      var l = gd._fullLayout, e = l && l.scene && l.scene.camera && l.scene.camera.eye;
      return e ? {x: e.x, y: e.y, z: e.z} : D.views[0].eye;
    }
    function setEye(e) {
      var upd = {};
      o.scenes().forEach(function (s) { upd[s + ".camera.eye"] = e; });
      return Plotly.relayout(gd, upd);
    }
    function zoomOf(e) { return R0 / Math.hypot(e.x, e.y, e.z); }
    function showZoom(z) {
      if (!sliderDown) el("zoom").value = Math.log2(z);
      el("zoom-v").innerHTML = "&times;" + (z >= 10 ? z.toFixed(0) : z.toFixed(1));
    }
    function applyZoom(z) {
      var e = currentEye(), k = (R0 / z) / Math.hypot(e.x, e.y, e.z);
      showZoom(z);
      return setEye({x: e.x * k, y: e.y * k, z: e.z * k});
    }
    // Applied at once; only if a previous zoom is still running is the newest request queued.
    function requestZoom(z) {
      z = Math.min(ZMAX, Math.max(ZMIN, z));
      lastReqZ = z;
      if (zoomBusy) { pendingZ = z; return; }
      runZoom(z);
    }
    function runZoom(z) {
      zoomBusy = true; pendingZ = null;
      function done() { zoomBusy = false; if (pendingZ !== null) runZoom(pendingZ); }
      applyZoom(z).then(done, done);
    }
    function zoomBy(f) {
      var base = (zoomBusy || pendingZ !== null) ? lastReqZ : zoomOf(currentEye());
      requestZoom(base * f);
    }
    function eyeFromEvent(ev) {
      var c = ev["scene.camera"] || ev["scene2.camera"];
      return (c && c.eye) || ev["scene.camera.eye"] || ev["scene2.camera.eye"] || null;
    }
    function onRelayout(ev) { var e = eyeFromEvent(ev); if (e) showZoom(zoomOf(e)); }
    function refresh() { showZoom(zoomOf(currentEye())); }

    var spinBtn = el("spin");
    function setSpinLabel() {
      spinBtn.innerHTML = spinning ? "&#9208; Stop" : "&#9654; Spin";
      spinBtn.classList.toggle("on", spinning);
    }
    function release() { inFlight = false; }
    function frame(t) {
      if (!spinning) return;
      var dt = lastT ? Math.min((t - lastT) / 1000, 0.1) : 0;
      lastT = t;
      if (!inFlight && dt > 0) {
        var e = currentEye(), th = SPIN_DEG_PER_S * Math.PI / 180 * dt, c = Math.cos(th), s = Math.sin(th);
        inFlight = true;
        setEye({x: e.x * c - e.y * s, y: e.x * s + e.y * c, z: e.z}).then(release, release);
      }
      requestAnimationFrame(frame);
    }
    function stopSpin() { if (spinning) { spinning = false; setSpinLabel(); } }
    spinBtn.addEventListener("click", function () {
      if (spinning) stopSpin(); else { spinning = true; lastT = 0; setSpinLabel(); requestAnimationFrame(frame); }
    });
    gd.addEventListener("pointerdown", stopSpin);

    el("zin").addEventListener("click", function () { zoomBy(1.25); });
    el("zout").addEventListener("click", function () { zoomBy(1 / 1.25); });
    el("zreset").addEventListener("click", function () { requestZoom(1); });
    el("zoom").addEventListener("input", function (ev) { requestZoom(Math.pow(2, +ev.target.value)); });
    el("zoom").addEventListener("pointerdown", function () { sliderDown = true; });
    window.addEventListener("pointerup", function () { sliderDown = false; });
    // Plotly's own wheel zoom jumps ~2x per notch, so the wheel is re-implemented at ~1.16x.
    gd.addEventListener("wheel", function (ev) {
      ev.preventDefault(); ev.stopPropagation();
      var unit = ev.deltaMode === 1 ? 33 : 1, k = ev.ctrlKey ? 0.01 : 0.0015;  // trackpad pinch = ctrl+wheel
      zoomBy(Math.exp(-ev.deltaY * unit * k));
    }, {capture: true, passive: false});

    return {currentEye: currentEye, setEye: setEye, refresh: refresh, onRelayout: onRelayout, stopSpin: stopSpin,
            zoomBy: zoomBy, requestZoom: requestZoom, setBusy: function (b) { inFlight = b; }};
  }

  /* ---- tab 1: training vs. evaluation ---- */
  function initOverview() {
    var gd = document.getElementById("umap-train-eval");
    var mode = "side", drawing = false, merged = null, size = 3, opacity = 0.5, touched = false;
    var DEFAULTS = {side: {size: 3, opacity: 0.5}, combined: {size: 2, opacity: 0.25}};  // 300k points are denser
    var tools = CameraTools({gd: gd, prefix: "umap", scenes: function () { return mode === "side" ? ["scene", "scene2"] : ["scene"]; }});

    function getMerged() {
      if (!merged) merged = {ot: concat(G.tOT, G.eOT), cd: concat(G.tCD, G.eCD)};
      return merged;
    }
    function trace(g, color, name, scene, showLegend) {
      return {type: "scatter3d", mode: "markers", x: g.x, y: g.y, z: g.z, name: name, scene: scene,
              showlegend: showLegend, hoverinfo: "skip", marker: {size: size, color: color, opacity: opacity}};
    }
    function sceneDef(x0, x1, eye) {
      function ax(i, t) { return {title: {text: t}, range: D.ranges[i], dtick: 1}; }
      return {domain: {x: [x0, x1], y: [0, 1]}, xaxis: ax(0, "UMAP 1"), yaxis: ax(1, "UMAP 2"), zaxis: ax(2, "UMAP 3"),
              aspectmode: "cube", camera: {eye: eye, center: {x: 0, y: 0, z: 0}, up: {x: 0, y: 0, z: 1}}};
    }
    // Annotation text is NOT constrained to its scene's domain width, so on a
    // narrow (phone) screen two side-by-side captions at x=0.25/0.75 collide.
    // Below the breakpoint, stack them full-width on separate lines instead
    // of placing them side by side, and shrink the fonts to match.
    var NARROW_PX = 640;
    function ann(text, x, y, fontSize) {
      return {text: text, xref: "paper", yref: "paper", x: x, y: y, xanchor: "center", yanchor: "bottom",
              showarrow: false, font: {size: fontSize}};
    }
    function counts(ot, cd) { return fmt(ot.n + cd.n) + " (" + fmt(cd.n) + " CD, " + fmt(ot.n) + " OT)"; }
    function build(eye) {
      var layout, data;
      var narrow = (gd.clientWidth || gd.getBoundingClientRect().width || 900) < NARROW_PX;
      var titleSize = narrow ? 13 : 18, annSize = narrow ? 11 : 13;
      if (mode === "side") {
        data = [trace(G.tOT, OT_C, "Other transient (OT)", "scene", true), trace(G.tCD, CD_C, "Call detection (CD)", "scene", true),
                trace(G.eOT, OT_C, "Other transient (OT)", "scene2", false), trace(G.eCD, CD_C, "Call detection (CD)", "scene2", false)];
        layout = {scene: sceneDef(0, 0.5, eye), scene2: sceneDef(0.5, 1, eye),
                  annotations: narrow
                    ? [ann("<b>Training dataset</b><br>n = " + counts(G.tOT, G.tCD), 0.5, 1.38, annSize),
                       ann("<b>Evaluation dataset</b><br>n = " + counts(G.eOT, G.eCD), 0.5, 1.0, annSize)]
                    : [ann("<b>Training dataset</b><br>n = " + counts(G.tOT, G.tCD), 0.25, 1.0, annSize),
                       ann("<b>Evaluation dataset</b><br>n = " + counts(G.eOT, G.eCD), 0.75, 1.0, annSize)]};
      } else {
        var m = getMerged();
        data = [trace(m.ot, OT_C, "Other transient (OT)", "scene", true), trace(m.cd, CD_C, "Call detection (CD)", "scene", true)];
        layout = {scene: sceneDef(0, 1, eye),
                  annotations: [ann("<b>All data points: training + evaluation</b><br>n = " + counts(m.ot, m.cd), 0.5, 1.0, annSize)]};
      }
      layout.title = {text: "<b>UMAP of the autoencoder latent space</b>", x: 0.5, xanchor: "center", y: 0.97, yanchor: "top", font: {size: titleSize}};
      layout.legend = {itemsizing: "constant", orientation: "h", x: 0.5, xanchor: "center", y: -0.02, yanchor: "top"};
      layout.margin = {l: 0, r: 0, t: narrow ? (mode === "side" ? 190 : 95) : 90, b: 60};
      layout.height = narrow ? 480 : 820;
      return {data: data, layout: layout};
    }
    function attachSync() {
      ["plotly_relayout", "plotly_relayouting"].forEach(function (n) { gd.removeAllListeners(n); });
      gd.on("plotly_relayout", tools.onRelayout);
      gd.on("plotly_relayouting", tools.onRelayout);
      if (mode !== "side") return;
      var busy = false;
      function mirror(ev) {
        if (busy) return;
        var src = ev["scene.camera"] ? "scene" : (ev["scene2.camera"] ? "scene2" : null);
        if (!src) return;
        var dst = src === "scene" ? "scene2" : "scene", upd = {};
        upd[dst + ".camera"] = ev[src + ".camera"];
        busy = true;
        Plotly.relayout(gd, upd).then(function () { busy = false; }, function () { busy = false; });
      }
      gd.on("plotly_relayout", mirror);
      gd.on("plotly_relayouting", mirror);
    }
    function showSliders() {
      document.getElementById("umap-size").value = size;
      document.getElementById("umap-size-v").textContent = size;
      document.getElementById("umap-opacity").value = opacity;
      document.getElementById("umap-opacity-v").textContent = opacity;
    }
    function markButtons() {
      document.querySelectorAll("#umap-modes [data-layout]").forEach(function (b) {
        b.classList.toggle("on", current === "overview" && b.getAttribute("data-layout") === mode);
      });
      var total = G.tOT.n + G.tCD.n + G.eOT.n + G.eCD.n;
      document.getElementById("umap-hint").textContent =
        "Drag to rotate \u00b7 zoom with the slider, \u2212/+ buttons, scroll wheel or the +/\u2212 keys \u00b7 " +
        "click a legend entry to hide/show CD or OT \u00b7 all " + fmt(total) + " points are drawn (none subsampled)" +
        (mode === "side" ? " \u00b7 the two panels rotate together" : "");
    }
    function draw(newMode) {
      var eye = (gd._fullLayout && gd._fullLayout.scene) ? tools.currentEye() : D.views[0].eye;
      mode = newMode; drawing = true; tools.setBusy(true);
      if (!touched) { size = DEFAULTS[mode].size; opacity = DEFAULTS[mode].opacity; showSliders(); }
      var b = build(eye);
      return Plotly.newPlot(gd, b.data, b.layout, CONFIG).then(function () {
        attachSync(); markButtons(); tools.refresh(); drawing = false; tools.setBusy(false);
      });
    }
    document.querySelectorAll("#umap-modes [data-layout]").forEach(function (b) {
      b.addEventListener("click", function () {
        var m = b.getAttribute("data-layout");
        if (m === "click") { showTab("click"); return; }
        showTab("overview");
        if (m !== mode && !drawing) draw(m);
      });
    });
    document.querySelectorAll("#umap-controls [data-view]").forEach(function (b) {
      b.addEventListener("click", function () { tools.setEye(D.views[+b.getAttribute("data-view")].eye); });
    });
    function slider(id, onChange) {
      var s = document.getElementById(id), lab = document.getElementById(id + "-v");
      s.addEventListener("input", function () { lab.textContent = s.value; });
      s.addEventListener("change", function () { touched = true; onChange(+s.value); });
    }
    slider("umap-size", function (v) { size = v; Plotly.restyle(gd, {"marker.size": v}); });
    slider("umap-opacity", function (v) { opacity = v; Plotly.restyle(gd, {"marker.opacity": v}); });
    draw("side");
    return {gd: gd, tools: tools, mode: function () { return mode; }};
  }

  /* ---- tab 2: click a point -> spectrogram (training dataset only) ---- */
  function initClick() {
    var gd = document.getElementById("uclick-plot"), C = D.click, S = C.scale;
    var slot = new Int32Array(decode(C.slot)), site = new Uint8Array(decode(C.site)), dasar = new Uint8Array(decode(C.dasar));
    var date = new Int32Array(decode(C.date)), time = new Int32Array(decode(C.time)), ty0 = new Uint8Array(decode(C.ty0));
    var ty = new Int8Array(decode(C.ty)), chg = new Uint8Array(decode(C.chg));
    var dur = new Float32Array(decode(C.dur)), fp = new Float32Array(decode(C.fp)), tp = new Float32Array(decode(C.tp));
    var DASAR = ["A", "D", "G"];
    var P = concat(G.tOT, G.tCD), N = P.n, NOT = G.tOT.n;   // same order as the metadata arrays: all OT, then all CD

    var lists = [[], [], [], []], nVerified = 0;             // index = hasSpectrogram * 2 + isCD
    for (var p = 0; p < N; p++) {
      var has = slot[p] >= 0 ? 1 : 0;
      nVerified += has;
      lists[has * 2 + (p >= NOT ? 1 : 0)].push(p);
    }
    // On a narrow screen the legend's longest entries ("Call detection (CD),
    // spectrogram available (9,323)") don't fit even stacked one per line, so
    // use a shorter label there instead.
    var clickNarrowLegend = (gd.clientWidth || gd.getBoundingClientRect().width || 900) < 640;
    var groups = lists.map(function (l, gi) {
      var pos = Int32Array.from(l), n = pos.length, x = new Float32Array(n), y = new Float32Array(n), z = new Float32Array(n);
      for (var i = 0; i < n; i++) { x[i] = P.x[pos[i]]; y[i] = P.y[pos[i]]; z[i] = P.z[pos[i]]; }
      var cd = gi % 2, has = gi >> 1;
      var name = clickNarrowLegend
        ? (cd ? "CD" : "OT") + (has ? ", has spectrogram" : ", no spectrogram") + " (" + fmt(n) + ")"
        : (cd ? "Call detection (CD)" : "Other transient (OT)") + (has ? ", spectrogram available" : ", no spectrogram") + " (" + fmt(n) + ")";
      return {pos: pos, trace: {
        type: "scatter3d", mode: "markers", x: x, y: y, z: z, hoverinfo: "none", showlegend: true,
        name: name,
        marker: {size: has ? 4 : 2, color: cd ? CD_C : OT_C, opacity: has ? 0.75 : 0.12}}};
    });
    var selected = {type: "scatter3d", mode: "markers", x: [], y: [], z: [], hoverinfo: "skip", showlegend: false, name: "selected",
                    marker: {size: 9, color: "#111", symbol: "diamond", line: {color: "#fff", width: 2}}};
    var R = D.rangesTrain;
    function ax(i, t) { return {title: {text: t}, range: R[i], dtick: 1}; }
    var clickNarrow = clickNarrowLegend;
    var layout = {
      scene: {xaxis: ax(0, "UMAP 1"), yaxis: ax(1, "UMAP 2"), zaxis: ax(2, "UMAP 3"), aspectmode: "cube",
              camera: {eye: D.views[0].eye, center: {x: 0, y: 0, z: 0}, up: {x: 0, y: 0, z: 1}}},
      // Horizontal legends don't wrap, so this tab's four fairly long entries
      // (e.g. "Call detection (CD), spectrogram available (9,323)") clip off
      // the edge on a narrow screen; stack them vertically there and let each
      // entry use the full legend width (Plotly otherwise sizes every column
      // to a single reference entry's width, clipping the longer ones).
      legend: clickNarrow
        ? {itemsizing: "constant", orientation: "v", font: {size: 11}, entrywidth: 1, entrywidthmode: "fraction",
           x: 0.5, xanchor: "center", y: -0.03, yanchor: "top"}
        : {itemsizing: "constant", orientation: "h", x: 0.5, xanchor: "center", y: -0.02, yanchor: "top"},
      margin: {l: 0, r: 0, t: 10, b: clickNarrow ? 230 : 110}, height: clickNarrow ? 440 : 760};
    var tools = CameraTools({gd: gd, prefix: "uclick", scenes: function () { return ["scene"]; }});

    var specEl = document.getElementById("uclick-spec"), emptyEl = document.getElementById("uclick-empty");
    var tbody = document.querySelector("#uclick-info tbody");
    function stemOf(p) {
      return "S" + site[p] + String(date[p]).slice(2, 4) + DASAR[dasar[p]] + "0T" + date[p] + "T" + pad(time[p], 6) + "_Type" + ty0[p];
    }
    function showSpec(sl) {
      var tile = Math.floor(sl / C.per_tile), k = sl % C.per_tile, c = k % C.cols, r = Math.floor(k / C.cols);
      specEl.style.backgroundImage = 'url("' + C.dir + "/tile_" + pad(tile, 3) + '.webp")';
      specEl.style.backgroundSize = (C.cols * C.cell_w * S) + "px " + (C.rows * C.cell_h * S) + "px";
      specEl.style.backgroundPosition = (-c * C.cell_w * S) + "px " + (-r * C.cell_h * S) + "px";
    }
    function row(k, v) { return "<tr><td>" + k + "</td><td>" + v + "</td></tr>"; }
    function select(p) {
      /*@test*/ window.__umapLast = p; /*@/test*/
      var d = String(date[p]), t = pad(time[p], 6), isCD = p >= NOT;
      document.getElementById("uclick-title").textContent = stemOf(p) + ".mat";
      if (slot[p] >= 0) {
        emptyEl.style.display = "none"; specEl.style.visibility = "visible"; showSpec(slot[p]);
      } else {
        specEl.style.visibility = "hidden"; emptyEl.style.display = "flex";
        document.getElementById("uclick-empty-msg").textContent =
          "No spectrogram available for this point (verified spectrograms exist for " + fmt(nVerified) + " of " + fmt(N) + " training points).";
      }
      tbody.innerHTML =
        row("Site / DASAR", "Site " + site[p] + " / DASAR " + DASAR[dasar[p]]) +
        row("Date, time (filename)", d.slice(0, 4) + "-" + d.slice(4, 6) + "-" + d.slice(6, 8) + " " + t.slice(0, 2) + ":" + t.slice(2, 4) + ":" + t.slice(4, 6)) +
        row("Original label", ty0[p] === 0 ? "Type 0 (other transient)" : "Type " + ty0[p] + " (manual call type)") +
        row("Reviewed label", (isCD ? "Call detection (CD)" : "Other transient (OT)") + (ty[p] === -128 ? "" : " &middot; review code " + ty[p])) +
        row("Changed by review", chg[p] ? "yes" : "no") +
        row("Duration (transient detector)", dur[p].toFixed(2) + " s") +
        row("Peak frequency (stored)", fp[p].toFixed(1) + " Hz") +
        row("Peak time (stored)", tp[p].toFixed(2) + " s");
      Plotly.restyle(gd, {x: [[P.x[p]]], y: [[P.y[p]]], z: [[P.z[p]]]}, [4]);
    }

    document.getElementById("uclick-hint").textContent =
      "Click any point to show its spectrogram \u00b7 bright points have a verified spectrogram (" + fmt(nVerified) + " of " + fmt(N) +
      "), faint points do not \u00b7 drag to rotate \u00b7 zoom with the slider, \u2212/+ buttons, scroll wheel or the +/\u2212 keys";
    Plotly.newPlot(gd, groups.map(function (g) { return g.trace; }).concat([selected]), layout, CONFIG).then(function () {
      // Plotly's gl3d "plotly_click" proved unreliable (a bare 3-D scatter fired 1 click in 12) and its hover
      // pick only hits when the cursor is within a few px of a point (18% of grid positions over the cloud,
      // and the verified points are the hardest to hit).  So the page picks for itself: it projects every
      // point with the live camera matrices (scene.glplot.cameraParams, the same ones the renderer uses),
      // finds the nearest point to the cursor within PICK_PX, preferring points that have a spectrogram,
      // and treats a press+release at the same spot as a click on it.
      var PICK_PX = 12, VERIFIED_BONUS_PX = 5, DEPTH_PX = 1.5;
      var scn = gd._fullLayout.scene._scene, glp = scn.glplot;
      var sc = scn.dataScale;
      var cache = null, cacheKey = "", wMin = 0, wMax = 1;
      function mat() {
        var cp = glp.cameraParams; return {m: cp.model, v: cp.view, p: cp.projection};
      }
      // Plotly scales each data axis by scene.dataScale before the model/view/projection matrices.
      // (Found empirically: with this scaling plotly's own hover-picked points land within ~3 px of the
      // cursor; without it, 76-457 px away.)
      function project(m, x, y, z) {                       // data -> clip space -> fractions of the canvas
        x *= sc[0]; y *= sc[1]; z *= sc[2];
        var X = x * m.m[0] + y * m.m[4] + z * m.m[8] + m.m[12], Y = x * m.m[1] + y * m.m[5] + z * m.m[9] + m.m[13],
            Z = x * m.m[2] + y * m.m[6] + z * m.m[10] + m.m[14], W = 1;
        var vx = X * m.v[0] + Y * m.v[4] + Z * m.v[8] + W * m.v[12], vy = X * m.v[1] + Y * m.v[5] + Z * m.v[9] + W * m.v[13],
            vz = X * m.v[2] + Y * m.v[6] + Z * m.v[10] + W * m.v[14], vw = X * m.v[3] + Y * m.v[7] + Z * m.v[11] + W * m.v[15];
        var cx = vx * m.p[0] + vy * m.p[4] + vz * m.p[8] + vw * m.p[12], cy = vx * m.p[1] + vy * m.p[5] + vz * m.p[9] + vw * m.p[13],
            cw = vx * m.p[3] + vy * m.p[7] + vz * m.p[11] + vw * m.p[15];
        return {x: (cx / cw * 0.5 + 0.5), y: (1 - (cy / cw * 0.5 + 0.5)), w: cw};
      }
      function canvasRect() {
        var c = gd.querySelector("canvas"); return (c || gd).getBoundingClientRect();
      }
      function projectAll() {                              // every point, once per camera change (a few ms for 100k)
        var m = mat(), r = canvasRect();
        var key = m.v.join(",") + "|" + m.p.join(",") + "|" + m.m.join(",") + "|" + r.width + "x" + r.height;
        if (key === cacheKey) return r;
        cache = cache || new Float32Array(N * 3);
        wMin = 1e9; wMax = -1e9;
        for (var p = 0; p < N; p++) {
          var q = project(m, P.x[p], P.y[p], P.z[p]);
          cache[p * 3] = q.x * r.width; cache[p * 3 + 1] = q.y * r.height; cache[p * 3 + 2] = q.w;
          if (q.w > 0) { if (q.w < wMin) wMin = q.w; if (q.w > wMax) wMax = q.w; }
        }
        cacheKey = key;
        return r;
      }
      // Nearest dot to the cursor in screen space.  Dots with a spectrogram get a small head start, and
      // among dots at almost the same spot the one nearer the camera wins (it is the one drawn on top).
      function nearest(clientX, clientY) {
        var r = projectAll(), mx = clientX - r.left, my = clientY - r.top, best = null, bestS = 1e9, span = (wMax - wMin) || 1;
        for (var i = 0; i < N; i++) {
          var w = cache[i * 3 + 2]; if (w <= 0) continue;                      // behind the camera
          var dx = cache[i * 3] - mx; if (dx > PICK_PX + 6 || dx < -PICK_PX - 6) continue;
          var dy = cache[i * 3 + 1] - my; if (dy > PICK_PX + 6 || dy < -PICK_PX - 6) continue;
          var d = Math.hypot(dx, dy);
          if (d > PICK_PX + (slot[i] >= 0 ? VERIFIED_BONUS_PX : 0)) continue;
          var s = d - (slot[i] >= 0 ? VERIFIED_BONUS_PX : 0) + DEPTH_PX * (w - wMin) / span;
          if (s < bestS) { bestS = s; best = i; }
        }
        return best;
      }
      /*@test*/
      window.__umapNearest = nearest;
      window.__umapCoords = function (p) { return [P.x[p], P.y[p], P.z[p]]; };
      window.__umapScreen = function (p) { var r = projectAll(); return {x: r.left + cache[p * 3], y: r.top + cache[p * 3 + 1], w: cache[p * 3 + 2]}; };
      window.__umapCandidates = function (cx, cy, rad) {
        var r = projectAll(), out = []; for (var i = 0; i < N; i++) { if (cache[i*3+2] <= 0) continue;
          if (Math.hypot(r.left + cache[i*3] - cx, r.top + cache[i*3+1] - cy) <= rad) out.push(i); } return out; };
      /*@/test*/
      var downPt = null, hoverRaf = 0, hoverPt = null;
      gd.addEventListener("pointerdown", function (ev) { downPt = {x: ev.clientX, y: ev.clientY}; }, true);
      gd.addEventListener("pointerup", function (ev) {
        var d = downPt; downPt = null;
        if (!d || Math.hypot(ev.clientX - d.x, ev.clientY - d.y) >= 5) return;      // it was a drag (rotate)
        var p = nearest(ev.clientX, ev.clientY);
        if (p !== null) select(p);
      }, true);
      gd.addEventListener("pointermove", function (ev) {
        if (ev.buttons) return;
        var x = ev.clientX, y = ev.clientY;
        if (hoverRaf) return;
        hoverRaf = requestAnimationFrame(function () {
          hoverRaf = 0; hoverPt = nearest(x, y); gd.style.cursor = hoverPt === null ? "" : "pointer";
        });
      }, true);
      gd.on("plotly_relayout", tools.onRelayout);
      gd.on("plotly_relayouting", tools.onRelayout);
      tools.refresh();
    });
    return {gd: gd, tools: tools, select: select, stemOf: stemOf};
  }

  /* ---- tabs and keyboard ---- */
  function showTab(name) {
    if (name === current) return;
    [A, B].forEach(function (v) { if (v) v.tools.stopSpin(); });
    current = name;
    document.querySelectorAll("#umap-modes [data-layout]").forEach(function (b) {
      var l = b.getAttribute("data-layout");
      b.classList.toggle("on", name === "click" ? l === "click" : l === A.mode());
    });
    document.getElementById("umap-panel-overview").style.display = name === "overview" ? "" : "none";
    document.getElementById("umap-panel-click").style.display = name === "click" ? "" : "none";
    if (name === "click" && !B) { B = initClick(); /*@test*/ window.__umapClick = B; /*@/test*/ }
    else Plotly.Plots.resize((name === "click" ? B : A).gd);
  }
  document.addEventListener("keydown", function (ev) {
    var t = ev.target, v = current === "click" ? B : A;
    if (!v || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (t && (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type !== "range"))) return;
    if (ev.key === "+" || ev.key === "=") { v.tools.zoomBy(1.25); ev.preventDefault(); }
    else if (ev.key === "-" || ev.key === "_") { v.tools.zoomBy(1 / 1.25); ev.preventDefault(); }
    else if (ev.key === "0") { v.tools.requestZoom(1); ev.preventDefault(); }
  });

  A = initOverview();
})();
"""


def build_fragment(packed: dict, test_hooks: bool = False) -> str:
    js = (RENDER_JS.replace("__DATA__", json.dumps(packed, separators=(",", ":")))
          .replace("__OT__", OT_COLOR).replace("__CD__", CD_COLOR))
    # Test hooks (window.__umap*) are only kept for local test builds, never in the published page.
    js = js.replace("/*@test*/", "").replace("/*@/test*/", "") if test_hooks else re.sub(r"/\*@test\*/.*?/\*@/test\*/", "", js, flags=re.S)
    return build_html() + '<script type="text/javascript">' + js + "</script>\n"


# --------------------------------------------------------------------------- page splice
BEGIN_MARK, END_MARK = "<!-- UMAP-INTERACTIVE-BEGIN -->", "<!-- UMAP-INTERACTIVE-END -->"


VIEWPORT_TAG = '<meta name="viewport" content="width=device-width, initial-scale=1" />'


def ensure_viewport(head: str) -> str:
    """Without this tag, mobile browsers render the page at a virtual desktop
    width (~980px) and scale the whole thing down, making every control and
    label tiny regardless of any responsive CSS -- this is the single most
    important fix for the page working on a phone."""
    if "name=\"viewport\"" in head:
        return re.sub(r'<meta name="viewport"[^>]*/?>', VIEWPORT_TAG, head, count=1)
    return head.replace('<meta charset="utf-8" />', f'<meta charset="utf-8" />\n  {VIEWPORT_TAG}', 1)


def ensure_title(text: str, title: str) -> str:
    head_end = text.index("</head>")
    head = ensure_viewport(text[:head_end])
    tag = f"<title>{title}</title>"
    if "<title" in head:
        head = re.sub(r"<title>.*?</title>", tag, head, count=1, flags=re.S)
    else:
        head = head.replace('<meta charset="utf-8" />', f'<meta charset="utf-8" />\n  {tag}', 1)
        if tag not in head:
            raise SystemExit("Could not place <title> in <head>")
    return head + text[head_end:]


def splice_into_page(page: Path, fragment: str, title: str) -> None:
    text = ensure_title(page.read_text(encoding="utf-8"), title)
    block = f"{BEGIN_MARK}\n{fragment}{END_MARK}\n"
    if BEGIN_MARK in text:
        start = text.index(BEGIN_MARK)
        end = text.index(END_MARK, start) + len(END_MARK) + 1
    else:
        marker = text.rfind('class="plotly-graph-div"')
        if marker < 0:
            raise SystemExit(f"No existing plotly-graph-div found in {page}")
        start = text.rfind("<div id=", 0, marker)
        end = text.rfind("</div>", 0, text.rfind("</body>"))
    if not (0 < start < end):
        raise SystemExit("Could not locate the figure boundaries safely; page left unmodified.")
    page.write_text(text[:start] + block + text[end:], encoding="utf-8")
    print(f"Page: replaced {end - start:,} chars with {len(block):,} chars; <title> = {title!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", type=Path, required=True, help="Plotly project page to update in place")
    ap.add_argument("--atlas-dir", type=Path, default=None, help=f"tile folder (default: <page dir>/{ATLAS_DIR_NAME})")
    ap.add_argument("--webp-quality", type=int, default=40)
    ap.add_argument("--title", default=PAGE_TITLE)
    ap.add_argument("--test-hooks", action="store_true", help="keep the window.__umap* testing hooks (local test builds only)")
    args = ap.parse_args()
    atlas_dir = args.atlas_dir or args.page.parent / ATLAS_DIR_NAME
    if atlas_dir.name != ATLAS_DIR_NAME:
        raise SystemExit(f"--atlas-dir must be named {ATLAS_DIR_NAME!r} (the page references it by that relative name)")

    data = load()
    atlas = atlas_mod.build(atlas_dir, args.webp_quality)
    for k, (xyz, cd) in (("train", data["train"]), ("eval", data["eval"])):
        print(f"{k}: {len(xyz):,} points ({int(cd.sum()):,} CD / {int((~cd).sum()):,} OT)")
    print(f"spectrograms: {atlas['n_slots']:,} images in {atlas['n_tiles']} tiles ({atlas['total_bytes'] / 1e6:.1f} MB), "
          f"covering {int((atlas['row_slots'] >= 0).sum()):,} of {len(atlas['row_slots']):,} training rows")
    splice_into_page(args.page, build_fragment(pack_data(data, atlas), args.test_hooks), args.title)


if __name__ == "__main__":
    main()
