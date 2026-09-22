"""
Interactive 3D/2D scatter GUI for latent embeddings.

Python equivalent of MATLAB Scatterplot_GUI.m / simple_scatter3_classic.m,
extended with click-to-inspect: selecting a point loads its spectrogram
(from the per-detection .mat database) and offers a "Play audio" button that
extracts + plays the corresponding clip from the source 24-hour .WAV file.
"""

import os
import platform
import subprocess
import tempfile
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Backend selection — try interactive backends in order of preference.
# matplotlib.use() is lazy, so actually import the backend module to detect
# missing dependencies (e.g. Tkinter) before committing to it.
# ---------------------------------------------------------------------------
def _select_backend():
    import importlib
    import matplotlib

    candidates = ["MacOSX", "Qt5Agg", "Qt6Agg", "TkAgg", "WXAgg"] \
        if platform.system() == "Darwin" \
        else ["Qt5Agg", "Qt6Agg", "TkAgg", "WXAgg", "MacOSX"]

    for backend in candidates:
        try:
            module_name = matplotlib.backend_registry.resolve_backend(backend)[0] \
                if hasattr(matplotlib, "backend_registry") else f"matplotlib.backends.backend_{backend.lower()}"
            importlib.import_module(module_name)
            matplotlib.use(backend, force=True)
            return backend
        except Exception:
            continue
    return None

_select_backend()

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# ---------------------------------------------------------------------------
# Colour map helpers
# ---------------------------------------------------------------------------
_CMAP = "tab10"


def _make_colormap(labels: np.ndarray):
    unique = np.unique(labels[np.isfinite(labels)])
    norm = plt.Normalize(vmin=unique.min(), vmax=unique.max())
    cmap = plt.get_cmap(_CMAP, max(len(unique), 2))
    colors = cmap(norm(labels))
    return colors, norm, cmap


def _play_wav_file(path: str) -> None:
    """Play a .wav file using the OS's native audio player (non-blocking)."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["afplay", path])
        elif system == "Linux":
            subprocess.Popen(["aplay", path])
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            print(f"Don't know how to play audio on {system}; clip saved to {path}")
    except FileNotFoundError:
        print(f"No audio player found for {system}; clip saved to {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def launch_gui(
    X: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Latent Embedding Viewer",
    filenames: Optional[list] = None,
    color_fields: Optional[dict] = None,
    image_folder: Optional[str] = None,
    wav_locator=None,
    pad_before: float = 2.0,
    pad_after: float = 3.0,
):
    """
    Launch an interactive scatter GUI.

    Parameters
    ----------
    X : (N, 2) or (N, 3) float array — embedding coordinates
    labels : (N,) float array — primary colour channel (e.g. call type)
    title : window title
    filenames : list of source filenames shown when a point is clicked
    color_fields : dict of {name: (N,) array} for additional colour channels
    image_folder : directory containing per-detection spectrogram .mat files
    wav_locator : a wav_locator.WavLocator instance, or None to disable audio
    pad_before, pad_after : seconds of audio to extract around a detection
    """
    if X.ndim != 2 or X.shape[1] not in (2, 3):
        raise ValueError(f"X must be (N,2) or (N,3), got shape {X.shape}")

    is_3d = X.shape[1] == 3
    n = X.shape[0]

    if labels is None:
        labels = np.zeros(n)

    # Build color-field menu
    all_color_fields: dict = {"type / labels": labels}
    if color_fields:
        all_color_fields.update(color_fields)
    color_names = list(all_color_fields.keys())

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 7.5), num=title)
    fig.patch.set_facecolor("#1e1e2e")

    # Main scatter axes (left ~58%)
    if is_3d:
        ax = fig.add_axes([0.02, 0.08, 0.56, 0.88], projection="3d")
        ax.set_facecolor("#1e1e2e")
    else:
        ax = fig.add_axes([0.02, 0.08, 0.56, 0.88])
        ax.set_facecolor("#1e1e2e")

    right = 0.63
    sw = 0.34  # right-column width

    # -----------------------------------------------------------------------
    # State
    # -----------------------------------------------------------------------
    state = {
        "colors": _make_colormap(labels)[0],
        "normalize": False,
        "current_field": color_names[0],
        "selected_index": None,
        "last_clip": None,       # cached extract_clip() result
        "last_clip_path": None,  # temp .wav path
    }

    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    def _get_X():
        return X_norm if state["normalize"] else X

    def _scatter():
        ax.cla()
        Xp = _get_X()
        c = state["colors"]
        if is_3d:
            ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2],
                       s=4, c=c, alpha=0.4, linewidths=0, picker=5)
            ax.set_xlabel("dim 1", color="white", fontsize=8)
            ax.set_ylabel("dim 2", color="white", fontsize=8)
            ax.set_zlabel("dim 3", color="white", fontsize=8)
            _apply_limits_3d(Xp)
        else:
            ax.scatter(Xp[:, 0], Xp[:, 1],
                       s=4, c=c, alpha=0.4, linewidths=0, picker=5)
            ax.set_xlabel("dim 1", color="white", fontsize=8)
            ax.set_ylabel("dim 2", color="white", fontsize=8)
            _apply_limits_2d(Xp)

        ax.set_title(title, color="white", fontsize=10)
        for spine in getattr(ax, "spines", {}).values():
            spine.set_edgecolor("#555")
        ax.tick_params(colors="white", labelsize=7)
        fig.canvas.draw_idle()

    def _apply_limits_3d(Xp):
        lx, ly, lz = sld_xlim.val, sld_ylim.val, sld_zlim.val
        ax.set_xlim3d(-lx, lx)
        ax.set_ylim3d(-ly, ly)
        ax.set_zlim3d(-lz, lz)
        ax.view_init(elev=sld_el.val, azim=sld_az.val)

    def _apply_limits_2d(Xp):
        ax.set_xlim(-sld_xlim.val, sld_xlim.val)
        ax.set_ylim(-sld_ylim.val, sld_ylim.val)

    def _default_lim(Xp, col):
        return float(np.percentile(np.abs(Xp[:, col]), 98)) * 1.05

    # -----------------------------------------------------------------------
    # Sliders (top of right column)
    # -----------------------------------------------------------------------
    slider_kw = dict(color="#444466", track_color="#333355")

    def _make_slider(y, label, lo, hi, v0):
        ax_s = fig.add_axes([right, y, sw, 0.022], facecolor="#2a2a3a")
        s = Slider(ax_s, label, lo, hi, valinit=v0, **slider_kw)
        s.label.set_color("white")
        s.valtext.set_color("white")
        return s

    Xp0 = _get_X()
    sld_xlim = _make_slider(0.945, "X ±", 0.1, 20, _default_lim(Xp0, 0))
    sld_ylim = _make_slider(0.905, "Y ±", 0.1, 20, _default_lim(Xp0, 1))

    if is_3d:
        sld_zlim = _make_slider(0.865, "Z ±", 0.1, 20, _default_lim(Xp0, 2))
        sld_az   = _make_slider(0.800, "Azimuth", 0, 360, 45)
        sld_el   = _make_slider(0.760, "Elevation", -90, 90, -25)
    else:
        sld_zlim = _make_slider(0.865, "Z ±", 0.1, 20, 5)
        sld_az   = _make_slider(0.800, "Azimuth", 0, 360, 45)
        sld_el   = _make_slider(0.760, "Elevation", -90, 90, -25)
        sld_zlim.ax.set_visible(False)
        sld_az.ax.set_visible(False)
        sld_el.ax.set_visible(False)

    def _on_slider(val):
        Xp = _get_X()
        _apply_limits_3d(Xp) if is_3d else _apply_limits_2d(Xp)
        fig.canvas.draw_idle()

    for s in (sld_xlim, sld_ylim, sld_zlim, sld_az, sld_el):
        s.on_changed(_on_slider)

    # -----------------------------------------------------------------------
    # Colour field radio buttons
    # -----------------------------------------------------------------------
    n_colors = len(color_names)
    radio_height = min(0.028 * n_colors + 0.02, 0.16)
    ax_radio = fig.add_axes([right, 0.60, sw, radio_height], facecolor="#2a2a3a")
    radio = RadioButtons(ax_radio, color_names, activecolor="#8888ff")
    ax_radio.set_title("Colour by", color="white", fontsize=8)
    for lbl in radio.labels:
        lbl.set_color("white")
        lbl.set_fontsize(7.5)

    def _on_radio(label):
        arr = all_color_fields[label]
        state["colors"] = _make_colormap(arr)[0]
        state["current_field"] = label
        _scatter()

    radio.on_clicked(_on_radio)

    # -----------------------------------------------------------------------
    # Normalise / reset-view buttons
    # -----------------------------------------------------------------------
    ax_btn_norm = fig.add_axes([right, 0.555, sw * 0.48, 0.032])
    btn_norm = Button(ax_btn_norm, "Toggle normalise", color="#333355", hovercolor="#555588")
    btn_norm.label.set_color("white")
    btn_norm.label.set_fontsize(7.5)

    def _on_norm(event):
        state["normalize"] = not state["normalize"]
        btn_norm.label.set_text("Normalised ON" if state["normalize"] else "Toggle normalise")
        Xp = _get_X()
        sld_xlim.set_val(_default_lim(Xp, 0))
        sld_ylim.set_val(_default_lim(Xp, 1))
        if is_3d:
            sld_zlim.set_val(_default_lim(Xp, 2))
        _scatter()

    btn_norm.on_clicked(_on_norm)

    ax_btn_rst = fig.add_axes([right + sw * 0.52, 0.555, sw * 0.48, 0.032])
    btn_rst = Button(ax_btn_rst, "Reset view", color="#333355", hovercolor="#555588")
    btn_rst.label.set_color("white")
    btn_rst.label.set_fontsize(7.5)

    def _on_reset(event):
        Xp = _get_X()
        sld_xlim.set_val(_default_lim(Xp, 0))
        sld_ylim.set_val(_default_lim(Xp, 1))
        if is_3d:
            sld_zlim.set_val(_default_lim(Xp, 2))
            sld_az.set_val(45)
            sld_el.set_val(-25)
        _scatter()

    btn_rst.on_clicked(_on_reset)

    # -----------------------------------------------------------------------
    # Spectrogram thumbnail panel
    # -----------------------------------------------------------------------
    spec_ax = fig.add_axes([right, 0.25, sw, 0.27], facecolor="#111122")
    spec_ax.axis("off")
    spec_ax.set_title("Spectrogram", color="white", fontsize=8, loc="left")
    spec_ax.text(
        0.5, 0.5, "Click a point to load\nits spectrogram",
        color="#666688", fontsize=8, ha="center", va="center",
        transform=spec_ax.transAxes,
    )

    # -----------------------------------------------------------------------
    # Play-audio button
    # -----------------------------------------------------------------------
    ax_btn_play = fig.add_axes([right, 0.185, sw, 0.035])
    btn_play = Button(ax_btn_play, "▶ Play audio clip", color="#2a5540", hovercolor="#357a58")
    btn_play.label.set_color("white")
    btn_play.label.set_fontsize(8)
    btn_play.ax.set_visible(wav_locator is not None)

    def _on_play(event):
        clip = state["last_clip"]
        if clip is None:
            info_text.set_text("No audio clip loaded yet — click a point first.")
            fig.canvas.draw_idle()
            return
        if state["last_clip_path"] is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            wav_locator.save_clip_wav(clip, tmp.name)
            state["last_clip_path"] = tmp.name
        _play_wav_file(state["last_clip_path"])

    btn_play.on_clicked(_on_play)

    # -----------------------------------------------------------------------
    # Click-to-inspect: info text + spectrogram + audio clip prep
    # -----------------------------------------------------------------------
    info_ax = fig.add_axes([right, 0.02, sw, 0.15], facecolor="#111122")
    info_ax.axis("off")
    info_text = info_ax.text(0.02, 0.95, "Click a point to inspect",
                             color="#aaaacc", fontsize=7.5,
                             transform=info_ax.transAxes,
                             verticalalignment="top", wrap=True)

    def _update_spectrogram(detection_stem: str):
        spec_ax.cla()
        spec_ax.axis("off")
        spec_ax.set_title("Spectrogram", color="white", fontsize=8, loc="left")
        if not image_folder:
            spec_ax.text(0.5, 0.5, "No --image-folder configured",
                         color="#666688", fontsize=7.5, ha="center", va="center",
                         transform=spec_ax.transAxes, wrap=True)
            return
        try:
            from spectrogram_loader import load_detection_spectrogram
            result = load_detection_spectrogram(image_folder, detection_stem)
            spec_ax.imshow(result["gram"], aspect="auto", origin="lower", cmap="inferno")
            spec_ax.set_title(f"{result['gram_name']}", color="white", fontsize=8, loc="left")
        except Exception as exc:
            spec_ax.text(0.5, 0.5, f"Spectrogram unavailable:\n{str(exc)[:120]}",
                         color="#cc8888", fontsize=7, ha="center", va="center",
                         transform=spec_ax.transAxes, wrap=True)

    def _prepare_audio(detection_stem: str) -> str:
        if wav_locator is None:
            return "Audio disabled (no --wav-base-dir configured)"
        try:
            clip = wav_locator.extract_clip(detection_stem, pad_before=pad_before, pad_after=pad_after)
            state["last_clip"] = clip
            state["last_clip_path"] = None
            dur = clip["clip_end_sec"] - clip["clip_start_sec"]
            return f"Audio ready: {dur:.1f}s from {os.path.basename(clip['wav_path'])}"
        except Exception as exc:
            state["last_clip"] = None
            state["last_clip_path"] = None
            return f"Audio unavailable: {str(exc)[:150]}"

    def _on_pick(event):
        ind = event.ind[0]
        state["selected_index"] = ind
        Xp = _get_X()
        lines = [f"Index : {ind}"]
        lines.append(f"Coords: ({Xp[ind, 0]:.3f}, {Xp[ind, 1]:.3f}"
                     + (f", {Xp[ind, 2]:.3f})" if is_3d else ")"))
        field = state["current_field"]
        val = all_color_fields[field][ind]
        lines.append(f"{field[:28]}: {val:.4g}")

        detection_stem = None
        if filenames is not None and ind < len(filenames):
            detection_stem = str(filenames[ind])
            lines.append(f"File  : {detection_stem[-50:]}")

        if detection_stem is not None:
            _update_spectrogram(detection_stem)
            lines.append(_prepare_audio(detection_stem))
        else:
            lines.append("No filename metadata available for this point.")

        info_text.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", _on_pick)

    # -----------------------------------------------------------------------
    # First draw
    # -----------------------------------------------------------------------
    _scatter()
    plt.show()

