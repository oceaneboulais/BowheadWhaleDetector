"""Build the interactive (3D, spinnable) + matching 2D companion version of the
"UMAP display of revised training dataset, for CDs only, colored by duration /
peak frequency / spectral slope" figure (paper fig:UMAP_different_labels3,
currently the static ``python_UMAP_color_peak_frequency.png``).

Data source
-----------
``paper/FinalDatabases_/latent_embeddings_3d_train_MATLAB_Angel_Final_26Aug2026.mat``
is the authoritative, full 100,000-sample training UMAP export (confirmed:
iscall counts match the published Table "Reviewed dataset changes" almost
exactly). It contains ``umap_embeddings_3d`` (100000,3), and per-sample
``features.iscall`` (CD/non-CD), ``features.duration1`` (s), ``features.fpeak``
(peak frequency, Hz), ``features.fmin``/``features.fmax`` (Hz).

IMPORTANT caveat on spectral slope: this file does NOT store a true per-sample
FM slope (that requires fitting the frequency-vs-time track from the raw
SNR-spectrogram pixels). The raw per-detection .mat files backing this
specific 100k export could not be located across the drives/directories
checked (R3D_2024_1's *_100K_ADG_*_centered dirs, local ThodeLab mirrors), so
this script instead derives an explicitly-labeled **unsigned proxy**,
``spectral_bandwidth_rate_hz_per_s = (fmax - fmin) / duration1`` -- the average
rate of frequency change implied by the signal's total bandwidth and
duration. This is NOT the same as a true signed FM slope (it cannot
distinguish an upsweep from a downsweep), and is labeled as such everywhere
it is plotted. If the true raw spectrograms become available, replace this
with ``bowhead.benchmark.generate_umap_paper_figures.spectrogram_features``'s
per-pixel linear-fit slope instead.

Standardization: coordinates are centered/scaled using the FULL 100,000-point
training set's own mean/std (matching ``plot_training_eval_grid`` and the
paper's own description of this normalization), so the resulting axis range
is naturally close to [-3, 3] in each dimension.

2D view: reuses the exact matplotlib ``elev=20, azim=60`` camera used for
the existing call-type plot ("Figure 14", ``plot_categorical_call_type``),
via ``mpl_toolkits.mplot3d.proj3d``, so the 2D projection's point arrangement
is directly visually comparable to that figure.

Run:
    PYTHONPATH=. .venv_ae/bin/python -m bowhead.benchmark.build_interactive_umap_cd_features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio

DEFAULT_MAT = Path("paper/FinalDatabases_/latent_embeddings_3d_train_MATLAB_Angel_Final_26Aug2026.mat")
AXIS_RANGE = (-3.0, 3.0)
VIEW_ELEV, VIEW_AZIM = 20.0, 60.0


def _load(mat_path: Path) -> dict:
    d = sio.loadmat(str(mat_path), simplify_cells=True)
    feats = d["features"]
    coords = np.asarray(d["umap_embeddings_3d"], dtype=np.float64)
    # Center/scale using the full 100k set, matching the paper's own convention.
    coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)

    iscall = np.asarray(feats["iscall"]).astype(bool)
    duration1 = np.asarray(feats["duration1"], dtype=np.float64)
    fpeak = np.asarray(feats["fpeak"], dtype=np.float64)
    fmin = np.asarray(feats["fmin"], dtype=np.float64)
    fmax = np.asarray(feats["fmax"], dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        bandwidth_rate = np.where(duration1 > 0, (fmax - fmin) / duration1, np.nan)

    cd = iscall
    return {
        "coords": coords[cd],
        "duration_s": duration1[cd],
        "peak_frequency_hz": fpeak[cd],
        "spectral_bandwidth_rate_hz_per_s": bandwidth_rate[cd],
        "n_cd": int(cd.sum()),
        "n_total": len(iscall),
    }


def _project_2d(coords: np.ndarray, elev: float, azim: float) -> np.ndarray:
    """Project standardized 3D coords to 2D using matplotlib's exact
    elev/azim camera transform, so the 2D arrangement matches the existing
    static call-type figure's view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(*AXIS_RANGE)
    ax.set_ylim(*AXIS_RANGE)
    ax.set_zlim(*AXIS_RANGE)
    ax.view_init(elev=elev, azim=azim)
    M = ax.get_proj()
    xs, ys, _ = proj3d.proj_transform(coords[:, 0], coords[:, 1], coords[:, 2], M)
    plt.close(fig)
    return np.column_stack([xs, ys])


def _build_html(data: dict, out_path: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    panels = [
        ("duration_s", "Duration (s)", "viridis"),
        ("peak_frequency_hz", "Peak frequency (Hz)", "viridis"),
        ("spectral_bandwidth_rate_hz_per_s",
         "Spectral bandwidth rate |\u0394f|/duration (Hz/s) -- unsigned proxy, not true FM slope",
         "viridis"),
    ]
    coords = data["coords"]
    marker_size = 4.5  # larger than the paper's static s=3.5 matplotlib marker

    fig3d = make_subplots(
        rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
        subplot_titles=[f"({chr(97+i)}) {label}" for i, (_, label, _) in enumerate(panels)],
    )
    coords2d = _project_2d(coords, VIEW_ELEV, VIEW_AZIM)
    fig2d = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"({chr(97+i)}) {label}" for i, (_, label, _) in enumerate(panels)],
    )

    for i, (key, label, cmap) in enumerate(panels):
        values = data[key]
        fig3d.add_trace(
            go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode="markers",
                marker=dict(size=marker_size, color=values, colorscale=cmap,
                            colorbar=dict(title=label, len=0.6) if i == 2 else None,
                            showscale=(i == 2), opacity=0.7),
                text=[f"{label}: {v:.3g}" for v in values],
                hoverinfo="text",
                name=label,
            ),
            row=1, col=i + 1,
        )
        fig2d.add_trace(
            go.Scattergl(
                x=coords2d[:, 0], y=coords2d[:, 1],
                mode="markers",
                marker=dict(size=marker_size, color=values, colorscale=cmap,
                            colorbar=dict(title=label, len=0.6) if i == 2 else None,
                            showscale=(i == 2), opacity=0.7),
                text=[f"{label}: {v:.3g}" for v in values],
                hoverinfo="text",
                name=label,
            ),
            row=1, col=i + 1,
        )

    scene_kwargs = dict(
        xaxis=dict(title="UMAP 1", range=list(AXIS_RANGE)),
        yaxis=dict(title="UMAP 2", range=list(AXIS_RANGE)),
        zaxis=dict(title="UMAP 3", range=list(AXIS_RANGE)),
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
    )
    fig3d.update_layout(
        title=(f"UMAP of revised training dataset (CDs only, n={data['n_cd']:,} "
               f"of {data['n_total']:,}): duration / peak frequency / spectral bandwidth rate"),
        scene=scene_kwargs, scene2=scene_kwargs, scene3=scene_kwargs,
        height=650, margin=dict(l=10, r=10, t=80, b=10),
    )
    fig2d.update_xaxes(title_text="UMAP projection 1 (elev=20, azim=60 view)")
    fig2d.update_yaxes(title_text="UMAP projection 2 (elev=20, azim=60 view)")
    fig2d.update_layout(
        title=(f"2D view matching Figure 14's camera angle (elev={VIEW_ELEV:g}, "
               f"azim={VIEW_AZIM:g}) -- CDs only, n={data['n_cd']:,}"),
        height=500, margin=dict(l=10, r=10, t=80, b=10),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("<html><head><meta charset='utf-8'></head><body>\n")
        f.write(fig3d.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write(fig2d.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>\n")
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    p.add_argument("--out", type=Path,
                   default=Path("paper/Figures/interactive_umap_cd_features.html"))
    args = p.parse_args()

    data = _load(args.mat)
    print(f"Loaded {data['n_cd']:,} CD samples out of {data['n_total']:,} total training samples.")
    _build_html(data, args.out)


if __name__ == "__main__":
    main()
