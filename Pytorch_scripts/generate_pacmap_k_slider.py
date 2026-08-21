#!/usr/bin/env python3
"""
Generate Interactive 3D PaCMAP HTML with k=2..5 Cluster Slider

Creates a fully self-contained interactive HTML file using Plotly where:
  - The 3D PaCMAP scatter plot is rotatable/zoomable
  - A slider at the bottom switches cluster colouring between k=2, 3, 4, 5
  - KMeans is re-run on the full 32-D latent vectors for each k value
  - The file is single-file HTML (no server needed to view locally)

USAGE:
    python3 generate_pacmap_k_slider.py --dir <model_directory>

EXAMPLE:
    python3 generate_pacmap_k_slider.py --dir LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.dir
"""

import os
import sys
import argparse
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans

try:
    import plotly.graph_objects as go
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("ERROR: plotly or pandas not installed.  Run: pip install plotly pandas")
    sys.exit(1)

K_VALUES = [2, 3, 4, 5]

# One visually distinct colour per cluster (up to 5 clusters max)
CLUSTER_COLOURS = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#FF7F00",  # orange
    "#984EA3",  # purple
]


def load_embeddings(directory):
    """Load PaCMAP 3D and latent embeddings from model directory."""
    pacmap_path = os.path.join(directory, "PaCMAP", "pacmap_embeddings_3d.mat")
    latent_path = os.path.join(directory, "MATLAB", "latent_embeddings.mat")

    if not os.path.exists(pacmap_path):
        print(f"ERROR: PaCMAP embeddings not found: {pacmap_path}")
        sys.exit(1)
    if not os.path.exists(latent_path):
        print(f"ERROR: Latent embeddings not found: {latent_path}")
        sys.exit(1)

    print(f"Loading PaCMAP 3D: {pacmap_path}")
    pdata = loadmat(pacmap_path)
    emb3d = pdata["pacmap_embeddings_3d"].astype(np.float32)

    print(f"Loading latent embeddings: {latent_path}")
    ldata = loadmat(latent_path)
    latent = ldata["latent_embeddings"].astype(np.float32)

    dataset_label = str(ldata.get("dataset_label", "Unknown"))
    print(f"  PaCMAP shape : {emb3d.shape}")
    print(f"  Latent shape : {latent.shape}")
    return emb3d, latent, dataset_label


def run_kmeans_for_all_k(latent, k_values):
    """Run KMeans on latent vectors for each k; returns dict k -> labels array."""
    labels_by_k = {}
    for k in k_values:
        print(f"  KMeans k={k} …", end=" ", flush=True)
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(latent)
        labels_by_k[k] = labels
        print("done")
    return labels_by_k


def build_figure(emb3d, labels_by_k, dataset_label):
    """Build Plotly figure with one set of cluster traces per k, plus a slider."""

    fig = go.Figure()

    # We'll add one Scatter3d trace per (k, cluster_id) combination.
    # Visibility will be toggled by the slider.
    trace_visibility_by_k = {k: [] for k in K_VALUES}

    for k in K_VALUES:
        labels = labels_by_k[k]
        for cluster_id in range(k):
            mask = labels == cluster_id
            colour = CLUSTER_COLOURS[cluster_id % len(CLUSTER_COLOURS)]
            trace = go.Scatter3d(
                x=emb3d[mask, 0],
                y=emb3d[mask, 1],
                z=emb3d[mask, 2],
                mode="markers",
                name=f"k={k}  Cluster {cluster_id}",
                legendgroup=f"k{k}",
                legendgrouptitle_text=f"k = {k}" if cluster_id == 0 else None,
                marker=dict(
                    size=2,
                    color=colour,
                    opacity=0.7,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    f"Cluster {cluster_id}<br>"
                    "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>"
                ),
                visible=(k == K_VALUES[0]),  # Only show k=2 initially
            )
            fig.add_trace(trace)

    # Build slider steps: each step shows only traces belonging to that k
    total_traces = sum(k for k in K_VALUES)  # total number of traces
    trace_index = 0
    slider_steps = []

    for k in K_VALUES:
        # Build a visibility list across ALL traces
        visibility = []
        idx = 0
        for kk in K_VALUES:
            for _ in range(kk):
                visibility.append(idx >= trace_index and idx < trace_index + k)
                idx += 1

        # Recalculate correctly
        visibility = []
        running = 0
        for kk in K_VALUES:
            for _ in range(kk):
                visibility.append(kk == k)
            running += kk

        step = dict(
            method="update",
            args=[
                {"visible": visibility},
                {"title": (
                    f"Interactive 3D PaCMAP — {dataset_label} | "
                    f"k = {k} clusters | {emb3d.shape[0]:,} samples"
                )},
            ],
            label=str(k),
        )
        slider_steps.append(step)
        trace_index += k

    sliders = [dict(
        active=0,
        currentvalue=dict(
            prefix="Number of clusters  k = ",
            font=dict(size=16, color="#333"),
        ),
        pad=dict(t=50, b=10),
        steps=slider_steps,
        tickcolor="#666",
        font=dict(size=14),
    )]

    fig.update_layout(
        title=dict(
            text=(
                f"Interactive 3D PaCMAP — {dataset_label} | "
                f"k = {K_VALUES[0]} clusters | {emb3d.shape[0]:,} samples"
            ),
            font=dict(size=16),
        ),
        sliders=sliders,
        scene=dict(
            xaxis_title="PaCMAP Dim 1",
            yaxis_title="PaCMAP Dim 2",
            zaxis_title="PaCMAP Dim 3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            xaxis=dict(showbackground=True, backgroundcolor="rgb(235,235,235)"),
            yaxis=dict(showbackground=True, backgroundcolor="rgb(235,235,235)"),
            zaxis=dict(showbackground=True, backgroundcolor="rgb(235,235,235)"),
        ),
        legend=dict(
            title="Clusters",
            itemsizing="constant",
            tracegroupgap=6,
        ),
        margin=dict(l=0, r=0, t=60, b=120),
        height=850,
        hovermode="closest",
    )

    # Informational annotation
    fig.add_annotation(
        text=(
            "<b>Controls:</b> drag to rotate · scroll to zoom · "
            "drag slider to change k · click legend to toggle clusters<br>"
            "<b>Method:</b> PaCMAP (Pairwise Controlled Manifold Approximation) "
            "· <b>Colours:</b> KMeans on 32-D latent space"
        ),
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#aaa",
        borderwidth=1,
        font=dict(size=11),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-contained interactive PaCMAP HTML with k=2..5 slider"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to model directory",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"ERROR: Directory not found: {args.dir}")
        sys.exit(1)

    print("=" * 70)
    print("Interactive PaCMAP k-Slider HTML Generation")
    print("=" * 70)
    print(f"Directory : {args.dir}")
    print(f"k values  : {K_VALUES}")
    print("=" * 70)

    emb3d, latent, dataset_label = load_embeddings(args.dir)

    print("\nRunning KMeans for each k value on latent vectors …")
    labels_by_k = run_kmeans_for_all_k(latent, K_VALUES)

    print("\nBuilding Plotly figure …")
    fig = build_figure(emb3d, labels_by_k, dataset_label)

    output_path = os.path.join(args.dir, "PaCMAP", "pacmap_3d_k_slider.html")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Writing HTML to: {output_path}")
    fig.write_html(
        output_path,
        include_plotlyjs="cdn",   # loads plotly from CDN — keeps file small
        full_html=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "pacmap_3d_k_slider",
                "height": 1200,
                "width": 1400,
                "scale": 2,
            },
        },
    )

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n{'=' * 70}")
    print(f"✓ Interactive HTML saved  ({size_mb:.1f} MB)")
    print(f"{'=' * 70}")
    print(f"\nOpen in browser:  open \"{output_path}\"")
    print(f"Network URL (after launching server):  http://132.239.169.101:6007/")
    print()


if __name__ == "__main__":
    main()
