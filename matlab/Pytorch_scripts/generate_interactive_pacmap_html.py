#!/usr/bin/env python3
"""
Generate Interactive 3D HTML Visualization for PaCMAP Embeddings

Creates an accessible, interactive 3D visualization using Plotly.
Users can rotate, zoom, pan, and hover over points to explore the latent space.

USAGE:
    python3 generate_interactive_pacmap_html.py --dir <model_directory>
    
EXAMPLE:
    python3 generate_interactive_pacmap_html.py --dir LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.dir
"""

import os
import sys
import argparse
import numpy as np
from scipy.io import loadmat

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("ERROR: plotly or pandas not installed. Run: pip install plotly pandas")
    sys.exit(1)


def generate_interactive_3d_html(directory):
    """Generate interactive 3D PaCMAP HTML visualization."""
    
    # Load PaCMAP 3D embeddings
    pacmap_path = os.path.join(directory, 'PaCMAP', 'pacmap_embeddings_3d.mat')
    
    if not os.path.exists(pacmap_path):
        print(f"ERROR: PaCMAP embeddings not found: {pacmap_path}")
        sys.exit(1)
    
    print(f"Loading PaCMAP 3D embeddings from: {pacmap_path}")
    data = loadmat(pacmap_path)
    
    embedding_3d = data['pacmap_embeddings_3d']
    clusters = data.get('clusters', None)
    
    # Flatten clusters if needed
    if clusters is not None and clusters.ndim > 1:
        clusters = clusters.flatten()
    
    optimal_k = data.get('optimal_k', np.array([[2]]))[0, 0] if 'optimal_k' in data else 2
    dataset_label = data.get('dataset_label', 'Unknown')
    
    print(f"  Loaded {embedding_3d.shape[0]:,} samples")
    print(f"  Clusters: {int(optimal_k)}")
    print(f"  Dataset: {dataset_label}")
    
    # Create DataFrame for plotly
    df = pd.DataFrame({
        'PaCMAP_1': embedding_3d[:, 0],
        'PaCMAP_2': embedding_3d[:, 1],
        'PaCMAP_3': embedding_3d[:, 2],
        'Cluster': clusters if clusters is not None else np.zeros(len(embedding_3d)),
        'Index': np.arange(len(embedding_3d))
    })
    
    # Convert cluster to categorical for better coloring
    df['Cluster'] = df['Cluster'].astype(int).astype(str)
    
    print("\nGenerating interactive 3D visualization...")
    
    # Create interactive 3D scatter plot
    fig = px.scatter_3d(
        df,
        x='PaCMAP_1',
        y='PaCMAP_2',
        z='PaCMAP_3',
        color='Cluster',
        hover_data={'Index': True, 'Cluster': True, 
                   'PaCMAP_1': ':.3f', 'PaCMAP_2': ':.3f', 'PaCMAP_3': ':.3f'},
        title=f'Interactive 3D PaCMAP Latent Space - {dataset_label} (k={int(optimal_k)})',
        opacity=0.7,
        height=900,
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Update marker size and layout
    fig.update_traces(marker=dict(size=2, line=dict(width=0)))
    
    fig.update_layout(
        scene=dict(
            xaxis_title='PaCMAP Dimension 1',
            yaxis_title='PaCMAP Dimension 2',
            zaxis_title='PaCMAP Dimension 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            xaxis=dict(showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            yaxis=dict(showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            zaxis=dict(showbackground=True, backgroundcolor='rgb(230, 230, 230)')
        ),
        font=dict(size=14),
        legend=dict(
            title=dict(text='Cluster'),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='closest'
    )
    
    # Add descriptive text
    fig.add_annotation(
        text=(
            f"<b>Dataset:</b> {dataset_label}<br>"
            f"<b>Samples:</b> {embedding_3d.shape[0]:,}<br>"
            f"<b>Clusters:</b> {int(optimal_k)}<br>"
            f"<b>Method:</b> PaCMAP (Pairwise Controlled Manifold Approximation)<br>"
            "<i>Hover over points for details. Click and drag to rotate.</i>"
        ),
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=11)
    )
    
    # Save HTML
    output_path = os.path.join(directory, 'PaCMAP', 'pacmap_3d_interactive.html')
    fig.write_html(
        output_path,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['hoverclosest', 'hovercompare'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'pacmap_3d',
                'height': 1200,
                'width': 1200,
                'scale': 2
            }
        }
    )
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"\n✓ Saved interactive 3D PaCMAP HTML")
    print(f"  Path: {output_path}")
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Samples: {len(embedding_3d):,}")
    print(f"  Clusters: {int(optimal_k)}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive 3D HTML visualization for PaCMAP embeddings'
    )
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='Path to model directory containing PaCMAP embeddings'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.exists(args.dir):
        print(f"ERROR: Directory not found: {args.dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("Interactive 3D PaCMAP HTML Generation")
    print("=" * 70)
    print(f"Model directory: {args.dir}")
    print("=" * 70)
    
    output_path = generate_interactive_3d_html(args.dir)
    
    print("\n" + "=" * 70)
    print("✓ Interactive HTML generation complete!")
    print("=" * 70)
    print(f"\nTo view the visualization, open in your browser:")
    print(f"  {output_path}")
    print("\nOr run:")
    print(f"  open {output_path}")
    print("\n")


if __name__ == '__main__':
    main()
