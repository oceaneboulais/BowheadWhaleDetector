#!/usr/bin/env python3
"""
Generate Interactive 5D PaCMAP Visualizations

Creates interactive visualizations for 5D PaCMAP embeddings using:
1. Parallel coordinates plot (all 5 dimensions simultaneously)
2. Multiple linked 3D scatter plots (different dimension combinations)
3. Interactive dimension selector

USAGE:
    python3 generate_5d_pacmap_visualization.py --dir <model_directory>
    
EXAMPLE:
    python3 generate_5d_pacmap_visualization.py --dir LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.dir
"""

import os
import sys
import argparse
import numpy as np
from scipy.io import loadmat

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("ERROR: plotly or pandas not installed. Run: pip install plotly pandas")
    sys.exit(1)


def generate_parallel_coordinates(embedding_5d, clusters, optimal_k, dataset_label, output_path, max_samples=10000):
    """Generate parallel coordinates plot for 5D embeddings."""
    
    print(f"\nGenerating parallel coordinates plot...")
    print(f"  Processing {len(embedding_5d)} samples...")
    
    # Subsample for performance
    n_samples = len(embedding_5d)
    if n_samples > max_samples:
        print(f"  Subsampling from {n_samples} to {max_samples} for performance...")
        indices = np.random.choice(n_samples, max_samples, replace=False)
        indices = np.sort(indices)
    else:
        indices = np.arange(n_samples)
        max_samples = n_samples
    
    embedding_subset = embedding_5d[indices]
    clusters_subset = clusters[indices] if clusters is not None else np.zeros(len(indices))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Dim 1': embedding_subset[:, 0],
        'Dim 2': embedding_subset[:, 1],
        'Dim 3': embedding_subset[:, 2],
        'Dim 4': embedding_subset[:, 3],
        'Dim 5': embedding_subset[:, 4],
        'Cluster': clusters_subset.astype(int)
    })
    
    # Create parallel coordinates plot
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df['Cluster'],
                colorscale='Viridis',
                showscale=True,
                cmin=df['Cluster'].min(),
                cmax=df['Cluster'].max()
            ),
            dimensions=[
                dict(label='Dimension 1', values=df['Dim 1']),
                dict(label='Dimension 2', values=df['Dim 2']),
                dict(label='Dimension 3', values=df['Dim 3']),
                dict(label='Dimension 4', values=df['Dim 4']),
                dict(label='Dimension 5', values=df['Dim 5']),
                dict(label='Cluster', values=df['Cluster'], 
                     tickvals=list(range(int(df['Cluster'].min()), int(df['Cluster'].max())+1)))
            ]
        )
    )
    
    fig.update_layout(
        title=dict(
            text=f'5D PaCMAP Parallel Coordinates<br><sub>Dataset: {dataset_label} | Clusters: {optimal_k} | Samples: {max_samples:,}</sub>',
            x=0.5,
            xanchor='center'
        ),
        font=dict(size=14),
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    print(f"  Saving HTML...")
    fig.write_html(
        output_path,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'pacmap_5d_parallel_coords',
                'height': 800,
                'width': 1400,
                'scale': 2
            }
        }
    )
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  ✓ Saved: {output_path}")
    print(f"    File size: {file_size:.1f} MB")
    print(f"    Samples: {max_samples:,}")


def generate_linked_3d_views(embedding_5d, clusters, optimal_k, dataset_label, output_path, max_samples=5000):
    """Generate multiple linked 3D scatter plots showing different dimension combinations."""
    
    print(f"\nGenerating linked 3D views...")
    print(f"  Processing {len(embedding_5d)} samples...")
    
    # Subsample for performance
    n_samples = len(embedding_5d)
    if n_samples > max_samples:
        print(f"  Subsampling from {n_samples} to {max_samples} for performance...")
        indices = np.random.choice(n_samples, max_samples, replace=False)
        indices = np.sort(indices)
    else:
        indices = np.arange(n_samples)
        max_samples = n_samples
    
    embedding_subset = embedding_5d[indices]
    clusters_subset = clusters[indices] if clusters is not None else np.zeros(len(indices))
    
    # Create subplots: 3 different 3D views
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Dims 1-2-3', 'Dims 3-4-5', 'Dims 1-3-5'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )
    
    # Color mapping by cluster
    colors = clusters_subset.astype(int)
    
    # View 1: Dimensions 1-2-3
    unique_clusters = sorted(np.unique(colors))
    for cluster_id in unique_clusters:
        mask = colors == cluster_id
        fig.add_trace(
            go.Scatter3d(
                x=embedding_subset[mask, 0],
                y=embedding_subset[mask, 1],
                z=embedding_subset[mask, 2],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(size=2, opacity=0.7),
                showlegend=bool(cluster_id == unique_clusters[0])  # Only show legend for first subplot
            ),
            row=1, col=1
        )
    
    # View 2: Dimensions 3-4-5
    for cluster_id in sorted(np.unique(colors)):
        mask = colors == cluster_id
        fig.add_trace(
            go.Scatter3d(
                x=embedding_subset[mask, 2],
                y=embedding_subset[mask, 3],
                z=embedding_subset[mask, 4],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(size=2, opacity=0.7),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # View 3: Dimensions 1-3-5
    for cluster_id in sorted(np.unique(colors)):
        mask = colors == cluster_id
        fig.add_trace(
            go.Scatter3d(
                x=embedding_subset[mask, 0],
                y=embedding_subset[mask, 2],
                z=embedding_subset[mask, 4],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(size=2, opacity=0.7),
                showlegend=False
            ),
            row=1, col=3
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'5D PaCMAP - Multiple 3D Projections<br><sub>Dataset: {dataset_label} | Clusters: {optimal_k} | Samples: {max_samples:,}</sub>',
            x=0.5,
            xanchor='center'
        ),
        font=dict(size=12),
        height=600,
        showlegend=True,
        legend=dict(x=1.05, y=0.5)
    )
    
    # Update axes labels
    fig.update_scenes(
        xaxis_title='Dim 1', yaxis_title='Dim 2', zaxis_title='Dim 3',
        row=1, col=1
    )
    fig.update_scenes(
        xaxis_title='Dim 3', yaxis_title='Dim 4', zaxis_title='Dim 5',
        row=1, col=2
    )
    fig.update_scenes(
        xaxis_title='Dim 1', yaxis_title='Dim 3', zaxis_title='Dim 5',
        row=1, col=3
    )
    
    print(f"  Saving HTML...")
    fig.write_html(
        output_path,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'pacmap_5d_linked_3d',
                'height': 800,
                'width': 1800,
                'scale': 2
            }
        }
    )
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  ✓ Saved: {output_path}")
    print(f"    File size: {file_size:.1f} MB")
    print(f"    Samples: {max_samples:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive 5D PaCMAP visualizations'
    )
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='Path to model directory'
    )
    parser.add_argument(
        '--max-samples-parcoords',
        type=int,
        default=10000,
        help='Maximum samples for parallel coordinates (default: 10000)'
    )
    parser.add_argument(
        '--max-samples-3d',
        type=int,
        default=5000,
        help='Maximum samples for 3D views (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.exists(args.dir):
        print(f"ERROR: Directory not found: {args.dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("5D PaCMAP Interactive Visualization")
    print("=" * 70)
    print(f"Model directory: {args.dir}")
    print(f"Max samples (parallel coords): {args.max_samples_parcoords}")
    print(f"Max samples (3D views): {args.max_samples_3d}")
    print("=" * 70)
    
    # Load 5D PaCMAP embeddings
    pacmap_5d_path = os.path.join(args.dir, 'PaCMAP', 'pacmap_embeddings_5d.mat')
    
    if not os.path.exists(pacmap_5d_path):
        print(f"\nERROR: 5D PaCMAP embeddings not found: {pacmap_5d_path}")
        print("Please run generate_pacmap_embeddings.py first to create 5D embeddings.")
        sys.exit(1)
    
    print(f"\nLoading 5D PaCMAP embeddings: {pacmap_5d_path}")
    data = loadmat(pacmap_5d_path)
    
    embedding_5d = data['pacmap_embeddings_5d']
    clusters = data.get('clusters', None)
    if clusters is not None and clusters.ndim > 1:
        clusters = clusters.flatten()
    optimal_k = data.get('optimal_k', np.array([[2]]))[0, 0] if 'optimal_k' in data else 2
    dataset_label = data.get('dataset_label', ['Unknown'])[0] if 'dataset_label' in data else 'Unknown'
    
    print(f"  Loaded embeddings: {embedding_5d.shape}")
    print(f"  Dataset: {dataset_label}")
    print(f"  Clusters: {optimal_k}")
    
    # Generate parallel coordinates plot
    output_dir = os.path.join(args.dir, 'PaCMAP')
    parcoords_path = os.path.join(output_dir, 'pacmap_5d_parallel_coordinates.html')
    generate_parallel_coordinates(
        embedding_5d, clusters, int(optimal_k), dataset_label,
        parcoords_path, max_samples=args.max_samples_parcoords
    )
    
    # Generate linked 3D views
    linked_3d_path = os.path.join(output_dir, 'pacmap_5d_linked_3d_views.html')
    generate_linked_3d_views(
        embedding_5d, clusters, int(optimal_k), dataset_label,
        linked_3d_path, max_samples=args.max_samples_3d
    )
    
    print("\n" + "=" * 70)
    print("✓ 5D PaCMAP visualizations generated!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - PaCMAP/pacmap_5d_parallel_coordinates.html")
    print(f"    (Interactive parallel coordinates plot with dimension brushing)")
    print(f"  - PaCMAP/pacmap_5d_linked_3d_views.html")
    print(f"    (Three rotatable 3D views showing different dimension combinations)")
    print("\n")


if __name__ == '__main__':
    main()
