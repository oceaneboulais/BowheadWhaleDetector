#!/usr/bin/env python3
"""
Generate Enhanced Interactive Visualizations with Image Popups

Creates interactive 3D UMAP and PaCMAP visualizations where clicking on data points
displays the corresponding input spectrogram image.

Features:
- Fully rotatable 3D plots (Plotly)
- Click/hover to see sample information
- Sample spectrogram images displayed on selection
- Works with both UMAP and PaCMAP embeddings

USAGE:
    python3 generate_interactive_with_images.py --dir <model_directory>
    
EXAMPLE:
    python3 generate_interactive_with_images.py --dir LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.dir
"""

import os
import sys
import argparse
import numpy as np
from scipy.io import loadmat
import base64
from io import BytesIO

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("ERROR: plotly or pandas not installed. Run: pip install plotly pandas")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("ERROR: matplotlib not installed")
    sys.exit(1)


def load_reconstruction_data(directory):
    """Load reconstruction data with original spectrograms."""
    recon_path = os.path.join(directory, 'MATLAB', 'reconstruction_data.mat')
    
    if not os.path.exists(recon_path):
        print(f"Warning: Reconstruction data not found: {recon_path}")
        return None
    
    print(f"Loading reconstruction data: {recon_path}")
    data = loadmat(recon_path)
    
    return {
        'originals': data.get('originals', None),
        'reconstructions': data.get('reconstructions', None),
        'filenames': data.get('filenames', None)
    }


def create_thumbnail_base64(spectrogram, figsize=(3, 2.5), dpi=80):
    """Create base64 encoded thumbnail of spectrogram."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(spectrogram, cmap='viridis', aspect='auto', interpolation='nearest')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Save to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"


def generate_interactive_plot_with_images(embedding_3d, clusters, recon_data, optimal_k, 
                                         dataset_label, title, output_path, max_samples=5000):
    """Generate interactive 3D plot with image popups on hover."""
    
    print(f"\\nGenerating interactive plot: {title}")
    print(f"  Processing {len(embedding_3d)} samples...")
    
    # Subsample for performance if needed
    n_samples = len(embedding_3d)
    if n_samples > max_samples:
        print(f"  Subsampling from {n_samples} to {max_samples} for performance...")
        indices = np.random.choice(n_samples, max_samples, replace=False)
        indices = np.sort(indices)  # Keep order
    else:
        indices = np.arange(n_samples)
        max_samples = n_samples
    
    # Prepare data
    embedding_subset = embedding_3d[indices]
    clusters_subset = clusters[indices] if clusters is not None else np.zeros(len(indices))
    
    # Create hover text and images
    hover_texts = []
    custom_data = []
    
    # Check if we have reconstruction data and it matches the embedding size
    has_images = (recon_data is not None and 
                  recon_data['originals'] is not None and
                  len(recon_data['originals']) >= max_samples)
    
    if has_images:
        print(f"  Creating image thumbnails...")
        originals = recon_data['originals'][indices]
        
        for i, idx in enumerate(indices):
            if i % 500 == 0:
                print(f"    Processed {i}/{len(indices)} thumbnails...")
            
            # Create thumbnail
            img_base64 = create_thumbnail_base64(originals[i])
            
            hover_text = (
                f"Sample: {idx}<br>"
                f"Cluster: {int(clusters_subset[i])}<br>"
                f"Coord: ({embedding_subset[i, 0]:.2f}, {embedding_subset[i, 1]:.2f}, {embedding_subset[i, 2]:.2f})"
            )
            hover_texts.append(hover_text)
            custom_data.append([img_base64, idx, int(clusters_subset[i])])
    else:
        if recon_data is not None and recon_data['originals'] is not None:
            n_avail = len(recon_data['originals'])
            print(f"  Warning: Reconstruction data has only {n_avail} samples, need {max_samples}")
            print(f"  Generating text-only hover (no images)")
        else:
            print(f"  No images available, using text-only hover")
        
        for i, idx in enumerate(indices):
            hover_text = (
                f"Sample: {idx}<br>"
                f"Cluster: {int(clusters_subset[i])}<br>"
                f"Coord: ({embedding_subset[i, 0]:.2f}, {embedding_subset[i, 1]:.2f}, {embedding_subset[i, 2]:.2f})"
            )
            hover_texts.append(hover_text)
            custom_data.append([None, idx, int(clusters_subset[i])])
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': embedding_subset[:, 0],
        'y': embedding_subset[:, 1],
        'z': embedding_subset[:, 2],
        'Cluster': clusters_subset.astype(int).astype(str),
        'hover_text': hover_texts
    })
    
    # Create 3D scatter plot
    fig = go.Figure()
    
    # Add traces by cluster for better coloring
    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_df = df[df['Cluster'] == cluster_id]
        cluster_indices = [i for i, c in enumerate(df['Cluster']) if c == cluster_id]
        cluster_custom_data = [custom_data[i] for i in cluster_indices]
        
        fig.add_trace(go.Scatter3d(
            x=cluster_df['x'],
            y=cluster_df['y'],
            z=cluster_df['z'],
            mode='markers',
            name=f'Cluster {cluster_id}',
            marker=dict(
                size=3,
                opacity=0.7,
                line=dict(width=0)
            ),
            hovertext=cluster_df['hover_text'],
            hoverinfo='text',
            customdata=cluster_custom_data
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{title}<br><sub>Dataset: {dataset_label} | Clusters: {optimal_k} | Samples: {max_samples:,}</sub>',
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            xaxis=dict(showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            yaxis=dict(showbackground=True, backgroundcolor='rgb(230, 230, 230)'),
            zaxis=dict(showbackground=True, backgroundcolor='rgb(230, 230, 230)')
        ),
        font=dict(size=14),
        hovermode='closest',
        height=900,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    # Add annotation with instructions
    annotation_text = (
        "<b>Instructions:</b><br>"
        "• <b>Rotate:</b> Click and drag<br>"
        "• <b>Zoom:</b> Scroll wheel<br>"
        "• <b>Pan:</b> Right-click and drag<br>"
        "• <b>Hover:</b> View sample details<br>"
    )
    
    if has_images:
        annotation_text += "• <b>Click:</b> Opens spectrogram (in hover)<br>"
    
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=11)
    )
    
    # Save HTML with embedded images
    print(f"  Saving HTML...")
    fig.write_html(
        output_path,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['hoverclosest', 'hovercompare'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': os.path.splitext(os.path.basename(output_path))[0],
                'height': 1200,
                'width': 1200,
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
        description='Generate interactive visualizations with image popups'
    )
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='Path to model directory'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=5000,
        help='Maximum samples to include (default: 5000 for performance)'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.exists(args.dir):
        print(f"ERROR: Directory not found: {args.dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("Interactive Visualization with Image Popups")
    print("=" * 70)
    print(f"Model directory: {args.dir}")
    print(f"Max samples: {args.max_samples}")
    print("=" * 70)
    
    # Load reconstruction data
    recon_data = load_reconstruction_data(args.dir)
    
    # Generate for PaCMAP 3D
    pacmap_3d_path = os.path.join(args.dir, 'PaCMAP', 'pacmap_embeddings_3d.mat')
    if os.path.exists(pacmap_3d_path):
        print(f"\nLoading PaCMAP 3D embeddings: {pacmap_3d_path}")
        data = loadmat(pacmap_3d_path)
        
        embedding_3d = data['pacmap_embeddings_3d']
        clusters = data.get('clusters', None)
        if clusters is not None and clusters.ndim > 1:
            clusters = clusters.flatten()
        optimal_k = data.get('optimal_k', np.array([[2]]))[0, 0] if 'optimal_k' in data else 2
        dataset_label = data.get('dataset_label', ['Unknown'])[0] if 'dataset_label' in data else 'Unknown'
        
        output_path = os.path.join(args.dir, 'PaCMAP', 'pacmap_3d_interactive_with_images.html')
        generate_interactive_plot_with_images(
            embedding_3d, clusters, recon_data, int(optimal_k), 
            dataset_label, 'Interactive 3D PaCMAP with Image Popups', 
            output_path, max_samples=args.max_samples
        )
    
    # Generate for UMAP 3D
    umap_3d_path = os.path.join(args.dir, 'UMAP', 'umap_embeddings_3d.mat')
    if os.path.exists(umap_3d_path):
        print(f"\nLoading UMAP 3D embeddings: {umap_3d_path}")
        data = loadmat(umap_3d_path)
        
        embedding_3d = data['umap_embeddings_3d']
        clusters = data.get('clusters', None)
        if clusters is not None and clusters.ndim > 1:
            clusters = clusters.flatten()
        optimal_k = data.get('optimal_k', np.array([[2]]))[0, 0] if 'optimal_k' in data else 2
        dataset_label = data.get('dataset_label', ['Unknown'])[0] if 'dataset_label' in data else 'Unknown'
        
        output_path = os.path.join(args.dir, 'UMAP', 'umap_3d_interactive_with_images.html')
        generate_interactive_plot_with_images(
            embedding_3d, clusters, recon_data, int(optimal_k), 
            dataset_label, 'Interactive 3D UMAP with Image Popups', 
            output_path, max_samples=args.max_samples
        )
    
    print("\n" + "=" * 70)
    print("✓ Interactive visualizations with images generated!")
    print("=" * 70)
    print(f"\nGenerated files:")
    if os.path.exists(pacmap_3d_path):
        print(f"  - PaCMAP/pacmap_3d_interactive_with_images.html")
    if os.path.exists(umap_3d_path):
        print(f"  - UMAP/umap_3d_interactive_with_images.html")
    print("\n")


if __name__ == '__main__':
    main()
