#!/usr/bin/env python3
"""
Generate PaCMAP Embeddings for Latent Representations

PaCMAP (Pairwise Controlled Manifold Approximation) is a dimensionality reduction
method that preserves both local and global structure using three types of pairs:
- Neighbor pairs (pair_neighbors): preserve local structure
- Mid-near pairs (pair_MN): preserve global structure  
- Further pairs (pair_FP): optimize overall embedding quality

This script generates 2D and 3D PaCMAP embeddings from existing latent representations.

USAGE:
    python3 generate_pacmap_embeddings.py --dir <model_directory>
    
EXAMPLE:
    python3 generate_pacmap_embeddings.py --dir LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.dir
"""

import os
import sys
import argparse
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

try:
    import pacmap
    PACMAP_AVAILABLE = True
except ImportError:
    PACMAP_AVAILABLE = False
    print("ERROR: pacmap not installed. Installing...")
    print("Run: pip install pacmap")
    sys.exit(1)


def load_latent_embeddings(directory):
    """Load latent embeddings from MATLAB directory."""
    latent_path = os.path.join(directory, 'MATLAB', 'latent_embeddings.mat')
    
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"Latent embeddings not found: {latent_path}")
    
    print(f"Loading latent embeddings from: {latent_path}")
    data = loadmat(latent_path)
    
    # Extract latent embeddings and metadata
    latent_embeddings = data['latent_embeddings']
    clusters = data.get('clusters', None)
    
    # Flatten clusters if it's a 2D array
    if clusters is not None and clusters.ndim > 1:
        clusters = clusters.flatten()
    
    optimal_k = data.get('optimal_k', np.array([[5]]))[0, 0] if 'optimal_k' in data else 5
    dataset_label = data.get('dataset_label', 'Unknown')
    
    print(f"  Loaded {latent_embeddings.shape[0]} samples, {latent_embeddings.shape[1]} latent dimensions")
    print(f"  Dataset: {dataset_label}, Clusters: {optimal_k}")
    
    return {
        'latent_embeddings': latent_embeddings,
        'clusters': clusters,
        'optimal_k': int(optimal_k),
        'dataset_label': dataset_label,
        'original_filenames': data.get('original_filenames', None),
        'reconstruction_filenames': data.get('reconstruction_filenames', None)
    }


def generate_pacmap_2d(latent_embeddings, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42):
    """
    Generate 2D PaCMAP embeddings.
    
    Parameters:
    -----------
    latent_embeddings : ndarray
        High-dimensional latent representations (N, D)
    n_neighbors : int
        Number of neighbors to consider for local structure (default: 10)
    MN_ratio : float
        Ratio of mid-near pairs to neighbor pairs (default: 0.5)
    FP_ratio : float
        Ratio of further pairs to neighbor pairs (default: 2.0)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    embedding_2d : ndarray
        2D PaCMAP embeddings (N, 2)
    """
    print(f"\nGenerating 2D PaCMAP embeddings...")
    print(f"  n_neighbors={n_neighbors}, MN_ratio={MN_ratio}, FP_ratio={FP_ratio}")
    
    embedding = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        random_state=random_state
    )
    
    embedding_2d = embedding.fit_transform(latent_embeddings)
    print(f"  ✓ 2D PaCMAP shape: {embedding_2d.shape}")
    
    return embedding_2d


def generate_pacmap_3d(latent_embeddings, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42):
    """
    Generate 3D PaCMAP embeddings.
    
    Parameters:
    -----------
    latent_embeddings : ndarray
        High-dimensional latent representations (N, D)
    n_neighbors : int
        Number of neighbors to consider for local structure (default: 10)
    MN_ratio : float
        Ratio of mid-near pairs to neighbor pairs (default: 0.5)
    FP_ratio : float
        Ratio of further pairs to neighbor pairs (default: 2.0)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    embedding_3d : ndarray
        3D PaCMAP embeddings (N, 3)
    """
    print(f"\nGenerating 3D PaCMAP embeddings...")
    print(f"  n_neighbors={n_neighbors}, MN_ratio={MN_ratio}, FP_ratio={FP_ratio}")
    
    embedding = pacmap.PaCMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        random_state=random_state
    )
    
    embedding_3d = embedding.fit_transform(latent_embeddings)
    print(f"  ✓ 3D PaCMAP shape: {embedding_3d.shape}")
    
    return embedding_3d


def generate_pacmap_5d(latent_embeddings, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42):
    """
    Generate 5D PaCMAP embeddings.
    
    Parameters:
    -----------
    latent_embeddings : ndarray
        High-dimensional latent representations (N, D)
    n_neighbors : int
        Number of neighbors to consider for local structure (default: 10)
    MN_ratio : float
        Ratio of mid-near pairs to neighbor pairs (default: 0.5)
    FP_ratio : float
        Ratio of further pairs to neighbor pairs (default: 2.0)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    embedding_5d : ndarray
        5D PaCMAP embeddings (N, 5)
    """
    print(f"\nGenerating 5D PaCMAP embeddings...")
    print(f"  n_neighbors={n_neighbors}, MN_ratio={MN_ratio}, FP_ratio={FP_ratio}")
    
    embedding = pacmap.PaCMAP(
        n_components=5,
        n_neighbors=n_neighbors,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        random_state=random_state
    )
    
    embedding_5d = embedding.fit_transform(latent_embeddings)
    print(f"  ✓ 5D PaCMAP shape: {embedding_5d.shape}")
    
    return embedding_5d


def plot_pacmap_2d(embedding_2d, clusters, optimal_k, save_path):
    """Plot 2D PaCMAP embeddings with cluster coloring."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if clusters is not None:
        scatter = ax.scatter(
            embedding_2d[:, 0], 
            embedding_2d[:, 1],
            c=clusters,
            cmap='tab10',
            s=1,
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.set_title(f'2D PaCMAP Latent Space (k={optimal_k})', fontsize=14, weight='bold')
    else:
        ax.scatter(
            embedding_2d[:, 0], 
            embedding_2d[:, 1],
            s=1,
            alpha=0.6
        )
        ax.set_title('2D PaCMAP Latent Space', fontsize=14, weight='bold')
    
    ax.set_xlabel('PaCMAP 1', fontsize=12)
    ax.set_ylabel('PaCMAP 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved 2D plot: {save_path}")


def plot_pacmap_3d(embedding_3d, clusters, optimal_k, save_path):
    """Plot 3D PaCMAP embeddings with cluster coloring."""
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    if clusters is not None:
        scatter = ax.scatter(
            embedding_3d[:, 0],
            embedding_3d[:, 1],
            embedding_3d[:, 2],
            c=clusters,
            cmap='tab10',
            s=1,
            alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='Cluster', shrink=0.5)
        ax.set_title(f'3D PaCMAP Latent Space (k={optimal_k})', fontsize=14, weight='bold')
    else:
        ax.scatter(
            embedding_3d[:, 0],
            embedding_3d[:, 1],
            embedding_3d[:, 2],
            s=1,
            alpha=0.6
        )
        ax.set_title('3D PaCMAP Latent Space', fontsize=14, weight='bold')
    
    ax.set_xlabel('PaCMAP 1', fontsize=11)
    ax.set_ylabel('PaCMAP 2', fontsize=11)
    ax.set_zlabel('PaCMAP 3', fontsize=11)
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved 3D plot: {save_path}")


def save_pacmap_embeddings(directory, embedding_2d, embedding_3d, embedding_5d, metadata):
    """Save PaCMAP embeddings to MATLAB-compatible format."""
    pacmap_dir = os.path.join(directory, 'PaCMAP')
    os.makedirs(pacmap_dir, exist_ok=True)
    
    # Save 2D embeddings
    save_data_2d = {
        'pacmap_embeddings_2d': embedding_2d,
        'clusters': metadata['clusters'],
        'optimal_k': metadata['optimal_k'],
        'dataset_label': metadata['dataset_label'],
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if metadata['original_filenames'] is not None:
        save_data_2d['original_filenames'] = metadata['original_filenames']
    if metadata['reconstruction_filenames'] is not None:
        save_data_2d['reconstruction_filenames'] = metadata['reconstruction_filenames']
    
    path_2d = os.path.join(pacmap_dir, 'pacmap_embeddings_2d.mat')
    savemat(path_2d, save_data_2d)
    print(f"  ✓ Saved 2D embeddings: {path_2d}")
    
    # Save 3D embeddings
    save_data_3d = {
        'pacmap_embeddings_3d': embedding_3d,
        'clusters': metadata['clusters'],
        'optimal_k': metadata['optimal_k'],
        'dataset_label': metadata['dataset_label'],
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if metadata['original_filenames'] is not None:
        save_data_3d['original_filenames'] = metadata['original_filenames']
    if metadata['reconstruction_filenames'] is not None:
        save_data_3d['reconstruction_filenames'] = metadata['reconstruction_filenames']
    
    path_3d = os.path.join(pacmap_dir, 'pacmap_embeddings_3d.mat')
    savemat(path_3d, save_data_3d)
    print(f"  ✓ Saved 3D embeddings: {path_3d}")
    
    # Save 5D embeddings
    save_data_5d = {
        'pacmap_embeddings_5d': embedding_5d,
        'clusters': metadata['clusters'],
        'optimal_k': metadata['optimal_k'],
        'dataset_label': metadata['dataset_label'],
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if metadata['original_filenames'] is not None:
        save_data_5d['original_filenames'] = metadata['original_filenames']
    if metadata['reconstruction_filenames'] is not None:
        save_data_5d['reconstruction_filenames'] = metadata['reconstruction_filenames']
    
    path_5d = os.path.join(pacmap_dir, 'pacmap_embeddings_5d.mat')
    savemat(path_5d, save_data_5d)
    print(f"  ✓ Saved 5D embeddings: {path_5d}")
    
    return pacmap_dir


def generate_interactive_3d_html(embedding_3d, clusters, optimal_k, save_path):
    """Generate interactive 3D PaCMAP plot using plotly."""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        # Create DataFrame for plotly
        import pandas as pd
        df = pd.DataFrame({
            'PaCMAP_1': embedding_3d[:, 0],
            'PaCMAP_2': embedding_3d[:, 1],
            'PaCMAP_3': embedding_3d[:, 2],
            'Cluster': clusters if clusters is not None else np.zeros(len(embedding_3d))
        })
        
        # Create interactive 3D scatter plot
        fig = px.scatter_3d(
            df,
            x='PaCMAP_1',
            y='PaCMAP_2',
            z='PaCMAP_3',
            color='Cluster',
            color_continuous_scale='viridis' if clusters is None else None,
            title=f'Interactive 3D PaCMAP Latent Space (k={optimal_k})' if clusters is not None else 'Interactive 3D PaCMAP Latent Space',
            opacity=0.6,
            height=800
        )
        
        fig.update_traces(marker=dict(size=2))
        fig.update_layout(
            scene=dict(
                xaxis_title='PaCMAP 1',
                yaxis_title='PaCMAP 2',
                zaxis_title='PaCMAP 3'
            )
        )
        
        fig.write_html(save_path)
        print(f"  ✓ Saved interactive 3D HTML: {save_path}")
        
    except ImportError:
        print("  ⚠ Warning: plotly not installed, skipping interactive HTML generation")


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaCMAP embeddings from latent representations'
    )
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='Path to model directory containing MATLAB/latent_embeddings.mat'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=10,
        help='Number of neighbors for local structure (default: 10)'
    )
    parser.add_argument(
        '--mn-ratio',
        type=float,
        default=0.5,
        help='Ratio of mid-near pairs to neighbor pairs (default: 0.5)'
    )
    parser.add_argument(
        '--fp-ratio',
        type=float,
        default=2.0,
        help='Ratio of further pairs to neighbor pairs (default: 2.0)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.exists(args.dir):
        print(f"ERROR: Directory not found: {args.dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("PaCMAP Embedding Generation")
    print("=" * 70)
    print(f"Model directory: {args.dir}")
    print(f"PaCMAP parameters:")
    print(f"  n_neighbors: {args.n_neighbors}")
    print(f"  MN_ratio: {args.mn_ratio}")
    print(f"  FP_ratio: {args.fp_ratio}")
    print(f"  random_state: {args.random_state}")
    print("=" * 70)
    
    # Load latent embeddings
    data = load_latent_embeddings(args.dir)
    latent_embeddings = data['latent_embeddings']
    clusters = data['clusters']
    optimal_k = data['optimal_k']
    
    # Generate PaCMAP embeddings
    embedding_2d = generate_pacmap_2d(
        latent_embeddings,
        n_neighbors=args.n_neighbors,
        MN_ratio=args.mn_ratio,
        FP_ratio=args.fp_ratio,
        random_state=args.random_state
    )
    
    embedding_3d = generate_pacmap_3d(
        latent_embeddings,
        n_neighbors=args.n_neighbors,
        MN_ratio=args.mn_ratio,
        FP_ratio=args.fp_ratio,
        random_state=args.random_state
    )
    
    embedding_5d = generate_pacmap_5d(
        latent_embeddings,
        n_neighbors=args.n_neighbors,
        MN_ratio=args.mn_ratio,
        FP_ratio=args.fp_ratio,
        random_state=args.random_state
    )
    
    # Save embeddings
    print("\nSaving PaCMAP embeddings...")
    pacmap_dir = save_pacmap_embeddings(args.dir, embedding_2d, embedding_3d, embedding_5d, data)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_pacmap_2d(
        embedding_2d,
        clusters,
        optimal_k,
        os.path.join(pacmap_dir, 'pacmap_2d.png')
    )
    
    plot_pacmap_3d(
        embedding_3d,
        clusters,
        optimal_k,
        os.path.join(pacmap_dir, 'pacmap_3d.png')
    )
    
    # Generate interactive 3D HTML
    generate_interactive_3d_html(
        embedding_3d,
        clusters,
        optimal_k,
        os.path.join(pacmap_dir, 'pacmap_3d_interactive.html')
    )
    
    print("\n" + "=" * 70)
    print("✓ PaCMAP embedding generation complete!")
    print("=" * 70)
    print(f"\nResults saved to: {pacmap_dir}")
    print("\nGenerated files:")
    print("  - pacmap_embeddings_2d.mat (2D embeddings)")
    print("  - pacmap_embeddings_3d.mat (3D embeddings)")
    print("  - pacmap_embeddings_5d.mat (5D embeddings)")
    print("  - pacmap_2d.png (2D visualization)")
    print("  - pacmap_3d.png (3D visualization)")
    print("  - pacmap_3d_interactive.html (interactive 3D plot)")
    print("\n")


if __name__ == '__main__':
    main()
