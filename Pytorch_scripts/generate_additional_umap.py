#!/usr/bin/env python3
"""
Generate Additional UMAP Embeddings (3D and 5D)

This script generates 3D and 5D UMAP embeddings from existing latent representations.
It reads the latent embeddings from MATLAB/latent_embeddings.mat and generates:
- 3D UMAP embeddings → saved to UMAP/umap_embeddings_3d.mat
- 5D UMAP embeddings → saved to UMAP/umap_embeddings_5d.mat

USAGE:
    # Single directory
    python3 generate_additional_umap.py --dir LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir
    
    # All LD16 and LD32 directories
    python3 generate_additional_umap.py --all
    
    # Custom directories
    python3 generate_additional_umap.py --dir dir1 --dir dir2 --dir dir3
"""

import os
import sys
import argparse
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("ERROR: umap-learn not installed. Run: pip install umap-learn")
    sys.exit(1)


# Default directories for --all flag
DEFAULT_DIRS = [
    'LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir',
    'LD16/Autoencoder_v14_100E_16LD_32C_Manual_100K_Date20260122-190056.dir',
    'LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir',
    'LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir'
]


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
        'optimal_k': optimal_k,
        'dataset_label': dataset_label,
        'original_filenames': data.get('original_filenames', None),
        'reconstruction_filenames': data.get('reconstruction_filenames', None)
    }


def generate_umap_embeddings(latent_embeddings, n_components=3, n_neighbors=15, min_dist=0.1, random_state=42):
    """Generate UMAP embeddings with specified dimensionality."""
    print(f"  Computing {n_components}D UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=False
    )
    
    embeddings = reducer.fit_transform(latent_embeddings)
    """Investigate whether we can run this algorithm on subset of latent embeddings and see if we can get the same UMAP coordinates; explore other methods assocaited with UMAP"""
    print(f"  ✓ Generated {n_components}D UMAP embeddings: {embeddings.shape}")
    
    return embeddings


def plot_3d_umap(embeddings, clusters, optimal_k, dataset_label, save_path):
    """Create 3D scatter plot of UMAP embeddings."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use modern matplotlib colormap API
    try:
        cmap = plt.colormaps.get_cmap('tab10')
    except AttributeError:
        # Fallback for older matplotlib versions
        cmap = plt.cm.get_cmap('tab10', optimal_k)
    
    for cluster_id in range(optimal_k):
        mask = clusters == cluster_id
        if cmap.N >= optimal_k:
            color = cmap(cluster_id)
        else:
            color = cmap(cluster_id / optimal_k)
        ax.scatter(embeddings[mask, 0], 
                  embeddings[mask, 1], 
                  embeddings[mask, 2],
                  c=[color], alpha=0.7, s=20, label=f'Cluster {cluster_id}')
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    ax.set_title(f'3D UMAP Latent Space (k={optimal_k})')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=(2 if optimal_k > 5 else 1))
    
    plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', 
               ha='right', va='bottom', fontsize=7, style='italic', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
    print(f"  ✓ Saved 3D visualization: {save_path}")


def save_umap_embeddings(directory, data_dict, n_components):
    """Save UMAP embeddings to .mat file."""
    umap_dir = os.path.join(directory, 'UMAP')
    os.makedirs(umap_dir, exist_ok=True)
    
    filename = f'umap_embeddings_{n_components}d.mat'
    save_path = os.path.join(umap_dir, filename)
    
    savemat(save_path, data_dict)
    print(f"  ✓ Saved {n_components}D UMAP embeddings: {save_path}")
    
    return save_path


def process_directory(directory, n_neighbors=15, min_dist=0.1, random_state=42):
    """Process a single directory: load latents, generate 3D/5D UMAPs, save results."""
    print(f"\n{'='*80}")
    print(f"Processing: {directory}")
    print(f"{'='*80}")
    
    if not os.path.exists(directory):
        print(f"  ✗ ERROR: Directory not found: {directory}")
        return False
    
    try:
        # Load latent embeddings and metadata
        data = load_latent_embeddings(directory)
        latent_embeddings = data['latent_embeddings']
        clusters = data['clusters']
        optimal_k = data['optimal_k']
        dataset_label = data['dataset_label']
        
        # Generate 3D UMAP embeddings
        print(f"\n[3D UMAP]")
        umap_3d = generate_umap_embeddings(
            latent_embeddings, 
            n_components=3, 
            n_neighbors=n_neighbors, 
            min_dist=min_dist,
            random_state=random_state
        )
        
        # Save 3D UMAP embeddings
        umap_3d_data = {
            'latent_embeddings': latent_embeddings,
            'umap_embeddings_3d': umap_3d,
            'clusters': clusters,
            'optimal_k': optimal_k,
            'dataset_label': dataset_label,
            'original_filenames': data.get('original_filenames', []),
            'reconstruction_filenames': data.get('reconstruction_filenames', []),
            'umap_params': {
                'n_components': 3,
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'random_state': random_state
            }
        }
        save_umap_embeddings(directory, umap_3d_data, 3)
        
        # Plot 3D UMAP visualization
        if clusters is not None:
            plot_path = os.path.join(directory, 'UMAP', 'umap_latent_3d.png')
            plot_3d_umap(umap_3d, clusters, optimal_k, dataset_label, plot_path)
        
        # Generate 5D UMAP embeddings
        print(f"\n[5D UMAP]")
        umap_5d = generate_umap_embeddings(
            latent_embeddings, 
            n_components=5, 
            n_neighbors=n_neighbors, 
            min_dist=min_dist,
            random_state=random_state
        )
        
        # Save 5D UMAP embeddings
        umap_5d_data = {
            'latent_embeddings': latent_embeddings,
            'umap_embeddings_5d': umap_5d,
            'clusters': clusters,
            'optimal_k': optimal_k,
            'dataset_label': dataset_label,
            'original_filenames': data.get('original_filenames', []),
            'reconstruction_filenames': data.get('reconstruction_filenames', []),
            'umap_params': {
                'n_components': 5,
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'random_state': random_state
            }
        }
        save_umap_embeddings(directory, umap_5d_data, 5)
        
        print(f"\n✓ SUCCESS: Completed processing {directory}")
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR processing {directory}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D and 5D UMAP embeddings from existing latent representations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all default LD16/LD32 directories
  python3 generate_additional_umap.py --all
  
  # Process specific directory
  python3 generate_additional_umap.py --dir LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir
  
  # Process multiple directories with custom UMAP parameters
  python3 generate_additional_umap.py --dir dir1 --dir dir2 --n-neighbors 30 --min-dist 0.05
        """
    )
    
    parser.add_argument('--dir', action='append', dest='directories',
                       help='Directory to process (can be specified multiple times)')
    parser.add_argument('--all', action='store_true',
                       help='Process all default LD16 and LD32 directories')
    parser.add_argument('--n-neighbors', type=int, default=15,
                       help='UMAP n_neighbors parameter (default: 15)')
    parser.add_argument('--min-dist', type=float, default=0.1,
                       help='UMAP min_dist parameter (default: 0.1)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Determine which directories to process
    if args.all:
        directories = DEFAULT_DIRS
        print(f"Processing all default directories ({len(directories)} total)")
    elif args.directories:
        directories = args.directories
        print(f"Processing {len(directories)} specified directories")
    else:
        parser.print_help()
        print("\nERROR: Must specify either --all or --dir")
        sys.exit(1)
    
    # Process each directory
    print(f"\nUMAP Parameters:")
    print(f"  n_neighbors: {args.n_neighbors}")
    print(f"  min_dist: {args.min_dist}")
    print(f"  random_state: {args.random_state}")
    
    success_count = 0
    fail_count = 0
    
    for directory in directories:
        success = process_directory(
            directory,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.random_state
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total directories: {len(directories)}")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed:  {fail_count}")
    
    if fail_count == 0:
        print(f"\n🎉 All directories processed successfully!")
    else:
        print(f"\n⚠️  Some directories failed to process. Check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    if not UMAP_AVAILABLE:
        print("ERROR: umap-learn package not installed")
        print("Install with: pip install umap-learn")
        sys.exit(1)
    
    main()
