#!/usr/bin/env python3
"""
Generate UMAP embeddings and visualization for a trained model.
"""
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

try:
    import umap
    UMAP = umap.UMAP
except Exception:
    print("ERROR: UMAP not installed. Install with: pip install umap-learn")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model directory
MODEL_DIR = '/Users/oboulais/Desktop/Bowhead_DL_Project/LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir'

# UMAP Parameters
UMAP_PARAMS = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2,
    'metric': 'euclidean',
    'random_state': 42,
    'verbose': True
}

print("="*70)
print("UMAP Embedding Generator")
print("="*70)
print(f"\nModel directory: {MODEL_DIR}")
print(f"\nUMAP Parameters:")
for key, value in UMAP_PARAMS.items():
    if key != 'verbose':
        print(f"  {key:20s}: {value}")
print("="*70)

# ============================================================================
# STEP 1: Load latent embeddings
# ============================================================================

matlab_dir = os.path.join(MODEL_DIR, 'MATLAB')
umap_dir = os.path.join(MODEL_DIR, 'UMAP')
os.makedirs(umap_dir, exist_ok=True)

embeddings_path = os.path.join(matlab_dir, 'latent_embeddings.mat')

if not os.path.exists(embeddings_path):
    print(f"\n❌ ERROR: Embeddings file not found: {embeddings_path}")
    sys.exit(1)

print(f"\nLoading latent embeddings from: {embeddings_path}")
data = loadmat(embeddings_path)

latent_embeddings = data['latent_embeddings']
n_samples, n_dims = latent_embeddings.shape
print(f"  ✓ Loaded {n_samples:,} samples with {n_dims}D latent vectors")

# Load existing clustering info if available
clusters = data.get('clusters', None)
optimal_k = int(data.get('optimal_k', 2)[0, 0]) if 'optimal_k' in data else 2
dataset_label = str(data.get('dataset_label', ['Unknown'])[0]) if 'dataset_label' in data else 'Unknown'
filenames = data.get('original_filenames', None)
reconstruction_filenames = data.get('reconstruction_filenames', None)

if clusters is not None:
    clusters = clusters.flatten()
    print(f"  ✓ Using existing {optimal_k} clusters from t-SNE")

# ============================================================================
# STEP 2: Compute UMAP embeddings
# ============================================================================

print(f"\nComputing UMAP embeddings on {n_samples:,} samples...")
start_time = datetime.now()

reducer = UMAP(**UMAP_PARAMS)
umap_embeddings = reducer.fit_transform(latent_embeddings)

elapsed_time = (datetime.now() - start_time).total_seconds()
print(f"  ✓ UMAP complete in {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
print(f"  ✓ Output shape: {umap_embeddings.shape}")

# ============================================================================
# STEP 3: Generate visualization
# ============================================================================

print(f"\nGenerating UMAP visualization...")

cmap = plt.cm.get_cmap('tab10', optimal_k)
plt.figure(figsize=(7, 6))

if clusters is not None:
    # Plot with clusters
    for cluster_id in range(optimal_k):
        mask = clusters == cluster_id
        color = cmap(cluster_id)
        plt.scatter(umap_embeddings[mask, 0], umap_embeddings[mask, 1], 
                   c=[color], alpha=0.85, s=28, label=f'Cluster {cluster_id}')
else:
    # Plot without clusters
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], 
               c='blue', alpha=0.85, s=28)

plt.title(f'UMAP Latent Space (k={optimal_k})')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
if clusters is not None:
    plt.legend(loc='upper right', fontsize=8, framealpha=0.9, 
              ncol=(2 if optimal_k > 5 else 1))
plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', 
           ha='right', va='bottom', fontsize=7, style='italic', alpha=0.6)
plt.tight_layout()

# Save plot
plot_path = os.path.join(umap_dir, 'umap_latent.png')
plt.savefig(plot_path, dpi=160)
plt.close()
print(f"  ✓ Saved plot to: {plot_path}")

# ============================================================================
# STEP 4: Save UMAP embeddings to .mat file
# ============================================================================

print(f"\nSaving UMAP embeddings...")

umap_data = {
    'latent_embeddings': latent_embeddings,
    'umap_embeddings': umap_embeddings,
    'clusters': clusters if clusters is not None else np.zeros(n_samples, dtype=int),
    'optimal_k': optimal_k,
    'dataset_label': dataset_label,
    'umap_n_neighbors': UMAP_PARAMS['n_neighbors'],
    'umap_min_dist': UMAP_PARAMS['min_dist'],
    'umap_metric': UMAP_PARAMS['metric'],
    'umap_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

# Add filenames if available
if filenames is not None:
    umap_data['original_filenames'] = filenames
if reconstruction_filenames is not None:
    umap_data['reconstruction_filenames'] = reconstruction_filenames

mat_path = os.path.join(umap_dir, 'umap_embeddings.mat')
savemat(mat_path, umap_data)

file_size_mb = os.path.getsize(mat_path) / (1024 * 1024)
print(f"  ✓ Saved UMAP data to: {mat_path}")
print(f"  ✓ File size: {file_size_mb:.1f} MB")

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("="*70)
print("✓ UMAP GENERATION COMPLETE!")
print("="*70)
print()
print(f"Model: {os.path.basename(MODEL_DIR)}")
print(f"  Samples:          {n_samples:,}")
print(f"  Latent dims:      {n_dims}D")
print(f"  Clusters:         {optimal_k}")
print()
print(f"Output files in: {umap_dir}")
print(f"  - umap_latent.png         (visualization)")
print(f"  - umap_embeddings.mat     (MATLAB data)")
print()
print(f"UMAP computation time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
print("="*70)
