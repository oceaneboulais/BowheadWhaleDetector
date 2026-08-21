# UMAP Embeddings Guide

## Overview

This guide explains how to generate **3D and 5D UMAP embeddings** from your trained autoencoder models.

The existing models already have **2D UMAP embeddings**. This tool adds higher-dimensional representations that can be useful for:
- More detailed latent space analysis in MATLAB
- Better cluster separation in higher dimensions
- Advanced dimensionality reduction pipelines
- Statistical analysis requiring >2 dimensions

## Quick Start

### Generate All UMAP Embeddings (Recommended)

```bash
# Activate environment
source activate_venv.sh

# Generate 3D and 5D UMAP for all LD16/LD32 models
./generate_all_umap.sh
```

This will process all 4 trained models:
- ✓ LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir
- ✓ LD16/Autoencoder_v14_100E_16LD_32C_Manual_100K_Date20260122-190056.dir
- ✓ LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir
- ✓ LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir

### Generate for Specific Directory

```bash
source activate_venv.sh

python3 generate_additional_umap.py --dir LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir
```

## Output Files

Each model directory will have new files in the `UMAP/` folder:

### Before (Original 2D UMAP)
```
UMAP/
  └── umap_embeddings.mat     # Original 2D embeddings
```

### After (2D + 3D + 5D UMAP)
```
UMAP/
  ├── umap_embeddings.mat     # Original 2D embeddings
  ├── umap_embeddings_3d.mat  # NEW: 3D embeddings
  ├── umap_embeddings_5d.mat  # NEW: 5D embeddings
  └── umap_latent_3d.png      # NEW: 3D visualization
```

## MATLAB File Contents

Each `.mat` file contains:

### umap_embeddings_3d.mat
```matlab
latent_embeddings       % Original latent space (N × 16 or N × 32)
umap_embeddings_3d      % 3D UMAP coordinates (N × 3)
clusters                % Cluster assignments (N × 1)
optimal_k               % Number of clusters (scalar)
dataset_label           % Dataset name (string)
original_filenames      % Source file names (N × 1 cell)
reconstruction_filenames% Reconstruction file names (N × 1 cell)
umap_params             % UMAP parameters (struct)
```

### umap_embeddings_5d.mat
```matlab
latent_embeddings       % Original latent space (N × 16 or N × 32)
umap_embeddings_5d      % 5D UMAP coordinates (N × 5)
clusters                % Cluster assignments (N × 1)
optimal_k               % Number of clusters (scalar)
dataset_label           % Dataset name (string)
original_filenames      % Source file names (N × 1 cell)
reconstruction_filenames% Reconstruction file names (N × 1 cell)
umap_params             % UMAP parameters (struct)
```

## Using in MATLAB

### Load and Visualize 3D UMAP

```matlab
% Load 3D UMAP embeddings
data = load('LD32/...dir/UMAP/umap_embeddings_3d.mat');

% Extract data
umap_3d = data.umap_embeddings_3d;
clusters = data.clusters;
k = double(data.optimal_k);

% Create 3D scatter plot
figure;
gscatter(umap_3d(:,1), umap_3d(:,2), umap_3d(:,3), clusters);
xlabel('UMAP 1');
ylabel('UMAP 2');
zlabel('UMAP 3');
title(sprintf('3D UMAP Latent Space (k=%d)', k));
grid on;
view(3);
rotate3d on;
```

### Load and Analyze 5D UMAP

```matlab
% Load 5D UMAP embeddings
data = load('LD32/...dir/UMAP/umap_embeddings_5d.mat');

% Extract data
umap_5d = data.umap_embeddings_5d;  % N × 5 matrix
clusters = data.clusters;

% Compute pairwise distances in 5D space
distances = pdist(umap_5d, 'euclidean');
distance_matrix = squareform(distances);

% Analyze cluster separation
silhouette_vals = silhouette(umap_5d, clusters);
mean_silhouette = mean(silhouette_vals);
fprintf('Mean Silhouette Score (5D): %.3f\n', mean_silhouette);

% PCA on 5D UMAP for visualization
[coeff, score, ~] = pca(umap_5d);
figure;
gscatter(score(:,1), score(:,2), clusters);
xlabel('PC1 of 5D UMAP');
ylabel('PC2 of 5D UMAP');
title('PCA Projection of 5D UMAP');
```

### Compare 2D vs 3D vs 5D Cluster Quality

```matlab
% Load all three embeddings
data_2d = load('UMAP/umap_embeddings.mat');
data_3d = load('UMAP/umap_embeddings_3d.mat');
data_5d = load('UMAP/umap_embeddings_5d.mat');

% Extract embeddings
umap_2d = data_2d.umap_embeddings;
umap_3d = data_3d.umap_embeddings_3d;
umap_5d = data_5d.umap_embeddings_5d;
clusters = data_2d.clusters;

% Compute silhouette scores
silhouette_2d = mean(silhouette(umap_2d, clusters));
silhouette_3d = mean(silhouette(umap_3d, clusters));
silhouette_5d = mean(silhouette(umap_5d, clusters));

% Display comparison
fprintf('Cluster Quality Comparison:\n');
fprintf('  2D UMAP: %.3f\n', silhouette_2d);
fprintf('  3D UMAP: %.3f\n', silhouette_3d);
fprintf('  5D UMAP: %.3f\n', silhouette_5d);

% Plot comparison
figure;
bar([silhouette_2d, silhouette_3d, silhouette_5d]);
set(gca, 'XTickLabel', {'2D', '3D', '5D'});
ylabel('Mean Silhouette Score');
title('Cluster Quality vs UMAP Dimensionality');
ylim([0 1]);
grid on;
```

## Advanced Usage

### Custom UMAP Parameters

```bash
# More neighbors for smoother manifold
python3 generate_additional_umap.py --all --n-neighbors 30

# Tighter clusters with smaller min-dist
python3 generate_additional_umap.py --all --min-dist 0.05

# Custom parameters for exploratory analysis
python3 generate_additional_umap.py --all --n-neighbors 50 --min-dist 0.01
```

### Process Specific Directories Only

```bash
# Just the LD32 models
python3 generate_additional_umap.py \
  --dir LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir \
  --dir LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir
```

### Reproducibility

```bash
# Set custom random seed for reproducible embeddings
python3 generate_additional_umap.py --all --random-state 12345
```

## UMAP Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | 3 or 5 | Target dimensionality (automatically set) |
| `n_neighbors` | 15 | Balance between local/global structure. Higher = more global. |
| `min_dist` | 0.1 | Minimum distance between points. Lower = tighter clusters. |
| `random_state` | 42 | Random seed for reproducibility |

## Troubleshooting

### UMAP Not Installed

```bash
# Install umap-learn
source activate_venv.sh
pip install umap-learn

# Verify installation
python3 -c "import umap; print(f'UMAP version: {umap.__version__}')"
```

### Memory Issues (Large Datasets)

If you encounter memory errors with very large datasets:

```python
# Edit generate_additional_umap.py and reduce n_neighbors
--n-neighbors 10  # Instead of default 15
```

### "Latent embeddings not found"

The script needs existing latent embeddings from trained models. Make sure:
1. The model has been trained (checkpoint files exist)
2. `MATLAB/latent_embeddings.mat` exists in the directory
3. You're using the correct directory path

## File Size Reference

Approximate `.mat` file sizes:

| Samples | 2D UMAP | 3D UMAP | 5D UMAP | Total |
|---------|---------|---------|---------|-------|
| 10K     | ~2 MB   | ~3 MB   | ~5 MB   | ~10 MB |
| 50K     | ~10 MB  | ~15 MB  | ~25 MB  | ~50 MB |
| 100K    | ~20 MB  | ~30 MB  | ~50 MB  | ~100 MB |

## Performance Notes

- **3D UMAP**: ~1-5 minutes per 50K samples
- **5D UMAP**: ~2-8 minutes per 50K samples
- Processing all 4 directories: ~10-30 minutes total

Uses the same UMAP parameters as the original 2D embeddings (n_neighbors=15, min_dist=0.1) for consistency.

## What Gets Generated?

For each directory, the script will:
1. ✓ Load latent embeddings from `MATLAB/latent_embeddings.mat`
2. ✓ Generate 3D UMAP projection
3. ✓ Save `UMAP/umap_embeddings_3d.mat`
4. ✓ Create 3D visualization `UMAP/umap_latent_3d.png`
5. ✓ Generate 5D UMAP projection
6. ✓ Save `UMAP/umap_embeddings_5d.mat`
7. ✓ Print summary statistics

## Command Reference

```bash
# Quick commands
./generate_all_umap.sh                    # Process all directories
python3 generate_additional_umap.py --all # Same as above
python3 generate_additional_umap.py -h    # Show help

# Specific directory
python3 generate_additional_umap.py --dir <directory>

# Multiple specific directories
python3 generate_additional_umap.py --dir dir1 --dir dir2

# Custom parameters
python3 generate_additional_umap.py --all \
  --n-neighbors 30 \
  --min-dist 0.05 \
  --random-state 12345
```

## Next Steps

After generating the embeddings:

1. **Visualize in MATLAB**: Use the code examples above
2. **Cluster Analysis**: Compare cluster quality across dimensionalities
3. **Statistical Tests**: Use 5D embeddings for more robust statistics
4. **Export for Papers**: Use 3D visualization PNG files

## References

- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [UMAP Parameter Guide](https://umap-learn.readthedocs.io/en/latest/parameters.html)
- Original training scripts: `Autoencoder_v02_LD16_20251118.py`, `Autoencoder_v02_LD32_20251118.py`
