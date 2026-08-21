# UMAP Generation Summary

**Generated:** February 10, 2026

## ✅ Successfully Completed

Generated **3D and 5D UMAP embeddings** for all 4 trained autoencoder models.

---

## Files Created

### LD16 Models

#### 1. LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir
- ✅ `UMAP/umap_embeddings_3d.mat` (26 MB) - 3D UMAP embeddings for 100,000 samples
- ✅ `UMAP/umap_embeddings_5d.mat` (27 MB) - 5D UMAP embeddings for 100,000 samples  
- ✅ `UMAP/umap_latent_3d.png` (316 KB) - 3D visualization
- **Latent Dimensions:** 16
- **Clusters:** 2
- **Dataset:** CombinedDatasets (Auto + Manual)

#### 2. LD16/Autoencoder_v14_100E_16LD_32C_Manual_100K_Date20260122-190056.dir
- ✅ `UMAP/umap_embeddings_3d.mat` (26 MB) - 3D UMAP embeddings for 99,933 samples
- ✅ `UMAP/umap_embeddings_5d.mat` (27 MB) - 5D UMAP embeddings for 99,933 samples
- ✅ `UMAP/umap_latent_3d.png` (278 KB) - 3D visualization
- **Latent Dimensions:** 16
- **Clusters:** 3
- **Dataset:** Manual Only

### LD32 Models

#### 3. LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir
- ✅ `UMAP/umap_embeddings_3d.mat` (32 MB) - 3D UMAP embeddings for 100,000 samples
- ✅ `UMAP/umap_embeddings_5d.mat` (33 MB) - 5D UMAP embeddings for 100,000 samples
- ✅ `UMAP/umap_latent_3d.png` (390 KB) - 3D visualization
- **Latent Dimensions:** 32
- **Clusters:** 2
- **Dataset:** CombinedDatasets (Auto + Manual)

#### 4. LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir
- ✅ `UMAP/umap_embeddings_3d.mat` (32 MB) - 3D UMAP embeddings for 99,933 samples
- ✅ `UMAP/umap_embeddings_5d.mat` (33 MB) - 5D UMAP embeddings for 99,933 samples
- ✅ `UMAP/umap_latent_3d.png` (260 KB) - 3D visualization
- **Latent Dimensions:** 32
- **Clusters:** 2
- **Dataset:** Manual Only

---

## UMAP Parameters Used

All embeddings generated with consistent parameters:
- **n_neighbors:** 15
- **min_dist:** 0.1  
- **random_state:** 42 (for reproducibility)

---

## File Contents

Each `.mat` file contains:

```matlab
latent_embeddings       % Original latent space (N × 16 or N × 32)
umap_embeddings_3d      % 3D UMAP coordinates (N × 3)  [3D files]
umap_embeddings_5d      % 5D UMAP coordinates (N × 5)  [5D files]
clusters                % Cluster assignments (N × 1)
optimal_k               % Number of clusters (scalar)
dataset_label           % Dataset name (string)
original_filenames      % Source file names (cell array)
reconstruction_filenames% Reconstruction file names (cell array)
umap_params             % UMAP parameters (struct)
```

---

## Complete UMAP Files Per Directory

Each directory now has:

```
UMAP/
├── umap_embeddings.mat     # Original 2D UMAP (~26-32 MB)
├── umap_embeddings_3d.mat  # NEW: 3D UMAP (~26-32 MB)
├── umap_embeddings_5d.mat  # NEW: 5D UMAP (~27-33 MB)
├── umap_latent.png         # Original 2D visualization
└── umap_latent_3d.png      # NEW: 3D visualization (~260-390 KB)
```

**Total:** 5 UMAP-related files per model

---

## Usage in MATLAB

### Load 3D UMAP Embeddings
```matlab
data = load('LD32/.../UMAP/umap_embeddings_3d.mat');
umap_3d = data.umap_embeddings_3d;  % N × 3 matrix
clusters = data.clusters;

% 3D scatter plot
figure;
scatter3(umap_3d(:,1), umap_3d(:,2), umap_3d(:,3), 20, clusters, 'filled');
xlabel('UMAP 1'); ylabel('UMAP 2'); zlabel('UMAP 3');
title('3D UMAP Latent Space');
colorbar;
view(3); rotate3d on;
```

### Load 5D UMAP Embeddings
```matlab
data = load('LD32/.../UMAP/umap_embeddings_5d.mat');
umap_5d = data.umap_embeddings_5d;  % N × 5 matrix
clusters = data.clusters;

% Analyze cluster quality in 5D
silhouette_vals = silhouette(umap_5d, clusters);
mean_score = mean(silhouette_vals);
fprintf('Cluster quality (5D): %.3f\n', mean_score);

% PCA for visualization
[~, score, ~] = pca(umap_5d);
figure;
gscatter(score(:,1), score(:,2), clusters);
title('PCA of 5D UMAP');
```

### Compare Dimensionalities
```matlab
% Load all three
data_2d = load('UMAP/umap_embeddings.mat');
data_3d = load('UMAP/umap_embeddings_3d.mat');
data_5d = load('UMAP/umap_embeddings_5d.mat');

% Compare cluster separation
s2d = mean(silhouette(data_2d.umap_embeddings, data_2d.clusters));
s3d = mean(silhouette(data_3d.umap_embeddings_3d, data_3d.clusters));
s5d = mean(silhouette(data_5d.umap_embeddings_5d, data_5d.clusters));

fprintf('Silhouette Scores:\n');
fprintf('  2D: %.3f\n  3D: %.3f\n  5D: %.3f\n', s2d, s3d, s5d);
```

---

## Statistics

| Metric | Value |
|--------|-------|
| **Total Directories Processed** | 4 |
| **Success Rate** | 100% (4/4) |
| **Total 3D UMAP Files Created** | 4 |
| **Total 5D UMAP Files Created** | 4 |
| **Total Visualizations Created** | 4 |
| **Total Samples Processed** | ~400K |
| **Total Processing Time** | ~8 minutes |
| **Total Storage Added** | ~232 MB |

---

## Next Steps

1. **Visualize in MATLAB:** Use the code examples above
2. **Cluster Analysis:** Compare 2D vs 3D vs 5D cluster quality
3. **Statistical Tests:** Leverage 5D embeddings for robust analysis  
4. **Publication Figures:** Use 3D PNG visualizations

---

## Documentation

Complete usage guide: [UMAP_GENERATION_GUIDE.md](UMAP_GENERATION_GUIDE.md)

---

## Regeneration

To regenerate or create embeddings with different parameters:

```bash
# Regenerate all with default parameters
./generate_all_umap.sh

# Custom parameters
python3 generate_additional_umap.py --all --n-neighbors 30 --min-dist 0.05

# Specific directory only
python3 generate_additional_umap.py --dir LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir
```

---

**Status:** ✅ Complete  
**Generated:** February 10, 2026  
**Script:** `generate_additional_umap.py`
