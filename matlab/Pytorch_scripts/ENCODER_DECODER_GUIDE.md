# Encoder/Decoder Inference Scripts Guide

Two standalone scripts for using trained autoencoder models:

1. **`extract_latent_embeddings.py`** - Extract latent embeddings (encoder only)
2. **`reconstruct_from_latent.py`** - Reconstruct spectrograms (decoder only)

Both scripts use **frozen weights** from trained models.

---

## Quick Start

### Activate Environment

```bash
source venv_bowhead/bin/activate
```

### Extract Latent Embeddings (Encoder)

```bash
# Single file
python3 extract_latent_embeddings.py \
    --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder.pth \
    --input BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir/S308A0T20080828T000045_Type0.mat \
    --output latent_embedding.mat

# Batch directory
python3 extract_latent_embeddings.py \
    --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder.pth \
    --input_dir BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir \
    --output_dir latent_embeddings_output \
    --batch_size 64
```

### Reconstruct Spectrograms (Decoder)

```bash
# From extracted embeddings
python3 reconstruct_from_latent.py \
    --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder.pth \
    --input latent_embeddings_output/latent_embeddings.mat \
    --output reconstructed_spectrograms.mat \
    --save_images \
    --image_dir reconstruction_images
```

---

## Usage Examples

### Example 1: Extract + Reconstruct Pipeline

```bash
# Step 1: Extract latent embeddings from 1000 files
python3 extract_latent_embeddings.py \
    --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder.pth \
    --input_dir BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir \
    --output_dir latent_output \
    --file_limit 1000

# Step 2: Reconstruct from those embeddings
python3 reconstruct_from_latent.py \
    --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder.pth \
    --input latent_output/latent_embeddings.mat \
    --output reconstructions.mat \
    --save_images \
    --max_images 50
```

### Example 2: Test Model Quality (Round-Trip)

```bash
# Extract embeddings
python3 extract_latent_embeddings.py \
    --model LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir/trained_model/autoencoder.pth \
    --latent_dim 16 \
    --input_dir BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir \
    --output_dir test_latent \
    --file_limit 100

# Reconstruct
python3 reconstruct_from_latent.py \
    --model LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir/trained_model/autoencoder.pth \
    --latent_dim 16 \
    --input test_latent/latent_embeddings.mat \
    --output test_recon.mat \
    --save_images \
    --image_dir test_images
```

### Example 3: Use LD32 v14 Model

```bash
# Extract with v14 architecture
python3 extract_latent_embeddings.py \
    --model LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir/trained_model/autoencoder.pth \
    --latent_dim 32 \
    --input_dir BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir \
    --output_dir v14_latent

# Reconstruct
python3 reconstruct_from_latent.py \
    --model LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir/trained_model/autoencoder.pth \
    --latent_dim 32 \
    --input v14_latent/latent_embeddings.mat \
    --output v14_recon.mat
```

---

## Command Line Arguments

### `extract_latent_embeddings.py`

#### Model Parameters
- `--model PATH` (required) - Path to trained `.pth` file
- `--latent_dim INT` - Latent dimension (default: 32)
- `--base_channels INT` - Base channels (default: 32)
- `--nrow INT` - Input height (default: 121)
- `--ncol INT` - Input width (default: 104)
- `--extra_conv` - Use 4-layer encoder architecture

#### Single File Mode
- `--input PATH` - Single `.mat` file to process
- `--output PATH` - Output `.mat` file for embedding

#### Batch Mode
- `--input_dir PATH` - Directory containing `.mat` files
- `--output_dir PATH` - Directory for batch output
- `--batch_size INT` - Batch size (default: 64)
- `--file_limit INT` - Limit number of files (for testing)

#### Processing Options
- `--no_normalize` - Skip min-max normalization
- `--device {auto,cpu,cuda,mps}` - Device selection (default: auto)

### `reconstruct_from_latent.py`

#### Model Parameters
- `--model PATH` (required) - Path to trained `.pth` file
- `--latent_dim INT` - Latent dimension (default: 32)
- `--base_channels INT` - Base channels (default: 32)
- `--nrow INT` - Output height (default: 121)
- `--ncol INT` - Output width (default: 104)
- `--extra_conv` - Use 4-layer decoder architecture

#### Input/Output
- `--input PATH` (required) - Input `.mat` file with latent embeddings
- `--output PATH` (required) - Output `.mat` file for reconstructions
- `--batch_size INT` - Batch size (default: 64)

#### Visualization Options
- `--save_images` - Save PNG images of reconstructions
- `--image_dir PATH` - Directory for images (default: reconstructed_images)
- `--max_images INT` - Maximum images to save (default: 100)

#### Processing Options
- `--device {auto,cpu,cuda,mps}` - Device selection (default: auto)
- `--quiet` - Suppress progress output

---

## Output File Formats

### Latent Embeddings (from encoder)

**Single file mode** (`--input` + `--output`):
```matlab
% Contents of latent_embedding.mat
latent_embedding    % Shape: (latent_dim,)
original_shape      % [121, 104]
input_file          % 'S308A0T20080828T000045_Type0.mat'
```

**Batch mode** (`--input_dir` + `--output_dir`):
```matlab
% Contents of latent_embeddings.mat in output_dir
latent_embeddings   % Shape: (N, latent_dim)
filenames           % Cell array of N filenames
num_samples         % N
latent_dim          % 16 or 32
```

### Reconstructed Spectrograms (from decoder)

```matlab
% Contents of reconstructed_spectrograms.mat
reconstructed_spectrograms  % Shape: (N, 121, 104)
filenames                   % Original filenames (if available)
num_samples                 % N
shape                       % [N, 121, 104]
```

---

## Performance Benchmarks

### LD32 Model on MPS (Apple Silicon)

**Encoder (extract_latent_embeddings.py)**:
- Single file: ~5ms per file
- Batch mode: ~200 files/sec (batch_size=64)
- 100K files: ~8 minutes

**Decoder (reconstruct_from_latent.py)**:
- Single embedding: ~5ms per reconstruction
- Batch mode: ~200 reconstructions/sec (batch_size=64)
- 100K embeddings: ~8 minutes

### Memory Usage

| Operation | Batch Size | Memory (GPU/MPS) |
|-----------|-----------|------------------|
| Encoder LD16 | 64 | ~500 MB |
| Encoder LD32 | 64 | ~800 MB |
| Decoder LD16 | 64 | ~600 MB |
| Decoder LD32 | 64 | ~1 GB |

---

## Use Cases

### 1. Feature Extraction for Analysis

Extract latent embeddings for clustering, classification, or similarity analysis:

```bash
python3 extract_latent_embeddings.py \
    --model LD32/.../autoencoder.pth \
    --input_dir BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir \
    --output_dir analysis_embeddings

# Then use in MATLAB, Python (sklearn), etc.
```

### 2. Reconstruction Quality Assessment

Test model performance by comparing original vs reconstructed:

```bash
# Extract → Reconstruct → Compare
python3 extract_latent_embeddings.py --model ... --input_dir test_data --output_dir test_latent
python3 reconstruct_from_latent.py --model ... --input test_latent/latent_embeddings.mat --output test_recon.mat --save_images
```

### 3. Denoising

Run noisy spectrograms through encoder→decoder to denoise:

```bash
python3 extract_latent_embeddings.py --model ... --input noisy.mat --output latent.mat
python3 reconstruct_from_latent.py --model ... --input latent.mat --output denoised.mat
```

### 4. Dimensionality Reduction

Use encoder as dimensionality reduction (121×104=12584 → 32):

```bash
python3 extract_latent_embeddings.py \
    --model LD32/.../autoencoder.pth \
    --input_dir large_dataset \
    --output_dir reduced_features

# Now use 32D vectors instead of 12584D spectrograms
```

### 5. Generative Sampling

Sample latent space and generate new spectrograms (requires custom latent sampling):

```python
# Create synthetic latent vectors
import numpy as np
from scipy.io import savemat

# Sample from learned distribution
latents = np.random.randn(100, 32) * 0.5  # 100 samples, 32D
savemat('synthetic_latents.mat', {'latent_embeddings': latents})
```

```bash
# Generate spectrograms
python3 reconstruct_from_latent.py \
    --model LD32/.../autoencoder.pth \
    --input synthetic_latents.mat \
    --output synthetic_spectrograms.mat \
    --save_images
```

---

## Model Compatibility

### Available Models

| Model | Latent Dim | Base Channels | Version |
|-------|-----------|---------------|---------|
| LD16 v13 AutoManual | 16 | 32 | v13 |
| LD16 v14 Manual | 16 | 32 | v14 |
| LD32 v13 AutoManual | 32 | 32 | v13 |
| LD32 v14 Manual | 32 | 32 | v14 |

### Architecture Parameters

**Standard (3-layer)**: `--latent_dim 32 --base_channels 32`
- Input: 121×104
- Encoder: Conv→Pool (×3)
- Latent: 32D
- Decoder: ConvTranspose (×3)

**Extra Conv (4-layer)**: `--latent_dim 32 --base_channels 32 --extra_conv`
- Input: 121×104
- Encoder: Conv→Pool (×4)
- Latent: 32D
- Decoder: ConvTranspose (×4)

### Weight Freezing

Both scripts automatically freeze all model parameters:

```python
for param in model.parameters():
    param.requires_grad = False
```

This ensures:
- ✓ No gradient computation (faster)
- ✓ No accidental weight updates
- ✓ Deterministic inference
- ✓ Reduced memory usage

---

## Troubleshooting

### Error: "RuntimeError: No valid .mat files found"

**Problem**: Input directory has no `.mat` files or wrong structure

**Solution**:
```bash
# Check directory contents
ls BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir/*.mat | head -5
```

### Error: "ValueError: No latent embedding field found"

**Problem**: Input `.mat` file doesn't have expected fields

**Solution**: Check file contents in MATLAB/Python:
```matlab
load('latent_embedding.mat');
whos
```

Expected fields: `latent_embedding` or `latent_embeddings`

### Error: "CUDA out of memory"

**Problem**: Batch size too large for GPU

**Solution**: Reduce batch size
```bash
--batch_size 32  # or 16
```

### Warning: "Skipped [filename]: Failed to load"

**Problem**: Corrupted or incompatible `.mat` file

**Solution**: This is normal - corrupted files are automatically skipped. Check if many files are skipped.

### Wrong Output Shape

**Problem**: Using wrong architecture parameters

**Solution**: Match training architecture:
```bash
# LD16 models
--latent_dim 16 --base_channels 32

# LD32 models
--latent_dim 32 --base_channels 32
```

---

## Integration with MATLAB

### Load Latent Embeddings in MATLAB

```matlab
% Load batch embeddings
data = load('latent_embeddings_output/latent_embeddings.mat');
embeddings = data.latent_embeddings;  % N×32 matrix
filenames = data.filenames;

% Clustering
[idx, C] = kmeans(embeddings, 5);

% Dimensionality reduction
[coeff, score, ~] = pca(embeddings);
scatter(score(:,1), score(:,2), 10, idx, 'filled');
```

### Load Reconstructions in MATLAB

```matlab
% Load reconstructed spectrograms
data = load('reconstructed_spectrograms.mat');
recons = data.reconstructed_spectrograms;  % N×121×104

% Visualize
figure;
for i = 1:9
    subplot(3,3,i);
    imagesc(squeeze(recons(i,:,:)));
    colormap jet;
    title(sprintf('Sample %d', i));
end
```

---

## Advanced Examples

### Compare LD16 vs LD32 Reconstructions

```bash
# Extract with LD16
python3 extract_latent_embeddings.py \
    --model LD16/.../autoencoder.pth \
    --latent_dim 16 \
    --input_dir test_data \
    --output_dir ld16_latent

# Extract with LD32
python3 extract_latent_embeddings.py \
    --model LD32/.../autoencoder.pth \
    --latent_dim 32 \
    --input_dir test_data \
    --output_dir ld32_latent

# Reconstruct both
python3 reconstruct_from_latent.py --model LD16/.../autoencoder.pth --latent_dim 16 --input ld16_latent/latent_embeddings.mat --output ld16_recon.mat --save_images --image_dir ld16_images
python3 reconstruct_from_latent.py --model LD32/.../autoencoder.pth --latent_dim 32 --input ld32_latent/latent_embeddings.mat --output ld32_recon.mat --save_images --image_dir ld32_images

# Compare visually in ld16_images/ vs ld32_images/
```

### Batch Process All Datasets

```bash
#!/bin/bash
# extract_all_datasets.sh

MODELS=(
    "LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir/trained_model/autoencoder.pth"
    "LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder.pth"
)

DATASETS=(
    "BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir"
    "BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir"
)

for model in "${MODELS[@]}"; do
    # Determine latent_dim from path
    if [[ $model == *"LD16"* ]]; then
        latent_dim=16
        tag="LD16"
    else
        latent_dim=32
        tag="LD32"
    fi
    
    for dataset in "${DATASETS[@]}"; do
        dataset_name=$(basename "$dataset" .dir)
        output_dir="embeddings_${tag}_${dataset_name}"
        
        echo "Processing: $model + $dataset → $output_dir"
        
        python3 extract_latent_embeddings.py \
            --model "$model" \
            --latent_dim $latent_dim \
            --input_dir "$dataset" \
            --output_dir "$output_dir" \
            --batch_size 64
    done
done
```

---

## Summary

✓ **Encoder Script**: Extract latent embeddings for analysis  
✓ **Decoder Script**: Reconstruct spectrograms from embeddings  
✓ **Frozen Weights**: Inference only, no training  
✓ **Batch Processing**: Efficient parallel processing  
✓ **Multiple Formats**: `.mat` files + optional PNG images  
✓ **Compatible**: All LD16/LD32 trained models  

For questions or issues, see the main project README or training script documentation.
