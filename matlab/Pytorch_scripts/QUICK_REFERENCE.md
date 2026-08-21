# Quick Reference - Encoder/Decoder Scripts

## ✅ Scripts Created and Tested

1. **`extract_latent_embeddings.py`** - Extract latent embeddings (encoder)
2. **`reconstruct_from_latent.py`** - Reconstruct spectrograms (decoder)

Both scripts **freeze model weights** for inference only.

---

## Your Models

| Model | Path | Latent Dim |
|-------|------|-----------|
| **LD16 v13 AutoManual** | `LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir/trained_model/autoencoder_clean.pth` | 16 |
| **LD16 v14 Manual** | `LD16/Autoencoder_v14_100E_16LD_32C_Manual_100K_Date20260122-190106.dir/trained_model/autoencoder_clean.pth` | 16 |
| **LD32 v13 AutoManual** | `LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder_clean.pth` | 32 |
| **LD32 v14 Manual** | `LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir/trained_model/autoencoder_clean.pth` | 32 |

---

## Common Commands

### Extract Latent Embeddings

```bash
# Activate environment
source venv_bowhead/bin/activate

# Single file - LD32
python3 extract_latent_embeddings.py \
    --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder_clean.pth \
    --input BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir/S308A0T20080828T000045_Type0.mat \
    --output latent_embedding.mat

# Batch directory - LD32 (100K files)
python3 extract_latent_embeddings.py \
    --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder_clean.pth \
    --input_dir BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir \
    --output_dir latent_embeddings_output \
    --batch_size 64

# Batch directory - LD16
python3 extract_latent_embeddings.py \
    --model LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir/trained_model/autoencoder_clean.pth \
    --latent_dim 16 \
    --input_dir BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir \
    --output_dir latent_embeddings_ld16 \
    --batch_size 64
```

### Reconstruct Spectrograms

```bash
# From single latent embedding
python3 reconstruct_from_latent.py \
    --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder_clean.pth \
    --input latent_embedding.mat \
    --output reconstructed.mat

# From batch (with image visualization)
python3 reconstruct_from_latent.py \
    --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder_clean.pth \
    --input latent_embeddings_output/latent_embeddings.mat \
    --output reconstructed_batch.mat \
    --save_images \
    --image_dir reconstruction_images \
    --max_images 50
```

---

## Test Results ✅

### Single File Test
```
✓ Extracted: S308A0T20080828T000045_Type0.mat → (32,)
  Processing time: ~5ms
  Output: test_single_latent.mat (520 bytes)
```

### Batch Test (100 files)
```
✓ EXTRACTION COMPLETE
  Total files: 100
  Output shape: (100, 32)
  Total time: 0.5s (217 files/sec)
  Output: test_batch_latent/latent_embeddings.mat (50 KB)
```

### Reconstruction Test (100 samples)
```
✓ RECONSTRUCTION COMPLETE
  Total samples: 100
  Output shape: (100, 121, 104)
  Total time: 0.3s (360 samples/sec)
  Output: test_batch_reconstruction.mat (4.8 MB)
  Images: 10 PNG files (51-57 KB each)
```

---

## Performance Estimates

### Your 100K Dataset

| Operation | LD16 | LD32 | Time (MPS) |
|-----------|------|------|------------|
| Extract 100K embeddings | 6.8 MB output | 13 MB output | ~8 min |
| Reconstruct 100K spectrograms | 4.8 GB output | 4.8 GB output | ~5 min |

**Device**: Apple Silicon MPS (Metal Performance Shaders)  
**Speed**: ~200-360 samples/sec for both encoding and decoding

---

## Common Use Cases

### 1. Feature Extraction for Clustering

```bash
# Extract embeddings
python3 extract_latent_embeddings.py \
    --model LD32/.../autoencoder_clean.pth \
    --input_dir BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir \
    --output_dir embeddings_for_clustering

# Use in MATLAB
# load('embeddings_for_clustering/latent_embeddings.mat')
# [idx, C] = kmeans(latent_embeddings, 5);
```

### 2. Quality Assessment (Round-Trip)

```bash
# Extract then reconstruct
python3 extract_latent_embeddings.py --model LD32/.../autoencoder_clean.pth --input_dir test_data --output_dir test_latent --file_limit 100
python3 reconstruct_from_latent.py --model LD32/.../autoencoder_clean.pth --input test_latent/latent_embeddings.mat --output test_recon.mat --save_images --image_dir comparison_images
```

### 3. Denoising Pipeline

```bash
# Run noisy spectrograms through encoder→decoder
python3 extract_latent_embeddings.py --model LD32/.../autoencoder_clean.pth --input noisy.mat --output latent.mat
python3 reconstruct_from_latent.py --model LD32/.../autoencoder_clean.pth --input latent.mat --output denoised.mat
```

---

## Output Formats

### Latent Embeddings (.mat)

**Single file**:
```
latent_embedding    (32,)
original_shape      [121, 104]
input_file          'S308A0T20080828T000045_Type0.mat'
```

**Batch**:
```
latent_embeddings   (N, 32)
filenames           {N×1 cell}
num_samples         N
latent_dim          32
```

### Reconstructed Spectrograms (.mat)

```
reconstructed_spectrograms   (N, 121, 104)
filenames                    {N×1 cell}
num_samples                  N
shape                        [N, 121, 104]
```

### Visualization (.png images)

- Resolution: 150 DPI
- Size: ~50-60 KB per image
- Colormap: viridis
- Axes: Time (0-3s) × Frequency (0-500 Hz)

---

## Tips

1. **Use `--file_limit` for testing**: Test on 100 files first before processing all 100K
2. **Adjust `--batch_size`**: Reduce if you get memory errors (64 → 32 → 16)
3. **Save images sparingly**: Use `--max_images 100` instead of processing all 100K
4. **Monitor GPU/MPS memory**: Scripts automatically use MPS on Apple Silicon
5. **Match architecture parameters**: Use `--latent_dim 16` for LD16 models

---

## Troubleshooting

**"FileNotFoundError: autoencoder.pth"**  
→ Model files are named `autoencoder_clean.pth` not `autoencoder.pth`

**"Arrays have incompatible sizes"**  
→ Use `--latent_dim 16` for LD16 models, `--latent_dim 32` for LD32 models

**"CUDA/MPS out of memory"**  
→ Reduce batch size: `--batch_size 32` or `--batch_size 16`

**Slow performance**  
→ Check device: Script should show "Device: mps" on Mac, "Device: cuda" on NVIDIA GPU

---

## Documentation

See **[ENCODER_DECODER_GUIDE.md](ENCODER_DECODER_GUIDE.md)** for complete documentation including:
- All command line arguments
- Advanced examples
- MATLAB integration
- Architecture details
- Performance benchmarks

---

## Summary

✅ **Both scripts tested and working**  
✅ **Processing speed: 200-360 samples/sec on MPS**  
✅ **Frozen weights (inference only)**  
✅ **Support all 4 trained models (LD16/LD32, v13/v14)**  
✅ **Output formats: .mat files + optional PNG images**  

Ready to use with your full 100K dataset!
