# Bowhead Whale Autoencoder Architecture Diagram

## Overview
This diagram visualizes the architecture of the autoencoder defined in `python_scripts/Autoencoder_v02_MultiGram_20260211.py`.

## Generated Files
- **Source**: `PlotNeuralNet/pyexamples/bowhead_autoencoder.py`
- **PDF**: `bowhead_autoencoder.pdf` (also copied to project root)

## Architecture Summary

### Input
- **Shape**: [2, 121, 104]
- **Channels**: 2 (SNR + NTV spectrograms)
- **Dimensions**: 121 frequency bins × 104 time bins

### Encoder (3 Convolutional Blocks)
1. **Block 1**: Conv2d(2→32, 3×3) + BatchNorm + ReLU + MaxPool(2)
   - Output: [32, 60, 52]
2. **Block 2**: Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool(2)
   - Output: [64, 30, 26]
3. **Block 3**: Conv2d(64→128, 3×3) + BatchNorm + ReLU + MaxPool(2)
   - Output: [128, 15, 13]

### Latent Space (Bottleneck)
- **Flatten**: 128 × 15 × 13 = 24,960 features
- **FC1**: Linear(24960 → 64) + ReLU
- **FC2**: Linear(64 → 32) **← Latent representation (32-dim)**
- **FC3**: Linear(32 → 64) + ReLU
- **FC4**: Linear(64 → 24960) + ReLU
- **Reshape**: [128, 15, 13]

### Decoder (3 Transposed Convolutional Blocks)
1. **Block 1**: ConvTranspose2d(128→64, 2×2, stride=2) + BatchNorm + ReLU
   - Output: [64, 30, 26]
2. **Block 2**: ConvTranspose2d(64→32, 2×2, stride=2) + BatchNorm + ReLU
   - Output: [32, 60, 52]
3. **Block 3**: ConvTranspose2d(32→2, 2×2, stride=2)
   - Output: [2, 121, 104] (reconstructed SNR + NTV)

### Output
- **Shape**: [2, 121, 104]
- **Channels**: 2 (reconstructed SNR + NTV)

## Model Parameters
- **Total Parameters**: ~3,359,010
- **Latent Dimension**: 32
- **Base Channels**: 32
- **Training Loss**: MSE (Mean Squared Error)
- **Optimizer**: Adam (lr=0.001)

## How to Regenerate the Diagram

1. **Edit the architecture** (if needed):
   ```bash
   cd PlotNeuralNet/pyexamples
   nano bowhead_autoencoder.py
   ```

2. **Regenerate PDF**:
   ```bash
   cd PlotNeuralNet/pyexamples
   bash ../tikzmake.sh bowhead_autoencoder
   ```

   This will:
   - Run the Python script to generate `.tex` file
   - Compile with `pdflatex` to create PDF
   - Open the PDF automatically (macOS)

3. **Manual compilation** (if needed):
   ```bash
   cd PlotNeuralNet/pyexamples
   python3 bowhead_autoencoder.py    # Generates .tex file
   pdflatex bowhead_autoencoder.tex  # Compiles to PDF
   ```

## Customization Guide

### Modify Colors
In `bowhead_autoencoder.py`, colors are defined globally:
- `\ConvColor`: Convolutional layers (yellow-orange)
- `\PoolColor`: Pooling layers (red)
- `\UnpoolColor`: Unpooling/upsampling (blue-green)
- `\SoftmaxColor`: Fully connected layers (magenta)

### Adjust Layer Sizes
To change visual proportions, modify the `height`, `width`, and `depth` parameters in each layer:
```python
to_ConvConvRelu(name='enc1', s_filer=52, n_filer=(32,32), 
                offset="(2,0,0)", to="(input-east)", 
                width=(3,3),      # Visual thickness
                height=30,        # Spatial height
                depth=21,         # Spatial depth
                caption="Conv+BN+ReLU")
```

### Add/Remove Layers
- **4-layer variant**: Add extra Conv+Pool block in encoder and matching decoder block
- **Different latent dim**: Update `s_filer` parameter in latent space blocks
- **Skip connections**: Use `to_skip()` function (see U-Net example)

## Architecture Variants

### Extra Convolutional Layer (--extra-conv)
When training with `--extra-conv`, the model uses 4 encoder/decoder blocks instead of 3:
- Additional pooling reduces spatial dimensions to [16, 7, 6]
- More features extracted at deeper levels

### Different Input Types
The model supports:
- `--gram-type SNR_gram`: Single-channel SNR only
- `--gram-type NTV_gram`: Single-channel NTV only
- `--gram-type BOTH`: 2-channel (SNR + NTV) - **shown in diagram**

## Related Files
- **Training script**: `python_scripts/Autoencoder_v02_MultiGram_20260211.py`
- **Architecture visualization**: `architecture_diagrams_latex/autoencoder_arch.tex`
- **PlotNeuralNet documentation**: `PlotNeuralNet/README.md`
- **Example reference**: `PlotNeuralNet/examples/AlexNet/alexnet.pdf`

## References
- PlotNeuralNet: https://github.com/HarisIqbal88/PlotNeuralNet
- Original paper architecture based on standard convolutional autoencoders
- Optimized for bioacoustic spectrogram analysis
