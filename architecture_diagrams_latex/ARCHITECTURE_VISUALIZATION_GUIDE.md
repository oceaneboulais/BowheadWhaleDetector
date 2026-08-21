# Autoencoder Architecture Visualization Guide

## Overview
The `visualize_architecture.py` script generates publication-quality block diagrams showing the complete architecture of your autoencoders.

## Generated Diagrams

### Three Diagram Styles

1. **Detailed Diagram** (`autoencoder_detailed_*.png`)
   - Horizontal flow from input to output
   - Shows every layer with precise dimensions
   - Includes all convolutional, pooling, and linear layers
   - Color-coded by operation type
   - Best for technical documentation

2. **Vertical Diagram** (`autoencoder_vertical_*.png`)
   - Top-to-bottom flow
   - Complete layer-by-layer breakdown
   - Useful for presentations and slides
   - Shows dimension transformations clearly

3. **Simplified Diagram** (`autoencoder_simple_*.png`)
   - High-level overview
   - Shows only major components (Encoder → Latent → Decoder)
   - Includes compression ratio
   - Perfect for papers and presentations

## Current Diagrams Generated

### Standard (3-layer) Architecture
- ✅ **16D Latent, 32 Channels**: `autoencoder_*_16LD_32C_3L.png`
- ✅ **32D Latent, 32 Channels**: `autoencoder_*_32LD_32C_3L.png`

### Hybrid (4-layer) Architecture
- ✅ **32D Latent, 32 Channels, Extra Conv**: `autoencoder_*_32LD_32C_4L.png`

All diagrams saved to: `architecture_diagrams/`

## Usage

### Basic Usage
```bash
# Generate all diagrams for 32D latent, 32 channels
python3 python_scripts/visualize_architecture.py \
    --latent-dim 32 \
    --channels 32 \
    --dpi 300
```

### Custom Configuration
```bash
# 16D latent dimension
python3 python_scripts/visualize_architecture.py --latent-dim 16

# 4-layer architecture (extra_conv)
python3 python_scripts/visualize_architecture.py --extra-conv

# Generate only simplified diagram
python3 python_scripts/visualize_architecture.py --style simple

# High-resolution for publication (600 DPI)
python3 python_scripts/visualize_architecture.py --dpi 600

# Custom input dimensions
python3 python_scripts/visualize_architecture.py --nrow 120 --ncol 104
```

### Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--nrow` | 121 | Input spectrogram height |
| `--ncol` | 104 | Input spectrogram width |
| `--latent-dim` | 32 | Latent space dimensionality |
| `--channels` | 32 | Base number of channels |
| `--extra-conv` | False | Use 4-layer architecture |
| `--output-dir` | architecture_diagrams | Output directory |
| `--dpi` | 300 | Image resolution |
| `--style` | all | Diagram style: all, detailed, vertical, simple |

## Architecture Details

### 3-Layer (Standard) Architecture
```
Input (121×104) 
  → Conv1+BN+ReLU (32 channels)
  → MaxPool (60×52)
  → Conv2+BN+ReLU (64 channels)
  → MaxPool (30×26)
  → Conv3+BN+ReLU (128 channels)
  → MaxPool (15×13)
  → Flatten
  → Linear (latent_dim * 2)
  → Latent (latent_dim)
  → Linear (latent_dim * 2)
  → Linear (15×13×128)
  → Reshape
  → Deconv1+BN+ReLU
  → Deconv2+BN+ReLU
  → Deconv3
  → Output (121×104)
```

**Compression Ratio**: 
- 32D latent: 12584 → 32 = **393×** compression
- 16D latent: 12584 → 16 = **786×** compression

### 4-Layer (Extra Conv) Architecture
```
Input (121×104)
  → 4 Conv+Pool layers (32 → 64 → 128 → 256 channels)
  → Flatten
  → Linear layers
  → Latent (latent_dim)
  → Linear layers
  → Reshape
  → 4 Deconv layers
  → Output (121×104)
```

**Compression Ratio**: 393× (for 32D latent)

## Color Coding

The diagrams use consistent color coding:
- 🔵 **Blue**: Input/Output layers
- 🟣 **Purple**: Convolution layers (+ BatchNorm + ReLU)
- 🟠 **Orange**: MaxPool layers
- 🟢 **Green**: Linear (fully connected) layers
- 🔴 **Red**: Latent space
- 🟥 **Dark Pink**: Deconvolution layers (+ BatchNorm + ReLU)

## Examples

### For Your Current Models

#### LD32 Models (Standard)
```bash
# Matches: Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K
python3 python_scripts/visualize_architecture.py \
    --latent-dim 32 \
    --channels 32
```

#### LD16 Models (Standard)  
```bash
# Matches: Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K
python3 python_scripts/visualize_architecture.py \
    --latent-dim 16 \
    --channels 32
```

#### LD32 Hybrid (4-layer)
```bash
# Matches: Autoencoder_v15_*_Hybrid_*
python3 python_scripts/visualize_architecture.py \
    --latent-dim 32 \
    --channels 32 \
    --extra-conv
```

## Quick Reference Commands

### Generate All Variants
```bash
# Standard architectures
python3 python_scripts/visualize_architecture.py --latent-dim 16 --channels 32
python3 python_scripts/visualize_architecture.py --latent-dim 32 --channels 32

# Hybrid architecture
python3 python_scripts/visualize_architecture.py --latent-dim 32 --channels 32 --extra-conv
```

### High-Quality Publication Export
```bash
# 600 DPI for journal publications
python3 python_scripts/visualize_architecture.py \
    --latent-dim 32 \
    --channels 32 \
    --dpi 600 \
    --style simple
```

### Presentation Slides
```bash
# Use simplified or vertical for PowerPoint/Keynote
python3 python_scripts/visualize_architecture.py \
    --latent-dim 32 \
    --style vertical \
    --dpi 150
```

## File Sizes

Typical file sizes at 300 DPI:
- Detailed: ~280-305 KB
- Vertical: ~265-300 KB  
- Simplified: ~140-145 KB

## Tips

1. **For papers**: Use simplified diagram at 600 DPI
2. **For presentations**: Use vertical or simplified at 150-300 DPI
3. **For documentation**: Use detailed diagram at 300 DPI
4. **For posters**: Use detailed at 600 DPI

## Customization

The script can be easily modified to:
- Change color schemes (edit `self.colors` dictionary)
- Adjust box sizes and spacing
- Add additional annotations
- Include layer parameters (kernel size, stride, etc.)

## Troubleshooting

**Low resolution output**: Increase `--dpi` value
**Overlapping labels**: Adjust font sizes in the script
**Missing matplotlib**: Install with `pip install matplotlib`

## Output Location

All diagrams are saved to: `architecture_diagrams/`

File naming convention: `autoencoder_{style}_{latent_dim}LD_{channels}C_{layers}L.png`
- `style`: detailed, vertical, or simple
- `latent_dim`: 16 or 32
- `channels`: 32 (base channels)
- `layers`: 3L (standard) or 4L (extra_conv)
