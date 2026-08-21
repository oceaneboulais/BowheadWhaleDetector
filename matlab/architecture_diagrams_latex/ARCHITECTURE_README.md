# Bowhead Whale Autoencoder Architecture Diagrams

This directory contains clear, professional architecture diagrams for the Bowhead whale call autoencoder.

## Files

1. **bowhead_autoencoder_simple.tex** - Standalone LaTeX diagram with large, non-overlapping text
2. **bowhead_autoencoder_architecture.py** - PlotNeuralNet-based generator (alternative)

## Quick Start

### Option 1: Compile Locally (Recommended)

```bash
cd architecture_diagrams_latex
pdflatex bowhead_autoencoder_simple.tex
```

This will generate `bowhead_autoencoder_simple.pdf` with a clear block diagram.

### Option 2: Use Overleaf (No installation required)

1. Go to [Overleaf](https://www.overleaf.com)
2. Create new project → Upload Project
3. Upload `bowhead_autoencoder_simple.tex`
4. Click "Recompile"
5. Download PDF

### Option 3: Generate with PlotNeuralNet

```bash
cd architecture_diagrams_latex
python3 bowhead_autoencoder_architecture.py
pdflatex bowhead_autoencoder_architecture.tex
```

## Architecture Overview

The diagram shows:

### **ENCODER** (Compression)
- **Input:** 1×121×104 whale call spectrogram
- **Conv1:** 5×5 kernel, 32 filters (captures broad N/U call patterns)
- **Conv2:** 3×3 kernel, 64 filters (refines features)
- **Conv3:** 3×3 kernel, 128 filters (deep features)
- **Flatten:** 26,624 features
- **Dense layers:** 64 → **32 (latent space)**

### **BOTTLENECK**
- **Latent dimension:** 32 (compressed representation)
- **Compression ratio:** 378:1

### **DECODER** (Reconstruction)
- Mirror of encoder with transposed convolutions
- Reconstructs 1×121×104 output

## Design Features

✅ **Large text** - All labels use \Large or \LARGE font sizes  
✅ **Non-overlapping** - Proper spacing with tikz positioning  
✅ **Color-coded** - Different colors for encoder/decoder/latent  
✅ **Annotations** - Model stats and training config included  
✅ **Professional** - Drop shadows, proper alignment, clear arrows

## Requirements

- **pdflatex** (from TeX Live, MiKTeX, or MacTeX)
- **tikz package** (usually included)

### Install LaTeX on macOS:
```bash
brew install --cask mactex
```

### Install LaTeX on Linux:
```bash
sudo apt-get install texlive-full
```

## Customization

To modify the diagram, edit `bowhead_autoencoder_simple.tex`:

- **Colors:** Defined at top with `\definecolor`
- **Sizes:** Adjust `minimum width` and `minimum height` in `block` style
- **Spacing:** Change `node distance` parameter
- **Fonts:** Modify `\Large`, `\LARGE`, `\Huge` commands

## Output

The PDF shows:
- Clear block diagram with flow from input → latent → output
- Layer specifications (kernel sizes, filters, dimensions)
- Model statistics (3.3M parameters, 32-dim latent)
- Training configuration
- Architectural explanation

Perfect for papers, presentations, or documentation!
