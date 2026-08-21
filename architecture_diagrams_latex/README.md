# PlotNeuralNet Autoencoder Diagrams

## Quick Start

### 1. Install PlotNeuralNet
```bash
cd /Users/oboulais/Public/Bowhead_DL_Project
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
```

### 2. Install LaTeX (if needed)
```bash
# macOS
brew install --cask mactex

# Ubuntu/Debian
sudo apt-get install texlive-full

# After install, refresh PATH
eval "$(/usr/libexec/path_helper)"
```

### 3. Generate Diagram
```bash
# Generate LaTeX code
python3 visualize_architecture_plotneuralnet.py --latent-dim 32 --channels 32

# Compile to PDF
cd architecture_diagrams_latex
pdflatex autoencoder_arch.tex
```

## Advantages of PlotNeuralNet

✅ **Publication-quality** 3D diagrams with proper LaTeX typography
✅ **Highly customizable** colors, sizes, and labels
✅ **Vector graphics** (PDF) - infinite zoom without quality loss
✅ **LaTeX integration** - use in papers, presentations, posters
✅ **Professional appearance** - commonly used in top-tier papers

## Comparison

| Feature | Matplotlib (current) | PlotNeuralNet |
|---------|---------------------|---------------|
| Output Format | PNG/JPG (raster) | PDF (vector) |
| Typography | Basic | LaTeX quality |
| 3D Rendering | 2D projections | True 3D with perspective |
| Customization | Python code | LaTeX styling |
| File Size (typical) | 200-500 KB | 50-150 KB |
| Zoom Quality | Pixelated at high zoom | Perfect at any zoom |
| Use in Papers | Acceptable | Preferred |

## Generated Files

- `autoencoder_arch.tex` - LaTeX source code
- `autoencoder_arch.pdf` - Compiled diagram (publication-ready)
- `autoencoder_arch.aux`, `.log` - LaTeX compilation artifacts (can delete)

## Customization

Edit the generated `.tex` file to customize:
- Colors: `\def\ConvColor{rgb:yellow,5;red,2.5;white,5}`
- Spacing: `shift={(X,Y,Z)}` coordinates
- Labels: `caption=`, `xlabel=`, `zlabel=`
- Sizes: `height=`, `width=`, `depth=`

## Troubleshooting

**Error: `! LaTeX Error: File 'init.tex' not found`**
- Solution: Make sure PlotNeuralNet is cloned in the correct location
- Check path: `/Users/oboulais/Public/Bowhead_DL_Project/PlotNeuralNet/`

**Error: `pdflatex: command not found`**
- Solution: Install MacTeX and refresh PATH
- Run: `eval "$(/usr/libexec/path_helper)"`

**Diagram looks wrong/misaligned**
- Solution: Adjust spacing in `shift={(x,y,z)}` values
- Increase x-offset between layers for more spacing

## Example Usage

```bash
# Default 121×104 → 32D
python3 visualize_architecture_plotneuralnet.py

# Custom configuration
python3 visualize_architecture_plotneuralnet.py \
    --latent-dim 16 \
    --channels 64 \
    --extra-conv \
    --in-channels 2

# Compile
cd architecture_diagrams_latex
pdflatex autoencoder_arch.tex
open autoencoder_arch.pdf  # macOS
```

## Integration with Papers

To include in LaTeX documents:
```latex
\documentclass{article}
\usepackage{graphicx}
\begin{document}
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{architecture_diagrams_latex/autoencoder_arch.pdf}
    \caption{Autoencoder architecture with 32D latent space.}
    \label{fig:autoencoder}
\end{figure}
\end{document}
```
