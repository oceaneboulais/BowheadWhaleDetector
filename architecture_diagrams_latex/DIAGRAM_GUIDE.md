# Bowhead Whale Autoencoder Architecture Diagram

## Generated Files

✅ **High-quality 3D architecture diagram created!**

### Files:
- [`bowhead_autoencoder_with_samples.pdf`](bowhead_autoencoder_with_samples.pdf) - Vector format (288KB, scalable)
- [`bowhead_autoencoder_with_samples.png`](bowhead_autoencoder_with_samples.png) - Raster format (86KB)
- [`bowhead_autoencoder_simple.pdf`](bowhead_autoencoder_simple.pdf) - Alternative block diagram (77KB)

## Diagram Features

### ✅ Clear 3D Visualization
- **Large, non-overlapping text** on all layers
- **3D blocks** showing depth and channels
- **Sample spectrograms** on left (actual whale calls)
- **Horizontal flow** (input → latent → output)
- **Color-coded** layers for easy identification

### Architecture Details Shown

**ENCODER (Left to Right):**
1. **Input:** 1-channel spectrogram (1×121×104)
2. **Conv1:** 5×5 kernel, 32 filters **(HYBRID - captures N/U patterns)**
3. **MaxPool** → 32×60×52
4. **Conv2:** 3×3 kernel, 64 filters
5. **MaxPool** → 64×30×26
6. **Conv3:** 3×3 kernel, 128 filters
7. **MaxPool** → 128×15×13
8. **FC 64:** Dense layer
9. **LATENT:** 32-dimensional bottleneck

**DECODER (Right):**
10. **FC 64:** Dense layer
11. **Reshape:** Back to 128×15×13
12. **TransConv1:** 64 filters → 64×30×26
13. **TransConv2:** 32 filters → 32×60×52
14. **TransConv3:** 1 filter → **1×121×104 reconstruction**

## Key Innovations

🔬 **Hybrid Architecture:**
- First convolutional layer uses **5×5 kernel** to capture broad N-shaped and U-shaped whale call patterns
- Subsequent layers use efficient **3×3 kernels** for detail refinement
- This design optimally captures the sweep patterns characteristic of bowhead whale calls

📊 **Specifications:**
- **Parameters:** 3,359,105
- **Latent Dimension:** 32
- **Compression Ratio:** 378:1
- **Input:** Single-channel spectrogram (SNR_gram or NTV_gram)
- **Training:** MSE loss, Adam optimizer, Batch Normalization

## How to Regenerate

If you need to update the diagram with new architecture changes:

```bash
cd /Users/oboulais/Public/Bowhead_DL_Project/PlotNeuralNet/pyexamples

# Edit the Python script if needed
nano bowhead_autoencoder_with_samples.py

# Regenerate
python3 bowhead_autoencoder_with_samples.py

# Copy to main directory
cp bowhead_autoencoder_with_samples.{pdf,png} ../../architecture_diagrams_latex/
```

## Source Files

- **Generator Script:** `/PlotNeuralNet/pyexamples/bowhead_autoencoder_with_samples.py`
- **Sample Images:** `/PlotNeuralNet/pyexamples/sample_images/`
- **LaTeX Source:** `bowhead_autoencoder_with_samples.tex` (auto-generated)

## Usage

### For Publications:
Use the **PDF version** - it's vector-based and scales perfectly for papers and posters.

### For Presentations:
Use the **PNG version** - renders quickly in PowerPoint/Keynote.

### For Documentation:
Either format works. The PDF has better quality when zooming.

## Customization

To modify the diagram appearance, edit `bowhead_autoencoder_with_samples.py`:

```python
# Adjust layer sizes (width, height, depth)
to_Conv(name='input', width=1, height=40, depth=34, ...)

# Change labels
caption="Your Custom Label\\\\Details"

# Adjust spacing between layers
offset="(4.5,0,0)"  # Increase/decrease for more/less space

# Modify colors
# Colors are defined in the generated LaTeX preamble
```

## Technical Notes

- Built with **PlotNeuralNet** (3D LaTeX/TikZ visualization)
- Requires: `pdflatex`, `convert` (ImageMagick for PNG generation)
- LaTeX packages: `tikz`, `standalone`, `mathptmx`

Perfect for papers, presentations, and documentation! 📊🐋
