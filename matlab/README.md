# Bowhead Whale Deep Learning Detection & Classification System

## Overview

This project develops deep learning methods for detecting, classifying, and range-estimating bowhead whale calls in the Alaskan Beaufort Sea. The work builds upon previously established automated procedures for detecting and localizing frequency-modulated bowhead whale sounds in the presence of seismic airgun surveys, as described in Thode et al. (2012).

The original automated procedure was applied to four years of data collected from over 30 directional autonomous recording packages deployed across a 280 km span of continental shelf in the Alaskan Beaufort Sea. That system used cascaded neural networks trained on 219,471 manually flagged bowhead call examples from 2008 and 2009, achieving comparable spatial and temporal call distributions to manual analysis methods while processing substantially more data.

This repository extends that foundation by implementing modern deep learning architectures using PyTorch, focusing on unsupervised and semi-supervised approaches for feature extraction and call classification.

## Acknowledgments

This work was recommended to the Secretary of Commerce for funding under project 4509 (2410): "Development of a deep learning bowhead whale call detector, classifier, and range estimator."

## Repository Structure

### `/Pytorch_scripts/`
Core deep learning implementation scripts using PyTorch for training and applying autoencoder models to bowhead whale acoustic data.

#### Main Training Scripts

- **`Autoencoder_v02_LD16_20251118.py`** - Convolutional autoencoder with 16-dimensional latent space. Optimized for quick iterations with reduced output samples and efficient visualization. Includes t-SNE and UMAP dimensionality reduction for latent space exploration.

- **`Autoencoder_v02_LD32_20251118.py`** - Convolutional autoencoder with 32-dimensional latent space. Similar architecture to LD16 but with higher-dimensional latent representation for capturing more complex acoustic features.

- **`Autooencoder_11132025.py`** - Earlier autoencoder implementation with alternative architecture configurations.

#### Application Scripts

- **`Apply_Autoencoder.py`** - Inference script that loads a trained autoencoder model and extracts latent space vectors from acoustic spectrogram images. Processes images in batches and saves results as MATLAB-compatible .mat files for downstream analysis.

#### Utility Scripts

- **`check_dataset_size.py`** - Diagnostic tool for verifying dataset file counts in airgun and whale call directories. Helps ensure data availability before training runs.

#### Archived Scripts

The `Archived_Scripts/` directory contains earlier development versions and experimental implementations, including:
- Alternative autoencoder architectures
- Jupyter notebook implementations
- Diagnostic and demonstration scripts for analyzing model behavior
- Reconstruction quality assessment tools

### `/matlab/`
MATLAB preprocessing and analysis scripts for:
- Creating spectrogram datasets from raw acoustic data
- Computing directional and bearing metrics
- Evaluating overlap between manual and automated detections
- Clustering analysis and visualization
- Database indexing and assembly

Key master scripts:
- `master_create_datasets.m` - Generate spectrogram datasets
- `master_cluster_analysis.m` - Perform clustering analysis on features
- `master_replot_cluster_analysis.m` - Generate t-SNE visualizations
- `master_assemble_unsupervised_database.m` - Assemble unsupervised training database

### `/models/`
Trained model checkpoints and unsupervised databases:
- `combined_ae_*/` - Combined autoencoder models
- `trained_models/` - Production model weights
- `Unsupervised_database_*.dir/` - Preprocessed acoustic feature databases

### `/plots/`
Model visualizations, training curves, and analysis figures including:
- Model architecture schematics
- Training loss plots
- t-SNE and UMAP embeddings of latent space
- Cluster analysis results

### `/results/`
Training outputs, extracted features, and analysis results from model runs.

### `/Azigram_pulse_detector.dir/`
Specialized scripts for detecting pulses in azimuthal spectrograms (azigrams), including weighted median calculations and master tracking routines.

### `/OutreachProject/`
Interactive browser-based tools for education and outreach:
- **Whale Call Theremin** - Interactive Theremin-style synthesizer for playing back bowhead whale calls
- Browser-based GUI with real-time frequency visualization
- 7 call types with ~70,000 classified examples from the Beaufort Sea
- Extracts frequency contours from spectrograms and synthesizes audio using Web Audio API
- See [OutreachProject/README_WHALE_PLAYER.md](OutreachProject/README_WHALE_PLAYER.md) for usage instructions

## Workflow

### Data Preparation (MATLAB)
1. Extract spectrograms from raw GSI acoustic recordings
2. Apply directional processing and bearing extraction
3. Assemble unsupervised databases with and without airgun contamination
4. Generate normalized spectrogram images for deep learning

### Model Training (PyTorch)
1. Configure autoencoder architecture (latent dimension, channels, layers)
2. Train on spectrogram image databases
3. Monitor reconstruction quality and latent space organization
4. Extract embeddings for clustering and classification

### Analysis and Visualization
1. Apply trained encoder to new acoustic data
2. Extract latent feature vectors
3. Perform clustering analysis (K-means, hierarchical)
4. Visualize embeddings with t-SNE/UMAP
5. Evaluate detection and classification performance

## Dependencies

### Python Environment
- PyTorch (deep learning framework)
- NumPy, SciPy (numerical computing)
- scikit-learn (clustering, dimensionality reduction)
- UMAP-learn (manifold learning)
- matplotlib (visualization)
- tqdm (progress tracking)

A virtual environment configuration is provided in `.venv_py31018/`.

### MATLAB
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- Image Processing Toolbox (for spectrogram generation)

## References

Thode, A. M., Kim, K. H., Blackwell, S. B., Greene, C. R., Jr., Conrad, A. S., & Michael, A. (2012). Automated detection and localization of bowhead whale sounds in the presence of seismic airgun surveys. *The Journal of the Acoustical Society of America*, 131(5), 3726–3747. https://doi.org/10.1121/1.3699247

v0.01 Autoencoder architecture inspired by Duane, Daniel & Kroeger, Nicholas & Freeman, Simon & Freeman, Lauren. (2025). Unsupervised clustering of biological sounds in a Hawaiian coral reef. The Journal of the Acoustical Society of America. 157. A49-A49. 10.1121/10.0037353. 

## Contact

For questions about this codebase, please refer to the repository maintainers or the associated project documentation.

---

*This research contributes to long-term acoustic monitoring efforts in the Arctic and advances automated bioacoustic analysis methods for marine mammal conservation.*
