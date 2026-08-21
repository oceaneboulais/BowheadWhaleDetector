#!/usr/bin/env python3
"""
GUARANTEED FRESH START:
- No model loading or transfer learning
- No checkpoint resuming
- Random initialization only (controlled by seed)
- Each run completely independent

PERFORMANCE OPTIMIZATIONS:
- Minimal output samples (30 instead of 5000) for faster JPEG generation
- Limited t-SNE samples (30 instead of 1000) for instant visualization
- Non-blocking plots (plt.close() instead of plt.show())
- Efficient DataLoader streaming from disk

FULL_DATASET MODE:
- Set FULL_DATASET = True or use --full-dataset flag
- Recursively discovers ALL .mat files in /Users/thode/Public/Spectrogram_Image_Database.dir
- Generates interactive 3D UMAP visualization (HTML)
- Comprehensive latent space analysis across entire database

USAGE:

# Normal mode (default datasets):
python3 Autoencoder_v03_20251212_Hybrid_Filter.py

# Full dataset mode (comprehensive database):
python3 Autoencoder_v03_20251212_Hybrid_Filter.py --full-dataset

"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for faster plotting
import matplotlib.pyplot as plt
import os
import sys
import glob
import math
import torch.nn.functional as F
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import Union, Optional, Tuple, List
from datetime import datetime
import time
import gc  # For explicit garbage collection
try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:
    TSNE = None
    KMeans = None
    silhouette_score = None

try:
    import umap
    UMAP = umap.UMAP
except Exception:
    UMAP = None
import argparse

# ============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ============================================================================

# Dataset parameters
FULL_DATASET = False  # When True, recursively discovers ALL .mat files in all subdirectories
FULL_DATASET_PATH = "/Users/thode/Public/Spectrogram_Image_Database.dir"  # Path used when FULL_DATASET=True

# Architecture parameters
CHANNELS_DEFAULT = 32
LATENT_DIM_DEFAULT = 32
EXTRA_CONV_DEFAULT = False


# Training parameters (optimized for speed)
EPOCHS_DEFAULT = 100             # Full training for production
LR_DEFAULT = 1e-3
SEED_DEFAULT = 42

# Output parameters (minimized for speed)
NUMBER_OUTPUT_IMAGE_SAMPLES = 30   # Reduced from 5000 for faster JPEG generation
PANEL_GROUP_SIZE = 3
SHOW_ERROR_PLOTS = False
ENABLE_UMAP = True  # Set to False to skip UMAP computation
DEFAULT_VERSION_TAG = "15_100E_32LD_32C_Hybrid_5x5_3x3_CombinedDatasets_100K"
TSNE_MAX_SAMPLES = None  # None => use all samples in dataset


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_global_seed(seed: int):
    """Set random seeds for reproducible initialization from scratch."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_output_directory(version_tag: Optional[str] = None) -> str:
    """Create unique timestamped output directory with organized subdirectories."""
    tag = (version_tag or DEFAULT_VERSION_TAG).strip().replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"Autoencoder_v{tag}_Date{timestamp}.dir"
    
    # Output to specified directory
    results_dir = "/Users/oboulais/Desktop/Bowhead_DL_Project/LD32_Hybrid"
    os.makedirs(results_dir, exist_ok=True)
    
    output_dir = os.path.join(results_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create organized subdirectories
    os.makedirs(os.path.join(output_dir, 'MATLAB'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image_results'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'trained_model'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'UMAP'), exist_ok=True)
    
    return output_dir


def ensure_dir_exists(filepath: str):
    """Ensure parent directory exists before writing file."""
    dir_path = os.path.dirname(filepath)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def _minmax_norm(im: np.ndarray, auto_skip_if_unit: bool = True) -> np.ndarray:
    """Min-max normalize image to [0, 1] range."""
    im = im.astype(np.float32)
    im_min = float(np.min(im))
    im_max = float(np.max(im))
    rng = im_max - im_min
    if rng < 1e-8:
        return np.zeros_like(im, dtype=np.float32)
    if auto_skip_if_unit and (-1e-4 <= im_min <= 1.0 + 1e-4) and (-1e-4 <= im_max <= 1.0 + 1e-4):
        return im
    return (im - im_min) / rng


# ============================================================================
# DATASET CLASS
# ============================================================================

class SNRDataset(Dataset):
    """Memory-efficient Dataset loading .mat files on-demand."""
    
    def __init__(self, directory: str, normalize: bool = True, 
                 seed: Optional[int] = None, show_summary: bool = False, full_dataset: bool = False):
        self.normalize = normalize
        self.file_paths: list[str] = []
        
        target_shape = None
        if full_dataset:
            print(f"  [FULL_DATASET MODE] Recursively scanning ALL subdirectories for .mat files in: {directory}")
        else:
            print(f"  Scanning for .mat files in: {directory}")
        sys.stdout.flush()
        
        # Recursively discover all .mat files (this already searches all subdirectories)
        mat_files = sorted(glob.glob(os.path.join(directory, '**', '*.mat'), recursive=True))
        total_files = len(mat_files)
        print(f"  Found {total_files} .mat files across all subdirectories, validating shapes...")
        sys.stdout.flush()
        
        for i, fp in enumerate(mat_files):
            # Progress update every 5000 files
            if i % 5000 == 0 and i > 0:
                print(f"    Validated {i}/{total_files} files ({100*i/total_files:.1f}%)...")
                sys.stdout.flush()
            try:
                m = loadmat(fp)
                im = m.get('SNR_gram', None)
                if im is None or not isinstance(im, np.ndarray) or im.ndim != 2:
                    continue
                if target_shape is None:
                    target_shape = im.shape
                if im.shape == target_shape:
                    self.file_paths.append(fp)
            except Exception:
                continue
        
        if not self.file_paths:
            raise RuntimeError(f"No valid .mat files found in {directory}")
        
        self.target_shape = target_shape
        
        if seed is not None:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(self.file_paths))
            self.file_paths = [self.file_paths[i] for i in indices]
        
        if show_summary:
            print(f"SNRDataset: {len(self)} files with shape {target_shape}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        try:
            m = loadmat(fp)
            im = m['SNR_gram']
            if self.normalize:
                im = _minmax_norm(im)
            else:
                im = im.astype(np.float32)
            tensor = torch.from_numpy(im).unsqueeze(0)
            return tensor, 0
        except Exception as e:
            print(f"Warning: Failed to load {fp}: {e}")
            h, w = self.target_shape
            return torch.zeros((1, h, w), dtype=torch.float32), 0


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ImprovedAutoencoder(nn.Module):
    """
    TRUE HYBRID Autoencoder with configurable first-layer kernel size.
    ALWAYS INITIALIZED FROM SCRATCH - NO PRETRAINED WEIGHTS.
    
    HYBRID ARCHITECTURE:
        - First conv layer: Uses kernel_size (default 5×5) to capture broad N/U curve shapes
        - Subsequent layers: Always use 3×3 to refine details efficiently
    
    This design captures the overall sweep pattern of whale calls in the first layer,
    then refines local features in deeper layers.
    
    Args:
        kernel_size: First layer kernel size (3 or 5). 5×5 recommended for N/U calls.
    """
    
    def __init__(self, nrow=121, ncol=104, latent_dim=LATENT_DIM_DEFAULT, 
                 base_channels=CHANNELS_DEFAULT, extra_conv=EXTRA_CONV_DEFAULT,
                 kernel_size=5):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        self.extra_conv = extra_conv
        self.kernel_size = kernel_size
        
        # HYBRID: First layer uses kernel_size, rest use 3×3
        first_padding = (kernel_size - 1) // 2  # 2 for 5×5, 1 for 3×3
        
        if extra_conv:
            nrow_reduced = nrow // 16
            ncol_reduced = ncol // 16
        else:
            nrow_reduced = nrow // 8
            ncol_reduced = ncol // 8
        
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        
        if extra_conv:
            self.encoder = nn.Sequential(
                # Layer 1: Large kernel (5×5) captures broad N/U curve patterns
                nn.Conv2d(1, c1, kernel_size, padding=first_padding),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # Layer 2+: Standard 3×3 kernels refine features
                nn.Conv2d(c1, c2, 3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(c2, c3, 3, padding=1),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(c3, c4, 3, padding=1),
                nn.BatchNorm2d(c4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            flat_size = c4 * nrow_reduced * ncol_reduced
        else:
            self.encoder = nn.Sequential(
                # Layer 1: Large kernel (5×5) captures broad N/U curve patterns
                nn.Conv2d(1, c1, kernel_size, padding=first_padding),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                # Layer 2+: Standard 3×3 kernels refine features
                nn.Conv2d(c1, c2, 3, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(c2, c3, 3, padding=1),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            flat_size = c3 * nrow_reduced * ncol_reduced
        
        self.to_latent = nn.Sequential(
            nn.Linear(flat_size, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, flat_size),
            nn.ReLU(inplace=True)
        )
        
        if extra_conv:
            pad_h = (nrow - nrow_reduced * 16) % 2
            pad_w = (ncol - ncol_reduced * 16) % 2
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(c4, c3, 2, stride=2),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c3, c2, 2, stride=2),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c2, c1, 2, stride=2),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c1, 1, 2, stride=2, output_padding=(pad_h, pad_w)),
            )
        else:
            pad_h = (nrow - nrow_reduced * 8) % 2
            pad_w = (ncol - ncol_reduced * 8) % 2
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(c3, c2, 2, stride=2),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c2, c1, 2, stride=2),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c1, 1, 2, stride=2, output_padding=(pad_h, pad_w)),
            )
        
        self.flat_size = flat_size
        self.nrow_reduced = nrow_reduced
        self.ncol_reduced = ncol_reduced
        self.base_channels = base_channels
        self.c_out = c4 if extra_conv else c3

    def forward(self, x):
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.to_latent(x_flat)
        x_recon = self.from_latent(latent)
        x_recon = x_recon.view(x_recon.size(0), self.c_out, self.nrow_reduced, self.ncol_reduced)
        output = self.decoder(x_recon)
        return output, latent


def match_shape_center(recon: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Center-crop or pad reconstruction to match target dimensions."""
    _, _, rH, rW = recon.shape
    tH, tW = target_hw
    if rH > tH:
        dh = (rH - tH) // 2
        recon = recon[:, :, dh:dh + tH, :]
        rH = tH
    if rW > tW:
        dw = (rW - tW) // 2
        recon = recon[:, :, :, dw:dw + tW]
        rW = tW
    padH = tH - rH
    padW = tW - rW
    if padH > 0 or padW > 0:
        pad_left = padW // 2 if padW > 0 else 0
        pad_right = padW - pad_left if padW > 0 else 0
        pad_top = padH // 2 if padH > 0 else 0
        pad_bottom = padH - pad_top if padH > 0 else 0
        recon = F.pad(recon, (pad_left, pad_right, pad_top, pad_bottom))
    return recon


def select_samples_for_outputs(dataset: Dataset, n_samples: int, seed: Optional[int]) -> Tuple[torch.Tensor, List[str]]:
    """Select random samples for JPEG panel generation."""
    if dataset is None or len(dataset) == 0:
        raise RuntimeError("No data available for output sampling")
    rng = np.random.default_rng(seed)
    total = len(dataset)
    k = min(n_samples, total)
    indices = rng.choice(total, size=k, replace=False) if total > k else np.arange(total)
    samples = []
    filenames = []
    
    # Collect all file paths (handle both single dataset and ConcatDataset)
    all_file_paths = []
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            if hasattr(ds, 'file_paths'):
                all_file_paths.extend(ds.file_paths)
    elif hasattr(dataset, 'file_paths'):
        all_file_paths = dataset.file_paths
    
    for idx in indices:
        sample, _ = dataset[int(idx)]
        samples.append(sample.unsqueeze(0))
        if all_file_paths and int(idx) < len(all_file_paths):
            filenames.append(os.path.basename(all_file_paths[int(idx)]))
        else:
            filenames.append(f"sample_{int(idx):06d}")
    return torch.cat(samples, dim=0).float(), filenames


def save_reconstruction_panels(model: nn.Module, samples: torch.Tensor, output_dir: str,
                               target_hw: tuple[int, int], base_name: str = "recon_panel",
                               dataset_label: str = "", filenames: list[str] = None, 
                               show_error: bool = SHOW_ERROR_PLOTS, epochs: int = 100,
                               latent_dim: int = 32, channels: int = 64, device: torch.device = None) -> int:
    """Save JPEG panels showing reconstructions with proper axis labels."""
    if samples is None or samples.shape[0] == 0:
        return 0
    if device is None:
        device = next(model.parameters()).device
    os.makedirs(output_dir, exist_ok=True)
    num_samples = samples.shape[0]
    group_count = math.ceil(num_samples / PANEL_GROUP_SIZE)
    panels_written = 0
    n_rows = 3 if show_error else 2
    
    # Spectrogram parameters for axis labels
    nrow, ncol = target_hw  # 121 rows (freq bins), 104 cols (time bins)
    freq_max_hz = 500.0  # Typical max frequency for whale calls
    time_duration_sec = 3.0  # Duration in seconds
    
    for group_idx in range(group_count):
        start = group_idx * PANEL_GROUP_SIZE
        end = min(start + PANEL_GROUP_SIZE, num_samples)
        batch = samples[start:end].to(device)
        with torch.no_grad():
            recon, _ = model(batch)
            recon = match_shape_center(recon, target_hw)
        orig_np = batch.squeeze(1).cpu().numpy()
        recon_np = recon.squeeze(1).cpu().numpy()
        
        cols = recon_np.shape[0]
        fig, axes = plt.subplots(n_rows, cols, figsize=(3.5 * cols, 3.5 * n_rows))
        if cols == 1:
            axes = np.expand_dims(axes, axis=1)
        
        for col in range(cols):
            sample_idx = start + col
            
            # Set filename as title
            if filenames and sample_idx < len(filenames) and filenames[sample_idx]:
                title = filenames[sample_idx].replace('.mat', '')
                if len(title) > 30:
                    title = title[:27] + '...'
            else:
                title = f'Input {sample_idx + 1}'
            
            # Original spectrogram
            im0 = axes[0, col].imshow(orig_np[col], cmap='viridis', origin='lower', aspect='auto',
                                      extent=[0, time_duration_sec, 0, freq_max_hz])
            axes[0, col].set_title(title, fontsize=7)
            axes[0, col].set_ylabel('Frequency (Hz)', fontsize=6)
            if col == 0:
                axes[0, col].tick_params(axis='both', labelsize=5)
            else:
                axes[0, col].set_yticks([])
            axes[0, col].set_xticks([])
            
            # Reconstruction
            im1 = axes[1, col].imshow(recon_np[col], cmap='viridis', origin='lower', aspect='auto',
                                      extent=[0, time_duration_sec, 0, freq_max_hz])
            axes[1, col].set_ylabel('Frequency (Hz)', fontsize=6)
            axes[1, col].set_xlabel('Time (s)', fontsize=6)
            if col == 0:
                axes[1, col].tick_params(axis='both', labelsize=5)
            else:
                axes[1, col].set_yticks([])
            axes[1, col].tick_params(axis='x', labelsize=5)
            
            if show_error:
                diff_np = np.abs(orig_np - recon_np)
                im2 = axes[2, col].imshow(diff_np[col], cmap='hot', origin='lower', aspect='auto',
                                         extent=[0, time_duration_sec, 0, freq_max_hz])
                axes[2, col].set_ylabel('Frequency (Hz)', fontsize=6)
                axes[2, col].set_xlabel('Time (s)', fontsize=6)
                if col == 0:
                    axes[2, col].tick_params(axis='both', labelsize=5)
                else:
                    axes[2, col].set_yticks([])
                axes[2, col].tick_params(axis='x', labelsize=5)
        
        # Add row labels on the left
        fig.text(0.02, 0.75 if not show_error else 0.83, 'Input', rotation=90, 
                va='center', ha='center', fontsize=8, weight='bold')
        fig.text(0.02, 0.5 if not show_error else 0.5, 'Reconstruction', rotation=90,
                va='center', ha='center', fontsize=8, weight='bold')
        if show_error:
            fig.text(0.02, 0.17, 'Error', rotation=90, va='center', ha='center', 
                    fontsize=8, weight='bold')
        
        # Add title at top with model parameters
        title_str = f'Autoencoder Reconstructions (epochs={epochs}, latent_dim={latent_dim}, channels={channels})'
        fig.suptitle(title_str, fontsize=10, y=0.98)
        
        # Add dataset label at bottom right
        if dataset_label:
            fig.text(0.98, 0.01, f'{dataset_label}', ha='right', va='bottom', 
                    fontsize=6, style='italic', alpha=0.6)
        
        panel_path = os.path.join(output_dir, f"{base_name}_{group_idx + 1:03d}.jpg")
        plt.tight_layout(rect=[0.03, 0.02, 1, 0.96])
        plt.savefig(panel_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        panels_written += 1
    return panels_written


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_autoencoder_from_scratch(data_dir: str, n_samples: int = 15, latent_dim: int = LATENT_DIM_DEFAULT,
                                   channels: int = CHANNELS_DEFAULT, seed: Optional[int] = SEED_DEFAULT,
                                   epochs: int = EPOCHS_DEFAULT, lr: float = LR_DEFAULT,
                                   tsne_samples: Optional[int] = None, extra_conv: bool = EXTRA_CONV_DEFAULT,
                                   batch_size: int = 32, output_samples: int = NUMBER_OUTPUT_IMAGE_SAMPLES,
                                   version_tag: str = DEFAULT_VERSION_TAG, show_error: bool = SHOW_ERROR_PLOTS,
                                   k_clusters: int = 2, tsne_perplexity: Optional[float] = None,
                                   kernel_size: int = 3, enable_umap: bool = ENABLE_UMAP, full_dataset: bool = FULL_DATASET):
    """
    
    FRESH START GUARANTEES:
    - New model initialized with random weights (controlled by seed)
    - No checkpoint loading or transfer learning
    - Independent of any previous runs
    - Deterministic if seed is set
    
    PERFORMANCE OPTIMIZATIONS:
    - Non-blocking plots (no plt.show())
    - Limited output samples for fast JPEG generation
    - Reduced t-SNE samples for instant visualization
    - Efficient DataLoader streaming
    - Explicit garbage collection after major operations
    """
    # Start timing
    script_start_time = time.time()
    
    # STEP 0: Set device (GPU if available, else CPU)
    # Prioritize MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: {device} (Apple Metal)")
        # MPS optimization: Enable fallback to CPU for unsupported ops
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print(f"MPS fallback enabled for unsupported operations")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        # Enable TF32 for faster training on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (WARNING: No GPU detected)")
    sys.stdout.flush()
    
    # STEP 1: Initialize random seed for reproducible FROM-SCRATCH initialization
    if seed is not None:
        set_global_seed(int(seed))
        print(f"Random seed set to {seed} for reproducible initialization")
    
    output_dir = create_output_directory(version_tag)
    print(f"="*70)
    print(f"FAST CLEAN AUTOENCODER - Training from Scratch")
    print(f"="*70)
    print(f"Output: {output_dir}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: epochs={epochs}, lr={lr}, latent_dim={latent_dim}, channels={channels}")
    print(f"="*70)
    
    # STEP 2: Load datasets from both directories
    if full_dataset:
        print(f"\n{'='*70}")
        print(f"[FULL_DATASET MODE ENABLED]")
        print(f"Recursively discovering ALL .mat files in all subdirectories")
        print(f"{'='*70}")
    
    if isinstance(data_dir, str):
        # Single directory (backward compatibility)
        print(f"\nLoading data from: {data_dir}")
        dataset_label = os.path.basename(data_dir.rstrip('/'))
        dataset = SNRDataset(data_dir, normalize=True, seed=seed, show_summary=True, full_dataset=full_dataset)
    elif isinstance(data_dir, (list, tuple)):
        # Multiple directories - combine them
        print(f"\nLoading data from {len(data_dir)} directories:")
        datasets = []
        for i, dir_path in enumerate(data_dir, 1):
            print(f"  [{i}/{len(data_dir)}] {dir_path}")
            ds = SNRDataset(dir_path, normalize=True, seed=seed, show_summary=False, full_dataset=full_dataset)
            datasets.append(ds)
            print(f"    -> Loaded {len(ds)} samples")
        dataset = ConcatDataset(datasets)
        dataset_label = f"Combined_{len(data_dir)}_datasets"
        print(f"\nTotal combined samples: {len(dataset)}")
    else:
        raise ValueError(f"data_dir must be str or list/tuple, got {type(data_dir)}")
    
    # Configure DataLoader for optimal GPU performance
    # Use multiple workers for parallel data loading (reduces GPU idle time)
    # pin_memory=True enables faster CPU->GPU transfer (CUDA only, not supported on MPS)
    # persistent_workers=True keeps workers alive between epochs (faster)
    num_workers = 4 if len(dataset) > 1000 else 2  # More workers for larger datasets
    pin_memory = True if device.type == 'cuda' else False  # Only CUDA supports pin_memory
    prefetch_factor = 2 if num_workers > 0 else None
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle for better training
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"DataLoader config: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
    sys.stdout.flush()
    
    # Load visualization samples
    if tsne_samples is None:
        tsne_samples = n_samples
    viz_samples = min(tsne_samples, len(dataset))
    print(f"Loading {viz_samples} samples for visualization...")
    data_list = []
    for i in range(viz_samples):
        sample, _ = dataset[i]
        data_list.append(sample.unsqueeze(0))
    data_tensor = torch.cat(data_list, dim=0) if data_list else None
    
    # STEP 3: Initialize model FROM SCRATCH (no pretrained weights)
    nrow, ncol = data_tensor.shape[-2], data_tensor.shape[-1]
    print(f"\nInitializing NEW model from random weights...")
    model = ImprovedAutoencoder(nrow=nrow, ncol=ncol, latent_dim=latent_dim, 
                                base_channels=channels, extra_conv=extra_conv,
                                kernel_size=kernel_size)
    model = model.to(device)  # Move model to GPU
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: {4 if extra_conv else 3} conv layers, latent_dim={latent_dim}, kernel={kernel_size}x{kernel_size}")
    print(f"Model on device: {next(model.parameters()).device}")
    
    # STEP 4: Train from scratch
    print(f"\nTraining model from scratch for {epochs} epochs...")
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()  # Ensure output is written immediately
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Enable Automatic Mixed Precision (AMP) for faster training on CUDA
    use_amp = device.type == 'cuda'  # AMP works best on CUDA
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print(f"Automatic Mixed Precision (AMP) enabled for faster training")
        sys.stdout.flush()
    
    model.train()
    
    losses = []
    epoch_times = []
    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_data, _ in train_loader:
            # Move batch to GPU (non_blocking=True for async transfer)
            batch_data = batch_data.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if use_amp:
                # Mixed precision training (faster on CUDA)
                with torch.cuda.amp.autocast():
                    output, _ = model(batch_data)
                    output = match_shape_center(output, (nrow, ncol))
                    loss = criterion(output, batch_data)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                output, _ = model(batch_data)
                output = match_shape_center(output, (nrow, ncol))
                loss = criterion(output, batch_data)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        losses.append(avg_loss)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        with torch.no_grad():
            o_min = float(output.min().cpu())
            o_max = float(output.max().cpu())
            o_mean = float(output.mean().cpu())
        
        # Print progress every epoch for monitoring
        elapsed = time.time() - training_start_time
        eta_seconds = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
        eta_minutes = eta_seconds / 60
        print(f"  Epoch {epoch:3d}/{epochs}: Loss={avg_loss:.4f} | out[{o_min:.3f}, {o_max:.3f}] | {epoch_time:.1f}s | ETA: {eta_minutes:.1f}min")
        sys.stdout.flush()  # Force write to log file
        
        # Save checkpoint every 10 epochs for recovery
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, 'trained_model', f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'losses': losses,
            }, checkpoint_path)
            print(f"  >> Checkpoint saved: {checkpoint_path}")
            sys.stdout.flush()
    
    training_elapsed = time.time() - training_start_time
    print(f"\nTraining complete: {training_elapsed:.1f}s ({training_elapsed/60:.1f}min)")
    print(f"  Avg per epoch: {training_elapsed/epochs:.1f}s")
    print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()
    
    # Save model weights to trained_model subdirectory
    model_path = os.path.join(output_dir, 'trained_model', 'autoencoder_clean.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")
    sys.stdout.flush()
    
    # Save model weights as MATLAB-compatible .mat file in MATLAB subdirectory
    mat_path = os.path.join(output_dir, 'MATLAB', 'autoencoder_clean.mat')
    state_dict = model.state_dict()
    mat_dict = {key: value.cpu().numpy() for key, value in state_dict.items()}
    savemat(mat_path, mat_dict)
    print(f"Saved model to MATLAB format: {mat_path}")
    sys.stdout.flush()
    
    # Clear training memory
    del train_loader
    gc.collect()
    
    # STEP 5: Extract latent embeddings
    model.eval()
    tsne_sample_count = min(TSNE_MAX_SAMPLES, len(dataset)) if TSNE_MAX_SAMPLES else len(dataset)
    print(f"\nExtracting {tsne_sample_count} latent embeddings for t-SNE analysis...")
    all_latent = []
    with torch.no_grad():
        for i in range(tsne_sample_count):
            sample, _ = dataset[i]
            sample = sample.unsqueeze(0).to(device)  # Move to GPU
            _, latent = model(sample)
            all_latent.append(latent.cpu())
    improved_latent_full = torch.cat(all_latent, dim=0)
    
    # Compute reconstructions on viz subset
    with torch.no_grad():
        data_tensor = data_tensor.to(device)  # Move to GPU
        improved_recon, _ = model(data_tensor)
        improved_recon = match_shape_center(improved_recon, (nrow, ncol))
    
    # STEP 6: Generate visualizations
    data_np = data_tensor.squeeze(1).cpu().numpy()
    print(f"\nGenerating visualizations...")
    
    # Plot 1: Reconstruction comparison
    vmin_data = data_np.min()
    vmax_data = data_np.max()
    cols = min(10, data_np.shape[0])
    n_rows = 3 if show_error else 2
    fig, axes = plt.subplots(n_rows, cols, figsize=(15, 6 if n_rows == 2 else 9))
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for i in range(cols):
        axes[0, i].imshow(data_np[i], cmap='viridis', origin='lower', aspect='auto', vmin=vmin_data, vmax=vmax_data)
        axes[0, i].set_title(f'Input {i+1}')
        axes[0, i].axis('off')
        imp_recon = improved_recon[i, 0].cpu().numpy()
        axes[1, i].imshow(imp_recon, cmap='viridis', origin='lower', aspect='auto', vmin=vmin_data, vmax=vmax_data)
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
        if show_error:
            diff = np.abs(data_np[i] - imp_recon)
            axes[2, i].imshow(diff, cmap='hot', origin='lower', aspect='auto')
            axes[2, i].set_title('Error')
            axes[2, i].axis('off')
    
    plt.suptitle(f'Autoencoder Reconstructions (epochs={epochs}, latent_dim={latent_dim})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_results', 'reconstructions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Training loss
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (epochs={epochs})')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_results', 'training_loss.png'), dpi=150)
    plt.close()
    
    # Plot 3: t-SNE with clustering (auto-find optimal k)
    # Extract latent embeddings first (save even if t-SNE fails)
    imp_z = improved_latent_full.detach().cpu().numpy()
    # Handle both single directory (string) and multiple directories (list)
    if isinstance(data_dir, list):
        dataset_label = "CombinedDatasets"
    else:
        dataset_label = os.path.basename(data_dir.rstrip('/'))
    
    if TSNE is not None and improved_latent_full.shape[0] > 2:
        try:
            print(f"Computing t-SNE on {improved_latent_full.shape[0]} samples...")
            perplexity = min(30.0, (imp_z.shape[0] - 1) / 3.0) if tsne_perplexity is None else tsne_perplexity
            perplexity = max(2.0, min(perplexity, imp_z.shape[0] - 1))
            
            emb = TSNE(n_components=2, random_state=int(seed) if seed else 0, 
                      perplexity=perplexity, learning_rate='auto').fit_transform(imp_z)
            
            # Auto-find optimal k if k_clusters is None or 0
            optimal_k = k_clusters
            if KMeans is not None and silhouette_score is not None and (k_clusters is None or k_clusters == 0):
                print("Finding optimal number of clusters...")
                max_k = min(10, imp_z.shape[0] // 2)  # Test up to 10 clusters
                silhouette_scores = []
                k_range = range(2, max_k + 1)
                
                for k in k_range:
                    kmeans_temp = KMeans(n_clusters=k, n_init='auto', random_state=int(seed) if seed else 0)
                    labels_temp = kmeans_temp.fit_predict(imp_z)
                    score = silhouette_score(imp_z, labels_temp)
                    silhouette_scores.append(score)
                    print(f"  k={k}: silhouette={score:.3f}")
                
                # Choose k with highest silhouette score
                optimal_k = k_range[np.argmax(silhouette_scores)]
                print(f"Optimal k={optimal_k} (silhouette={max(silhouette_scores):.3f})")
                
                # Save elbow plot
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.plot(list(k_range), silhouette_scores, 'bo-')
                plt.xlabel('Number of clusters (k)')
                plt.ylabel('Silhouette Score')
                plt.title('Optimal k Selection')
                plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'image_results', 'optimal_k_analysis.png'), dpi=150)
                plt.close()
            elif k_clusters is None or k_clusters == 0:
                optimal_k = 2  # Fallback if silhouette unavailable
            
            # Perform final clustering with optimal k
            if KMeans is not None:
                kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=int(seed) if seed else 0)
                clusters = kmeans.fit_predict(imp_z)
            else:
                clusters = (emb[:, 0] > np.median(emb[:, 0])).astype(int)
            
            # Generate color map for all clusters
            cmap = plt.cm.get_cmap('tab10', optimal_k)
            plt.figure(figsize=(7, 6))
            
            # Plot each cluster separately for legend
            for cluster_id in range(optimal_k):
                mask = clusters == cluster_id
                color = cmap(cluster_id)
                plt.scatter(emb[mask, 0], emb[mask, 1], 
                           c=[color], alpha=0.85, s=28, label=f'Cluster {cluster_id}')
            
            plt.title(f't-SNE Latent Space (k={optimal_k}, perplexity={perplexity:.1f})')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=(2 if optimal_k > 5 else 1))
            
            # Add dataset label in bottom right (discrete/subtle)
            plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', 
                       ha='right', va='bottom', fontsize=7, style='italic', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'image_results', 'tsne_latent.png'), dpi=160)
            plt.close()
            
            # Save latent embeddings and t-SNE results for re-plotting without retraining
            latent_data = {
                'latent_embeddings': imp_z,
                'tsne_embeddings': emb,
                'clusters': clusters,
                'optimal_k': optimal_k,
                'perplexity': perplexity,
                'dataset_label': dataset_label
            }
            savemat(os.path.join(output_dir, 'MATLAB', 'latent_embeddings.mat'), latent_data)
            print(f"Saved latent embeddings to MATLAB/latent_embeddings.mat (re-plot without retraining!)")
            
        except Exception as e:
            print(f"Warning: t-SNE visualization skipped: {e}")
            # Still save latent embeddings even if t-SNE failed
            emb = np.zeros((imp_z.shape[0], 2))  # Dummy t-SNE embeddings
            clusters = np.zeros(imp_z.shape[0], dtype=int)
            optimal_k = 2
            perplexity = 30.0
    else:
        print("Warning: TSNE not available or insufficient samples")
        emb = np.zeros((imp_z.shape[0], 2))
        clusters = np.zeros(imp_z.shape[0], dtype=int)
        optimal_k = 2
        perplexity = 30.0
    
    # Plot 4: 3D Interactive UMAP visualization (if enabled and available)
    if enable_umap and UMAP is not None and improved_latent_full.shape[0] > 2:
        try:
            print(f"\nGenerating 3D UMAP visualization (interactive HTML)...")
            umap_3d = umap.UMAP(n_components=3, random_state=seed if seed else 42, n_neighbors=15, min_dist=0.1)
            umap_emb_3d = umap_3d.fit_transform(imp_z)
            
            # Save 3D UMAP data to .mat file
            umap_3d_data = {
                'umap_3d_embeddings': umap_emb_3d,
                'clusters': clusters,
                'optimal_k': optimal_k,
                'dataset_label': dataset_label,
                'original_filenames': filenames if 'filenames' in locals() else np.array([], dtype=object)
            }
            umap_3d_mat_path = os.path.join(output_dir, 'UMAP', 'umap_3d_embeddings.mat')
            savemat(umap_3d_mat_path, umap_3d_data)
            print(f"Saved 3D UMAP data to: {umap_3d_mat_path}")
            
            # Create interactive 3D plot with Plotly (if available)
            if PLOTLY_AVAILABLE:
                print("Creating interactive 3D UMAP plot with Plotly...")
                
                # Create hover text with filenames if available
                hover_text = []
                if 'filenames' in locals() and len(filenames) == len(umap_emb_3d):
                    hover_text = [f"File: {fn}<br>Cluster: {c}" for fn, c in zip(filenames, clusters)]
                else:
                    hover_text = [f"Sample {i}<br>Cluster: {c}" for i, c in enumerate(clusters)]
                
                fig = go.Figure(data=[go.Scatter3d(
                    x=umap_emb_3d[:, 0],
                    y=umap_emb_3d[:, 1],
                    z=umap_emb_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=clusters,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Cluster"),
                        line=dict(width=0.5, color='white')
                    ),
                    text=hover_text,
                    hoverinfo='text'
                )])
                
                fig.update_layout(
                    title=f'3D UMAP Visualization - {dataset_label}<br>'
                          f'Latent Dim: {latent_dim}, Clusters: {optimal_k}, Samples: {len(umap_emb_3d)}',
                    scene=dict(
                        xaxis_title='UMAP 1',
                        yaxis_title='UMAP 2',
                        zaxis_title='UMAP 3',
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.5)
                        )
                    ),
                    width=1200,
                    height=900,
                    hovermode='closest'
                )
                
                # Save interactive HTML
                html_path = os.path.join(output_dir, 'UMAP', 'umap_3d_interactive.html')
                fig.write_html(html_path)
                print(f"Saved interactive 3D UMAP to: {html_path}")
                print(f"  -> Open this file in a web browser for interactive exploration")
            else:
                print("Warning: Plotly not available. Install with: pip install plotly")
                print("         3D UMAP data saved but interactive HTML not generated.")
            
        except Exception as e:
            print(f"Warning: 3D UMAP generation failed: {e}")
        
        # Still generate the 2D UMAP for comparison
        try:
            print(f"Computing UMAP on {improved_latent_full.shape[0]} samples...")
            umap_reducer = UMAP(n_components=2, random_state=int(seed) if seed else 0, n_neighbors=15, min_dist=0.1)
            umap_emb = umap_reducer.fit_transform(imp_z)
            
            # Use same clusters from t-SNE/k-means if available, otherwise compute new ones
            if KMeans is not None and optimal_k > 0:
                umap_clusters = clusters  # Reuse clusters from t-SNE section
            else:
                umap_clusters = (umap_emb[:, 0] > np.median(umap_emb[:, 0])).astype(int)
                optimal_k = 2
            
            # Generate UMAP plot
            cmap = plt.cm.get_cmap('tab10', optimal_k)
            plt.figure(figsize=(7, 6))
            
            for cluster_id in range(optimal_k):
                mask = umap_clusters == cluster_id
                color = cmap(cluster_id)
                plt.scatter(umap_emb[mask, 0], umap_emb[mask, 1], 
                           c=[color], alpha=0.85, s=28, label=f'Cluster {cluster_id}')
            
            plt.title(f'UMAP Latent Space (k={optimal_k})')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=(2 if optimal_k > 5 else 1))
            plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}', 
                       ha='right', va='bottom', fontsize=7, style='italic', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'UMAP', 'umap_latent.png'), dpi=160)
            plt.close()
            
            # Save UMAP embeddings
            umap_data = {
                'latent_embeddings': imp_z,
                'umap_embeddings': umap_emb,
                'clusters': umap_clusters,
                'optimal_k': optimal_k,
                'dataset_label': dataset_label,
                'original_filenames': filenames if 'filenames' in locals() else np.array([]),
                'reconstruction_filenames': reconstruction_filenames if 'reconstruction_filenames' in locals() else np.array([])
            }
            savemat(os.path.join(output_dir, 'UMAP', 'umap_embeddings.mat'), umap_data)
            print(f"Saved UMAP embeddings to UMAP/umap_embeddings.mat")
            
        except Exception as e:
            print(f"Warning: UMAP visualization skipped: {e}")
    elif not enable_umap:
        print("UMAP visualization disabled (enable_umap=False)")
    else:
        print("Warning: UMAP not available or insufficient samples")
    
    # ALWAYS save latent embeddings (even if t-SNE failed)
    print(f"Saving latent embeddings ({imp_z.shape[0]} samples, {imp_z.shape[1]}-dim)...")
    
    # Extract filenames from dataset (only basenames, matching embedding order)
    if isinstance(dataset, ConcatDataset):
        # For concatenated datasets, collect file_paths from all sub-datasets
        all_file_paths = []
        for ds in dataset.datasets:
            all_file_paths.extend(ds.file_paths)
        filenames = np.array([os.path.basename(all_file_paths[i]) for i in range(min(tsne_sample_count, len(all_file_paths)))], dtype=object)
    else:
        # For single dataset
        filenames = np.array([os.path.basename(dataset.file_paths[i]) for i in range(tsne_sample_count)], dtype=object)
    
    # Generate reconstruction filenames (original name with _reconstr suffix)
    reconstruction_filenames = np.array([f"{os.path.splitext(fn)[0]}_reconstr.mat" for fn in filenames], dtype=object)
    
    latent_data = {
        'latent_embeddings': imp_z,
        'tsne_embeddings': emb,
        'clusters': clusters,
        'optimal_k': optimal_k,
        'perplexity': perplexity,
        'dataset_label': dataset_label,
        'original_filenames': filenames,
        'reconstruction_filenames': reconstruction_filenames
    }
    embeddings_path = os.path.join(output_dir, 'MATLAB', 'latent_embeddings.mat')
    savemat(embeddings_path, latent_data)
    print(f"Saved latent embeddings to: {embeddings_path}")
    print(f"  -> Includes 'original_filenames' and 'reconstruction_filenames' fields")
    print(f"  -> Mapping {len(filenames)} embeddings to source files")
    print("  -> Use replot_tsne_from_saved.py to re-plot with different k values!")
    
    # STEP 7: Save JPEG reconstruction panels
    try:
        panel_samples, panel_filenames = select_samples_for_outputs(dataset, output_samples, seed)
        panels_written = save_reconstruction_panels(model, panel_samples, 
                                                    os.path.join(output_dir, 'image_results'), 
                                                    (nrow, ncol),
                                                    dataset_label=dataset_label, filenames=panel_filenames, 
                                                    show_error=show_error, epochs=epochs, 
                                                    latent_dim=latent_dim, channels=channels, device=device)
        print(f"Saved {panels_written} JPEG panel(s) ({panel_samples.shape[0]} samples)")
    except Exception as e:
        print(f"Warning: JPEG panels skipped: {e}")
    
    # STEP 8: Save reconstruction data with filenames
    sample_data = {
        'originals': data_np,
        'reconstructions': improved_recon.squeeze().cpu().numpy(),
        'filenames': panel_filenames
    }
    savemat(os.path.join(output_dir, 'MATLAB', 'reconstruction_data.mat'), sample_data)
    
    script_elapsed = time.time() - script_start_time
    timing_log = [
        f"FAST CLEAN AUTOENCODER - Training from Scratch",
        f"=" * 70,
        f"Start: {datetime.fromtimestamp(script_start_time).strftime('%Y-%m-%d %H:%M:%S')}",
        f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total runtime: {script_elapsed:.1f}s ({script_elapsed/60:.1f}min)",
        f"",
        f"Configuration:",
        f"  Dataset: {dataset_label}",
        f"  Files: {len(dataset)}",
        f"  Epochs: {epochs}",
        f"  Learning rate: {lr}",
        f"  Batch size: {batch_size}",
        f"  Latent dim: {latent_dim}",
        f"  Channels: {channels}",
        f"  Seed: {seed}",
        f"",
        f"Performance:",
        f"  Training: {training_elapsed:.1f}s ({training_elapsed/60:.1f}min)",
        f"  Per epoch: {training_elapsed/epochs:.1f}s",
        f"  Other ops: {script_elapsed - training_elapsed:.1f}s",
        f"",
        f"Results:",
        f"  Final loss: {losses[-1]:.6f}",
        f"  Model saved: autoencoder_clean.pth",
        f"  FRESH START: Model trained from random initialization",
    ]
    
    with open(os.path.join(output_dir, 'timing_log.txt'), 'w') as f:
        f.write('\n'.join(timing_log))
    
    print(f"\n{'='*70}")
    print(f"COMPLETE! Total: {script_elapsed:.1f}s ({script_elapsed/60:.1f}min)")
    print(f"  Training: {training_elapsed:.1f}s | Other: {script_elapsed - training_elapsed:.1f}s")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")


# ========================== 
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid_Filter_Autoencoder")
    parser.add_argument("--data-dir", 
                       nargs='+',
                       default=[
                           "/Users/oboulais/Desktop/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir",
                           "/Users/oboulais/Desktop/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir"
                       ],
                       help=f"One or more directories containing .mat files (default: both datasets, or {FULL_DATASET_PATH} when --full-dataset is used)")
    parser.add_argument("--n-samples", type=int, default=15, help="Visualization samples")
    parser.add_argument("--tsne-samples", type=int, default=None, help="t-SNE samples (default: n-samples)")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM_DEFAULT, help="Latent dimension")
    parser.add_argument("--channels", type=int, default=CHANNELS_DEFAULT, help="Base channels")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help=f"Training epochs (default: {EPOCHS_DEFAULT})")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT, help=f"Learning rate (default: {LR_DEFAULT})")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help=f"Random seed (default: {SEED_DEFAULT})")
    parser.add_argument("--k-clusters", type=int, default=0, help="KMeans clusters for t-SNE (0=auto-detect optimal k)")
    parser.add_argument("--tsne-perplexity", type=float, default=None, help="t-SNE perplexity")
    parser.add_argument("--extra-conv", action='store_true', default=EXTRA_CONV_DEFAULT, help="Use 4 conv layers")
    parser.add_argument("--no-extra-conv", dest='extra_conv', action='store_false', help="Use 3 conv layers")
    parser.add_argument("--kernel-size", type=int, default=5, choices=[3, 5], 
                       help="Conv kernel size: 3x3 or 5x5 (default, better for N/U-shaped calls)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-samples", type=int, default=NUMBER_OUTPUT_IMAGE_SAMPLES, help="JPEG panel samples")
    parser.add_argument("--show-error", action='store_true', default=SHOW_ERROR_PLOTS, help="Show error row")
    parser.add_argument("--no-error", dest='show_error', action='store_false', help="Hide error row")
    parser.add_argument("--enable-umap", action='store_true', default=ENABLE_UMAP, help="Enable UMAP visualization")
    parser.add_argument("--no-umap", dest='enable_umap', action='store_false', help="Disable UMAP visualization")
    parser.add_argument("--full-dataset", action='store_true', default=FULL_DATASET, 
                       help="Enable FULL_DATASET mode: recursively discover ALL .mat files in all subdirectories")
    parser.add_argument("--no-full-dataset", dest='full_dataset', action='store_false', help="Disable FULL_DATASET mode")
    parser.add_argument("--version-tag", type=str, default=DEFAULT_VERSION_TAG, help="Version tag")
    args = parser.parse_args()
    
    # Override data_dir when FULL_DATASET mode is enabled
    if args.full_dataset:
        # Use the comprehensive spectrogram database for full dataset mode
        args.data_dir = [FULL_DATASET_PATH]
        print(f"[FULL_DATASET MODE] Using comprehensive database: {args.data_dir[0]}")
    
    train_autoencoder_from_scratch(
        data_dir=args.data_dir,
        n_samples=args.n_samples,
        latent_dim=args.latent_dim,
        channels=args.channels,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        tsne_samples=args.tsne_samples,
        extra_conv=args.extra_conv,
        batch_size=args.batch_size,
        output_samples=args.output_samples,
        version_tag=args.version_tag,
        show_error=args.show_error,
        k_clusters=args.k_clusters,
        tsne_perplexity=args.tsne_perplexity,
        kernel_size=args.kernel_size,
        enable_umap=args.enable_umap,
        full_dataset=args.full_dataset
    )
