#!/usr/bin/env python3
"""
VARIATIONAL AUTOENCODER (VAE) - Training from Scratch

KEY DIFFERENCES FROM STANDARD AUTOENCODER:
- Encoder outputs: μ (mean) and log(σ²) (log variance)
- Reparameterization trick: z = μ + σ × ε (where ε ~ N(0,1))
- Loss function: Reconstruction loss + KL divergence
- Enables generative sampling from learned distribution

GUARANTEED FRESH START:
- No model loading or transfer learning
- No checkpoint resuming
- Random initialization only (controlled by seed)
- Each run completely independent

USAGE:
source .venv_py31018/bin/activate
python3 Autoencoder_v02_MultiGram_VAE_20260330.py --gram-type BOTH
"""
import os
import sys
import argparse
import math
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datetime import datetime
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

import warnings
warnings.filterwarnings('ignore')
from typing import Optional, Tuple, List

# ============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ============================================================================

# Architecture parameters
CHANNELS_DEFAULT = 32
LATENT_DIM_DEFAULT = 32
EXTRA_CONV_DEFAULT = False

# Training parameters 
EPOCHS_DEFAULT = 100
LR_DEFAULT = 1e-3
SEED_DEFAULT = 42
KL_WEIGHT_DEFAULT = 0.001  # VAE-specific: weight for KL divergence loss

# Output parameters 
NUMBER_OUTPUT_IMAGE_SAMPLES = 30
PANEL_GROUP_SIZE = 3
SHOW_ERROR_PLOTS = False
ENABLE_UMAP = True
DEFAULT_VERSION_TAG = "v13VAE_100E_32LD_32C_AutoManual_Combined_100K"
TSNE_MAX_SAMPLES = None

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
    results_dir = "/Users/oboulais/Public/Bowhead_DL_Project/LD32"
    os.makedirs(results_dir, exist_ok=True)
    
    output_dir = os.path.join(results_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create organized subdirectories
    os.makedirs(os.path.join(output_dir, 'MATLAB'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image_results'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image_results', 'SNR'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image_results', 'NTV'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image_results', 'spectrogram'), exist_ok=True)
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
        return im.astype(np.float32)
    return (im - im_min) / rng


# ============================================================================
# DATASET CLASS
# ============================================================================

class SNRDataset(Dataset):
    """Memory-efficient Dataset loading .mat files on-demand."""
    
    def __init__(self, directory: str, normalize: bool = True, 
                 seed: Optional[int] = None, show_summary: bool = False,
                 max_samples: Optional[int] = None, gram_type: str = 'SNR_gram'):
        self.directory = directory
        self.normalize = normalize
        self.gram_type = gram_type
        self.load_both = (gram_type == 'BOTH')
        
        # Get all .mat files
        self.file_paths = sorted([
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith('.mat')
        ])
        
        if not self.file_paths:
            raise ValueError(f"No .mat files found in {directory}")
        
        # Limit number of samples if requested
        if max_samples is not None and max_samples < len(self.file_paths):
            self.file_paths = self.file_paths[:max_samples]
        
        # Shuffle if seed provided
        if seed is not None:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(self.file_paths))
            self.file_paths = [self.file_paths[i] for i in indices]
        
        # Get dimensions from first sample
        test_data = loadmat(self.file_paths[0])
        test_key = 'SNR_gram' if 'SNR_gram' in test_data else list(test_data.keys())[0]
        self.target_shape = test_data[test_key].shape
        
        if show_summary:
            print(f"Dataset: {directory}")
            print(f"  Files: {len(self.file_paths)}")
            print(f"  Shape: {self.target_shape}")
            print(f"  Gram type: {gram_type}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        fp = self.file_paths[idx]
        
        if self.load_both:
            # Load BOTH SNR_gram and NTV_gram as 2-channel input
            try:
                m = loadmat(fp)
                snr = m['SNR_gram']
                ntv = m['NTV_gram']
                if self.normalize:
                    snr = _minmax_norm(snr)
                    ntv = _minmax_norm(ntv)
                else:
                    snr = snr.astype(np.float32)
                    ntv = ntv.astype(np.float32)
                tensor = torch.from_numpy(np.stack([snr, ntv], axis=0))
                return tensor, 0
            except Exception as e:
                print(f"Warning: Failed to load {fp} (BOTH): {e}")
                h, w = self.target_shape
                return torch.zeros((2, h, w), dtype=torch.float32), 0
        else:
            # Load single gram type as 1-channel input
            try:
                m = loadmat(fp)
                im = m[self.gram_type]
                if self.normalize:
                    im = _minmax_norm(im)
                else:
                    im = im.astype(np.float32)
                tensor = torch.from_numpy(im).unsqueeze(0)
                return tensor, 0
            except Exception as e:
                print(f"Warning: Failed to load {fp} ({self.gram_type}): {e}")
                h, w = self.target_shape
                return torch.zeros((1, h, w), dtype=torch.float32), 0


# ============================================================================
# VAE MODEL ARCHITECTURE
# ============================================================================

class ImprovedVariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder with batch normalization.
    
    Key Features:
    - Encoder outputs μ and log(σ²) for latent distribution
    - Reparameterization trick for sampling: z = μ + σ × ε
    - Generates diverse outputs through probabilistic latent space
    - ALWAYS INITIALIZED FROM SCRATCH - NO PRETRAINED WEIGHTS
    """
    
    def __init__(self, nrow=121, ncol=104, latent_dim=LATENT_DIM_DEFAULT, 
                 base_channels=CHANNELS_DEFAULT, extra_conv=EXTRA_CONV_DEFAULT, in_channels=1):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        self.extra_conv = extra_conv
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
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
        
        # Encoder: same as standard autoencoder
        if extra_conv:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, c1, 3, padding=1),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
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
                nn.Conv2d(in_channels, c1, 3, padding=1),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
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
        
        # VAE-specific: Split into mu and log_var branches
        self.fc_mu = nn.Sequential(
            nn.Linear(flat_size, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.fc_log_var = nn.Sequential(
            nn.Linear(flat_size, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # Decoder: from latent to reconstruction
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
                nn.ConvTranspose2d(c1, in_channels, 2, stride=2, output_padding=(pad_h, pad_w)),
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
                nn.ConvTranspose2d(c1, in_channels, 2, stride=2, output_padding=(pad_h, pad_w)),
            )
        
        self.flat_size = flat_size
        self.nrow_reduced = nrow_reduced
        self.ncol_reduced = ncol_reduced
        self.base_channels = base_channels
        self.c_out = c4 if extra_conv else c3

    def encode(self, x):
        """Encode input to latent distribution parameters."""
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        mu = self.fc_mu(x_flat)
        log_var = self.fc_log_var(x_flat)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = μ + σ × ε
        where ε ~ N(0, 1)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to reconstruction."""
        x_recon = self.from_latent(z)
        x_recon = x_recon.view(x_recon.size(0), self.c_out, self.nrow_reduced, self.ncol_reduced)
        output = self.decoder(x_recon)
        return output

    def forward(self, x):
        """
        Forward pass through VAE.
        Returns: reconstruction, input, mu, log_var
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


def vae_loss_function(recon_x, x, mu, log_var, kl_weight=0.001):
    """
    VAE loss = Reconstruction loss + KL divergence
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution
        kl_weight: Weight for KL divergence term
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = kl_loss / x.size(0)  # Average over batch
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss


def match_shape_center(recon: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Center-crop or pad reconstruction to match target dimensions."""
    _, _, rH, rW = recon.shape
    tH, tW = target_hw
    if rH > tH:
        startH = (rH - tH) // 2
        recon = recon[:, :, startH:startH+tH, :]
    if rW > tW:
        startW = (rW - tW) // 2
        recon = recon[:, :, :, startW:startW+tW]
    padH = tH - rH
    padW = tW - rW
    if padH > 0 or padW > 0:
        pad_top = padH // 2
        pad_bottom = padH - pad_top
        pad_left = padW // 2
        pad_right = padW - pad_left
        recon = nn.functional.pad(recon, (pad_left, pad_right, pad_top, pad_bottom))
    return recon


def select_samples_for_outputs(dataset: Dataset, n_samples: int, seed: Optional[int]) -> Tuple[torch.Tensor, List[str]]:
    """Select random samples for JPEG panel generation."""
    if dataset is None or len(dataset) == 0:
        return torch.zeros((0, 1, 121, 104)), []
    rng = np.random.default_rng(seed)
    total = len(dataset)
    k = min(n_samples, total)
    indices = rng.choice(total, size=k, replace=False) if total > k else np.arange(total)
    samples = []
    filenames = []
    
    # Collect all file paths
    all_file_paths = []
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            all_file_paths.extend(ds.file_paths)
    elif hasattr(dataset, 'file_paths'):
        all_file_paths = dataset.file_paths
    
    for idx in indices:
        sample, _ = dataset[idx]
        samples.append(sample.unsqueeze(0))
        if idx < len(all_file_paths):
            filenames.append(os.path.basename(all_file_paths[idx]))
        else:
            filenames.append(f"sample_{idx}.mat")
    return torch.cat(samples, dim=0), filenames


def save_reconstruction_panels(model: nn.Module, samples: torch.Tensor, output_dir: str,
                               target_hw: Tuple[int, int], base_name: str = "recon_panel",
                               dataset_label: str = "", filenames: Optional[List[str]] = None, 
                               show_error: bool = SHOW_ERROR_PLOTS, epochs: int = 100,
                               latent_dim: int = 32, channels: int = 64, device: torch.device = None) -> int:
    """Save JPEG panels showing reconstructions."""
    if samples is None or samples.shape[0] == 0:
        return 0
    if device is None:
        device = next(model.parameters()).device
    os.makedirs(output_dir, exist_ok=True)
    num_samples = samples.shape[0]
    group_count = math.ceil(num_samples / PANEL_GROUP_SIZE)
    panels_written = 0
    n_rows = 3 if show_error else 2
    
    nrow, ncol = target_hw
    freq_max_hz = 500.0
    time_duration_sec = 3.0
    
    model.eval()
    for group_idx in range(group_count):
        start_idx = group_idx * PANEL_GROUP_SIZE
        end_idx = min(start_idx + PANEL_GROUP_SIZE, num_samples)
        batch = samples[start_idx:end_idx].to(device)
        
        # Handle 2-channel visualization
        batch_np = batch.cpu().numpy()
        if batch_np.shape[1] == 2:
            batch_display = np.mean(batch_np, axis=1)
        else:
            batch_display = batch_np.squeeze(1)
        
        with torch.no_grad():
            recon, _, _ = model(batch)
            recon = match_shape_center(recon, target_hw)
        
        recon_np_raw = recon.cpu().numpy()
        if recon_np_raw.shape[1] == 2:
            recon_display = np.mean(recon_np_raw, axis=1)
        else:
            recon_display = recon_np_raw.squeeze(1)
        
        n_in_group = end_idx - start_idx
        fig, axes = plt.subplots(n_rows, n_in_group, figsize=(n_in_group * 3, n_rows * 2.5))
        if n_in_group == 1:
            axes = axes.reshape(-1, 1)
        
        for col_idx in range(n_in_group):
            orig_img = batch_display[col_idx]
            recon_img = recon_display[col_idx]
            
            # Original
            im0 = axes[0, col_idx].imshow(orig_img, aspect='auto', cmap='inferno', origin='lower')
            axes[0, col_idx].set_title(f'Input {start_idx+col_idx+1}')
            axes[0, col_idx].set_xlabel('Time (s)')
            axes[0, col_idx].set_ylabel('Frequency (Hz)')
            
            # Reconstruction
            im1 = axes[1, col_idx].imshow(recon_img, aspect='auto', cmap='inferno', origin='lower')
            axes[1, col_idx].set_title(f'VAE Recon')
            axes[1, col_idx].set_xlabel('Time (s)')
            axes[1, col_idx].set_ylabel('Frequency (Hz)')
            
            # Error (if enabled)
            if show_error:
                error = np.abs(orig_img - recon_img)
                im2 = axes[2, col_idx].imshow(error, aspect='auto', cmap='hot', origin='lower')
                axes[2, col_idx].set_title(f'|Error|')
                plt.colorbar(im2, ax=axes[2, col_idx], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'VAE Reconstructions (Group {group_idx+1}/{group_count})', fontsize=14)
        plt.tight_layout()
        
        panel_filename = f"{base_name}_group{group_idx+1:03d}.jpg"
        panel_path = os.path.join(output_dir, panel_filename)
        plt.savefig(panel_path, dpi=120, format='jpg', bbox_inches='tight')
        plt.close()
        panels_written += 1
    
    return panels_written


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_vae_from_scratch(data_dir: str, n_samples: int = 15, latent_dim: int = LATENT_DIM_DEFAULT,
                          channels: int = CHANNELS_DEFAULT, seed: Optional[int] = SEED_DEFAULT,
                          epochs: int = EPOCHS_DEFAULT, lr: float = LR_DEFAULT,
                          tsne_samples: Optional[int] = None, extra_conv: bool = EXTRA_CONV_DEFAULT,
                          batch_size: int = 32, output_samples: int = NUMBER_OUTPUT_IMAGE_SAMPLES,
                          version_tag: str = DEFAULT_VERSION_TAG, show_error: bool = SHOW_ERROR_PLOTS,
                          k_clusters: int = 2, tsne_perplexity: Optional[float] = None,
                          max_samples_per_dataset: Optional[int] = None, gram_type: str = 'SNR_gram',
                          kl_weight: float = KL_WEIGHT_DEFAULT):
    """
    Train Variational Autoencoder from scratch.
    
    VAE-specific parameter:
        kl_weight: Weight for KL divergence term in loss function
    """
    script_start_time = time.time()
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
    sys.stdout.flush()
    
    # Initialize random seed
    if seed is not None:
        set_global_seed(int(seed))
    
    output_dir = create_output_directory(version_tag)
    print(f"="*70)
    print(f"VARIATIONAL AUTOENCODER (VAE) - Training from Scratch")
    print(f"="*70)
    print(f"Output: {output_dir}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: epochs={epochs}, lr={lr}, latent_dim={latent_dim}, channels={channels}")
    print(f"VAE-specific: kl_weight={kl_weight}")
    print(f"="*70)
    
    # Load datasets
    if isinstance(data_dir, str):
        print(f"Loading dataset from: {data_dir}")
        dataset = SNRDataset(data_dir, normalize=True, seed=seed, show_summary=True,
                            max_samples=max_samples_per_dataset, gram_type=gram_type)
    elif isinstance(data_dir, (list, tuple)):
        print(f"Loading {len(data_dir)} datasets...")
        datasets = []
        for i, dir_path in enumerate(data_dir):
            print(f"  [{i+1}/{len(data_dir)}] {dir_path}")
            ds = SNRDataset(dir_path, normalize=True, seed=seed, show_summary=True,
                           max_samples=max_samples_per_dataset, gram_type=gram_type)
            datasets.append(ds)
        dataset = ConcatDataset(datasets)
        print(f"Combined dataset: {len(dataset)} samples")
    else:
        raise ValueError(f"data_dir must be str or list/tuple, got {type(data_dir)}")
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Load visualization samples
    if tsne_samples is None:
        tsne_samples = n_samples
    viz_samples = min(tsne_samples, len(dataset))
    print(f"\nLoading {viz_samples} samples for visualization...")
    data_list = []
    for i in range(viz_samples):
        sample, _ = dataset[i]
        data_list.append(sample.unsqueeze(0))
    data_tensor = torch.cat(data_list, dim=0) if data_list else None
    
    # Initialize VAE model
    in_channels = data_tensor.shape[1]
    nrow, ncol = data_tensor.shape[-2], data_tensor.shape[-1]
    print(f"\nInitializing NEW VAE model from random weights...")
    print(f"Input shape: {data_tensor.shape} (channels={in_channels})")
    model = ImprovedVariationalAutoencoder(nrow=nrow, ncol=ncol, latent_dim=latent_dim, 
                                          base_channels=channels, extra_conv=extra_conv, in_channels=in_channels)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: {4 if extra_conv else 3} conv layers, latent_dim={latent_dim}, in_channels={in_channels}")
    print(f"VAE architecture: Encoder → (μ, log_var) → z = μ + σε → Decoder")
    
    # Train VAE
    print(f"\nTraining VAE from scratch for {epochs} epochs...")
    sys.stdout.flush()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    losses = []
    recon_losses = []
    kl_losses = []
    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        n_batches = 0
        
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(batch_data)
            recon_batch = match_shape_center(recon_batch, (nrow, ncol))
            
            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, batch_data, mu, log_var, kl_weight)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon_loss / n_batches
        avg_kl = epoch_kl_loss / n_batches
        losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        
        epoch_time = time.time() - epoch_start
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] Loss: {avg_loss:.6f} (Recon: {avg_recon:.6f}, KL: {avg_kl:.6f}) Time: {epoch_time:.1f}s")
            sys.stdout.flush()
    
    training_elapsed = time.time() - training_start_time
    print(f"\nTraining complete: {training_elapsed:.1f}s ({training_elapsed/60:.1f}min)")
    print(f"  Avg per epoch: {training_elapsed/epochs:.1f}s")
    sys.stdout.flush()
    
    # Save model
    model_path = os.path.join(output_dir, 'trained_model', 'vae_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved VAE model to: {model_path}")
    
    mat_path = os.path.join(output_dir, 'MATLAB', 'vae_model.mat')
    state_dict = model.state_dict()
    mat_dict = {key: value.cpu().numpy() for key, value in state_dict.items()}
    savemat(mat_path, mat_dict)
    print(f"Saved VAE model to MATLAB format: {mat_path}")
    
    # Extract latent embeddings
    model.eval()
    tsne_sample_count = min(TSNE_MAX_SAMPLES, len(dataset)) if TSNE_MAX_SAMPLES else len(dataset)
    print(f"\nExtracting {tsne_sample_count} latent embeddings...")
    all_latent = []
    with torch.no_grad():
        for i in range(tsne_sample_count):
            sample, _ = dataset[i]
            sample_batch = sample.unsqueeze(0).to(device)
            mu, _ = model.encode(sample_batch)  # Use mean of latent distribution
            all_latent.append(mu.cpu())
    improved_latent_full = torch.cat(all_latent, dim=0)
    
    # Compute reconstructions
    with torch.no_grad():
        data_tensor_gpu = data_tensor.to(device)
        improved_recon, _, _ = model(data_tensor_gpu)
        improved_recon = match_shape_center(improved_recon, (nrow, ncol)).cpu()
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    data_np_raw = data_tensor.cpu().numpy()
    if data_np_raw.shape[1] == 2:
        data_np = np.mean(data_np_raw, axis=1)
    else:
        data_np = data_np_raw.squeeze(1)
    
    # Plot 1: Reconstruction comparison
    cols = min(10, data_np.shape[0])
    n_rows = 3 if show_error else 2
    fig, axes = plt.subplots(n_rows, cols, figsize=(15, 6 if n_rows == 2 else 9))
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    improved_recon_np = improved_recon.cpu().numpy()
    if improved_recon_np.shape[1] == 2:
        improved_recon_display = np.mean(improved_recon_np, axis=1)
    else:
        improved_recon_display = improved_recon_np.squeeze(1)
    
    for i in range(cols):
        axes[0, i].imshow(data_np[i], aspect='auto', cmap='inferno')
        axes[0, i].set_title(f'Input {i+1}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(improved_recon_display[i], aspect='auto', cmap='inferno')
        axes[1, i].set_title(f'VAE Recon')
        axes[1, i].axis('off')
        
        if show_error:
            error = np.abs(data_np[i] - improved_recon_display[i])
            axes[2, i].imshow(error, aspect='auto', cmap='hot')
            axes[2, i].set_title(f'|Error|')
            axes[2, i].axis('off')
    
    plt.suptitle(f'VAE Reconstructions (epochs={epochs}, latent_dim={latent_dim})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_results', 'reconstructions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Training losses
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(recon_losses)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss (MSE)')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(kl_losses)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(f'VAE Training Losses (epochs={epochs}, kl_weight={kl_weight})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_results', 'training_losses.png'), dpi=150)
    plt.close()
    
    # t-SNE and UMAP (same as standard autoencoder)
    imp_z = improved_latent_full.detach().cpu().numpy()
    
    if isinstance(data_dir, list):
        dataset_label = "CombinedDatasets"
    else:
        dataset_label = os.path.basename(data_dir.rstrip('/'))
    
    # Extract filenames
    if isinstance(dataset, ConcatDataset):
        all_file_paths = []
        for ds in dataset.datasets:
            all_file_paths.extend(ds.file_paths)
        filenames = np.array([os.path.basename(all_file_paths[i]) for i in range(min(tsne_sample_count, len(all_file_paths)))], dtype=object)
    else:
        filenames = np.array([os.path.basename(dataset.file_paths[i]) for i in range(tsne_sample_count)], dtype=object)
    
    reconstruction_filenames = np.array([f"{os.path.splitext(fn)[0]}_reconstr.mat" for fn in filenames], dtype=object)
    
    # Placeholder for t-SNE/clustering
    emb = np.zeros((imp_z.shape[0], 2))
    clusters = np.zeros(imp_z.shape[0], dtype=int)
    optimal_k = 2
    perplexity = 30.0
    
    if TSNE is not None and improved_latent_full.shape[0] > 2:
        try:
            print(f"Computing t-SNE on {improved_latent_full.shape[0]} samples...")
            perplexity = min(30.0, (imp_z.shape[0] - 1) / 3.0) if tsne_perplexity is None else tsne_perplexity
            perplexity = max(2.0, min(perplexity, imp_z.shape[0] - 1))
            
            emb = TSNE(n_components=2, random_state=int(seed) if seed else 0, 
                      perplexity=perplexity, learning_rate='auto').fit_transform(imp_z)
            
            if KMeans is not None and silhouette_score is not None and (k_clusters is None or k_clusters == 0):
                print("Finding optimal number of clusters...")
                max_k = min(10, imp_z.shape[0] // 2)
                silhouette_scores = []
                k_range = range(2, max_k + 1)
                
                for k in k_range:
                    kmeans_temp = KMeans(n_clusters=k, n_init='auto', random_state=int(seed) if seed else 0)
                    labels_temp = kmeans_temp.fit_predict(imp_z)
                    score = silhouette_score(imp_z, labels_temp)
                    silhouette_scores.append(score)
                
                optimal_k = k_range[np.argmax(silhouette_scores)]
                print(f"Optimal k={optimal_k} (silhouette={max(silhouette_scores):.3f})")
            elif k_clusters is None or k_clusters == 0:
                optimal_k = 2
            else:
                optimal_k = k_clusters
            
            if KMeans is not None:
                kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=int(seed) if seed else 0)
                clusters = kmeans.fit_predict(imp_z)
            
            # Plot t-SNE
            cmap = plt.cm.get_cmap('tab10', optimal_k)
            plt.figure(figsize=(7, 6))
            
            for cluster_id in range(optimal_k):
                mask = clusters == cluster_id
                color = cmap(cluster_id)
                plt.scatter(emb[mask, 0], emb[mask, 1], 
                           c=[color], alpha=0.85, s=28, label=f'Cluster {cluster_id}')
            
            plt.title(f't-SNE VAE Latent Space (k={optimal_k})')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'image_results', 'tsne_latent.png'), dpi=160)
            plt.close()
        except Exception as e:
            print(f"Warning: t-SNE skipped: {e}")
    
    # Save latent embeddings
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
    
    # UMAP (2D, 3D, 5D)
    if ENABLE_UMAP and UMAP is not None and improved_latent_full.shape[0] > 2:
        try:
            print(f"Computing UMAP (2D, 3D, 5D) on {improved_latent_full.shape[0]} samples...")
            umap_clusters = clusters
            cmap = plt.cm.get_cmap('tab10', optimal_k)
            
            # 2D UMAP
            print("  Computing 2D UMAP...")
            umap_reducer_2d = UMAP(n_components=2, random_state=int(seed) if seed else 42, 
                                   n_neighbors=15, min_dist=0.1)
            umap_emb_2d = umap_reducer_2d.fit_transform(imp_z)
            
            plt.figure(figsize=(7, 6))
            for cluster_id in range(optimal_k):
                mask = umap_clusters == cluster_id
                color = cmap(cluster_id)
                plt.scatter(umap_emb_2d[mask, 0], umap_emb_2d[mask, 1], 
                           c=[color], alpha=0.85, s=28, label=f'Cluster {cluster_id}')
            
            plt.title(f'UMAP 2D VAE Latent Space (k={optimal_k})')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'UMAP', 'umap_2d_latent.png'), dpi=160)
            plt.close()
            print(f"    ✓ 2D UMAP complete")
            
            # 3D UMAP
            print("  Computing 3D UMAP...")
            umap_reducer_3d = UMAP(n_components=3, random_state=int(seed) if seed else 42,
                                   n_neighbors=15, min_dist=0.1)
            umap_emb_3d = umap_reducer_3d.fit_transform(imp_z)
            
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            for cluster_id in range(optimal_k):
                mask = umap_clusters == cluster_id
                color = cmap(cluster_id)
                ax.scatter(umap_emb_3d[mask, 0], umap_emb_3d[mask, 1], umap_emb_3d[mask, 2],
                          c=[color], alpha=0.85, s=28, label=f'Cluster {cluster_id}')
            
            ax.set_title(f'UMAP 3D VAE Latent Space (k={optimal_k})')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_zlabel('UMAP 3')
            ax.legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'UMAP', 'umap_3d_latent.png'), dpi=160)
            plt.close()
            print(f"    ✓ 3D UMAP complete")
            
            # 5D UMAP
            print("  Computing 5D UMAP...")
            umap_reducer_5d = UMAP(n_components=5, random_state=int(seed) if seed else 42,
                                   n_neighbors=15, min_dist=0.1)
            umap_emb_5d = umap_reducer_5d.fit_transform(imp_z)
            print(f"    ✓ 5D UMAP complete (no visualization, saved to .mat)")
            
            # Save all UMAP embeddings
            umap_data = {
                'latent_embeddings': imp_z,
                'umap_embeddings_2d': umap_emb_2d,
                'umap_embeddings_3d': umap_emb_3d,
                'umap_embeddings_5d': umap_emb_5d,
                'clusters': umap_clusters,
                'optimal_k': optimal_k,
                'dataset_label': dataset_label,
                'original_filenames': filenames,
                'reconstruction_filenames': reconstruction_filenames
            }
            savemat(os.path.join(output_dir, 'UMAP', 'umap_embeddings.mat'), umap_data)
            print(f"✓ Saved UMAP embeddings (2D, 3D, 5D) to UMAP/umap_embeddings.mat")
        except Exception as e:
            print(f"Warning: UMAP skipped: {e}")
    
    # Save timing log
    script_elapsed = time.time() - script_start_time
    timing_log = [
        f"VARIATIONAL AUTOENCODER (VAE) - Training from Scratch",
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
        f"  KL weight: {kl_weight}",
        f"",
        f"Performance:",
        f"  Training: {training_elapsed:.1f}s ({training_elapsed/60:.1f}min)",
        f"  Per epoch: {training_elapsed/epochs:.1f}s",
        f"",
        f"Results:",
        f"  Final total loss: {losses[-1]:.6f}",
        f"  Final recon loss: {recon_losses[-1]:.6f}",
        f"  Final KL loss: {kl_losses[-1]:.6f}",
        f"  Model saved: vae_model.pth",
    ]
    
    with open(os.path.join(output_dir, 'timing_log.txt'), 'w') as f:
        f.write('\n'.join(timing_log))
    
    print(f"\n{'='*70}")
    print(f"COMPLETE! Total: {script_elapsed:.1f}s ({script_elapsed/60:.1f}min)")
    print(f"  Training: {training_elapsed:.1f}s")
    print(f"  Final total loss: {losses[-1]:.6f}")
    print(f"  Final recon loss: {recon_losses[-1]:.6f}")
    print(f"  Final KL loss: {kl_losses[-1]:.6f}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variational Autoencoder (VAE) v02")
    parser.add_argument("--data-dir", 
                       nargs='+',
                       default=[
                          "/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214_centered.dir",
                          "/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214_centered.dir"
                       ],
                       help="One or more directories containing .mat files")
    parser.add_argument("--max-samples-per-dataset", type=int, default=50000,
                       help="Maximum samples from each dataset (default: 50000)")
    parser.add_argument("--n-samples", type=int, default=15, help="Visualization samples")
    parser.add_argument("--tsne-samples", type=int, default=None, help="t-SNE samples")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM_DEFAULT, help="Latent dimension")
    parser.add_argument("--channels", type=int, default=CHANNELS_DEFAULT, help="Base channels")
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help=f"Training epochs (default: {EPOCHS_DEFAULT})")
    parser.add_argument("--lr", type=float, default=LR_DEFAULT, help=f"Learning rate (default: {LR_DEFAULT})")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help=f"Random seed (default: {SEED_DEFAULT})")
    parser.add_argument("--kl-weight", type=float, default=KL_WEIGHT_DEFAULT, 
                       help=f"KL divergence weight (default: {KL_WEIGHT_DEFAULT})")
    parser.add_argument("--k-clusters", type=int, default=0, help="KMeans clusters (0=auto-detect)")
    parser.add_argument("--tsne-perplexity", type=float, default=None, help="t-SNE perplexity")
    parser.add_argument("--extra-conv", action='store_true', default=EXTRA_CONV_DEFAULT, help="Use 4 conv layers")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-samples", type=int, default=NUMBER_OUTPUT_IMAGE_SAMPLES, help="JPEG panel samples")
    parser.add_argument("--show-error", action='store_true', default=SHOW_ERROR_PLOTS, help="Show error row")
    parser.add_argument("--version-tag", type=str, default=DEFAULT_VERSION_TAG, help="Version tag")
    parser.add_argument("--gram-type", type=str, default='SNR_gram', 
                       choices=['SNR_gram', 'NTV_gram', 'Polar_gram', 'KEtoPE_gram', 'BOTH'],
                       help="Spectrogram type (default: SNR_gram, use BOTH for 2-channel)")
    args = parser.parse_args()
    
    train_vae_from_scratch(
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
        max_samples_per_dataset=args.max_samples_per_dataset,
        gram_type=args.gram_type,
        kl_weight=args.kl_weight
    )
