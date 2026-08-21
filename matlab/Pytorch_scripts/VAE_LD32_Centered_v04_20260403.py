#!/usr/bin/env python3
"""
SPECTROGRAM AUTOENCODER - Clean single-channel implementation

Loads standard spectrograms from .mat files and trains a convolutional autoencoder.

FEATURES:
- Single-channel spectrogram input
- Convolutional autoencoder with explicit weight initialization
- UMAP (2D, 3D, 5D) and t-SNE visualizations
- Automatic clustering analysis
- MATLAB-compatible output files
- Full reconstruction visualization

USAGE:
    source .venv_py31018/bin/activate
    python3 Autoencoder_Spectrogram_20260403.py \
        --data-dir /path/to/spectrogram/dataset \
        --epochs 100 \
        --latent-dim 32

EXPECTED DATA FORMAT:
    - .mat files containing a field named 'spectrogram' (2D numpy array)
    - All spectrograms must have the same shape
    - Files organized in a single directory or with subdirectories
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import glob
import math
import torch.nn.functional as F
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datetime import datetime
import time
import gc

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
from typing import Optional, Tuple, List

# ============================================================================
# CONFIGURATION
# ============================================================================

CHANNELS_DEFAULT = 32
LATENT_DIM_DEFAULT = 32
EXTRA_CONV_DEFAULT = False
EPOCHS_DEFAULT = 100
LR_DEFAULT = 1e-3
SEED_DEFAULT = 42
NUMBER_OUTPUT_IMAGE_SAMPLES = 30
SHOW_ERROR_PLOTS = False
ENABLE_UMAP = True
DEFAULT_VERSION_TAG = "SpectrogramVAE_100E_32LD"
TSNE_MAX_SAMPLES = None
MAX_SAMPLES_PER_DATASET_DEFAULT = 50000
KL_WEIGHT_DEFAULT = 0.001

# ============================================================================
# UTILITIES
# ============================================================================

def set_global_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_weights(model: nn.Module, seed: Optional[int] = None):
    """Initialize all model weights from scratch."""
    if seed is not None:
        set_global_seed(seed)
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def verify_random_weights(model: nn.Module) -> dict:
    """Verify that weights are properly initialized (not zeros)."""
    stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            stats[name] = {
                'mean': float(param.data.mean()),
                'std': float(param.data.std()),
                'min': float(param.data.min()),
                'max': float(param.data.max())
            }
    return stats


def match_shape_center(output: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """Center-crop or pad output to match target shape."""
    _, _, h, w = output.shape
    h_target, w_target = target_shape
    
    if h == h_target and w == w_target:
        return output
    
    if h > h_target:
        h_start = (h - h_target) // 2
        output = output[:, :, h_start:h_start+h_target, :]
    elif h < h_target:
        pad_h = h_target - h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        output = F.pad(output, (0, 0, pad_top, pad_bottom))
    
    _, _, h, w = output.shape
    if w > w_target:
        w_start = (w - w_target) // 2
        output = output[:, :, :, w_start:w_start+w_target]
    elif w < w_target:
        pad_w = w_target - w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        output = F.pad(output, (pad_left, pad_right, 0, 0))
    
    return output


def create_output_directory(version_tag: str) -> str:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    base_dir = '/Users/oboulais/Public/Bowhead_DL_Project/LD32'
    output_dir = os.path.join(base_dir, f'Autoencoder_{version_tag}_Date{timestamp}.dir')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'trained_model'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'image_results'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'MATLAB'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'UMAP'), exist_ok=True)
    
    return output_dir


# ============================================================================
# DATASET
# ============================================================================

class SpectrogramDataset(Dataset):
    """
    Memory-efficient dataset for loading spectrograms from .mat files.
    
    Expected .mat file structure:
        - 'spectrogram': 2D numpy array (frequency x time)
        
    Args:
        directory: Path to directory containing .mat files
        normalize: Whether to normalize spectrograms to [0, 1]
        seed: Random seed for shuffling
        show_summary: Print dataset summary
        max_samples: Maximum number of samples to load
        field_name: Name of the field in .mat files containing the spectrogram
    """
    
    def __init__(self, directory: str, normalize: bool = True,
                 seed: Optional[int] = None, show_summary: bool = False,
                 max_samples: Optional[int] = None, field_name: str = 'spectrogram'):
        self.normalize = normalize
        self.field_name = field_name
        self.file_paths: List[str] = []
        
        target_shape = None
        mat_files = sorted(glob.glob(os.path.join(directory, '**', '*.mat'), recursive=True))
        
        for fp in mat_files:
            try:
                m = loadmat(fp)
                im = m.get(self.field_name, None)
                if im is None or not isinstance(im, np.ndarray) or im.ndim != 2:
                    continue
                
                if target_shape is None:
                    target_shape = im.shape
                if im.shape == target_shape:
                    self.file_paths.append(fp)
            except Exception:
                continue
        
        if not self.file_paths:
            raise RuntimeError(f"No valid .mat files found in {directory} with field '{self.field_name}'")
        
        self.target_shape = target_shape
        
        # Shuffle before limiting
        if seed is not None:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(self.file_paths))
            self.file_paths = [self.file_paths[i] for i in indices]
        
        # Limit to max_samples
        total_found = len(self.file_paths)
        if max_samples is not None and max_samples < len(self.file_paths):
            self.file_paths = self.file_paths[:max_samples]
        
        if show_summary:
            if max_samples is not None and len(self.file_paths) < total_found:
                print(f"Spectrogram Dataset: {len(self.file_paths)} samples (limited from {total_found}) with shape {target_shape}")
            else:
                print(f"Spectrogram Dataset: {len(self.file_paths)} samples with shape {target_shape}")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and return a single spectrogram."""
        m = loadmat(self.file_paths[idx])
        im = m[self.field_name].astype(np.float32)
        
        if self.normalize:
            im_min = im.min()
            im_max = im.max()
            if im_max > im_min:
                im = (im - im_min) / (im_max - im_min)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        im = torch.from_numpy(im).unsqueeze(0)
        return im, idx


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SpectrogramVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for single-channel spectrograms.
    
    Architecture:
        Encoder: Conv layers -> Flatten -> (μ, log σ²) branches
        Reparameterization: z = μ + σ × ε  (where ε ~ N(0,1))
        Decoder: Linear -> Reshape -> ConvTranspose layers
    
    Args:
        nrow: Number of frequency bins (height)
        ncol: Number of time bins (width)
        latent_dim: Dimension of latent space
        base_channels: Base number of convolutional channels
        extra_conv: Whether to add an extra convolutional layer
    """
    
    def __init__(self, nrow: int, ncol: int, latent_dim: int = 32,
                 base_channels: int = 32, extra_conv: bool = False):
        super().__init__()
        
        self.nrow = nrow
        self.ncol = ncol
        self.latent_dim = latent_dim
        self.extra_conv = extra_conv
        
        # Encoder
        if extra_conv:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
            compression = 16
            final_channels = base_channels * 4
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
            compression = 8
            final_channels = base_channels * 4
        
        # Calculate flattened size after encoder
        h_enc = math.ceil(nrow / compression)
        w_enc = math.ceil(ncol / compression)
        self.flat_size = final_channels * h_enc * w_enc
        self.h_enc = h_enc
        self.w_enc = w_enc
        self.final_channels = final_channels
        
        # VAE Bottleneck - Probabilistic encoding
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.flat_size)
        
        # Decoder
        if extra_conv:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(final_channels, base_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
        else:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(final_channels, base_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ × ε
        where ε ~ N(0,1)
        
        This allows gradients to flow through the sampling operation.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Returns:
            reconstruction: Reconstructed spectrogram
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            z: Sampled latent vector
        """
        # Encode
        encoded = self.encoder(x)
        flat = encoded.view(encoded.size(0), -1)
        
        # Probabilistic latent encoding
        mu = self.fc_mu(flat)
        log_var = self.fc_log_var(flat)
        
        # Sample latent vector using reparameterization trick
        z = self.reparameterize(mu, log_var)
        
        # Decode
        decoded_flat = self.from_latent(z)
        decoded = decoded_flat.view(-1, self.final_channels, self.h_enc, self.w_enc)
        reconstruction = self.decoder(decoded)
        
        return reconstruction, mu, log_var, z


# ============================================================================
# TRAINING
# ============================================================================

def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, log_var: torch.Tensor, 
                     kl_weight: float = 0.001) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss = Reconstruction loss + KL divergence loss
    
    Args:
        recon_x: Reconstructed data
        x: Original data
        mu: Mean of latent distribution
        log_var: Log variance of latent distribution
        kl_weight: Weight for KL divergence term (balances reconstruction vs regularization)
    
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction MSE loss
        kl_loss: KL divergence loss (unweighted)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence loss: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = kl_loss / x.size(0)  # Normalize by batch size
    
    # Total loss with weighted KL term
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss


def train_autoencoder(data_dir: str, n_samples: int = 15, latent_dim: int = LATENT_DIM_DEFAULT,
                      channels: int = CHANNELS_DEFAULT, seed: Optional[int] = SEED_DEFAULT,
                      epochs: int = EPOCHS_DEFAULT, lr: float = LR_DEFAULT,
                      tsne_samples: Optional[int] = None, extra_conv: bool = EXTRA_CONV_DEFAULT,
                      batch_size: int = 32, output_samples: int = NUMBER_OUTPUT_IMAGE_SAMPLES,
                      version_tag: str = DEFAULT_VERSION_TAG, show_error: bool = SHOW_ERROR_PLOTS,
                      k_clusters: int = 2, tsne_perplexity: Optional[float] = None,
                      max_samples_per_dataset: Optional[int] = None,
                      field_name: str = 'spectrogram', kl_weight: float = KL_WEIGHT_DEFAULT):
    """
    Train Variational Autoencoder (VAE) from scratch.
    
    Args:
        data_dir: Directory or list of directories containing .mat files
        n_samples: Number of samples for visualization
        latent_dim: Dimension of latent space
        channels: Base number of convolutional channels
        seed: Random seed for reproducibility
        epochs: Number of training epochs
        lr: Learning rate
        tsne_samples: Number of samples for t-SNE (None = use n_samples)
        extra_conv: Add extra convolutional layer
        batch_size: Batch size for training
        output_samples: Number of output visualization samples
        version_tag: Version tag for output directory
        show_error: Show error plots in reconstruction visualization
        k_clusters: Number of clusters for k-means (0 = auto-select)
        tsne_perplexity: Perplexity for t-SNE (None = auto)
        max_samples_per_dataset: Maximum samples per dataset
        field_name: Name of spectrogram field in .mat files
        kl_weight: Weight for KL divergence term in loss (default: 0.001)
    """
    script_start_time = time.time()
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: {device} (Apple Metal)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} - {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (WARNING: No GPU detected)")
    sys.stdout.flush()
    
    # Initialize random seed
    if seed is not None:
        set_global_seed(int(seed))
        print(f"Random seed set to {seed}")
    
    output_dir = create_output_directory(version_tag)
    print(f"="*70)
    print(f"SPECTROGRAM VARIATIONAL AUTOENCODER (VAE)")
    print(f"="*70)
    print(f"Output: {output_dir}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: epochs={epochs}, lr={lr}, latent_dim={latent_dim}, channels={channels}, kl_weight={kl_weight}")
    print(f"="*70)
    
    # Load dataset(s)
    if isinstance(data_dir, str):
        print(f"\nLoading data from: {data_dir}")
        if max_samples_per_dataset:
            print(f"  Limiting to {max_samples_per_dataset:,} samples")
        dataset_label = os.path.basename(data_dir.rstrip('/'))
        dataset = SpectrogramDataset(data_dir, normalize=True, seed=seed, show_summary=True,
                                    max_samples=max_samples_per_dataset, field_name=field_name)
    elif isinstance(data_dir, (list, tuple)):
        print(f"\nLoading data from {len(data_dir)} directories:")
        if max_samples_per_dataset:
            print(f"  Limiting each dataset to {max_samples_per_dataset:,} samples")
        datasets = []
        for i, dir_path in enumerate(data_dir, 1):
            print(f"  [{i}] {dir_path}")
            ds = SpectrogramDataset(dir_path, normalize=True, seed=seed, show_summary=True,
                                   max_samples=max_samples_per_dataset, field_name=field_name)
            datasets.append(ds)
            print(f"      Using {len(ds):,} samples")
        dataset = ConcatDataset(datasets)
        dataset_label = f"Combined_{len(data_dir)}_datasets"
        print(f"\nTotal combined samples: {len(dataset):,}")
    else:
        raise ValueError(f"data_dir must be str or list/tuple, got {type(data_dir)}")
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
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
    
    # Initialize model
    nrow, ncol = data_tensor.shape[-2], data_tensor.shape[-1]
    print(f"\nInitializing VAE model from random weights...")
    print(f"Input shape: {data_tensor.shape}")
    
    model = SpectrogramVAE(nrow=nrow, ncol=ncol, latent_dim=latent_dim,
                          base_channels=channels, extra_conv=extra_conv)
    
    print(f"Explicitly initializing all {sum(p.numel() for p in model.parameters()):,} parameters...")
    initialize_weights(model, seed=seed)
    
    weight_stats = verify_random_weights(model)
    print(f"✓ Weight initialization verified:")
    print(f"  - {len(weight_stats)} parameter tensors initialized")
    print(f"  - Encoder conv1 weight std: {weight_stats.get('encoder.0.weight', {}).get('std', 0):.6f}")
    print(f"  - All weights are non-zero")
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: {4 if extra_conv else 3} conv layers, latent_dim={latent_dim}")
    print(f"VAE: Probabilistic encoding with KL divergence (weight={kl_weight})")
    print(f"="*70)
    
    # Train
    print(f"\nTraining VAE for {epochs} epochs...")
    sys.stdout.flush()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    losses = []
    recon_losses = []
    kl_losses = []
    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        batch_count = 0
        
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            # VAE forward pass returns (recon, mu, log_var, z)
            output, mu, log_var, z = model(batch_data)
            output = match_shape_center(output, (nrow, ncol))
            
            # VAE loss computation
            total_loss, recon_loss, kl_loss = vae_loss_function(output, batch_data, mu, log_var, kl_weight)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            batch_count += 1
        
        avg_total_loss = epoch_total_loss / batch_count if batch_count > 0 else 0.0
        avg_recon_loss = epoch_recon_loss / batch_count if batch_count > 0 else 0.0
        avg_kl_loss = epoch_kl_loss / batch_count if batch_count > 0 else 0.0
        
        losses.append(avg_total_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        epoch_time = time.time() - epoch_start
        
        with torch.no_grad():
            o_min = float(output.min().cpu())
            o_max = float(output.max().cpu())
        
        elapsed = time.time() - training_start_time
        eta_seconds = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
        eta_minutes = eta_seconds / 60
        print(f"  Epoch {epoch:3d}/{epochs}: Total={avg_total_loss:.4f} | Recon={avg_recon_loss:.4f} | KL={avg_kl_loss:.4f} | out[{o_min:.3f}, {o_max:.3f}] | {epoch_time:.1f}s | ETA: {eta_minutes:.1f}min")
        sys.stdout.flush()
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, 'trained_model', f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_loss': avg_total_loss,
                'recon_loss': avg_recon_loss,
                'kl_loss': avg_kl_loss,
                'losses': losses,
                'recon_losses': recon_losses,
                'kl_losses': kl_losses,
            }, checkpoint_path)
            print(f"  >> Checkpoint saved: {checkpoint_path}")
            sys.stdout.flush()
    
    training_elapsed = time.time() - training_start_time
    print(f"\nTraining complete: {training_elapsed:.1f}s ({training_elapsed/60:.1f}min)")
    sys.stdout.flush()
    
    # Save model
    model_path = os.path.join(output_dir, 'trained_model', 'spectrogram_vae.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved VAE model to: {model_path}")
    
    # Save model with descriptive .pt filename including dimensions
    descriptive_model_path = os.path.join(output_dir, 'trained_model', f'vae_clean_{nrow}x{ncol}_clean.pt')
    torch.save(model.state_dict(), descriptive_model_path)
    print(f"Saved VAE model to: {descriptive_model_path}")
    
    mat_path = os.path.join(output_dir, 'MATLAB', 'spectrogram_vae.mat')
    state_dict = model.state_dict()
    mat_dict = {key.replace('.', '_'): value.cpu().numpy() for key, value in state_dict.items()}
    savemat(mat_path, mat_dict)
    print(f"Saved VAE model to MATLAB format: {mat_path}")
    sys.stdout.flush()
    
    del train_loader
    gc.collect()
    
    # Extract latent embeddings
    model.eval()
    tsne_sample_count = min(TSNE_MAX_SAMPLES, len(dataset)) if TSNE_MAX_SAMPLES else len(dataset)
    print(f"\nExtracting {tsne_sample_count} latent embeddings...")
    all_latent = []
    with torch.no_grad():
        for i in range(tsne_sample_count):
            sample, _ = dataset[i]
            sample = sample.unsqueeze(0).to(device)
            _, mu, log_var, z = model(sample)
            all_latent.append(z.cpu())  # Use sampled z for embeddings
    latent_full = torch.cat(all_latent, dim=0)
    
    # Compute reconstructions
    with torch.no_grad():
        data_tensor = data_tensor.to(device)
        recon, _, _, _ = model(data_tensor)  # VAE returns 4 values
        recon = match_shape_center(recon, (nrow, ncol))
    
    # Visualizations
    data_np = data_tensor.squeeze(1).cpu().numpy()
    print(f"\nGenerating visualizations...")
    
    # Plot 1: Reconstructions
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
        recon_np = recon[i, 0].cpu().numpy()
        axes[1, i].imshow(recon_np, cmap='viridis', origin='lower', aspect='auto', vmin=vmin_data, vmax=vmax_data)
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
        if show_error:
            diff = np.abs(data_np[i] - recon_np)
            axes[2, i].imshow(diff, cmap='hot', origin='lower', aspect='auto')
            axes[2, i].set_title('Error')
            axes[2, i].axis('off')
    
    plt.suptitle(f'Spectrogram VAE (epochs={epochs}, latent_dim={latent_dim}, kl_weight={kl_weight})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_results', 'reconstructions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Training loss (Total, Reconstruction, KL Divergence)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Total Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(recon_losses, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('Reconstruction Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(kl_losses, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'VAE Training Losses (epochs={epochs}, kl_weight={kl_weight})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_results', 'training_loss.png'), dpi=150)
    plt.close()
    
    # Plot 3: t-SNE with clustering
    latent_np = latent_full.detach().cpu().numpy()
    if isinstance(data_dir, list):
        dataset_label = "CombinedDatasets"
    else:
        dataset_label = os.path.basename(data_dir.rstrip('/'))
    
    if TSNE is not None and latent_full.shape[0] > 2:
        try:
            print(f"Computing t-SNE...")
            perplexity = min(30.0, (latent_np.shape[0] - 1) / 3.0) if tsne_perplexity is None else tsne_perplexity
            perplexity = max(2.0, min(perplexity, latent_np.shape[0] - 1))
            
            emb = TSNE(n_components=2, random_state=int(seed) if seed else 0,
                      perplexity=perplexity, learning_rate='auto').fit_transform(latent_np)
            
            # Auto-find optimal k
            optimal_k = k_clusters
            if KMeans is not None and silhouette_score is not None and (k_clusters is None or k_clusters == 0):
                print("Finding optimal number of clusters...")
                max_k = min(10, latent_np.shape[0] // 2)
                silhouette_scores = []
                k_range = range(2, max_k + 1)
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=int(seed) if seed else 0, n_init=10)
                    cluster_labels = kmeans.fit_predict(latent_np)
                    score = silhouette_score(latent_np, cluster_labels)
                    silhouette_scores.append(score)
                
                optimal_k = k_range[np.argmax(silhouette_scores)]
                print(f"Optimal k={optimal_k} (silhouette={max(silhouette_scores):.3f})")
            elif k_clusters is None or k_clusters == 0:
                optimal_k = 2
            
            # Clustering
            if KMeans is not None:
                kmeans = KMeans(n_clusters=optimal_k, random_state=int(seed) if seed else 0, n_init=10)
                clusters = kmeans.fit_predict(latent_np)
            else:
                clusters = np.zeros(latent_np.shape[0], dtype=int)
            
            # Plot
            cmap_colors = plt.cm.get_cmap('tab10', optimal_k)
            plt.figure(figsize=(7, 6))
            
            for cluster_id in range(optimal_k):
                mask = clusters == cluster_id
                plt.scatter(emb[mask, 0], emb[mask, 1], c=[cmap_colors(cluster_id)],
                           label=f'Cluster {cluster_id}', s=20, alpha=0.6)
            
            plt.title(f't-SNE Latent Space (k={optimal_k}, perplexity={perplexity:.1f})')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend(loc='upper right', fontsize=8, framealpha=0.9)
            plt.figtext(0.99, 0.01, f'Dataset: {dataset_label}',
                       ha='right', va='bottom', fontsize=7, style='italic', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'image_results', 'tsne_latent.png'), dpi=160)
            plt.close()
        except Exception as e:
            print(f"Warning: t-SNE failed - {e}")
            emb = np.zeros((latent_np.shape[0], 2))
            clusters = np.zeros(latent_np.shape[0], dtype=int)
            optimal_k = 2
            perplexity = 30.0
    else:
        print("Warning: TSNE not available")
        emb = np.zeros((latent_np.shape[0], 2))
        clusters = np.zeros(latent_np.shape[0], dtype=int)
        optimal_k = 2
        perplexity = 30.0
    
    # Save latent embeddings
    print(f"Saving latent embeddings...")
    
    if isinstance(dataset, ConcatDataset):
        all_file_paths = []
        for ds in dataset.datasets:
            all_file_paths.extend(ds.file_paths)
        filenames = np.array([os.path.basename(all_file_paths[i]) for i in range(min(tsne_sample_count, len(all_file_paths)))], dtype=object)
    else:
        filenames = np.array([os.path.basename(dataset.file_paths[i]) for i in range(tsne_sample_count)], dtype=object)
    
    reconstruction_filenames = np.array([f"{os.path.splitext(fn)[0]}_reconstr.mat" for fn in filenames], dtype=object)
    
    latent_data = {
        'latent_embeddings': latent_np,
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
    
    # Plot 4: UMAP
    if ENABLE_UMAP and UMAP is not None and latent_full.shape[0] > 2:
        print(f"Computing UMAP embeddings...")
        try:
            umap_2d = UMAP(n_components=2, random_state=int(seed) if seed else 0, n_neighbors=15, min_dist=0.1)
            umap_embeddings_2d = umap_2d.fit_transform(latent_np)
            
            umap_3d = UMAP(n_components=3, random_state=int(seed) if seed else 0, n_neighbors=15, min_dist=0.1)
            umap_embeddings_3d = umap_3d.fit_transform(latent_np)
            
            umap_5d = UMAP(n_components=5, random_state=int(seed) if seed else 0, n_neighbors=15, min_dist=0.1)
            umap_embeddings_5d = umap_5d.fit_transform(latent_np)
            
            # Save consolidated UMAP
            umap_data = {
                'umap_embeddings_2d': umap_embeddings_2d,
                'umap_embeddings_3d': umap_embeddings_3d,
                'umap_embeddings_5d': umap_embeddings_5d,
                'clusters': clusters,
                'optimal_k': optimal_k
            }
            umap_path = os.path.join(output_dir, 'UMAP', 'umap_embeddings.mat')
            savemat(umap_path, umap_data)
            print(f"Saved UMAP embeddings (2D, 3D, 5D) to: {umap_path}")
            
            # Plot 2D UMAP
            plt.figure(figsize=(7, 6))
            for cluster_id in range(optimal_k):
                mask = clusters == cluster_id
                plt.scatter(umap_embeddings_2d[mask, 0], umap_embeddings_2d[mask, 1],
                           c=[cmap_colors(cluster_id)], label=f'Cluster {cluster_id}',
                           s=20, alpha=0.6)
            plt.title('UMAP 2D Projection')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.legend(loc='upper right', fontsize=8, framealpha=0.9)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'UMAP', 'umap_2d.png'), dpi=160)
            plt.close()
            
            # Plot 3D UMAP
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            for cluster_id in range(optimal_k):
                mask = clusters == cluster_id
                ax.scatter(umap_embeddings_3d[mask, 0], umap_embeddings_3d[mask, 1], umap_embeddings_3d[mask, 2],
                          c=[cmap_colors(cluster_id)], label=f'Cluster {cluster_id}', s=20, alpha=0.6)
            ax.set_title('UMAP 3D Projection')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_zlabel('UMAP 3')
            ax.legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'UMAP', 'umap_3d.png'), dpi=160)
            plt.close()
            
            print(f"✓ UMAP complete: 2D, 3D, 5D embeddings saved")
        except Exception as e:
            print(f"Warning: UMAP failed - {e}")
    
    # Final summary
    total_elapsed = time.time() - script_start_time
    print(f"\n{'='*70}")
    print(f"VAE TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"Final total loss: {losses[-1]:.6f}")
    print(f"Final reconstruction loss: {recon_losses[-1]:.6f}")
    print(f"Final KL divergence: {kl_losses[-1]:.6f}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Spectrogram Variational Autoencoder (VAE)')
    
    parser.add_argument('--data-dir',
                       nargs='+',
                       default=[
                           "/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214_centered.dir",
                           "/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214_centered.dir"
                       ],
                       help='One or more directories containing .mat files')
    parser.add_argument('--field-name', type=str, default='SNR_gram',
                       help='Field name in .mat files (default: SNR_gram)')
    parser.add_argument('--epochs', type=int, default=EPOCHS_DEFAULT,
                       help=f'Number of epochs (default: {EPOCHS_DEFAULT})')
    parser.add_argument('--latent-dim', type=int, default=LATENT_DIM_DEFAULT,
                       help=f'Latent dimension (default: {LATENT_DIM_DEFAULT})')
    parser.add_argument('--channels', type=int, default=CHANNELS_DEFAULT,
                       help=f'Base channels (default: {CHANNELS_DEFAULT})')
    parser.add_argument('--lr', type=float, default=LR_DEFAULT,
                       help=f'Learning rate (default: {LR_DEFAULT})')
    parser.add_argument('--seed', type=int, default=SEED_DEFAULT,
                       help=f'Random seed (default: {SEED_DEFAULT})')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--max-samples', type=int, default=50000,
                       help='Maximum samples per dataset (default: 50000)')
    parser.add_argument('--extra-conv', action='store_true',
                       help='Add extra convolutional layer')
    parser.add_argument('--version-tag', type=str, default=DEFAULT_VERSION_TAG,
                       help='Version tag for output directory')
    parser.add_argument('--k-clusters', type=int, default=0,
                       help='Number of clusters (0=auto-select)')
    parser.add_argument('--kl-weight', type=float, default=KL_WEIGHT_DEFAULT,
                       help=f'KL divergence weight (default: {KL_WEIGHT_DEFAULT})')
    
    args = parser.parse_args()
    
    train_autoencoder(
        data_dir=args.data_dir,
        latent_dim=args.latent_dim,
        channels=args.channels,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.lr,
        extra_conv=args.extra_conv,
        batch_size=args.batch_size,
        version_tag=args.version_tag,
        k_clusters=args.k_clusters,
        max_samples_per_dataset=args.max_samples,
        field_name=args.field_name,
        kl_weight=args.kl_weight
    )


if __name__ == '__main__':
    main()
