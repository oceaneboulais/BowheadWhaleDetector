#!/usr/bin/env python3
"""
DECODER-ONLY SCRIPT: Reconstruct Images from Latent Embeddings

Loads a trained autoencoder and uses only the decoder portion (with frozen weights)
to reconstruct spectrograms from latent embeddings.

USAGE:
    source venv_bowhead/bin/activate
    
    # Single latent embedding
    python3 reconstruct_from_latent.py \
        --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder.pth \
        --input latent_embedding.mat \
        --output reconstructed_spectrogram.mat
    
    # Batch of latent embeddings
    python3 reconstruct_from_latent.py \
        --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder.pth \
        --input latent_embeddings.mat \
        --output reconstructed_batch.mat \
        --batch_size 64
    
    # Generate visualizations
    python3 reconstruct_from_latent.py \
        --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder.pth \
        --input latent_embeddings.mat \
        --output reconstructed_batch.mat \
        --save_images \
        --image_dir reconstructed_images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import argparse
from typing import Optional, Tuple
import time
import math

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ============================================================================
# MODEL ARCHITECTURE (must match training script)
# ============================================================================

class ImprovedAutoencoder(nn.Module):
    """Autoencoder architecture - matches Autoencoder_v02_LD32_20251118.py"""
    
    def __init__(self, nrow=121, ncol=104, latent_dim=32, 
                 base_channels=32, extra_conv=False):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        self.extra_conv = extra_conv
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
        
        if extra_conv:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, c1, 3, padding=1),
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
                nn.Conv2d(1, c1, 3, padding=1),
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

    def decode(self, latent):
        """Reconstruct images from latent embeddings."""
        x_recon = self.from_latent(latent)
        x_recon = x_recon.view(x_recon.size(0), self.c_out, self.nrow_reduced, self.ncol_reduced)
        output = self.decoder(x_recon)
        return output

    def forward(self, x):
        """Full forward pass (for compatibility when loading weights)."""
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.to_latent(x_flat)
        x_recon = self.from_latent(latent)
        x_recon = x_recon.view(x_recon.size(0), self.c_out, self.nrow_reduced, self.ncol_reduced)
        output = self.decoder(x_recon)
        return output, latent


def match_shape_center(recon: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_trained_model(model_path: str, device: torch.device,
                      nrow: int = 121, ncol: int = 104,
                      latent_dim: int = 32, base_channels: int = 32,
                      extra_conv: bool = False) -> nn.Module:
    """Load trained autoencoder with frozen weights."""
    model = ImprovedAutoencoder(
        nrow=nrow, ncol=ncol, latent_dim=latent_dim,
        base_channels=base_channels, extra_conv=extra_conv
    )
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"✓ Loaded model from: {model_path}")
    print(f"  Architecture: {latent_dim}D latent → {nrow}×{ncol}")
    print(f"  Device: {device}")
    print(f"  All weights frozen (requires_grad=False)")
    
    return model


def load_latent_embeddings(input_path: str) -> Tuple[np.ndarray, Optional[list]]:
    """Load latent embeddings from .mat file.
    
    Returns:
        latent_embeddings: (N, latent_dim) array
        filenames: Optional list of filenames
    """
    try:
        mat_data = loadmat(input_path)
        
        # Try different possible field names
        if 'latent_embedding' in mat_data:
            latents = mat_data['latent_embedding']
        elif 'latent_embeddings' in mat_data:
            latents = mat_data['latent_embeddings']
        elif 'latent' in mat_data:
            latents = mat_data['latent']
        else:
            raise ValueError(f"No latent embedding field found in {input_path}")
        
        # Ensure 2D array (N, latent_dim)
        if latents.ndim == 1:
            latents = latents[np.newaxis, :]
        
        # Load filenames if available
        filenames = None
        if 'filenames' in mat_data:
            filenames = mat_data['filenames']
            if isinstance(filenames, np.ndarray):
                filenames = [str(f) for f in filenames.flat]
        
        return latents, filenames
        
    except Exception as e:
        raise RuntimeError(f"Failed to load {input_path}: {e}")


def save_large_array(output_path: str, reconstructions: np.ndarray, 
                     filenames: Optional[list] = None, verbose: bool = True):
    """Save reconstructions using appropriate format based on size.
    
    MATLAB v5 format has 2GB limit. For large arrays:
    - Use HDF5 (.h5) format if h5py available
    - Otherwise split into multiple .mat files
    
    Args:
        output_path: Output file path (.mat or .h5)
        reconstructions: (N, H, W) array of reconstructions
        filenames: Optional list of filenames
        verbose: Print progress messages
    """
    # Calculate array size in bytes
    array_size_bytes = reconstructions.nbytes
    array_size_gb = array_size_bytes / (1024**3)
    
    # MATLAB v5 format limit is ~2GB (actually 2^31 - 1 bytes)
    MAT_V5_LIMIT = 2 * 1024**3
    
    if array_size_bytes < MAT_V5_LIMIT:
        # Small enough for regular .mat file
        if verbose:
            print(f"Saving {reconstructions.shape[0]} reconstructions ({array_size_gb:.2f} GB) to .mat file...")
        
        savemat(output_path, {
            'reconstructed_spectrograms': reconstructions,
            'filenames': filenames if filenames else [],
            'num_samples': reconstructions.shape[0],
            'shape': reconstructions.shape
        })
        
        if verbose:
            print(f"✓ Saved to: {output_path}")
    
    else:
        # Too large for MATLAB v5 format
        if verbose:
            print(f"⚠ Array is too large for MATLAB v5 format ({array_size_gb:.2f} GB > 2 GB)")
        
        # Try HDF5 format first
        if HAS_H5PY and (output_path.endswith('.h5') or output_path.endswith('.hdf5')):
            if verbose:
                print(f"Using HDF5 format (.h5): {output_path}")
            
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('reconstructed_spectrograms', data=reconstructions, 
                               compression='gzip', compression_opts=4)
                f.create_dataset('num_samples', data=reconstructions.shape[0])
                f.create_dataset('shape', data=reconstructions.shape)
                
                if filenames:
                    # Store filenames as variable-length strings
                    dt = h5py.string_dtype(encoding='utf-8')
                    f.create_dataset('filenames', data=filenames, dtype=dt)
            
            if verbose:
                print(f"✓ Saved to HDF5: {output_path}")
                print(f"  Load in MATLAB: data = h5read('{os.path.basename(output_path)}', '/reconstructed_spectrograms');")
        
        elif HAS_H5PY:
            # Auto-switch to HDF5
            h5_path = output_path.replace('.mat', '.h5')
            if verbose:
                print(f"Auto-switching to HDF5 format: {h5_path}")
            
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset('reconstructed_spectrograms', data=reconstructions, 
                               compression='gzip', compression_opts=4)
                f.create_dataset('num_samples', data=reconstructions.shape[0])
                f.create_dataset('shape', data=reconstructions.shape)
                
                if filenames:
                    dt = h5py.string_dtype(encoding='utf-8')
                    f.create_dataset('filenames', data=filenames, dtype=dt)
            
            if verbose:
                print(f"✓ Saved to HDF5: {h5_path}")
                print(f"  Load in MATLAB: data = h5read('{os.path.basename(h5_path)}', '/reconstructed_spectrograms');")
        
        else:
            # Split into multiple .mat files
            if verbose:
                print(f"h5py not available. Splitting into multiple .mat files...")
            
            num_samples = reconstructions.shape[0]
            # Calculate chunk size to keep each file under 1.8GB (safety margin)
            bytes_per_sample = reconstructions[0].nbytes
            chunk_size = int((1.8 * 1024**3) / bytes_per_sample)
            num_chunks = math.ceil(num_samples / chunk_size)
            
            base_path = output_path.replace('.mat', '')
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, num_samples)
                chunk_data = reconstructions[start_idx:end_idx]
                chunk_filenames = filenames[start_idx:end_idx] if filenames else []
                
                chunk_path = f"{base_path}_part{i+1:03d}.mat"
                
                savemat(chunk_path, {
                    'reconstructed_spectrograms': chunk_data,
                    'filenames': chunk_filenames,
                    'num_samples': chunk_data.shape[0],
                    'shape': chunk_data.shape,
                    'chunk_info': f'Part {i+1}/{num_chunks}, samples {start_idx}-{end_idx-1}'
                })
                
                if verbose:
                    print(f"  ✓ Saved part {i+1}/{num_chunks}: {chunk_path} ({chunk_data.shape[0]} samples)")
            
            if verbose:
                print(f"✓ Saved {num_chunks} files: {base_path}_part001.mat to {base_path}_part{num_chunks:03d}.mat")


# ============================================================================
# RECONSTRUCTION FUNCTIONS
# ============================================================================

def reconstruct_single(model: nn.Module, latent: np.ndarray,
                       target_hw: Tuple[int, int],
                       device: torch.device = None) -> np.ndarray:
    """Reconstruct spectrogram from single latent embedding."""
    if device is None:
        device = next(model.parameters()).device
    
    # Convert to tensor
    if latent.ndim == 1:
        latent_tensor = torch.from_numpy(latent).unsqueeze(0).float().to(device)  # 1×latent_dim
    else:
        latent_tensor = torch.from_numpy(latent).float().to(device)
    
    # Decode
    with torch.no_grad():
        reconstruction = model.decode(latent_tensor)
        reconstruction = match_shape_center(reconstruction, target_hw)
    
    # Convert to numpy
    recon_np = reconstruction.squeeze().cpu().numpy()  # H×W
    
    return recon_np


def reconstruct_batch(model: nn.Module, latents: np.ndarray,
                     target_hw: Tuple[int, int],
                     batch_size: int = 64,
                     device: torch.device = None,
                     verbose: bool = True) -> np.ndarray:
    """Reconstruct spectrograms from batch of latent embeddings."""
    if device is None:
        device = next(model.parameters()).device
    
    num_samples = latents.shape[0]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"BATCH RECONSTRUCTION: {num_samples} samples")
        print(f"{'='*70}")
    
    all_reconstructions = []
    start_time = time.time()
    
    # Process in batches
    for i in range(0, num_samples, batch_size):
        batch_latents = latents[i:i+batch_size]
        
        # Convert to tensor
        latent_tensor = torch.from_numpy(batch_latents).float().to(device)
        
        # Decode
        with torch.no_grad():
            reconstructions = model.decode(latent_tensor)
            reconstructions = match_shape_center(reconstructions, target_hw)
        
        # Convert to numpy
        recons_np = reconstructions.squeeze(1).cpu().numpy()  # N×H×W
        all_reconstructions.append(recons_np)
        
        # Progress update
        if verbose:
            processed = min(i + batch_size, num_samples)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  [{processed:6d}/{num_samples}] {rate:6.1f} samples/sec | Batch shape: {recons_np.shape}")
    
    # Concatenate all batches
    all_recons = np.concatenate(all_reconstructions, axis=0)  # (N, H, W)
    
    if verbose:
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✓ RECONSTRUCTION COMPLETE")
        print(f"  Total samples: {num_samples}")
        print(f"  Output shape: {all_recons.shape}")
        print(f"  Total time: {total_time:.1f}s ({num_samples/total_time:.1f} samples/sec)")
        print(f"{'='*70}\n")
    
    return all_recons


def save_reconstruction_images(reconstructions: np.ndarray, output_dir: str,
                               filenames: Optional[list] = None,
                               max_images: int = 100,
                               freq_max_hz: float = 500.0,
                               time_duration_sec: float = 3.0):
    """Save reconstructed spectrograms as PNG images."""
    os.makedirs(output_dir, exist_ok=True)
    
    num_to_save = min(reconstructions.shape[0], max_images)
    
    print(f"\nSaving {num_to_save} reconstruction images to: {output_dir}")
    
    for i in range(num_to_save):
        recon = reconstructions[i]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot spectrogram
        im = ax.imshow(recon, cmap='viridis', origin='lower', aspect='auto',
                      extent=[0, time_duration_sec, 0, freq_max_hz])
        
        # Labels
        if filenames and i < len(filenames):
            title = filenames[i].replace('.mat', '')
        else:
            title = f'Reconstruction {i+1}'
        
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', fontsize=10)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Amplitude', fontsize=10)
        
        # Save
        output_path = os.path.join(output_dir, f'reconstruction_{i:05d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if (i + 1) % 20 == 0:
            print(f"  Saved {i+1}/{num_to_save} images...")
    
    print(f"✓ Saved {num_to_save} images to {output_dir}\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct spectrograms from latent embeddings using trained autoencoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model parameters
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained autoencoder .pth file')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension (default: 32)')
    parser.add_argument('--base_channels', type=int, default=32,
                       help='Base channels (default: 32)')
    parser.add_argument('--nrow', type=int, default=121,
                       help='Output height (default: 121)')
    parser.add_argument('--ncol', type=int, default=104,
                       help='Output width (default: 104)')
    parser.add_argument('--extra_conv', action='store_true',
                       help='Use extra conv layer architecture')
    
    # Input/output
    parser.add_argument('--input', type=str, required=True,
                       help='Input .mat file containing latent embeddings')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file for reconstructed spectrograms (.mat or .h5). '
                            'Auto-switches to HDF5 for arrays >2GB')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing (default: 64)')
    
    # Visualization options
    parser.add_argument('--save_images', action='store_true',
                       help='Save reconstructed spectrograms as PNG images')
    parser.add_argument('--image_dir', type=str, default='reconstructed_images',
                       help='Directory for saved images (default: reconstructed_images)')
    parser.add_argument('--max_images', type=int, default=100,
                       help='Maximum number of images to save (default: 100)')
    
    # Processing options
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cpu, cuda, mps (default: auto)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n{'='*70}")
    print(f"SPECTROGRAM RECONSTRUCTION FROM LATENT EMBEDDINGS")
    print(f"{'='*70}")
    
    # Load model
    model = load_trained_model(
        args.model, device,
        nrow=args.nrow, ncol=args.ncol,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        extra_conv=args.extra_conv
    )
    
    # Load latent embeddings
    print(f"\nLoading latent embeddings from: {args.input}")
    latents, filenames = load_latent_embeddings(args.input)
    print(f"✓ Loaded {latents.shape[0]} latent embeddings (dim={latents.shape[1]})")
    
    # Reconstruct
    target_hw = (args.nrow, args.ncol)
    
    if latents.shape[0] == 1:
        # Single reconstruction
        recon = reconstruct_single(model, latents, target_hw, device)
        reconstructions = recon[np.newaxis, :, :]  # Add batch dimension
    else:
        # Batch reconstruction
        reconstructions = reconstruct_batch(
            model, latents, target_hw,
            batch_size=args.batch_size,
            device=device,
            verbose=not args.quiet
        )
    
    # Save reconstructions
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    save_large_array(
        args.output, 
        reconstructions,
        filenames=filenames,
        verbose=not args.quiet
    )
    
    # Save images if requested
    if args.save_images:
        save_reconstruction_images(
            reconstructions, args.image_dir,
            filenames=filenames,
            max_images=args.max_images
        )
    
    print(f"\n✓ All operations complete!\n")


if __name__ == '__main__':
    main()
