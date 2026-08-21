#!/usr/bin/env python3
"""
ENCODER-ONLY SCRIPT: Extract Latent Embeddings from Input Images

Loads a trained autoencoder and uses only the encoder portion (with frozen weights)
to extract latent embeddings from input spectrograms.

USAGE:
    source venv_bowhead/bin/activate
    
    # Single file
    python3 extract_latent_embeddings.py \
        --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder.pth \
        --input BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir/S308A0T20080828T000045_Type0.mat \
        --output latent_embedding.mat
    
    # Batch directory processing
    python3 extract_latent_embeddings.py \
        --model LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder.pth \
        --input_dir BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir \
        --output_dir latent_embeddings_output \
        --batch_size 64
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat, savemat
import os
import sys
import argparse
import glob
from typing import Optional, Tuple
import time


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
        
        # Decoder and from_latent not needed for encoding, but included for compatibility
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

    def encode(self, x):
        """Extract latent embeddings from input images."""
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.to_latent(x_flat)
        return latent

    def forward(self, x):
        """Full forward pass (for compatibility when loading weights)."""
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.to_latent(x_flat)
        x_recon = self.from_latent(latent)
        x_recon = x_recon.view(x_recon.size(0), self.c_out, self.nrow_reduced, self.ncol_reduced)
        output = self.decoder(x_recon)
        return output, latent


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _minmax_norm(im: np.ndarray) -> np.ndarray:
    """Min-max normalize image to [0, 1] range."""
    im = im.astype(np.float32)
    im_min = float(np.min(im))
    im_max = float(np.max(im))
    rng = im_max - im_min
    if rng < 1e-8:
        return np.zeros_like(im, dtype=np.float32)
    return (im - im_min) / rng


def load_snr_gram(file_path: str, normalize: bool = True) -> np.ndarray:
    """Load SNR_gram from .mat file."""
    try:
        mat_data = loadmat(file_path)
        snr_gram = mat_data.get('SNR_gram', None)
        if snr_gram is None or not isinstance(snr_gram, np.ndarray) or snr_gram.ndim != 2:
            raise ValueError(f"Invalid SNR_gram in {file_path}")
        
        if normalize:
            snr_gram = _minmax_norm(snr_gram)
        else:
            snr_gram = snr_gram.astype(np.float32)
        
        return snr_gram
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {e}")


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
    print(f"  Architecture: {nrow}×{ncol} → {latent_dim}D latent")
    print(f"  Device: {device}")
    print(f"  All weights frozen (requires_grad=False)")
    
    return model


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_single_file(model: nn.Module, input_path: str, output_path: str,
                       normalize: bool = True, device: torch.device = None) -> np.ndarray:
    """Extract latent embedding from single .mat file."""
    if device is None:
        device = next(model.parameters()).device
    
    # Load input
    snr_gram = load_snr_gram(input_path, normalize=normalize)
    
    # Convert to tensor
    input_tensor = torch.from_numpy(snr_gram).unsqueeze(0).unsqueeze(0).to(device)  # 1×1×H×W
    
    # Extract latent embedding
    with torch.no_grad():
        latent = model.encode(input_tensor)
    
    # Convert to numpy
    latent_np = latent.cpu().numpy().squeeze()  # Shape: (latent_dim,)
    
    # Save to .mat file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    savemat(output_path, {
        'latent_embedding': latent_np,
        'original_shape': snr_gram.shape,
        'input_file': os.path.basename(input_path)
    })
    
    print(f"✓ Extracted: {os.path.basename(input_path)} → {latent_np.shape}")
    print(f"  Saved to: {output_path}")
    
    return latent_np


def extract_batch_directory(model: nn.Module, input_dir: str, output_dir: str,
                            normalize: bool = True, batch_size: int = 64,
                            device: torch.device = None, file_limit: Optional[int] = None) -> Tuple[np.ndarray, list]:
    """Extract latent embeddings from all .mat files in directory."""
    if device is None:
        device = next(model.parameters()).device
    
    # Find all .mat files
    mat_files = sorted(glob.glob(os.path.join(input_dir, '**', '*.mat'), recursive=True))
    
    if file_limit is not None:
        mat_files = mat_files[:file_limit]
    
    if not mat_files:
        raise RuntimeError(f"No .mat files found in {input_dir}")
    
    print(f"\n{'='*70}")
    print(f"BATCH EXTRACTION: {len(mat_files)} files")
    print(f"{'='*70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_latents = []
    all_filenames = []
    
    start_time = time.time()
    
    # Process in batches
    for i in range(0, len(mat_files), batch_size):
        batch_files = mat_files[i:i+batch_size]
        batch_tensors = []
        batch_names = []
        
        # Load batch
        for file_path in batch_files:
            try:
                snr_gram = load_snr_gram(file_path, normalize=normalize)
                tensor = torch.from_numpy(snr_gram).unsqueeze(0)  # 1×H×W
                batch_tensors.append(tensor)
                batch_names.append(os.path.basename(file_path))
            except Exception as e:
                print(f"  Warning: Skipped {os.path.basename(file_path)}: {e}")
                continue
        
        if not batch_tensors:
            continue
        
        # Stack into batch
        batch = torch.stack(batch_tensors, dim=0).to(device)  # N×1×H×W
        
        # Extract latent embeddings
        with torch.no_grad():
            latents = model.encode(batch)
        
        # Convert to numpy and store
        latents_np = latents.cpu().numpy()  # N×latent_dim
        all_latents.append(latents_np)
        all_filenames.extend(batch_names)
        
        # Progress update
        processed = min(i + batch_size, len(mat_files))
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"  [{processed:6d}/{len(mat_files)}] {rate:6.1f} files/sec | Batch shape: {latents_np.shape}")
    
    # Concatenate all batches
    all_latents_np = np.vstack(all_latents)  # (total_samples, latent_dim)
    
    # Save combined output
    output_path = os.path.join(output_dir, 'latent_embeddings.mat')
    savemat(output_path, {
        'latent_embeddings': all_latents_np,
        'filenames': all_filenames,
        'num_samples': len(all_filenames),
        'latent_dim': all_latents_np.shape[1]
    })
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ EXTRACTION COMPLETE")
    print(f"  Total files: {len(all_filenames)}")
    print(f"  Output shape: {all_latents_np.shape}")
    print(f"  Total time: {total_time:.1f}s ({len(all_filenames)/total_time:.1f} files/sec)")
    print(f"  Saved to: {output_path}")
    print(f"{'='*70}\n")
    
    return all_latents_np, all_filenames


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract latent embeddings from spectrograms using trained autoencoder',
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
                       help='Input height (default: 121)')
    parser.add_argument('--ncol', type=int, default=104,
                       help='Input width (default: 104)')
    parser.add_argument('--extra_conv', action='store_true',
                       help='Use extra conv layer architecture')
    
    # Input/output (single file mode)
    parser.add_argument('--input', type=str,
                       help='Single input .mat file (mutually exclusive with --input_dir)')
    parser.add_argument('--output', type=str,
                       help='Output .mat file for single file mode')
    
    # Input/output (batch mode)
    parser.add_argument('--input_dir', type=str,
                       help='Input directory containing .mat files (mutually exclusive with --input)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for batch mode')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing (default: 64)')
    parser.add_argument('--file_limit', type=int, default=None,
                       help='Limit number of files to process (for testing)')
    
    # Processing options
    parser.add_argument('--no_normalize', action='store_true',
                       help='Skip min-max normalization')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cpu, cuda, mps (default: auto)')
    
    args = parser.parse_args()
    
    # Validate input modes
    if args.input and args.input_dir:
        parser.error("Cannot specify both --input and --input_dir")
    if not args.input and not args.input_dir:
        parser.error("Must specify either --input or --input_dir")
    
    # Single file mode validation
    if args.input and not args.output:
        parser.error("--output required when using --input")
    
    # Batch mode validation
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir required when using --input_dir")
    
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
    print(f"LATENT EMBEDDING EXTRACTION")
    print(f"{'='*70}")
    
    # Load model
    model = load_trained_model(
        args.model, device,
        nrow=args.nrow, ncol=args.ncol,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        extra_conv=args.extra_conv
    )
    
    normalize = not args.no_normalize
    
    # Execute appropriate mode
    if args.input:
        # Single file mode
        extract_single_file(
            model, args.input, args.output,
            normalize=normalize, device=device
        )
    else:
        # Batch directory mode
        extract_batch_directory(
            model, args.input_dir, args.output_dir,
            normalize=normalize,
            batch_size=args.batch_size,
            device=device,
            file_limit=args.file_limit
        )
    
    print(f"\n✓ All operations complete!\n")


if __name__ == '__main__':
    main()
