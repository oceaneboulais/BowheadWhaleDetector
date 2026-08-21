#!/usr/bin/env python3
"""
Generate UMAP and t-SNE embeddings from a trained autoencoder model.

Usage:
    python generate_embeddings.py --model_dir <path_to_model_dir> [--max_samples 10000]

Example:
    python generate_embeddings.py --model_dir ../results/Autoencoder_v14_100E_32LD_32C_Hybrid5x5-3x3_CombinedDatasets_100K_Date20260103-091003.dir
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat, savemat
from typing import List, Optional

# Try importing UMAP and t-SNE
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False
    print("Warning: sklearn not found. Install with: pip install scikit-learn")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not found. Install with: pip install umap-learn")


# ============================================================================
# MODEL ARCHITECTURE (must match training script)
# ============================================================================

class ImprovedAutoencoder(nn.Module):
    """Autoencoder architecture (must match training configuration).
    
    HYBRID ARCHITECTURE:
        - First conv layer: Uses kernel_size (default 5×5) to capture broad N/U curve shapes
        - Subsequent layers: Always use 3×3 to refine details efficiently
    """
    
    def __init__(self, nrow=121, ncol=104, latent_dim=32, 
                 base_channels=32, extra_conv=False, kernel_size=5):
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


def load_mat_files(directory: str, max_samples: Optional[int] = None) -> List[str]:
    """Find all .mat files in directory."""
    print(f"Searching for .mat files in: {directory}")
    mat_files = sorted(glob.glob(os.path.join(directory, '**', '*.mat'), recursive=True))
    
    if not mat_files:
        raise RuntimeError(f"No .mat files found in {directory}")
    
    # Validate and filter
    valid_files = []
    target_shape = None
    
    for fp in mat_files:
        try:
            m = loadmat(fp)
            im = m.get('SNR_gram', None)
            if im is None or not isinstance(im, np.ndarray) or im.ndim != 2:
                continue
            if target_shape is None:
                target_shape = im.shape
            if im.shape == target_shape:
                valid_files.append(fp)
        except Exception:
            continue
    
    if not valid_files:
        raise RuntimeError(f"No valid .mat files with consistent shape found")
    
    print(f"Found {len(valid_files)} valid .mat files with shape {target_shape}")
    
    if max_samples and len(valid_files) > max_samples:
        print(f"Limiting to {max_samples} samples")
        valid_files = valid_files[:max_samples]
    
    return valid_files


def extract_latent_embeddings(model: nn.Module, file_paths: List[str], 
                              device: torch.device, batch_size: int = 32) -> np.ndarray:
    """Extract latent embeddings from .mat files using trained model."""
    model.eval()
    all_latents = []
    
    print(f"\nExtracting latent embeddings from {len(file_paths)} samples...")
    
    batch = []
    for i, fp in enumerate(file_paths):
        try:
            m = loadmat(fp)
            im = m['SNR_gram']
            im = _minmax_norm(im)
            tensor = torch.from_numpy(im).unsqueeze(0)  # Add channel dim
            batch.append(tensor)
            
            if len(batch) == batch_size or i == len(file_paths) - 1:
                batch_tensor = torch.stack(batch).to(device)
                with torch.no_grad():
                    _, latent = model(batch_tensor)
                    all_latents.append(latent.cpu().numpy())
                batch = []
                
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(file_paths)} samples...")
        
        except Exception as e:
            print(f"Warning: Failed to load {fp}: {e}")
            continue
    
    embeddings = np.vstack(all_latents)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings


def compute_tsne(embeddings: np.ndarray, perplexity: float = 30.0, 
                random_state: int = 42) -> np.ndarray:
    """Compute t-SNE embedding."""
    if not HAS_TSNE:
        print("Skipping t-SNE (sklearn not installed)")
        return None
    
    print(f"\nComputing t-SNE (perplexity={perplexity})...")
    perplexity = min(perplexity, (embeddings.shape[0] - 1) / 3.0)
    perplexity = max(2.0, perplexity)
    
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity,
                learning_rate='auto', n_jobs=-1)
    tsne_embedding = tsne.fit_transform(embeddings)
    print(f"t-SNE embedding shape: {tsne_embedding.shape}")
    return tsne_embedding


def compute_umap(embeddings: np.ndarray, n_neighbors: int = 15, 
                min_dist: float = 0.1, random_state: int = 42) -> np.ndarray:
    """Compute UMAP embedding."""
    if not HAS_UMAP:
        print("Skipping UMAP (umap-learn not installed)")
        return None
    
    print(f"\nComputing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                       min_dist=min_dist, random_state=random_state)
    umap_embedding = reducer.fit_transform(embeddings)
    print(f"UMAP embedding shape: {umap_embedding.shape}")
    return umap_embedding


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate UMAP and t-SNE embeddings from trained autoencoder')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Path to model directory containing autoencoder_clean.pth')
    parser.add_argument('--data_dir', type=str, nargs='+', default=None,
                       help='Path(s) to data directory(ies) with .mat files (if different from training)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (default: all)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for embedding extraction (default: 32)')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension (default: 32)')
    parser.add_argument('--channels', type=int, default=32,
                       help='Base channels (default: 32)')
    parser.add_argument('--extra_conv', action='store_true',
                       help='Use extra convolutional layer (4 layers instead of 3)')
    parser.add_argument('--kernel_size', type=int, default=5,
                       help='First layer kernel size (default: 5 for hybrid 5x5-3x3)')
    parser.add_argument('--tsne_perplexity', type=float, default=30.0,
                       help='t-SNE perplexity (default: 30.0)')
    parser.add_argument('--umap_neighbors', type=int, default=15,
                       help='UMAP n_neighbors (default: 15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check dependencies
    if not HAS_TSNE and not HAS_UMAP:
        print("\nERROR: Neither sklearn nor umap-learn is installed!")
        print("Install with: pip install scikit-learn umap-learn")
        return
    
    # Find model file (check both root dir and checkpoints subfolder)
    model_path = os.path.join(args.model_dir, 'autoencoder_clean.pth')
    if not os.path.exists(model_path):
        # Try checkpoints subfolder
        model_path = os.path.join(args.model_dir, 'checkpoints', 'autoencoder_clean.pth')
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found in {args.model_dir} or {args.model_dir}/checkpoints/")
            return
    
    print(f"\nLoading model from: {model_path}")
    
    # Determine data directories
    if args.data_dir:
        data_dirs = args.data_dir
    else:
        # Try to find data directories from model training
        # Look for common dataset directories in the repo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        
        possible_dirs = [
            os.path.join(repo_root, 'models', '/Users/oboulais/Desktop/BowheadDeepLearningMATLAB/results/Autoencoder_v14_100E_32LD_32C_Hybrid5x5-3x3_CombinedDatasets_100K_Date20260103-091003.dir'),
            os.path.join(repo_root, 'models', '/Users/oboulais/Desktop/BowheadDeepLearningMATLAB/results/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir'),
            os.path.join(repo_root, 'models', '/Users/oboulais/Desktop/BowheadDeepLearningMATLAB/results/Autoencoder_v12_100E_32LD_32C_TrueHybrid5x5-3x3_CombinedDatasets_100K_Date20251224-104051.dir')

        ]
        
        data_dirs = [d for d in possible_dirs if os.path.exists(d)]
        
        if not data_dirs:
            print("ERROR: No data directories found. Specify with --data_dir")
            return
        
        print(f"\nAuto-detected data directories:")
        for d in data_dirs:
            print(f"  - {d}")
    
    # Load .mat files from all directories
    all_files = []
    for data_dir in data_dirs:
        files = load_mat_files(data_dir, max_samples=args.max_samples)
        all_files.extend(files)
    
    if args.max_samples and len(all_files) > args.max_samples:
        all_files = all_files[:args.max_samples]
    
    print(f"\nTotal files to process: {len(all_files)}")
    
    # Get sample shape from first file
    m = loadmat(all_files[0])
    sample_shape = m['SNR_gram'].shape
    nrow, ncol = sample_shape
    print(f"Sample shape: {nrow} x {ncol}")
    
    # Initialize model
    model = ImprovedAutoencoder(nrow=nrow, ncol=ncol, latent_dim=args.latent_dim,
                               base_channels=args.channels, extra_conv=args.extra_conv,
                               kernel_size=args.kernel_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Extract latent embeddings
    embeddings = extract_latent_embeddings(model, all_files, device, args.batch_size)
    
    # Compute t-SNE
    tsne_embedding = compute_tsne(embeddings, perplexity=args.tsne_perplexity, 
                                  random_state=args.seed)
    
    # Compute UMAP
    umap_embedding = compute_umap(embeddings, n_neighbors=args.umap_neighbors,
                                  random_state=args.seed)
    
    # Extract filenames
    filenames = [os.path.basename(f) for f in all_files]
    
    # Save to .mat file
    output_name = f"CombinedDataset_{args.latent_dim}LD_umap_tsne_embeddings.mat"
    output_path = os.path.join(args.model_dir, output_name)
    
    save_dict = {
        'latent_embeddings': embeddings,
        'filenames': np.array(filenames, dtype=object),
    }
    
    if tsne_embedding is not None:
        save_dict['tsne_embedding'] = tsne_embedding
    
    if umap_embedding is not None:
        save_dict['umap_embedding'] = umap_embedding
    
    print(f"\nSaving embeddings to: {output_path}")
    savemat(output_path, save_dict)
    print(f"✓ Saved successfully!")
    
    print(f"\nSummary:")
    print(f"  - Latent embeddings: {embeddings.shape}")
    if tsne_embedding is not None:
        print(f"  - t-SNE embedding: {tsne_embedding.shape}")
    if umap_embedding is not None:
        print(f"  - UMAP embedding: {umap_embedding.shape}")
    print(f"  - Filenames: {len(filenames)}")


if __name__ == '__main__':
    main()
