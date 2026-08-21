#!/usr/bin/env python3
"""
Extract latent vectors from new dataset using existing trained autoencoder.

This script:
1. Loads a trained autoencoder model (weights frozen)
2. Loads new data from Unsupervised_database_AutoWithAirguns_100K_Y08101214_centered.dir
3. Extracts latent embeddings via single forward pass (no training)
4. Saves latent vectors and reconstruction data

USAGE:
    python3 extract_latent_airguns.py
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import time
from typing import Optional, List, Tuple

# Model architecture (must match the trained model from Autoencoder_v03)
class ImprovedAutoencoder(nn.Module):
    """
    Hybrid autoencoder architecture matching the trained model.
    First conv layer uses 5×5 kernel, rest use 3×3.
    """
    
    def __init__(self, nrow=121, ncol=104, latent_dim=32, 
                 base_channels=32, extra_conv=False, kernel_size=5):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        self.latent_dim = latent_dim
        self.extra_conv = extra_conv
        self.kernel_size = kernel_size
        
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
                nn.Conv2d(1, c1, kernel_size, padding=first_padding),
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
                nn.Conv2d(1, c1, kernel_size, padding=first_padding),
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
        self.c3 = c3
        self.c4 = c4 if extra_conv else None

    def encode(self, x):
        """Extract latent representation"""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.to_latent(x)
        return z

    def decode(self, z):
        """Reconstruct from latent"""
        x = self.from_latent(z)
        if self.extra_conv:
            x = x.view(-1, self.c4, self.nrow_reduced, self.ncol_reduced)
        else:
            x = x.view(-1, self.c3, self.nrow_reduced, self.ncol_reduced)
        x = self.decoder(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class SNRDataset(Dataset):
    """Dataset for loading .mat files"""
    
    def __init__(self, directory: str, normalize: bool = True, 
                 max_samples: Optional[int] = None, gram_type: str = 'SNR_gram'):
        self.directory = directory
        self.normalize = normalize
        self.gram_type = gram_type
        
        # Find all .mat files
        print(f"Scanning directory: {directory}")
        self.file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.mat') and not file.startswith('airgun_index'):
                    self.file_paths.append(os.path.join(root, file))
        
        if max_samples:
            self.file_paths = self.file_paths[:max_samples]
        
        print(f"Found {len(self.file_paths)} .mat files")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        filepath = self.file_paths[idx]
        try:
            mat_data = loadmat(filepath)
            
            if self.gram_type == 'BOTH':
                snr = mat_data.get('SNR_gram', mat_data.get('snr_gram', None))
                ntv = mat_data.get('NTV_gram', mat_data.get('ntv_gram', None))
                if snr is None or ntv is None:
                    print(f"Warning: Missing SNR_gram or NTV_gram in {filepath}, using zeros")
                    gram = np.zeros((2, 121, 104), dtype=np.float32)
                else:
                    snr = snr.astype(np.float32)
                    ntv = ntv.astype(np.float32)
                    if self.normalize:
                        snr = self._minmax_norm(snr)
                        ntv = self._minmax_norm(ntv)
                    gram = np.stack([snr, ntv], axis=0)
            else:
                gram = mat_data.get(self.gram_type, mat_data.get(self.gram_type.lower(), None))
                if gram is None:
                    print(f"Warning: Missing {self.gram_type} in {filepath}, using zeros")
                    gram = np.zeros((121, 104), dtype=np.float32)
                else:
                    gram = gram.astype(np.float32)
                    if self.normalize:
                        gram = self._minmax_norm(gram)
                gram = np.expand_dims(gram, axis=0)
            
            return torch.from_numpy(gram), filepath
        except Exception as e:
            print(f"Warning: Error loading {filepath}: {e}, using zeros")
            # Return zero tensor on error to continue processing
            if self.gram_type == 'BOTH':
                gram = np.zeros((2, 121, 104), dtype=np.float32)
            else:
                gram = np.zeros((1, 121, 104), dtype=np.float32)
            return torch.from_numpy(gram), filepath
    
    def _minmax_norm(self, im: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]"""
        im_min = float(np.min(im))
        im_max = float(np.max(im))
        rng = im_max - im_min
        if rng < 1e-8:
            return im
        return (im - im_min) / rng


def extract_latent_vectors(model_dir: str, data_dir: str, output_dir: str, 
                           batch_size: int = 32, device: str = 'auto'):
    """
    Extract latent vectors from new dataset using trained model.
    
    Args:
        model_dir: Directory containing trained model (autoencoder_clean.pth)
        data_dir: Directory containing new .mat files
        output_dir: Directory to save latent vectors
        batch_size: Batch size for inference
        device: Device to use ('auto', 'cuda', 'mps', or 'cpu')
    """
    
    start_time = time.time()
    
    # Setup device
    if device == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device)
        print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_subdir = os.path.join(output_dir, f"Latent_Airguns_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)
    print(f"\nOutput directory: {output_subdir}")
    
    # Load trained model
    model_path = os.path.join(model_dir, 'trained_model', 'autoencoder_clean.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"\nLoading trained model from: {model_path}")
    
    # Initialize model with same architecture
    model = ImprovedAutoencoder(
        nrow=121,
        ncol=104,
        latent_dim=32,
        base_channels=32,
        extra_conv=False,
        kernel_size=3  # Model was trained with 3×3 first layer (not 5×5)
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Latent dimension: {model.latent_dim}")
    
    # Load new dataset
    print(f"\nLoading dataset from: {data_dir}")
    dataset = SNRDataset(data_dir, normalize=True, gram_type='SNR_gram')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Extract latent vectors
    print(f"\nExtracting latent vectors (batch_size={batch_size})...")
    all_latents = []
    all_filenames = []
    all_reconstructions = []
    
    with torch.no_grad():
        for batch_idx, (batch_data, batch_files) in enumerate(dataloader):
            batch_data = batch_data.to(device)
            
            # Forward pass
            recon, latent = model(batch_data)
            
            # Store results
            all_latents.append(latent.cpu().numpy())
            all_filenames.extend(batch_files)
            all_reconstructions.append(recon.cpu().numpy())
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(dataset)} samples...")
    
    # Concatenate all batches
    latents = np.concatenate(all_latents, axis=0)
    reconstructions = np.concatenate(all_reconstructions, axis=0)
    filenames = np.array([os.path.basename(f) for f in all_filenames], dtype=object)
    
    print(f"\n✓ Extraction complete!")
    print(f"  Total samples: {latents.shape[0]}")
    print(f"  Latent shape: {latents.shape}")
    print(f"  Time elapsed: {time.time() - start_time:.1f}s")
    
    # Save results
    print(f"\nSaving results...")
    
    # Save as .mat file (MATLAB compatible)
    mat_output = os.path.join(output_subdir, 'latent_embeddings_airguns.mat')
    savemat(mat_output, {
        'latent_vectors': latents,
        'filenames': filenames,
        'reconstruction_filenames': np.array([f"{os.path.splitext(fn)[0]}_reconstr.mat" for fn in filenames], dtype=object)
    })
    print(f"  ✓ Saved MATLAB file: {mat_output}")
    
    # Save as .npz file (Python/NumPy)
    npz_output = os.path.join(output_subdir, 'latent_embeddings_airguns.npz')
    np.savez_compressed(npz_output,
                       latent_vectors=latents,
                       filenames=filenames,
                       reconstructions=reconstructions)
    print(f"  ✓ Saved NumPy file: {npz_output}")
    
    # Save summary
    summary_path = os.path.join(output_subdir, 'extraction_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Latent Vector Extraction Summary\n")
        f.write(f"="*70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Output: {output_subdir}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Total samples: {latents.shape[0]}\n")
        f.write(f"  Latent dimension: {latents.shape[1]}\n")
        f.write(f"  Processing time: {time.time() - start_time:.1f}s\n")
        f.write(f"  Device: {device}\n")
    print(f"  ✓ Saved summary: {summary_path}")
    
    print(f"\n{'='*70}")
    print(f"✓ All done! Latent vectors extracted successfully.")
    print(f"{'='*70}")
    
    return output_subdir


if __name__ == "__main__":
    # Configuration
    MODEL_DIR = "/Users/oboulais/Public/Bowhead_DL_Project/LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.dir"
    DATA_DIR = "/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214_centered.dir"
    OUTPUT_DIR = "/Users/oboulais/Public/Bowhead_DL_Project/LD32"
    
    print("="*70)
    print("Latent Vector Extraction from New Dataset")
    print("="*70)
    print(f"Trained model: {MODEL_DIR}")
    print(f"New dataset: {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)
    
    # Run extraction
    output_path = extract_latent_vectors(
        model_dir=MODEL_DIR,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        batch_size=64,  # Larger batch for faster inference
        device='auto'
    )
    
    print(f"\nResults saved to: {output_path}")
