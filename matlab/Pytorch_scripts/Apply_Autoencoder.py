#!/usr/bin/env python3
"""
Apply a trained autoencoder to extract latent vectors from a dataset.

COMPUTATIONAL ORDER:
1. Parse command-line arguments (see main block at bottom)
2. Load trained model weights
3. Scan image folder and create dataset
4. Process images in batches through encoder
5. Collect latent vectors into matrix
6. Save results to .mat file

This script:
1. Loads a trained autoencoder from a specified directory
2. Processes all images from a target image folder
3. Extracts latent space vectors for each image
4. Saves the latent vectors as a matrix in the image folder
"""
import torch
import torch.nn as nn
import numpy as np
import os
import glob
import argparse
from datetime import datetime
from scipy.io import loadmat, savemat
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---- Default values ----
CHANNELS_DEFAULT = 128
LATENT_DIM_DEFAULT = 32
EXTRA_CONV_DEFAULT = False


# ============================================================================
# HELPER FUNCTIONS (Called during processing)
# ============================================================================

def _minmax_norm(im: np.ndarray, auto_skip_if_unit: bool = True) -> np.ndarray:
    """
    Min-max normalize unless data already resides in [0, 1].
    
    Used during image loading to ensure consistent input scaling.
    """
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
# MODEL ARCHITECTURE (Must match training script exactly!)
# ============================================================================

class ImprovedAutoencoder(nn.Module):
    """
    Improved autoencoder architecture (must match training script).
    
    CRITICAL: All parameters (latent_dim, base_channels, extra_conv) must
    match the values used during training, or the loaded weights won't work!
    """
    def __init__(self, nrow=121, ncol=104, latent_dim=LATENT_DIM_DEFAULT, base_channels=CHANNELS_DEFAULT,
                 extra_conv=EXTRA_CONV_DEFAULT):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        self.extra_conv = extra_conv
        
        # Calculate reduced dimensions based on depth
        if extra_conv:
            nrow_reduced = nrow // 16
            ncol_reduced = ncol // 16
        else:
            nrow_reduced = nrow // 8
            ncol_reduced = ncol // 8
        
        # Encoder with batch norm
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8  # Only used if extra_conv=True
        
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
        
        # Store decoder parameters (not needed for extraction but required for architecture)
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, flat_size),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (not used for latent extraction but part of model)
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
    
    def encode(self, x):
        """
        Extract latent vectors without reconstruction.
        
        This is the KEY method used for latent vector extraction.
        Only runs the encoder portion, skipping the decoder.
        """
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.to_latent(x_flat)
        return latent


# ============================================================================
# DATASET CLASS (Handles loading .mat files)
# ============================================================================

class ImageDataset(Dataset):
    """
    Dataset for loading .mat files containing SNR_gram images.
    
    Scans the image folder recursively, validates shapes, and provides
    batch-wise access to images during processing.
    """
    def __init__(self, image_folder: str, normalize: bool = True):
        """
        Initialize dataset by scanning folder for .mat files.
        
        RUNS DURING STEP 2: Validates all files have consistent shapes.
        """
        self.normalize = normalize
        self.file_paths = sorted(glob.glob(os.path.join(image_folder, '**', '*.mat'), recursive=True))
        
        if not self.file_paths:
            raise RuntimeError(f"No .mat files found in {image_folder}")
        
        # Determine target shape from first valid file
        self.target_shape = None
        valid_files = []
        for fp in self.file_paths:
            try:
                m = loadmat(fp)
                im = m.get('SNR_gram', None)
                if im is None or not isinstance(im, np.ndarray) or im.ndim != 2:
                    continue
                if self.target_shape is None:
                    self.target_shape = im.shape
                if im.shape == self.target_shape:
                    valid_files.append(fp)
            except Exception:
                continue
        
        self.file_paths = valid_files
        if not self.file_paths:
            raise RuntimeError(f"No valid .mat files with consistent shapes found in {image_folder}")
        
        print(f"Found {len(self.file_paths)} valid images with shape {self.target_shape}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Load and return a single image.
        
        CALLED DURING STEP 4: Loads images on-demand during batch processing.
        """
        fp = self.file_paths[idx]
        try:
            m = loadmat(fp)
            im = m['SNR_gram']
            
            if self.normalize:
                im = _minmax_norm(im)
            else:
                im = im.astype(np.float32)
            
            tensor = torch.from_numpy(im).unsqueeze(0)  # Add channel dimension
            return tensor, fp
        except Exception as e:
            print(f"Warning: Failed to load {fp}: {e}")
            h, w = self.target_shape
            return torch.zeros((1, h, w), dtype=torch.float32), fp


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_model_info_from_dir(model_dir: str) -> dict:
    """
    Extract version tag and timestamp from model directory name.
    
    RUNS DURING STEP 1: Parses directory name like
    "Autoencoder_v01_Date20251114-120000.dir" to get version "01"
    """
    dirname = os.path.basename(model_dir.rstrip('/'))
    # Expected format: Autoencoder_vXX_DateYYYYMMDD-HHMMSS.dir
    info = {'version': 'unknown', 'timestamp': 'unknown'}
    
    if 'Autoencoder_v' in dirname:
        parts = dirname.split('_')
        for part in parts:
            if part.startswith('v') and len(part) > 1:
                info['version'] = part[1:]  # Remove 'v' prefix
            if part.startswith('Date') and len(part) > 4:
                info['timestamp'] = part[4:]  # Remove 'Date' prefix
    
    return info


def apply_autoencoder(model_dir: str, image_folder: str, batch_size: int = 32,
                      latent_dim: int = LATENT_DIM_DEFAULT, channels: int = CHANNELS_DEFAULT,
                      extra_conv: bool = EXTRA_CONV_DEFAULT, normalize: bool = True):
    """
    Apply trained autoencoder to extract latent vectors from all images.
    
    MAIN PROCESSING FUNCTION - Called after command-line arguments are parsed.
    
    Args:
        model_dir: Directory containing the trained model (e.g., Autoencoder_v01_Date20251114-*.dir)
        image_folder: Directory containing .mat files to process
        batch_size: Batch size for processing
        latent_dim: Latent dimension (must match training)
        channels: Base channels (must match training)
        extra_conv: Whether extra conv layer was used (must match training)
        normalize: Apply min-max normalization to inputs
    """
    # ========================================================================
    # STEP 1: Locate and validate the trained model file
    # ========================================================================
    print(f"Loading model from: {model_dir}")
    print(f"Processing images from: {image_folder}")
    
    # Find the trained model weights file
    model_path = os.path.join(model_dir, 'improved_autoencoder.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Extract version tag from directory name (e.g., "01" from "Autoencoder_v01_Date...")
    model_info = extract_model_info_from_dir(model_dir)
    version_tag = model_info['version']
    
    # ========================================================================
    # STEP 2: Load and prepare the image dataset
    # ========================================================================
    # Create dataset that loads all .mat files from the image folder
    dataset = ImageDataset(image_folder, normalize=normalize)
    # Create dataloader for efficient batch processing
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Get image dimensions from first sample to configure model architecture
    sample, _ = dataset[0]
    nrow, ncol = sample.shape[-2], sample.shape[-1]
    print(f"Image dimensions: {nrow} x {ncol}")
    
    # ========================================================================
    # STEP 3: Initialize model architecture and load trained weights
    # ========================================================================
    # Create model with same architecture as training (must match exactly!)
    model = ImprovedAutoencoder(nrow=nrow, ncol=ncol, latent_dim=latent_dim,
                                base_channels=channels, extra_conv=extra_conv)
    
    # Load the trained weights from disk
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode (disables dropout, batch norm updates, etc.)
    print(f"Model loaded successfully with latent_dim={latent_dim}")
    
    # ========================================================================
    # STEP 4: Process all images and extract latent vectors
    # ========================================================================
    # Initialize lists to collect results from all batches
    all_latents = []
    all_filenames = []
    
    print(f"\nExtracting latent vectors from {len(dataset)} images...")
    # Process images in batches (more efficient than one-by-one)
    with torch.no_grad():  # Disable gradient computation (we're not training)
        for batch_data, batch_files in tqdm(dataloader, desc="Processing batches"):
            # Extract latent vectors for this batch using the encoder
            latents = model.encode(batch_data)
            # Store latent vectors and corresponding filenames
            all_latents.append(latents.cpu().numpy())
            all_filenames.extend(batch_files)
    
    # ========================================================================
    # STEP 5: Combine all batches into single matrix
    # ========================================================================
    # Stack all batch results into one large matrix: (n_images, latent_dim)
    # Example: 50,000 images with latent_dim=64 → 50,000 x 64 matrix
    latent_matrix = np.vstack(all_latents)  # Shape: (n_images, latent_dim)
    print(f"\nExtracted latent vectors: {latent_matrix.shape}")
    print(f"  - Number of images: {latent_matrix.shape[0]}")
    print(f"  - Latent dimension: {latent_matrix.shape[1]}")
    
    # ========================================================================
    # STEP 6: Prepare output filename with timestamp
    # ========================================================================
    # Create unique filename: LatentProcessor_v01_Date20251114-153045.mat
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = f"LatentProcessor_v{version_tag}_Date{timestamp}.mat"
    output_path = os.path.join(image_folder, output_filename)
    
    # ========================================================================
    # STEP 7: Package results with metadata
    # ========================================================================
    # Prepare comprehensive data structure for MATLAB compatibility
    output_data = {
        'latent_vectors': latent_matrix,  # Shape: (n_images, latent_dim) - THE MAIN OUTPUT
        'filenames': np.array(all_filenames, dtype=object),  # Corresponding image paths
        'model_dir': model_dir,  # Where the model came from
        'model_version': version_tag,  # Model version (e.g., "01")
        'latent_dim': latent_dim,  # Dimensionality of latent space
        'channels': channels,  # Model architecture parameter
        'extra_conv': extra_conv,  # Model architecture parameter
        'image_shape': np.array([nrow, ncol]),  # Original image dimensions
        'extraction_timestamp': timestamp,  # When vectors were created
        'num_images': latent_matrix.shape[0]  # Total number of processed images
    }
    
    # ========================================================================
    # STEP 8: Save results to .mat file in the image folder
    # ========================================================================
    # Write the latent vectors and metadata to a MATLAB-compatible .mat file
    savemat(output_path, output_data)
    print(f"\nSaved latent vectors to: {output_path}")
    print(f"  - Output format: {latent_matrix.shape[0]} x {latent_matrix.shape[1]} matrix")
    print(f"  - File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return output_path, latent_matrix


if __name__ == "__main__":
    # ========================================================================
    # SCRIPT EXECUTION BEGINS HERE
    # ========================================================================
    # Step 1: Parse command-line arguments to get model and data paths
    parser = argparse.ArgumentParser(description="Apply trained autoencoder to extract latent vectors")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Path to trained model directory (e.g., Autoencoder_v01_Date20251114-120000.dir)")
    parser.add_argument("--image-folder", type=str, required=True,
                       help="Path to folder containing .mat images to process")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for processing (default: 32)")
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM_DEFAULT,
                       help=f"Latent dimension (must match training, default: {LATENT_DIM_DEFAULT})")
    parser.add_argument("--channels", type=int, default=CHANNELS_DEFAULT,
                       help=f"Base channels (must match training, default: {CHANNELS_DEFAULT})")
    parser.add_argument("--extra-conv", action='store_true', default=EXTRA_CONV_DEFAULT,
                       help=f"Use extra conv layer (must match training, default: {EXTRA_CONV_DEFAULT})")
    parser.add_argument("--no-normalize", action='store_true',
                       help="Disable input normalization")
    
    args = parser.parse_args()
    
    # Step 2: Call main processing function with parsed arguments
    apply_autoencoder(
        model_dir=args.model_dir,
        image_folder=args.image_folder,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        channels=args.channels,
        extra_conv=args.extra_conv,
        normalize=not args.no_normalize
    )
