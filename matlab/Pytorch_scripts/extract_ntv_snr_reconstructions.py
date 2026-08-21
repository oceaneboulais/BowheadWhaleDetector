#!/usr/bin/env python3
"""
Extract separate NTV and SNR reconstructions from a trained autoencoder.
This script loads the trained model and generates reconstructions for both
SNR_gram and NTV_gram data separately, saving them as .mat files.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from scipy.io import loadmat, savemat
import glob
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = "/Users/oboulais/Public/Bowhead_DL_Project/LD32/Autoencoder_v100E_32LD_32C_Auto_SNR+NTV_100K_Date20260213-150900.dir"
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model", "autoencoder_clean.pth")
DATA_DIR = "/Users/oboulais/Desktop/Bowhead_DL_Project/BCB_Whale_Datasets"

OUTPUT_DIR = os.path.join(MODEL_DIR, "reconstructions_separate")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# MODEL ARCHITECTURE (MUST MATCH TRAINING)
# ============================================================================

class ImprovedAutoencoder(nn.Module):
    """Same architecture as training script"""

    def __init__(self, nrow=121, ncol=104, latent_dim=32,
                 base_channels=32, extra_conv=False, in_channels=1):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        self.extra_conv = extra_conv
        self.in_channels = in_channels

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
    """Dataset for loading single-channel spectrograms"""

    def __init__(self, directory: str, gram_type: str = 'SNR_gram', normalize: bool = True):
        self.normalize = normalize
        self.gram_type = gram_type
        self.file_paths = []

        mat_files = sorted(glob.glob(os.path.join(directory, '**', '*.mat'), recursive=True))
        target_shape = None

        for fp in mat_files:
            try:
                m = loadmat(fp)
                im = m.get(self.gram_type, None)
                if im is None or not isinstance(im, np.ndarray) or im.ndim != 2:
                    continue

                if target_shape is None:
                    target_shape = im.shape
                if im.shape == target_shape:
                    self.file_paths.append(fp)
            except Exception:
                continue

        if not self.file_paths:
            raise RuntimeError(f"No valid .mat files with {gram_type} found in {directory}")

        self.target_shape = target_shape
        print(f"Loaded {len(self.file_paths)} files with {gram_type} (shape: {target_shape})")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        fp = self.file_paths[idx]

        try:
            m = loadmat(fp)
            im = m[self.gram_type]
            if self.normalize:
                im = _minmax_norm(im)
            else:
                im = im.astype(np.float32)
            tensor = torch.from_numpy(im).unsqueeze(0)
            filename = os.path.basename(fp)
            return tensor, filename
        except Exception as e:
            h, w = self.target_shape
            return torch.zeros((1, h, w), dtype=torch.float32), f"error_{idx}"


def extract_reconstructions(gram_type: str = 'SNR_gram', max_samples: int = None):
    """Extract reconstructions for given gram type"""

    print(f"\n{'='*70}")
    print(f"Extracting {gram_type} Reconstructions")
    print(f"{'='*70}")

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    loaded = torch.load(MODEL_PATH, map_location=device)

    # The saved checkpoint is a state_dict directly
    state_dict = loaded

    # Find first conv layer to determine in_channels
    first_conv_weight = None
    for key, val in state_dict.items():
        if 'encoder' in key and '0.weight' in key:  # First conv2d
            first_conv_weight = val
            break

    if first_conv_weight is not None:
        in_channels = first_conv_weight.shape[1]
    else:
        in_channels = 2  # Default to 2 for SNR+NTV

    print(f"Model input channels from checkpoint: {in_channels}")

    # Initialize model with correct input channels
    model = ImprovedAutoencoder(nrow=121, ncol=104, latent_dim=32,
                                base_channels=32, extra_conv=False, in_channels=in_channels)
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Find data directories
    dataset_dirs = []
    if os.path.exists(DATA_DIR):
        for subdir in os.listdir(DATA_DIR):
            full_path = os.path.join(DATA_DIR, subdir)
            if os.path.isdir(full_path):
                dataset_dirs.append(full_path)

    if not dataset_dirs:
        print(f"Warning: No dataset directories found in {DATA_DIR}")
        return

    print(f"Found {len(dataset_dirs)} dataset directories")

    # Process each dataset
    all_originals = []
    all_reconstructions = []
    all_filenames = []

    for data_path in dataset_dirs:
        print(f"\nProcessing: {os.path.basename(data_path)}")

        try:
            dataset = SNRDataset(data_path, gram_type=gram_type, normalize=True)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

            sample_count = 0
            for batch_data, filenames in loader:
                if max_samples and sample_count >= max_samples:
                    break

                # Model expects 2-channel input, so we need to handle 1-channel input
                # by duplicating the channel or padding
                if batch_data.shape[1] == 1:
                    # Duplicate the single channel to 2 channels for the model
                    batch_data_2ch = torch.cat([batch_data, batch_data], dim=1)
                else:
                    batch_data_2ch = batch_data

                batch_data_2ch = batch_data_2ch.to(device)
                with torch.no_grad():
                    recon, _ = model(batch_data_2ch)

                # Handle shape mismatch
                if recon.shape[2:] != batch_data.shape[2:]:
                    # Center-crop reconstruction
                    rH, rW = recon.shape[2:]
                    tH, tW = batch_data.shape[2:]
                    if rH > tH:
                        dh = (rH - tH) // 2
                        recon = recon[:, :, dh:dh+tH, :]
                    if rW > tW:
                        dw = (rW - tW) // 2
                        recon = recon[:, :, :, dw:dw+tW]

                # Convert to numpy
                orig_np = batch_data.cpu().numpy().squeeze(1)  # Remove channel dim
                recon_np = recon.cpu().numpy().squeeze(1)

                all_originals.append(orig_np)
                all_reconstructions.append(recon_np)
                all_filenames.extend(filenames)
                sample_count += len(filenames)

            print(f"  Processed {sample_count} samples")

        except Exception as e:
            print(f"  Error processing {data_path}: {e}")

    if not all_originals:
        print(f"No data found for {gram_type}")
        return

    # Concatenate all data
    originals_stacked = np.vstack(all_originals)
    reconstructions_stacked = np.vstack(all_reconstructions)

    print(f"\nTotal samples: {len(all_filenames)}")
    print(f"Original shape: {originals_stacked.shape}")
    print(f"Reconstruction shape: {reconstructions_stacked.shape}")

    # Save to .mat file
    output_file = os.path.join(OUTPUT_DIR, f"{gram_type}_reconstructions.mat")
    save_dict = {
        'originals': originals_stacked,
        'reconstructions': reconstructions_stacked,
        'filenames': np.array(all_filenames, dtype=object)
    }
    savemat(output_file, save_dict)
    print(f"Saved to: {output_file}")

    return originals_stacked, reconstructions_stacked, all_filenames


if __name__ == "__main__":
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Extract SNR reconstructions
    extract_reconstructions(gram_type='SNR_gram', max_samples=None)

    # Extract NTV reconstructions
    extract_reconstructions(gram_type='NTV_gram', max_samples=None)

    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output saved to: {OUTPUT_DIR}")
