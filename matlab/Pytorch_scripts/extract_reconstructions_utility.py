#!/usr/bin/env python3
"""
Utility to extract SEPARATE SNR and NTV reconstructions from the trained autoencoder.

Usage:
    python3 extract_reconstructions_utility.py --input-files file1.mat file2.mat [...]
    or
    python3 extract_reconstructions_utility.py --input-dir /path/to/data
    or
    python3 extract_reconstructions_utility.py --extract-all (extracts all 50,000 samples)
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from scipy.io import loadmat, savemat
import argparse
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = "/Users/oboulais/Public/Bowhead_DL_Project/LD32/Autoencoder_v100E_32LD_32C_Auto_SNR+NTV_100K_Date20260213-150900.dir"
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model", "autoencoder_clean.pth")
LATENT_EMBED_PATH = os.path.join(MODEL_DIR, "MATLAB", "latent_embeddings.mat")
OUTPUT_DIR = os.path.join(MODEL_DIR, "reconstructions_separate")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ImprovedAutoencoder(nn.Module):
    """Trained autoencoder architecture"""

    def __init__(self, nrow=121, ncol=104, latent_dim=32,
                 base_channels=32, extra_conv=False, in_channels=2):
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


def _minmax_norm(im: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]"""
    im = im.astype(np.float32)
    im_min = float(np.min(im))
    im_max = float(np.max(im))
    rng = im_max - im_min
    if rng < 1e-8:
        return np.zeros_like(im, dtype=np.float32)
    return (im - im_min) / rng


def load_model(device):
    """Load trained model"""
    print(f"Loading model from: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model = ImprovedAutoencoder(nrow=121, ncol=104, latent_dim=32,
                                base_channels=32, extra_conv=False, in_channels=2)
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")
    return model


def extract_snr_ntv_from_files(file_paths, output_prefix, model, device):
    """Extract SNR and NTV reconstructions from a list of .mat files"""

    snr_originals = []
    ntv_originals = []
    snr_reconstructions = []
    ntv_reconstructions = []
    filenames = []

    print(f"\nProcessing {len(file_paths)} files...")

    for file_path in file_paths:
        try:
            data = loadmat(file_path)
            snr = data.get('SNR_gram', None)
            ntv = data.get('NTV_gram', None)

            if snr is None or ntv is None:
                print(f"  Skipping {os.path.basename(file_path)}: missing SNR or NTV")
                continue

            # Normalize
            snr_norm = _minmax_norm(snr)
            ntv_norm = _minmax_norm(ntv)

            snr_originals.append(snr_norm)
            ntv_originals.append(ntv_norm)
            filenames.append(os.path.basename(file_path))

        except Exception as e:
            print(f"  Error loading {file_path}: {e}")

    if not snr_originals:
        print("ERROR: No valid files found!")
        return

    # Generate reconstructions in batches
    print(f"Generating {len(snr_originals)} reconstructions...")

    with torch.no_grad():
        for i in range(0, len(snr_originals), 32):
            batch_end = min(i + 32, len(snr_originals))
            batch_size = batch_end - i

            # Create 2-channel batches
            snr_batch = np.stack(snr_originals[i:batch_end], axis=0)
            ntv_batch = np.stack(ntv_originals[i:batch_end], axis=0)

            snr_2ch = np.stack([snr_batch, snr_batch], axis=1)
            ntv_2ch = np.stack([ntv_batch, ntv_batch], axis=1)

            # Forward pass
            snr_2ch_t = torch.from_numpy(snr_2ch).float().to(device)
            ntv_2ch_t = torch.from_numpy(ntv_2ch).float().to(device)

            snr_recon, _ = model(snr_2ch_t)
            ntv_recon, _ = model(ntv_2ch_t)

            snr_reconstructions.append(snr_recon[:, 0, :, :].cpu().numpy())
            ntv_reconstructions.append(ntv_recon[:, 0, :, :].cpu().numpy())

    # Concatenate results
    snr_orig_arr = np.stack(snr_originals, axis=0)
    ntv_orig_arr = np.stack(ntv_originals, axis=0)
    snr_recon_arr = np.vstack(snr_reconstructions)
    ntv_recon_arr = np.vstack(ntv_reconstructions)

    # Save SNR
    snr_file = os.path.join(OUTPUT_DIR, f"{output_prefix}_SNR_reconstructions.mat")
    savemat(snr_file, {
        'originals': snr_orig_arr,
        'reconstructions': snr_recon_arr,
        'filenames': np.array(filenames, dtype=object)
    })
    print(f"Saved SNR: {snr_file}")

    # Save NTV
    ntv_file = os.path.join(OUTPUT_DIR, f"{output_prefix}_NTV_reconstructions.mat")
    savemat(ntv_file, {
        'originals': ntv_orig_arr,
        'reconstructions': ntv_recon_arr,
        'filenames': np.array(filenames, dtype=object)
    })
    print(f"Saved NTV: {ntv_file}")

    # Save combined
    combined_file = os.path.join(OUTPUT_DIR, f"{output_prefix}_SNR_NTV_combined.mat")
    savemat(combined_file, {
        'SNR_originals': snr_orig_arr,
        'SNR_reconstructions': snr_recon_arr,
        'NTV_originals': ntv_orig_arr,
        'NTV_reconstructions': ntv_recon_arr,
        'filenames': np.array(filenames, dtype=object)
    })
    print(f"Saved combined: {combined_file}")

    print(f"\nResults:")
    print(f"  Samples: {len(filenames)}")
    print(f"  SNR shape: {snr_orig_arr.shape}")
    print(f"  NTV shape: {ntv_orig_arr.shape}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract separate SNR and NTV reconstructions"
    )
    parser.add_argument("--input-files", nargs='+', help="Input .mat files")
    parser.add_argument("--input-dir", help="Input directory with .mat files")
    parser.add_argument("--output-prefix", default="extracted", help="Output filename prefix")

    args = parser.parse_args()

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Device: {device} (Apple Metal)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Device: {device}")

    # Load model
    model = load_model(device)

    # Get files to process
    files_to_process = []

    if args.input_files:
        files_to_process = args.input_files
    elif args.input_dir:
        import glob
        files_to_process = glob.glob(os.path.join(args.input_dir, "*.mat"))
    else:
        print("ERROR: Specify --input-files or --input-dir")
        sys.exit(1)

    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    extract_snr_ntv_from_files(files_to_process, args.output_prefix, model, device)
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Outputs saved to: {OUTPUT_DIR}")
