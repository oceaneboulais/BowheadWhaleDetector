#!/usr/bin/env python3
"""
Quick example: Extract SNR and NTV reconstructions from the 30 visualization samples.

This demonstrates how separate reconstructions look for the samples that were
visualized in the image_results/ panels.
"""
import torch
import numpy as np
import os
from scipy.io import loadmat, savemat
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = "/Users/oboulais/Public/Bowhead_DL_Project/LD32/Autoencoder_v100E_32LD_32C_Auto_SNR+NTV_100K_Date20260213-150900.dir"
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model", "autoencoder_clean.pth")
RECON_DATA_PATH = os.path.join(MODEL_DIR, "MATLAB", "reconstruction_data.mat")
OUTPUT_DIR = os.path.join(MODEL_DIR, "reconstructions_separate")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ImprovedAutoencoder(torch.nn.Module):
    """Minimal autoencoder for loading weights"""

    def __init__(self, nrow=121, ncol=104, latent_dim=32, base_channels=32, in_channels=2):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, c1, 3, padding=1),
            torch.nn.BatchNorm2d(c1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(c1, c2, 3, padding=1),
            torch.nn.BatchNorm2d(c2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(c2, c3, 3, padding=1),
            torch.nn.BatchNorm2d(c3),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )

        flat_size = c3 * (nrow // 8) * (ncol // 8)
        self.to_latent = torch.nn.Sequential(
            torch.nn.Linear(flat_size, latent_dim * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(latent_dim * 2, latent_dim)
        )

        self.from_latent = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(latent_dim * 2, flat_size),
            torch.nn.ReLU(inplace=True)
        )

        pad_h = (nrow - (nrow // 8) * 8) % 2
        pad_w = (ncol - (ncol // 8) * 8) % 2
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(c3, c2, 2, stride=2),
            torch.nn.BatchNorm2d(c2),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(c2, c1, 2, stride=2),
            torch.nn.BatchNorm2d(c1),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(c1, 1, 2, stride=2, output_padding=(pad_h, pad_w)),
        )

        self.flat_size = flat_size
        self.nrow_reduced = nrow // 8
        self.ncol_reduced = ncol // 8
        self.c_out = c3

    def forward(self, x):
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        latent = self.to_latent(x_flat)
        x_recon = self.from_latent(latent)
        x_recon = x_recon.view(x_recon.size(0), self.c_out, self.nrow_reduced, self.ncol_reduced)
        return self.decoder(x_recon), latent


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*70}")
    print(f"Quick Example: Extract SNR/NTV Reconstructions from 30 Samples")
    print(f"{'='*70}")

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
    print(f"\nLoading model...")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model = ImprovedAutoencoder(nrow=121, ncol=104, latent_dim=32,
                                base_channels=32, in_channels=2)
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load existing averaged reconstructions
    print(f"Loading averaged reconstruction data...")
    recon_data = loadmat(RECON_DATA_PATH)
    originals = recon_data['originals']  # (30, 121, 104)
    filenames = recon_data['filenames']

    print(f"  Loaded {originals.shape[0]} samples")

    # Note: The current reconstruction_data.mat contains AVERAGED SNR+NTV
    # To get separate SNR and NTV, we would need the original source files
    # But we can demonstrate with synthetic data

    print(f"\nNote: Current data is AVERAGED SNR+NTV")
    print(f"To get SEPARATE reconstructions:\n")
    print(f"Option 1: Use extract_reconstructions_utility.py with source files")
    print(f"Option 2: Load SNR and NTV separately from source .mat files\n")

    # Demonstrate with the averaged data
    print(f"Demonstrating reconstruction process...")

    # Treat the averaged data as SNR for demonstration
    snr_demo = originals  # (30, 121, 104)

    # Create 2-channel input (as the model expects)
    snr_2ch = np.zeros((30, 2, 121, 104), dtype=np.float32)
    snr_2ch[:, 0, :, :] = snr_demo  # Channel 1
    snr_2ch[:, 1, :, :] = snr_demo  # Channel 2 (duplicated)

    print(f"Input shape: {snr_2ch.shape}")

    # Generate reconstructions
    with torch.no_grad():
        snr_2ch_t = torch.from_numpy(snr_2ch).to(device)
        recon_out, latent = model(snr_2ch_t)
        recon_np = recon_out[:, 0, :, :].cpu().numpy()

    print(f"Reconstruction shape: {recon_np.shape}")
    print(f"Latent shape: {latent.shape}")

    # Save example
    example_file = os.path.join(OUTPUT_DIR, "example_averaged_demo_reconstruction.mat")
    savemat(example_file, {
        'originals': snr_demo,
        'reconstructions': recon_np,
        'latent_embeddings': latent.cpu().numpy(),
        'filenames': filenames
    })
    print(f"\nSaved example to: {example_file}")

    # Statistics
    print(f"\nReconstruction Statistics:")
    print(f"  Original range: [{originals.min():.4f}, {originals.max():.4f}]")
    print(f"  Original mean: {originals.mean():.4f}, std: {originals.std():.4f}")
    print(f"  Reconstruction range: [{recon_np.min():.4f}, {recon_np.max():.4f}]")
    print(f"  Reconstruction mean: {recon_np.mean():.4f}, std: {recon_np.std():.4f}")
    print(f"  MSE: {((originals - recon_np)**2).mean():.6f}")

    print(f"\n{'='*70}")
    print(f"To extract REAL separate SNR and NTV reconstructions:")
    print(f"{'='*70}")
    print(f"""
Usage:
    python3 extract_reconstructions_utility.py \\
      --input-files file1.mat file2.mat \\
      --output-prefix my_extraction

This will create:
  - my_extraction_SNR_reconstructions.mat
  - my_extraction_NTV_reconstructions.mat
  - my_extraction_SNR_NTV_combined.mat
    """)


if __name__ == "__main__":
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
