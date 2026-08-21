#!/usr/bin/env python3
"""
Extract SEPARATE NTV and SNR reconstructions from the trained 2-channel autoencoder.

The autoencoder was trained on 2-channel input [SNR, NTV]. This script:
1. Loads the reconstruction filenames from the saved metadata
2. Loads the original SNR and NTV data for those files
3. Generates separate reconstructions by passing each through the model
4. Saves both original and reconstructed data for SNR and NTV separately
5. Creates visualization panels matching the image_results format
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from scipy.io import loadmat, savemat
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import math
from typing import List, Optional, Tuple

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = "/Users/oboulais/Public/Bowhead_DL_Project/LD32/Autoencoder_v100E_32LD_32C_Auto_SNR+NTV_100K_Date20260213-150900.dir"
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model", "autoencoder_clean.pth")
LATENT_EMBED_PATH = os.path.join(MODEL_DIR, "MATLAB", "latent_embeddings.mat")

# Find source data directories
SEARCH_DIRS = [
    "/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir",
    "/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir",
    "/Users/oboulais/Desktop/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir",
    "/Users/oboulais/Desktop/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir",
    "/Users/oboulais/Desktop",
    "/Users/oboulais/Public",
]

OUTPUT_DIR = os.path.join(MODEL_DIR, "reconstructions_separate")
IMAGE_RESULTS_DIR = os.path.join(OUTPUT_DIR, "image_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_RESULTS_DIR, exist_ok=True)

# Visualization parameters
PANEL_GROUP_SIZE = 3  # Number of samples per panel
SHOW_ERROR_PLOTS = True  # Show error difference maps
MAX_PANELS = 50  # Maximum number of panels to generate (3 samples each = 150 images max)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ImprovedAutoencoder(nn.Module):
    """Same architecture as training script"""

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
    """Min-max normalize image to [0, 1] range."""
    im = im.astype(np.float32)
    im_min = float(np.min(im))
    im_max = float(np.max(im))
    rng = im_max - im_min
    if rng < 1e-8:
        return np.zeros_like(im, dtype=np.float32)
    return (im - im_min) / rng


def save_reconstruction_panels(originals: np.ndarray, reconstructions: np.ndarray,
                               filenames: List[str], output_dir: str,
                               base_name: str = "recon_panel",
                               data_type: str = "SNR",
                               show_error: bool = True) -> int:
    """
    Save JPEG panels showing separate reconstructions (SNR or NTV).
    
    Args:
        originals: Original spectrograms [N, H, W]
        reconstructions: Reconstructed spectrograms [N, H, W]
        filenames: List of filenames for titles
        output_dir: Directory to save panels
        base_name: Base name for panel files
        data_type: "SNR" or "NTV" for labeling
        show_error: Whether to show error difference maps
        
    Returns:
        Number of panels written
    """
    os.makedirs(output_dir, exist_ok=True)
    num_samples = originals.shape[0]
    group_count = math.ceil(num_samples / PANEL_GROUP_SIZE)
    panels_written = 0
    n_rows = 3 if show_error else 2
    
    # Spectrogram parameters for axis labels
    nrow, ncol = originals.shape[1], originals.shape[2]  # 121 rows (freq bins), 104 cols (time bins)
    freq_max_hz = 500.0  # Typical max frequency for whale calls
    time_duration_sec = 3.0  # Duration in seconds
    
    for group_idx in range(group_count):
        start = group_idx * PANEL_GROUP_SIZE
        end = min(start + PANEL_GROUP_SIZE, num_samples)
        
        orig_group = originals[start:end]
        recon_group = reconstructions[start:end]
        
        cols = len(orig_group)
        fig, axes = plt.subplots(n_rows, cols, figsize=(3.5 * cols, 3.5 * n_rows))
        if cols == 1:
            axes = np.expand_dims(axes, axis=1)
        
        for col in range(cols):
            sample_idx = start + col
            
            # Set filename as title
            if sample_idx < len(filenames) and filenames[sample_idx]:
                title = filenames[sample_idx].replace('.mat', '')
                if len(title) > 30:
                    title = title[:27] + '...'
            else:
                title = f'{data_type} {sample_idx + 1}'
            
            # Original spectrogram
            im0 = axes[0, col].imshow(orig_group[col], cmap='viridis', origin='lower', aspect='auto',
                                      extent=[0, time_duration_sec, 0, freq_max_hz])
            axes[0, col].set_title(title, fontsize=7)
            axes[0, col].set_ylabel('Frequency (Hz)', fontsize=6)
            if col == 0:
                axes[0, col].tick_params(axis='both', labelsize=5)
            else:
                axes[0, col].set_yticks([])
            axes[0, col].set_xticks([])
            
            # Reconstruction
            im1 = axes[1, col].imshow(recon_group[col], cmap='viridis', origin='lower', aspect='auto',
                                      extent=[0, time_duration_sec, 0, freq_max_hz])
            axes[1, col].set_ylabel('Frequency (Hz)', fontsize=6)
            if not show_error:
                axes[1, col].set_xlabel('Time (s)', fontsize=6)
            if col == 0:
                axes[1, col].tick_params(axis='both', labelsize=5)
            else:
                axes[1, col].set_yticks([])
            if not show_error:
                axes[1, col].tick_params(axis='x', labelsize=5)
            else:
                axes[1, col].set_xticks([])
            
            if show_error:
                diff = np.abs(orig_group[col] - recon_group[col])
                im2 = axes[2, col].imshow(diff, cmap='hot', origin='lower', aspect='auto',
                                         extent=[0, time_duration_sec, 0, freq_max_hz])
                axes[2, col].set_ylabel('Frequency (Hz)', fontsize=6)
                axes[2, col].set_xlabel('Time (s)', fontsize=6)
                if col == 0:
                    axes[2, col].tick_params(axis='both', labelsize=5)
                else:
                    axes[2, col].set_yticks([])
                axes[2, col].tick_params(axis='x', labelsize=5)
        
        # Add row labels on the left
        fig.text(0.02, 0.75 if not show_error else 0.83, f'{data_type} Input', rotation=90, 
                va='center', ha='center', fontsize=8, weight='bold')
        fig.text(0.02, 0.5 if not show_error else 0.5, f'{data_type} Reconstruction', rotation=90,
                va='center', ha='center', fontsize=8, weight='bold')
        if show_error:
            fig.text(0.02, 0.17, 'Error', rotation=90, va='center', ha='center', 
                    fontsize=8, weight='bold')
        
        # Add title at top
        title_str = f'{data_type}-gram Reconstructions (2-Channel Autoencoder, LD=32)'
        fig.suptitle(title_str, fontsize=10, y=0.98)
        
        panel_path = os.path.join(output_dir, f"{base_name}_{group_idx + 1:03d}.jpg")
        plt.tight_layout(rect=[0.03, 0.02, 1, 0.96])
        plt.savefig(panel_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        panels_written += 1
    
    return panels_written


def find_source_data_files(filenames: list) -> dict:
    """Find source .mat files for given filenames."""
    print(f"\nSearching for {len(filenames)} source data files...")

    found_files = {}
    basename_to_full = {}

    # Search for all .mat files in the known dataset directories
    for search_dir in SEARCH_DIRS:
        if not os.path.exists(search_dir):
            continue
        print(f"  Searching in: {search_dir}")

        # Direct file search (faster than recursive glob)
        if os.path.isdir(search_dir):
            try:
                for filename in os.listdir(search_dir):
                    if filename.endswith('.mat'):
                        full_path = os.path.join(search_dir, filename)
                        if os.path.isfile(full_path):
                            basename_to_full[filename] = full_path
            except Exception as e:
                print(f"    Warning: Could not list directory: {e}")

    # Match filenames
    for fn in filenames:
        if fn in basename_to_full:
            found_files[fn] = basename_to_full[fn]

    print(f"  Found {len(found_files)} / {len(filenames)} files ({100*len(found_files)/len(filenames):.1f}%)")
    
    if len(found_files) < len(filenames) * 0.5:
        print(f"\n  WARNING: Less than 50% of files found!")
        print(f"  Missing {len(filenames) - len(found_files)} files")
        print(f"  Check that source data directories are accessible")
    
    return found_files


def extract_snr_ntv_reconstructions():
    """Main extraction function"""

    print(f"\n{'='*70}")
    print(f"Extracting Separate SNR and NTV Reconstructions")
    print(f"{'='*70}")

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Device: {device} (Apple Metal)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: {device} (CUDA)")
    else:
        device = torch.device('cpu')
        print(f"Device: {device} (CPU)")

    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model = ImprovedAutoencoder(nrow=121, ncol=104, latent_dim=32,
                                base_channels=32, extra_conv=False, in_channels=2)
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded successfully")

    # Load filenames from latent embeddings metadata
    print(f"\nLoading file metadata from: {LATENT_EMBED_PATH}")
    try:
        embedding_data = loadmat(LATENT_EMBED_PATH)
        filenames_arr = embedding_data.get('original_filenames', None)

        if filenames_arr is not None:
            # Convert from numpy array of objects
            # Handle shape (1, N) or (N, 1) or (N,)
            filenames = [str(fn[0]) if isinstance(fn, np.ndarray) else str(fn)
                        for fn in filenames_arr.flat]
            print(f"Found {len(filenames)} filenames in metadata")
            print(f"  Array shape: {filenames_arr.shape}")
        else:
            print("No 'original_filenames' field in latent_embeddings.mat")
            return
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    # Find source data files
    source_files = find_source_data_files(filenames)

    if not source_files:
        print("ERROR: Could not find source data files!")
        print("The script needs access to the original .mat files with SNR_gram and NTV_gram data.")
        return

    # Load SNR and NTV data
    print(f"\nLoading original SNR and NTV data...")
    print(f"  Found {len(source_files)} files to process")
    snr_originals = []
    ntv_originals = []
    processed_files = []
    skipped_count = 0

    report_interval = max(1, len(filenames) // 20)  # Report every 5%
    
    for i, fn in enumerate(filenames):
        if i % report_interval == 0 and i > 0:
            print(f"  Processed {i}/{len(filenames)} ({100*i/len(filenames):.1f}%) - Loaded: {len(snr_originals)}, Skipped: {skipped_count}")

        if fn not in source_files:
            skipped_count += 1
            continue

        try:
            data = loadmat(source_files[fn])
            snr_gram = data.get('SNR_gram', None)
            ntv_gram = data.get('NTV_gram', None)

            if snr_gram is not None and ntv_gram is not None:
                snr_norm = _minmax_norm(snr_gram)
                ntv_norm = _minmax_norm(ntv_gram)

                snr_originals.append(snr_norm)
                ntv_originals.append(ntv_norm)
                processed_files.append(fn)
        except Exception as e:
            skipped_count += 1

    print(f"  Final: Loaded {len(snr_originals)} SNR/NTV pairs, Skipped {skipped_count}")

    if not snr_originals:
        print("\nERROR: Could not load any SNR/NTV data!")
        print("Check that:")
        print("  1. Source .mat files are accessible")
        print("  2. Files contain 'SNR_gram' and 'NTV_gram' fields")
        return

    # Generate reconstructions
    print(f"\nGenerating reconstructions...")
    print(f"  Processing in batches of 32...")
    snr_reconstructions = []
    ntv_reconstructions = []

    total_batches = math.ceil(len(snr_originals) / 32)
    
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(snr_originals), 32)):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{total_batches} ({100*(batch_idx+1)/total_batches:.1f}%)")

            batch_end = min(i + 32, len(snr_originals))
            batch_size = batch_end - i

            # Create 2-channel input for each gram type
            snr_batch = np.stack(snr_originals[i:batch_end], axis=0)  # [B, H, W]
            ntv_batch = np.stack(ntv_originals[i:batch_end], axis=0)  # [B, H, W]

            # Stack as 2-channel for model
            snr_2ch = np.stack([snr_batch, snr_batch], axis=1)  # [B, 2, H, W]
            ntv_2ch = np.stack([ntv_batch, ntv_batch], axis=1)  # [B, 2, H, W]

            # Convert to torch tensors
            snr_2ch_torch = torch.from_numpy(snr_2ch).float().to(device)
            ntv_2ch_torch = torch.from_numpy(ntv_2ch).float().to(device)

            # Generate reconstructions
            snr_recon, _ = model(snr_2ch_torch)
            ntv_recon, _ = model(ntv_2ch_torch)

            # Extract first channel (output is [B, 1, H, W])
            snr_recon_np = snr_recon[:, 0, :, :].cpu().numpy()
            ntv_recon_np = ntv_recon[:, 0, :, :].cpu().numpy()

            snr_reconstructions.append(snr_recon_np)
            ntv_reconstructions.append(ntv_recon_np)
    
    print(f"  Reconstruction complete!")

    # Concatenate all batches
    snr_recon_all = np.vstack(snr_reconstructions)
    ntv_recon_all = np.vstack(ntv_reconstructions)
    snr_orig_all = np.stack(snr_originals, axis=0)
    ntv_orig_all = np.stack(ntv_originals, axis=0)

    print(f"\nFinal shapes:")
    print(f"  SNR originals: {snr_orig_all.shape}")
    print(f"  SNR reconstructions: {snr_recon_all.shape}")
    print(f"  NTV originals: {ntv_orig_all.shape}")
    print(f"  NTV reconstructions: {ntv_recon_all.shape}")

    # Save SNR data
    snr_output_file = os.path.join(OUTPUT_DIR, "SNR_gram_reconstructions.mat")
    snr_data = {
        'originals': snr_orig_all,
        'reconstructions': snr_recon_all,
        'filenames': np.array(processed_files, dtype=object)
    }
    savemat(snr_output_file, snr_data)
    print(f"\nSaved SNR data to: {snr_output_file}")

    # Save NTV data
    ntv_output_file = os.path.join(OUTPUT_DIR, "NTV_gram_reconstructions.mat")
    ntv_data = {
        'originals': ntv_orig_all,
        'reconstructions': ntv_recon_all,
        'filenames': np.array(processed_files, dtype=object)
    }
    savemat(ntv_output_file, ntv_data)
    print(f"Saved NTV data to: {ntv_output_file}")

    # Also save combined data for reference
    combined_output_file = os.path.join(OUTPUT_DIR, "SNR_NTV_reconstructions_combined.mat")
    combined_data = {
        'SNR_originals': snr_orig_all,
        'SNR_reconstructions': snr_recon_all,
        'NTV_originals': ntv_orig_all,
        'NTV_reconstructions': ntv_recon_all,
        'filenames': np.array(processed_files, dtype=object)
    }
    savemat(combined_output_file, combined_data)
    print(f"Saved combined data to: {combined_output_file}")

    # Generate visualization panels
    print(f"\n{'='*70}")
    print(f"Generating Visualization Panels")
    print(f"{'='*70}")

    # Select subset for visualization
    # Generate panels up to MAX_PANELS (each panel has PANEL_GROUP_SIZE samples)
    max_viz_samples = MAX_PANELS * PANEL_GROUP_SIZE
    num_viz_samples = min(max_viz_samples, len(processed_files))
    viz_indices = list(range(num_viz_samples))
    print(f"\nVisualizing {num_viz_samples} of {len(processed_files)} samples")
    print(f"  This will create ~{math.ceil(num_viz_samples / PANEL_GROUP_SIZE)} panels per data type")

    # SNR panels
    print(f"\nGenerating SNR-gram panels...")
    snr_panels = save_reconstruction_panels(
        snr_orig_all[viz_indices],
        snr_recon_all[viz_indices],
        [processed_files[i] for i in viz_indices],
        IMAGE_RESULTS_DIR,
        base_name="SNR_recon_panel",
        data_type="SNR",
        show_error=SHOW_ERROR_PLOTS
    )
    print(f"  Written {snr_panels} SNR panel(s)")

    # NTV panels
    print(f"\nGenerating NTV-gram panels...")
    ntv_panels = save_reconstruction_panels(
        ntv_orig_all[viz_indices],
        ntv_recon_all[viz_indices],
        [processed_files[i] for i in viz_indices],
        IMAGE_RESULTS_DIR,
        base_name="NTV_recon_panel",
        data_type="NTV",
        show_error=SHOW_ERROR_PLOTS
    )
    print(f"  Written {ntv_panels} NTV panel(s)")

    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Total samples processed: {len(processed_files)}")
    print(f"  Visualized samples: {num_viz_samples}")
    print(f"  SNR panels: {snr_panels}")
    print(f"  NTV panels: {ntv_panels}")
    print(f"  Images saved to: {IMAGE_RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    extract_snr_ntv_reconstructions()
    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
