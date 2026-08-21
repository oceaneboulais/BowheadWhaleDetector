#!/usr/bin/env python3
"""
Summary of available reconstruction data and guide for extracting NTV/SNR reconstructions.

The autoencoder was trained on 2-channel input [SNR_gram, NTV_gram] combined.
The reconstruction_data.mat file contains averaged reconstructions.

This script shows what's available and provides guidance for separate extraction.
"""
import os
import scipy.io
import numpy as np

MODEL_DIR = "/Users/oboulais/Public/Bowhead_DL_Project/LD32/Autoencoder_v100E_32LD_32C_Auto_SNR+NTV_100K_Date20260213-150900.dir"
RECON_DATA_PATH = os.path.join(MODEL_DIR, "MATLAB", "reconstruction_data.mat")
LATENT_PATH = os.path.join(MODEL_DIR, "MATLAB", "latent_embeddings.mat")

print("="*70)
print("RECONSTRUCTION DATA AVAILABILITY ANALYSIS")
print("="*70)

# Load and analyze reconstruction data
print("\n1. RECONSTRUCTION_DATA.MAT")
print("-" * 70)
try:
    recon_data = scipy.io.loadmat(RECON_DATA_PATH)

    for key in ['originals', 'reconstructions', 'filenames']:
        if key in recon_data:
            val = recon_data[key]
            if isinstance(val, np.ndarray):
                if key == 'filenames':
                    print(f"  {key:20s}: shape={val.shape}, dtype={val.dtype}")
                    print(f"                     Sample filenames:")
                    for i, fn in enumerate(val.flat[:3]):
                        print(f"                       [{i}] {fn}")
                else:
                    print(f"  {key:20s}: shape={val.shape}, dtype={val.dtype}")
                    print(f"                     Range: [{val.min():.4f}, {val.max():.4f}]")
                    print(f"                     Mean: {val.mean():.4f}, Std: {val.std():.4f}")

    print(f"\n  DATA CHARACTERISTICS:")
    print(f"  - This contains AVERAGED SNR+NTV reconstructions")
    print(f"  - 30 visualization samples: shape (30, 121, 104)")
    print(f"  - Values in range [0, 1] (normalized)")

except Exception as e:
    print(f"  Error: {e}")

# Load and analyze latent embeddings
print("\n2. LATENT_EMBEDDINGS.MAT")
print("-" * 70)
try:
    latent_data = scipy.io.loadmat(LATENT_PATH)

    key_info = {
        'latent_embeddings': 'Full latent representations',
        'tsne_embeddings': '2D t-SNE projection',
        'clusters': 'Cluster assignments',
        'original_filenames': 'Source filenames',
        'reconstruction_filenames': 'Expected reconstruction filenames'
    }

    for key, desc in key_info.items():
        if key in latent_data:
            val = latent_data[key]
            if isinstance(val, np.ndarray):
                print(f"  {key:25s}: shape={val.shape}, dtype={val.dtype}")
                print(f"                           {desc}")

    orig_files = latent_data['original_filenames']
    print(f"\n  TOTAL EMBEDDINGS: {orig_files.size} samples")
    print(f"  Sample files:")
    for i, fn in enumerate(orig_files.flat[:3]):
        print(f"    [{i}] {fn}")

except Exception as e:
    print(f"  Error: {e}")

# Information about separate extraction
print("\n3. SEPARATE SNR/NTV EXTRACTION")
print("-" * 70)
print("""
To extract SEPARATE SNR and NTV reconstructions:

Option A: Using original source data (if available):
  1. Load SNR_gram from source .mat files
  2. Load NTV_gram from source .mat files
  3. Pass each as duplicated 2-channel input [grad, grad] to the model
  4. Extract reconstructions separately

  Required: Access to original Bowhead_DL_Project/BCB_Whale_Datasets directory

Option B: From combined reconstructions:
  The current reconstruction_data.mat shows AVERAGED results from both channels.
  To get separate SNR and NTV reconstructions, you need to re-run the model
  inference on the individual source spectrograms.

Instructions for Option A:
  1. Ensure source data directories are accessible
  2. Use the provided extraction script:
     python3 extract_ntv_snr_separate_reconstructions.py
  3. Script will:
     - Load 50,000 samples from training directories
     - Generate SNR reconstructions → SNR_gram_reconstructions.mat
     - Generate NTV reconstructions → NTV_gram_reconstructions.mat
     - Generate combined file → SNR_NTV_reconstructions_combined.mat
""")

# Current outputs
print("\n4. AVAILABLE OUTPUT FILES")
print("-" * 70)
output_dir = os.path.join(MODEL_DIR, "MATLAB")
if os.path.exists(output_dir):
    files = os.listdir(output_dir)
    for f in sorted(files):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / (1024*1024)
            print(f"  {f}")
            print(f"    Size: {size_mb:.2f} MB")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("""
1. Check if source Whale_Datasets directories are accessible
   find ~ -name "*Whale*" -o -name "*BCB*" 2>/dev/null

2. If accessible, run the extraction script:
   source ~/.venv_py31018/bin/activate
   python extract_ntv_snr_separate_reconstructions.py

3. The output will be saved to:
   ./LD32/Autoencoder_v100E.../reconstructions_separate/

4. Files generated:
   - SNR_gram_reconstructions.mat (originals + reconstructions)
   - NTV_gram_reconstructions.mat (originals + reconstructions)
   - SNR_NTV_reconstructions_combined.mat (all together)
""")
