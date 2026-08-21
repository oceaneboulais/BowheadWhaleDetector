#!/usr/bin/env python3
"""
Extract sample spectrograms for autoencoder architecture diagram.
Creates input/output examples showing SNR, NTV, and combined spectrograms.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
import os

# Model directory
MODEL_DIR = "/Users/oboulais/Public/Bowhead_DL_Project/LD32/Autoencoder_v100E_32LD_32C_Auto_SNR+NTV_100K_Date20260213-150900.dir"
OUTPUT_DIR = "/Users/oboulais/Public/Bowhead_DL_Project/PlotNeuralNet/pyexamples/sample_images"

# Load a sample from the data directory
DATA_DIR = "/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir"

def minmax_norm(im):
    """Normalize to [0, 1]"""
    im = im.astype(np.float32)
    im_min = float(np.min(im))
    im_max = float(np.max(im))
    rng = im_max - im_min
    if rng < 1e-8:
        return np.zeros_like(im, dtype=np.float32)
    return (im - im_min) / rng

def create_sample_images():
    """Create sample images for the architecture diagram"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find a good sample file
    import glob
    mat_files = glob.glob(os.path.join(DATA_DIR, '**', '*.mat'), recursive=True)
    
    if not mat_files:
        print(f"No .mat files found in {DATA_DIR}")
        return
    
    # Load first file with both SNR_gram and NTV_gram
    sample_file = None
    snr_gram = None
    ntv_gram = None
    
    for fp in mat_files[:100]:  # Check first 100 files
        try:
            m = loadmat(fp)
            if 'SNR_gram' in m and 'NTV_gram' in m:
                snr_gram = minmax_norm(m['SNR_gram'])
                ntv_gram = minmax_norm(m['NTV_gram'])
                sample_file = os.path.basename(fp)
                print(f"Using sample: {sample_file}")
                break
        except:
            continue
    
    if snr_gram is None or ntv_gram is None:
        print("Could not find valid sample with both SNR_gram and NTV_gram")
        return
    
    # Spectrogram parameters
    freq_max_hz = 500.0
    time_duration_sec = 3.0
    nrow, ncol = snr_gram.shape
    
    # Create individual images
    fig_size = (4, 3)
    dpi = 150
    
    # 1. SNR gram input
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    im = ax.imshow(snr_gram, cmap='viridis', origin='lower', aspect='auto',
                   extent=[0, time_duration_sec, 0, freq_max_hz])
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    ax.set_title('SNR Spectrogram (Input)', fontsize=11, weight='bold')
    plt.colorbar(im, ax=ax, label='Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'input_snr.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    print("✓ Created input_snr.png")
    
    # 2. NTV gram input
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    im = ax.imshow(ntv_gram, cmap='viridis', origin='lower', aspect='auto',
                   extent=[0, time_duration_sec, 0, freq_max_hz])
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    ax.set_title('NTV Spectrogram (Input)', fontsize=11, weight='bold')
    plt.colorbar(im, ax=ax, label='Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'input_ntv.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    print("✓ Created input_ntv.png")
    
    # 3. Combined/averaged spectrogram
    combined = (snr_gram + ntv_gram) / 2.0
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    im = ax.imshow(combined, cmap='viridis', origin='lower', aspect='auto',
                   extent=[0, time_duration_sec, 0, freq_max_hz])
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Frequency (Hz)', fontsize=10)
    ax.set_title('Combined (SNR+NTV)', fontsize=11, weight='bold')
    plt.colorbar(im, ax=ax, label='Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'input_combined.png'), dpi=dpi, bbox_inches='tight')
    plt.close()
    print("✓ Created input_combined.png")
    
    # Load model and generate reconstruction if available
    try:
        sys.path.append('/Users/oboulais/Public/Bowhead_DL_Project/python_scripts')
        from Autoencoder_v02_MultiGram_20260211 import ImprovedAutoencoder
        
        model_path = os.path.join(MODEL_DIR, 'trained_model', 'autoencoder_clean.pth')
        if os.path.exists(model_path):
            # Initialize model
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            model = ImprovedAutoencoder(nrow=121, ncol=104, latent_dim=32, base_channels=32, in_channels=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            
            # Create 2-channel input
            input_tensor = torch.from_numpy(np.stack([snr_gram, ntv_gram], axis=0)).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                output, latent = model(input_tensor)
            
            # Extract reconstructions
            output_np = output.cpu().numpy()[0]
            snr_recon = output_np[0]
            ntv_recon = output_np[1]
            combined_recon = (snr_recon + ntv_recon) / 2.0
            latent_np = latent.cpu().numpy()[0]
            
            # 4. SNR reconstruction
            fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
            im = ax.imshow(snr_recon, cmap='viridis', origin='lower', aspect='auto',
                           extent=[0, time_duration_sec, 0, freq_max_hz])
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Frequency (Hz)', fontsize=10)
            ax.set_title('SNR Reconstruction', fontsize=11, weight='bold')
            plt.colorbar(im, ax=ax, label='Amplitude')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'output_snr.png'), dpi=dpi, bbox_inches='tight')
            plt.close()
            print("✓ Created output_snr.png")
            
            # 5. NTV reconstruction
            fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
            im = ax.imshow(ntv_recon, cmap='viridis', origin='lower', aspect='auto',
                           extent=[0, time_duration_sec, 0, freq_max_hz])
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Frequency (Hz)', fontsize=10)
            ax.set_title('NTV Reconstruction', fontsize=11, weight='bold')
            plt.colorbar(im, ax=ax, label='Amplitude')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'output_ntv.png'), dpi=dpi, bbox_inches='tight')
            plt.close()
            print("✓ Created output_ntv.png")
            
            # 6. Combined reconstruction
            fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
            im = ax.imshow(combined_recon, cmap='viridis', origin='lower', aspect='auto',
                           extent=[0, time_duration_sec, 0, freq_max_hz])
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Frequency (Hz)', fontsize=10)
            ax.set_title('Combined Reconstruction', fontsize=11, weight='bold')
            plt.colorbar(im, ax=ax, label='Amplitude')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'output_combined.png'), dpi=dpi, bbox_inches='tight')
            plt.close()
            print("✓ Created output_combined.png")
            
            # 7. Latent representation visualization
            fig, ax = plt.subplots(figsize=(6, 2), dpi=dpi)
            ax.bar(range(len(latent_np)), latent_np, color='steelblue', width=0.8)
            ax.set_xlabel('Latent Dimension', fontsize=10)
            ax.set_ylabel('Activation', fontsize=10)
            ax.set_title('Latent Space (32-dim)', fontsize=11, weight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'latent_space.png'), dpi=dpi, bbox_inches='tight')
            plt.close()
            print("✓ Created latent_space.png")
            
    except Exception as e:
        print(f"Could not generate reconstructions: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✓ All sample images saved to: {OUTPUT_DIR}")
    print(f"  - input_snr.png")
    print(f"  - input_ntv.png")
    print(f"  - input_combined.png")
    print(f"  - output_snr.png (if model available)")
    print(f"  - output_ntv.png (if model available)")
    print(f"  - output_combined.png (if model available)")
    print(f"  - latent_space.png (if model available)")

if __name__ == '__main__':
    create_sample_images()
