#!/usr/bin/env python3
"""
Enhance TensorBoard Dashboard with Training Results

Adds comprehensive training results from LD32 model directories to TensorBoard:
- Training loss curves
- Reconstruction comparison images
- t-SNE visualizations
- Optimal k analysis
- Sample spectrograms

USAGE:
    python3 enhance_tensorboard_with_training_results.py --dir <model_directory>
    
EXAMPLE:
    python3 enhance_tensorboard_with_training_results.py --dir LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20260416-180022.dir
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("ERROR: tensorboard not installed. Run: pip install tensorboard")
    sys.exit(1)


def add_training_images(writer, image_dir):
    """Add training result images to TensorBoard."""
    print("\nAdding training result images...")
    
    image_files = {
        'training_loss.png': 'training/loss_curve',
        'optimal_k_analysis.png': 'analysis/optimal_k',
        'tsne_latent.png': 'visualizations/tsne',
        'reconstructions.png': 'reconstructions/overview'
    }
    
    for filename, tag in image_files.items():
        filepath = os.path.join(image_dir, filename)
        if os.path.exists(filepath):
            try:
                img = Image.open(filepath)
                img_array = np.array(img)
                
                # Convert to CHW format for TensorBoard (channels, height, width)
                if len(img_array.shape) == 3:
                    img_array = np.transpose(img_array, (2, 0, 1))
                elif len(img_array.shape) == 2:
                    img_array = np.expand_dims(img_array, 0)
                
                writer.add_image(tag, img_array, global_step=0, dataformats='CHW')
                print(f"  ✓ Added {filename} → {tag}")
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
        else:
            print(f"  ⚠ Not found: {filename}")


def add_reconstruction_panels(writer, image_dir):
    """Add reconstruction panel images to TensorBoard."""
    print("\nAdding reconstruction panels...")
    
    count = 0
    for i in range(1, 20):  # Check up to 20 panels
        filename = f'recon_panel_{i:03d}.jpg'
        filepath = os.path.join(image_dir, filename)
        
        if os.path.exists(filepath):
            try:
                img = Image.open(filepath)
                img_array = np.array(img)
                
                if len(img_array.shape) == 3:
                    img_array = np.transpose(img_array, (2, 0, 1))
                elif len(img_array.shape) == 2:
                    img_array = np.expand_dims(img_array, 0)
                
                writer.add_image(f'reconstructions/panel_{i:03d}', img_array, global_step=0, dataformats='CHW')
                count += 1
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
        else:
            break
    
    print(f"  ✓ Added {count} reconstruction panels")


def add_sample_spectrograms(writer, image_dir):
    """Add sample spectrogram images to TensorBoard."""
    print("\nAdding sample spectrograms...")
    
    # Check SNR and NTV subdirectories
    for subdir in ['SNR', 'NTV', 'spectrogram']:
        subdir_path = os.path.join(image_dir, subdir)
        if os.path.exists(subdir_path):
            files = [f for f in os.listdir(subdir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            count = 0
            for filename in files[:20]:  # Limit to 20 samples
                filepath = os.path.join(subdir_path, filename)
                try:
                    img = Image.open(filepath)
                    img_array = np.array(img)
                    
                    if len(img_array.shape) == 3:
                        img_array = np.transpose(img_array, (2, 0, 1))
                    elif len(img_array.shape) == 2:
                        img_array = np.expand_dims(img_array, 0)
                    
                    tag = f'samples/{subdir}/{os.path.splitext(filename)[0]}'
                    writer.add_image(tag, img_array, global_step=0, dataformats='CHW')
                    count += 1
                except Exception as e:
                    print(f"  ✗ Error loading {filename}: {e}")
            
            if count > 0:
                print(f"  ✓ Added {count} {subdir} samples")


def add_timing_info(writer, directory):
    """Add timing information as text to TensorBoard."""
    timing_file = os.path.join(directory, 'timing_log.txt')
    if os.path.exists(timing_file):
        print("\nAdding timing information...")
        try:
            with open(timing_file, 'r') as f:
                timing_text = f.read()
            writer.add_text('info/timing', f"```\n{timing_text}\n```", global_step=0)
            print("  ✓ Added timing log")
        except Exception as e:
            print(f"  ✗ Error loading timing log: {e}")


def add_readme(writer, directory):
    """Add README as text to TensorBoard."""
    readme_file = os.path.join(directory, 'README.md')
    if os.path.exists(readme_file):
        print("\nAdding README...")
        try:
            with open(readme_file, 'r') as f:
                readme_text = f.read()
            writer.add_text('info/readme', readme_text, global_step=0)
            print("  ✓ Added README")
        except Exception as e:
            print(f"  ✗ Error loading README: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Enhance TensorBoard dashboard with training results'
    )
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='Path to model directory'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.exists(args.dir):
        print(f"ERROR: Directory not found: {args.dir}")
        sys.exit(1)
    
    print("=" * 70)
    print("Enhancing TensorBoard Dashboard with Training Results")
    print("=" * 70)
    print(f"Model directory: {args.dir}")
    print("=" * 70)
    
    # TensorBoard log directory
    log_dir = os.path.join(args.dir, 'tensorboard_logs')
    if not os.path.exists(log_dir):
        print(f"ERROR: TensorBoard logs not found: {log_dir}")
        print("Run generate_tensorboard_dashboard.py first")
        sys.exit(1)
    
    # Initialize TensorBoard writer (append mode)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Add training images
    image_dir = os.path.join(args.dir, 'image_results')
    if os.path.exists(image_dir):
        add_training_images(writer, image_dir)
        add_reconstruction_panels(writer, image_dir)
        add_sample_spectrograms(writer, image_dir)
    else:
        print(f"⚠ Warning: image_results directory not found: {image_dir}")
    
    # Add timing and README
    add_timing_info(writer, args.dir)
    add_readme(writer, args.dir)
    
    # Add metadata
    metadata = f"""
# Enhanced TensorBoard Dashboard

**Directory:** `{args.dir}`  
**Enhanced:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contents:
- Training loss curves
- Reconstruction comparison panels
- Sample spectrograms (SNR & NTV)
- t-SNE latent space visualization
- Optimal k-means analysis
- Latent space embeddings (32D, UMAP, PaCMAP)
- Model timing information
- Training configuration details

## Access:
This TensorBoard is accessible via the public URL provided by ngrok or cloudflared tunnel.
"""
    writer.add_text('info/dashboard', metadata, global_step=0)
    
    # Close writer
    writer.close()
    
    print("\n" + "=" * 70)
    print("✓ TensorBoard dashboard enhanced with training results!")
    print("=" * 70)
    print(f"\nRefresh your TensorBoard browser to see the new content:")
    print(f"  https://lurk-party-squirt.ngrok-free.dev")
    print("\n")


if __name__ == '__main__':
    main()
