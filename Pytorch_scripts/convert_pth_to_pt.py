#!/usr/bin/env python3
"""
Convert .pth autoencoder files to .pt format with descriptive naming.

Usage:
    python3 convert_pth_to_pt.py /path/to/folder/trained_model
"""

import torch
import sys
import os

def convert_pth_to_pt(trained_model_folder):
    """
    Convert autoencoder .pth files to .pt format.
    
    Args:
        trained_model_folder: Path to the trained_model directory
    """
    # Check if folder exists
    if not os.path.exists(trained_model_folder):
        print(f"ERROR: Folder not found: {trained_model_folder}")
        sys.exit(1)
    
    # Look for the main model files
    pth_files = [
        'autoencoder_clean.pth',
        'autoencoder_120x104.pth',
        'checkpoint_epoch100.pth'
    ]
    
    source_file = None
    for pth_file in pth_files:
        pth_path = os.path.join(trained_model_folder, pth_file)
        if os.path.exists(pth_path):
            source_file = pth_path
            print(f"Found source file: {pth_file}")
            break
    
    if source_file is None:
        print(f"ERROR: No suitable .pth file found in {trained_model_folder}")
        print(f"Looked for: {', '.join(pth_files)}")
        sys.exit(1)
    
    # Load the model state dict
    print(f"Loading model from: {source_file}")
    
    try:
        if 'checkpoint' in source_file:
            # Checkpoint files contain more than just state_dict
            checkpoint = torch.load(source_file, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"  Loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            else:
                state_dict = checkpoint
        else:
            # Regular .pth files are just state_dict
            state_dict = torch.load(source_file, map_location='cpu')
        
        print(f"  Model has {len(state_dict)} parameter tensors")
        
        # Try to infer dimensions from the model architecture
        # Look for the first conv layer output or final layer input
        dims = "120x104"  # Default based on folder name
        
        # Check if we can find dimension info in filenames
        if 'autoencoder_120x104.pth' in os.listdir(trained_model_folder):
            dims = "120x104"
        
        # Create output filename
        output_filename = f"autoencoder_clean_{dims}_clean.pt"
        output_path = os.path.join(trained_model_folder, output_filename)
        
        # Save as .pt
        print(f"Saving to: {output_filename}")
        torch.save(state_dict, output_path)
        
        # Verify 
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ Successfully created: {output_path}")
        print(f"  File size: {file_size_mb:.1f} MB")
        
    except Exception as e:
        print(f"ERROR during conversion: {e}")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 convert_pth_to_pt.py /path/to/trained_model")
        print("\nExample:")
        print("  python3 convert_pth_to_pt.py /Users/oboulais/Public/Bowhead_DL_Project/LD32/Autoencoder_*/trained_model")
        sys.exit(1)
    
    folder = sys.argv[1]
    convert_pth_to_pt(folder)
