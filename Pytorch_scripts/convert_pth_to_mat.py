#!/usr/bin/env python3
"""
Convert PyTorch .pth model file to MATLAB .mat format
"""
import torch
from scipy.io import savemat
import os
import sys

# Input .pth file path
pth_path = '/Users/oboulais/Desktop/Bowhead_DL_Project/LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder_clean.pth'

# Find the base directory (ending in .dir) and create MATLAB subdirectory
base_dir = os.path.dirname(os.path.dirname(pth_path))  # Go up from trained_model/
matlab_dir = os.path.join(base_dir, 'MATLAB')
os.makedirs(matlab_dir, exist_ok=True)

# Output .mat file path in MATLAB folder
mat_filename = os.path.basename(pth_path).replace('.pth', '.mat')
mat_path = os.path.join(matlab_dir, mat_filename)

print(f"Loading PyTorch model from: {pth_path}")

# Load the PyTorch state dict
state_dict = torch.load(pth_path, map_location='cpu')

print(f"Converting {len(state_dict)} tensors to numpy arrays...")

# Convert all tensors to numpy arrays
mat_dict = {key: value.cpu().numpy() for key, value in state_dict.items()}

print(f"Saving to MATLAB format: {mat_path}")

# Save as .mat file
savemat(mat_path, mat_dict)

print("Conversion complete!")
print(f"\nOutput file: {mat_path}")
print(f"File size: {os.path.getsize(mat_path) / (1024*1024):.2f} MB")
print(f"\nIn MATLAB, load with: weights = load('{os.path.basename(mat_path)}');")
