#!/usr/bin/env python3
"""Quick script to inspect the structure of whale call .mat files"""
import scipy.io as sio
import numpy as np
import glob
import os

DATA_DIR = "/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir"

# Get a sample file
sample_files = glob.glob(os.path.join(DATA_DIR, "S308A0T*Type1.mat"))[:1]

if sample_files:
    print(f"Examining: {os.path.basename(sample_files[0])}")
    data = sio.loadmat(sample_files[0])
    
    print("\nKeys in .mat file:")
    for key in data.keys():
        if not key.startswith('__'):
            print(f"  {key}: {type(data[key])}, shape: {np.array(data[key]).shape}")
    
    # Try to understand the structure
    for key in data.keys():
        if not key.startswith('__'):
            print(f"\n{key} content preview:")
            print(data[key])

# Count files by type
print("\n" + "="*60)
print("File counts by type:")
for call_type in range(1, 8):
    pattern = os.path.join(DATA_DIR, f"*Type{call_type}.mat")
    count = len(glob.glob(pattern))
    print(f"Type {call_type}: {count:,} files")
