#!/usr/bin/env python3
"""Check input channels from previous MultiGram runs"""
from scipy.io import loadmat
import os

# Check the most recent MultiGram run
base_dir = '/Users/oboulais/Public/Bowhead_DL_Project/LD32'
run_dirs = [
    'Autoencoder_vv13_100E_32LD_32C_AutoManual_Combined_100K_Date20260221-182034.dir',
    'Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir'
]

for run_dir in run_dirs:
    dir_path = os.path.join(base_dir, run_dir)
    model_path = os.path.join(dir_path, 'MATLAB', 'autoencoder_clean.mat')
    
    if os.path.exists(model_path):
        print(f"\n{'='*70}")
        print(f"Checking: {run_dir}")
        print(f"{'='*70}")
        
        model = loadmat(model_path)
        
        # Look for first encoder conv layer to see in_channels
        encoder_keys = [k for k in model.keys() if 'encoder' in k and 'weight' in k]
        
        if encoder_keys:
            # Find the first encoder layer (should be encoder.0.weight or similar)
            first_key = sorted([k for k in encoder_keys if '.0.' in k or '_0_' in k])[0]
            weights = model[first_key]
            
            print(f"\nFirst encoder layer: {first_key}")
            print(f"Weight shape: {weights.shape}")
            
            if len(weights.shape) == 4:
                # Conv2d weight format: [out_channels, in_channels, kernel_h, kernel_w]
                out_ch, in_ch, kh, kw = weights.shape
                print(f"\n  ✓ Output channels (filters): {out_ch}")
                print(f"  ✓ INPUT CHANNELS: {in_ch}")
                print(f"  ✓ Kernel size: {kh}x{kw}")
                
                if in_ch == 1:
                    print(f"\n  ⚠️  This model was trained with SINGLE-CHANNEL input (SNR or NTV only)")
                elif in_ch == 2:
                    print(f"\n  ✅ This model was trained with 2-CHANNEL input (BOTH: SNR + NTV)")
                elif in_ch == 3:
                    print(f"\n  ❌ WARNING! This model was trained with 3-CHANNEL input (UNUSUAL!)")
                else:
                    print(f"\n  ❓ Unexpected number of input channels: {in_ch}")
        else:
            print("  No encoder weights found")
    else:
        print(f"\nNot found: {model_path}")
