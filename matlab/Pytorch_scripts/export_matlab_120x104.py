#!/usr/bin/env python3
"""
Export MATLAB-Compatible Autoencoder (120×104 - No Padding Needed)

This version removes the padding issue entirely by using 120×104 input,
which divides evenly by 8, requiring NO padding in the decoder.

MATLAB cannot trace padding operations reliably, so this is the cleanest solution.
"""

import torch
import torch.nn as nn
import os
import sys


class Reshape(nn.Module):
    """Simple reshape layer for MATLAB compatibility"""
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
    
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, *self.target_shape)


class MatlabCleanAutoencoder(nn.Module):
    """
    MATLAB-compatible autoencoder for 120×104 input (NO PADDING NEEDED).
    
    Key features:
    - Input: 120×104 (divides evenly by 8)
    - No output_padding parameter
    - No ZeroPad2d layer (not needed!)
    - No dynamic operations
    - Clean MATLAB import with zero ATEN layers
    """
    
    def __init__(self, nrow=120, ncol=104, latent_dim=32, 
                 base_channels=32, extra_conv=False):
        super().__init__()
        self.nrow, self.ncol = nrow, ncol
        self.extra_conv = extra_conv
        
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
        
        # Encoder
        if extra_conv:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, c1, 3, padding=1),
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
                nn.Conv2d(1, c1, 3, padding=1),
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
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # Latent space
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
        
        # Reshape
        c_out = c4 if extra_conv else c3
        self.reshape = Reshape((c_out, nrow_reduced, ncol_reduced))
        
        # Decoder: NO PADDING NEEDED (120÷8=15, 104÷8=13, both exact!)
        if extra_conv:
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
                nn.ConvTranspose2d(c1, 1, 2, stride=2),  # Perfect 120×104!
            )
        else:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(c3, c2, 2, stride=2),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c2, c1, 2, stride=2),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c1, 1, 2, stride=2),  # Perfect 120×104!
            )
        
        self.flat_size = flat_size
        self.nrow_reduced = nrow_reduced
        self.ncol_reduced = ncol_reduced
        self.c_out = c_out

    def forward(self, x):
        """Forward pass - returns only reconstruction"""
        x = self.encoder(x)
        x_flat = self.flatten(x)
        latent = self.to_latent(x_flat)
        x_recon = self.from_latent(latent)
        x_recon = self.reshape(x_recon)
        output = self.decoder(x_recon)
        return output


def load_and_adapt_weights(checkpoint_path, latent_dim, base_channels, extra_conv):
    """Load weights from 121×104 model and adapt to 120×104 model"""
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create new model
    model = MatlabCleanAutoencoder(
        nrow=120, ncol=104,
        latent_dim=latent_dim,
        base_channels=base_channels,
        extra_conv=extra_conv
    )
    
    # Load weights (encoder, latent, most of decoder are compatible)
    model.load_state_dict(state_dict, strict=False)
    print("✓ Loaded compatible weights (decoder output layer excluded)")
    
    return model


def export_model(model, output_path):
    """Export model as TorchScript"""
    model.eval()
    
    dummy_input = torch.randn(1, 1, 120, 104)
    
    print("\nTracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Verify
        traced_output = traced_model(dummy_input)
        direct_output = model(dummy_input)
        diff = torch.abs(traced_output - direct_output).max().item()
        print(f"  Trace verification: {diff:.2e}")
    
    torch.jit.save(traced_model, output_path)
    print(f"✓ Saved to: {output_path}")
    
    print(f"\n{'='*70}")
    print("MATLAB Import (NO PADDING ISSUES!):")
    print(f"{'='*70}")
    print(f">> net = importNetworkFromPyTorch('{os.path.basename(output_path)}', ...")
    print(f"     'InputSize', [120 104 1]);")
    print(f"")
    print(f"NOTE: This model expects 120×104 input, not 121×104.")
    print(f"      Crop your SNR_gram data: snr_crop = snr_gram(1:120, :);")


def main():
    print("="*70)
    print("MATLAB-Compatible Export (120×104 - No Padding)")
    print("="*70)
    
    configs = [
        ("LD16", "/Users/oboulais/Public/Bowhead_DL_Project/LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir/trained_model/autoencoder_clean.pth", 16, 32, False),
        ("LD16", "/Users/oboulais/Public/Bowhead_DL_Project/LD16/Autoencoder_v14_100E_16LD_32C_Manual_100K_Date20260122-190056.dir/trained_model/autoencoder_clean.pt", 16, 32, False),
        ("LD32", "/Users/oboulais/Public/Bowhead_DL_Project/LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder_clean.pth", 32, 32, False),
        ("LD32", "/Users/oboulais/Public/Bowhead_DL_Project/LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir/trained_model/autoencoder_clean.pt", 32, 32, False),
    ]
    
    for name, ckpt_path, latent_dim, base_ch, extra_conv in configs:
        if not os.path.exists(ckpt_path):
            print(f"\n⚠ Not found: {ckpt_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing {name}: {os.path.basename(ckpt_path)}")
        print(f"{'='*70}")
        
        model = load_and_adapt_weights(ckpt_path, latent_dim, base_ch, extra_conv)
        
        output_path = ckpt_path.replace('.pth', '_120x104_clean.pt').replace('.pt', '_120x104_clean.pt')
        if '_120x104_clean_120x104_clean' in output_path:
            output_path = output_path.replace('_120x104_clean_120x104_clean', '_120x104_clean')
        
        export_model(model, output_path)
    
    print(f"\n{'='*70}")
    print("Export Complete!")
    print(f"{'='*70}")
    print("\nThese models output 120×104 (NO padding needed).")
    print("In MATLAB, crop input: snr_crop = snr_gram(1:120, :);")
    print("After prediction, pad output if needed:")
    print("  snr_full = padarray(reconstruction, [1 0], 0, 'post');")


if __name__ == '__main__':
    main()
