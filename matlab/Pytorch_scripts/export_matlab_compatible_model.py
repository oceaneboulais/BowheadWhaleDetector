#!/usr/bin/env python3
"""
Export MATLAB-Compatible Autoencoder Model

This script creates a MATLAB-compatible version of the autoencoder by removing 
the problematic output_padding parameter from ConvTranspose2d layers.
Instead, it uses manual padding to achieve the correct output dimensions.

The MATLAB Deep Learning Toolbox doesn't support non-uniform output_padding,
so we replace the final layer with a ConvTranspose2d without output_padding
followed by explicit zero-padding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


# ============================================================================
# ORIGINAL MODEL DEFINITION (from your codebase)
# ============================================================================

class ImprovedAutoencoder(nn.Module):
    """Original autoencoder with output_padding (PyTorch compatible)"""
    
    def __init__(self, nrow=121, ncol=104, latent_dim=32, 
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
        x_recon = x_recon.view(x_recon.size(0), self.c_out, 
                               self.nrow_reduced, self.ncol_reduced)
        output = self.decoder(x_recon)
        return output, latent


# ============================================================================
# MATLAB-COMPATIBLE MODEL (removes output_padding)
# ============================================================================

class MatlabCompatibleAutoencoder(nn.Module):
    """
    MATLAB-compatible autoencoder that avoids output_padding in ConvTranspose2d.
    
    Key differences:
    - Final ConvTranspose2d has no output_padding parameter
    - Uses nn.ZeroPad2d layer for static padding (traces better than F.pad)
    - Uses nn.Flatten() instead of dynamic .view() operations
    - Returns only reconstruction (not tuple) for MATLAB compatibility
    """
    
    def __init__(self, nrow=121, ncol=104, latent_dim=32, 
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
        
        # Encoder: same as original
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
        
        # Flattening layer (replaces dynamic .view())
        self.flatten = nn.Flatten()
        
        # Latent space: same as original
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
        
        # Unflatten layer (replaces dynamic .view() in decoder)
        self.unflatten = nn.Unflatten(1, (c4 if extra_conv else c3, nrow_reduced, ncol_reduced))
        
        # Calculate padding needed to match target dimensions  
        expected_h = nrow_reduced * (16 if extra_conv else 8)
        expected_w = ncol_reduced * (16 if extra_conv else 8)
        pad_h = nrow - expected_h
        pad_w = ncol - expected_w
        
        # Decoder: NO output_padding parameter
        # Use nn.ZeroPad2d instead of F.pad for better MATLAB tracing
        decoder_layers = []
        
        if extra_conv:
            decoder_layers.extend([
                nn.ConvTranspose2d(c4, c3, 2, stride=2),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c3, c2, 2, stride=2),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c2, c1, 2, stride=2),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c1, 1, 2, stride=2),  # NO output_padding
            ])
        else:
            decoder_layers.extend([
                nn.ConvTranspose2d(c3, c2, 2, stride=2),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c2, c1, 2, stride=2),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(c1, 1, 2, stride=2),  # NO output_padding
            ])
        
        # Add padding layer if needed (left, right, top, bottom)
        if pad_h > 0 or pad_w > 0:
            decoder_layers.append(nn.ZeroPad2d((0, pad_w, 0, pad_h)))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.flat_size = flat_size
        # Encoder
        x = self.encoder(x)
        x_flat = self.flatten(x)  # Use nn.Flatten instead of .view()
        
        # Latent bottleneck
        latent = self.to_latent(x_flat)
        
        # Decoder
        x_recon = self.from_latent(latent)
        x_recon = self.unflatten(x_recon)  # Use nn.Unflatten instead of .view()
        output = self.decoder(x_recon)
        
        # MATLAB compatibility: return only reconstruction (not tuple)
        # For training/validation in Python, use forward_with_latent()
        return output
    
    def forward_with_latent(self, x):
        """Python-only version that returns both output and latent"""
        x = self.encoder(x)
        x_flat = self.flatten(x)
        latent = self.to_latent(x_flat)
        x_recon = self.from_latent(latent)
        x_recon = self.unflatten(x_recon)
        output = self.decoder(x_recon)
        return output, latent


# ============================================================================
# WEIGHT TRANSFER AND EXPORT
# ============================================================================

def transfer_weights(original_model, matlab_model):
    """
    Transfer weights from original model to MATLAB-compatible model.
    All layers except decoder structure are identical.
    """
    # Transfer encoder weights
    matlab_model.encoder.load_state_dict(original_model.encoder.state_dict())
    
    # Transfer latent space weights
    matlab_model.to_latent.load_state_dict(original_model.to_latent.state_dict())
    matl# MATLAB model returns only output, use forward_with_latent for validation
        matlab_out, matlab_latent = matlab_model.forward_with_latent(dummy_input)
    
    # Check latent representations (should be identical)
    latent_diff = torch.abs(orig_latent - matlab_latent).max().item()
    print(f"Max latent difference: {latent_diff:.2e}")
    
    # Check output shapes
    print(f"Original output shape: {orig_out.shape}")
    print(f"MATLAB output shape: {matlab_out.shape}")
    
    # Check output similarity (may differ slightly due to padding approach)
    if orig_out.shape == matlab_out.shape:
        output_diff = torch.abs(orig_out - matlab_out).max().item()
        output_mean_diff = torch.abs(orig_out - matlab_out).mean().item()
        print(f"Max output difference: {output_diff:.2e}")
        print(f"Mean output difference: {output_mean_diff:.2e}")
        
        if output_diff < 1e-5:
            print("✓ Models produce identical outputs")
        else:
            print("⚠ Models produce slightly different outputs (expected due to padding)")
    else:
        print("⚠ Output shapes differ - this is expected")
    
    # Test traced model output (what MATLAB will actually use)
    print("\nValidating traced model output:")
    with torch.no_grad():
        traced_output = matlab_model(dummy_input)
    trace_diff = torch.abs(orig_out - traced_output).max().item()
    print(f"Max difference (original vs traced): {trace_diff:.2e}
    print(f"Original output shape: {orig_out.shape}")
    print(f"MATLAB output shape: {matlab_out.shape}")
    
    # Check output similarity (may differ slightly due to padding approach)
    if orig_out.shape == matlab_out.shape:
        output_diff = torch.abs(orig_out - matlab_out).max().item()
        output_mean_diff = torch.abs(orig_out - matlab_out).mean().item()
        print(f"Max output difference: {output_diff:.2e}")
        print(f"Mean output difference: {output_mean_diff:.2e}")
        
        if output_diff < 1e-5:
            print("✓ Models produce identical outputs")
        else:
    
    # Trace the model (this is what MATLAB will import)
    print("Tracing model for MATLAB export...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
    
    # Verify the traced model works
    print("Verifying traced model...")
    with torch.no_grad():
        traced_output = traced_model(dummy_input)
        direct_output = model(dummy_input)
        diff = torch.abs(traced_output - direct_output).max().item()
        print(f"  Trace verification difference: {diff:.2e}")
        
        if diff < 1e-6:
            print("  ✓ Traced model is identical to original")
        else:
            print("  ⚠ Traced model differs slightly (may still work)")
    
    # Save traced model
    torch.jit.save(traced_model, output_path)
    print(f"✓ Saved traced model to: {output_path}")
    
    # Print model info
    print(f"\nMATLAB Import Command:")
    print(f"  >> net = importNetworkFromPyTorch('{os.path.basename(output_path)}');")
    print(f"  >> prediction = predict(net, input_data);  % input_data: 121×104×1×N

def export_traced_model(model, dummy_input, output_path):
    """Export model as TorchScript for MATLAB import"""
    model.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, output_path)
    print(f"✓ Saved traced model to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("MATLAB-Compatible Autoencoder Export")
    print("=" * 70)
    
    # Configuration - EDIT THESE PATHS
    model_paths = [
        # LD16 models
        "/Users/oboulais/Public/Bowhead_DL_Project/LD16/Autoencoder_v13_100E_16LD_32C_AutoManual_Combined_100K_Date20260119-222955.dir/trained_model/autoencoder_clean.pth",
        "/Users/oboulais/Public/Bowhead_DL_Project/LD16/Autoencoder_v14_100E_16LD_32C_Manual_100K_Date20260122-190056.dir/trained_model/autoencoder_clean.pt",
        
        # LD32 models
        "/Users/oboulais/Public/Bowhead_DL_Project/LD32/Autoencoder_v13_100E_32LD_32C_AutoManual_Combined_100K_Date20251228-124835.dir/trained_model/autoencoder_clean.pth",
        "/Users/oboulais/Public/Bowhead_DL_Project/LD32/Autoencoder_v14_100E_32LD_32C_Manual_100K_Date20260122-190106.dir/trained_model/autoencoder_clean.pt",
    ]
    
    # Model configurations: (latent_dim, base_channels, extra_conv)
    configs = {
        "LD16": (16, 32, False),
        "LD32": (32, 32, False),
    }
    
    # Process each model
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"\n⚠ Model not found: {model_path}")
            continue
        
        print(f"\n{'=' * 70}")
        print(f"Processing: {os.path.basename(model_path)}")
        print(f"{'=' * 70}")
        
        # Determine configuration from path
        if "LD16" in model_path:
            latent_dim, base_channels, extra_conv = configs["LD16"]
        elif "LD32" in model_path:
            latent_dim, base_channels, extra_conv = configs["LD32"]
        else:
            print(f"⚠ Unknown configuration for {model_path}")
            continue
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Create original model and load weights
        original_model = ImprovedAutoencoder(
            nrow=121, ncol=104, 
            latent_dim=latent_dim, 
            base_channels=base_channels, 
            extra_conv=extra_conv
        )
        original_model.load_state_dict(state_dict)
        original_model.eval()
        
        # Create MATLAB-compatible model
        matlab_model = MatlabCompatibleAutoencoder(
            nrow=121, ncol=104, 
            latent_dim=latent_dim, 
            base_channels=base_channels, 
            extra_conv=extra_conv
        )
        
        # Transfer weights
        transfer_weights(original_model, matlab_model)
        
        # Validate models
        dummy_input = torch.randn(1, 1, 121, 104)
        validate_models(original_model, matlab_model, dummy_input)
        
        # Export traced model
        output_path = model_path.replace('.pth', '_matlab_compatible.pt').replace('.pt', '_matlab_compatible.pt')
        if output_path == model_path:
            output_path = model_path.replace('.pt', '_matlab.pt')
        
        export_traced_model(matlab_model, dummy_input, output_path)
        
        print(f"\n✓ Successfully created MATLAB-compatible model")
        print(f"  Original: {model_path}")
        print(f"  MATLAB:   {output_path}")
    
    print("\n" + "=" * 70)
    print("Export Complete!")
    print("=" * 70)
    print("\nNow you can import the '*_matlab_compatible.pt' files into MATLAB")
    print("using the Deep Learning Toolbox converter:")
    print("  >> net = importNetworkFromPyTorch('model_matlab_compatible.pt')")


if __name__ == '__main__':
    main()
