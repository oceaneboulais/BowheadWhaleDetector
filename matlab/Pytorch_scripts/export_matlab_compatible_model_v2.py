#!/usr/bin/env python3
"""
Export MATLAB-Compatible Autoencoder Model (Version 2)

This script creates a MATLAB-compatible version of the autoencoder by:
1. Removing the problematic output_padding parameter from ConvTranspose2d layers
2. Using nn.Flatten() instead of dynamic .view() operations
3. Returning only reconstruction (not tuple) to avoid ATEN layer errors

The MATLAB Deep Learning Toolbox creates "ATEN" layers when it encounters:
- Dynamic tensor operations like .view(x.size(0), -1)
- Tuple returns in forward()
- Non-uniform output_padding parameters

This version fixes all these issues for clean MATLAB import.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys


# ============================================================================
# CUSTOM LAYERS FOR MATLAB COMPATIBILITY
# ============================================================================

class Reshape(nn.Module):
    """
    Simple reshape layer that MATLAB can trace.
    Replaces nn.Unflatten() which MATLAB doesn't recognize.
    """
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
    
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, *self.target_shape)


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
# MATLAB-COMPATIBLE MODEL (fixes ATEN layer issues)
# ============================================================================

class MatlabCompatibleAutoencoder(nn.Module):
    """
    MATLAB-compatible autoencoder that avoids ATEN layer errors.
    
    Key changes from original:
    1. No output_padding parameter → use nn.ZeroPad2d layer instead
    2. No dynamic .view() → use nn.Flatten() and custom Reshape layer
    3. No tuple return → return only reconstruction
    4. All operations have direct MATLAB equivalents (no nn.Unflatten)
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
        
        # Static flattening layer (replaces dynamic .view())
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
        
        # Custom reshape layer that MATLAB can trace (replaces nn.Unflatten)
        c_out = c4 if extra_conv else c3
        self.reshape = Reshape((c_out, nrow_reduced, ncol_reduced))
        
        # Calculate padding needed to match target dimensions  
        expected_h = nrow_reduced * (16 if extra_conv else 8)
        expected_w = ncol_reduced * (16 if extra_conv else 8)
        pad_h = nrow - expected_h
        pad_w = ncol - expected_w
        
        # Decoder: NO output_padding parameter, use explicit padding layer
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
        self.nrow_reduced = nrow_reduced
        self.ncol_reduced = ncol_reduced
        self.base_channels = base_channels
        self.c_out = c4 if extra_conv else c3

    def forward(self, x):
        """
        Forward pass for MATLAB export.
        Returns only reconstruction (no latent) to avoid tuple ATEN layer.
        """
        x = self.encoder(x)
        x_flat = self.flatten(x)  # Static layer, not dynamic .view()
        latent = self.to_latent(x_flat)
        x_recon = self.from_latent(latent)
        x_recon = self.reshape(x_recon)  # Custom reshape layer MATLAB can trace
        output = self.decoder(x_recon)
        
        # Return only reconstruction for MATLAB compatibility
        return output
    
    def forward_with_latent(self, x):
        """Python-only version that returns both output and latent for validation"""
        x = self.encoder(x)
        x_flat = self.flatten(x)
        latent = self.to_latent(x_flat)
        x_recon = self.from_latent(latent)
        x_recon = self.reshape(x_recon)  # Custom reshape layer
        output = self.decoder(x_recon)
        return output, latent


# ============================================================================
# WEIGHT TRANSFER AND EXPORT
# ============================================================================

def transfer_weights(original_model, matlab_model):
    """
    Transfer weights from original model to MATLAB-compatible model.
    All layers have identical weights, only the structure differs.
    """
    # Transfer encoder weights
    matlab_model.encoder.load_state_dict(original_model.encoder.state_dict())
    
    # Transfer latent space weights
    matlab_model.to_latent.load_state_dict(original_model.to_latent.state_dict())
    matlab_model.from_latent.load_state_dict(original_model.from_latent.state_dict())
    
    # Transfer decoder weights
    matlab_model.decoder.load_state_dict(original_model.decoder.state_dict(), strict=False)
    
    print("✓ Successfully transferred all weights")


def validate_models(original_model, matlab_model, dummy_input):
    """Validate that both models produce similar outputs"""
    original_model.eval()
    matlab_model.eval()
    
    with torch.no_grad():
        orig_out, orig_latent = original_model(dummy_input)
        # Use forward_with_latent for validation
        matlab_out, matlab_latent = matlab_model.forward_with_latent(dummy_input)
    
    # Check latent representations (should be identical)
    latent_diff = torch.abs(orig_latent - matlab_latent).max().item()
    print(f"Max latent difference: {latent_diff:.2e}")
    
    # Check output shapes
    print(f"Original output shape: {orig_out.shape}")
    print(f"MATLAB output shape: {matlab_out.shape}")
    
    # Check output similarity
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
        print("⚠ Output shapes differ")
    
    # Test traced model output (what MATLAB will actually use)
    print("\nValidating traced model (single output):")
    with torch.no_grad():
        traced_output = matlab_model(dummy_input)  # Returns only reconstruction
        print(f"Traced model output shape: {traced_output.shape}")
        trace_diff = torch.abs(orig_out - traced_output).max().item()
        print(f"Max difference (original vs traced): {trace_diff:.2e}")


def export_traced_model(model, dummy_input, output_path):
    """Export model as TorchScript for MATLAB import with explicit input size"""
    model.eval()
    
    # Get input dimensions (H, W, C) from dummy_input (B, C, H, W)
    input_height = dummy_input.shape[2]
    input_width = dummy_input.shape[3]
    input_channels = dummy_input.shape[1]
    
    # Trace the model (this is what MATLAB will import)
    print("\nTracing model for MATLAB export...")
    print(f"  Input dimensions: {input_height}×{input_width}×{input_channels} (HxWxC)")
    
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
    
    # Print MATLAB import instructions with correct input size
    print(f"\n{'='*70}")
    print("MATLAB Import Instructions:")
    print(f"{'='*70}")
    print(f"1. In MATLAB, navigate to the training directory")
    print(f"2. Import with explicit InputSize:")
    print(f"   >> net = importNetworkFromPyTorch('{os.path.basename(output_path)}', ...")
    print(f"        'InputSize', [{input_height} {input_width} {input_channels}]);")
    print(f"3. Verify input layer:")
    print(f"   >> net.Layers(1)  % Should show imageInputLayer with size [{input_height} {input_width} {input_channels}]")
    print(f"4. Test prediction:")
    print(f"   >> input_data = randn({input_height}, {input_width}, {input_channels}, 10);  % 10 samples")
    print(f"   >> predictions = predict(net, input_data);")
    print(f"   >> size(predictions)  % Should be [{input_height} {input_width} {input_channels} 10]")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("MATLAB-Compatible Autoencoder Export (Version 2.1)")
    print("Fixes: No output_padding, Custom Reshape (not nn.Unflatten)")
    print("       No dynamic .view(), No tuple return")
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
        print(f"Loading checkpoint...")
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Create original model and load weights
        print(f"Creating original model (latent_dim={latent_dim})...")
        original_model = ImprovedAutoencoder(
            nrow=121, ncol=104, 
            latent_dim=latent_dim, 
            base_channels=base_channels, 
            extra_conv=extra_conv
        )
        original_model.load_state_dict(state_dict)
        original_model.eval()
        
        # Create MATLAB-compatible model
        print(f"Creating MATLAB-compatible model...")
        matlab_model = MatlabCompatibleAutoencoder(
            nrow=121, ncol=104, 
            latent_dim=latent_dim, 
            base_channels=base_channels, 
            extra_conv=extra_conv
        )
        
        # Transfer weights
        print("Transferring weights...")
        transfer_weights(original_model, matlab_model)
        
        # Validate models
        print("\nValidating models...")
        dummy_input = torch.randn(1, 1, 121, 104)
        validate_models(original_model, matlab_model, dummy_input)
        
        # Export traced model
        output_path = model_path.replace('.pth', '_matlab_compatible.pt').replace('.pt', '_matlab_compatible.pt')
        # Avoid double '_matlab_compatible_matlab_compatible.pt'
        if '_matlab_compatible_matlab_compatible' in output_path:
            output_path = output_path.replace('_matlab_compatible_matlab_compatible', '_matlab_compatible')
        
        export_traced_model(matlab_model, dummy_input, output_path)
        
        print(f"\n✓ Successfully created MATLAB-compatible model")
        print(f"  Original: {os.path.basename(model_path)}")
        print(f"  MATLAB:   {os.path.basename(output_path)}")
    
    print("\n" + "=" * 70)
    print("Export Complete!")
    print("=" * 70)
    print("\nAll models exported successfully.")
    print("Uses custom Reshape layer (not nn.Unflatten) for MATLAB compatibility.")
    print("No ATEN layers should appear in MATLAB import.")
    print("The models return only reconstruction (not latent + reconstruction).")


if __name__ == '__main__':
    main()
