#!/usr/bin/env python3
"""Quick model architecture test - no dataset scanning."""
import torch
import sys
sys.path.insert(0, '/Users/oboulais/Public/Bowhead_DL_Project')

from Autoencoder_MultiFeature_v01 import (
    DualChannelAutoencoder,
    BearingConditionedAutoencoder,
)

print("="*70)
print("Quick Model Architecture Test")
print("="*70)

# Test 1: Dual-Channel Autoencoder
print("\n[1/2] Testing DualChannelAutoencoder...")
try:
    model = DualChannelAutoencoder(nrow=121, ncol=104, latent_dim=32, base_channels=32)
    params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created: {params:,} parameters")
    
    # Forward pass test
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(2, 2, 121, 104)  # 2-channel input
        recon, latent = model(dummy_input)
    
    print(f"  ✓ Forward pass: Input {dummy_input.shape} → Latent {latent.shape} → Output {recon.shape}")
    assert recon.shape == (2, 2, 121, 104), f"Expected output shape (2, 2, 121, 104), got {recon.shape}"
    assert latent.shape == (2, 32), f"Expected latent shape (2, 32), got {latent.shape}"
    print("  ✓ Output shapes correct")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Bearing-Conditioned Autoencoder
print("\n[2/2] Testing BearingConditionedAutoencoder...")
try:
    model = BearingConditionedAutoencoder(nrow=121, ncol=104, latent_dim=32, base_channels=32)
    params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created: {params:,} parameters")
    
    # Forward pass test
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(2, 1, 121, 104)  # 1-channel input
        dummy_bearing = torch.randn(2, 1)  # Bearing scalar
        recon, latent = model(dummy_input, dummy_bearing)
    
    print(f"  ✓ Forward pass: Input {dummy_input.shape} + Bearing {dummy_bearing.shape} → Latent {latent.shape} → Output {recon.shape}")
    assert recon.shape == (2, 1, 121, 104), f"Expected output shape (2, 1, 121, 104), got {recon.shape}"
    assert latent.shape == (2, 32), f"Expected latent shape (2, 32), got {latent.shape}"
    print("  ✓ Output shapes correct")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nModels are working correctly. Ready to train!")
print("\nQuick start:")
print("  ./run_all_multifeature.sh")
print("\nOr run individual modes:")
print("  python3 Autoencoder_MultiFeature_v01.py --mode snr_ntv --database auto --epochs 10 --max_samples 1000")
print("  python3 Autoencoder_MultiFeature_v01.py --mode snr_bearing --database manual --epochs 10 --max_samples 1000")
