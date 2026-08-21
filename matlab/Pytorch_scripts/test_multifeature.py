#!/usr/bin/env python3
"""
Quick test script to verify multi-feature autoencoder setup.
Tests that datasets load correctly and models can process data.
"""
import torch
import sys
import os
from scipy.io import loadmat
import glob

# Add current directory to path
sys.path.insert(0, '/Users/oboulais/Public/Bowhead_DL_Project')

from Autoencoder_MultiFeature_v01 import (
    DualChannelDataset,
    BearingConditionedDataset,
    DualChannelAutoencoder,
    BearingConditionedAutoencoder,
    DATABASE_PATHS
)

def test_dataset(dataset_class, name, db_path):
    """Test dataset loading."""
    print(f"\n{'='*70}")
    print(f"Testing {name}")
    print(f"{'='*70}")
    
    try:
        print(f"  Scanning directory (this may take a moment)...")
        dataset = dataset_class(
            db_path,
            normalize=True,
            seed=42,
            show_summary=False,  # Don't show summary to reduce output
            max_samples=5  # Just load 5 samples for quick testing
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Test loading a sample
        sample = dataset[0]
        if isinstance(sample[0], tuple):
            # Bearing-conditioned
            data, bearing = sample[0]
            print(f"✓ Sample loaded: SNR shape={data.shape}, Bearing={bearing.item():.3f}")
        else:
            # Dual-channel
            data = sample[0]
            print(f"✓ Sample loaded: Shape={data.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model(model_class, name, input_shape, has_bearing=False):
    """Test model forward pass."""
    print(f"\n{'='*70}")
    print(f"Testing {name}")
    print(f"{'='*70}")
    
    try:
        model = model_class(
            nrow=121, ncol=104,
            latent_dim=32,
            base_channels=32,
            extra_conv=False
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created: {total_params:,} parameters")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            if has_bearing:
                dummy_input = torch.randn(2, 1, 121, 104)  # Batch of 2
                dummy_bearing = torch.randn(2, 1)
                recon, latent = model(dummy_input, dummy_bearing)
            else:
                dummy_input = torch.randn(2, *input_shape, 121, 104)  # Batch of 2
                recon, latent = model(dummy_input)
        
        print(f"✓ Forward pass successful:")
        print(f"  Input shape: {dummy_input.shape}")
        if has_bearing:
            print(f"  Bearing shape: {dummy_bearing.shape}")
        print(f"  Latent shape: {latent.shape}")
        print(f"  Output shape: {recon.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_mat_file_contents():
    """Check what fields are available in sample .mat files."""
    print(f"\n{'='*70}")
    print("Checking .mat File Contents")
    print(f"{'='*70}")
    
    for db_name, db_path in DATABASE_PATHS.items():
        print(f"\n{db_name.upper()} Database:")
        if not os.path.exists(db_path):
            print(f"  ✗ Path not found: {db_path}")
            continue
        
        mat_files = glob.glob(os.path.join(db_path, '*.mat'))[:1]
        if not mat_files:
            print(f"  ✗ No .mat files found")
            continue
        
        try:
            m = loadmat(mat_files[0])
            print(f"  ✓ Sample file: {os.path.basename(mat_files[0])}")
            print(f"  Available fields:")
            for key in m.keys():
                if not key.startswith('__'):
                    val = m[key]
                    shape = getattr(val, 'shape', 'N/A')
                    print(f"    - {key}: shape={shape}")
        except Exception as e:
            print(f"  ✗ Error reading .mat file: {e}")


def main():
    print("="*70)
    print("Multi-Feature Autoencoder - System Test")
    print("="*70)
    
    # Check .mat file contents
    check_mat_file_contents()
    
    # Test datasets
    print("\n" + "="*70)
    print("DATASET TESTS")
    print("="*70)
    
    results = []
    
    for db_name, db_path in DATABASE_PATHS.items():
        if os.path.exists(db_path):
            # Test dual-channel dataset
            result = test_dataset(
                DualChannelDataset,
                f"DualChannelDataset ({db_name})",
                db_path
            )
            results.append(("DualChannel-" + db_name, result))
            
            # Test bearing-conditioned dataset
            result = test_dataset(
                BearingConditionedDataset,
                f"BearingConditionedDataset ({db_name})",
                db_path
            )
            results.append(("BearingConditioned-" + db_name, result))
        else:
            print(f"\n⚠ Skipping {db_name}: path not found")
    
    # Test models
    print("\n" + "="*70)
    print("MODEL TESTS")
    print("="*70)
    
    result = test_model(
        DualChannelAutoencoder,
        "DualChannelAutoencoder",
        input_shape=(2,),
        has_bearing=False
    )
    results.append(("DualChannelAutoencoder", result))
    
    result = test_model(
        BearingConditionedAutoencoder,
        "BearingConditionedAutoencoder",
        input_shape=(1,),
        has_bearing=True
    )
    results.append(("BearingConditionedAutoencoder", result))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to train.")
        print("\nTo start training, run:")
        print("  ./run_all_multifeature.sh")
        print("\nOr individual runs:")
        print("  python3 Autoencoder_MultiFeature_v01.py --mode snr_ntv --database auto")
        print("  python3 Autoencoder_MultiFeature_v01.py --mode snr_bearing --database manual")
    else:
        print("\n⚠ Some tests failed. Please check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
