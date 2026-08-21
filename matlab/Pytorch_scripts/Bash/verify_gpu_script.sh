#!/bin/bash
# Quick verification that GPU script is ready

cd /Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/Pytorch_scripts

echo "========================================"
echo "GPU Training Script Verification"
echo "========================================"
echo ""

# Check 1: GPU support in local script
echo "✓ Checking local script for GPU support..."
if grep -q "device = torch.device" Autoencoder_v02_20251118.py; then
    echo "  ✓ GPU support found (line $(grep -n 'device = torch.device' Autoencoder_v02_20251118.py | head -1 | cut -d: -f1))"
else
    echo "  ✗ WARNING: GPU support NOT found"
    exit 1
fi

# Check 2: Required imports
echo "✓ Checking required imports..."
if grep -q "import sys" Autoencoder_v02_20251118.py; then
    echo "  ✓ sys module imported"
else
    echo "  ✗ WARNING: sys module missing"
    exit 1
fi

# Check 3: Model device placement
echo "✓ Checking model device placement..."
if grep -q "model = model.to(device)" Autoencoder_v02_20251118.py; then
    echo "  ✓ Model moved to device"
else
    echo "  ✗ WARNING: Model not moved to device"
    exit 1
fi

# Check 4: Data device placement
echo "✓ Checking data device placement..."
if grep -q "batch_data = batch_data.to(device)" Autoencoder_v02_20251118.py; then
    echo "  ✓ Batch data moved to device"
else
    echo "  ✗ WARNING: Batch data not moved to device"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ All checks passed!"
echo "========================================"
echo ""
echo "Ready to launch GPU training with:"
echo "  ./start_remote_training.sh 100"
echo ""
echo "Or use all-in-one command:"
echo "  ./launch_training.sh"
echo "========================================"
