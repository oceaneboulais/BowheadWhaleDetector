# Multi-Feature Autoencoder - MATLAB-Compatible Version

## ✅ Issues Addressed

### 1. Database Paths ✓
The script now uses the correct database paths:
- **Auto Database**: `Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir`
- **Manual Database**: `Unsupervised_database_Manual_100K_Y08101214.dir`

Both paths verified and exist on your system.

### 2. MATLAB Import Issues ✓
**Problem**: MATLAB Deep Learning Toolbox doesn't support `output_padding` parameter in ConvTranspose2d layers.

**Solution**: All models now use MATLAB-compatible architecture:
- Removed `output_padding` from all ConvTranspose2d layers
- Added `nn.ZeroPad2d` layers instead for proper output dimensions
- Models automatically export MATLAB-compatible `.pt` files after training

## 🎯 Three Training Modes

### Mode 1: SNR + NTV (Auto Database)
```bash
python3 Autoencoder_MultiFeature_v01.py \
    --mode snr_ntv \
    --database auto \
    --epochs 100 \
    --latent_dim 32 \
    --channels 32
```
- **Input**: 2 channels [SNR_gram, NTV_gram]
- **Output**: 2 channels [reconstructed SNR, reconstructed NTV]
- **Database**: Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir

### Mode 2: SNR + NTV (Manual Database)
```bash
python3 Autoencoder_MultiFeature_v01.py \
    --mode snr_ntv \
    --database manual \
    --epochs 100 \
    --latent_dim 32 \
    --channels 32
```
- **Input**: 2 channels [SNR_gram, NTV_gram]
- **Output**: 2 channels [reconstructed SNR, reconstructed NTV]
- **Database**: Unsupervised_database_Manual_100K_Y08101214.dir

### Mode 3: SNR + Bearing (Manual Database)
```bash
python3 Autoencoder_MultiFeature_v01.py \
    --mode snr_bearing \
    --database manual \
    --epochs 100 \
    --latent_dim 32 \
    --channels 32
```
- **Input**: 1 channel SNR_gram + bearing scalar
- **Output**: 1 channel reconstructed SNR
- **Conditioning**: Bearing concatenated to latent space
- **Database**: Unsupervised_database_Manual_100K_Y08101214.dir

## 🚀 Quick Start - Run All Three

```bash
cd /Users/oboulais/Public/Bowhead_DL_Project
./run_all_multifeature.sh
```

This will train all three variants sequentially with 100 epochs each.

## 📦 Output Files

Each training run creates:
```
MultiFeature_Results/
└── Autoencoder_{mode}_{database}_Date{timestamp}.dir/
    ├── trained_model/
    │   ├── autoencoder_clean.pth              # Standard PyTorch checkpoint
    │   └── autoencoder_clean_matlab_compatible.pt  # MATLAB-ready traced model
    ├── training_loss.png                       # Loss curve
    ├── image_results/                          # Visualizations
    ├── MATLAB/                                 # MATLAB exports
    └── UMAP/                                   # UMAP analysis
```

## 🔧 MATLAB Import Instructions

After training completes, import into MATLAB:

```matlab
% Navigate to the trained model directory
cd('/Users/oboulais/Public/Bowhead_DL_Project/MultiFeature_Results/Autoencoder_snr_ntv_auto_Date{YOUR_TIMESTAMP}.dir/trained_model')

% Import the MATLAB-compatible model
net = importNetworkFromPyTorch('autoencoder_clean_matlab_compatible.pt');

% Verify the network
analyzeNetwork(net)

% Test with data
testInput = randn(121, 104, 2, 1, 'single');  % For dual-channel (SNR+NTV)
% OR
testInput = randn(121, 104, 1, 1, 'single');  % For bearing-conditioned

output = predict(net, testInput);
disp(['Output size: ', num2str(size(output))]);
```

## ⚠️ NO MORE MATLAB ERRORS

The `output_padding` error has been completely eliminated:
- ❌ Old: `nn.ConvTranspose2d(c1, 2, 2, stride=2, output_padding=(pad_h, pad_w))`
- ✅ New: `nn.ConvTranspose2d(c1, 2, 2, stride=2)` + `nn.ZeroPad2d((0, pad_w, 0, pad_h))`

MATLAB will now successfully import and run the models!

## 🎓 Model Architecture Details

### MATLAB-Compatible Decoder Structure
```python
# For dual-channel (SNR+NTV):
ConvTranspose2d(128→64) + BatchNorm + ReLU
ConvTranspose2d(64→32) + BatchNorm + ReLU  
ConvTranspose2d(32→2)                      # 2 output channels, NO output_padding
ZeroPad2d(0, 0, 0, 1)                     # Add 1 pixel padding to match 121x104

# For bearing-conditioned:
ConvTranspose2d(128→64) + BatchNorm + ReLU
ConvTranspose2d(64→32) + BatchNorm + ReLU
ConvTranspose2d(32→1)                      # 1 output channel, NO output_padding
ZeroPad2d(0, 0, 0, 1)                     # Add 1 pixel padding to match 121x104
```

## 📊 Expected Output Dimensions

All models produce correct output shapes:
- **SNR + NTV**: [Batch, 2, 121, 104]
- **SNR + Bearing**: [Batch, 1, 121, 104]
- **Latent**: [Batch, 32]

## 🔍 Verification

Run quick test to verify everything works:
```bash
python3 quick_test.py
```

Expected output:
```
✓ ALL TESTS PASSED!
Models are working correctly. Ready to train!
```

## 💡 Tips

### Quick Test Run (Fast)
```bash
python3 Autoencoder_MultiFeature_v01.py \
    --mode snr_ntv \
    --database auto \
    --epochs 5 \
    --max_samples 1000 \
    --batch_size 32
```

### Full Production Run
```bash
python3 Autoencoder_MultiFeature_v01.py \
    --mode snr_ntv \
    --database auto \
    --epochs 100 \
    --batch_size 32
```

### Adjust Learning Rate
```bash
--lr 5e-4  # Lower for stability
--lr 5e-3  # Higher for faster training
```

## 📝 Model Checkpoint Contents

```python
{
    'state_dict': <model_weights>,
    'config': {
        'mode': 'snr_ntv' or 'snr_bearing',
        'database': 'auto' or 'manual',
        'latent_dim': 32,
        'base_channels': 32,
        'nrow': 121,
        'ncol': 104,
    },
    'loss_history': [losses_per_epoch]
}
```

## ✨ Summary

**All systems ready for training!**

✅ Database paths verified  
✅ MATLAB compatibility ensured  
✅ Models tested and validated  
✅ Automatic traced model export  
✅ No output_padding errors  

**Ready to train all three variants with:**
```bash
./run_all_multifeature.sh
```
