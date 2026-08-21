# Multi-Feature Autoencoder System

## Overview

This system provides autoencoder variants that incorporate multiple features beyond just the spectrogram:

1. **SNR + NTV (Dual-Channel)**: Processes both SNR_gram and NTV_gram as a 2-channel input
2. **SNR + Bearing (Conditioned)**: Processes SNR_gram conditioned on bearing scalar value

## Available Data Fields

Each .mat file contains:
- `SNR_gram`: (121, 104) - Signal-to-noise ratio spectrogram
- `NTV_gram`: (121, 104) - Normalized transport velocity gram
- `bearing`: (1, 1) - Bearing angle in degrees
- `KEtoPE_gram`: (121, 104) - Kinetic to potential energy ratio
- `Polar_gram`: (121, 104) - Polar representation

## Architecture Variants

### 1. Dual-Channel Autoencoder (`snr_ntv`)
- **Input**: 2 channels [SNR_gram, NTV_gram]
- **Output**: 2 channels [reconstructed SNR, reconstructed NTV]
- **Use case**: Jointly learning features from both SNR and NTV

### 2. Bearing-Conditioned Autoencoder (`snr_bearing`)
- **Input**: 1 channel SNR_gram + bearing scalar
- **Output**: 1 channel reconstructed SNR
- **Conditioning**: Bearing value concatenated to latent representation
- **Use case**: Learning directional-aware features

## Usage

### Quick Start - Run All Three Variants

```bash
cd /Users/oboulais/Public/Bowhead_DL_Project
./run_all_multifeature.sh
```

This will execute:
1. SNR + NTV on Auto database
2. SNR + NTV on Manual database  
3. SNR + Bearing on Manual database

### Individual Runs

**Run 1: SNR + NTV for Auto Database**
```bash
python3 Autoencoder_MultiFeature_v01.py \
    --mode snr_ntv \
    --database auto \
    --epochs 100 \
    --latent_dim 32 \
    --channels 32 \
    --batch_size 32
```

**Run 2: SNR + NTV for Manual Database**
```bash
python3 Autoencoder_MultiFeature_v01.py \
    --mode snr_ntv \
    --database manual \
    --epochs 100 \
    --latent_dim 32 \
    --channels 32 \
    --batch_size 32
```

**Run 3: SNR + Bearing for Manual Database**
```bash
python3 Autoencoder_MultiFeature_v01.py \
    --mode snr_bearing \
    --database manual \
    --epochs 100 \
    --latent_dim 32 \
    --channels 32 \
    --batch_size 32
```

## Command-Line Arguments

```
--mode           Mode: 'snr_ntv' or 'snr_bearing' (required)
--database       Database: 'auto' or 'manual' (required)
--epochs         Number of training epochs (default: 100)
--latent_dim     Latent dimension size (default: 32)
--channels       Base number of channels (default: 32)
--lr             Learning rate (default: 1e-3)
--batch_size     Batch size (default: 32)
--max_samples    Maximum samples to use (default: all)
```

## Database Paths

The script automatically uses these paths:
- **Auto**: `/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns_100K_Y08101214.dir`
- **Manual**: `/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir`

## Output Structure

Results are saved to: `/Users/oboulais/Public/Bowhead_DL_Project/MultiFeature_Results/`

Each run creates a timestamped directory:
```
Autoencoder_{mode}_{database}_Date{timestamp}.dir/
├── MATLAB/                    # MATLAB export files
├── image_results/             # Visualization outputs
├── trained_model/             
│   └── autoencoder_clean.pth  # Trained model checkpoint
├── UMAP/                      # UMAP analysis
└── training_loss.png          # Loss curve
```

## Model Checkpoint Contents

Each saved model includes:
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
    'loss_history': [list of losses per epoch]
}
```

## Loading Trained Models

### Dual-Channel Model
```python
import torch
from Autoencoder_MultiFeature_v01 import DualChannelAutoencoder

# Load checkpoint
checkpoint = torch.load('path/to/autoencoder_clean.pth')
config = checkpoint['config']

# Create model
model = DualChannelAutoencoder(
    nrow=121, ncol=104,
    latent_dim=config['latent_dim'],
    base_channels=config['base_channels'],
    extra_conv=False
)

# Load weights
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Use model
import numpy as np
from scipy.io import loadmat

data = loadmat('sample.mat')
snr = data['SNR_gram']
ntv = data['NTV_gram']

# Normalize
snr = (snr - snr.min()) / (snr.max() - snr.min())
ntv = (ntv - ntv.min()) / (ntv.max() - ntv.min())

# Create input tensor
input_tensor = torch.from_numpy(np.stack([snr, ntv], axis=0)).unsqueeze(0)

# Get reconstruction and latent
with torch.no_grad():
    recon, latent = model(input_tensor)

print(f"Latent shape: {latent.shape}")  # [1, 32]
print(f"Reconstruction shape: {recon.shape}")  # [1, 2, 121, 104]
```

### Bearing-Conditioned Model
```python
from Autoencoder_MultiFeature_v01 import BearingConditionedAutoencoder

# Load and create model (same as above)
model = BearingConditionedAutoencoder(...)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Use model with bearing
data = loadmat('sample.mat')
snr = data['SNR_gram']
bearing = float(data['bearing'].flatten()[0])

# Normalize
snr = (snr - snr.min()) / (snr.max() - snr.min())
bearing_norm = bearing / 360.0

# Create input tensors
snr_tensor = torch.from_numpy(snr).unsqueeze(0).unsqueeze(0)
bearing_tensor = torch.tensor([[bearing_norm]], dtype=torch.float32)

# Get reconstruction and latent
with torch.no_grad():
    recon, latent = model(snr_tensor, bearing_tensor)

print(f"Latent shape: {latent.shape}")  # [1, 32]
print(f"Reconstruction shape: {recon.shape}")  # [1, 1, 121, 104]
```

## Architecture Details

### Dual-Channel Encoder
1. Conv2d(2→32) + BN + ReLU + MaxPool
2. Conv2d(32→64) + BN + ReLU + MaxPool
3. Conv2d(64→128) + BN + ReLU + MaxPool
4. Flatten → Linear(flat_size → 64) → Linear(64 → 32)

### Dual-Channel Decoder
1. Linear(32 → 64) → Linear(64 → flat_size)
2. Reshape
3. ConvTranspose2d(128→64) + BN + ReLU
4. ConvTranspose2d(64→32) + BN + ReLU
5. ConvTranspose2d(32→2)

### Bearing-Conditioned Architecture
- Same as single-channel but with bearing concatenated at bottleneck
- Encoder: SNR → features → concat(features, bearing) → latent
- Decoder: concat(latent, bearing) → features → SNR reconstruction

## Comparison with Original Autoencoder

| Feature | Original | SNR+NTV | SNR+Bearing |
|---------|----------|---------|-------------|
| Input Channels | 1 (SNR) | 2 (SNR+NTV) | 1 (SNR) + scalar |
| Output Channels | 1 | 2 | 1 |
| Conditioning | None | None | Bearing |
| Use Case | SNR-only | Multi-modal | Directional |

## Tips for Training

1. **Batch Size**: Reduce if you run out of memory
   ```bash
   --batch_size 16
   ```

2. **Learning Rate**: Adjust if loss doesn't converge
   ```bash
   --lr 5e-4  # Lower for stability
   --lr 5e-3  # Higher for faster training
   ```

3. **Quick Test**: Use fewer samples for testing
   ```bash
   --max_samples 1000 --epochs 10
   ```

4. **Monitor Training**: Check `training_loss.png` periodically

## Next Steps After Training

1. **Analyze Latent Space**: Use t-SNE/UMAP visualization
2. **Compare Reconstructions**: Visual quality assessment
3. **Feature Analysis**: Compare latent representations across modes
4. **Classification**: Use latent features for downstream tasks

## Troubleshooting

**Issue**: CUDA out of memory  
**Solution**: Reduce batch size: `--batch_size 16` or `--batch_size 8`

**Issue**: Dataset not found  
**Solution**: Check paths in `DATABASE_PATHS` dictionary in the script

**Issue**: NTV_gram or bearing missing from .mat files  
**Solution**: Verify data files contain required fields using:
```python
from scipy.io import loadmat
m = loadmat('yourfile.mat')
print(m.keys())
```

## Future Extensions

Potential additions:
- Add KEtoPE_gram as third channel
- Add Polar_gram conditioning
- Multi-task learning (predict bearing from latent)
- Variational autoencoder (VAE) variant
- Attention mechanisms for multi-modal fusion
