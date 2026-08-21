#!/bin/bash
# Run training locally on your MacBook

DATASET="/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/models/Unsupervised_database_With_Airguns.dir"

# Training parameters
EPOCHS=${1:-100}
LATENT_DIM=${2:-32}
CHANNELS=${3:-64}
BATCH_SIZE=${4:-32}
SEED=${5:-42}

echo "========================================"
echo "Local MacBook Training"
echo "========================================"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Latent Dim: $LATENT_DIM"
echo "Channels: $CHANNELS"
echo "Batch Size: $BATCH_SIZE"
echo "Seed: $SEED"
echo ""
echo "Device: Will auto-detect GPU/CPU"
echo "Results will save to: results/"
echo "========================================"
echo ""

# Activate virtual environment
source .venv_py31018/bin/activate

# Run training directly (not in background - you'll see output)
python Autoencoder_v02_20251118.py \
  --data-dir $DATASET \
  --epochs $EPOCHS \
  --latent-dim $LATENT_DIM \
  --channels $CHANNELS \
  --batch-size $BATCH_SIZE \
  --seed $SEED

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Results saved to: results/"
echo ""
