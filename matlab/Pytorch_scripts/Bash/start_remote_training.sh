#!/bin/bash
# Start training on garibaldi in background (allows closing laptop)

SERVER="oboulais@garibaldi.ucsd.edu"
REMOTE_DIR="~/BowheadDeepLearningMATLAB/Pytorch_scripts"
REMOTE_RESULTS="~/BowheadDeepLearningMATLAB/results"
LOCAL_RESULTS="/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results"
DATASET="/home/oboulais/BCB_Whale_Datasets/Unsupervised_database_AutoWithAirguns.dir"

# Training parameters (always default to 100 epochs)
EPOCHS=${1:-100}
LATENT_DIM=${2:-32}
CHANNELS=${3:-64}
BATCH_SIZE=${4:-32}
SEED=${5:-42}

echo "========================================"
echo "Starting GPU Training on Garibaldi"
echo "========================================"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS (GPU-accelerated)"
echo "Latent Dim: $LATENT_DIM"
echo "Channels: $CHANNELS"
echo "Batch Size: $BATCH_SIZE"
echo "Seed: $SEED"
echo ""
echo "Training will run in background with nohup"
echo "Results will be auto-downloaded when complete"
echo "You can close your laptop after this starts"
echo "========================================"
echo ""

# Copy updated GPU-enabled script to garibaldi
echo "1. Copying latest GPU-enabled script to garibaldi..."
scp Autoencoder_v02_20251118.py $SERVER:$REMOTE_DIR/
echo "   ✓ Script uploaded"
echo ""

# Verify GPU support in uploaded script
echo "2. Verifying GPU support in script..."
ssh $SERVER "grep -q 'device = torch.device' $REMOTE_DIR/Autoencoder_v02_20251118.py && echo '   ✓ GPU support confirmed' || echo '   ✗ WARNING: GPU code not found'"
echo ""

# Start training in background
echo "3. Starting GPU training process..."
ssh $SERVER "cd $REMOTE_DIR && \
  source .venv_garibaldi/bin/activate && \
  nohup python -u Autoencoder_v02_20251118.py \
    --data-dir $DATASET \
    --epochs $EPOCHS \
    --latent-dim $LATENT_DIM \
    --channels $CHANNELS \
    --batch-size $BATCH_SIZE \
    --seed $SEED \
    > training_gpu.log 2>&1 &"

# Wait a moment and check if it started
sleep 3
echo ""
echo "4. Checking if GPU training started..."
ssh $SERVER "pgrep -af 'python.*Autoencoder' && echo '   ✓ SUCCESS: Training is running!' || echo '   ✗ WARNING: Process not found'"
echo ""

# Check GPU detection in log
echo "5. Verifying GPU detection..."
sleep 2
ssh $SERVER "head -5 $REMOTE_DIR/training_gpu.log | grep -i 'cuda\|gpu' || echo '   ⚠ Waiting for log...'"

echo ""
echo "========================================"
echo "GPU Training Started!"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  ./monitor_training.sh"
echo ""
echo "Watch live log:"
echo "  ssh $SERVER 'tail -f $REMOTE_DIR/training_gpu.log'"
echo ""
echo "Download results when complete:"
echo "  ./download_results.sh"
echo ""
echo "Or auto-download (waits for completion):"
echo "  ./start_remote_training.sh $EPOCHS && ./wait_and_download.sh"
echo ""
echo "Stop training (if needed):"
echo "  ssh $SERVER 'pkill -f python.*Autoencoder'"
echo ""
echo "You can now close your laptop safely!"
echo "Training on GPU should take ~2-3 hours for 100 epochs"
echo "========================================"
