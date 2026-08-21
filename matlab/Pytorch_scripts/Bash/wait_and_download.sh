#!/bin/bash
# Wait for training to complete and automatically download results

SERVER="oboulais@garibaldi.ucsd.edu"
REMOTE_DIR="~/BowheadDeepLearningMATLAB/Pytorch_scripts"
REMOTE_RESULTS="~/BowheadDeepLearningMATLAB/results"
LOCAL_RESULTS="/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/results"

echo "========================================"
echo "Waiting for Training to Complete"
echo "========================================"
echo ""

# Check if training is running
while true; do
    RUNNING=$(ssh $SERVER "pgrep -f 'python.*Autoencoder'" 2>/dev/null)
    
    if [ -z "$RUNNING" ]; then
        echo "Training complete! Starting download..."
        break
    else
        # Get last line of log for progress
        PROGRESS=$(ssh $SERVER "tail -1 $REMOTE_DIR/training_gpu.log 2>/dev/null | grep 'Epoch'" || echo "Training in progress...")
        echo "[$(date '+%H:%M:%S')] $PROGRESS"
        sleep 60  # Check every minute
    fi
done

echo ""
echo "========================================"
echo "Downloading Results"
echo "========================================"
echo ""

# Find the most recent result directory
LATEST_DIR=$(ssh $SERVER "ls -t $REMOTE_RESULTS | head -1")
echo "Latest result: $LATEST_DIR"
echo ""

# Download with rsync
echo "Downloading to: $LOCAL_RESULTS/$LATEST_DIR"
rsync -avz --progress $SERVER:$REMOTE_RESULTS/$LATEST_DIR/ $LOCAL_RESULTS/$LATEST_DIR/

echo ""
echo "========================================"
echo "Download Complete!"
echo "========================================"
echo "Results saved to:"
echo "  $LOCAL_RESULTS/$LATEST_DIR/"
echo ""
echo "Key files:"
echo "  - autoencoder_clean.pth (trained model)"
echo "  - latent_embeddings.mat (embeddings + filenames)"
echo "  - tsne_latent.png (visualization)"
echo "  - reconstructions.png (sample outputs)"
echo "========================================"
