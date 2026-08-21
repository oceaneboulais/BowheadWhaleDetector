#!/bin/bash
# Download results from garibaldi

SERVER="oboulais@garibaldi.ucsd.edu"
REMOTE_DIR="~/BowheadDeepLearningMATLAB"
LOCAL_DIR="/Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB"

echo "========================================"
echo "Downloading Results from Garibaldi"
echo "========================================"
echo ""

# List available result directories
echo "Available result directories:"
ssh $SERVER "ls -lhtr $REMOTE_DIR/results/ 2>/dev/null | tail -10"
echo ""

# Ask which one to download (or download all recent)
echo "Downloading all results..."
rsync -avz --progress $SERVER:$REMOTE_DIR/results/ $LOCAL_DIR/results_from_garibaldi/

echo ""
echo "========================================"
echo "Download Complete!"
echo "========================================"
echo "Results saved to: $LOCAL_DIR/results_from_garibaldi/"
echo ""
