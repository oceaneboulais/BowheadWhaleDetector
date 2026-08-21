#!/bin/bash
# Monitor training progress on garibaldi

SERVER="oboulais@garibaldi.ucsd.edu"
REMOTE_DIR="~/BowheadDeepLearningMATLAB/Pytorch_scripts"

echo "========================================"
echo "Garibaldi Training Monitor"
echo "========================================"
echo ""

# Check if training process is running
echo "1. Checking if training is running..."
ssh $SERVER "pgrep -af 'python.*Autoencoder' || echo 'No training process found'"
echo ""

# Show last 30 lines of log
echo "2. Last 30 lines of training log:"
echo "----------------------------------------"
ssh $SERVER "tail -30 $REMOTE_DIR/training.log 2>/dev/null || echo 'No log file found yet'"
echo ""

# Show disk usage of results directory
echo "3. Results directory size:"
ssh $SERVER "du -sh $REMOTE_DIR/../results/ 2>/dev/null || echo 'No results yet'"
echo ""

# Show GPU usage if nvidia-smi is available
echo "4. GPU status (if available):"
ssh $SERVER "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null || echo 'nvidia-smi not available'"
echo ""

echo "========================================"
echo "Use 'tail -f' to watch live updates:"
echo "  ssh $SERVER 'tail -f $REMOTE_DIR/training.log'"
echo "========================================"
