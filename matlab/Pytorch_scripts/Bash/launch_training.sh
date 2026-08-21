#!/bin/bash
# Quick launcher: Start training on garibaldi GPU and auto-download when done

cd /Users/oceaneboulais/Github/ThodeLab/BowheadDeepLearningMATLAB/Pytorch_scripts

echo "========================================"
echo "GPU Training + Auto-Download"
echo "========================================"
echo ""
echo "This will:"
echo "  1. Upload GPU-enabled script to garibaldi"
echo "  2. Start 100-epoch training on RTX 2080 Ti"
echo "  3. Wait for completion (~2-3 hours)"
echo "  4. Auto-download results to local results/"
echo ""
echo "You can close this terminal after training starts"
echo "Press Ctrl+C to cancel auto-download (training continues)"
echo ""
read -p "Press Enter to start..."
echo ""

# Start training
./start_remote_training.sh 100

# Wait and download
./wait_and_download.sh
