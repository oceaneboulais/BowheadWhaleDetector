#!/usr/bin/env bash
# Standalone, offline progress monitor for the 100K-matched training pipeline.
# Run this directly in your own Terminal — it needs no internet connection,
# no Copilot CLI session, nothing but your local Mac. Safe to leave running.
#
# Usage:
#   cd /Users/oceaneboulais/Github/BowheadWhaleDetector
#   ./scripts/watch_100k_matched_pipeline.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/runs/logs_100k_matched"

echo "Watching $LOG_DIR"
echo "Press Ctrl+C to stop watching (this does NOT stop the pipeline)."
echo ""

while true; do
    clear
    echo "=================================================================="
    echo " 100K-matched pipeline status — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================================="
    echo ""
    echo "--- Steps completed ---"
    for i in 1 2 3 4 5; do
        if [ -f "$LOG_DIR/.step${i}_done" ]; then
            echo "  [x] Step $i"
        else
            echo "  [ ] Step $i"
        fi
    done
    echo ""
    echo "--- Currently running process (if any) ---"
    ps aux | grep -E "bowhead\.(data\.build_dataset|train\.train_ae|train\.train_cnn|eval\.score_eval_dataset)" \
        | grep -v grep \
        | awk '{printf "  PID %s  CPU %s%%  TIME %s  ", $2, $3, $10; for(i=11;i<=NF;i++) printf "%s ", $i; print ""}' \
        | cut -c1-160
    echo ""
    echo "--- Last 15 lines of pipeline.log ---"
    tail -n 15 "$LOG_DIR/pipeline.log" 2>/dev/null
    echo ""
    echo "(refreshing every 15s — Ctrl+C to exit; pipeline keeps running)"
    sleep 15
done
