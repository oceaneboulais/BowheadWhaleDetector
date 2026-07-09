#!/usr/bin/env bash
# Live view of the two matched-100K CNN trainings (scratch + warm-start).
# Run this directly in your own Terminal.app — no internet or chat session needed.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/runs/logs_100k_matched"

while true; do
    clear
    echo "=================================================================="
    echo " Matched-100K CNN training — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================================="
    echo ""
    echo "--- Running processes ---"
    ps aux | grep "bowhead.train.train_cnn" | grep -v grep \
        | awk '{printf "  PID %s  CPU %s%%  TIME %s  TAG ", $2, $3, $10; for(i=11;i<=NF;i++) if($i=="--tag") print $(i+1)}'
    echo ""
    echo "--- Scratch CNN (last 8 lines) ---"
    tail -n 8 "$LOG_DIR/03_train_scratch_cnn.log" 2>/dev/null
    echo ""
    echo "--- Warm-start CNN (last 8 lines) ---"
    tail -n 8 "$LOG_DIR/04_train_warmstart_cnn.log" 2>/dev/null
    echo ""
    echo "(refreshing every 10s — Ctrl+C to exit; training keeps running)"
    sleep 10
done
