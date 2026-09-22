#!/usr/bin/env bash
# Constant-Q (log-frequency-warped) counterpart to
# scripts/run_100k_matched_pipeline.sh: retrains the autoencoder, scratch CNN,
# and warm-start CNN with --freq-warp on the SAME already-built matched 100K
# dataset (data/spectrograms_100k_matched.npz -- no external drive needed,
# step 1 is skipped since that file already exists), then runs the call-type
# probing/clustering study (bowhead/benchmark/run_calltype_study.py) with
# --freq-warp against the resulting checkpoints, to fill in
# paper/ml_paper/cqt_manuscript.tex's Table 1.
#
# Resumable: each step writes a ".stepN_done" marker under runs/logs_100k_matched_cqt/;
# re-running skips already-completed steps.
#
# Usage:
#   nohup ./scripts/run_100k_matched_cqt_pipeline.sh > runs/logs_100k_matched_cqt/pipeline.log 2>&1 &
#   disown
#   tail -f runs/logs_100k_matched_cqt/pipeline.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$REPO_ROOT/.venv_ae/bin/python3}"   # this venv has torch/scipy/sklearn + MPS

DATA="$REPO_ROOT/data/spectrograms_100k_matched.npz"
AE_OUT_DIR="$REPO_ROOT/runs/ae_100k_matched_cqt"
AE_CKPT="$AE_OUT_DIR/autoencoder_clean.pt"
SCRATCH_TAG="scratch_100k_matched_cqt"
WARMSTART_TAG="warmstart_100k_matched_cqt"
CALLTYPE_OUT="$REPO_ROOT/runs/calltype_study_cqt"

LOG_DIR="$REPO_ROOT/runs/logs_100k_matched_cqt"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if [ ! -f "$DATA" ]; then
    log "ERROR: expected dataset not found: $DATA"
    exit 1
fi

# --------------------------------------------------------------------------
# Step 1: retrain the 32-D autoencoder with the constant-Q-style warp
# --------------------------------------------------------------------------
STEP1_DONE="$LOG_DIR/.step1_done"
if [ ! -f "$STEP1_DONE" ]; then
    log "STEP 1/4: Retraining autoencoder with --freq-warp (epochs=100, patience=10)"
    "$PY" -m bowhead.train.train_ae \
        --data "$DATA" \
        --out-dir "$AE_OUT_DIR" \
        --epochs 100 --batch 32 --lr 1e-3 --patience 10 --seed 42 --device auto \
        --freq-warp \
        2>&1 | tee "$LOG_DIR/01_train_ae_cqt.log"
    touch "$STEP1_DONE"
else
    log "STEP 1/4: already done, skipping ($AE_CKPT)"
fi

# --------------------------------------------------------------------------
# Step 2: scratch (random-init) CNN with the constant-Q-style warp
# --------------------------------------------------------------------------
STEP2_DONE="$LOG_DIR/.step2_done"
if [ ! -f "$STEP2_DONE" ]; then
    log "STEP 2/4: Training scratch CNN with --freq-warp"
    "$PY" -m bowhead.train.train_cnn \
        --data "$DATA" \
        --tag "$SCRATCH_TAG" \
        --seed 42 --device auto --freq-warp \
        2>&1 | tee "$LOG_DIR/02_train_scratch_cnn_cqt.log"
    touch "$STEP2_DONE"
else
    log "STEP 2/4: already done, skipping (runs/$SCRATCH_TAG/best.pt)"
fi

# --------------------------------------------------------------------------
# Step 3: warm-start CNN, initialised from the Step-1 constant-Q AE checkpoint
# --------------------------------------------------------------------------
STEP3_DONE="$LOG_DIR/.step3_done"
if [ ! -f "$STEP3_DONE" ]; then
    log "STEP 3/4: Training warm-start CNN with --freq-warp (init from $AE_CKPT)"
    "$PY" -m bowhead.train.train_cnn \
        --data "$DATA" \
        --tag "$WARMSTART_TAG" \
        --warm-start "$AE_CKPT" \
        --seed 42 --device auto --freq-warp \
        2>&1 | tee "$LOG_DIR/03_train_warmstart_cnn_cqt.log"
    touch "$STEP3_DONE"
else
    log "STEP 3/4: already done, skipping (runs/$WARMSTART_TAG/best.pt)"
fi

# --------------------------------------------------------------------------
# Step 4: call-type linear-probe + clustering study (fills in cqt_manuscript's
#         Table 1), reusing the identical protocol as ml_manuscript.tex
# --------------------------------------------------------------------------
STEP4_DONE="$LOG_DIR/.step4_done"
if [ ! -f "$STEP4_DONE" ]; then
    log "STEP 4/4: Running call-type probing/clustering study with --freq-warp"
    "$PY" -m bowhead.benchmark.run_calltype_study \
        --data "$DATA" \
        --ae "$AE_CKPT" \
        --scratch "$REPO_ROOT/runs/$SCRATCH_TAG/best.pt" \
        --warmstart "$REPO_ROOT/runs/$WARMSTART_TAG/best.pt" \
        --out "$CALLTYPE_OUT" \
        --device auto --seed 42 --freq-warp \
        2>&1 | tee "$LOG_DIR/04_calltype_study_cqt.log"
    touch "$STEP4_DONE"
else
    log "STEP 4/4: already done, skipping ($CALLTYPE_OUT/results.json)"
fi

log "Pipeline complete. Results: $CALLTYPE_OUT/results.json"
