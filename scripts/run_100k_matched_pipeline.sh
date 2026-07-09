#!/usr/bin/env bash
# Fully offline, resumable pipeline that trains the autoencoder, scratch CNN,
# and warm-start CNN on one identical, deterministic 100,000-sample dataset
# (50,000 manually-verified calls + 50,000 auto-detected non-calls), giving
# true three-way parity for the JASA ablation.
#
# Why this exists: the original AE (Autoencoder_v13, Apr 2026) was trained on
# 50K+50K samples subsampled from the Auto_100K/Manual_100K .mat databases,
# but those source directories have since grown (see build log below), so the
# *exact* original 100K files can no longer be reconstructed. This script
# instead builds ONE new deterministic 100K subset (seed=42) from the current
# directories and trains all three models (AE, scratch CNN, warm-start CNN)
# on it, so their comparison is on identical, reproducible data even if it
# isn't literally the April-2026 file list.
#
# 100% OFFLINE: every step reads local .mat files and trains a from-scratch
# custom architecture (no torch.hub / pretrained downloads / network calls
# anywhere in this pipeline) so it is safe to run with no internet connection.
#
# Resumable: each step writes a ".stepN_done" marker; re-running the script
# skips already-completed steps, so it's safe to interrupt (e.g. laptop sleep)
# and restart.
#
# Usage:
#   nohup ./scripts/run_100k_matched_pipeline.sh > runs/logs_100k_matched/pipeline.log 2>&1 &
#   disown
#   tail -f runs/logs_100k_matched/pipeline.log

set -euo pipefail

# --------------------------------------------------------------------------
# Paths (edit here if the external drive mount point changes)
# --------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/usr/local/bin/python3.8}"   # this interpreter has torch/scipy/sklearn

BCB_ROOT="/Volumes/R3D_2024_1/BowheadDeepLearningMATLAB/BCB_Whale_Datasets"
AUTO_DIR="$BCB_ROOT/Unsupervised_database_Auto_100K_ADG_Y08101214_centered_16Apr2026.dir"
MANUAL_DIR="$BCB_ROOT/Unsupervised_database_Manual_100K_ADG_Y08101214_centered_16Apr2026.dir"
EVAL_DIR="$BCB_ROOT/Unsupervised_database_Evaluation_200K_8Auto1Manual_ADG_Y08101214_centered_06May2026.dir"

DATA_OUT="$REPO_ROOT/data/spectrograms_100k_matched.npz"
AE_OUT_DIR="$REPO_ROOT/runs/ae_100k_matched"
AE_CKPT="$AE_OUT_DIR/autoencoder_clean.pt"
SCRATCH_TAG="scratch_100k_matched"
WARMSTART_TAG="warmstart_100k_matched"

LOG_DIR="$REPO_ROOT/runs/logs_100k_matched"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

require_mounted() {
    if [ ! -d "$1" ]; then
        log "ERROR: expected directory not found: $1"
        log "Is the R3D_2024_1 drive mounted?"
        exit 1
    fi
}

# --------------------------------------------------------------------------
# Step 1: build one deterministic 100K dataset (50K manual + 50K auto, seed=42)
# --------------------------------------------------------------------------
STEP1_DONE="$LOG_DIR/.step1_done"
if [ ! -f "$STEP1_DONE" ]; then
    log "STEP 1/5: Building matched 100K dataset (50K manual + 50K auto, seed=42)"
    require_mounted "$AUTO_DIR"
    require_mounted "$MANUAL_DIR"
    "$PY" -m bowhead.data.build_dataset \
        --manual-dir "$MANUAL_DIR" \
        --auto-dir "$AUTO_DIR" \
        --max-per-class 50000 \
        --seed 42 \
        --out "$DATA_OUT" \
        2>&1 | tee "$LOG_DIR/01_build_dataset.log"
    touch "$STEP1_DONE"
else
    log "STEP 1/5: already done, skipping ($DATA_OUT)"
fi

# --------------------------------------------------------------------------
# Step 2: retrain the 32-D autoencoder on the matched 100K dataset
# --------------------------------------------------------------------------
STEP2_DONE="$LOG_DIR/.step2_done"
if [ ! -f "$STEP2_DONE" ]; then
    log "STEP 2/5: Retraining autoencoder on matched 100K dataset (epochs=100, patience=10)"
    "$PY" -m bowhead.train.train_ae \
        --data "$DATA_OUT" \
        --out-dir "$AE_OUT_DIR" \
        --epochs 100 --batch 32 --lr 1e-3 --patience 10 --seed 42 --device auto \
        2>&1 | tee "$LOG_DIR/02_train_ae.log"
    touch "$STEP2_DONE"
else
    log "STEP 2/5: already done, skipping ($AE_CKPT)"
fi

# --------------------------------------------------------------------------
# Step 3: train the scratch (random-init) CNN on the same matched dataset
# --------------------------------------------------------------------------
STEP3_DONE="$LOG_DIR/.step3_done"
if [ ! -f "$STEP3_DONE" ]; then
    log "STEP 3/5: Training scratch CNN on matched 100K dataset"
    "$PY" -m bowhead.train.train_cnn \
        --data "$DATA_OUT" \
        --tag "$SCRATCH_TAG" \
        --seed 42 --device auto \
        2>&1 | tee "$LOG_DIR/03_train_scratch_cnn.log"
    touch "$STEP3_DONE"
else
    log "STEP 3/5: already done, skipping (runs/$SCRATCH_TAG/best.pt)"
fi

# --------------------------------------------------------------------------
# Step 4: train the warm-start CNN, initialised from the Step-2 AE checkpoint,
#         on the identical matched dataset
# --------------------------------------------------------------------------
STEP4_DONE="$LOG_DIR/.step4_done"
if [ ! -f "$STEP4_DONE" ]; then
    log "STEP 4/5: Training warm-start CNN on matched 100K dataset"
    if [ ! -f "$AE_CKPT" ]; then
        log "ERROR: AE checkpoint not found at $AE_CKPT (did Step 2 finish?)"
        exit 1
    fi
    "$PY" -m bowhead.train.train_cnn \
        --data "$DATA_OUT" \
        --warm-start "$AE_CKPT" \
        --tag "$WARMSTART_TAG" \
        --seed 42 --device auto \
        2>&1 | tee "$LOG_DIR/04_train_warmstart_cnn.log"
    touch "$STEP4_DONE"
else
    log "STEP 4/5: already done, skipping (runs/$WARMSTART_TAG/best.pt)"
fi

# --------------------------------------------------------------------------
# Step 5: score both matched-data models on the held-out Evaluation_200K set
#         and rebuild a dedicated comparison HTML (does not overwrite the
#         existing docs/detection_curves.html)
# --------------------------------------------------------------------------
STEP5_DONE="$LOG_DIR/.step5_done"
if [ ! -f "$STEP5_DONE" ]; then
    log "STEP 5/5: Scoring matched-data models on held-out Evaluation_200K"
    require_mounted "$EVAL_DIR"
    "$PY" -m bowhead.eval.score_eval_dataset \
        --eval-dir "$EVAL_DIR" \
        --model "$SCRATCH_TAG"   "$REPO_ROOT/runs/$SCRATCH_TAG/best.pt" \
        --model "$WARMSTART_TAG" "$REPO_ROOT/runs/$WARMSTART_TAG/best.pt" \
        --npz-out "$REPO_ROOT/runs/pr_curves_100k_matched.npz" \
        --html-out "$REPO_ROOT/docs/detection_curves_100k_matched.html" \
        --device auto \
        2>&1 | tee "$LOG_DIR/05_score_eval.log"
    touch "$STEP5_DONE"
else
    log "STEP 5/5: already done, skipping"
fi

log "PIPELINE COMPLETE"
log "  Dataset:            $DATA_OUT"
log "  AE checkpoint:       $AE_CKPT"
log "  Scratch CNN:         runs/$SCRATCH_TAG/best.pt"
log "  Warm-start CNN:      runs/$WARMSTART_TAG/best.pt"
log "  Comparison HTML:     docs/detection_curves_100k_matched.html"
