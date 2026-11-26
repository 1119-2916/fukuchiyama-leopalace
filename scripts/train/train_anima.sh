#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Train the Anima LoRA with kohya-ss/sd-scripts (anima_train_network.py).
#
# Runs on the training machine (RTX 6000), not in this repo's CI.
# Prerequisites:
#   - sd-scripts cloned and its venv active, or SD_SCRIPTS_DIR pointing at it
#   - ANIMA_BASE_PATH: path/ID of the Anima Base checkpoint
#   - training-data/anima/ assembled (scripts/assemble_training.sh)
# Usage: scripts/train/train_anima.sh [--dry-run]
set -eu
cd "$(dirname "$0")/../.."

DATASET_DIR="training-data/anima"
if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]; then
  echo "error: $DATASET_DIR is empty; assemble it first (scripts/assemble_training.sh)" >&2
  exit 2
fi
: "${ANIMA_BASE_PATH:?set ANIMA_BASE_PATH to the Anima Base checkpoint}"
SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-../sd-scripts}"
TRAIN_PY="$SD_SCRIPTS_DIR/anima_train_network.py"
if [ ! -f "$TRAIN_PY" ]; then
  echo "error: $TRAIN_PY not found; clone kohya-ss/sd-scripts or set SD_SCRIPTS_DIR" >&2
  exit 2
fi

# Starting-point hyperparameters for 30-50 images (record actuals per run
# in docs/training.md). Verify flag names against the sd-scripts version
# on the training machine before the first real run.
CMD="accelerate launch $TRAIN_PY
  --pretrained_model_name_or_path $ANIMA_BASE_PATH
  --dataset_config configs/anima/dataset.toml
  --output_dir outputs/anima
  --output_name okakikai-anima
  --network_dim 32
  --network_alpha 16
  --learning_rate 1e-4
  --lr_scheduler cosine
  --lr_warmup_steps 100
  --max_train_epochs 10
  --save_every_n_epochs 2
  --mixed_precision bf16
  --gradient_checkpointing
  --cache_latents
  --cache_text_encoder_outputs
  --optimizer_type adamw_torch
  --seed 42"

if [ "${1:-}" = "--dry-run" ]; then
  echo "$CMD"
  exit 0
fi
mkdir -p outputs/anima
# shellcheck disable=SC2086
exec $CMD
