#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Train the MOSS-Image LoRA with the in-repo Diffusers+PEFT trainer
# (scripts/train/train_moss_lora.py). Runs on the training machine;
# see that script's header for the points to verify before a real run.
# Usage: scripts/train/train_moss.sh [--dry-run] [extra args...]
set -eu
cd "$(dirname "$0")/../.."
DATASET_DIR="training-data/moss"
if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]; then
  echo "error: $DATASET_DIR is empty; assemble it first (scripts/assemble_training.sh)" >&2
  exit 2
fi
if [ "${1:-}" = "--dry-run" ]; then
  echo "python3 scripts/train/train_moss_lora.py --dataset $DATASET_DIR --output-dir outputs/moss"
  exit 0
fi
mkdir -p outputs/moss
exec python3 scripts/train/train_moss_lora.py --dataset "$DATASET_DIR" \
  --output-dir outputs/moss "$@"
