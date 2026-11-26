#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Shared launcher for AI Toolkit jobs. Not called directly; use
# train_z_image.sh / train_flux_klein.sh, which pass model + config.
#   $1 = model slot (z-image | flux-klein), $2 = config yaml, $3 = --dry-run?
set -eu
MODEL="$1"; CONFIG="$2"; DRY="${3:-}"
cd "$(dirname "$0")/../.."

DATASET_DIR="training-data/$MODEL"
if [ ! -d "$DATASET_DIR" ] || [ -z "$(ls -A "$DATASET_DIR" 2>/dev/null)" ]; then
  echo "error: $DATASET_DIR is empty; assemble it first (scripts/assemble_training.sh)" >&2
  exit 2
fi
AI_TOOLKIT_DIR="${AI_TOOLKIT_DIR:-../ai-toolkit}"
RUN_PY="$AI_TOOLKIT_DIR/run.py"
if [ ! -f "$RUN_PY" ]; then
  echo "error: $RUN_PY not found; clone ostris/ai-toolkit or set AI_TOOLKIT_DIR" >&2
  exit 2
fi
CMD="python3 $RUN_PY $CONFIG"
if [ "$DRY" = "--dry-run" ]; then
  echo "$CMD"
  exit 0
fi
mkdir -p "outputs/$MODEL"
exec $CMD
