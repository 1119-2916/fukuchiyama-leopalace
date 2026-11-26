#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Train the Z-Image Base LoRA with AI Toolkit (configs/z-image/ai_toolkit.yaml).
exec "$(dirname "$0")/train_ai_toolkit.sh" z-image configs/z-image/ai_toolkit.yaml "${1:-}"
