#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Train the FLUX.3 Image Klein Base LoRA with AI Toolkit
# (configs/flux-klein/ai_toolkit.yaml). Gated model: HF_TOKEN required.
exec "$(dirname "$0")/train_ai_toolkit.sh" flux-klein configs/flux-klein/ai_toolkit.yaml "${1:-}"
