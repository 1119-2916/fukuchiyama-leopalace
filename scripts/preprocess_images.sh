#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Wrapper: per-model image preprocessing for LoRA training.
# All arguments are passed through to preprocess_images.py (needs Pillow).
exec python3 "$(dirname "$0")/preprocess_images.py" "$@"
