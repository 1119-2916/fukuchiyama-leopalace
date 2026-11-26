#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Wrapper: normalize dataset images for LoRA training.
# All arguments are passed through to normalize_images.py (needs Pillow).
exec python3 "$(dirname "$0")/normalize_images.py" "$@"
