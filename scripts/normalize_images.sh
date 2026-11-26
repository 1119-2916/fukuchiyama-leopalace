#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Wrapper: runs from the repo root so relative paths (dataset/, outputs)
# always resolve the same way, matching caption_batch.sh.
cd "$(dirname "$0")/.."
exec python3 scripts/normalize_images.py "$@"
