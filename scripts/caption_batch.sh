#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Wrapper for scripts/caption_batch.py -- all arguments are passed through.
#
# Examples:
#   ./scripts/caption_batch.sh --dry-run
#   ./scripts/caption_batch.sh --trigger-word fukuchiyama_style --limit 5
#   CAPTION_PROVIDER=anthropic ./scripts/caption_batch.sh --overwrite
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 scripts/caption_batch.py "$@"
