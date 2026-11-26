#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Wrapper: runs from the repo root so relative paths resolve the same
# way as the other tools. All arguments pass through to generate.py.
cd "$(dirname "$0")/.."
exec python3 scripts/generate.py "$@"
