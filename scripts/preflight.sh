#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
exec python3 "$(dirname "$0")/preflight.py" "$@"
