#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Wrapper: assemble a trainer-ready dataset from a chosen caption run.
# All arguments are passed through to assemble_training.py.
exec python3 "$(dirname "$0")/assemble_training.py" "$@"
