#!/usr/bin/env bash
# GLYPHMIND — one-command launcher (macOS / Linux)
set -euo pipefail
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
exec "$PY" run.py "$@"
