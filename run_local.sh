#!/usr/bin/env bash
# Run a Python entry point using the isolated kit-less environment.
# Example: ./run_local.sh supermimic/train.py --num_envs 1024 --pi_batch 1024
set -euo pipefail
cd "$(dirname "$0")"
REPO="$PWD"
if [ ! -x "$REPO/.venv/bin/python" ]; then
  echo "Run bash tools/setup_local.sh first." >&2
  exit 1
fi
export VIRTUAL_ENV="$REPO/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export ISAACLAB_PATH="$REPO/IsaacLab"
export UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV"
export UV_NO_SYNC=1
export PYTHONUNBUFFERED=1
export WARP_CACHE_PATH="$REPO/.cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$REPO/.cache/inductor"
export MPLCONFIGDIR="$REPO/.cache/matplotlib"
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR"
exec "$VIRTUAL_ENV/bin/python" "$@"
