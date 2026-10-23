#!/usr/bin/env bash
# Install the pinned kit-less stack in a project-local venv, without Docker.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
ISAACLAB_COMMIT=46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
if [ ! -d IsaacLab ]; then
  git init IsaacLab
  git -C IsaacLab remote add origin https://github.com/isaac-sim/IsaacLab.git
  git -C IsaacLab fetch --depth 1 origin "$ISAACLAB_COMMIT"
  git -C IsaacLab checkout --detach FETCH_HEAD
fi
if [ "$(git -C IsaacLab rev-parse HEAD)" != "$ISAACLAB_COMMIT" ]; then
  echo "IsaacLab must be checked out at $ISAACLAB_COMMIT" >&2
  exit 1
fi
export UV_PROJECT_ENVIRONMENT="$REPO/.venv"
export UV_CACHE_DIR="$REPO/.cache/uv"
export UV_PYTHON=/usr/bin/python3.12
export UV_PYTHON_PREFERENCE=only-system
export UV_HTTP_TIMEOUT=120
export UV_HTTP_RETRIES=5
export UV_LINK_MODE=hardlink
cd IsaacLab
# Same locked extras as docker/Dockerfile.kitless in the pinned checkout.
uv sync --frozen --inexact --extra all --extra importers --extra test
