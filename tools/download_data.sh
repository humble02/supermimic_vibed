#!/bin/bash
# Download the G1-retargeted LAFAN1 motions and the G1 description, then build the MJCF/USD assets.
# LAFAN1 (Ubisoft La Forge) is CC BY-NC-ND 4.0; the Unitree G1 description is BSD-3-Clause.
# Usage: tools/download_data.sh [--local|--download-only] (default: Docker conversion)
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:---docker}
case "$MODE" in
  --docker|--local|--download-only) ;;
  *) echo "Usage: $0 [--local|--download-only]" >&2; exit 2 ;;
esac
HF=https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset
API=https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main
list() { curl -fsSL --retry 3 "$API/$1" | python3 -c "import json,sys; [print(x['path']) for x in json.load(sys.stdin) if x['type'] == 'file']"; }
mkdir -p data/lafan1_g1 data/g1_description/meshes
MOTION_FILES=$(list g1)
for f in $MOTION_FILES; do curl -fsSL --retry 3 -o "data/lafan1_g1/$(basename "$f")" "$HF/resolve/main/$f"; done
curl -fsSL --retry 3 -o data/g1_description/g1_29dof_rev_1_0.urdf "$HF/resolve/main/robot_description/g1/g1_29dof_rev_1_0.urdf"
MESH_FILES=$(list robot_description/g1/meshes)
for f in $MESH_FILES; do curl -fsSL --retry 3 -o "data/g1_description/meshes/$(basename "$f")" "$HF/resolve/main/$f"; done
echo "motions: $(ls data/lafan1_g1 | wc -l) clips"
# Asset conversion needs mujoco + the standalone USD importers.
case "$MODE" in
  --local)
    bash run_local.sh tools/urdf_to_mjcf.py
    bash run_local.sh tools/mjcf_to_usd.py
    ;;
  --docker)
    docker exec supermimic bash -c "cd /workspace/supermimic && python tools/urdf_to_mjcf.py && python tools/mjcf_to_usd.py"
    ;;
esac
