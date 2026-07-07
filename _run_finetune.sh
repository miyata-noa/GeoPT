#!/bin/bash
# Wrapper: runs a finetune script inside NGC container, logs to ./logs/, fixes ownership.
# Usage: bash _run_finetune.sh <script_path> <tag>
set -e
SCRIPT="$1"
TAG="$2"
if [ -z "$SCRIPT" ] || [ -z "$TAG" ]; then
  echo "Usage: $0 <script_path> <tag>" >&2
  exit 2
fi
cd "$(dirname "$0")"
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/${TAG}_${TS}.log"
HOST_UID=$(id -u)
HOST_GID=$(id -g)
echo "[wrapper] START $TS tag=$TAG script=$SCRIPT log=$LOG" | tee "$LOG"
docker run --rm --gpus all --shm-size=8g \
  -v "$HOME/projects/GeoPT:/workspace/GeoPT" \
  -w /workspace/GeoPT \
  nvcr.io/nvidia/pytorch:25.10-py3 \
  bash -lc "pip install --quiet h5py timm matplotlib einops pyvista vtk huggingface_hub 'scipy>=1.11' tqdm 2>&1 | tail -3 && python -c 'import torch; print(\"[torch]\", torch.__version__, \"cuda=\", torch.cuda.is_available(), torch.cuda.get_device_name(0))' && echo '[wrapper] launching $SCRIPT' && bash '$SCRIPT'; rc=\$?; echo '[wrapper] script exit code:' \$rc; chown -R ${HOST_UID}:${HOST_GID} /workspace/GeoPT/results /workspace/GeoPT/checkpoints /workspace/GeoPT/logs 2>/dev/null || true; exit \$rc" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "[wrapper] END $(date +%Y%m%d_%H%M%S) rc=$RC" | tee -a "$LOG"
exit $RC
