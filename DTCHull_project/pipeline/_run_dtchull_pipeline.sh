#!/bin/bash
# DTCHull pipeline: DL+preprocess -> GeoPT 200ep -> scratch 200ep
# Fixed: --hf_repo GeoPT/Downstream_Physics_Simulation (HaixuWu/GeoPT is 401);
#        set -eo pipefail so STEP failures abort the pipeline.
set -eo pipefail
# script lives in DTCHull_project/pipeline/ ; run everything from repo root
cd "$(dirname "$0")/../.."
mkdir -p DTCHull_project/logs hf_cache
TS=$(date +%Y%m%d_%H%M%S)
PIPELOG=DTCHull_project/logs/dtchull_pipeline_${TS}.log
HOST_UID=$(id -u)
HOST_GID=$(id -g)

log() { echo "[pipeline $(date +%H:%M:%S)] $*" | tee -a "$PIPELOG"; }
die() { log "FATAL: $*"; exit 1; }

log "=== START pipeline $TS ==="

# ---------- STEP 1: DL + preprocess ----------
log "STEP 1: DTCHull download + preprocess (i=1..130, ~20GB)"
docker run --rm --gpus all --shm-size=8g \
  -e PYTHONUNBUFFERED=1 \
  -e HF_HOME=/workspace/GeoPT/hf_cache \
  -v "$HOME/projects/GeoPT:/workspace/GeoPT" \
  -w /workspace/GeoPT \
  nvcr.io/nvidia/pytorch:25.10-py3 \
  bash -lc "pip install --quiet h5py timm matplotlib einops pyvista vtk huggingface_hub 'scipy>=1.11' tqdm 2>&1 | tail -3 && python DTCHull_project/preprocess/DTCHull_process.py --hf_repo GeoPT/Downstream_Physics_Simulation --hf_subdir DTCHull --outdir ./DTCHull_project/data --i_start 1 --i_end 130 --skip_existing; rc=\$?; chown -R ${HOST_UID}:${HOST_GID} /workspace/GeoPT/DTCHull_project/data /workspace/GeoPT/hf_cache 2>/dev/null || true; exit \$rc" 2>&1 | tee -a "$PIPELOG"
[ -d DTCHull_project/data ] && [ "$(ls DTCHull_project/data/ 2>/dev/null | wc -l)" -gt 0 ] || die "STEP 1 produced no files"
log "STEP 1 done: $(ls DTCHull_project/data/ | wc -l) files, $(du -sh DTCHull_project/data/ | cut -f1)"

# ---------- STEP 2: GeoPT 200ep ----------
log "STEP 2: GeoPT_dtchull.sh (200 epochs)"
bash _run_finetune.sh DTCHull_project/scripts/finetune/GeoPT_dtchull.sh geopt_dtchull 2>&1 | tee -a "$PIPELOG"
log "STEP 2 done"

# ---------- STEP 3: scratch 200ep ----------
log "STEP 3: Transolver_dtchull.sh scratch (200 epochs)"
bash _run_finetune.sh DTCHull_project/scripts/scratch_normal_cond/Transolver_dtchull.sh transolver_dtchull_scratch 2>&1 | tee -a "$PIPELOG"
log "STEP 3 done"

log "=== ALL DONE ==="
log "finished at $(date +%Y%m%d_%H%M%S)"
