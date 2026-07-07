#!/bin/bash
# AirCraft data-reduction sweep: ntrain in {20,40,60,80,100}, GeoPT 20ep + scratch 200ep each.
# Waits for newest DTCHull pipeline log to contain "=== ALL DONE ===".
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p logs scripts/sweep

# ---- wait for newest DTCHull pipeline to finish ----
echo "[sweep] $(date +%H:%M:%S) waiting for DTCHull pipeline 'ALL DONE' marker..."
while :; do
  DTCHULL_LOG=$(ls -t logs/dtchull_pipeline_*.log 2>/dev/null | head -1)
  if [ -n "$DTCHULL_LOG" ] && grep -q "=== ALL DONE ===" "$DTCHULL_LOG"; then
    echo "[sweep] $(date +%H:%M:%S) DTCHull pipeline done ($DTCHULL_LOG), starting sweep"
    break
  fi
  sleep 60
done

# ---- start sweep ----
TS=$(date +%Y%m%d_%H%M%S)
SWEEPLOG=logs/aircraft_sweep_${TS}.log
ORIG_GEO=scripts/finetune/GeoPT_craft.sh
ORIG_SCR=scripts/from_scratch_normal_cond/Transolver_craft.sh
log() { echo "[sweep $(date +%H:%M:%S)] $*" | tee -a "$SWEEPLOG"; }
log "=== START sweep $TS ==="

for NTRAIN in 20 40 60 80 100; do
  log "--- ntrain=$NTRAIN ---"
  GEO_SCRIPT=scripts/sweep/GeoPT_craft_n${NTRAIN}.sh
  SCR_SCRIPT=scripts/sweep/Transolver_craft_n${NTRAIN}.sh
  sed "s/--ntrain 100/--ntrain ${NTRAIN}/; s/craft_geopt_8layers/craft_geopt_8layers_n${NTRAIN}/" "$ORIG_GEO" > "$GEO_SCRIPT"
  sed "s/--ntrain 100/--ntrain ${NTRAIN}/; s/craft_transolver_8layers_normal_cond/craft_transolver_8layers_normal_cond_n${NTRAIN}/" "$ORIG_SCR" > "$SCR_SCRIPT"

  log "GeoPT 20ep (ntrain=$NTRAIN)"
  bash _run_finetune.sh "$GEO_SCRIPT" "geopt_craft_n${NTRAIN}" 2>&1 | tee -a "$SWEEPLOG"
  log "scratch 200ep (ntrain=$NTRAIN)"
  bash _run_finetune.sh "$SCR_SCRIPT" "transolver_craft_scratch_n${NTRAIN}" 2>&1 | tee -a "$SWEEPLOG"
done

log "=== ALL DONE ==="
log "finished at $(date +%Y%m%d_%H%M%S)"
