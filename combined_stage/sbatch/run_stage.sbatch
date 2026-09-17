#!/bin/bash
#SBATCH --job-name=cs_stage
#SBATCH --partition=compute
#SBATCH --exclude=cn12,cn07
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#
# ONE combined-stage optimization (no ladder), resumable:
#   cd star_lite_design/combined_stage
#   CONFIG=<yaml> SEED=<vmec namelist> OUTDIR=<run dir> [COILS=<coils json>] [XPOINT=<xpoint json>] \
#   [PLASMA_TARGET=<namelist the plasma walks to from the cold-solved SEED>] [SET="k=v k=v"] \
#     sbatch --job-name=<name> --output=<run dir>/logs/slurm_%j.out sbatch/run_stage.sbatch
# Re-submitting the same command resumes from <run dir>/state/checkpoint.npz (outer_maxiter counts over the whole run).
set -euo pipefail
: "${CONFIG:?set CONFIG}" "${SEED:?set SEED}" "${OUTDIR:?set OUTDIR}"
cd "${CS_DIR:-$SLURM_SUBMIT_DIR}"
[ -f combined_stage_vmex.py ] || { echo "run from star_lite_design/combined_stage or set CS_DIR" >&2; exit 2; }
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/data/common/raffael.wendlinger/envs/simflare/bin/python
RESUME=()
if [ -f "$OUTDIR/state/checkpoint.npz" ] || [ -f "$OUTDIR/checkpoint.npz" ]; then RESUME=(--resume); fi
SETS=()
for s in ${SET:-}; do SETS+=(--set "$s"); done
echo "host $(hostname)  start $(date)  config $CONFIG  seed $SEED  outdir $OUTDIR  ${RESUME[*]:-fresh}"
"$PY" -u combined_stage_vmex.py --config "$CONFIG" --vmec-input "$SEED" --outdir "$OUTDIR" \
  ${COILS:+--coils "$COILS"} ${XPOINT:+--xpoint "$XPOINT"} ${PLASMA_TARGET:+--plasma-target "$PLASMA_TARGET"} \
  ${SETS[@]+"${SETS[@]}"} ${RESUME[@]+"${RESUME[@]}"}
echo "end $(date)"
