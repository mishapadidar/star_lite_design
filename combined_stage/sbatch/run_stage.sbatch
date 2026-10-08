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
# PIN_CORES=<cpu list> runs the whole process under taskset. Needed with virtual casing: on several cores JAX/XLA CPU
# returns nondeterministic garbage from virtual_casing_jax (|B_plasma| up to 1e10 T at random points; identical inputs,
# 5 of 7 calls; OMP/BLAS/Eigen threads, async dispatch and the legacy XLA runtime do not fix it -- only one core does;
# jobs 2324-2326). Explains the irreproducible runs 911 / 1072.
# BUT pinning to ONE core deadlocks VMEX (its implicit solve calls back into Python from inside an XLA executable; a
# 1-thread pool cannot run the callback -- jobs 2328/2406, py-spy). 2 or 4 cores are deterministic (job 2418): for
# virtual-casing runs request --cpus-per-task=4 and leave PIN_CORES unset (the SLURM cgroup already limits the cores).
PIN=()
if [ -n "${PIN_CORES:-}" ]; then PIN=(taskset -c "$PIN_CORES"); fi
echo "host $(hostname)  start $(date)  config $CONFIG  seed $SEED  outdir $OUTDIR  ${RESUME[*]:-fresh}  ${PIN[*]:-unpinned}"
${PIN[@]+"${PIN[@]}"} "$PY" -u combined_stage_vmex.py --config "$CONFIG" --vmec-input "$SEED" --outdir "$OUTDIR" \
  ${COILS:+--coils "$COILS"} ${XPOINT:+--xpoint "$XPOINT"} ${PLASMA_TARGET:+--plasma-target "$PLASMA_TARGET"} \
  ${SETS[@]+"${SETS[@]}"} ${RESUME[@]+"${RESUME[@]}"}
echo "end $(date)"
