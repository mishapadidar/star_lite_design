#!/bin/bash
#SBATCH --job-name=cs_ladder
#SBATCH --partition=compute
#SBATCH --exclude=cn12,cn07
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#
# Continuation ladder of combined-stage optimizations (see ../run_ladder.py).
#   cd star_lite_design/combined_stage
#   LADDER=ladders/hsx_variant_vacuum.yaml SEED=<namelist> OUTDIR=<dir> sbatch sbatch/ladder.sbatch
# Optional first-stage seeds: COILS=<coils json> XPOINT=<xpoint json>. SETS="a.b=1 c.d=2": --set for every stage.
# Re-submitting the same command resumes: finished stages are skipped.
set -euo pipefail
: "${LADDER:?set LADDER}" "${SEED:?set SEED}" "${OUTDIR:?set OUTDIR}"
# $0 is SLURM's spooled copy, so locate combined_stage/ via CS_DIR or the submit directory
cd "${CS_DIR:-$SLURM_SUBMIT_DIR}"
[ -f run_ladder.py ] || { echo "run from star_lite_design/combined_stage or set CS_DIR" >&2; exit 2; }
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/data/common/raffael.wendlinger/envs/simflare/bin/python
echo "host $(hostname)  start $(date)  ladder ${LADDER}  seed ${SEED}  outdir ${OUTDIR}"
# STARTUP_JITTER=<s>: random delay before Python starts, so jobs released together (e.g. when drained nodes come
# back) do not all read JAX's ~1000 .pyc files over NFS at once (the 2026-10-17/18 startup hangs)
if [ "${STARTUP_JITTER:-0}" -gt 0 ]; then d=$((RANDOM % STARTUP_JITTER)); echo "startup jitter ${d}s"; sleep "$d"; fi
SETARGS=()
for item in ${SETS:-}; do SETARGS+=(--set "$item"); done
"$PY" -u run_ladder.py "$LADDER" --vmec-input "$SEED" --outdir "$OUTDIR" ${COILS:+--coils "$COILS"} ${XPOINT:+--xpoint "$XPOINT"} ${SETARGS[@]+"${SETARGS[@]}"} ${SMOKE:+--smoke}
echo "end $(date)"
