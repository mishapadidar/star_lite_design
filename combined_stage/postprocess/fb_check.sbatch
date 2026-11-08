#!/bin/bash
#SBATCH --job-name=cs_fbcheck
#SBATCH --partition=compute
#SBATCH --exclude=cn12,cn07
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#
# Free-boundary Poincaré check of ONE combined-stage result: coils-only vs coils + plasma (virtual casing) field.
#   RUN=<run dir> WOUT=<wout.nc> COILS=<coils json> XPOINT=<xpoint json> \
#     sbatch --output=<run dir>/logs/fbcheck_%j.out combined_stage/postprocess/fb_check.sbatch
# Writes <run dir>/fb_check/{prepare.npz,fmm.npz,poincare_fb.png,fb_summary.yaml}. Step 2 needs the OpenMP fmm3dpy of
# the cadmag env (python 3.11); steps 1 and 3 run in simflare.
set -euo pipefail
: "${RUN:?}" "${WOUT:?}" "${COILS:?}" "${XPOINT:?}"
PP=/data/common/raffael.wendlinger/simflare/ext/star_lite_design/combined_stage/postprocess
PY=/data/common/raffael.wendlinger/envs/simflare/bin/python
FMMPY=/data/common/raffael.wendlinger/envs/cadmag/bin/python
OUT="$RUN/fb_check"
mkdir -p "$OUT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$PP"
echo "host $(hostname) start $(date) run $RUN"
[ -f "$OUT/prepare.npz" ] || "$PY" -u fb_field_prepare.py "$WOUT" "$XPOINT" "$OUT/prepare.npz" ${PREP_ARGS:-}
[ -f "$OUT/fmm.npz" ] || "$FMMPY" -u fb_field_fmm.py "$OUT/prepare.npz" "$OUT/fmm.npz"
OMP_NUM_THREADS=1 "$PY" -u fb_poincare.py "$OUT/prepare.npz" "$OUT/fmm.npz" "$WOUT" "$COILS" "$XPOINT" "$OUT" \
  --nproc "${SLURM_CPUS_PER_TASK:-16}" ${TRACE_ARGS:-}
echo "end $(date)"
