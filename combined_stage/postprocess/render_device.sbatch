#!/bin/bash
#SBATCH --job-name=cs_device
#SBATCH --partition=compute
#SBATCH --exclude=cn12,cn07
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# Array-campaign device data <run dir>/device and figures <run dir>/figures for ONE combined-stage checkpoint:
#   RUN=<run dir> TAG=<final|outerNNN> [STEPS=all|figures] [SURFACE_GUESS=<archive json>[:index]] \
#     sbatch --output=<run dir>/logs/device_%j.out combined_stage/postprocess/render_device.sbatch
# STEPS=all (default): design archive + summary.txt, LCFS and 90/80/70 % surfaces, manifold xs plot, elongation
# (the array/ scripts in runjobs.sh order), then the figures. STEPS=figures: only the nine-angle Poincaré grid and the
# iso/top/left/right renders of an existing device folder.
set -uo pipefail
: "${RUN:?set RUN}" "${TAG:?set TAG}"
STEPS=${STEPS:-all}
SLD=/data/common/raffael.wendlinger/simflare/ext/star_lite_design
PY=/data/common/raffael.wendlinger/envs/simflare/bin/python
VTKPY=/data/common/raffael.wendlinger/envs/vtkrender/bin/python
PP=$SLD/combined_stage/postprocess
DEV="$RUN/device"
FIG="$RUN/figures"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
unset DISPLAY
mkdir -p "$DEV" "$FIG"
cd "$SLD/array"
echo "host $(hostname)  start $(date)  run $RUN  tag $TAG  steps $STEPS"

if [ "$STEPS" = all ]; then
  "$PY" -u "$PP/mk_device_archive.py" "$RUN" "$TAG" "$DEV" ${SURFACE_GUESS:+--surface-guess "$SURFACE_GUESS"} || { echo "archive build failed"; exit 1; }
  J=$(ls "$DEV"/design_opt_final_*.json | head -1)
  "$PY" -u mk_LCFS.py "$J" || echo "WARNING: mk_LCFS.py failed"
  "$PY" -u mk_manifolds.py "$J" || echo "WARNING: mk_manifolds.py failed"
  "$PY" -u plot_manifolds.py "$J" || echo "WARNING: plot_manifolds.py failed"
  for f in "$DEV"/xs_*.png; do [ -f "$f" ] && mv "$f" "$FIG/xs_manifolds.png"; done
  "$PY" -u mk_elongation.py "$J" || echo "WARNING: mk_elongation.py failed"
fi
"$PY" -u "$PP/poincare_grid.py" "$DEV" --out-dir "$FIG" || echo "WARNING: poincare_grid.py failed"
VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow "$VTKPY" "$PP/render_views.py" "$DEV" "$FIG" || echo "WARNING: render_views.py failed"
echo "end $(date)"
