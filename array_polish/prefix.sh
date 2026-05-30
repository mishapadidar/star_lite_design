#!/bin/bash
set -uo pipefail

margin="$1"
well="$2"
Z="$3"
distance="$4"
on_vessel="$5"
config="$6"
vessel_id="$7"
mono="$8"
ncoils="$9"

if [ "$well" = "OFF" ]; then
  well_str="OFF"
else
  well_str=$(python -c 'import sys; print(float(sys.argv[1]))' "$well")
fi
margin_str=$(printf "%.2f" "$margin" | sed 's/\./p/')

HOME_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
INITIAL_OUTPUT="/mnt/home/agiuliani/ceph/STAR_LITE/star_lite_design/array_initial/output"
# Lookup into array_initial uses the task name WITHOUT ncoils (that run had no
# ncoils dimension); the polish output dir appends ncoils so each scan value
# gets its own directory.
INITIAL_TASK="margin=${margin_str}_well=${well_str}_Z=${Z}_onvessel=${on_vessel}_distance=${distance}_configID=${config}_vesselID=${vessel_id}_mono=${mono}"
TASK="${INITIAL_TASK}_ncoils=${ncoils}"

SCRATCH="${TMPDIR:-/tmp/$USER}/disbatch_${SLURM_JOB_ID:-local}_${TASK}"
RUN="$SCRATCH/run"
LOG="$SCRATCH/log.out"

mkdir -p "$RUN" "$HOME_DIR/output" "$HOME_DIR/logs"

# Duplicate stdout/stderr to $LOG while keeping the engine pipe alive.
exec > >(tee "$LOG") 2>&1

# Polish is defined only for mono=1 (M=I) and mono=2 (tr(M)=2).
if [ "$mono" -eq 1 ]; then
  MONO_CONSTRAINT="identity"
elif [ "$mono" -eq 2 ]; then
  MONO_CONSTRAINT="trace"
else
  echo "SKIP: mono=$mono is not 1 or 2 (polish only runs for these)"
  exit 0
fi

INITIAL_TASK_DIR="$INITIAL_OUTPUT/$INITIAL_TASK"
INITIAL_JSON="$INITIAL_TASK_DIR/design_opt_final.json"
if [ ! -f "$INITIAL_JSON" ]; then
  echo "SKIP: $INITIAL_JSON not found (no array_initial result for this task)"
  exit 0
fi

# Filtered sync: only invoked on successful exit. Copies just the artifacts
# produced by the polish step over to $HOME_DIR/output so we never duplicate
# or overwrite anything in array_initial/output.
sync_back_filtered() {
  if [ -d "$RUN/output" ]; then
    rsync -a \
      --include='*/' \
      --include='singular.json' \
      --include='singular.yaml' \
      --include='summary.txt' \
      --include='max_rel_error.txt' \
      --include='poincare*.txt' \
      --include='xpoint.txt' \
      --include='legs.txt' \
      --include='vessel_cross_*.txt' \
      --include='surface_cross_*.txt' \
      --include='fixed_points_*.txt' \
      --include='xs_*.png' \
      --include='scene_*.png' \
      --include='sc*.vts' \
      --include='aux_coils_*.vtu' \
      --include='surf_opt_*_final.vts' \
      --include='curves_opt_final.vtu' \
      --include='ma_opt_final.vtu' \
      --include='xpoint_curves_opt_final.vtu' \
      --include='vessel_opt_final.vtr' \
      --exclude='*' \
      "$RUN/output/" "$HOME_DIR/output/"
  fi
}

# Always preserved so the run is traceable, regardless of success/failure.
sync_log() {
  [ -f "$LOG" ] && rsync -a "$LOG" "$HOME_DIR/logs/${TASK}.out" || true
}

cleanup() {
  status=$?
  if [ "$status" -eq 0 ]; then
    sync_back_filtered
  fi
  sync_log
  rm -rf "$SCRATCH"
  exit "$status"
}
trap cleanup EXIT

# Stage source code into the run dir, excluding outputs/logs.
rsync -a --exclude output --exclude logs --exclude '*_disBatch_*' "$HOME_DIR/" "$RUN/"

# Only design_opt_final.json is brought over from the array_initial task dir;
# every other file the polish workflow needs (singular.json, the various
# *_opt_final.* VTKs, poincare/vessel/surface txt files, xs/scene PNGs) is
# produced by the python scripts that run below.
OUT_DIR_REL="output/$TASK"
mkdir -p "$RUN/$OUT_DIR_REL"
cp "$INITIAL_JSON" "$RUN/$OUT_DIR_REL/design_opt_final.json"

cd "$RUN"

echo "Host: $(hostname)"
echo "Scratch: $SCRATCH"
echo "Started: $(date)"
echo "Task: $TASK"
echo "Mono: $mono ($MONO_CONSTRAINT)"
echo "ncoils: $ncoils"

OUT_DIR="./$OUT_DIR_REL"
input_json="$OUT_DIR/design_opt_final.json"
singular_json="$OUT_DIR/singular.json"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONPATH="/mnt/home/agiuliani/ceph/STAR_LITE:${PYTHONPATH:-}"

source /mnt/home/agiuliani/ceph/STAR_LITE/venv/bin/activate
./boozer_singular.py "$input_json" "$MONO_CONSTRAINT" "$ncoils"

# Abort immediately if boozer_singular.py did not write singular.json (Newton
# solver failed, or the script errored out). The cleanup trap will sync the
# full run dir + log so the failure can be inspected.
if [ ! -f "$singular_json" ]; then
  echo "ERROR: $singular_json not produced — Newton solver failed or boozer_singular.py errored"
  deactivate
  exit 1
fi
echo "Wrote $singular_json"

# Recompute the summary metrics on the polished device (same venv). Reuses the
# thresholds and modular-coil rows from the array_initial summary.txt.
./compute_summary.py "$singular_json" "$INITIAL_TASK_DIR/summary.txt"
deactivate

module load modules/2.3-20240529
module load paraview/5.10.1
source /mnt/home/agiuliani/ceph/STAR_LITE/venv_matplotlib/bin/activate
export PYTHONPATH="${PYTHONPATH}:/mnt/home/agiuliani/ceph/STAR_LITE/"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# Manifolds + plotting consume the new singular.json (which holds the polished
# Boozer surfaces, the SingularPeriodicFieldLine xpoints, and the unchanged
# axes / sdf objects).
#./mk_manifolds_disk.py "$singular_json"
./mk_manifolds_STAR_RZ.py "$singular_json"
./plot_manifolds_disk.py "$singular_json"

# Paraview renders the scene from the VTK files staged from array_initial.
xvfb-run -a pvbatch --force-offscreen-rendering mk_paraview.py "$OUT_DIR" || exit 1

echo "Finished: $(date)"
