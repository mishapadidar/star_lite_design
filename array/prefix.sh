#!/bin/bash
set -uo pipefail

# Merged initial-optimization + polish workflow.
#
#   bash ./prefix.sh <margin> <well> <Z> <distance> <on_vessel> <config> \
#                    <vessel_id> <mono> <attempt> <null(DN|SN)>
#
# 1. boozer_all.py produces the unpolished device (num_aux=0); its base modular
#    coils are perturbed (seeded by the device ID) for this attempt.
# 2. The num_aux=0 device is traced + plotted + rendered.
# 3. If mono is 1 or 2, num_aux = 1..NUM_AUX_MAX is scanned, polishing the
#    freshly-computed num_aux=0 design (boozer_singular.py) into its own
#    num_aux=N directory. A failed polish for one num_aux is logged and skipped
#    (it does not fail the task). The polish reuses THIS run's num_aux=0 design
#    from local scratch, so there is no cross-task dependency.

margin="$1"
well="$2"
Z="$3"
distance="$4"
on_vessel="$5"
config="$6"
vessel_id="$7"
mono="$8"
attempt="$9"
null="${10}"

# Maximum number of auxiliary planar coils to try when polishing (mono=1,2).
NUM_AUX_MAX=7

if [ "$well" = "OFF" ]; then
  well_str="OFF"
else
  well_str=$(python -c 'import sys; print(float(sys.argv[1]))' "$well")
fi
margin_str=$(printf "%.2f" "$margin" | sed 's/\./p/')

HOME_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Folder-name builder; must match boozer_all.py's TASK_NAME exactly so the
# device IDs (and thus the perturbation seed) line up. Argument: num_aux.
task_name() {
  echo "margin=${margin_str}_well=${well_str}_Z=${Z}_onvessel=${on_vessel}_distance=${distance}_configID=${config}_vesselID=${vessel_id}_mono=${mono}_null=${null}_num_aux=${1}_attempt=${attempt}"
}

INIT_NAME="$(task_name 0)"

SCRATCH="${TMPDIR:-/tmp/$USER}/disbatch_${SLURM_JOB_ID:-local}_${INIT_NAME}"
RUN="$SCRATCH/run"
LOG="$SCRATCH/log.out"

mkdir -p "$RUN" "$HOME_DIR/output"  "$HOME_DIR/logs"

# Duplicate stdout/stderr to $LOG while keeping the engine pipe alive.
exec > >(tee "$LOG") 2>&1

# Filtered sync of every produced output dir (init + any polished). Only run on
# success, so failed runs never litter ceph with half-built directories.
sync_back_filtered() {
  if [ -d "$RUN/output" ]; then
    rsync -a \
      --include='*/' \
      --include='design_opt_final.json' \
      --include='design_opt_final.yaml' \
      --include='singular.json' \
      --include='singular.yaml' \
      --include='summary.txt' \
      --include='max_rel_error.txt' \
      --include='nonQS.txt' \
      --include='scene_*.png' \
      --include='xs_*.png' \
      --include='poincare*.txt' \
      --include='xpoint.txt' \
      --include='legs.txt' \
      --include='vessel_cross_*.txt' \
      --include='surface_cross_*.txt' \
      --include='fixed_points_*.txt' \
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
  [ -f "$LOG" ] && rsync -a "$LOG" "$HOME_DIR/logs/${INIT_NAME}.out" || true
}

cleanup() {
  status=$?
  # On success: filtered sync of all produced dirs. On failure: log only.
  # Always wipe scratch.
  if [ "$status" -eq 0 ]; then
    sync_back_filtered
  fi
  sync_log
  rm -rf "$SCRATCH"
  exit "$status"
}
trap cleanup EXIT

rsync -a --exclude output --exclude logs --exclude '*_disBatch_*' "$HOME_DIR/" "$RUN/"
mkdir -p "$SCRATCH/convert"
rsync -a "$HOME_DIR/../convert/" "$SCRATCH/convert/"

cd "$RUN"

echo "Host: $(hostname)"
echo "Scratch: $SCRATCH"
echo "Started: $(date)"
echo "Init task: $INIT_NAME"
echo "mono=$mono attempt=$attempt"

INIT_DIR="./output/$INIT_NAME"
INIT_JSON="$INIT_DIR/design_opt_final.json"
INIT_SUMMARY="$INIT_DIR/summary.txt"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONPATH="/mnt/home/agiuliani/ceph/STAR_LITE:${PYTHONPATH:-}"

# ───────────────────────── Phase A: solves (boozer venv) ──────────────────
source /mnt/home/agiuliani/ceph/STAR_LITE/venv/bin/activate

# (1) Initial optimization -> num_aux=0 device.
./boozer_all.py \
  --margin "$margin" \
  --well "$well" \
  --Z "$Z" \
  --distance "$distance" \
  --on-vessel "$on_vessel" \
  --config "$config" \
  --vessel-id "$vessel_id" \
  --mono "$mono" \
  --num-aux 0 \
  --attempt "$attempt" \
  --null "$null"

if [ ! -f "$INIT_JSON" ]; then
  echo "ERROR: $INIT_JSON not produced — boozer_all.py failed"
  deactivate
  exit 1
fi

# nonQS gate on the initial device.
nonqs_file="$INIT_DIR/nonQS.txt"
if [ ! -f "$nonqs_file" ]; then
  echo "ERROR: $nonqs_file not found"
  deactivate
  exit 1
fi
if awk '{ exit !($1 > 20) }' "$nonqs_file"; then
  echo "ERROR: nonQS = $(cat "$nonqs_file")% > 20%, aborting"
  deactivate
  exit 1
fi
echo "nonQS = $(cat "$nonqs_file")% (OK)"

# Directories to render in Phase B, paired with the json the tracer consumes.
RENDER_DIRS=("$INIT_DIR")
RENDER_JSONS=("$INIT_JSON")

# (2) Polish (mono=1 -> M=I, mono=2 -> tr(M)=2) over num_aux = 1..NUM_AUX_MAX.
if [ "$mono" -eq 1 ]; then
  MONO_CONSTRAINT="identity"
elif [ "$mono" -eq 2 ]; then
  MONO_CONSTRAINT="trace"
else
  MONO_CONSTRAINT=""
fi

if [ -n "$MONO_CONSTRAINT" ]; then
  for num_aux in $(seq 1 "$NUM_AUX_MAX"); do
    POLISH_NAME="$(task_name "$num_aux")"
    POLISH_DIR="./output/$POLISH_NAME"
    mkdir -p "$POLISH_DIR"
    # Reuse THIS run's num_aux=0 design as the polish input.
    cp "$INIT_JSON" "$POLISH_DIR/design_opt_final.json"

    echo "--- polishing num_aux=$num_aux ($MONO_CONSTRAINT) ---"
    ./boozer_singular.py "$POLISH_DIR/design_opt_final.json" "$MONO_CONSTRAINT" "$num_aux" || true

    if [ -f "$POLISH_DIR/singular.json" ]; then
      echo "polish num_aux=$num_aux succeeded"
      ./compute_summary.py "$POLISH_DIR/singular.json" "$INIT_SUMMARY" || true
      RENDER_DIRS+=("$POLISH_DIR")
      RENDER_JSONS+=("$POLISH_DIR/singular.json")
    else
      echo "polish num_aux=$num_aux failed (no singular.json) — skipping"
      rm -rf "$POLISH_DIR"
    fi
  done
fi
deactivate

# ──────────────── Phase B: trace + plot + render (matplotlib venv) ─────────
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

# mk_manifolds + plot_manifolds support all of mono=0,1,2; the init device reads
# design_opt_final.json, polished devices read singular.json.
for i in "${!RENDER_DIRS[@]}"; do
  d="${RENDER_DIRS[$i]}"
  j="${RENDER_JSONS[$i]}"
  echo "=== rendering $d (input $j) ==="
  ./mk_manifolds.py "$j"
  ./plot_manifolds.py "$j"
  xvfb-run -a pvbatch --force-offscreen-rendering mk_paraview.py "$d" || exit 1
done

echo "Finished: $(date)"
