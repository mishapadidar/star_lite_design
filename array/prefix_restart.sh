#!/bin/bash
set -uo pipefail

# RESTART variant of prefix.sh: skip the boozer_all.py initial optimization and
# restart a previously-run prefix.sh FROM the boozer_singular_opt.py polish step.
#
#   bash ./prefix_restart.sh <margin> <well> <Z> <distance> <on_vessel> <config> \
#                            <vessel_id> <mono> <attempt> <null(DN|SN)> [num_aux]
#
# Identical to prefix.sh EXCEPT step (1): instead of running boozer_all.py, it
# copies the design_opt_final.json (+ sibling .yaml and max_rel_error.txt) that a
# previous prefix.sh run produced -- from the persistent output directory dictated
# by THESE arguments -- into local scratch (/tmp/...). The SAME max-relative-error
# gate as prefix.sh is then applied; if it fails the restart aborts. Everything
# from boozer_singular_opt.py onward (polish + trace + plot + render) is unchanged.
# Use this to re-run a rewritten boozer_singular_opt.py without recomputing the
# (expensive) num_aux=0 device.

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

# Number of auxiliary planar coils for the polish (mono=1,2): either devices
# have 0 (unpolished) or exactly this many aux coils.
NUM_AUX_POLISH="${11:-10}"

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
      --include='design_polished_final.json' \
      --include='design_polished_final.yaml' \
      --include='singular.json' \
      --include='singular.yaml' \
      --include='summary.txt' \
      --include='max_rel_error.txt' \
      --include='scene_*.png' \
      --include='xs_*.png' \
      --include='poincare*.txt' \
      --include='xpoint.txt' \
      --include='phis.txt' \
      --include='xpoint_type.txt' \
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
      --include='xpoint_singular_curves_opt_final.vtu' \
      --include='vessel_opt_final.vtr' \
      --include='*xpoint_deletion*' \
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

# A mono=0 device has no monodromy polish, so there is nothing for the
# boozer_singular_opt.py restart to do. Fail fast and abort the task -- the EXIT
# trap above cleans up scratch ($SCRATCH) and preserves the log.
if [ "$mono" -eq 0 ]; then
  echo "ERROR: mono=0 has no polish step — nothing to restart from boozer_singular_opt.py; aborting task"
  exit 1
fi

rsync -a --exclude output --exclude logs --exclude '*_disBatch_*' "$HOME_DIR/" "$RUN/"
mkdir -p "$SCRATCH/convert"
rsync -a "$HOME_DIR/../convert/" "$SCRATCH/convert/"

cd "$RUN"

echo "Host: $(hostname)"
echo "Scratch: $SCRATCH"
echo "Started: $(date)"
echo "Init task: $INIT_NAME"
echo "mono=$mono attempt=$attempt (RESTART from boozer_singular_opt.py)"

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

# (1) RESTART: reuse the num_aux=0 device from a previous prefix.sh run instead of
# recomputing it. Copy design_opt_final.json (+ sibling .yaml and the previous
# max_rel_error.txt) from the persistent output directory dictated by the
# arguments into local scratch (/tmp/...), where boozer_singular_opt.py reads it.
SRC_INIT_DIR="$HOME_DIR/output/$INIT_NAME"
if [ ! -f "$SRC_INIT_DIR/design_opt_final.json" ] || [ ! -f "$SRC_INIT_DIR/design_opt_final.yaml" ]; then
  echo "ERROR: $SRC_INIT_DIR/design_opt_final.{json,yaml} not found — nothing to restart from"
  deactivate
  exit 1
fi
mkdir -p "$INIT_DIR"
cp "$SRC_INIT_DIR/design_opt_final.json" "$INIT_JSON"
cp "$SRC_INIT_DIR/design_opt_final.yaml" "$INIT_DIR/design_opt_final.yaml"
[ -f "$SRC_INIT_DIR/max_rel_error.txt" ] && cp "$SRC_INIT_DIR/max_rel_error.txt" "$INIT_DIR/max_rel_error.txt"
echo "restart: copied design_opt_final.{json,yaml}(+max_rel_error.txt) from $SRC_INIT_DIR to $INIT_DIR"

# Max-relative-error gate on the (reused) initial device: boozer_all.py wrote the
# largest relative constraint error (a fraction) to max_rel_error.txt. If it
# exceeds 0.1% the device did not meet its constraints, so do NOT proceed to
# polishing / tracing / rendering -- exactly as prefix.sh.
maxerr_file="$INIT_DIR/max_rel_error.txt"
if [ ! -f "$maxerr_file" ]; then
  echo "ERROR: $maxerr_file not found"
  deactivate
  exit 1
fi
if awk '{ exit !($1 > 0.001) }' "$maxerr_file"; then
  echo "ERROR: max relative error = $(cat "$maxerr_file") > 0.1%, aborting"
  deactivate
  exit 1
fi
echo "max relative error = $(cat "$maxerr_file") (<= 0.1%, OK)"

# Phase B renders ONLY the polished device(s) that succeed; the restart never
# re-renders the reused initial design_opt_final.json device. These arrays start
# empty and are populated by the polish loop below only on a successful polish.
RENDER_DIRS=()
RENDER_JSONS=()

# (2) Polish (mono=1 -> M=I, mono=2 -> tr(M)=2) over num_aux = 1..NUM_AUX_MAX.
if [ "$mono" -eq 1 ]; then
  MONO_CONSTRAINT="identity"
elif [ "$mono" -eq 2 ]; then
  MONO_CONSTRAINT="trace"
else
  MONO_CONSTRAINT=""
fi

if [ -n "$MONO_CONSTRAINT" ]; then
  for num_aux in "$NUM_AUX_POLISH"; do
    POLISH_NAME="$(task_name "$num_aux")"
    POLISH_DIR="./output/$POLISH_NAME"
    mkdir -p "$POLISH_DIR"

    echo "--- polishing+optimizing num_aux=$num_aux ($MONO_CONSTRAINT) ---"
    # Copy THIS run's num_aux=0 design (json + sibling yaml) into the polish dir;
    # boozer_singular_opt.py reads ALL its parameters from that yaml (weights,
    # thresholds, monodromy constraint, config id, num_aux=10, well state) and
    # writes design_polished_final.json (+ .yaml) IN PLACE in the same dir, with
    # the boozer surfaces/axes re-solved on the combined modular+aux coil set.
    cp "$INIT_JSON" "$POLISH_DIR/design_opt_final.json"
    cp "$INIT_DIR/design_opt_final.yaml" "$POLISH_DIR/design_opt_final.yaml"

    ./boozer_singular_opt.py "$POLISH_DIR/design_opt_final.json" --num-aux "$num_aux" || true

    if [ ! -f "$POLISH_DIR/design_polished_final.json" ]; then
      echo "singular optimization num_aux=$num_aux failed (no design_polished_final.json) — skipping"
      rm -rf "$POLISH_DIR"
    else
      # Verify the POLISHED device also meets its constraints to 0.1% (the polish
      # writes its own max_rel_error.txt). If it exceeds 0.1%, keep the results
      # but do NOT trace/plot/render it.
      polish_err="$POLISH_DIR/max_rel_error.txt"
      if [ -f "$polish_err" ] && awk '{ exit !($1 > 0.001) }' "$polish_err"; then
        echo "polished num_aux=$num_aux exceeds 0.1% (max rel err = $(cat "$polish_err")) — not rendering"
      else
        echo "singular optimization num_aux=$num_aux succeeded (max rel err = $(cat "$polish_err" 2>/dev/null) <= 0.1%)"
        RENDER_DIRS+=("$POLISH_DIR")
        RENDER_JSONS+=("$POLISH_DIR/design_polished_final.json")
      fi
    fi
  done
fi
deactivate

# ──────────────── Phase B: trace + plot + render (matplotlib venv) ─────────
# Render ONLY the polished device(s) that succeeded (RENDER_DIRS); the restart
# never re-renders the initial device. If the polish/optimization did not succeed
# there is nothing to render, so skip Phase B entirely (no module/venv load) and
# let the EXIT trap sync back whatever the polish produced.
if [ "${#RENDER_DIRS[@]}" -eq 0 ]; then
  echo "no polished device succeeded — skipping trace/plot/render"
else
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

  # Each polished device is traced from its design_polished_final.json (which
  # carries the COMBINED modular+aux coil set, written by the finalize step of
  # boozer_singular_opt.py). RENDER_JSONS holds the right path per device.
  for i in "${!RENDER_DIRS[@]}"; do
    d="${RENDER_DIRS[$i]}"
    j="${RENDER_JSONS[$i]}"
    echo "=== rendering $d (input $j) ==="
    ./mk_manifolds.py "$j"
    ./plot_manifolds.py "$j"
    xvfb-run -a pvbatch --force-offscreen-rendering mk_paraview.py "$d" || exit 1
  done
fi

echo "Finished: $(date)"
