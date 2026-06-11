#!/bin/bash
set -uo pipefail

# Master driver. Orchestrates the per-device workflow by calling lower-level steps,
# each of which loads its OWN venv (so there is no env/venv leakage between phases):
#   run_boozer_all.sh -> initial num_aux=0 optimization   (boozer venv)
#   run_polish.sh     -> singular polish + coil opt        (boozer venv)
#   run_render.sh     -> trace + plot + paraview render    (matplotlib venv + paraview)
# The master sets up local scratch, applies the max-relative-error gates, decides
# what to keep, and COPIES (shards) kept results to ceph. Nothing is written to ceph
# for a failed / out-of-spec device except the run log.
#
#   bash ./prefix.sh <margin> <well> <Z> <distance> <on_vessel> <config> \
#                    <vessel_id> <mono> <null(DN|SN)> [num_aux] [AR] [attempt]
#
# Flow: boozer_all -> gate. If it passes, render the num_aux=0 device and copy it to
# ceph BEFORE polishing. Then (mono 1/2) polish at num_aux=NUM_AUX_POLISH; if the
# polished device passes its own 0.1% gate, render + copy it; otherwise discard it
# (nothing to ceph but the log), exactly as a failed boozer_all is treated.

margin="$1"
well="$2"
Z="$3"
distance="$4"
on_vessel="$5"
config="$6"
vessel_id="$7"
mono="$8"
null="$9"

# Number of auxiliary planar coils for the polish (mono=1,2): devices have either
# 0 (unpolished) or exactly this many aux coils.
NUM_AUX_POLISH="${10:-10}"

# Aspect-ratio knob forwarded to boozer_all.py (--AR): 0 = leave AR as-is, 1 = reduce
# the plasma aspect ratio toward ~5. It is part of the device identity, so it also
# goes into the folder name (task_name) below.
AR="${11:-0}"

# Attempt index (perturbation seed). LAST positional arg so the prefix args mirror
# the folder-name tail (..._num_aux_AR_attempt).
attempt="${12}"

if [ "$well" = "OFF" ]; then
  well_str="OFF"
else
  well_str=$(python -c 'import sys; print(float(sys.argv[1]))' "$well")
fi
margin_str=$(printf "%.2f" "$margin" | sed 's/\./p/')

HOME_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Folder-name builder; must match boozer_all.py's TASK_NAME exactly so the device
# IDs (and thus the perturbation seed) line up. Argument: num_aux.
task_name() {
  echo "margin=${margin_str}_well=${well_str}_Z=${Z}_onvessel=${on_vessel}_distance=${distance}_configID=${config}_vesselID=${vessel_id}_mono=${mono}_null=${null}_num_aux=${1}_AR=${AR}_attempt=${attempt}"
}

# Shard folder name -> a <=256-bucket subdir (md5 prefix) so that no directory on
# ceph (output/, logs/) ever holds more than ~500 entries: output/<shard>/<device>
# and logs/<shard>/<name>.out. Pure function of the (deterministic) folder name, so
# it matches prefix_restart.sh and readers that discover devices by content (e.g.
# device_browser's os.walk) need no change.
shard() { printf '%s' "$1" | md5sum | cut -c1-2; }

INIT_NAME="$(task_name 0)"

SCRATCH="${TMPDIR:-/tmp/$USER}/disbatch_${SLURM_JOB_ID:-local}_${INIT_NAME}"
RUN="$SCRATCH/run"
LOG="$SCRATCH/log.out"

mkdir -p "$RUN" "$HOME_DIR/output"  "$HOME_DIR/logs"

# Duplicate stdout/stderr to $LOG while keeping the engine pipe alive.
exec > >(tee "$LOG") 2>&1

# Filter for what gets copied back from a kept device dir.
SYNC_INCLUDES=(
  --include='*/'
  --include='design_opt_final_*.json'
  --include='design_opt_final_*.yaml'
  --include='design_polished_final_*.json'
  --include='design_polished_final_*.yaml'
  --include='singular.json'
  --include='singular.yaml'
  --include='summary.txt'
  --include='max_rel_error.txt'
  --include='scene_*.png'
  --include='xs_*.png'
  --include='poincare*.txt'
  --include='xpoint.txt'
  --include='phis.txt'
  --include='xpoint_type.txt'
  --include='legs.txt'
  --include='snowflake_discriminant.txt'
  --include='vessel_cross_*.txt'
  --include='surface_cross_*.txt'
  --include='fixed_points_*.txt'
  --include='sc*.vts'
  --include='aux_coils_*.vtu'
  --include='surf_opt_*_final.vts'
  --include='curves_opt_final.vtu'
  --include='ma_opt_final.vtu'
  --include='xpoint_curves_opt_final.vtu'
  --include='xpoint_singular_curves_opt_final.vtu'
  --include='vessel_opt_final.vtr'
  --include='*xpoint_deletion*'
  --exclude='*'
)

# Copy ONE local device dir into its ceph shard: output/<shard>/<device>/...
# Called explicitly by the master only for devices that should be KEPT.
sync_dir() {
  local d="${1%/}" name dest
  [ -d "$d" ] || return 0
  name="$(basename "$d")"
  dest="$HOME_DIR/output/$(shard "$name")"
  mkdir -p "$dest"
  rsync -a "${SYNC_INCLUDES[@]}" "$d/" "$dest/$name/"
}

# Always preserved so the run is traceable, regardless of success/failure.
sync_log() {
  [ -f "$LOG" ] || return 0
  local dest="$HOME_DIR/logs/$(shard "$INIT_NAME")"
  mkdir -p "$dest"
  rsync -a "$LOG" "$dest/${INIT_NAME}.out" || true
}

cleanup() {
  status=$?
  # Device dirs are copied to ceph EXPLICITLY (sync_dir) at the points where they
  # are decided to be kept, so cleanup only preserves the log and wipes scratch.
  sync_log
  rm -rf "$SCRATCH"
  exit "$status"
}
trap cleanup EXIT

rsync -a --exclude 'output*' --exclude 'logs*' --exclude '*_disBatch_*' "$HOME_DIR/" "$RUN/"
mkdir -p "$SCRATCH/convert"
rsync -a "$HOME_DIR/../convert/" "$SCRATCH/convert/"

cd "$RUN"

echo "Host: $(hostname)"
echo "Scratch: $SCRATCH"
echo "Started: $(date)"
echo "Init task: $INIT_NAME"
echo "mono=$mono attempt=$attempt"

INIT_DIR="./output/$INIT_NAME"

# ───────────────────── (1) initial optimization -> num_aux=0 ──────────────────
bash run_boozer_all.sh \
  "$margin" "$well" "$Z" "$distance" "$on_vessel" \
  "$config" "$vessel_id" "$mono" "$attempt" "$null" "$AR"

# boozer_all.py writes design_opt_final_<DEVICE_ID>.json (the device ID is in the
# name); locate it by glob (one per device dir).
INIT_JSON="$(ls "$INIT_DIR"/design_opt_final_*.json 2>/dev/null | head -1)"
if [ -z "$INIT_JSON" ] || [ ! -f "$INIT_JSON" ]; then
  echo "ERROR: design_opt_final_*.json not produced in $INIT_DIR — boozer_all.py failed"
  exit 1
fi

# Max-relative-error gate on the initial device: boozer_all.py writes the largest
# relative constraint error (a fraction) to max_rel_error.txt. If it exceeds 0.1%
# the device did not meet its constraints -> abort (nothing to ceph but the log).
maxerr_file="$INIT_DIR/max_rel_error.txt"
if [ ! -f "$maxerr_file" ]; then
  echo "ERROR: $maxerr_file not found"
  exit 1
fi
if awk '{ exit !($1 > 0.001) }' "$maxerr_file"; then
  echo "ERROR: max relative error = $(cat "$maxerr_file") > 0.1%, aborting"
  exit 1
fi
echo "max relative error = $(cat "$maxerr_file") (<= 0.1%, OK)"

# ─────────── (2) boozer_all passed: render the num_aux=0 device, copy to ceph ──
# Render BEFORE polishing so the init device is secured on ceph independently of
# whatever happens to the polished device.
echo "=== rendering init device $INIT_DIR ==="
if ! bash run_render.sh "$INIT_JSON" "$INIT_DIR"; then
  echo "ERROR: init render failed"
  exit 1
fi
sync_dir "$INIT_DIR"

# ───────────────────────────── (3) polish (mono 1/2) ──────────────────────────
if [ "$mono" -eq 1 ]; then
  MONO_CONSTRAINT="identity"
elif [ "$mono" -eq 2 ]; then
  MONO_CONSTRAINT="trace"
else
  MONO_CONSTRAINT=""
fi

if [ -n "$MONO_CONSTRAINT" ]; then
  num_aux="$NUM_AUX_POLISH"
  POLISH_NAME="$(task_name "$num_aux")"
  POLISH_DIR="./output/$POLISH_NAME"
  mkdir -p "$POLISH_DIR"

  echo "--- polishing+optimizing num_aux=$num_aux ($MONO_CONSTRAINT) ---"
  # boozer_singular_opt.py reads the input json + its sibling yaml and writes
  # design_polished_final_<DEVICE_ID>.json (+ .yaml) IN PLACE in the polish dir.
  # Copy the init json + paired yaml in, preserving their (ID-suffixed) basenames.
  INIT_JSON_BASE="$(basename "$INIT_JSON")"
  INIT_YAML="${INIT_JSON%.json}.yaml"
  cp "$INIT_JSON" "$POLISH_DIR/$INIT_JSON_BASE"
  cp "$INIT_YAML" "$POLISH_DIR/$(basename "$INIT_YAML")"

  bash run_polish.sh "$POLISH_DIR/$INIT_JSON_BASE" "$num_aux" || true

  # The copied init design was only the polish INPUT; remove it (and its yaml) now
  # the polish has finished, leaving only the polished design in the dir.
  rm -f "$POLISH_DIR/$INIT_JSON_BASE" "$POLISH_DIR/$(basename "$INIT_YAML")"

  POLISHED_JSON="$(ls "$POLISH_DIR"/design_polished_final_*.json 2>/dev/null | head -1)"
  if [ -z "$POLISHED_JSON" ]; then
    echo "singular optimization num_aux=$num_aux failed (no design_polished_final_*.json) — discarding (nothing to ceph)"
    rm -rf "$POLISH_DIR"
  else
    # Same gate as boozer_all: a polished device that exceeds 0.1% is discarded
    # entirely (nothing to ceph but the log) -- symmetric with the init device.
    polish_err="$POLISH_DIR/max_rel_error.txt"
    if [ ! -f "$polish_err" ] || awk '{ exit !($1 > 0.001) }' "$polish_err"; then
      echo "polished num_aux=$num_aux out of spec (max rel err = $(cat "$polish_err" 2>/dev/null)) — discarding (nothing to ceph except log)"
      rm -rf "$POLISH_DIR"
    else
      echo "singular optimization num_aux=$num_aux succeeded (max rel err = $(cat "$polish_err") <= 0.1%)"
      echo "=== rendering polished device $POLISH_DIR ==="
      if ! bash run_render.sh "$POLISHED_JSON" "$POLISH_DIR"; then
        echo "ERROR: polished render failed"
        exit 1
      fi
      sync_dir "$POLISH_DIR"
    fi
  fi
fi

echo "Finished: $(date)"
